//! Dedicated TCP transport for real OpenRaft RPC traffic.
//!
//! `network.rs` provides the crate's general-purpose cluster RPC transport
//! (`NetworkManager`/`NetworkService`, health checks, shard operations, BFT
//! messages) via the `RpcMessage` enum. OpenRaft's own RPC types
//! (`AppendEntriesRequest`, `VoteRequest`, `InstallSnapshotRequest`, and their
//! responses) don't fit that shape — they carry full log entries, vote/term
//! state, and membership/snapshot metadata that `RpcMessage` was never
//! designed to hold. This module is a small, separate, dedicated transport
//! built specifically for OpenRaft:
//!
//! - [`OxirsRaftNetworkFactory`] / [`OxirsRaftNetworkClient`]: the *outbound*
//!   side, implementing `openraft`'s [`RaftNetworkFactory`]/[`RaftNetwork`]
//!   traits so `openraft::Raft` can send RPCs to peers.
//! - [`serve_raft_rpc`]: the *inbound* side — an accept loop that answers
//!   those RPCs on the local `openraft::Raft` instance. `network.rs` never
//!   grew an accept loop of its own (see the comment at
//!   `NetworkManager::start_background_tasks`); this is that loop, scoped to
//!   Raft traffic only.
//!
//! Wire format matches `network.rs`'s existing convention exactly: a
//! big-endian `u32` length prefix followed by an oxicode-encoded body
//! (COOLJAPAN policy: oxicode, never bincode).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use openraft::error::{
    InstallSnapshotError, NetworkError as OpenraftNetworkError, RPCError, RaftError, RemoteError,
    Timeout, Unreachable,
};
use openraft::network::{RPCOption, RPCTypes, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use openraft::{BasicNode, Raft};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

use crate::raft::{OxirsNodeId, OxirsTypeConfig};

/// Maximum size of a single Raft RPC wire frame. Generous enough for
/// reasonably large replication batches (`Config::max_payload_entries`
/// entries of RDF commands) plus snapshot chunks
/// (`Config::snapshot_max_chunk_size`, default 3 MiB), while still bounding
/// allocation on behalf of a corrupt or hostile peer.
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Wire envelope for an outbound OpenRaft RPC request, concretely typed to
/// [`OxirsTypeConfig`] — this crate only ever uses one type config, so there
/// is no need to keep this generic.
#[derive(Debug, Serialize, Deserialize)]
enum RaftWireRequest {
    AppendEntries(AppendEntriesRequest<OxirsTypeConfig>),
    Vote(VoteRequest<OxirsNodeId>),
    InstallSnapshot(InstallSnapshotRequest<OxirsTypeConfig>),
    /// Raft §3.10 leadership-transfer signal: instruct the receiving node to
    /// immediately start an election (`Raft::trigger().elect()`), as OpenRaft
    /// 0.9.24 has no native leader-transfer API. Sent by the current leader to
    /// a caught-up target during a graceful handoff.
    TimeoutNow,
}

/// Wire envelope for the corresponding response. Each variant carries the
/// exact `Result` shape the matching `Raft<C>` inbound handler produces.
/// `RaftError` and `InstallSnapshotError` both derive `Serialize`/
/// `Deserialize` under openraft's `serde` feature (confirmed in the vendored
/// source: `openraft-0.9.24/src/error.rs`), so these round-trip losslessly —
/// nothing here is stringified away or silently swallowed.
#[derive(Debug, Serialize, Deserialize)]
enum RaftWireResponse {
    AppendEntries(Result<AppendEntriesResponse<OxirsNodeId>, RaftError<OxirsNodeId>>),
    Vote(Result<VoteResponse<OxirsNodeId>, RaftError<OxirsNodeId>>),
    InstallSnapshot(
        Result<InstallSnapshotResponse<OxirsNodeId>, RaftError<OxirsNodeId, InstallSnapshotError>>,
    ),
    /// Acknowledgement of a [`RaftWireRequest::TimeoutNow`]: `Ok(())` once the
    /// receiver has triggered its election, or a stringified `Fatal` error if
    /// its Raft core has already shut down. Stringified because `Fatal`'s
    /// serde round-trip is not needed here — the sender only needs
    /// success/failure to decide whether to fail loud.
    TimeoutNow(Result<(), String>),
}

/// Transport-level failures from this module's own plumbing (connect,
/// timeout, codec, framing) — distinct from the OpenRaft-level errors they
/// eventually get mapped into by [`OxirsRaftNetworkClient`]. Implements
/// `std::error::Error` (via `thiserror`) so it can be fed directly into
/// `openraft::error::{Unreachable, NetworkError}::new`.
#[derive(Debug, thiserror::Error)]
enum RaftTransportError {
    #[error("connect to {addr}: {source}")]
    Connect {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("io error talking to {addr}: {source}")]
    Io {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("frame to/from {addr} ({len} bytes) exceeds max message size ({max} bytes)")]
    FrameTooLarge {
        addr: SocketAddr,
        len: usize,
        max: usize,
    },
    #[error("failed to encode raft RPC for {addr}: {message}")]
    Encode { addr: SocketAddr, message: String },
    #[error("failed to decode raft RPC frame from {addr}: {message}")]
    Decode { addr: SocketAddr, message: String },
    #[error("no known network address for peer node {peer}")]
    UnknownPeer { peer: OxirsNodeId },
    #[error("peer {addr} replied with a response of the wrong RPC kind")]
    MismatchedResponse { addr: SocketAddr },
    #[error("raft RPC to {addr} timed out after {timeout:?}")]
    Timeout { addr: SocketAddr, timeout: Duration },
}

/// Write `message` to `stream` as a length-prefixed oxicode frame: a
/// big-endian `u32` byte count followed by the serialized body. Mirrors
/// `network.rs::write_frame`'s exact wire shape (COOLJAPAN policy: oxicode,
/// never bincode). `max_size` is a parameter (not just the module constant)
/// so tests can exercise the oversize guard cheaply.
async fn write_frame<W, T>(
    stream: &mut W,
    message: &T,
    peer_addr: SocketAddr,
    max_size: usize,
) -> Result<(), RaftTransportError>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let body =
        oxicode::serde::encode_to_vec(message, oxicode::config::standard()).map_err(|e| {
            RaftTransportError::Encode {
                addr: peer_addr,
                message: e.to_string(),
            }
        })?;
    if body.len() > max_size {
        return Err(RaftTransportError::FrameTooLarge {
            addr: peer_addr,
            len: body.len(),
            max: max_size,
        });
    }
    let io_err = |source: std::io::Error| RaftTransportError::Io {
        addr: peer_addr,
        source,
    };
    let len = body.len() as u32;
    stream.write_all(&len.to_be_bytes()).await.map_err(io_err)?;
    stream.write_all(&body).await.map_err(io_err)?;
    stream.flush().await.map_err(io_err)?;
    Ok(())
}

/// Read a single length-prefixed oxicode frame written by [`write_frame`].
/// Used by the client, which always expects exactly one response frame per
/// request — any EOF here is a genuine (reportable) failure.
async fn read_frame<R, T>(
    stream: &mut R,
    peer_addr: SocketAddr,
    max_size: usize,
) -> Result<T, RaftTransportError>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    try_read_frame(stream, peer_addr, max_size)
        .await?
        .ok_or(RaftTransportError::Io {
            addr: peer_addr,
            source: std::io::Error::from(std::io::ErrorKind::UnexpectedEof),
        })
}

/// Like [`read_frame`], but distinguishes a clean end-of-connection (no bytes
/// at all available for a new frame — `Ok(None)`) from a genuinely truncated
/// or corrupt frame (`Err`). Used by the server accept loop: a peer that
/// opens a connection, sends one request, reads the response, and closes
/// (exactly what [`OxirsRaftNetworkClient::call`] does) must end that
/// connection's task quietly rather than logging a spurious error.
async fn try_read_frame<R, T>(
    stream: &mut R,
    peer_addr: SocketAddr,
    max_size: usize,
) -> Result<Option<T>, RaftTransportError>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let io_err = |source: std::io::Error| RaftTransportError::Io {
        addr: peer_addr,
        source,
    };

    let mut len_buf = [0u8; 4];
    let first_byte = stream.read(&mut len_buf[0..1]).await.map_err(io_err)?;
    if first_byte == 0 {
        // Peer closed the connection cleanly between frames.
        return Ok(None);
    }
    stream
        .read_exact(&mut len_buf[1..4])
        .await
        .map_err(io_err)?;

    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_size {
        return Err(RaftTransportError::FrameTooLarge {
            addr: peer_addr,
            len,
            max: max_size,
        });
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await.map_err(io_err)?;

    let (message, _) = oxicode::serde::decode_from_slice(&body, oxicode::config::standard())
        .map_err(|e| RaftTransportError::Decode {
            addr: peer_addr,
            message: e.to_string(),
        })?;
    Ok(Some(message))
}

/// Factory that hands `openraft::Raft` a fresh [`OxirsRaftNetworkClient`] per
/// target node. Per [`RaftNetworkFactory::new_client`]'s contract this must
/// not eagerly connect — it just captures enough context (this node's own id,
/// for error reporting, and the shared peer-address registry as a fallback)
/// for the client to dial lazily on the first real RPC.
pub(crate) struct OxirsRaftNetworkFactory {
    self_id: OxirsNodeId,
    peer_addresses: Arc<RwLock<HashMap<OxirsNodeId, SocketAddr>>>,
}

impl OxirsRaftNetworkFactory {
    pub(crate) fn new(
        self_id: OxirsNodeId,
        peer_addresses: Arc<RwLock<HashMap<OxirsNodeId, SocketAddr>>>,
    ) -> Self {
        Self {
            self_id,
            peer_addresses,
        }
    }
}

impl RaftNetworkFactory<OxirsTypeConfig> for OxirsRaftNetworkFactory {
    type Network = OxirsRaftNetworkClient;

    async fn new_client(&mut self, target: OxirsNodeId, node: &BasicNode) -> Self::Network {
        // `node.addr` is what openraft itself was told about this target at
        // `initialize()`/membership-change time, so it is authoritative.
        // Parsing it here (once) avoids re-parsing on every RPC; the shared
        // registry remains as a fallback for the (should-never-happen) case
        // that it fails to parse.
        let addr = node.addr.parse::<SocketAddr>().ok();
        OxirsRaftNetworkClient {
            self_id: self.self_id,
            target,
            addr,
            peer_addresses: Arc::clone(&self.peer_addresses),
        }
    }
}

/// A [`RaftNetwork`] client bound to a single target node. Connects fresh (a
/// new [`TcpStream`]) per RPC, matching this crate's existing transport style
/// (`network.rs::NetworkManager::exchange_rpc`) rather than pooling
/// connections — openraft already serializes/paces replication per target
/// internally, so a fresh connection per call keeps this module simple
/// without a meaningful throughput cost.
pub(crate) struct OxirsRaftNetworkClient {
    self_id: OxirsNodeId,
    target: OxirsNodeId,
    addr: Option<SocketAddr>,
    peer_addresses: Arc<RwLock<HashMap<OxirsNodeId, SocketAddr>>>,
}

impl OxirsRaftNetworkClient {
    async fn resolve_address(&self) -> Result<SocketAddr, RaftTransportError> {
        if let Some(addr) = self.addr {
            return Ok(addr);
        }
        self.peer_addresses
            .read()
            .await
            .get(&self.target)
            .copied()
            .ok_or(RaftTransportError::UnknownPeer { peer: self.target })
    }

    /// Connect, send `request`, and read back one response frame — all
    /// within `hard_ttl`. Maps a local timeout to
    /// [`RaftTransportError::Timeout`] rather than letting the caller hang.
    async fn call(
        &self,
        addr: SocketAddr,
        request: RaftWireRequest,
        hard_ttl: Duration,
    ) -> Result<RaftWireResponse, RaftTransportError> {
        let attempt = async {
            let mut stream = TcpStream::connect(addr)
                .await
                .map_err(|e| RaftTransportError::Connect { addr, source: e })?;
            // Heartbeats and vote requests are small, latency-sensitive
            // messages; Nagle's algorithm (on by default) can hold a small
            // write for up to ~40ms waiting to coalesce it with more data
            // that will never come on a fresh, single-request-then-close
            // connection. That delay eats directly into the election-timeout
            // budget, so disable it. This is a one-line latency fix, not
            // connection pooling.
            if let Err(e) = stream.set_nodelay(true) {
                tracing::debug!("failed to set TCP_NODELAY on raft RPC connection to {addr}: {e}");
            }
            // This connection exists for exactly one request/response and is
            // then discarded (see this type's doc comment on why: fresh
            // connection per RPC, no pooling). A normal close leaves it in
            // TIME_WAIT on this (the connection-initiating) side for up to
            // ~60s; under the RPC rate a live cluster generates (a fresh
            // connection per heartbeat/vote/replication call, from every
            // node to every peer), that accumulates fast enough to exhaust
            // the local ephemeral port range under sustained load, which
            // then surfaces as `connect()` failing with EADDRNOTAVAIL
            // ("Can't assign requested address") — confirmed empirically
            // under this workspace's heavy shared-machine test load. Since
            // we already know there is nothing left to send once the
            // response has been read, an abortive close (RST) is safe here
            // and skips TIME_WAIT entirely. `set_zero_linger` (not the
            // deprecated `set_linger`) is the correct API for this: a
            // *non-zero* linger blocks the thread in `close()`, but a zero
            // linger does not, so Tokio carves it out as its own method.
            if let Err(e) = stream.set_zero_linger() {
                tracing::debug!(
                    "failed to set zero SO_LINGER on raft RPC connection to {addr}: {e}"
                );
            }
            write_frame(&mut stream, &request, addr, MAX_MESSAGE_SIZE).await?;
            read_frame(&mut stream, addr, MAX_MESSAGE_SIZE).await
        };
        let result = match tokio::time::timeout(hard_ttl, attempt).await {
            Ok(result) => result,
            Err(_elapsed) => Err(RaftTransportError::Timeout {
                addr,
                timeout: hard_ttl,
            }),
        };
        if let Err(ref e) = result {
            // Routine in a distributed system (a peer being temporarily or
            // permanently down is an expected condition, not a bug) — debug,
            // not warn/error. `RPCError::Unreachable`/`Network` above is what
            // actually surfaces this to openraft's own retry/backoff logic.
            tracing::debug!("raft RPC call to {addr} (hard_ttl={hard_ttl:?}) failed: {e}");
        }
        result
    }

    /// Map a transport-level failure to the openraft-level [`RPCError`] shape
    /// each `RaftNetwork` method must return. Local transport failures never
    /// panic and never fabricate a fake success — every branch here reports a
    /// real failure, just categorized the way openraft expects:
    /// - [`RaftTransportError::Timeout`] → `RPCError::Timeout` (with the
    ///   proper `action`/`id`/`target` context, so openraft's own timeout
    ///   bookkeeping/metrics stay accurate).
    /// - Connection failures / unknown peer → `RPCError::Unreachable`, which
    ///   tells openraft to back off before retrying.
    /// - Any other transport hiccup (mid-stream I/O error, codec failure,
    ///   oversize frame) → `RPCError::Network`, which openraft may retry
    ///   immediately.
    fn transport_err_to_rpc<E>(
        &self,
        err: RaftTransportError,
        action: RPCTypes,
        hard_ttl: Duration,
    ) -> RPCError<OxirsNodeId, BasicNode, RaftError<OxirsNodeId, E>>
    where
        E: std::error::Error,
    {
        match err {
            RaftTransportError::Timeout { .. } => RPCError::Timeout(Timeout {
                action,
                id: self.self_id,
                target: self.target,
                timeout: hard_ttl,
            }),
            RaftTransportError::Connect { .. } | RaftTransportError::UnknownPeer { .. } => {
                RPCError::Unreachable(Unreachable::new(&err))
            }
            RaftTransportError::Io { .. }
            | RaftTransportError::FrameTooLarge { .. }
            | RaftTransportError::Encode { .. }
            | RaftTransportError::Decode { .. }
            | RaftTransportError::MismatchedResponse { .. } => {
                RPCError::Network(OpenraftNetworkError::new(&err))
            }
        }
    }
}

impl RaftNetwork<OxirsTypeConfig> for OxirsRaftNetworkClient {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<OxirsTypeConfig>,
        option: RPCOption,
    ) -> Result<
        AppendEntriesResponse<OxirsNodeId>,
        RPCError<OxirsNodeId, BasicNode, RaftError<OxirsNodeId>>,
    > {
        let hard_ttl = option.hard_ttl();
        let addr = self
            .resolve_address()
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::AppendEntries, hard_ttl))?;
        let wire_resp = self
            .call(addr, RaftWireRequest::AppendEntries(rpc), hard_ttl)
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::AppendEntries, hard_ttl))?;
        match wire_resp {
            RaftWireResponse::AppendEntries(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(RPCError::Network(OpenraftNetworkError::new(
                &RaftTransportError::MismatchedResponse { addr },
            ))),
        }
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<OxirsNodeId>,
        option: RPCOption,
    ) -> Result<VoteResponse<OxirsNodeId>, RPCError<OxirsNodeId, BasicNode, RaftError<OxirsNodeId>>>
    {
        let hard_ttl = option.hard_ttl();
        let addr = self
            .resolve_address()
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::Vote, hard_ttl))?;
        let wire_resp = self
            .call(addr, RaftWireRequest::Vote(rpc), hard_ttl)
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::Vote, hard_ttl))?;
        match wire_resp {
            RaftWireResponse::Vote(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(RPCError::Network(OpenraftNetworkError::new(
                &RaftTransportError::MismatchedResponse { addr },
            ))),
        }
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<OxirsTypeConfig>,
        option: RPCOption,
    ) -> Result<
        InstallSnapshotResponse<OxirsNodeId>,
        RPCError<OxirsNodeId, BasicNode, RaftError<OxirsNodeId, InstallSnapshotError>>,
    > {
        let hard_ttl = option.hard_ttl();
        let addr = self
            .resolve_address()
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::InstallSnapshot, hard_ttl))?;
        let wire_resp = self
            .call(addr, RaftWireRequest::InstallSnapshot(rpc), hard_ttl)
            .await
            .map_err(|e| self.transport_err_to_rpc(e, RPCTypes::InstallSnapshot, hard_ttl))?;
        match wire_resp {
            RaftWireResponse::InstallSnapshot(result) => {
                result.map_err(|e| RPCError::RemoteError(RemoteError::new(self.target, e)))
            }
            _ => Err(RPCError::Network(OpenraftNetworkError::new(
                &RaftTransportError::MismatchedResponse { addr },
            ))),
        }
    }
}

/// Deliver a leadership-transfer ([`RaftWireRequest::TimeoutNow`]) signal to
/// `target` at `addr` and wait for its acknowledgement. Opens a fresh
/// connection (matching this module's per-RPC connection style), sends the
/// signal, and reads back the target's [`RaftWireResponse::TimeoutNow`]. Fails
/// loud on connect/codec/timeout errors, on a mismatched response kind, or if
/// the target reports its Raft core could not start an election — the caller
/// (`RaftNode::transfer_leadership`) surfaces any error rather than pretending
/// the handoff succeeded.
pub(crate) async fn send_timeout_now(
    self_id: OxirsNodeId,
    target: OxirsNodeId,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    // A generous fixed budget: the receiver only flips an internal flag and
    // returns, so this should be fast, but a loaded host can still add jitter.
    const TIMEOUT: Duration = Duration::from_secs(5);

    let attempt = async {
        let mut stream = TcpStream::connect(addr)
            .await
            .map_err(|e| RaftTransportError::Connect { addr, source: e })?;
        if let Err(e) = stream.set_nodelay(true) {
            tracing::debug!("failed to set TCP_NODELAY on leadership-transfer connection: {e}");
        }
        write_frame(
            &mut stream,
            &RaftWireRequest::TimeoutNow,
            addr,
            MAX_MESSAGE_SIZE,
        )
        .await?;
        let response: RaftWireResponse = read_frame(&mut stream, addr, MAX_MESSAGE_SIZE).await?;
        match response {
            RaftWireResponse::TimeoutNow(Ok(())) => Ok(()),
            RaftWireResponse::TimeoutNow(Err(message)) => Err(RaftTransportError::Decode {
                addr,
                message: format!("target {target} refused leadership transfer: {message}"),
            }),
            _ => Err(RaftTransportError::MismatchedResponse { addr }),
        }
    };

    let _ = self_id;
    match tokio::time::timeout(TIMEOUT, attempt).await {
        Ok(result) => result.map_err(anyhow::Error::from),
        Err(_) => Err(anyhow::Error::from(RaftTransportError::Timeout {
            addr,
            timeout: TIMEOUT,
        })),
    }
}

/// Inbound accept loop for Raft RPCs — the counterpart `network.rs` never
/// grew (see the comment at `NetworkManager::start_background_tasks`, which
/// explicitly skips a listener task because `TcpListener` doesn't support
/// cloning). Accepts connections forever, spawning one task per connection
/// that answers each request frame by calling straight into the LOCAL
/// `raft` instance's inbound handlers (`Raft::append_entries`/`vote`/
/// `install_snapshot` — distinct from the outbound `RaftNetwork` trait
/// methods above) and writing back the result. Never panics and never
/// fabricates a response: a malformed frame or I/O error just ends that one
/// connection's task; the accept loop itself keeps serving other peers.
pub(crate) async fn serve_raft_rpc(listener: TcpListener, raft: Raft<OxirsTypeConfig>) {
    loop {
        let (stream, peer_addr) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!("raft RPC listener accept() failed: {e}; continuing to listen");
                continue;
            }
        };
        // See the matching comment in `OxirsRaftNetworkClient::call`: small
        // latency-sensitive RPCs (heartbeats, votes) should not pay Nagle's
        // coalescing delay on either end of the connection.
        if let Err(e) = stream.set_nodelay(true) {
            tracing::debug!(
                "failed to set TCP_NODELAY on raft RPC connection from {peer_addr}: {e}"
            );
        }
        let raft = raft.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_raft_connection(stream, peer_addr, raft).await {
                tracing::debug!("raft RPC connection from {peer_addr} ended: {e}");
            }
        });
    }
}

/// Serve requests on a single accepted connection until the peer closes it
/// or a genuine I/O/codec error occurs.
async fn serve_raft_connection(
    mut stream: TcpStream,
    peer_addr: SocketAddr,
    raft: Raft<OxirsTypeConfig>,
) -> Result<(), RaftTransportError> {
    loop {
        let request: RaftWireRequest =
            match try_read_frame(&mut stream, peer_addr, MAX_MESSAGE_SIZE).await? {
                Some(request) => request,
                None => return Ok(()), // peer closed cleanly between requests
            };

        let response = match request {
            RaftWireRequest::AppendEntries(rpc) => {
                RaftWireResponse::AppendEntries(raft.append_entries(rpc).await)
            }
            RaftWireRequest::Vote(rpc) => RaftWireResponse::Vote(raft.vote(rpc).await),
            RaftWireRequest::InstallSnapshot(rpc) => {
                RaftWireResponse::InstallSnapshot(raft.install_snapshot(rpc).await)
            }
            RaftWireRequest::TimeoutNow => {
                // Leadership-transfer signal from the current leader: start an
                // election right now. The target's up-to-date log means Raft's
                // election restriction lets it win over stale voters.
                let result = raft.trigger().elect().await.map_err(|e| e.to_string());
                RaftWireResponse::TimeoutNow(result)
            }
        };

        write_frame(&mut stream, &response, peer_addr, MAX_MESSAGE_SIZE).await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::Vote;

    /// Round-tripping a request frame through `write_frame`/`read_frame`
    /// must reproduce an identical value, proving the oxicode framing (not
    /// bincode/serde_json) carries real openraft RPC types intact.
    #[tokio::test]
    async fn request_frame_round_trips() {
        let original = RaftWireRequest::Vote(VoteRequest::new(Vote::new(3, 7u64), None));

        let (mut a, mut b) = tokio::io::duplex(64 * 1024);
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("valid addr");
        let writer = tokio::spawn(async move {
            write_frame(&mut a, &original, addr, MAX_MESSAGE_SIZE)
                .await
                .expect("write_frame failed");
        });
        let decoded: RaftWireRequest = read_frame(&mut b, addr, MAX_MESSAGE_SIZE)
            .await
            .expect("read_frame failed");
        writer.await.expect("writer task panicked");

        match decoded {
            RaftWireRequest::Vote(v) => {
                assert_eq!(v.vote.leader_id().term, 3);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// A clean peer disconnect between frames must surface as `Ok(None)`,
    /// not an error — this is what lets `serve_raft_connection` end quietly
    /// for the (expected, normal) "one request per connection" client style.
    #[tokio::test]
    async fn try_read_frame_reports_clean_eof_as_none() {
        let (a, b) = tokio::io::duplex(1024);
        drop(a); // simulate the peer closing immediately
        let mut b = b;
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("valid addr");
        let result: Option<RaftWireRequest> = try_read_frame(&mut b, addr, MAX_MESSAGE_SIZE)
            .await
            .expect("try_read_frame should not error on clean EOF");
        assert!(result.is_none());
    }

    /// An oversize outgoing frame must be rejected rather than silently
    /// truncated or sent anyway.
    #[tokio::test]
    async fn write_frame_rejects_oversize_message() {
        let msg = RaftWireRequest::Vote(VoteRequest::new(Vote::new(1, 1u64), None));
        let (mut a, _b) = tokio::io::duplex(1024);
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("valid addr");

        // A max_size of 1 byte guarantees the oversize guard trips, exactly
        // mirroring `network.rs::test_frame_rejects_oversize_message`.
        let err = write_frame(&mut a, &msg, addr, 1)
            .await
            .expect_err("oversize frame must be rejected");
        assert!(
            matches!(err, RaftTransportError::FrameTooLarge { .. }),
            "unexpected error: {err}"
        );
    }

    /// The client must report an honest, categorized error (not a fabricated
    /// success) when the target peer's address is unknown.
    #[tokio::test]
    async fn resolve_address_fails_loudly_for_unknown_peer() {
        let client = OxirsRaftNetworkClient {
            self_id: 1,
            target: 99,
            addr: None,
            peer_addresses: Arc::new(RwLock::new(HashMap::new())),
        };
        let err = client
            .resolve_address()
            .await
            .expect_err("must fail for an unregistered peer");
        assert!(matches!(err, RaftTransportError::UnknownPeer { peer: 99 }));
    }
}
