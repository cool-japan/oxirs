//! A single cluster node backed by a real TCP listener.
//!
//! `TcpClusterNode` listens on a dynamically-assigned local port (bind to
//! `0.0.0.0:0` and read back the OS-assigned address via
//! `TcpListener::local_addr()`).  Peers are registered after construction so
//! that `TcpClusterNetwork` can wire the mesh before gossip rounds start.
//!
//! ## Gossip loop
//!
//! Every `gossip_interval_ms` the node selects
//! `fanout.resolve(peers.len())` peers uniformly at random (using the
//! `scirs2_core::random` generator per COOLJAPAN policy) and connects a
//! fresh TCP stream to each.  It sends one `ClusterMessage::Gossip` for
//! every entry in its local `GossipState`.  LWW reconciliation on the
//! receiver side means stale re-sends are idempotent.
//!
//! ## Listener loop
//!
//! Each accepted connection is handled in its own spawned task.  Messages are
//! read in a loop until the connection closes or the `CancellationToken` fires.
//! Supported incoming messages:
//! - `Gossip` → merge into local state
//! - `Ping`   → reply with `Pong`
//! - `Replicate` → reply with `ReplicateAck { success: true }`
//! - Others are silently ignored
//!
//! ## Transport security
//!
//! Without [`TcpNodeConfig::tls`], anyone who can connect can read the gossip
//! and change the state, so such a node should only listen on loopback (a
//! warning is logged otherwise). With it, every connection runs over mutually
//! authenticated TLS: the listener accepts only clients with a trusted
//! certificate, and gossip is sent only to peers that present a trusted
//! certificate for the expected name.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use rustls::pki_types::ServerName;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{interval, timeout};
use tokio_util::sync::CancellationToken;

use crate::gossip::fanout::GossipFanout;
use crate::tls::TlsManager;

use super::codec::{ClusterMessage, MessageCodec};

// ─────────────────────────────────────────────────────────────────────────────
// GossipState
// ─────────────────────────────────────────────────────────────────────────────

/// Shared key-value state propagated via gossip.
///
/// Each entry is `(value, version)`.  Versions are compared on merge:
/// an incoming entry replaces the stored one only when its version is
/// strictly greater.
#[derive(Default, Clone)]
pub struct GossipState {
    /// `key → (value, version)` — last-write-wins by version number.
    pub entries: HashMap<String, (u64, u64)>,
}

impl GossipState {
    /// Merge an incoming entry.
    ///
    /// Returns `true` if the stored entry was updated (i.e. `version` was
    /// strictly greater than the previously stored version, or the key is new).
    pub fn set(&mut self, key: &str, value: u64, version: u64) -> bool {
        let entry = self.entries.entry(key.to_owned()).or_insert((0, 0));
        if version > entry.1 {
            *entry = (value, version);
            true
        } else {
            false
        }
    }

    /// Return `(value, version)` for `key`, or `None` if absent.
    pub fn get(&self, key: &str) -> Option<(u64, u64)> {
        self.entries.get(key).copied()
    }

    /// Number of distinct keys stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the state has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// How long a TLS handshake with a peer may take.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Configuration for a single `TcpClusterNode`.
#[derive(Debug, Clone)]
pub struct TcpNodeConfig {
    /// Human-readable node identifier (used in gossip messages).
    pub node_id: String,
    /// Address to bind; use `127.0.0.1:0` to let the OS assign a free port.
    /// Without [`TcpNodeConfig::tls`], bind only to loopback.
    pub bind_addr: SocketAddr,
    /// Fanout policy for gossip peer selection.
    pub fanout: GossipFanout,
    /// How often to run a gossip round (milliseconds).
    pub gossip_interval_ms: u64,
    /// Mutually authenticated TLS for every connection of this node. It is
    /// initialized by [`TcpClusterNode::start`], which fails if it cannot be.
    /// `None` runs plaintext and unauthenticated.
    pub tls: Option<Arc<TlsManager>>,
}

impl TcpNodeConfig {
    /// Convenience constructor binding to `127.0.0.1:port`.
    ///
    /// Pass `port = 0` to let the OS pick a free ephemeral port.
    pub fn localhost(node_id: &str, port: u16) -> Self {
        Self {
            node_id: node_id.to_owned(),
            bind_addr: SocketAddr::from(([127, 0, 0, 1], port)),
            fanout: GossipFanout::Unbounded,
            gossip_interval_ms: 50,
            tls: None,
        }
    }

    /// Run every connection of the node over mutually authenticated TLS.
    pub fn with_tls(mut self, tls: Arc<TlsManager>) -> Self {
        self.tls = Some(tls);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`TcpClusterNode`] operations.
#[derive(Debug)]
pub enum TcpNodeError {
    /// Failed to bind the TCP listener.
    BindError(std::io::Error),
    /// Failed to send a message to a peer.
    SendError(String),
    /// TLS was configured but could not be initialized.
    Tls(String),
    /// Node has been shut down.
    Shutdown,
}

impl std::fmt::Display for TcpNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TcpNodeError::BindError(e) => write!(f, "TCP bind failed: {e}"),
            TcpNodeError::SendError(s) => write!(f, "send error: {s}"),
            TcpNodeError::Tls(s) => write!(f, "TLS could not be initialized: {s}"),
            TcpNodeError::Shutdown => write!(f, "node has been shut down"),
        }
    }
}

impl std::error::Error for TcpNodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TcpNodeError::BindError(e) => Some(e),
            TcpNodeError::SendError(_) | TcpNodeError::Tls(_) | TcpNodeError::Shutdown => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Node
// ─────────────────────────────────────────────────────────────────────────────

/// A single cluster node running real TCP sockets.
///
/// Spawn with [`TcpClusterNode::start`], wire peers with [`add_peer`][Self::add_peer],
/// read/write state with [`get`][Self::get] / [`set`][Self::set],
/// and shut down with [`shutdown`][Self::shutdown].
pub struct TcpClusterNode {
    config: TcpNodeConfig,
    /// Actual bound address (may differ from `config.bind_addr` when port was 0).
    bound_addr: SocketAddr,
    state: Arc<RwLock<GossipState>>,
    peers: Arc<RwLock<Vec<Peer>>>,
    cancel: CancellationToken,
    /// Global version counter: monotonically incremented on every local `set`.
    version: Arc<AtomicU64>,
}

impl TcpClusterNode {
    /// Bind the TCP listener and start background tasks.
    ///
    /// Returns once the listener is bound and the background tasks are running.
    ///
    /// # Errors
    ///
    /// Returns [`TcpNodeError::Tls`] if TLS is configured but cannot be
    /// initialized (nothing is bound then), and [`TcpNodeError::BindError`]
    /// if the port could not be acquired.
    pub async fn start(config: TcpNodeConfig) -> Result<Self, TcpNodeError> {
        match &config.tls {
            Some(tls) => tls
                .initialize()
                .await
                .map_err(|e| TcpNodeError::Tls(e.to_string()))?,
            None if !config.bind_addr.ip().is_loopback() => tracing::warn!(
                "TCP cluster node {} listens on {} without TLS: anyone who can connect can \
                 read the gossip and change the state",
                config.node_id,
                config.bind_addr
            ),
            None => {}
        }

        let listener = TcpListener::bind(config.bind_addr)
            .await
            .map_err(TcpNodeError::BindError)?;
        let bound_addr = listener.local_addr().map_err(TcpNodeError::BindError)?;

        let state: Arc<RwLock<GossipState>> = Arc::default();
        let peers: Arc<RwLock<Vec<Peer>>> = Arc::default();
        let cancel = CancellationToken::new();
        let version = Arc::new(AtomicU64::new(1));

        // Spawn listener task.
        let state_clone = Arc::clone(&state);
        let cancel_clone = cancel.clone();
        let node_id_clone = config.node_id.clone();
        let tls_listener = config.tls.clone();
        tokio::spawn(async move {
            run_listener(
                listener,
                state_clone,
                node_id_clone,
                cancel_clone,
                tls_listener,
            )
            .await;
        });

        // Spawn gossip timer task.
        let state_gossip = Arc::clone(&state);
        let peers_gossip = Arc::clone(&peers);
        let cancel_gossip = cancel.clone();
        let gossip_interval = config.gossip_interval_ms;
        let fanout = config.fanout;
        let node_id_gossip = config.node_id.clone();
        let tls_gossip = config.tls.clone();
        tokio::spawn(async move {
            run_gossip_loop(
                node_id_gossip,
                fanout,
                gossip_interval,
                state_gossip,
                peers_gossip,
                cancel_gossip,
                tls_gossip,
            )
            .await;
        });

        Ok(Self {
            config,
            bound_addr,
            state,
            peers,
            cancel,
            version,
        })
    }

    /// Register a peer address to gossip with. With TLS, the peer must
    /// present a trusted certificate for the IP address of `addr`.
    pub fn add_peer(&self, addr: SocketAddr) {
        self.peers.write().push(Peer {
            addr,
            server_name: None,
        });
    }

    /// Register a peer whose TLS certificate is issued to `server_name`
    /// (for example `node-3`). Without TLS the name is not used.
    pub fn add_peer_with_server_name(&self, addr: SocketAddr, server_name: impl Into<String>) {
        self.peers.write().push(Peer {
            addr,
            server_name: Some(server_name.into()),
        });
    }

    /// Store a key-value pair in the local gossip state.
    ///
    /// The internal monotone version counter is incremented so that subsequent
    /// gossip rounds carry a strictly greater version than any prior one,
    /// guaranteeing LWW ordering across nodes in a single-process test.
    pub fn set(&self, key: &str, value: u64) {
        let ver = self.version.fetch_add(1, Ordering::Relaxed) + 1;
        self.state.write().set(key, value, ver);
    }

    /// Store a key-value pair with an explicit version number.
    ///
    /// This bypasses the internal monotone counter and is intended for test
    /// scenarios that need deterministic LWW ordering (e.g. to prove that a
    /// higher explicit version beats a lower one regardless of call timing).
    ///
    /// The internal counter is advanced to `version.max(current_counter)` so
    /// subsequent [`set`][Self::set] calls still produce strictly greater
    /// versions.
    pub fn set_with_version(&self, key: &str, value: u64, version: u64) {
        self.state.write().set(key, value, version);
        // Keep the internal counter at least at `version` so future `set`
        // calls produce strictly larger versions.
        let mut current = self.version.load(Ordering::Relaxed);
        loop {
            if current >= version {
                break;
            }
            match self.version.compare_exchange_weak(
                current,
                version,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current = v,
            }
        }
    }

    /// Read the current value for `key` from the local state.
    ///
    /// Returns `None` if the key has not converged to this node yet.
    pub fn get(&self, key: &str) -> Option<u64> {
        self.state.read().get(key).map(|(v, _ver)| v)
    }

    /// Number of distinct keys currently in the local state.
    pub fn state_len(&self) -> usize {
        self.state.read().len()
    }

    /// Signal all background tasks to stop.
    ///
    /// Does not wait for task completion; caller should `tokio::time::sleep`
    /// or `JoinHandle::await` if ordering matters.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// The logical node identifier.
    pub fn node_id(&self) -> &str {
        &self.config.node_id
    }

    /// The OS-assigned bound address (canonical for peer registration).
    pub fn addr(&self) -> SocketAddr {
        self.bound_addr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background tasks
// ─────────────────────────────────────────────────────────────────────────────

/// A gossip peer: where to connect, and the name its TLS certificate must
/// carry (its IP address when `None`).
#[derive(Debug, Clone)]
struct Peer {
    addr: SocketAddr,
    server_name: Option<String>,
}

/// Listener task: accept incoming connections and dispatch them to handler tasks.
async fn run_listener(
    listener: TcpListener,
    state: Arc<RwLock<GossipState>>,
    node_id: String,
    cancel: CancellationToken,
    tls: Option<Arc<TlsManager>>,
) {
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = listener.accept() => {
                match result {
                    Ok((stream, peer)) => {
                        let state_clone = Arc::clone(&state);
                        let node_id_clone = node_id.clone();
                        let cancel_clone = cancel.clone();
                        let tls = tls.clone();
                        tokio::spawn(async move {
                            match tls {
                                None => {
                                    handle_connection(stream, state_clone, node_id_clone, cancel_clone).await;
                                }
                                Some(tls) => match accept_tls(&tls, stream).await {
                                    Ok(stream) => {
                                        handle_connection(stream, state_clone, node_id_clone, cancel_clone).await;
                                    }
                                    Err(e) => {
                                        tracing::warn!("refused TCP cluster connection from {peer}: {e}");
                                    }
                                },
                            }
                        });
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

/// Complete the server side of a TLS handshake; the client must present a
/// trusted certificate.
async fn accept_tls(
    tls: &TlsManager,
    stream: TcpStream,
) -> Result<tokio_rustls::server::TlsStream<TcpStream>, String> {
    let acceptor = tls.get_acceptor().await.map_err(|e| e.to_string())?;
    match timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(e)) => Err(format!("TLS handshake failed: {e}")),
        Err(_) => Err(format!(
            "TLS handshake timed out after {TLS_HANDSHAKE_TIMEOUT:?}"
        )),
    }
}

/// Handle a single accepted connection.
///
/// Reads messages until EOF or cancellation, then closes the stream.
async fn handle_connection<S>(
    stream: S,
    state: Arc<RwLock<GossipState>>,
    node_id: String,
    cancel: CancellationToken,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (mut reader, mut writer) = tokio::io::split(stream);

    loop {
        let msg = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = MessageCodec::read(&mut reader) => {
                match result {
                    Ok(m) => m,
                    Err(_) => break, // Connection closed or error
                }
            }
        };

        match msg {
            ClusterMessage::Gossip {
                key,
                value,
                version,
                ..
            } => {
                state.write().set(&key, value, version);
            }
            ClusterMessage::Ping { sender_id: _, seq } => {
                let pong = ClusterMessage::Pong {
                    sender_id: node_id.clone(),
                    seq,
                };
                if MessageCodec::write(&mut writer, &pong).await.is_err() {
                    break;
                }
            }
            ClusterMessage::Replicate { index, .. } => {
                let ack = ClusterMessage::ReplicateAck {
                    follower_id: node_id.clone(),
                    index,
                    success: true,
                };
                if MessageCodec::write(&mut writer, &ack).await.is_err() {
                    break;
                }
            }
            // Pong and ReplicateAck are not expected on an incoming connection
            // from a peer — ignore them silently.
            ClusterMessage::Pong { .. } | ClusterMessage::ReplicateAck { .. } => {}
        }
    }
}

/// Gossip timer task: periodically push local state to a random peer subset.
async fn run_gossip_loop(
    node_id: String,
    fanout: GossipFanout,
    interval_ms: u64,
    state: Arc<RwLock<GossipState>>,
    peers: Arc<RwLock<Vec<Peer>>>,
    cancel: CancellationToken,
    tls: Option<Arc<TlsManager>>,
) {
    let mut ticker = interval(Duration::from_millis(interval_ms));

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {}
        }

        // Snapshot state and peers under lock, then release.
        let snapshot: Vec<(String, u64, u64)> = {
            let st = state.read();
            st.entries
                .iter()
                .map(|(k, (v, ver))| (k.clone(), *v, *ver))
                .collect()
        };

        if snapshot.is_empty() {
            continue;
        }

        let selected = {
            let all_peers: Vec<Peer> = peers.read().clone();
            let count = fanout.resolve(all_peers.len());
            select_random_peers(&all_peers, count)
        };

        for peer in &selected {
            gossip_to_peer(&node_id, peer, &snapshot, tls.as_deref()).await;
        }
    }
}

/// Select up to `count` addresses uniformly at random (without replacement).
///
/// Uses a simple Fisher-Yates partial shuffle without an external RNG crate.
/// We seed from `std::time::SystemTime` to get variation across rounds.
fn select_random_peers<T: Clone>(peers: &[T], count: usize) -> Vec<T> {
    if count == 0 || peers.is_empty() {
        return Vec::new();
    }
    let count = count.min(peers.len());
    let mut indices: Vec<usize> = (0..peers.len()).collect();

    // Cheap determinism-free shuffle using a seed from system time.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;

    // Xorshift64 PRNG — good enough for random peer selection.
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    if state == 0 {
        state = 1;
    }

    for i in 0..count {
        // Generate next random index in [i, len).
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = i + (state as usize % (peers.len() - i));
        indices.swap(i, j);
    }

    indices[..count].iter().map(|&i| peers[i].clone()).collect()
}

/// Open a short-lived connection to `peer` (over TLS when `tls` is set) and
/// send all snapshot entries.
async fn gossip_to_peer(
    node_id: &str,
    peer: &Peer,
    snapshot: &[(String, u64, u64)],
    tls: Option<&TlsManager>,
) {
    let Ok(stream) = TcpStream::connect(peer.addr).await else {
        return; // Peer might not be ready yet; skip this round.
    };
    let Some(tls) = tls else {
        send_snapshot(stream, node_id, snapshot).await;
        return;
    };

    let server_name = match &peer.server_name {
        Some(name) => match ServerName::try_from(name.clone()) {
            Ok(name) => name,
            Err(e) => {
                tracing::warn!(
                    "peer {} has an invalid server name {name:?}: {e}",
                    peer.addr
                );
                return;
            }
        },
        None => ServerName::from(peer.addr.ip()),
    };
    let connector = match tls.get_connector().await {
        Ok(connector) => connector,
        Err(e) => {
            tracing::warn!("no TLS connector for gossip to {}: {e}", peer.addr);
            return;
        }
    };
    match timeout(
        TLS_HANDSHAKE_TIMEOUT,
        connector.connect(server_name, stream),
    )
    .await
    {
        Ok(Ok(stream)) => send_snapshot(stream, node_id, snapshot).await,
        Ok(Err(e)) => tracing::debug!("TLS handshake with peer {} failed: {e}", peer.addr),
        Err(_) => tracing::debug!("TLS handshake with peer {} timed out", peer.addr),
    }
}

/// Write one `Gossip` message per snapshot entry, then close the stream.
async fn send_snapshot<S>(mut stream: S, node_id: &str, snapshot: &[(String, u64, u64)])
where
    S: AsyncWrite + Unpin,
{
    for (key, value, version) in snapshot {
        let msg = ClusterMessage::Gossip {
            sender_id: node_id.to_owned(),
            key: key.clone(),
            value: *value,
            version: *version,
        };
        if MessageCodec::write(&mut stream, &msg).await.is_err() {
            return; // Stream closed; abort this peer's batch.
        }
    }
    // An orderly close (TLS close_notify, TCP FIN) after the last message.
    let _ = stream.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_state_lww() {
        let mut state = GossipState::default();
        assert!(state.set("k", 10, 1));
        assert!(!state.set("k", 99, 1)); // same version — ignored
        assert!(state.set("k", 42, 2)); // higher version — accepted
        assert_eq!(state.get("k"), Some((42, 2)));
    }

    #[test]
    fn test_gossip_state_len() {
        let mut state = GossipState::default();
        state.set("a", 1, 1);
        state.set("b", 2, 1);
        assert_eq!(state.len(), 2);
        assert!(!state.is_empty());
    }

    #[test]
    fn test_node_config_localhost() {
        let cfg = TcpNodeConfig::localhost("n1", 0);
        assert_eq!(cfg.node_id, "n1");
        assert_eq!(cfg.bind_addr.port(), 0);
    }

    #[test]
    fn test_select_random_peers_empty() {
        let peers: Vec<SocketAddr> = vec![];
        let result = select_random_peers(&peers, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_random_peers_count_capped() {
        let peers: Vec<SocketAddr> = (0..5)
            .map(|i| SocketAddr::from(([127, 0, 0, 1], 10000 + i)))
            .collect();
        let result = select_random_peers(&peers, 100);
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_start_and_addr() {
        let cfg = TcpNodeConfig::localhost("test-node", 0);
        let node = TcpClusterNode::start(cfg).await.expect("start");
        assert_eq!(node.node_id(), "test-node");
        assert_ne!(
            node.addr().port(),
            0,
            "OS should have assigned a non-zero port"
        );
        node.shutdown();
    }

    #[tokio::test]
    async fn test_set_and_get() {
        let cfg = TcpNodeConfig::localhost("n1", 0);
        let node = TcpClusterNode::start(cfg).await.expect("start");
        node.set("foo", 42);
        assert_eq!(node.get("foo"), Some(42));
        node.shutdown();
    }
}

#[cfg(test)]
mod tls_tests {
    use super::*;
    use crate::tcp_cluster::network::TcpClusterNetwork;
    use crate::tls::test_support::{install_provider, node_config};
    use crate::tls::TlsConfig;
    use std::path::Path;

    /// TLS identities for `ids`, each in its own directory under `dir`, that
    /// all trust one bundle holding the certificates of `trusted`.
    async fn identities(dir: &Path, ids: &[u64], trusted: &[u64]) -> Vec<Arc<TlsManager>> {
        install_provider();
        let bundle = dir.join("cluster.pem");
        let mut pems = String::new();
        let mut configs = Vec::new();
        for &id in ids {
            let node_dir = dir.join(format!("node-{id}"));
            std::fs::create_dir_all(&node_dir).expect("cert dir");
            let (config, pem) = node_config(&node_dir, id, &bundle, true).await;
            if trusted.contains(&id) {
                pems.push_str(&pem);
            }
            configs.push((id, config));
        }
        std::fs::write(&bundle, pems).expect("bundle");
        configs
            .into_iter()
            .map(|(id, config)| Arc::new(TlsManager::new(config, id)))
            .collect()
    }

    /// Send one gossip message in plaintext to `addr`.
    async fn inject_plaintext_gossip(addr: SocketAddr, key: &str) {
        let mut stream = TcpStream::connect(addr).await.expect("connect");
        let message = ClusterMessage::Gossip {
            sender_id: "intruder".into(),
            key: key.into(),
            value: 666,
            version: u64::MAX / 2,
        };
        // The write may fail once a TLS node closes the connection.
        let _ = MessageCodec::write(&mut stream, &message).await;
    }

    #[tokio::test]
    async fn test_tls_mesh_converges() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let tls = identities(dir.path(), &[1, 2, 3], &[1, 2, 3]).await;
        let net = TcpClusterNetwork::spawn_tls(tls, GossipFanout::Unbounded, 20)
            .await
            .expect("spawn over TLS");
        net.set_on(0, "k", 9);
        let stats = net.wait_converged("k", 9, Duration::from_secs(5)).await;
        assert!(stats.converged, "the TLS mesh did not converge: {stats:?}");
        net.shutdown_all();
    }

    /// A node with TLS does not apply gossip from a plaintext connection,
    /// while a node without TLS does.
    #[tokio::test]
    async fn test_tls_node_ignores_plaintext_gossip() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let tls = identities(dir.path(), &[1], &[1]).await;
        let secured = TcpClusterNode::start(
            TcpNodeConfig::localhost("secured", 0).with_tls(Arc::clone(&tls[0])),
        )
        .await
        .expect("start with TLS");
        let open = TcpClusterNode::start(TcpNodeConfig::localhost("open", 0))
            .await
            .expect("start");

        inject_plaintext_gossip(open.addr(), "injected").await;
        inject_plaintext_gossip(secured.addr(), "injected").await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while open.get("injected").is_none() && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            open.get("injected"),
            Some(666),
            "the plaintext control saw nothing"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            secured.get("injected"),
            None,
            "a TLS node applied plaintext gossip"
        );
        secured.shutdown();
        open.shutdown();
    }

    /// A node whose certificate the receiver does not trust cannot gossip
    /// into it, even over TLS.
    #[tokio::test]
    async fn test_untrusted_node_cannot_gossip() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        // Both trust node 1 only: node 2's certificate is refused by node 1.
        let tls = identities(dir.path(), &[1, 2], &[1]).await;
        let receiver = TcpClusterNode::start(
            TcpNodeConfig::localhost("receiver", 0).with_tls(Arc::clone(&tls[0])),
        )
        .await
        .expect("receiver");
        let mut sender_config = TcpNodeConfig::localhost("sender", 0).with_tls(Arc::clone(&tls[1]));
        sender_config.gossip_interval_ms = 20;
        let sender = TcpClusterNode::start(sender_config).await.expect("sender");
        sender.add_peer_with_server_name(receiver.addr(), "node-1");
        sender.set("from-untrusted", 1);

        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(receiver.get("from-untrusted"), None);
        receiver.shutdown();
        sender.shutdown();
    }

    #[tokio::test]
    async fn test_tls_that_cannot_initialize_stops_start() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let broken = Arc::new(TlsManager::new(
            TlsConfig {
                enabled: true,
                cert_dir: dir.path().to_path_buf(),
                ca_file: None,
                auto_rotate: false,
                ..Default::default()
            },
            1,
        ));
        let outcome =
            TcpClusterNode::start(TcpNodeConfig::localhost("broken", 0).with_tls(broken)).await;
        assert!(matches!(outcome, Err(TcpNodeError::Tls(_))));
    }
}
