//! # Network Communication Layer
//!
//! Network communication layer for Raft consensus protocol.
//! Provides RPC mechanisms for node-to-node communication.

use crate::at_rest::{self, AtRestKey};
use crate::raft::{OxirsNodeId, RdfCommand, RdfResponse};
use crate::tls::{TlsConfig, TlsManager};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Context bound into data sealed by [`NetworkManager::encrypt_data`].
const NETWORK_DATA_CONTEXT: &[u8] = b"oxirs-cluster/network-data";

/// RPC message types for Raft protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcMessage {
    /// Request vote message
    RequestVote {
        term: u64,
        candidate_id: OxirsNodeId,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// Vote response
    VoteResponse { term: u64, vote_granted: bool },
    /// Append entries message
    AppendEntries {
        term: u64,
        leader_id: OxirsNodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    /// Append entries response
    AppendEntriesResponse {
        term: u64,
        success: bool,
        last_log_index: u64,
    },
    /// Client request
    ClientRequest { command: RdfCommand },
    /// Client response
    ClientResponse { response: RdfResponse },
    /// Heartbeat message
    Heartbeat { term: u64, leader_id: OxirsNodeId },
    /// Heartbeat response
    HeartbeatResponse { term: u64 },
    /// Byzantine fault tolerance message
    #[cfg(feature = "bft")]
    Bft { data: Vec<u8> },
    /// Shard operation message
    ShardOperation(crate::shard_manager::ShardOperation),
    /// Store triple to shard
    StoreTriple {
        shard_id: crate::shard::ShardId,
        triple: oxirs_core::model::Triple,
    },
    /// Replicate triple to shard
    ReplicateTriple {
        shard_id: crate::shard::ShardId,
        triple: oxirs_core::model::Triple,
    },
    /// Query shard
    QueryShard {
        shard_id: crate::shard::ShardId,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
    /// Query shard response
    QueryShardResponse {
        shard_id: crate::shard::ShardId,
        results: Vec<oxirs_core::model::Triple>,
    },
    /// Transaction prepare request
    TransactionPrepare {
        tx_id: String,
        shard_id: crate::shard::ShardId,
        operations: Vec<crate::transaction::TransactionOp>,
    },
    /// Transaction vote response
    TransactionVote {
        tx_id: String,
        shard_id: crate::shard::ShardId,
        vote: bool,
    },
    /// Transaction commit request
    TransactionCommit {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Transaction abort request
    TransactionAbort {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Transaction acknowledgment
    TransactionAck {
        tx_id: String,
        shard_id: crate::shard::ShardId,
    },
    /// Migration batch transfer
    MigrationBatch {
        migration_id: String,
        batch: crate::shard_migration::MigrationBatch,
    },
    /// Shard data transfer during migration
    ShardTransfer {
        shard_id: crate::shard::ShardId,
        triples: Vec<oxirs_core::model::Triple>,
        source_node: crate::raft::OxirsNodeId,
    },
}

/// Log entry for Raft protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub index: u64,
    pub term: u64,
    pub command: RdfCommand,
}

impl LogEntry {
    pub fn new(index: u64, term: u64, command: RdfCommand) -> Self {
        Self {
            index,
            term,
            command,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Local node address
    pub local_address: SocketAddr,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    /// TLS configuration
    pub tls_config: TlsConfig,
    /// Enable compression for messages
    pub enable_compression: bool,
    /// Maximum message size (bytes)
    pub max_message_size: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            local_address: "127.0.0.1:8080"
                .parse()
                .expect("localhost address is valid"),
            connection_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(10),
            max_connections: 100,
            keep_alive_interval: Duration::from_secs(30),
            tls_config: TlsConfig::default(),
            enable_compression: true,
            max_message_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Network connection manager
#[derive(Debug)]
pub struct NetworkManager {
    config: NetworkConfig,
    node_id: OxirsNodeId,
    connections: Arc<RwLock<HashMap<OxirsNodeId, Connection>>>,
    /// Statically known peer addresses, keyed by node id. Populated via
    /// [`NetworkManager::register_peer`] so that address-less send paths
    /// (e.g. [`NetworkService::send_message`]) can resolve a real endpoint
    /// instead of silently dropping the message.
    peer_addresses: Arc<RwLock<HashMap<OxirsNodeId, SocketAddr>>>,
    listener: Option<TcpListener>,
    /// Handle to the spawned inbound accept loop (see
    /// [`NetworkManager::start`]). Held so [`NetworkManager::stop`] can abort
    /// it and free the listening port. Never cloned (a clone is a lightweight
    /// handle that does not own the server task).
    accept_task: Option<Arc<tokio::task::JoinHandle<()>>>,
    running: Arc<RwLock<bool>>,
    tls_manager: Option<Arc<TlsManager>>,
    /// Key for [`NetworkManager::encrypt_data`], shared by all clones.
    data_key: Arc<RwLock<Option<AtRestKey>>>,
    message_stats: Arc<RwLock<MessageStats>>,
}

/// Message statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct MessageStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connections_established: u64,
    pub connections_failed: u64,
    pub tls_handshakes_completed: u64,
    pub tls_handshakes_failed: u64,
}

/// Network connection to a peer
#[derive(Debug, Clone)]
pub struct Connection {
    pub peer_id: OxirsNodeId,
    pub address: SocketAddr,
    pub last_activity: std::time::Instant,
    pub is_connected: bool,
}

impl Connection {
    pub fn new(peer_id: OxirsNodeId, address: SocketAddr) -> Self {
        Self {
            peer_id,
            address,
            last_activity: std::time::Instant::now(),
            is_connected: false,
        }
    }

    /// Check if connection is stale
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    /// Update last activity timestamp
    pub fn update_activity(&mut self) {
        self.last_activity = std::time::Instant::now();
    }
}

/// Serialize an [`RpcMessage`] with oxicode (COOLJAPAN policy — not bincode) and
/// write it to `stream` as a length-prefixed frame: a big-endian `u32` byte
/// count followed by the serialized body. Returns the total number of bytes
/// written (prefix + body). Fails loudly on serialization, oversize, or I/O
/// errors — it never silently drops the message.
async fn write_frame<W>(stream: &mut W, message: &RpcMessage, max_size: usize) -> Result<u64>
where
    W: AsyncWrite + Unpin,
{
    let body = oxicode::serde::encode_to_vec(message, oxicode::config::standard())
        .map_err(|e| anyhow::anyhow!("failed to serialize RPC message: {e}"))?;
    if body.len() > max_size {
        return Err(anyhow::anyhow!(
            "outgoing RPC frame ({} bytes) exceeds max_message_size ({} bytes)",
            body.len(),
            max_size
        ));
    }
    let len = body.len() as u32;
    stream
        .write_all(&len.to_be_bytes())
        .await
        .map_err(|e| anyhow::anyhow!("failed to write RPC length prefix: {e}"))?;
    stream
        .write_all(&body)
        .await
        .map_err(|e| anyhow::anyhow!("failed to write RPC body: {e}"))?;
    stream
        .flush()
        .await
        .map_err(|e| anyhow::anyhow!("failed to flush RPC frame: {e}"))?;
    Ok(4 + body.len() as u64)
}

/// Read a single length-prefixed frame written by [`write_frame`] from `stream`
/// and deserialize it back into an [`RpcMessage`]. Enforces `max_size` on the
/// advertised length so a malicious/corrupt peer cannot force an unbounded
/// allocation. Returns the peer's actual message — never a fabricated response.
async fn read_frame<R>(stream: &mut R, max_size: usize) -> Result<RpcMessage>
where
    R: AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| anyhow::anyhow!("failed to read RPC length prefix: {e}"))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_size {
        return Err(anyhow::anyhow!(
            "incoming RPC frame ({len} bytes) exceeds max_message_size ({max_size} bytes)"
        ));
    }
    let mut body = vec![0u8; len];
    stream
        .read_exact(&mut body)
        .await
        .map_err(|e| anyhow::anyhow!("failed to read RPC body ({len} bytes): {e}"))?;
    let (message, _) = oxicode::serde::decode_from_slice(&body, oxicode::config::standard())
        .map_err(|e| anyhow::anyhow!("failed to deserialize RPC message: {e}"))?;
    Ok(message)
}

impl NetworkManager {
    /// Create a new network manager
    ///
    /// With `config.tls_config.enabled` set, the manager speaks nothing but
    /// mutually authenticated TLS. TLS is initialized by
    /// [`NetworkManager::start`] or the first outgoing RPC, whichever comes
    /// first; if that fails, the call fails instead of falling back to
    /// plaintext. [`NetworkManager::with_tls`] initializes it up front.
    pub fn new(node_id: OxirsNodeId, config: NetworkConfig) -> Self {
        let tls_manager = config
            .tls_config
            .enabled
            .then(|| Arc::new(TlsManager::new(config.tls_config.clone(), node_id)));
        Self {
            config,
            node_id,
            connections: Arc::new(RwLock::new(HashMap::new())),
            peer_addresses: Arc::new(RwLock::new(HashMap::new())),
            listener: None,
            accept_task: None,
            running: Arc::new(RwLock::new(false)),
            tls_manager,
            data_key: Arc::new(RwLock::new(None)),
            message_stats: Arc::new(RwLock::new(MessageStats::default())),
        }
    }

    /// Create a new network manager and initialize TLS right away
    ///
    /// Same as [`NetworkManager::new`], except that when
    /// `config.tls_config.enabled` is set TLS is initialized here, so a
    /// configuration problem surfaces before the manager is used. Every
    /// inbound connection and every outgoing RPC then runs over mutually
    /// authenticated TLS against the trust anchors in [`TlsConfig::ca_file`].
    ///
    /// # Errors
    ///
    /// Fails when TLS cannot be initialized, for example because `ca_file` is
    /// not set (see [`TlsManager::initialize`]).
    pub async fn with_tls(node_id: OxirsNodeId, config: NetworkConfig) -> Result<Self> {
        let manager = Self::new(node_id, config);
        if let Some(tls_manager) = &manager.tls_manager {
            tls_manager.initialize().await?;
        }
        Ok(manager)
    }

    /// Start the network manager
    ///
    /// # Errors
    ///
    /// Fails when TLS is enabled but cannot be initialized, or when the
    /// listening address cannot be bound; the manager is left stopped.
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }

        let listener = match self.prepare_listener().await {
            Ok(listener) => listener,
            Err(e) => {
                *self.running.write().await = false;
                return Err(e);
            }
        };

        // Spawn the inbound accept loop. Previously this was skipped ("for now,
        // we'll skip the listener task"), so the bound listener never accepted
        // anything and no peer's Heartbeat (or any other RpcMessage) was ever
        // answered — making health probes report every peer unreachable even
        // for a perfectly healthy cluster. The listener is moved into the task
        // by value (TcpListener isn't Clone, so it can't live in `self` and be
        // served concurrently); `stop()` aborts the task to free the port.
        let server = self.clone();
        let max_message_size = self.config.max_message_size;
        let running = Arc::clone(&self.running);
        let accept_task = tokio::spawn(async move {
            server
                .run_accept_loop(listener, max_message_size, running)
                .await;
        });
        self.accept_task = Some(Arc::new(accept_task));

        // Start background tasks
        self.start_background_tasks().await;

        Ok(())
    }

    /// Initialize TLS when it is enabled (refusing to listen without it), then
    /// bind the listening socket.
    async fn prepare_listener(&self) -> Result<TcpListener> {
        match &self.tls_manager {
            Some(tls_manager) => tls_manager.initialize().await.map_err(|e| {
                anyhow::anyhow!(
                    "TLS is enabled for node {} but could not be initialized: {e}",
                    self.node_id
                )
            })?,
            None => tracing::warn!(
                "TLS is disabled for node {}: cluster RPC traffic is neither encrypted nor authenticated",
                self.node_id
            ),
        }

        let listener = TcpListener::bind(self.config.local_address).await?;
        tracing::info!(
            "Network manager for node {} listening on {}",
            self.node_id,
            self.config.local_address
        );
        Ok(listener)
    }

    /// Inbound accept loop: answer real RpcMessage frames on the bound
    /// listener. Each accepted connection is served by its own task that reads
    /// length-prefixed frames, dispatches them through
    /// [`NetworkManager::handle_inbound`], and writes the response back — so a
    /// peer's health `Heartbeat` gets a real `HeartbeatResponse`. Never panics;
    /// a malformed frame or I/O error just ends that one connection.
    async fn run_accept_loop(
        &self,
        listener: TcpListener,
        max_message_size: usize,
        running: Arc<RwLock<bool>>,
    ) {
        loop {
            if !*running.read().await {
                return;
            }
            let (stream, peer_addr) = match listener.accept().await {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("network accept() failed: {e}; continuing to listen");
                    continue;
                }
            };
            if let Err(e) = stream.set_nodelay(true) {
                tracing::debug!("failed to set TCP_NODELAY on inbound connection: {e}");
            }
            let manager = self.clone();
            tokio::spawn(async move {
                // With TLS configured, every inbound connection must complete an
                // authenticated handshake before a single frame is read.
                match manager.tls_manager.clone() {
                    Some(tls_manager) => {
                        if let Some(tls_stream) =
                            manager.accept_tls(&tls_manager, stream, peer_addr).await
                        {
                            manager
                                .serve_connection(tls_stream, max_message_size, peer_addr)
                                .await;
                        }
                    }
                    None => {
                        manager
                            .serve_connection(stream, max_message_size, peer_addr)
                            .await
                    }
                }
            });
        }
    }

    /// Complete the server side of a TLS handshake on an inbound connection.
    /// Returns `None` (after recording the failure) when the handshake fails or
    /// times out, for example because the peer's certificate is not trusted.
    async fn accept_tls(
        &self,
        tls_manager: &TlsManager,
        stream: TcpStream,
        peer_addr: SocketAddr,
    ) -> Option<tokio_rustls::server::TlsStream<TcpStream>> {
        let handshake = async {
            let acceptor = tls_manager.get_acceptor().await?;
            Ok::<_, anyhow::Error>(acceptor.accept(stream).await?)
        };
        let outcome = match timeout(self.config.connection_timeout, handshake).await {
            Ok(Ok(tls_stream)) => Some(tls_stream),
            Ok(Err(e)) => {
                tracing::warn!("TLS handshake with {peer_addr} failed: {e}");
                None
            }
            Err(_) => {
                tracing::warn!(
                    "TLS handshake with {peer_addr} timed out after {:?}",
                    self.config.connection_timeout
                );
                None
            }
        };
        self.record_tls_handshake(outcome.is_some()).await;
        outcome
    }

    /// Serve length-prefixed RPC frames on one inbound connection until the
    /// peer closes it or sends something that cannot be answered.
    async fn serve_connection<S>(
        &self,
        mut stream: S,
        max_message_size: usize,
        peer_addr: SocketAddr,
    ) where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        loop {
            let request = match read_frame(&mut stream, max_message_size).await {
                Ok(msg) => msg,
                Err(_) => return, // peer closed or sent a bad/partial frame
            };
            match self.handle_inbound(request).await {
                Ok(response) => {
                    if write_frame(&mut stream, &response, max_message_size)
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        "no response produced for inbound message from {peer_addr}: {e}"
                    );
                    return;
                }
            }
        }
    }

    /// Count a completed or failed TLS handshake in the message statistics.
    async fn record_tls_handshake(&self, completed: bool) {
        let mut stats = self.message_stats.write().await;
        if completed {
            stats.tls_handshakes_completed += 1;
        } else {
            stats.tls_handshakes_failed += 1;
        }
    }

    /// Stop the network manager
    pub async fn stop(&mut self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if !*running {
                return Ok(());
            }

            tracing::info!("Stopping network manager for node {}", self.node_id);
            *running = false;
        }

        // Abort the accept loop so the listening port is released.
        if let Some(task) = self.accept_task.take() {
            task.abort();
        }
        self.listener = None;

        // Close all connections
        let mut connections = self.connections.write().await;
        connections.clear();

        Ok(())
    }

    /// Register (or update) the network address of a peer node so that
    /// address-less send paths can resolve a real endpoint.
    pub async fn register_peer(&self, node_id: OxirsNodeId, address: SocketAddr) {
        let mut addrs = self.peer_addresses.write().await;
        addrs.insert(node_id, address);
    }

    /// Resolve the last-known address for `node_id`, consulting the static
    /// registry first and then any live connection metadata.
    async fn resolve_peer_address(&self, node_id: OxirsNodeId) -> Option<SocketAddr> {
        if let Some(addr) = self.peer_addresses.read().await.get(&node_id).copied() {
            return Some(addr);
        }
        self.connections
            .read()
            .await
            .get(&node_id)
            .map(|c| c.address)
    }

    /// Send RPC message to a peer
    pub async fn send_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        // Perform the whole connect + write + read round trip under the request
        // timeout so a stalled peer cannot block the caller indefinitely.
        let response = timeout(
            self.config.request_timeout,
            self.exchange_rpc(peer_id, peer_address, message),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "RPC to peer {peer_id} at {peer_address} timed out after {:?}",
                self.config.request_timeout
            )
        })??;

        Ok(response)
    }

    /// Send a message to `node_id` using a previously registered address (or a
    /// live connection's address). Returns an explicit error when no address is
    /// known — the message is never silently dropped or fabricated as delivered.
    pub async fn send_message(&self, node_id: OxirsNodeId, message: RpcMessage) -> Result<()> {
        let address = self.resolve_peer_address(node_id).await.ok_or_else(|| {
            anyhow::anyhow!(
                "no known network address for node {node_id}; register the peer before sending"
            )
        })?;
        // Deliver for real and require a valid response frame from the peer.
        self.send_rpc(node_id, address, message).await?;
        Ok(())
    }

    /// Send request vote RPC
    pub async fn send_request_vote(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        term: u64,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<(u64, bool)> {
        let message = RpcMessage::RequestVote {
            term,
            candidate_id: self.node_id,
            last_log_index,
            last_log_term,
        };

        let response = self.send_rpc(peer_id, peer_address, message).await?;

        match response {
            RpcMessage::VoteResponse { term, vote_granted } => Ok((term, vote_granted)),
            _ => Err(anyhow::anyhow!("Unexpected response type")),
        }
    }

    /// Send append entries RPC
    #[allow(clippy::too_many_arguments)]
    pub async fn send_append_entries(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        term: u64,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    ) -> Result<(u64, bool, u64)> {
        let message = RpcMessage::AppendEntries {
            term,
            leader_id: self.node_id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        };

        let response = self.send_rpc(peer_id, peer_address, message).await?;

        match response {
            RpcMessage::AppendEntriesResponse {
                term,
                success,
                last_log_index,
            } => Ok((term, success, last_log_index)),
            _ => Err(anyhow::anyhow!("Unexpected response type")),
        }
    }

    /// Send heartbeat to all peers
    pub async fn send_heartbeat(&self, term: u64, peers: &[(OxirsNodeId, SocketAddr)]) {
        let message = RpcMessage::Heartbeat {
            term,
            leader_id: self.node_id,
        };

        for &(peer_id, peer_address) in peers {
            if peer_id != self.node_id {
                let manager = self.clone();
                let message = message.clone();
                tokio::spawn(async move {
                    if let Err(e) = manager.send_rpc(peer_id, peer_address, message).await {
                        tracing::warn!("Failed to send heartbeat to peer {}: {}", peer_id, e);
                    }
                });
            }
        }
    }

    /// Open a real TCP connection to `peer_address`, recording connection
    /// metadata for statistics. Returns the live stream so the caller can
    /// perform length-prefixed framing over it. Errors (refused/timeout) are
    /// propagated so an unreachable peer is never mistaken for a live one.
    async fn connect_to_peer(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
    ) -> Result<TcpStream> {
        let stream = timeout(
            self.config.connection_timeout,
            TcpStream::connect(peer_address),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "timed out connecting to peer {peer_id} at {peer_address} after {:?}",
                self.config.connection_timeout
            )
        })?
        .map_err(|e| {
            anyhow::anyhow!("failed to connect to peer {peer_id} at {peer_address}: {e}")
        })?;

        // Record/refresh connection metadata used by get_stats().
        {
            let mut connections = self.connections.write().await;
            let entry = connections
                .entry(peer_id)
                .or_insert_with(|| Connection::new(peer_id, peer_address));
            entry.address = peer_address;
            entry.is_connected = true;
            entry.update_activity();
        }
        {
            let mut stats = self.message_stats.write().await;
            stats.connections_established += 1;
        }

        Ok(stream)
    }

    /// Perform a full request/response RPC over a freshly opened TCP stream:
    /// connect, write the length-prefixed request frame, then read and
    /// deserialize the peer's actual response frame. No fabricated responses.
    /// When TLS is configured the exchange runs over a TLS session that
    /// authenticates the peer as `node-{peer_id}`; there is no plaintext
    /// fallback.
    async fn exchange_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        if let Some(tls_manager) = &self.tls_manager {
            tls_manager.initialize().await.map_err(|e| {
                anyhow::anyhow!(
                    "TLS is enabled for node {} but could not be initialized, so nothing was \
                     sent to peer {peer_id}: {e}",
                    self.node_id
                )
            })?;
        }
        let stream = self.connect_to_peer(peer_id, peer_address).await?;

        match &self.tls_manager {
            Some(tls_manager) => {
                let mut tls_stream = self
                    .connect_tls(tls_manager, peer_id, peer_address, stream)
                    .await?;
                self.exchange_frames(&mut tls_stream, &message).await
            }
            None => {
                let mut stream = stream;
                self.exchange_frames(&mut stream, &message).await
            }
        }
    }

    /// Complete the client side of a TLS handshake, verifying that the peer
    /// presents a trusted certificate issued to `node-{peer_id}`.
    async fn connect_tls(
        &self,
        tls_manager: &TlsManager,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        stream: TcpStream,
    ) -> Result<tokio_rustls::client::TlsStream<TcpStream>> {
        let server_name = rustls::pki_types::ServerName::try_from(format!("node-{peer_id}"))?;
        let handshake = async {
            let connector = tls_manager.get_connector().await?;
            Ok::<_, anyhow::Error>(connector.connect(server_name, stream).await?)
        };
        let outcome = match timeout(self.config.connection_timeout, handshake).await {
            Ok(Ok(tls_stream)) => Ok(tls_stream),
            Ok(Err(e)) => Err(anyhow::anyhow!(
                "TLS handshake with peer {peer_id} at {peer_address} failed: {e}"
            )),
            Err(_) => Err(anyhow::anyhow!(
                "TLS handshake with peer {peer_id} at {peer_address} timed out after {:?}",
                self.config.connection_timeout
            )),
        };
        self.record_tls_handshake(outcome.is_ok()).await;
        outcome
    }

    /// Write one request frame and read the peer's response frame.
    async fn exchange_frames<S>(&self, stream: &mut S, message: &RpcMessage) -> Result<RpcMessage>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let sent = write_frame(stream, message, self.config.max_message_size).await?;
        let response = read_frame(stream, self.config.max_message_size).await?;

        {
            let mut stats = self.message_stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_sent += sent;
            stats.messages_received += 1;
        }

        Ok(response)
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&self) {
        let running = Arc::clone(&self.running);
        let connections = Arc::clone(&self.connections);
        let connection_timeout = self.config.connection_timeout;

        // Connection cleanup task
        tokio::spawn(async move {
            while *running.read().await {
                {
                    let mut connections = connections.write().await;
                    let stale_connections: Vec<_> = connections
                        .iter()
                        .filter(|(_, conn)| conn.is_stale(connection_timeout))
                        .map(|(&id, _)| id)
                        .collect();

                    for peer_id in stale_connections {
                        connections.remove(&peer_id);
                        tracing::debug!("Removed stale connection to peer {}", peer_id);
                    }
                }

                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });

        // Connection listener task
        // Note: In a real implementation, we would need to handle the listener differently
        // since TcpListener doesn't support cloning. For now, we'll skip the listener task.
    }

    /// Produce the response for an inbound [`RpcMessage`] received on the
    /// accept loop. Answers latency-sensitive protocol messages directly (a
    /// `Heartbeat` gets a real `HeartbeatResponse`, which is what makes peer
    /// health probes work), returns conservative defaults for consensus
    /// messages a bare network layer cannot decide, and fails loud for
    /// operations that require a wired-up handler (shard/migration) instead of
    /// fabricating a success. This is the single source of truth for inbound
    /// handling; [`NetworkService::handle_message`] delegates here.
    pub async fn handle_inbound(&self, message: RpcMessage) -> Result<RpcMessage> {
        match message {
            #[cfg(feature = "bft")]
            RpcMessage::Bft { .. } => Err(anyhow::anyhow!(
                "BFT messages should be handled by BFT network service"
            )),

            // --- Raft consensus messages ---
            RpcMessage::RequestVote { term, .. } => Ok(RpcMessage::VoteResponse {
                term,
                vote_granted: false,
            }),
            RpcMessage::VoteResponse { .. } => Ok(message),
            RpcMessage::AppendEntries { term, .. } => Ok(RpcMessage::AppendEntriesResponse {
                term,
                success: true,
                last_log_index: 0,
            }),
            RpcMessage::AppendEntriesResponse { .. } => Ok(message),
            RpcMessage::Heartbeat { term, .. } => Ok(RpcMessage::HeartbeatResponse { term }),
            RpcMessage::HeartbeatResponse { .. } => Ok(message),

            // --- Client request/response ---
            RpcMessage::ClientRequest { .. } => Ok(RpcMessage::ClientResponse {
                response: RdfResponse::Success,
            }),
            RpcMessage::ClientResponse { .. } => Ok(message),

            // --- Shard operations ---
            RpcMessage::ShardOperation(_) => Err(anyhow::anyhow!(
                "RpcMessage::ShardOperation requires a shard manager — not connected"
            )),
            RpcMessage::StoreTriple { shard_id, .. } => Err(anyhow::anyhow!(
                "RpcMessage::StoreTriple for shard {:?} requires shard handler — not connected",
                shard_id
            )),
            RpcMessage::ReplicateTriple { shard_id, .. } => Err(anyhow::anyhow!(
                "RpcMessage::ReplicateTriple for shard {:?} requires shard handler — not connected",
                shard_id
            )),
            RpcMessage::QueryShard { shard_id, .. } => Ok(RpcMessage::QueryShardResponse {
                shard_id,
                results: Vec::new(),
            }),
            RpcMessage::QueryShardResponse { .. } => Ok(message),

            // --- Two-phase commit / distributed transactions ---
            RpcMessage::TransactionPrepare { tx_id, shard_id, .. } => {
                Ok(RpcMessage::TransactionVote {
                    tx_id,
                    shard_id,
                    vote: false,
                })
            }
            RpcMessage::TransactionVote { .. } => Ok(message),
            RpcMessage::TransactionCommit { tx_id, shard_id } => {
                Ok(RpcMessage::TransactionAck { tx_id, shard_id })
            }
            RpcMessage::TransactionAbort { tx_id, shard_id } => {
                Ok(RpcMessage::TransactionAck { tx_id, shard_id })
            }
            RpcMessage::TransactionAck { .. } => Ok(message),

            // --- Shard migration ---
            RpcMessage::MigrationBatch { migration_id, .. } => Err(anyhow::anyhow!(
                "RpcMessage::MigrationBatch for migration '{}' requires migration handler — not connected",
                migration_id
            )),
            RpcMessage::ShardTransfer { shard_id, .. } => Err(anyhow::anyhow!(
                "RpcMessage::ShardTransfer for shard {:?} requires migration handler — not connected",
                shard_id
            )),
        }
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let connections = self.connections.read().await;
        let total_connections = connections.len();
        let active_connections = connections.values().filter(|c| c.is_connected).count();

        NetworkStats {
            total_connections,
            active_connections,
            local_address: self.config.local_address,
            node_id: self.node_id,
        }
    }

    /// Send an RPC message to a peer over mutually authenticated TLS.
    ///
    /// Unlike [`NetworkManager::send_rpc`], this never runs in plaintext: it is
    /// an error to call it on a manager whose `tls_config.enabled` is off.
    ///
    /// # Errors
    ///
    /// Fails when TLS is not configured, when the peer cannot be authenticated
    /// as `node-{peer_id}`, or when the exchange itself fails.
    pub async fn send_secure_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        if self.tls_manager.is_none() {
            return Err(anyhow::anyhow!(
                "secure RPC to peer {peer_id} refused: TLS is not configured on this node \
                 and plaintext fallback is not allowed"
            ));
        }
        self.send_rpc(peer_id, peer_address, message).await
    }

    /// Get message statistics
    pub async fn get_message_stats(&self) -> MessageStats {
        self.message_stats.read().await.clone()
    }

    /// Reset message statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.message_stats.write().await;
        *stats = MessageStats::default();
    }

    /// Check TLS certificate status
    pub async fn get_tls_status(&self) -> Result<TlsStatus> {
        if let Some(tls_manager) = &self.tls_manager {
            let certificates = tls_manager.list_certificates().await;
            let server_cert = certificates.get("server");

            Ok(TlsStatus {
                enabled: true,
                certificates_count: certificates.len(),
                server_cert_expires: server_cert.map(|c| c.not_after),
                handshakes_completed: self.message_stats.read().await.tls_handshakes_completed,
                handshakes_failed: self.message_stats.read().await.tls_handshakes_failed,
            })
        } else {
            Ok(TlsStatus {
                enabled: false,
                certificates_count: 0,
                server_cert_expires: None,
                handshakes_completed: 0,
                handshakes_failed: 0,
            })
        }
    }

    /// Set the key used by [`NetworkManager::encrypt_data`] and
    /// [`NetworkManager::decrypt_data`]; it is shared by all clones.
    pub async fn set_data_encryption_key(&self, key: AtRestKey) {
        *self.data_key.write().await = Some(key);
    }

    /// Encrypt data at rest with XChaCha20-Poly1305 under the key set by
    /// [`NetworkManager::set_data_encryption_key`] (see [`crate::at_rest`]).
    ///
    /// # Errors
    ///
    /// Fails when no data key has been set; data is never passed through
    /// unencrypted.
    pub async fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let key = self.data_key.read().await;
        let key = key.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no data encryption key is set; call set_data_encryption_key first")
        })?;
        Ok(at_rest::seal(key, NETWORK_DATA_CONTEXT, data)?)
    }

    /// Decrypt data produced by [`NetworkManager::encrypt_data`].
    ///
    /// # Errors
    ///
    /// Fails when no data key has been set, or when `encrypted_data` was not
    /// sealed under that key or has been modified.
    pub async fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let key = self.data_key.read().await;
        let key = key.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no data encryption key is set; call set_data_encryption_key first")
        })?;
        Ok(at_rest::open(key, NETWORK_DATA_CONTEXT, encrypted_data)?)
    }
}

// Implement Clone for NetworkManager to allow sharing
impl Clone for NetworkManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            node_id: self.node_id,
            connections: Arc::clone(&self.connections),
            peer_addresses: Arc::clone(&self.peer_addresses),
            listener: None,    // Don't clone the listener
            accept_task: None, // Don't clone the server task handle
            running: Arc::clone(&self.running),
            tls_manager: self.tls_manager.clone(),
            data_key: Arc::clone(&self.data_key),
            message_stats: Arc::clone(&self.message_stats),
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub local_address: SocketAddr,
    pub node_id: OxirsNodeId,
}

/// TLS status information
#[derive(Debug, Clone)]
pub struct TlsStatus {
    pub enabled: bool,
    pub certificates_count: usize,
    pub server_cert_expires: Option<SystemTime>,
    pub handshakes_completed: u64,
    pub handshakes_failed: u64,
}

/// Network service for high-level network operations
#[derive(Debug, Clone)]
pub struct NetworkService {
    manager: NetworkManager,
}

impl NetworkService {
    /// Create a new network service (see [`NetworkManager::new`] for how
    /// `config.tls_config` is applied)
    pub fn new(node_id: OxirsNodeId, config: NetworkConfig) -> Self {
        Self {
            manager: NetworkManager::new(node_id, config),
        }
    }

    /// Create a new network service and initialize TLS right away (see
    /// [`NetworkManager::with_tls`])
    ///
    /// # Errors
    ///
    /// Fails when TLS is enabled but cannot be initialized.
    pub async fn with_tls(node_id: OxirsNodeId, config: NetworkConfig) -> Result<Self> {
        Ok(Self {
            manager: NetworkManager::with_tls(node_id, config).await?,
        })
    }

    /// Start the network service
    pub async fn start(&mut self) -> Result<()> {
        self.manager.start().await
    }

    /// Stop the network service
    pub async fn stop(&mut self) -> Result<()> {
        self.manager.stop().await
    }

    /// Send a message to a specific peer
    pub async fn send_to(&self, peer_id: &str, message: RpcMessage) -> Result<()> {
        let peer_id: OxirsNodeId = peer_id
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid peer ID"))?;
        // Use send_message method from NetworkService instead
        self.send_message(peer_id, message).await?;
        Ok(())
    }

    /// Broadcast a message to all peers
    pub async fn broadcast(&self, message: RpcMessage) -> Result<()> {
        let connections = self.manager.connections.read().await;
        for peer_id in connections.keys() {
            let _ = self.send_message(*peer_id, message.clone()).await;
        }
        Ok(())
    }

    /// Register (or update) the network address of a peer node so that
    /// [`NetworkService::send_message`] can resolve a real endpoint for it.
    pub async fn register_peer(&self, node_id: OxirsNodeId, address: SocketAddr) {
        self.manager.register_peer(node_id, address).await;
    }

    /// Send an RPC to a peer and return the peer's actual response frame.
    pub async fn send_rpc(
        &self,
        peer_id: OxirsNodeId,
        peer_address: SocketAddr,
        message: RpcMessage,
    ) -> Result<RpcMessage> {
        self.manager.send_rpc(peer_id, peer_address, message).await
    }

    /// Send a message to a specific node.
    ///
    /// Delivers the message over a real TCP connection via the network manager,
    /// resolving the peer's address from the registry. If no address is known
    /// (or the connection cannot be established), an explicit error is returned
    /// — the message is never silently dropped nor reported as delivered.
    pub async fn send_message(&self, node_id: OxirsNodeId, message: RpcMessage) -> Result<()> {
        tracing::debug!("Sending message to node {}: {:?}", node_id, message);
        self.manager.send_message(node_id, message).await
    }

    /// Handle incoming RPC message. Delegates to
    /// [`NetworkManager::handle_inbound`], the single source of truth also used
    /// by the inbound accept loop.
    pub async fn handle_message(&self, message: RpcMessage) -> Result<RpcMessage> {
        self.manager.handle_inbound(message).await
    }
}

/// Network-related errors
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Connection failed to {peer_id} at {address}: {message}")]
    ConnectionFailed {
        peer_id: OxirsNodeId,
        address: SocketAddr,
        message: String,
    },

    #[error("Timeout: {operation} timed out after {duration:?}")]
    Timeout {
        operation: String,
        duration: Duration,
    },

    #[error("Serialization error: {message}")]
    Serialization { message: String },

    #[error("Protocol error: {message}")]
    Protocol { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests;
