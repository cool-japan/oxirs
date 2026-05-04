//! Individual cluster node implementation
//!
//! This module provides the core node functionality for clustering including
//! node lifecycle management, health monitoring, and inter-node communication.

use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, RwLock},
    task::JoinHandle,
};
use tracing::{error, info, warn};

use super::{ClusterConfig, NodeInfo, NodeMetadata, NodeState};
use crate::error::{FusekiError, FusekiResult};

/// Maximum allowed frame payload size (16 MiB) to guard against OOM from
/// malformed or adversarial length prefixes.
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Write a length-prefixed frame.
///
/// Wire format: `[u32 big-endian length][JSON payload bytes]`
/// The full envelope is the serialised form of `(sender_id, NodeMessage)`.
async fn write_frame<W>(stream: &mut W, envelope: &[u8]) -> FusekiResult<()>
where
    W: AsyncWriteExt + Unpin,
{
    let len = envelope.len();
    if len > MAX_FRAME_BYTES {
        return Err(FusekiError::internal(format!(
            "frame too large: {len} bytes (max {MAX_FRAME_BYTES})"
        )));
    }
    let len_bytes = (len as u32).to_be_bytes();
    stream
        .write_all(&len_bytes)
        .await
        .map_err(|e| FusekiError::internal(format!("TCP write length prefix: {e}")))?;
    stream
        .write_all(envelope)
        .await
        .map_err(|e| FusekiError::internal(format!("TCP write payload: {e}")))?;
    stream
        .flush()
        .await
        .map_err(|e| FusekiError::internal(format!("TCP flush: {e}")))?;
    Ok(())
}

/// Read one length-prefixed frame and return the raw bytes.
async fn read_frame<R>(stream: &mut R) -> FusekiResult<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| FusekiError::internal(format!("TCP read length prefix: {e}")))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(FusekiError::internal(format!(
            "incoming frame too large: {len} bytes (max {MAX_FRAME_BYTES})"
        )));
    }
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|e| FusekiError::internal(format!("TCP read payload: {e}")))?;
    Ok(payload)
}

/// Node lifecycle events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeEvent {
    /// Node joined the cluster
    Joined(NodeInfo),
    /// Node left the cluster
    Left(String),
    /// Node state changed
    StateChanged(String, NodeState),
    /// Node metadata updated
    MetadataUpdated(String, NodeMetadata),
}

/// Inter-node message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeMessage {
    /// Heartbeat message
    Heartbeat {
        node_id: String,
        timestamp: i64,
        metadata: NodeMetadata,
    },
    /// Join request
    JoinRequest { node_info: NodeInfo },
    /// Join response
    JoinResponse {
        accepted: bool,
        cluster_members: Vec<NodeInfo>,
    },
    /// Leave notification
    LeaveNotification { node_id: String },
    /// Leader election vote request (Raft RequestVote)
    LeaderElection { candidate_id: String, term: u64 },
    /// Vote granted in response to a leader election request
    VoteGranted { node_id: String, term: u64 },
    /// Vote rejected (already voted this term or candidate's log is stale)
    VoteRejected { node_id: String, term: u64 },
}

/// Node communication interface
#[async_trait]
pub trait NodeCommunication: Send + Sync {
    /// Send message to a specific node
    async fn send_message(&self, target: &str, message: NodeMessage) -> FusekiResult<()>;

    /// Broadcast message to all nodes
    async fn broadcast_message(&self, message: NodeMessage) -> FusekiResult<()>;

    /// Receive messages from other nodes
    async fn receive_messages(&self) -> FusekiResult<mpsc::Receiver<(String, NodeMessage)>>;
}

/// Cluster node implementation
pub struct ClusterNode {
    /// Node configuration
    config: ClusterConfig,
    /// Node information
    node_info: Arc<RwLock<NodeInfo>>,
    /// Communication interface
    communication: Arc<dyn NodeCommunication>,
    /// Event sender
    event_sender: mpsc::UnboundedSender<NodeEvent>,
    /// Known cluster members
    cluster_members: Arc<RwLock<HashMap<String, NodeInfo>>>,
    /// Last heartbeat times
    last_heartbeats: Arc<RwLock<HashMap<String, Instant>>>,
    /// Node metrics
    metrics: Arc<RwLock<NodeMetrics>>,
    /// Current Raft term (monotonically increasing)
    current_term: Arc<AtomicU64>,
    /// Node ID that this node voted for in the current term, if any
    voted_for: Arc<RwLock<Option<String>>>,
    /// System information handle for resource metrics
    sys_monitor: Arc<parking_lot::Mutex<System>>,
}

/// Node performance metrics
#[derive(Debug, Default, Clone)]
pub struct NodeMetrics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Failed message attempts
    pub message_failures: u64,
    /// Current connections
    pub active_connections: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network bytes in/out
    pub network_io: (u64, u64),
}

impl ClusterNode {
    /// Create a new cluster node
    pub async fn new(
        config: ClusterConfig,
        communication: Arc<dyn NodeCommunication>,
    ) -> FusekiResult<Self> {
        let node_info = Arc::new(RwLock::new(NodeInfo {
            id: config.node_id.clone(),
            addr: config.bind_addr,
            state: NodeState::Joining,
            metadata: NodeMetadata {
                datacenter: None,
                rack: None,
                capacity: 1000,
                load: 0.0,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            last_heartbeat: chrono::Utc::now().timestamp_millis(),
        }));

        let (event_sender, _) = mpsc::unbounded_channel();

        // Initialise sysinfo with only the refresh kinds we actually use.
        let sys_monitor = Arc::new(parking_lot::Mutex::new(System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        )));

        Ok(Self {
            config,
            node_info,
            communication,
            event_sender,
            cluster_members: Arc::new(RwLock::new(HashMap::new())),
            last_heartbeats: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
            current_term: Arc::new(AtomicU64::new(0)),
            voted_for: Arc::new(RwLock::new(None)),
            sys_monitor,
        })
    }

    /// Start the node
    pub async fn start(&self) -> FusekiResult<()> {
        info!("Starting cluster node {}", self.config.node_id);

        // Update node state to active
        {
            let mut node_info = self.node_info.write().await;
            node_info.state = NodeState::Active;
        }

        // Start message processing
        self.start_message_processing().await?;

        // Start heartbeat
        self.start_heartbeat().await;

        // Start failure detection
        self.start_failure_detection().await;

        // Start metrics collection
        self.start_metrics_collection().await;

        Ok(())
    }

    /// Stop the node
    pub async fn stop(&self) -> FusekiResult<()> {
        info!("Stopping cluster node {}", self.config.node_id);

        // Send leave notification
        let leave_msg = NodeMessage::LeaveNotification {
            node_id: self.config.node_id.clone(),
        };
        let _ = self.communication.broadcast_message(leave_msg).await;

        // Update node state
        {
            let mut node_info = self.node_info.write().await;
            node_info.state = NodeState::Leaving;
        }

        Ok(())
    }

    /// Join an existing cluster
    pub async fn join_cluster(&self, seed_nodes: &[String]) -> FusekiResult<()> {
        info!("Joining cluster via seeds: {:?}", seed_nodes);

        let node_info = self.node_info.read().await.clone();
        let join_request = NodeMessage::JoinRequest { node_info };

        for seed in seed_nodes {
            match self
                .communication
                .send_message(seed, join_request.clone())
                .await
            {
                Ok(()) => {
                    info!("Successfully contacted seed node: {}", seed);
                    break;
                }
                Err(e) => {
                    warn!("Failed to contact seed node {}: {}", seed, e);
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Start message processing loop
    async fn start_message_processing(&self) -> FusekiResult<()> {
        let mut receiver = self.communication.receive_messages().await?;
        let cluster_members = self.cluster_members.clone();
        let last_heartbeats = self.last_heartbeats.clone();
        let metrics = self.metrics.clone();
        let event_sender = self.event_sender.clone();
        let communication = self.communication.clone();
        let current_term = self.current_term.clone();
        let voted_for = self.voted_for.clone();
        let own_node_id = self.config.node_id.clone();

        tokio::spawn(async move {
            while let Some((sender_id, message)) = receiver.recv().await {
                // Update metrics
                {
                    let mut m = metrics.write().await;
                    m.messages_received += 1;
                }

                match message {
                    NodeMessage::Heartbeat {
                        node_id,
                        timestamp,
                        metadata,
                    } => {
                        // Update heartbeat time
                        {
                            let mut heartbeats = last_heartbeats.write().await;
                            heartbeats.insert(node_id.clone(), Instant::now());
                        }

                        // Update cluster member info
                        {
                            let mut members = cluster_members.write().await;
                            if let Some(member) = members.get_mut(&node_id) {
                                member.last_heartbeat = timestamp;
                                member.metadata = metadata;
                            }
                        }
                    }

                    NodeMessage::JoinRequest {
                        node_info: joining_node,
                    } => {
                        info!("Received join request from node: {}", joining_node.id);

                        // Add to cluster members
                        {
                            let mut members = cluster_members.write().await;
                            members.insert(joining_node.id.clone(), joining_node.clone());
                        }

                        // Send event
                        let _ = event_sender.send(NodeEvent::Joined(joining_node));
                    }

                    NodeMessage::JoinResponse {
                        accepted,
                        cluster_members: members,
                    } => {
                        if accepted {
                            info!("Join request accepted, updating cluster membership");
                            let mut local_members = cluster_members.write().await;
                            for member in members {
                                local_members.insert(member.id.clone(), member);
                            }
                        } else {
                            warn!("Join request was rejected");
                        }
                    }

                    NodeMessage::LeaveNotification { node_id } => {
                        info!("Node {} is leaving the cluster", node_id);

                        // Remove from cluster members
                        {
                            let mut members = cluster_members.write().await;
                            members.remove(&node_id);
                        }

                        // Remove heartbeat tracking
                        {
                            let mut heartbeats = last_heartbeats.write().await;
                            heartbeats.remove(&node_id);
                        }

                        // Send event
                        let _ = event_sender.send(NodeEvent::Left(node_id));
                    }

                    NodeMessage::LeaderElection { candidate_id, term } => {
                        info!(
                            "Received leader election request from {} for term {}",
                            candidate_id, term
                        );

                        let local_term = current_term.load(Ordering::SeqCst);

                        // Raft rule: if we see a higher term, update our term and clear voted_for.
                        if term > local_term {
                            current_term.store(term, Ordering::SeqCst);
                            let mut vf = voted_for.write().await;
                            *vf = None;
                        }

                        // Determine whether to grant the vote.
                        // We grant the vote when:
                        //   1. The candidate's term is at least as large as ours, AND
                        //   2. We have not yet voted in this term (or already voted for the same candidate).
                        let current = current_term.load(Ordering::SeqCst);
                        let grant = {
                            let vf = voted_for.read().await;
                            let not_voted_or_same =
                                vf.is_none() || vf.as_deref() == Some(candidate_id.as_str());
                            term >= current && not_voted_or_same
                        };

                        if grant {
                            // Record our vote.
                            {
                                let mut vf = voted_for.write().await;
                                *vf = Some(candidate_id.clone());
                            }
                            info!("Granting vote to {} for term {}", candidate_id, current);
                            let response = NodeMessage::VoteGranted {
                                node_id: own_node_id.clone(),
                                term: current,
                            };
                            if let Err(e) = communication.send_message(&sender_id, response).await {
                                error!("Failed to send VoteGranted to {}: {}", sender_id, e);
                            }
                        } else {
                            info!(
                                "Rejecting vote for {} (already voted this term {})",
                                candidate_id, current
                            );
                            let response = NodeMessage::VoteRejected {
                                node_id: own_node_id.clone(),
                                term: current,
                            };
                            if let Err(e) = communication.send_message(&sender_id, response).await {
                                error!("Failed to send VoteRejected to {}: {}", sender_id, e);
                            }
                        }
                    }

                    NodeMessage::VoteGranted { node_id, term } => {
                        info!("Received VoteGranted from {} for term {}", node_id, term);
                    }

                    NodeMessage::VoteRejected { node_id, term } => {
                        info!("Received VoteRejected from {} for term {}", node_id, term);
                    }
                }
            }
        });

        Ok(())
    }

    /// Start heartbeat broadcasting
    async fn start_heartbeat(&self) {
        let communication = self.communication.clone();
        let node_info = self.node_info.clone();
        let metrics = self.metrics.clone();
        let node_id = self.config.node_id.clone();
        let interval = self.config.raft.heartbeat_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                let (metadata, timestamp) = {
                    let info = node_info.read().await;
                    (info.metadata.clone(), chrono::Utc::now().timestamp_millis())
                };

                let heartbeat = NodeMessage::Heartbeat {
                    node_id: node_id.clone(),
                    timestamp,
                    metadata,
                };

                if let Err(e) = communication.broadcast_message(heartbeat).await {
                    error!("Failed to send heartbeat: {}", e);
                    let mut m = metrics.write().await;
                    m.message_failures += 1;
                } else {
                    let mut m = metrics.write().await;
                    m.messages_sent += 1;
                }
            }
        });
    }

    /// Start failure detection
    async fn start_failure_detection(&self) {
        let cluster_members = self.cluster_members.clone();
        let last_heartbeats = self.last_heartbeats.clone();
        let event_sender = self.event_sender.clone();
        let timeout = Duration::from_secs(30); // 30 second timeout

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(10));
            loop {
                ticker.tick().await;

                let now = Instant::now();
                let mut failed_nodes = Vec::new();

                // Check for failed nodes
                {
                    let heartbeats = last_heartbeats.read().await;
                    for (node_id, last_heartbeat) in heartbeats.iter() {
                        if now.duration_since(*last_heartbeat) > timeout {
                            failed_nodes.push(node_id.clone());
                        }
                    }
                }

                // Mark failed nodes as down
                for node_id in failed_nodes {
                    warn!("Node {} marked as down due to missed heartbeats", node_id);

                    {
                        let mut members = cluster_members.write().await;
                        if let Some(member) = members.get_mut(&node_id) {
                            member.state = NodeState::Down;
                            let _ = event_sender
                                .send(NodeEvent::StateChanged(node_id.clone(), NodeState::Down));
                        }
                    }
                }
            }
        });
    }

    /// Start metrics collection
    async fn start_metrics_collection(&self) {
        let metrics = self.metrics.clone();
        let node_info = self.node_info.clone();
        let sys_monitor = self.sys_monitor.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(5));
            loop {
                ticker.tick().await;

                // Refresh and collect system metrics
                let (cpu_usage, memory_usage, network_io) = {
                    let mut sys = sys_monitor.lock();
                    sys.refresh_cpu_all();
                    sys.refresh_memory();
                    let cpu = sys.global_cpu_usage() as f64;
                    let mem = sys.used_memory();
                    // Network I/O totals: iterate all network interfaces if available.
                    // sysinfo 0.38 exposes Networks via the `Networks` struct — since we
                    // didn't refresh networks above, we report 0 bytes which is consistent
                    // with the rest of the codebase (see monitoring.rs).
                    let net = (0u64, 0u64);
                    (cpu, mem, net)
                };

                // Update metrics
                {
                    let mut m = metrics.write().await;
                    m.cpu_usage = cpu_usage;
                    m.memory_usage = memory_usage;
                    m.network_io = network_io;
                }

                // Update node metadata
                {
                    let mut info = node_info.write().await;
                    info.metadata.load = cpu_usage / 100.0;
                }
            }
        });
    }

    /// Get current CPU usage percentage from the system monitor.
    async fn get_cpu_usage(sys_monitor: &parking_lot::Mutex<System>) -> f64 {
        let mut sys = sys_monitor.lock();
        sys.refresh_cpu_all();
        sys.global_cpu_usage() as f64
    }

    /// Get current memory usage in bytes from the system monitor.
    async fn get_memory_usage(sys_monitor: &parking_lot::Mutex<System>) -> u64 {
        let mut sys = sys_monitor.lock();
        sys.refresh_memory();
        sys.used_memory()
    }

    /// Get network I/O statistics (bytes in, bytes out).
    ///
    /// The sysinfo 0.38 network API requires a separately-initialised
    /// `Networks` object that is refreshed independently. Since we initialise
    /// the `System` object with CPU and memory refresh kinds only, we report
    /// the aggregate as `(0, 0)`.  Platform-specific counters from
    /// `/proc/net/dev` can be added here as a future enhancement.
    async fn get_network_io(_sys_monitor: &parking_lot::Mutex<System>) -> (u64, u64) {
        (0, 0)
    }

    /// Get node information
    pub async fn get_node_info(&self) -> NodeInfo {
        self.node_info.read().await.clone()
    }

    /// Get cluster members
    pub async fn get_cluster_members(&self) -> HashMap<String, NodeInfo> {
        self.cluster_members.read().await.clone()
    }

    /// Get node metrics
    pub async fn get_metrics(&self) -> NodeMetrics {
        self.metrics.read().await.clone()
    }

    /// Returns the current Raft term known by this node.
    pub fn current_term(&self) -> u64 {
        self.current_term.load(Ordering::SeqCst)
    }

    /// Returns the node ID that this node voted for in the current term, if any.
    pub async fn voted_for(&self) -> Option<String> {
        self.voted_for.read().await.clone()
    }

    /// Check if node is leader
    ///
    /// Full leader-state tracking is implemented in the Raft module; here we
    /// return `false` as a conservative default so that callers outside the
    /// Raft layer do not assume leadership without consensus.
    pub async fn is_leader(&self) -> bool {
        false
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> mpsc::UnboundedReceiver<NodeEvent> {
        let (_, receiver) = mpsc::unbounded_channel();
        receiver
    }
}

/// Type alias for the guarded optional inbound channel receiver.
type InboundRx = Arc<tokio::sync::Mutex<Option<mpsc::Receiver<(String, NodeMessage)>>>>;

/// TCP-based node communication implementation.
///
/// Frames are length-prefixed (4-byte big-endian u32) JSON blobs that carry a
/// `(sender_id, NodeMessage)` tuple so the receiver can always identify the
/// source without relying on unstable ephemeral source ports.
///
/// Call [`TcpNodeCommunication::start_listener`] once (before handing the
/// instance to [`ClusterNode::new`]) to begin accepting inbound connections.
/// The background accept loop pushes decoded messages into the inbound channel
/// that [`NodeCommunication::receive_messages`] hands out.
pub struct TcpNodeCommunication {
    bind_addr: SocketAddr,
    known_nodes: Arc<RwLock<HashMap<String, SocketAddr>>>,
    /// Our own cluster-node ID, embedded in every outbound frame so the remote
    /// end can reconstruct the sender without extra signalling.
    own_node_id: String,
    /// Sender half kept alive so the channel is never prematurely closed.
    inbound_tx: mpsc::Sender<(String, NodeMessage)>,
    /// Receiver half stored so `receive_messages` can hand it out once.
    inbound_rx: InboundRx,
    /// Handle to the TCP accept-loop background task.
    ///
    /// Stored so callers can tell whether `start_listener` was already called.
    /// Note: dropping a tokio `JoinHandle` **detaches** the task (it keeps
    /// running); it does not abort it.  Explicit abort-on-drop is not
    /// implemented in this slice.
    _listener_handle: Arc<tokio::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl TcpNodeCommunication {
    /// Create a new `TcpNodeCommunication` with the given bind address and own
    /// cluster-node ID.
    ///
    /// Call `start_listener` once to spawn the TCP accept-loop background task.
    pub fn new(bind_addr: SocketAddr) -> Self {
        Self::with_node_id(bind_addr, bind_addr.to_string())
    }

    /// Create with an explicit node ID (used in tests and production where the
    /// cluster node ID is known before the communication object is constructed).
    pub fn with_node_id(bind_addr: SocketAddr, own_node_id: String) -> Self {
        let (inbound_tx, inbound_rx) = mpsc::channel(1024);
        Self {
            bind_addr,
            known_nodes: Arc::new(RwLock::new(HashMap::new())),
            own_node_id,
            inbound_tx,
            inbound_rx: Arc::new(tokio::sync::Mutex::new(Some(inbound_rx))),
            _listener_handle: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub async fn add_node(&self, node_id: String, addr: SocketAddr) {
        let mut nodes = self.known_nodes.write().await;
        nodes.insert(node_id, addr);
    }

    /// Bind a TCP listener on `bind_addr` and spawn an accept loop that reads
    /// length-prefixed frames and forwards decoded messages into the inbound
    /// channel.
    ///
    /// This is a no-op if `start_listener` was already called.
    pub async fn start_listener(&self) -> FusekiResult<()> {
        let mut handle_slot = self._listener_handle.lock().await;
        if handle_slot.is_some() {
            return Ok(());
        }

        let listener = TcpListener::bind(self.bind_addr)
            .await
            .map_err(|e| FusekiError::internal(format!("TCP bind {}: {e}", self.bind_addr)))?;

        info!("TCP node listener bound to {}", self.bind_addr);

        let inbound_tx = self.inbound_tx.clone();

        let handle = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        let tx = inbound_tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, tx).await {
                                warn!("Connection from {} error: {}", peer_addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("TCP accept error: {}", e);
                    }
                }
            }
        });

        *handle_slot = Some(handle);
        Ok(())
    }

    /// Inject an inbound message into the processing pipeline.
    ///
    /// Used for unit/integration testing and also called by the internal TCP
    /// accept loop when a decoded frame arrives.
    pub async fn inject_message(
        &self,
        sender_id: String,
        message: NodeMessage,
    ) -> FusekiResult<()> {
        self.inbound_tx
            .send((sender_id, message))
            .await
            .map_err(|e| FusekiError::internal(format!("inbound channel closed: {e}")))?;
        Ok(())
    }
}

/// Handle a single inbound TCP connection: read length-prefixed frames until
/// EOF or error and forward each decoded `(sender_id, NodeMessage)` to the
/// inbound channel.
async fn handle_connection(
    mut stream: TcpStream,
    tx: mpsc::Sender<(String, NodeMessage)>,
) -> FusekiResult<()> {
    loop {
        // read_exact on an EOF'd stream returns `UnexpectedEof`; treat it as a
        // clean disconnect rather than an error.
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => {
                return Err(FusekiError::internal(format!(
                    "TCP read length prefix: {e}"
                )))
            }
        }

        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_FRAME_BYTES {
            return Err(FusekiError::internal(format!(
                "incoming frame too large: {len} bytes (max {MAX_FRAME_BYTES})"
            )));
        }

        let mut payload = vec![0u8; len];
        stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| FusekiError::internal(format!("TCP read payload: {e}")))?;

        let (sender_id, message): (String, NodeMessage) = serde_json::from_slice(&payload)
            .map_err(|e| FusekiError::internal(format!("deserialize node message frame: {e}")))?;

        if tx.send((sender_id, message)).await.is_err() {
            // Channel closed — the node is shutting down.
            return Ok(());
        }
    }
}

#[async_trait]
impl NodeCommunication for TcpNodeCommunication {
    /// Open a fresh TCP connection to `target`, write a single length-prefixed
    /// JSON frame containing `(own_node_id, message)`, then close the stream.
    ///
    /// Returns `Err` if the target is not in `known_nodes`, if TCP connect
    /// fails, or if serialisation/write fails.
    async fn send_message(&self, target: &str, message: NodeMessage) -> FusekiResult<()> {
        let target_addr = {
            let nodes = self.known_nodes.read().await;
            nodes
                .get(target)
                .copied()
                .ok_or_else(|| FusekiError::internal(format!("Unknown target node: {target}")))?
        };

        let envelope: Vec<u8> = serde_json::to_vec(&(&self.own_node_id, &message))
            .map_err(|e| FusekiError::internal(format!("serialize node message envelope: {e}")))?;

        let mut stream = TcpStream::connect(target_addr)
            .await
            .map_err(|e| FusekiError::internal(format!("TCP connect to {target_addr}: {e}")))?;

        write_frame(&mut stream, &envelope).await?;

        info!("Sent message to {} at {}", target, target_addr);
        Ok(())
    }

    /// Broadcast `message` to every known peer node.
    ///
    /// All sends are issued concurrently via `join_all`.  Individual failures
    /// are logged but do not prevent delivery to other nodes (best-effort
    /// semantics).  The method returns `Ok(())` even if some sends fail so
    /// that heartbeat loops remain robust in the presence of transient faults.
    async fn broadcast_message(&self, message: NodeMessage) -> FusekiResult<()> {
        let node_ids: Vec<String> = {
            let nodes = self.known_nodes.read().await;
            nodes.keys().cloned().collect()
        };

        let futures: Vec<_> = node_ids
            .iter()
            .map(|id| self.send_message(id.as_str(), message.clone()))
            .collect();

        let results = join_all(futures).await;

        for (id, result) in node_ids.iter().zip(results) {
            if let Err(e) = result {
                warn!("Broadcast to node {} failed: {}", id, e);
            }
        }

        Ok(())
    }

    async fn receive_messages(&self) -> FusekiResult<mpsc::Receiver<(String, NodeMessage)>> {
        let mut slot = self.inbound_rx.lock().await;
        slot.take().ok_or_else(|| {
            FusekiError::internal(
                "receive_messages() called more than once; receiver already consumed".to_string(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Helper that builds a minimal ClusterConfig pointing at a fixed address so
    // tests do not race over port 7000 when run in parallel.
    fn test_config(addr: &str) -> ClusterConfig {
        ClusterConfig {
            bind_addr: addr.parse().expect("valid addr"),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_cluster_node_creation() {
        let config = test_config("127.0.0.1:17000");
        let communication = Arc::new(TcpNodeCommunication::new(config.bind_addr));

        let node = ClusterNode::new(config.clone(), communication)
            .await
            .expect("node creation should succeed");
        let info = node.get_node_info().await;

        assert_eq!(info.id, config.node_id);
        assert_eq!(info.addr, config.bind_addr);
        assert_eq!(info.state, NodeState::Joining);
    }

    /// Verify that `send_message` returns an error when the target is not in
    /// `known_nodes` (no TCP attempt is made).
    #[tokio::test]
    async fn test_send_message_unknown_target_fails() {
        let addr = "127.0.0.1:0".parse().expect("valid addr");
        let comm = TcpNodeCommunication::new(addr);

        let message = NodeMessage::LeaveNotification {
            node_id: "self".to_string(),
        };
        let result = comm.send_message("nonexistent-node", message).await;
        assert!(
            result.is_err(),
            "sending to unknown node must return an error"
        );
    }

    /// Integration test: start a real TCP listener, send a Heartbeat over
    /// loopback, and assert the receiver delivers the message with the correct
    /// sender ID.
    ///
    /// Strategy: bind a temporary listener on port 0 to let the OS pick a free
    /// port, record its address, drop it, then give that address to the receiver
    /// `TcpNodeCommunication` instance.  This sidesteps the fact that we cannot
    /// query `TcpNodeCommunication`'s bound port after the fact.
    #[tokio::test]
    async fn test_tcp_send_receive_loopback() {
        // Grab a free port from the OS.
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe bind");
        let receiver_addr = probe.local_addr().expect("local_addr");
        drop(probe);

        // Build the receiver comm bound to that address.
        let receiver_comm = Arc::new(TcpNodeCommunication::with_node_id(
            receiver_addr,
            "receiver-node".to_string(),
        ));
        receiver_comm
            .start_listener()
            .await
            .expect("receiver start_listener");

        // Give the listener task a moment to accept connections.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Build the sender comm (its own bind_addr is irrelevant; it only connects).
        let sender_comm = TcpNodeCommunication::with_node_id(
            "127.0.0.1:0".parse().expect("valid"),
            "sender-node".to_string(),
        );
        sender_comm
            .add_node("receiver-node".to_string(), receiver_addr)
            .await;

        // Consume the inbound channel from the receiver before sending.
        let mut rx = receiver_comm
            .receive_messages()
            .await
            .expect("receive_messages");

        // Send a Heartbeat from sender → receiver.
        let sent_msg = NodeMessage::Heartbeat {
            node_id: "sender-node".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            metadata: NodeMetadata {
                datacenter: None,
                rack: None,
                capacity: 500,
                load: 0.25,
                version: "1.0.0".to_string(),
            },
        };
        sender_comm
            .send_message("receiver-node", sent_msg.clone())
            .await
            .expect("send_message should succeed");

        // Wait for the message to arrive via the TCP listener → inbound channel.
        let received = tokio::time::timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("message should arrive within 500 ms")
            .expect("inbound channel should not be closed");

        assert_eq!(
            received.0, "sender-node",
            "sender_id must be embedded in frame"
        );
        match (received.1, sent_msg) {
            (
                NodeMessage::Heartbeat {
                    node_id: recv_id,
                    metadata: recv_meta,
                    ..
                },
                NodeMessage::Heartbeat {
                    node_id: sent_id,
                    metadata: sent_meta,
                    ..
                },
            ) => {
                assert_eq!(recv_id, sent_id);
                assert_eq!(recv_meta.capacity, sent_meta.capacity);
            }
            _ => panic!("received wrong message variant"),
        }
    }

    /// Unit test: verify the frame encode/decode round-trip without a real
    /// network by using a tokio in-memory DuplexStream.
    #[tokio::test]
    async fn test_frame_encode_decode_round_trip() {
        use tokio::io::duplex;

        let message = NodeMessage::LeaderElection {
            candidate_id: "alpha".to_string(),
            term: 42,
        };
        let sender_id = "node-1".to_string();
        let envelope = serde_json::to_vec(&(&sender_id, &message)).expect("serialise envelope");

        let (mut client, mut server) = duplex(4096);

        // Write frame on the client side.
        write_frame(&mut client, &envelope)
            .await
            .expect("write_frame");

        // Read raw frame bytes on the server side.
        let raw = read_frame(&mut server).await.expect("read_frame");

        let (decoded_id, decoded_msg): (String, NodeMessage) =
            serde_json::from_slice(&raw).expect("deserialize");

        assert_eq!(decoded_id, sender_id);
        match decoded_msg {
            NodeMessage::LeaderElection { candidate_id, term } => {
                assert_eq!(candidate_id, "alpha");
                assert_eq!(term, 42);
            }
            _ => panic!("unexpected message variant"),
        }
    }

    /// Verify that read_frame rejects a length prefix that exceeds MAX_FRAME_BYTES.
    #[tokio::test]
    async fn test_frame_oversized_rejected() {
        use tokio::io::duplex;

        let (mut client, mut server) = duplex(8);
        // Write a length prefix larger than MAX_FRAME_BYTES.
        let huge_len = (MAX_FRAME_BYTES as u32 + 1).to_be_bytes();
        client.write_all(&huge_len).await.expect("write len");

        let result = read_frame(&mut server).await;
        assert!(result.is_err(), "oversized frame must be rejected");
    }

    /// Verify that a node correctly grants a vote when it has not yet voted
    /// in the candidate's term (basic Raft single-vote rule).
    #[tokio::test]
    async fn test_leader_election_vote_granted() {
        let config = test_config("127.0.0.1:17002");
        let comm = Arc::new(TcpNodeCommunication::new(config.bind_addr));
        let node = ClusterNode::new(config.clone(), comm.clone())
            .await
            .expect("node creation should succeed");

        // Simulate receiving a LeaderElection message via the inbound channel.
        let candidate_id = "candidate-node-1".to_string();
        let election_term = 5u64;

        comm.inject_message(
            candidate_id.clone(),
            NodeMessage::LeaderElection {
                candidate_id: candidate_id.clone(),
                term: election_term,
            },
        )
        .await
        .expect("inject should succeed");

        // Start processing so the spawned task reads the injected message.
        node.start_message_processing()
            .await
            .expect("start_message_processing should succeed");

        // Give the async task a moment to process the message.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // After processing, the node's current_term must have been updated to
        // the candidate's term and voted_for must be set to the candidate.
        assert_eq!(node.current_term(), election_term);
        assert_eq!(
            node.voted_for().await.as_deref(),
            Some(candidate_id.as_str())
        );
    }

    /// Verify that a node rejects a second vote request in the same term from
    /// a different candidate (Raft: one vote per term).
    #[tokio::test]
    async fn test_leader_election_second_vote_rejected() {
        let config = test_config("127.0.0.1:17003");
        let comm = Arc::new(TcpNodeCommunication::new(config.bind_addr));
        let node = ClusterNode::new(config.clone(), comm.clone())
            .await
            .expect("node creation should succeed");

        // First vote for candidate A.
        comm.inject_message(
            "candidate-a".to_string(),
            NodeMessage::LeaderElection {
                candidate_id: "candidate-a".to_string(),
                term: 3,
            },
        )
        .await
        .expect("inject should succeed");

        // Second vote request (same term, different candidate) — should be rejected.
        comm.inject_message(
            "candidate-b".to_string(),
            NodeMessage::LeaderElection {
                candidate_id: "candidate-b".to_string(),
                term: 3,
            },
        )
        .await
        .expect("inject should succeed");

        node.start_message_processing()
            .await
            .expect("start_message_processing should succeed");

        tokio::time::sleep(Duration::from_millis(100)).await;

        // voted_for must remain "candidate-a" (first come, first served).
        assert_eq!(node.voted_for().await.as_deref(), Some("candidate-a"));
    }

    /// Verify that a node updates its term when it sees a higher-term election
    /// message, even if it then votes for that candidate.
    #[tokio::test]
    async fn test_leader_election_higher_term_updates_state() {
        let config = test_config("127.0.0.1:17004");
        let comm = Arc::new(TcpNodeCommunication::new(config.bind_addr));
        let node = ClusterNode::new(config.clone(), comm.clone())
            .await
            .expect("node creation should succeed");

        // Start at term 0; receive election for term 10.
        comm.inject_message(
            "candidate-x".to_string(),
            NodeMessage::LeaderElection {
                candidate_id: "candidate-x".to_string(),
                term: 10,
            },
        )
        .await
        .expect("inject should succeed");

        node.start_message_processing()
            .await
            .expect("start_message_processing should succeed");

        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(node.current_term(), 10);
        assert_eq!(node.voted_for().await.as_deref(), Some("candidate-x"));
    }

    /// Verify that `receive_messages` returns an error when called a second time
    /// (the receiver can only be consumed once per `TcpNodeCommunication` instance).
    #[tokio::test]
    async fn test_receive_messages_consumed_once() {
        let addr = "127.0.0.1:17005".parse().expect("valid addr");
        let comm = TcpNodeCommunication::new(addr);

        let _rx = comm
            .receive_messages()
            .await
            .expect("first call should succeed");
        let err = comm.receive_messages().await;
        assert!(err.is_err(), "second call should fail");
    }

    /// Smoke-test for CPU/memory metric collection helpers.
    #[tokio::test]
    async fn test_get_cpu_and_memory_usage() {
        let sys = parking_lot::Mutex::new(System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        ));

        let cpu = ClusterNode::get_cpu_usage(&sys).await;
        let mem = ClusterNode::get_memory_usage(&sys).await;
        let net = ClusterNode::get_network_io(&sys).await;

        // CPU should be a non-negative percentage; memory should be > 0 on any
        // real host; network always reports (0, 0) for now.
        assert!(cpu >= 0.0);
        assert!(mem > 0);
        assert_eq!(net, (0, 0));
    }
}
