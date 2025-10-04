//! Raft consensus with optimized log compaction
//!
//! This module implements the Raft consensus algorithm optimized for RDF data,
//! with efficient log compaction and snapshot management.

#![allow(dead_code)]

use crate::model::{Triple, TriplePattern};
use crate::OxirsError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::interval;

/// Raft configuration
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Node ID
    pub node_id: String,
    /// Cluster peers
    pub peers: Vec<RaftPeer>,
    /// Election timeout range (ms)
    pub election_timeout: (u64, u64),
    /// Heartbeat interval (ms)
    pub heartbeat_interval: u64,
    /// Log compaction configuration
    pub compaction: CompactionConfig,
    /// Snapshot configuration
    pub snapshot: SnapshotConfig,
    /// Storage path
    pub storage_path: String,
}

/// Raft peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftPeer {
    /// Peer ID
    pub id: String,
    /// Peer address
    pub address: SocketAddr,
    /// Voting member
    pub voting: bool,
}

/// Log compaction configuration
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Enable automatic compaction
    pub auto_compact: bool,
    /// Compaction threshold (number of entries)
    pub threshold: usize,
    /// Minimum entries to keep
    pub min_entries: usize,
    /// Delta compression for similar entries
    pub delta_compression: bool,
    /// Batch size for compaction
    pub batch_size: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        CompactionConfig {
            auto_compact: true,
            threshold: 10000,
            min_entries: 1000,
            delta_compression: true,
            batch_size: 1000,
        }
    }
}

/// Vote request parameters
#[derive(Debug, Clone)]
struct VoteRequestParams {
    pub request_term: u64,
    pub candidate_id: String,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

/// Append entries request parameters  
#[derive(Debug, Clone)]
struct AppendEntriesParams {
    pub request_term: u64,
    pub leader_id: String,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<RaftLogEntry>,
    pub leader_commit: u64,
}

/// Snapshot configuration
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Enable automatic snapshots
    pub auto_snapshot: bool,
    /// Snapshot interval (entries)
    pub interval: usize,
    /// Incremental snapshots
    pub incremental: bool,
    /// Compression for snapshots
    pub compression: bool,
    /// Maximum concurrent snapshots
    pub max_concurrent: usize,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        SnapshotConfig {
            auto_snapshot: true,
            interval: 50000,
            incremental: true,
            compression: true,
            max_concurrent: 2,
        }
    }
}

/// Raft node states
#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
    Learner,
}

/// Log entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    /// Add triple
    AddTriple(Triple),
    /// Remove triple
    RemoveTriple(Triple),
    /// Batch add
    BatchAdd(Vec<Triple>),
    /// Batch remove
    BatchRemove(Vec<Triple>),
    /// Configuration change
    ConfigChange(ConfigChangeEntry),
    /// Snapshot marker
    SnapshotMarker(SnapshotInfo),
    /// Compacted entry
    CompactedEntry(CompactedData),
}

/// Configuration change entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChangeEntry {
    /// Change type
    pub change_type: ConfigChangeType,
    /// Peer info
    pub peer: RaftPeer,
}

/// Configuration change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigChangeType {
    AddNode,
    RemoveNode,
    PromoteToVoter,
    DemoteToLearner,
}

/// Snapshot information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// Snapshot ID
    pub id: String,
    /// Index of last included entry
    pub last_index: u64,
    /// Term of last included entry
    pub last_term: u64,
    /// Snapshot size
    pub size: usize,
    /// Checksum
    pub checksum: String,
}

/// Compacted log data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactedData {
    /// Start index
    pub start_index: u64,
    /// End index
    pub end_index: u64,
    /// Compacted size
    pub size: usize,
    /// Delta-compressed data
    pub data: Vec<u8>,
    /// Reference snapshot
    pub base_snapshot: Option<String>,
}

/// Raft message types
#[derive(Debug)]
pub enum RaftMessage {
    /// Request vote
    VoteRequest {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// Vote response
    VoteResponse { term: u64, vote_granted: bool },
    /// Append entries (heartbeat/replication)
    AppendEntries {
        term: u64,
        _leader_id: String,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftLogEntry>,
        leader_commit: u64,
    },
    /// Append entries response
    AppendResponse {
        term: u64,
        success: bool,
        match_index: u64,
        conflict_term: Option<u64>,
        conflict_index: Option<u64>,
    },
    /// Install snapshot
    InstallSnapshot {
        term: u64,
        _leader_id: String,
        last_included_index: u64,
        last_included_term: u64,
        offset: u64,
        data: Vec<u8>,
        done: bool,
    },
    /// Client request
    ClientRequest {
        request_id: String,
        entry: LogEntry,
        response_tx: oneshot::Sender<Result<(), OxirsError>>,
    },
}

/// Raft log entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftLogEntry {
    /// Entry index
    pub index: u64,
    /// Entry term
    pub term: u64,
    /// Entry data
    pub entry: LogEntry,
    /// Timestamp
    pub timestamp: u64,
}

/// Raft node implementation
pub struct RaftNode {
    /// Configuration
    config: RaftConfig,
    /// Current state
    state: Arc<RwLock<NodeState>>,
    /// Current term
    current_term: Arc<RwLock<u64>>,
    /// Voted for
    voted_for: Arc<RwLock<Option<String>>>,
    /// Log entries
    log: Arc<RwLock<RaftLog>>,
    /// Commit index
    commit_index: Arc<RwLock<u64>>,
    /// Last applied
    last_applied: Arc<RwLock<u64>>,
    /// Leader state
    leader_state: Arc<RwLock<Option<LeaderState>>>,
    /// Message channels
    message_tx: mpsc::Sender<RaftMessage>,
    message_rx: Arc<Mutex<mpsc::Receiver<RaftMessage>>>,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
    /// Statistics
    stats: Arc<RwLock<RaftStats>>,
}

/// Raft log with optimized compaction
struct RaftLog {
    /// Log entries
    entries: VecDeque<RaftLogEntry>,
    /// Compacted entries
    compacted: HashMap<u64, CompactedData>,
    /// Snapshot metadata
    snapshots: HashMap<String, SnapshotInfo>,
    /// Start index (after compaction)
    start_index: u64,
    /// Compaction state
    compaction_state: CompactionState,
}

/// Compaction state
struct CompactionState {
    /// Last compaction index
    last_compacted: u64,
    /// Pending compaction
    pending: Option<CompactionJob>,
    /// Compaction statistics
    stats: CompactionStats,
}

/// Compaction job
#[allow(dead_code)]
struct CompactionJob {
    /// Start index
    start: u64,
    /// End index
    end: u64,
    /// Start time
    start_time: Instant,
}

/// Compaction statistics
#[derive(Debug, Default)]
struct CompactionStats {
    /// Total compactions
    total_compactions: u64,
    /// Entries compacted
    entries_compacted: u64,
    /// Space saved
    space_saved_bytes: u64,
    /// Compression ratio
    compression_ratio: f64,
}

/// Leader-specific state
struct LeaderState {
    /// Next index for each peer
    next_index: HashMap<String, u64>,
    /// Match index for each peer
    match_index: HashMap<String, u64>,
    /// Replication progress
    replication_progress: HashMap<String, ReplicationProgress>,
    /// Pending client requests
    pending_requests: HashMap<String, PendingRequest>,
}

/// Replication progress tracking
struct ReplicationProgress {
    /// Last sent time
    last_sent: Instant,
    /// Consecutive failures
    failures: u32,
    /// In-flight entries
    in_flight: u64,
    /// Bandwidth estimate
    bandwidth_bps: f64,
}

/// Pending client request
struct PendingRequest {
    /// Request ID
    request_id: String,
    /// Log index
    log_index: u64,
    /// Response channel
    response_tx: oneshot::Sender<Result<(), OxirsError>>,
    /// Timeout
    timeout: Instant,
}

/// Raft statistics
#[derive(Debug, Default)]
struct RaftStats {
    /// Elections held
    elections_held: u64,
    /// Elections won
    elections_won: u64,
    /// Messages sent
    messages_sent: u64,
    /// Messages received
    messages_received: u64,
    /// Entries replicated
    entries_replicated: u64,
    /// Snapshots sent
    snapshots_sent: u64,
    /// Snapshots received
    snapshots_received: u64,
}

impl RaftNode {
    /// Create new Raft node
    pub async fn new(config: RaftConfig) -> Result<Self, OxirsError> {
        let (message_tx, message_rx) = mpsc::channel(10000);

        Ok(RaftNode {
            config,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            current_term: Arc::new(RwLock::new(0)),
            voted_for: Arc::new(RwLock::new(None)),
            log: Arc::new(RwLock::new(RaftLog::new())),
            commit_index: Arc::new(RwLock::new(0)),
            last_applied: Arc::new(RwLock::new(0)),
            leader_state: Arc::new(RwLock::new(None)),
            message_tx,
            message_rx: Arc::new(Mutex::new(message_rx)),
            shutdown: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(RaftStats::default())),
        })
    }

    /// Start Raft node
    pub async fn start(&self) -> Result<(), OxirsError> {
        // Start message processing
        self.spawn_message_processor();

        // Start election timer
        self.spawn_election_timer();

        // Start heartbeat timer (for leaders)
        self.spawn_heartbeat_timer();

        // Start log compaction
        if self.config.compaction.auto_compact {
            self.spawn_compaction_worker();
        }

        // Start snapshot manager
        if self.config.snapshot.auto_snapshot {
            self.spawn_snapshot_manager();
        }

        Ok(())
    }

    /// Submit a client request
    pub async fn submit(&self, entry: LogEntry) -> Result<(), OxirsError> {
        let state = self.state.read().await;
        if *state != NodeState::Leader {
            return Err(OxirsError::Store("Not the leader".to_string()));
        }

        let request_id = uuid::Uuid::new_v4().to_string();
        let (response_tx, response_rx) = oneshot::channel();

        self.message_tx
            .send(RaftMessage::ClientRequest {
                request_id,
                entry,
                response_tx,
            })
            .await
            .map_err(|_| OxirsError::Store("Failed to send request".to_string()))?;

        response_rx
            .await
            .map_err(|_| OxirsError::Store("Request cancelled".to_string()))?
    }

    /// Query committed data
    pub async fn query(&self, pattern: &TriplePattern) -> Result<Vec<Triple>, OxirsError> {
        let log = self.log.read().await;
        let last_applied = *self.last_applied.read().await;

        let mut results = Vec::new();
        let mut current_state = HashSet::new();

        // Apply log entries up to last_applied
        for entry in &log.entries {
            if entry.index > last_applied {
                break;
            }

            match &entry.entry {
                LogEntry::AddTriple(triple) => {
                    current_state.insert(triple.clone());
                }
                LogEntry::RemoveTriple(triple) => {
                    current_state.remove(triple);
                }
                LogEntry::BatchAdd(triples) => {
                    for triple in triples {
                        current_state.insert(triple.clone());
                    }
                }
                LogEntry::BatchRemove(triples) => {
                    for triple in triples {
                        current_state.remove(triple);
                    }
                }
                _ => {}
            }
        }

        // Filter by pattern
        for triple in current_state {
            if pattern.matches(&triple) {
                results.push(triple);
            }
        }

        Ok(results)
    }

    /// Spawn message processor
    fn spawn_message_processor(&self) {
        let message_rx = self.message_rx.clone();
        let state = self.state.clone();
        let current_term = self.current_term.clone();
        let voted_for = self.voted_for.clone();
        let log = self.log.clone();
        let commit_index = self.commit_index.clone();
        let leader_state = self.leader_state.clone();
        let stats = self.stats.clone();
        let node_id = self.config.node_id.clone();

        tokio::spawn(async move {
            let mut rx = message_rx.lock().await;
            while let Some(message) = rx.recv().await {
                let mut stats_guard = stats.write().await;
                stats_guard.messages_received += 1;
                drop(stats_guard);

                match message {
                    RaftMessage::VoteRequest {
                        term,
                        candidate_id,
                        last_log_index,
                        last_log_term,
                    } => {
                        Self::handle_vote_request(
                            &state,
                            &current_term,
                            &voted_for,
                            &log,
                            &node_id,
                            VoteRequestParams {
                                request_term: term,
                                candidate_id,
                                last_log_index,
                                last_log_term,
                            },
                        )
                        .await;
                    }
                    RaftMessage::AppendEntries {
                        term,
                        _leader_id,
                        prev_log_index,
                        prev_log_term,
                        entries,
                        leader_commit,
                    } => {
                        Self::handle_append_entries(
                            &state,
                            &current_term,
                            &log,
                            &commit_index,
                            AppendEntriesParams {
                                request_term: term,
                                leader_id: _leader_id,
                                prev_log_index,
                                prev_log_term,
                                entries,
                                leader_commit,
                            },
                        )
                        .await;
                    }
                    RaftMessage::ClientRequest {
                        request_id,
                        entry,
                        response_tx,
                    } => {
                        Self::handle_client_request(
                            &state,
                            &current_term,
                            &log,
                            &leader_state,
                            request_id,
                            entry,
                            response_tx,
                        )
                        .await;
                    }
                    _ => {}
                }
            }
        });
    }

    /// Spawn election timer
    fn spawn_election_timer(&self) {
        let state = self.state.clone();
        let current_term = self.current_term.clone();
        let voted_for = self.voted_for.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            #[allow(unused_imports)]
            use scirs2_core::random::{Random, Rng};

            loop {
                // Random election timeout
                let timeout = {
                    let mut random = Random::default();
                    random.gen_range(config.election_timeout.0..config.election_timeout.1)
                };
                tokio::time::sleep(Duration::from_millis(timeout)).await;

                if *shutdown.read().await {
                    break;
                }

                let current_state = state.read().await.clone();
                if current_state == NodeState::Follower || current_state == NodeState::Candidate {
                    // Start election
                    Self::start_election(&state, &current_term, &voted_for, &config, &stats).await;
                }
            }
        });
    }

    /// Spawn heartbeat timer
    fn spawn_heartbeat_timer(&self) {
        let state = self.state.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.heartbeat_interval));

            loop {
                interval.tick().await;

                if *shutdown.read().await {
                    break;
                }

                let current_state = state.read().await.clone();
                if current_state == NodeState::Leader {
                    // Send heartbeats
                    Self::send_heartbeats(&config).await;
                }
            }
        });
    }

    /// Spawn compaction worker
    fn spawn_compaction_worker(&self) {
        let log = self.log.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute

            loop {
                interval.tick().await;

                if *shutdown.read().await {
                    break;
                }

                // Check if compaction needed
                let mut log_guard = log.write().await;
                if log_guard.entries.len() > config.compaction.threshold {
                    Self::compact_log(&mut log_guard, &config.compaction).await;
                }
            }
        });
    }

    /// Spawn snapshot manager
    fn spawn_snapshot_manager(&self) {
        let log = self.log.clone();
        let commit_index = self.commit_index.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Check every 5 minutes

            loop {
                interval.tick().await;

                if *shutdown.read().await {
                    break;
                }

                let current_commit = *commit_index.read().await;
                let log_guard = log.read().await;

                // Check if snapshot needed
                if let Some(last_snapshot) =
                    log_guard.snapshots.values().max_by_key(|s| s.last_index)
                {
                    if current_commit - last_snapshot.last_index > config.snapshot.interval as u64 {
                        drop(log_guard);
                        Self::create_snapshot(&log, current_commit, &config.snapshot).await;
                    }
                } else if current_commit > config.snapshot.interval as u64 {
                    drop(log_guard);
                    Self::create_snapshot(&log, current_commit, &config.snapshot).await;
                }
            }
        });
    }

    /// Handle vote request
    async fn handle_vote_request(
        state: &Arc<RwLock<NodeState>>,
        current_term: &Arc<RwLock<u64>>,
        voted_for: &Arc<RwLock<Option<String>>>,
        log: &Arc<RwLock<RaftLog>>,
        node_id: &str,
        request: VoteRequestParams,
    ) {
        let mut term = current_term.write().await;
        let mut voted = voted_for.write().await;

        // Update term if needed
        if request.request_term > *term {
            *term = request.request_term;
            *voted = None;
            *state.write().await = NodeState::Follower;
        }

        // Check if we can vote
        let vote_granted = if request.request_term < *term
            || (voted.is_some() && voted.as_ref().unwrap() != &request.candidate_id)
        {
            false
        } else {
            // Check log up-to-date
            let log_guard = log.read().await;
            let our_last_index = log_guard.last_index();
            let our_last_term = log_guard.last_term();
            drop(log_guard);

            request.last_log_term > our_last_term
                || (request.last_log_term == our_last_term
                    && request.last_log_index >= our_last_index)
        };

        if vote_granted {
            *voted = Some(request.candidate_id);
        }

        // Send response (would use actual networking)
        tracing::info!(
            "Node {} vote response: term={}, granted={}",
            node_id,
            *term,
            vote_granted
        );
    }

    /// Handle append entries
    async fn handle_append_entries(
        state: &Arc<RwLock<NodeState>>,
        current_term: &Arc<RwLock<u64>>,
        log: &Arc<RwLock<RaftLog>>,
        commit_index: &Arc<RwLock<u64>>,
        request: AppendEntriesParams,
    ) {
        let mut term = current_term.write().await;

        // Update term if needed
        if request.request_term > *term {
            *term = request.request_term;
            *state.write().await = NodeState::Follower;
        }

        // Reject if term is old
        if request.request_term < *term {
            return;
        }

        // Reset to follower
        *state.write().await = NodeState::Follower;

        // Check log consistency
        let mut log_guard = log.write().await;
        let success = if request.prev_log_index == 0 {
            true
        } else if let Some(entry) = log_guard.get(request.prev_log_index) {
            entry.term == request.prev_log_term
        } else {
            false
        };

        if success {
            // Append entries
            for entry in request.entries {
                log_guard.append(entry);
            }

            // Update commit index
            if request.leader_commit > *commit_index.read().await {
                let last_index = log_guard.last_index();
                *commit_index.write().await = request.leader_commit.min(last_index);
            }
        }
    }

    /// Handle client request
    async fn handle_client_request(
        state: &Arc<RwLock<NodeState>>,
        current_term: &Arc<RwLock<u64>>,
        log: &Arc<RwLock<RaftLog>>,
        leader_state: &Arc<RwLock<Option<LeaderState>>>,
        request_id: String,
        entry: LogEntry,
        response_tx: oneshot::Sender<Result<(), OxirsError>>,
    ) {
        let current_state = state.read().await.clone();
        if current_state != NodeState::Leader {
            let _ = response_tx.send(Err(OxirsError::Store("Not the leader".to_string())));
            return;
        }

        // Append to log
        let term = *current_term.read().await;
        let mut log_guard = log.write().await;
        let index = log_guard.next_index();

        let raft_entry = RaftLogEntry {
            index,
            term,
            entry,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        log_guard.append(raft_entry);
        drop(log_guard);

        // Track pending request
        if let Some(ref mut leader) = *leader_state.write().await {
            leader.pending_requests.insert(
                request_id.clone(),
                PendingRequest {
                    request_id,
                    log_index: index,
                    response_tx,
                    timeout: Instant::now() + Duration::from_secs(5),
                },
            );
        }
    }

    /// Start election
    async fn start_election(
        state: &Arc<RwLock<NodeState>>,
        current_term: &Arc<RwLock<u64>>,
        voted_for: &Arc<RwLock<Option<String>>>,
        config: &RaftConfig,
        stats: &Arc<RwLock<RaftStats>>,
    ) {
        *state.write().await = NodeState::Candidate;
        let mut term = current_term.write().await;
        *term += 1;
        *voted_for.write().await = Some(config.node_id.clone());

        let mut stats_guard = stats.write().await;
        stats_guard.elections_held += 1;

        tracing::info!(
            "Node {} starting election for term {}",
            config.node_id,
            *term
        );

        // In real implementation, would send vote requests to all peers
        // and handle responses
    }

    /// Send heartbeats
    async fn send_heartbeats(config: &RaftConfig) {
        // In real implementation, would send heartbeat messages to all followers
        tracing::debug!("Leader {} sending heartbeats", config.node_id);
    }

    /// Compact log
    async fn compact_log(log: &mut RaftLog, config: &CompactionConfig) {
        if log.entries.len() <= config.min_entries {
            return;
        }
        let entries_to_compact = log.entries.len() - config.min_entries;

        tracing::info!(
            "Starting log compaction, compacting {} entries",
            entries_to_compact
        );

        // Create compaction job
        let start_index = log.start_index;
        let end_index = start_index + entries_to_compact as u64;

        log.compaction_state.pending = Some(CompactionJob {
            start: start_index,
            end: end_index,
            start_time: Instant::now(),
        });

        // Perform compaction (simplified)
        let mut compacted_data = Vec::new();
        let mut removed_entries = Vec::new();

        for _ in 0..entries_to_compact {
            if let Some(entry) = log.entries.pop_front() {
                // Serialize entry for compaction
                let serialized = bincode::serialize(&entry).unwrap();
                compacted_data.extend_from_slice(&serialized);
                removed_entries.push(entry);
            }
        }

        // Apply compression
        let compressed = if config.delta_compression {
            // Delta compression would go here
            zstd::encode_all(&compacted_data[..], 3).unwrap()
        } else {
            compacted_data
        };

        // Store compacted data
        let compressed_size = compressed.len();
        let compacted = CompactedData {
            start_index,
            end_index,
            size: compressed_size,
            data: compressed,
            base_snapshot: None,
        };

        log.compacted.insert(start_index, compacted);
        log.start_index = end_index;

        // Update compaction state
        log.compaction_state.last_compacted = end_index;
        log.compaction_state.pending = None;
        log.compaction_state.stats.total_compactions += 1;
        log.compaction_state.stats.entries_compacted += entries_to_compact as u64;
        log.compaction_state.stats.space_saved_bytes +=
            (removed_entries.len() * std::mem::size_of::<RaftLogEntry>()) as u64
                - compressed_size as u64;

        tracing::info!(
            "Log compaction completed, saved {} bytes",
            log.compaction_state.stats.space_saved_bytes
        );
    }

    /// Create snapshot
    async fn create_snapshot(
        log: &Arc<RwLock<RaftLog>>,
        last_index: u64,
        _config: &SnapshotConfig,
    ) {
        tracing::info!("Creating snapshot at index {}", last_index);

        // In real implementation, would create actual snapshot
        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let snapshot_info = SnapshotInfo {
            id: snapshot_id.clone(),
            last_index,
            last_term: 0, // Would get from log
            size: 0,      // Would calculate
            checksum: "dummy".to_string(),
        };

        let mut log_guard = log.write().await;
        log_guard.snapshots.insert(snapshot_id, snapshot_info);
    }
}

impl RaftLog {
    /// Create new log
    fn new() -> Self {
        RaftLog {
            entries: VecDeque::new(),
            compacted: HashMap::new(),
            snapshots: HashMap::new(),
            start_index: 1,
            compaction_state: CompactionState {
                last_compacted: 0,
                pending: None,
                stats: CompactionStats::default(),
            },
        }
    }

    /// Get entry at index
    fn get(&self, index: u64) -> Option<&RaftLogEntry> {
        if index < self.start_index {
            // Entry is compacted
            None
        } else {
            let offset = (index - self.start_index) as usize;
            self.entries.get(offset)
        }
    }

    /// Append entry
    fn append(&mut self, entry: RaftLogEntry) {
        self.entries.push_back(entry);
    }

    /// Get last index
    fn last_index(&self) -> u64 {
        if self.entries.is_empty() {
            self.start_index - 1
        } else {
            self.start_index + self.entries.len() as u64 - 1
        }
    }

    /// Get last term
    fn last_term(&self) -> u64 {
        self.entries.back().map(|e| e.term).unwrap_or(0)
    }

    /// Get next index
    fn next_index(&self) -> u64 {
        self.last_index() + 1
    }
}

/// Raft storage trait for persistence
#[async_trait]
pub trait RaftStorage: Send + Sync {
    /// Save current term
    async fn save_term(&mut self, term: u64) -> Result<(), OxirsError>;

    /// Load current term
    async fn load_term(&self) -> Result<u64, OxirsError>;

    /// Save voted for
    async fn save_voted_for(&mut self, voted_for: Option<String>) -> Result<(), OxirsError>;

    /// Load voted for
    async fn load_voted_for(&self) -> Result<Option<String>, OxirsError>;

    /// Append log entries
    async fn append_entries(&mut self, entries: Vec<RaftLogEntry>) -> Result<(), OxirsError>;

    /// Load log entries
    async fn load_entries(&self, start: u64, end: u64) -> Result<Vec<RaftLogEntry>, OxirsError>;

    /// Save snapshot
    async fn save_snapshot(
        &mut self,
        snapshot: SnapshotInfo,
        data: Vec<u8>,
    ) -> Result<(), OxirsError>;

    /// Load snapshot
    async fn load_snapshot(&self, id: &str) -> Result<(SnapshotInfo, Vec<u8>), OxirsError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[tokio::test]
    async fn test_raft_node_creation() {
        let config = RaftConfig {
            node_id: "node1".to_string(),
            peers: vec![],
            election_timeout: (150, 300),
            heartbeat_interval: 50,
            compaction: CompactionConfig::default(),
            snapshot: SnapshotConfig::default(),
            storage_path: "/tmp/raft_test".to_string(),
        };

        let node = RaftNode::new(config).await.unwrap();

        // Check initial state
        assert_eq!(*node.state.read().await, NodeState::Follower);
        assert_eq!(*node.current_term.read().await, 0);
        assert_eq!(*node.commit_index.read().await, 0);
    }

    #[tokio::test]
    async fn test_log_operations() {
        let mut log = RaftLog::new();

        // Add entries
        for i in 1..=10 {
            let entry = RaftLogEntry {
                index: i,
                term: 1,
                entry: LogEntry::AddTriple(Triple::new(
                    NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
                    NamedNode::new("http://example.org/p").unwrap(),
                    crate::model::Object::Literal(Literal::new(format!("value{i}"))),
                )),
                timestamp: i,
            };
            log.append(entry);
        }

        assert_eq!(log.last_index(), 10);
        assert_eq!(log.last_term(), 1);
        assert_eq!(log.entries.len(), 10);

        // Test get
        assert!(log.get(5).is_some());
        assert_eq!(log.get(5).unwrap().index, 5);
    }

    #[tokio::test]
    async fn test_log_compaction() {
        let mut log = RaftLog::new();

        // Add many entries
        for i in 1..=100 {
            let entry = RaftLogEntry {
                index: i,
                term: 1,
                entry: LogEntry::AddTriple(Triple::new(
                    NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
                    NamedNode::new("http://example.org/p").unwrap(),
                    crate::model::Object::Literal(Literal::new(format!("value{i}"))),
                )),
                timestamp: i,
            };
            log.append(entry);
        }

        let config = CompactionConfig {
            auto_compact: true,
            threshold: 50,
            min_entries: 10,
            delta_compression: true,
            batch_size: 10,
        };

        // Compact log
        RaftNode::compact_log(&mut log, &config).await;

        // Check compaction
        assert!(log.entries.len() <= config.min_entries);
        assert!(log.start_index > 1);
        assert!(!log.compacted.is_empty());
        assert_eq!(log.compaction_state.stats.total_compactions, 1);
    }
}
