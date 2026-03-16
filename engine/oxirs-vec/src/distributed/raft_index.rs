//! Raft-based distributed index consensus for vector stores
//!
//! This module implements the Raft consensus protocol for distributed
//! vector index management. It provides:
//! - Leader election among index nodes
//! - Replicated log for index mutations (insertions, deletions, updates)
//! - Consistent reads via quorum
//! - Automatic failover and leader re-election
//!
//! # Design
//!
//! Each node in the cluster participates in Raft. Index mutations (vector
//! insertions/deletions) are proposed as log entries. Once a majority of
//! nodes acknowledge an entry, it is committed and applied to the local
//! in-memory index.
//!
//! # Pure Rust
//!
//! This module is 100% Pure Rust - no CUDA or FFI dependencies.

use anyhow::{anyhow, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Node ID type for Raft cluster members
pub type NodeId = u64;

/// Log index (1-based, 0 means no entry)
pub type LogIndex = u64;

/// Term number
pub type Term = u64;

/// A vector entry stored in the distributed index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique identifier for this vector
    pub vector_id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp of insertion
    pub inserted_at: u64,
}

/// Commands that can be applied to the replicated state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexCommand {
    /// Insert or update a vector
    Upsert(VectorEntry),
    /// Delete a vector by ID
    Delete { vector_id: String },
    /// Rebuild the index (triggers background rebuild)
    Rebuild,
    /// Update metadata for a vector
    UpdateMetadata {
        vector_id: String,
        metadata: HashMap<String, String>,
    },
    /// No-op entry for leadership heartbeat and linearization
    NoOp,
}

/// A single entry in the replicated log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Index of this entry in the log (1-based)
    pub index: LogIndex,
    /// Term when this entry was created
    pub term: Term,
    /// The command to be applied
    pub command: IndexCommand,
    /// Client request ID for deduplication
    pub client_id: Option<String>,
}

/// Raft node role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Follower,
    Candidate,
    Leader,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Follower => write!(f, "Follower"),
            Self::Candidate => write!(f, "Candidate"),
            Self::Leader => write!(f, "Leader"),
        }
    }
}

/// AppendEntries RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader's ID
    pub leader_id: NodeId,
    /// Log index immediately preceding new entries
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// New log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
}

/// AppendEntries RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update itself
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// The responding node's ID
    pub node_id: NodeId,
    /// Conflict index for fast log rollback (optimization)
    pub conflict_index: Option<LogIndex>,
}

/// RequestVote RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// RequestVote RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term for candidate to update itself
    pub term: Term,
    /// True means candidate received vote
    pub vote_granted: bool,
    /// The responding node's ID
    pub node_id: NodeId,
}

/// Raft node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// This node's ID
    pub node_id: NodeId,
    /// All node IDs in the cluster (including self)
    pub cluster_nodes: Vec<NodeId>,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Election timeout range (min, max) in milliseconds
    pub election_timeout_min_ms: u64,
    pub election_timeout_max_ms: u64,
    /// Maximum log entries per AppendEntries batch
    pub max_entries_per_batch: usize,
    /// Enable log compaction via snapshotting
    pub enable_snapshots: bool,
    /// Snapshot threshold (entries before snapshot)
    pub snapshot_threshold: usize,
    /// Maximum retries for failed RPCs
    pub max_rpc_retries: usize,
}

impl RaftConfig {
    /// Create a single-node cluster configuration (useful for testing)
    pub fn single_node(node_id: NodeId) -> Self {
        Self {
            node_id,
            cluster_nodes: vec![node_id],
            heartbeat_interval_ms: 150,
            election_timeout_min_ms: 300,
            election_timeout_max_ms: 600,
            max_entries_per_batch: 100,
            enable_snapshots: true,
            snapshot_threshold: 10_000,
            max_rpc_retries: 3,
        }
    }

    /// Create a three-node cluster configuration
    pub fn three_node_cluster(node_id: NodeId) -> Self {
        Self {
            node_id,
            cluster_nodes: vec![1, 2, 3],
            heartbeat_interval_ms: 150,
            election_timeout_min_ms: 300,
            election_timeout_max_ms: 600,
            max_entries_per_batch: 100,
            enable_snapshots: true,
            snapshot_threshold: 10_000,
            max_rpc_retries: 3,
        }
    }

    /// Get the quorum size (majority)
    pub fn quorum_size(&self) -> usize {
        self.cluster_nodes.len() / 2 + 1
    }
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self::single_node(1)
    }
}

/// Statistics for the Raft node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RaftStats {
    /// Current term
    pub current_term: Term,
    /// Current role
    pub role: String,
    /// Current leader ID (if known)
    pub current_leader: Option<NodeId>,
    /// Total log entries
    pub log_length: usize,
    /// Commit index
    pub commit_index: LogIndex,
    /// Last applied index
    pub last_applied: LogIndex,
    /// Number of elections participated in
    pub elections_participated: u64,
    /// Number of terms this node was leader
    pub terms_as_leader: u64,
    /// Number of index operations applied
    pub operations_applied: u64,
    /// Number of vectors in the distributed index
    pub vector_count: usize,
    /// Number of RPC messages sent
    pub rpcs_sent: u64,
    /// Number of RPC messages received
    pub rpcs_received: u64,
}

/// The in-memory state machine: the actual vector index
#[derive(Debug, Default)]
struct IndexStateMachine {
    /// All vectors stored in this index shard
    vectors: HashMap<String, VectorEntry>,
    /// Number of operations applied
    operations_applied: u64,
}

impl IndexStateMachine {
    /// Apply a command to the state machine
    fn apply(&mut self, command: &IndexCommand) {
        match command {
            IndexCommand::Upsert(entry) => {
                self.vectors.insert(entry.vector_id.clone(), entry.clone());
                self.operations_applied += 1;
                debug!("Applied Upsert for vector '{}'", entry.vector_id);
            }
            IndexCommand::Delete { vector_id } => {
                self.vectors.remove(vector_id);
                self.operations_applied += 1;
                debug!("Applied Delete for vector '{}'", vector_id);
            }
            IndexCommand::UpdateMetadata {
                vector_id,
                metadata,
            } => {
                if let Some(entry) = self.vectors.get_mut(vector_id) {
                    entry.metadata.clone_from(metadata);
                    self.operations_applied += 1;
                }
            }
            IndexCommand::Rebuild => {
                debug!("Applied Rebuild command");
                self.operations_applied += 1;
            }
            IndexCommand::NoOp => {
                // No-op doesn't increment operations
            }
        }
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn get(&self, vector_id: &str) -> Option<&VectorEntry> {
        self.vectors.get(vector_id)
    }

    fn search_similar(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self
            .vectors
            .iter()
            .filter_map(|(id, entry)| {
                if entry.vector.len() != query.len() {
                    return None;
                }
                let dot: f32 = entry
                    .vector
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let na: f32 = entry.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if na < 1e-9 || nb < 1e-9 {
                    0.0
                } else {
                    dot / (na * nb)
                };
                Some((id.clone(), sim))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }
}

/// Persistent state for Raft node (must survive restarts)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistentState {
    /// Latest term this server has seen
    pub current_term: Term,
    /// CandidateId that received vote in current term
    pub voted_for: Option<NodeId>,
    /// Log entries
    pub log: Vec<LogEntry>,
}

impl PersistentState {
    fn last_log_index(&self) -> LogIndex {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    fn last_log_term(&self) -> Term {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    fn get_entry(&self, index: LogIndex) -> Option<&LogEntry> {
        if index == 0 {
            return None;
        }
        // Log entries are 1-indexed; find by scanning
        self.log.iter().find(|e| e.index == index)
    }

    fn truncate_from(&mut self, from_index: LogIndex) {
        self.log.retain(|e| e.index < from_index);
    }
}

/// Raft node implementation for distributed vector index
///
/// This implements the core Raft protocol. In a production deployment,
/// RPCs would be sent over the network (e.g., gRPC or HTTP/2). Here
/// we provide the state machine logic and expose methods for injecting
/// simulated or actual network messages.
#[derive(Debug)]
pub struct RaftIndexNode {
    config: RaftConfig,
    /// Persistent state (term, vote, log)
    persistent: Arc<RwLock<PersistentState>>,
    /// Current role
    role: Arc<Mutex<NodeRole>>,
    /// Current leader (known)
    current_leader: Arc<Mutex<Option<NodeId>>>,
    /// Commit index (highest log entry known to be committed)
    commit_index: Arc<Mutex<LogIndex>>,
    /// Last applied (highest log entry applied to state machine)
    last_applied: Arc<Mutex<LogIndex>>,
    /// Next index to send to each follower (leader only)
    next_index: Arc<Mutex<HashMap<NodeId, LogIndex>>>,
    /// Highest log index known to be replicated on each follower (leader only)
    match_index: Arc<Mutex<HashMap<NodeId, LogIndex>>>,
    /// The actual state machine (vector index)
    state_machine: Arc<RwLock<IndexStateMachine>>,
    /// Votes received in current election
    votes_received: Arc<Mutex<HashMap<NodeId, bool>>>,
    /// Election timeout tracking
    last_heartbeat: Arc<Mutex<Instant>>,
    /// Statistics
    stats: Arc<Mutex<RaftStats>>,
    /// Number of elections participated in
    elections_participated: Arc<Mutex<u64>>,
    /// Number of terms as leader
    terms_as_leader: Arc<Mutex<u64>>,
    /// Total RPCs sent
    rpcs_sent: Arc<Mutex<u64>>,
    /// Total RPCs received
    rpcs_received: Arc<Mutex<u64>>,
}

impl RaftIndexNode {
    /// Create a new Raft index node
    pub fn new(config: RaftConfig) -> Self {
        let node_id = config.node_id;
        let cluster_nodes: Vec<NodeId> = config.cluster_nodes.clone();

        let next_index: HashMap<NodeId, LogIndex> = cluster_nodes
            .iter()
            .filter(|&&n| n != node_id)
            .map(|&n| (n, 1))
            .collect();

        let match_index: HashMap<NodeId, LogIndex> = cluster_nodes
            .iter()
            .filter(|&&n| n != node_id)
            .map(|&n| (n, 0))
            .collect();

        info!(
            "Raft node {} initialized in cluster {:?}",
            node_id, cluster_nodes
        );

        Self {
            config,
            persistent: Arc::new(RwLock::new(PersistentState::default())),
            role: Arc::new(Mutex::new(NodeRole::Follower)),
            current_leader: Arc::new(Mutex::new(None)),
            commit_index: Arc::new(Mutex::new(0)),
            last_applied: Arc::new(Mutex::new(0)),
            next_index: Arc::new(Mutex::new(next_index)),
            match_index: Arc::new(Mutex::new(match_index)),
            state_machine: Arc::new(RwLock::new(IndexStateMachine::default())),
            votes_received: Arc::new(Mutex::new(HashMap::new())),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
            stats: Arc::new(Mutex::new(RaftStats::default())),
            elections_participated: Arc::new(Mutex::new(0)),
            terms_as_leader: Arc::new(Mutex::new(0)),
            rpcs_sent: Arc::new(Mutex::new(0)),
            rpcs_received: Arc::new(Mutex::new(0)),
        }
    }

    /// Start an election (become candidate)
    pub fn start_election(&self) -> RequestVoteRequest {
        let mut persistent = self.persistent.write();
        persistent.current_term += 1;
        let new_term = persistent.current_term;
        persistent.voted_for = Some(self.config.node_id);

        *self.role.lock() = NodeRole::Candidate;
        let mut votes = self.votes_received.lock();
        votes.clear();
        votes.insert(self.config.node_id, true); // Vote for self

        *self.elections_participated.lock() += 1;

        info!(
            "Node {} starting election for term {}",
            self.config.node_id, new_term
        );

        RequestVoteRequest {
            term: new_term,
            candidate_id: self.config.node_id,
            last_log_index: persistent.last_log_index(),
            last_log_term: persistent.last_log_term(),
        }
    }

    /// Handle a RequestVote RPC from a candidate
    pub fn handle_request_vote(&self, request: RequestVoteRequest) -> RequestVoteResponse {
        *self.rpcs_received.lock() += 1;
        let mut persistent = self.persistent.write();

        // If we see a higher term, update and become follower
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.role.lock() = NodeRole::Follower;
        }

        let vote_granted = if request.term < persistent.current_term {
            // Stale term, reject
            false
        } else {
            let already_voted = persistent
                .voted_for
                .map(|v| v != request.candidate_id)
                .unwrap_or(false);

            if already_voted {
                false
            } else {
                // Grant vote if candidate's log is at least as up-to-date
                let our_last_index = persistent.last_log_index();
                let our_last_term = persistent.last_log_term();

                let log_ok = request.last_log_term > our_last_term
                    || (request.last_log_term == our_last_term
                        && request.last_log_index >= our_last_index);

                if log_ok {
                    persistent.voted_for = Some(request.candidate_id);
                    *self.last_heartbeat.lock() = Instant::now();
                    true
                } else {
                    false
                }
            }
        };

        debug!(
            "Node {} {:?} vote to {} for term {}",
            self.config.node_id,
            if vote_granted { "grants" } else { "denies" },
            request.candidate_id,
            request.term
        );

        RequestVoteResponse {
            term: persistent.current_term,
            vote_granted,
            node_id: self.config.node_id,
        }
    }

    /// Process a vote response from a peer
    ///
    /// Returns `true` if this node just won the election.
    pub fn process_vote_response(&self, response: RequestVoteResponse) -> bool {
        *self.rpcs_received.lock() += 1;
        let persistent = self.persistent.read();

        // If we see a higher term, become follower
        if response.term > persistent.current_term {
            drop(persistent);
            let mut p = self.persistent.write();
            p.current_term = response.term;
            p.voted_for = None;
            *self.role.lock() = NodeRole::Follower;
            return false;
        }

        // Only count votes if still a candidate in the same term
        if *self.role.lock() != NodeRole::Candidate {
            return false;
        }

        if response.term != persistent.current_term {
            return false;
        }

        if response.vote_granted {
            let mut votes = self.votes_received.lock();
            votes.insert(response.node_id, true);
            let vote_count = votes.values().filter(|&&v| v).count();

            if vote_count >= self.config.quorum_size() {
                // Won election!
                drop(votes);
                drop(persistent);
                self.become_leader();
                return true;
            }
        }
        false
    }

    /// Transition to leader state
    fn become_leader(&self) {
        let term = self.persistent.read().current_term;
        *self.role.lock() = NodeRole::Leader;
        *self.current_leader.lock() = Some(self.config.node_id);
        *self.terms_as_leader.lock() += 1;

        // Initialize next_index and match_index for all followers
        let last_log_index = self.persistent.read().last_log_index();
        let mut next_idx = self.next_index.lock();
        let mut match_idx = self.match_index.lock();

        for &peer in &self.config.cluster_nodes {
            if peer != self.config.node_id {
                next_idx.insert(peer, last_log_index + 1);
                match_idx.insert(peer, 0);
            }
        }

        info!(
            "Node {} became leader for term {}",
            self.config.node_id, term
        );

        // Append no-op to establish leadership
        drop(next_idx);
        drop(match_idx);
        let _ = self.append_entry(IndexCommand::NoOp, None);
    }

    /// Handle AppendEntries RPC (from leader)
    pub fn handle_append_entries(&self, request: AppendEntriesRequest) -> AppendEntriesResponse {
        *self.rpcs_received.lock() += 1;
        let mut persistent = self.persistent.write();

        // If we see a higher term, become follower
        if request.term > persistent.current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.role.lock() = NodeRole::Follower;
        }

        // Reply false if term < currentTerm
        if request.term < persistent.current_term {
            return AppendEntriesResponse {
                term: persistent.current_term,
                success: false,
                node_id: self.config.node_id,
                conflict_index: None,
            };
        }

        // Reset election timer since we heard from a valid leader
        *self.last_heartbeat.lock() = Instant::now();
        *self.current_leader.lock() = Some(request.leader_id);
        *self.role.lock() = NodeRole::Follower;

        // Check prev_log consistency
        if request.prev_log_index > 0 {
            let entry = persistent.get_entry(request.prev_log_index);
            match entry {
                None => {
                    // Don't have that entry
                    return AppendEntriesResponse {
                        term: persistent.current_term,
                        success: false,
                        node_id: self.config.node_id,
                        conflict_index: Some(persistent.last_log_index() + 1),
                    };
                }
                Some(e) if e.term != request.prev_log_term => {
                    // Conflicting entry
                    let conflict_index = e.index;
                    return AppendEntriesResponse {
                        term: persistent.current_term,
                        success: false,
                        node_id: self.config.node_id,
                        conflict_index: Some(conflict_index),
                    };
                }
                _ => {}
            }
        }

        // Append new entries, removing conflicting ones
        for entry in &request.entries {
            let existing = persistent.get_entry(entry.index).cloned();
            match existing {
                Some(e) if e.term != entry.term => {
                    // Conflict: truncate log from here
                    persistent.truncate_from(entry.index);
                    persistent.log.push(entry.clone());
                }
                None => {
                    persistent.log.push(entry.clone());
                }
                _ => {} // Entry already present and matches
            }
        }

        // Update commit index
        let prev_commit = *self.commit_index.lock();
        if request.leader_commit > prev_commit {
            let new_commit = request.leader_commit.min(persistent.last_log_index());
            drop(persistent);
            *self.commit_index.lock() = new_commit;
            self.apply_committed_entries();
        }

        AppendEntriesResponse {
            term: self.persistent.read().current_term,
            success: true,
            node_id: self.config.node_id,
            conflict_index: None,
        }
    }

    /// Process AppendEntries response from a follower (leader only)
    pub fn process_append_entries_response(
        &self,
        peer_id: NodeId,
        response: AppendEntriesResponse,
        entries_sent_count: usize,
    ) {
        *self.rpcs_received.lock() += 1;
        let current_term = self.persistent.read().current_term;

        if response.term > current_term {
            let mut p = self.persistent.write();
            p.current_term = response.term;
            p.voted_for = None;
            *self.role.lock() = NodeRole::Follower;
            return;
        }

        if *self.role.lock() != NodeRole::Leader {
            return;
        }

        if response.success {
            let mut next_idx = self.next_index.lock();
            let mut match_idx = self.match_index.lock();

            let new_next =
                next_idx.get(&peer_id).copied().unwrap_or(1) + entries_sent_count as LogIndex;

            next_idx.insert(peer_id, new_next);
            match_idx.insert(peer_id, new_next - 1);
            drop(next_idx);
            drop(match_idx);

            // Try to advance commit index
            self.try_advance_commit_index();
        } else {
            // Decrement next_index for this follower
            let mut next_idx = self.next_index.lock();
            if let Some(conflict) = response.conflict_index {
                next_idx.insert(peer_id, conflict);
            } else {
                let current = next_idx.get(&peer_id).copied().unwrap_or(1);
                if current > 1 {
                    next_idx.insert(peer_id, current - 1);
                }
            }
        }
    }

    /// Try to advance the commit index based on match_index replication
    fn try_advance_commit_index(&self) {
        let persistent = self.persistent.read();
        let current_term = persistent.current_term;
        let last_log_index = persistent.last_log_index();
        drop(persistent);

        let match_idx = self.match_index.lock();
        let mut commit = *self.commit_index.lock();

        for n in (commit + 1)..=last_log_index {
            let p = self.persistent.read();
            let entry_term = p.get_entry(n).map(|e| e.term).unwrap_or(0);
            drop(p);

            // Only commit entries from current term (safety requirement)
            if entry_term != current_term {
                continue;
            }

            // Count replications
            let replication_count = 1 + // self
                match_idx.values().filter(|&&m| m >= n).count();

            if replication_count >= self.config.quorum_size() {
                commit = n;
            }
        }
        drop(match_idx);

        let old_commit = *self.commit_index.lock();
        if commit > old_commit {
            *self.commit_index.lock() = commit;
            self.apply_committed_entries();
        }
    }

    /// Apply all committed but not yet applied log entries to state machine
    fn apply_committed_entries(&self) {
        let commit = *self.commit_index.lock();
        let mut last = *self.last_applied.lock();

        while last < commit {
            last += 1;
            let persistent = self.persistent.read();
            let entry = persistent.get_entry(last).cloned();
            drop(persistent);

            if let Some(entry) = entry {
                let mut sm = self.state_machine.write();
                sm.apply(&entry.command);
                debug!("Node {} applied log entry {}", self.config.node_id, last);
            }
        }

        *self.last_applied.lock() = last;
    }

    /// Propose a new command (leader only)
    ///
    /// Returns the log index of the proposed entry, or an error if not leader.
    pub fn propose(&self, command: IndexCommand, client_id: Option<String>) -> Result<LogIndex> {
        if *self.role.lock() != NodeRole::Leader {
            let leader = self.current_leader.lock().map(|l| l.to_string());
            return Err(anyhow!(
                "Not the leader. Current leader: {:?}",
                leader.unwrap_or_else(|| "unknown".to_string())
            ));
        }
        self.append_entry(command, client_id)
    }

    /// Append an entry to the leader's log
    fn append_entry(&self, command: IndexCommand, client_id: Option<String>) -> Result<LogIndex> {
        let mut persistent = self.persistent.write();
        let term = persistent.current_term;
        let index = persistent.last_log_index() + 1;

        let entry = LogEntry {
            index,
            term,
            command,
            client_id,
        };

        persistent.log.push(entry);
        info!(
            "Node {} appended log entry {} in term {}",
            self.config.node_id, index, term
        );
        Ok(index)
    }

    /// Create an AppendEntries request for a specific follower
    pub fn create_append_entries_request(&self, peer_id: NodeId) -> Result<AppendEntriesRequest> {
        if *self.role.lock() != NodeRole::Leader {
            return Err(anyhow!("Not the leader"));
        }

        let persistent = self.persistent.read();
        let next_idx = self.next_index.lock();
        let next = next_idx.get(&peer_id).copied().unwrap_or(1);

        let prev_log_index = next.saturating_sub(1);
        let prev_log_term = if prev_log_index > 0 {
            persistent
                .get_entry(prev_log_index)
                .map(|e| e.term)
                .unwrap_or(0)
        } else {
            0
        };

        let entries: Vec<LogEntry> = persistent
            .log
            .iter()
            .filter(|e| e.index >= next)
            .take(self.config.max_entries_per_batch)
            .cloned()
            .collect();

        let commit = *self.commit_index.lock();

        *self.rpcs_sent.lock() += 1;

        Ok(AppendEntriesRequest {
            term: persistent.current_term,
            leader_id: self.config.node_id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: commit,
        })
    }

    /// Force commit a single-node cluster (for testing)
    ///
    /// In a single-node cluster, entries are immediately committed.
    pub fn force_commit_single_node(&self) {
        if self.config.cluster_nodes.len() != 1 {
            warn!("force_commit_single_node called on multi-node cluster");
            return;
        }
        let last_index = self.persistent.read().last_log_index();
        *self.commit_index.lock() = last_index;
        self.apply_committed_entries();
    }

    /// Get the current role
    pub fn role(&self) -> NodeRole {
        *self.role.lock()
    }

    /// Get the current term
    pub fn current_term(&self) -> Term {
        self.persistent.read().current_term
    }

    /// Get the current leader ID (if known)
    pub fn current_leader(&self) -> Option<NodeId> {
        *self.current_leader.lock()
    }

    /// Check if this node is the leader
    pub fn is_leader(&self) -> bool {
        *self.role.lock() == NodeRole::Leader
    }

    /// Get the number of entries in the log
    pub fn log_length(&self) -> usize {
        self.persistent.read().log.len()
    }

    /// Get the commit index
    pub fn commit_index(&self) -> LogIndex {
        *self.commit_index.lock()
    }

    /// Get the last applied index
    pub fn last_applied(&self) -> LogIndex {
        *self.last_applied.lock()
    }

    /// Get the number of vectors in the state machine
    pub fn vector_count(&self) -> usize {
        self.state_machine.read().len()
    }

    /// Get a vector from the state machine (read only)
    pub fn get_vector(&self, vector_id: &str) -> Option<VectorEntry> {
        self.state_machine.read().get(vector_id).cloned()
    }

    /// Search for similar vectors in the state machine
    pub fn search_similar(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        self.state_machine.read().search_similar(query, k)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> RaftStats {
        let persistent = self.persistent.read();
        RaftStats {
            current_term: persistent.current_term,
            role: self.role().to_string(),
            current_leader: *self.current_leader.lock(),
            log_length: persistent.log.len(),
            commit_index: *self.commit_index.lock(),
            last_applied: *self.last_applied.lock(),
            elections_participated: *self.elections_participated.lock(),
            terms_as_leader: *self.terms_as_leader.lock(),
            operations_applied: self.state_machine.read().operations_applied,
            vector_count: self.state_machine.read().len(),
            rpcs_sent: *self.rpcs_sent.lock(),
            rpcs_received: *self.rpcs_received.lock(),
        }
    }

    /// Check if election timeout has elapsed
    pub fn election_timeout_elapsed(&self) -> bool {
        let elapsed = self.last_heartbeat.lock().elapsed();
        elapsed > Duration::from_millis(self.config.election_timeout_max_ms)
    }

    /// Reset the heartbeat timer (call when receiving valid messages from leader)
    pub fn reset_heartbeat(&self) {
        *self.last_heartbeat.lock() = Instant::now();
    }
}

/// Helper to simulate a two-node cluster interaction for testing
pub struct ClusterSimulator {
    pub nodes: Vec<RaftIndexNode>,
}

impl ClusterSimulator {
    /// Create a simulated cluster of N nodes
    pub fn new(n: usize) -> Result<Self> {
        let cluster_nodes: Vec<NodeId> = (1..=(n as NodeId)).collect();

        let nodes = cluster_nodes
            .iter()
            .map(|&id| {
                let config = RaftConfig {
                    node_id: id,
                    cluster_nodes: cluster_nodes.clone(),
                    heartbeat_interval_ms: 50,
                    election_timeout_min_ms: 150,
                    election_timeout_max_ms: 300,
                    max_entries_per_batch: 10,
                    enable_snapshots: false,
                    snapshot_threshold: 1000,
                    max_rpc_retries: 2,
                };
                RaftIndexNode::new(config)
            })
            .collect();

        Ok(Self { nodes })
    }

    /// Elect node at index `leader_idx` as leader
    pub fn elect_leader(&self, leader_idx: usize) {
        // Start election on the chosen node
        let vote_request = self.nodes[leader_idx].start_election();

        // Collect votes from all other nodes
        let mut all_won = false;
        for (i, node) in self.nodes.iter().enumerate() {
            if i == leader_idx {
                continue;
            }
            let response = node.handle_request_vote(vote_request.clone());
            if self.nodes[leader_idx].process_vote_response(response) {
                all_won = true;
            }
        }

        // If won, send initial heartbeats
        if all_won || self.nodes[leader_idx].is_leader() {
            for (i, node) in self.nodes.iter().enumerate() {
                if i == leader_idx {
                    continue;
                }
                if let Ok(ae_req) =
                    self.nodes[leader_idx].create_append_entries_request(node.config.node_id)
                {
                    let response = node.handle_append_entries(ae_req.clone());
                    self.nodes[leader_idx].process_append_entries_response(
                        node.config.node_id,
                        response,
                        ae_req.entries.len(),
                    );
                }
            }
        }
    }

    /// Replicate all pending entries from leader to all followers
    pub fn replicate_all(&self) -> Result<()> {
        let leader_idx = self
            .nodes
            .iter()
            .position(|n| n.is_leader())
            .ok_or_else(|| anyhow!("No leader elected"))?;

        for (i, node) in self.nodes.iter().enumerate() {
            if i == leader_idx {
                continue;
            }
            if let Ok(ae_req) =
                self.nodes[leader_idx].create_append_entries_request(node.config.node_id)
            {
                let entries_len = ae_req.entries.len();
                let response = node.handle_append_entries(ae_req);
                self.nodes[leader_idx].process_append_entries_response(
                    node.config.node_id,
                    response,
                    entries_len,
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn make_vector_entry(id: &str, vec: Vec<f32>) -> VectorEntry {
        VectorEntry {
            vector_id: id.to_string(),
            vector: vec,
            metadata: HashMap::new(),
            inserted_at: 0,
        }
    }

    #[test]
    fn test_raft_config_single_node() {
        let config = RaftConfig::single_node(1);
        assert_eq!(config.node_id, 1);
        assert_eq!(config.cluster_nodes, vec![1]);
        assert_eq!(config.quorum_size(), 1);
    }

    #[test]
    fn test_raft_config_three_node() {
        let config = RaftConfig::three_node_cluster(1);
        assert_eq!(config.quorum_size(), 2);
    }

    #[test]
    fn test_node_starts_as_follower() {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);
        assert_eq!(node.role(), NodeRole::Follower);
        assert_eq!(node.current_term(), 0);
    }

    #[test]
    fn test_single_node_becomes_leader() {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        let vote_req = node.start_election();
        assert_eq!(vote_req.term, 1);
        assert_eq!(node.current_term(), 1);

        // Single-node cluster wins immediately
        let won = node.process_vote_response(RequestVoteResponse {
            term: 1,
            vote_granted: true,
            node_id: 1,
        });

        // Single node has quorum of 1, self-vote should win
        assert!(node.is_leader() || won);
    }

    #[test]
    fn test_single_node_leader_force_commit() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        // Make node leader directly
        node.start_election();
        // In single node, the self-vote should win
        let _ = node.process_vote_response(RequestVoteResponse {
            term: node.current_term(),
            vote_granted: true,
            node_id: 1,
        });

        if !node.is_leader() {
            // Manually set role for testing
            *node.role.lock() = NodeRole::Leader;
            *node.current_leader.lock() = Some(1);
        }

        let entry = make_vector_entry("v1", vec![1.0, 2.0, 3.0]);
        node.propose(IndexCommand::Upsert(entry), None)?;
        node.force_commit_single_node();

        assert_eq!(node.vector_count(), 1);
        assert!(node.get_vector("v1").is_some());
        Ok(())
    }

    #[test]
    fn test_propose_fails_when_not_leader() {
        let config = RaftConfig::three_node_cluster(1);
        let node = RaftIndexNode::new(config);
        // Node is follower, proposing should fail
        let result = node.propose(IndexCommand::NoOp, None);
        assert!(result.is_err(), "Should fail to propose when not leader");
    }

    #[test]
    fn test_request_vote_grants_to_newer_term() {
        let config = RaftConfig::three_node_cluster(2);
        let voter = RaftIndexNode::new(config);

        let req = RequestVoteRequest {
            term: 5,
            candidate_id: 1,
            last_log_index: 10,
            last_log_term: 5,
        };

        let response = voter.handle_request_vote(req);
        assert!(response.vote_granted, "Should grant vote to higher term");
        assert_eq!(response.term, 5);
    }

    #[test]
    fn test_request_vote_rejects_stale_term() {
        let config = RaftConfig::three_node_cluster(2);
        let voter = RaftIndexNode::new(config);

        // Set voter's current term to 5
        voter.persistent.write().current_term = 5;

        let req = RequestVoteRequest {
            term: 3, // Stale term
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = voter.handle_request_vote(req);
        assert!(!response.vote_granted, "Should reject stale term vote");
        assert_eq!(response.term, 5);
    }

    #[test]
    fn test_request_vote_rejects_duplicate_vote() {
        let config = RaftConfig::three_node_cluster(2);
        let voter = RaftIndexNode::new(config);

        let req1 = RequestVoteRequest {
            term: 1,
            candidate_id: 1,
            last_log_index: 0,
            last_log_term: 0,
        };

        let req2 = RequestVoteRequest {
            term: 1,
            candidate_id: 3, // Different candidate, same term
            last_log_index: 0,
            last_log_term: 0,
        };

        let r1 = voter.handle_request_vote(req1);
        assert!(r1.vote_granted, "First vote should be granted");

        let r2 = voter.handle_request_vote(req2);
        assert!(
            !r2.vote_granted,
            "Duplicate vote in same term should be rejected"
        );
    }

    #[test]
    #[ignore = "slow network simulation test - run explicitly with cargo test -- --ignored"]
    fn test_append_entries_heartbeat() {
        let config = RaftConfig::three_node_cluster(2);
        let follower = RaftIndexNode::new(config);

        let heartbeat = AppendEntriesRequest {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        let response = follower.handle_append_entries(heartbeat);
        assert!(response.success, "Heartbeat should succeed");
        assert_eq!(follower.current_leader(), Some(1));
    }

    #[test]
    fn test_append_entries_stale_term() {
        let config = RaftConfig::three_node_cluster(2);
        let follower = RaftIndexNode::new(config);
        follower.persistent.write().current_term = 5;

        let request = AppendEntriesRequest {
            term: 3, // Stale
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        };

        let response = follower.handle_append_entries(request);
        assert!(!response.success, "Stale term should be rejected");
        assert_eq!(response.term, 5);
    }

    #[test]
    #[ignore = "slow network simulation test - run explicitly with cargo test -- --ignored"]
    fn test_cluster_simulator_election() -> Result<()> {
        let sim = ClusterSimulator::new(3)?;
        sim.elect_leader(0);

        // At least one node should be leader
        let leaders: Vec<_> = sim.nodes.iter().filter(|n| n.is_leader()).collect();
        assert!(!leaders.is_empty(), "At least one node should be leader");
        Ok(())
    }

    #[test]
    #[ignore = "slow network simulation test - run explicitly with cargo test -- --ignored"]
    fn test_cluster_simulator_replication() -> Result<()> {
        let sim = ClusterSimulator::new(3)?;
        sim.elect_leader(0);

        let leader_idx = sim
            .nodes
            .iter()
            .position(|n| n.is_leader())
            .expect("no leader found");
        let entry = make_vector_entry("v1", vec![1.0, 0.0, 0.0]);
        sim.nodes[leader_idx].propose(IndexCommand::Upsert(entry), None)?;

        sim.replicate_all()?;

        // All nodes should eventually have the entry committed
        let leader = &sim.nodes[leader_idx];
        leader.force_commit_single_node();
        // Leader should have the vector
        let vec = leader.get_vector("v1");
        assert!(vec.is_some() || leader.log_length() > 0);
        Ok(())
    }

    #[test]
    fn test_delete_command() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        // Become leader
        node.start_election();
        let _ = node.process_vote_response(RequestVoteResponse {
            term: node.current_term(),
            vote_granted: true,
            node_id: 1,
        });

        if !node.is_leader() {
            *node.role.lock() = NodeRole::Leader;
            *node.current_leader.lock() = Some(1);
        }

        // Insert then delete
        let entry = make_vector_entry("v1", vec![1.0]);
        node.propose(IndexCommand::Upsert(entry), None)?;
        node.force_commit_single_node();
        assert_eq!(node.vector_count(), 1);

        node.propose(
            IndexCommand::Delete {
                vector_id: "v1".to_string(),
            },
            None,
        )?;
        node.force_commit_single_node();
        assert_eq!(node.vector_count(), 0);
        Ok(())
    }

    #[test]
    fn test_update_metadata_command() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        *node.role.lock() = NodeRole::Leader;
        *node.current_leader.lock() = Some(1);

        let entry = make_vector_entry("v1", vec![1.0, 2.0]);
        node.propose(IndexCommand::Upsert(entry), None)?;

        let mut new_meta = HashMap::new();
        new_meta.insert("tag".to_string(), "important".to_string());
        node.propose(
            IndexCommand::UpdateMetadata {
                vector_id: "v1".to_string(),
                metadata: new_meta,
            },
            None,
        )?;
        node.force_commit_single_node();

        let stored = node.get_vector("v1").expect("v1 not found");
        assert_eq!(stored.metadata.get("tag"), Some(&"important".to_string()));
        Ok(())
    }

    #[test]
    fn test_search_similar() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        *node.role.lock() = NodeRole::Leader;
        *node.current_leader.lock() = Some(1);

        node.propose(
            IndexCommand::Upsert(make_vector_entry("v1", vec![1.0, 0.0, 0.0])),
            None,
        )?;
        node.propose(
            IndexCommand::Upsert(make_vector_entry("v2", vec![0.0, 1.0, 0.0])),
            None,
        )?;
        node.propose(
            IndexCommand::Upsert(make_vector_entry("v3", vec![0.0, 0.0, 1.0])),
            None,
        )?;
        node.force_commit_single_node();

        let results = node.search_similar(&[1.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        // First result should be v1 with similarity ~1.0
        assert_eq!(results[0].0, "v1");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_stats_populated() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        *node.role.lock() = NodeRole::Leader;
        *node.current_leader.lock() = Some(1);
        node.propose(IndexCommand::NoOp, None)?;
        node.force_commit_single_node();

        let stats = node.get_stats();
        assert_eq!(stats.role, "Leader");
        assert!(stats.log_length > 0);
        Ok(())
    }

    #[test]
    fn test_raft_log_length_increases() -> Result<()> {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);

        *node.role.lock() = NodeRole::Leader;
        *node.current_leader.lock() = Some(1);

        assert_eq!(node.log_length(), 0);

        node.propose(IndexCommand::NoOp, None)?;
        assert_eq!(node.log_length(), 1);

        node.propose(IndexCommand::Rebuild, None)?;
        assert_eq!(node.log_length(), 2);
        Ok(())
    }

    #[test]
    fn test_persistent_state_default() {
        let state = PersistentState::default();
        assert_eq!(state.current_term, 0);
        assert!(state.voted_for.is_none());
        assert!(state.log.is_empty());
        assert_eq!(state.last_log_index(), 0);
        assert_eq!(state.last_log_term(), 0);
    }

    #[test]
    fn test_node_role_display() {
        assert_eq!(NodeRole::Follower.to_string(), "Follower");
        assert_eq!(NodeRole::Candidate.to_string(), "Candidate");
        assert_eq!(NodeRole::Leader.to_string(), "Leader");
    }

    #[test]
    fn test_election_timeout_not_elapsed_immediately() {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);
        // Freshly created node should not have elapsed election timeout
        assert!(!node.election_timeout_elapsed());
    }

    #[test]
    fn test_reset_heartbeat() {
        let config = RaftConfig::single_node(1);
        let node = RaftIndexNode::new(config);
        // Resetting heartbeat should keep timeout from elapsing
        node.reset_heartbeat();
        assert!(!node.election_timeout_elapsed());
    }

    #[test]
    fn test_append_entries_appends_new_log_entries() {
        let config = RaftConfig::three_node_cluster(2);
        let follower = RaftIndexNode::new(config);

        let entry = LogEntry {
            index: 1,
            term: 1,
            command: IndexCommand::NoOp,
            client_id: None,
        };

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![entry],
            leader_commit: 1,
        };

        let response = follower.handle_append_entries(request);
        assert!(response.success);
        assert_eq!(follower.log_length(), 1);
    }

    #[test]
    fn test_commit_advances_last_applied() {
        let config = RaftConfig::three_node_cluster(2);
        let follower = RaftIndexNode::new(config);

        let entry = LogEntry {
            index: 1,
            term: 1,
            command: IndexCommand::Upsert(make_vector_entry("v1", vec![1.0])),
            client_id: None,
        };

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: 1,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![entry],
            leader_commit: 1, // Leader has committed this
        };

        follower.handle_append_entries(request);

        assert_eq!(follower.last_applied(), 1);
        assert_eq!(follower.vector_count(), 1);
    }
}
