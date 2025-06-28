//! Raft consensus protocol implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, Mutex, RwLock},
    time::{interval, sleep},
};

use crate::{
    clustering::RaftConfig,
    error::{FusekiError, FusekiResult},
    store::Store,
};

/// Raft node states
#[derive(Debug, Clone, Copy, PartialEq)]
enum RaftState {
    Follower,
    Candidate,
    Leader,
}

/// Raft log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log index
    pub index: u64,
    /// Term when entry was created
    pub term: u64,
    /// Command data
    pub command: Command,
    /// Client request ID for deduplication
    pub client_id: Option<String>,
}

/// Commands that can be replicated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Store a key-value pair
    Set { key: String, value: Vec<u8> },
    /// Delete a key
    Delete { key: String },
    /// Configuration change
    ConfigChange { config: ClusterConfig },
    /// No-op for new leader establishment
    NoOp,
}

/// Cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Current members
    pub members: Vec<String>,
    /// New members (during reconfiguration)
    pub new_members: Option<Vec<String>>,
}

/// Persistent state (must be durable)
#[derive(Debug, Clone)]
struct PersistentState {
    /// Current term
    current_term: u64,
    /// Candidate that received vote in current term
    voted_for: Option<String>,
    /// Log entries
    log: Vec<LogEntry>,
}

/// Volatile state on all servers
#[derive(Debug, Clone)]
struct VolatileState {
    /// Index of highest log entry known to be committed
    commit_index: u64,
    /// Index of highest log entry applied to state machine
    last_applied: u64,
}

/// Volatile state on leaders
#[derive(Debug, Clone)]
struct LeaderState {
    /// Next log index to send to each server
    next_index: HashMap<String, u64>,
    /// Highest log index known to be replicated on each server
    match_index: HashMap<String, u64>,
}

/// RPC messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcMessage {
    AppendEntries(AppendEntriesRequest),
    AppendEntriesResponse(AppendEntriesResponse),
    RequestVote(RequestVoteRequest),
    RequestVoteResponse(RequestVoteResponse),
    InstallSnapshot(InstallSnapshotRequest),
    InstallSnapshotResponse(InstallSnapshotResponse),
}

/// AppendEntries RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: u64,
    /// Leader ID
    pub leader_id: String,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: u64,
    /// Term of prev_log_index entry
    pub prev_log_term: u64,
    /// Log entries to store
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: u64,
}

/// AppendEntries RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update itself
    pub term: u64,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// Follower's last log index (for fast backtracking)
    pub last_log_index: u64,
}

/// RequestVote RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term
    pub term: u64,
    /// Candidate requesting vote
    pub candidate_id: String,
    /// Index of candidate's last log entry
    pub last_log_index: u64,
    /// Term of candidate's last log entry
    pub last_log_term: u64,
}

/// RequestVote RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term for candidate to update itself
    pub term: u64,
    /// True means candidate received vote
    pub vote_granted: bool,
}

/// InstallSnapshot RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotRequest {
    /// Leader's term
    pub term: u64,
    /// Leader ID
    pub leader_id: String,
    /// Last included index
    pub last_included_index: u64,
    /// Last included term
    pub last_included_term: u64,
    /// Byte offset where chunk is positioned
    pub offset: u64,
    /// Raw bytes of snapshot chunk
    pub data: Vec<u8>,
    /// True if this is the last chunk
    pub done: bool,
}

/// InstallSnapshot RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotResponse {
    /// Current term for leader to update itself
    pub term: u64,
}

/// Raft consensus node
pub struct RaftNode {
    /// Node ID
    id: String,
    /// Configuration
    config: RaftConfig,
    /// Current state
    state: Arc<RwLock<RaftState>>,
    /// Persistent state
    persistent: Arc<RwLock<PersistentState>>,
    /// Volatile state
    volatile: Arc<RwLock<VolatileState>>,
    /// Leader state (when leader)
    leader_state: Arc<RwLock<Option<LeaderState>>>,
    /// Current leader
    current_leader: Arc<RwLock<Option<String>>>,
    /// Cluster configuration
    cluster_config: Arc<RwLock<ClusterConfig>>,
    /// RPC channel
    rpc_tx: mpsc::Sender<(String, RpcMessage)>,
    rpc_rx: Arc<Mutex<mpsc::Receiver<(String, RpcMessage)>>>,
    /// Election timer
    election_timer: Arc<RwLock<Instant>>,
    /// Storage backend
    store: Arc<Store>,
}

impl RaftNode {
    /// Create a new Raft node
    pub async fn new(id: String, config: RaftConfig, store: Arc<Store>) -> Result<Self> {
        let (rpc_tx, rpc_rx) = mpsc::channel(1000);

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(RaftState::Follower)),
            persistent: Arc::new(RwLock::new(PersistentState {
                current_term: 0,
                voted_for: None,
                log: vec![],
            })),
            volatile: Arc::new(RwLock::new(VolatileState {
                commit_index: 0,
                last_applied: 0,
            })),
            leader_state: Arc::new(RwLock::new(None)),
            current_leader: Arc::new(RwLock::new(None)),
            cluster_config: Arc::new(RwLock::new(ClusterConfig {
                members: vec![],
                new_members: None,
            })),
            rpc_tx,
            rpc_rx: Arc::new(Mutex::new(rpc_rx)),
            election_timer: Arc::new(RwLock::new(Instant::now())),
            store,
        })
    }

    /// Start the Raft node
    pub async fn start(&self) -> Result<()> {
        // Start RPC handler
        self.start_rpc_handler().await;

        // Start election timer
        self.start_election_timer().await;

        // Start heartbeat timer (for leaders)
        self.start_heartbeat_timer().await;

        // Start log applier
        self.start_log_applier().await;

        Ok(())
    }

    /// Bootstrap a new single-node cluster
    pub async fn bootstrap(&self) -> Result<()> {
        let mut config = self.cluster_config.write().await;
        config.members = vec![self.id.clone()];

        // Become leader immediately
        *self.state.write().await = RaftState::Leader;
        *self.current_leader.write().await = Some(self.id.clone());

        // Initialize leader state
        *self.leader_state.write().await = Some(LeaderState {
            next_index: HashMap::new(),
            match_index: HashMap::new(),
        });

        // Append no-op entry
        self.append_log_entry(Command::NoOp).await?;

        Ok(())
    }

    /// Start RPC handler
    async fn start_rpc_handler(&self) {
        let rpc_rx = self.rpc_rx.clone();
        let node = self.clone_refs();

        tokio::spawn(async move {
            let mut rx = rpc_rx.lock().await;
            while let Some((from, msg)) = rx.recv().await {
                match msg {
                    RpcMessage::AppendEntries(req) => {
                        let resp = node.handle_append_entries(req).await;
                        // Send response back
                        let _ = node
                            .send_rpc(&from, RpcMessage::AppendEntriesResponse(resp))
                            .await;
                    }
                    RpcMessage::RequestVote(req) => {
                        let resp = node.handle_request_vote(req).await;
                        // Send response back
                        let _ = node
                            .send_rpc(&from, RpcMessage::RequestVoteResponse(resp))
                            .await;
                    }
                    RpcMessage::InstallSnapshot(req) => {
                        let resp = node.handle_install_snapshot(req).await;
                        // Send response back
                        let _ = node
                            .send_rpc(&from, RpcMessage::InstallSnapshotResponse(resp))
                            .await;
                    }
                    _ => {}
                }
            }
        });
    }

    /// Start election timer
    async fn start_election_timer(&self) {
        let node = self.clone_refs();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(50));
            let mut rng = rand::thread_rng();

            loop {
                interval.tick().await;

                let state = *node.state.read().await;
                if state != RaftState::Leader {
                    let last_heartbeat = *node.election_timer.read().await;
                    let timeout =
                        rng.gen_range(config.election_timeout.0..config.election_timeout.1);

                    if last_heartbeat.elapsed() > timeout {
                        // Start election
                        node.start_election().await;
                    }
                }
            }
        });
    }

    /// Start heartbeat timer
    async fn start_heartbeat_timer(&self) {
        let node = self.clone_refs();
        let interval_duration = self.config.heartbeat_interval;

        tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                let state = *node.state.read().await;
                if state == RaftState::Leader {
                    node.send_heartbeats().await;
                }
            }
        });
    }

    /// Start log applier
    async fn start_log_applier(&self) {
        let node = self.clone_refs();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                let volatile = node.volatile.read().await;
                let last_applied = volatile.last_applied;
                let commit_index = volatile.commit_index;
                drop(volatile);

                if commit_index > last_applied {
                    node.apply_committed_entries(last_applied + 1, commit_index)
                        .await;
                }
            }
        });
    }

    /// Start election
    async fn start_election(&self) {
        tracing::info!("Node {} starting election", self.id);

        // Increment current term
        let mut persistent = self.persistent.write().await;
        persistent.current_term += 1;
        persistent.voted_for = Some(self.id.clone());
        let current_term = persistent.current_term;
        let last_log_index = persistent.log.len() as u64;
        let last_log_term = persistent.log.last().map(|e| e.term).unwrap_or(0);
        drop(persistent);

        // Transition to candidate
        *self.state.write().await = RaftState::Candidate;
        *self.election_timer.write().await = Instant::now();

        // Request votes from all other nodes
        let config = self.cluster_config.read().await;
        let mut votes = 1; // Vote for self
        let majority = (config.members.len() / 2) + 1;

        for member in &config.members {
            if member != &self.id {
                let req = RequestVoteRequest {
                    term: current_term,
                    candidate_id: self.id.clone(),
                    last_log_index,
                    last_log_term,
                };

                // Send vote request
                if let Ok(()) = self.send_rpc(member, RpcMessage::RequestVote(req)).await {
                    // In real implementation, would handle response asynchronously
                    votes += 1;
                }
            }
        }

        // Check if won election
        if votes >= majority {
            self.become_leader().await;
        }
    }

    /// Become leader
    async fn become_leader(&self) {
        tracing::info!("Node {} became leader", self.id);

        *self.state.write().await = RaftState::Leader;
        *self.current_leader.write().await = Some(self.id.clone());

        // Initialize leader state
        let config = self.cluster_config.read().await;
        let log_length = self.persistent.read().await.log.len() as u64;

        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for member in &config.members {
            if member != &self.id {
                next_index.insert(member.clone(), log_length + 1);
                match_index.insert(member.clone(), 0);
            }
        }

        *self.leader_state.write().await = Some(LeaderState {
            next_index,
            match_index,
        });

        // Append no-op entry
        self.append_log_entry(Command::NoOp).await.ok();
    }

    /// Send heartbeats to all followers
    async fn send_heartbeats(&self) {
        let leader_state = self.leader_state.read().await;
        if let Some(state) = leader_state.as_ref() {
            let config = self.cluster_config.read().await;
            let persistent = self.persistent.read().await;
            let volatile = self.volatile.read().await;

            for member in &config.members {
                if member != &self.id {
                    let next_idx = state.next_index.get(member).copied().unwrap_or(1);
                    let prev_idx = next_idx.saturating_sub(1);
                    let prev_term = if prev_idx > 0 {
                        persistent
                            .log
                            .get(prev_idx as usize - 1)
                            .map(|e| e.term)
                            .unwrap_or(0)
                    } else {
                        0
                    };

                    let req = AppendEntriesRequest {
                        term: persistent.current_term,
                        leader_id: self.id.clone(),
                        prev_log_index: prev_idx,
                        prev_log_term: prev_term,
                        entries: vec![],
                        leader_commit: volatile.commit_index,
                    };

                    // Send heartbeat
                    let _ = self.send_rpc(member, RpcMessage::AppendEntries(req)).await;
                }
            }
        }
    }

    /// Handle AppendEntries RPC
    async fn handle_append_entries(&self, req: AppendEntriesRequest) -> AppendEntriesResponse {
        let mut persistent = self.persistent.write().await;
        let current_term = persistent.current_term;

        // Reply false if term < currentTerm
        if req.term < current_term {
            return AppendEntriesResponse {
                term: current_term,
                success: false,
                last_log_index: persistent.log.len() as u64,
            };
        }

        // Update term if needed
        if req.term > current_term {
            persistent.current_term = req.term;
            persistent.voted_for = None;
        }

        // Reset election timer
        *self.election_timer.write().await = Instant::now();
        *self.state.write().await = RaftState::Follower;
        *self.current_leader.write().await = Some(req.leader_id.clone());

        // Check log consistency
        if req.prev_log_index > 0 {
            if let Some(entry) = persistent.log.get(req.prev_log_index as usize - 1) {
                if entry.term != req.prev_log_term {
                    return AppendEntriesResponse {
                        term: req.term,
                        success: false,
                        last_log_index: persistent.log.len() as u64,
                    };
                }
            } else {
                return AppendEntriesResponse {
                    term: req.term,
                    success: false,
                    last_log_index: persistent.log.len() as u64,
                };
            }
        }

        // Append new entries
        if !req.entries.is_empty() {
            persistent.log.truncate(req.prev_log_index as usize);
            persistent.log.extend(req.entries);
        }

        // Update commit index
        if req.leader_commit > self.volatile.read().await.commit_index {
            let mut volatile = self.volatile.write().await;
            volatile.commit_index = req.leader_commit.min(persistent.log.len() as u64);
        }

        AppendEntriesResponse {
            term: req.term,
            success: true,
            last_log_index: persistent.log.len() as u64,
        }
    }

    /// Handle RequestVote RPC
    async fn handle_request_vote(&self, req: RequestVoteRequest) -> RequestVoteResponse {
        let mut persistent = self.persistent.write().await;
        let current_term = persistent.current_term;

        // Reply false if term < currentTerm
        if req.term < current_term {
            return RequestVoteResponse {
                term: current_term,
                vote_granted: false,
            };
        }

        // Update term if needed
        if req.term > current_term {
            persistent.current_term = req.term;
            persistent.voted_for = None;
            *self.state.write().await = RaftState::Follower;
        }

        // Check if can grant vote
        let can_vote = persistent.voted_for.is_none()
            || persistent.voted_for.as_ref() == Some(&req.candidate_id);
        let log_ok = self.is_log_up_to_date(&persistent, req.last_log_index, req.last_log_term);

        let vote_granted = can_vote && log_ok;

        if vote_granted {
            persistent.voted_for = Some(req.candidate_id);
            *self.election_timer.write().await = Instant::now();
        }

        RequestVoteResponse {
            term: req.term,
            vote_granted,
        }
    }

    /// Handle InstallSnapshot RPC
    async fn handle_install_snapshot(
        &self,
        req: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        let current_term = self.persistent.read().await.current_term;

        // Reply immediately if term < currentTerm
        if req.term < current_term {
            return InstallSnapshotResponse { term: current_term };
        }

        // TODO: Implement snapshot installation

        InstallSnapshotResponse { term: req.term }
    }

    /// Check if candidate's log is at least as up-to-date as receiver's log
    fn is_log_up_to_date(
        &self,
        persistent: &PersistentState,
        last_log_index: u64,
        last_log_term: u64,
    ) -> bool {
        let my_last_index = persistent.log.len() as u64;
        let my_last_term = persistent.log.last().map(|e| e.term).unwrap_or(0);

        last_log_term > my_last_term
            || (last_log_term == my_last_term && last_log_index >= my_last_index)
    }

    /// Append a log entry
    async fn append_log_entry(&self, command: Command) -> Result<u64> {
        let mut persistent = self.persistent.write().await;
        let index = persistent.log.len() as u64 + 1;
        let term = persistent.current_term;

        persistent.log.push(LogEntry {
            index,
            term,
            command,
            client_id: None,
        });

        Ok(index)
    }

    /// Apply committed entries to state machine
    async fn apply_committed_entries(&self, start: u64, end: u64) {
        let persistent = self.persistent.read().await;

        for i in start..=end {
            if let Some(entry) = persistent.log.get(i as usize - 1) {
                // Apply to state machine
                match &entry.command {
                    Command::Set { key, value } => {
                        // TODO: Apply to store
                        tracing::debug!("Applied Set({}, ...)", key);
                    }
                    Command::Delete { key } => {
                        // TODO: Apply to store
                        tracing::debug!("Applied Delete({})", key);
                    }
                    Command::ConfigChange { config } => {
                        // TODO: Apply configuration change
                        tracing::debug!("Applied ConfigChange");
                    }
                    Command::NoOp => {
                        // No operation
                    }
                }
            }
        }

        let mut volatile = self.volatile.write().await;
        volatile.last_applied = end;
    }

    /// Send RPC to another node
    async fn send_rpc(&self, target: &str, message: RpcMessage) -> Result<()> {
        // TODO: Implement actual network RPC
        tracing::debug!("Sending {:?} to {}", message, target);
        Ok(())
    }

    /// Clone references for spawning tasks
    fn clone_refs(&self) -> RaftNodeRefs {
        RaftNodeRefs {
            id: self.id.clone(),
            config: self.config.clone(),
            state: self.state.clone(),
            persistent: self.persistent.clone(),
            volatile: self.volatile.clone(),
            leader_state: self.leader_state.clone(),
            current_leader: self.current_leader.clone(),
            cluster_config: self.cluster_config.clone(),
            election_timer: self.election_timer.clone(),
        }
    }
}

/// References to RaftNode fields for async tasks
struct RaftNodeRefs {
    id: String,
    config: RaftConfig,
    state: Arc<RwLock<RaftState>>,
    persistent: Arc<RwLock<PersistentState>>,
    volatile: Arc<RwLock<VolatileState>>,
    leader_state: Arc<RwLock<Option<LeaderState>>>,
    current_leader: Arc<RwLock<Option<String>>>,
    cluster_config: Arc<RwLock<ClusterConfig>>,
    election_timer: Arc<RwLock<Instant>>,
}

// Implement the same methods for RaftNodeRefs (simplified for the example)
impl RaftNodeRefs {
    async fn start_election(&self) {
        // Implementation would be similar to RaftNode::start_election
    }

    async fn become_leader(&self) {
        // Implementation would be similar to RaftNode::become_leader
    }

    async fn send_heartbeats(&self) {
        // Implementation would be similar to RaftNode::send_heartbeats
    }

    async fn handle_append_entries(&self, req: AppendEntriesRequest) -> AppendEntriesResponse {
        // Implementation would be similar to RaftNode::handle_append_entries
        AppendEntriesResponse {
            term: 0,
            success: false,
            last_log_index: 0,
        }
    }

    async fn handle_request_vote(&self, req: RequestVoteRequest) -> RequestVoteResponse {
        // Implementation would be similar to RaftNode::handle_request_vote
        RequestVoteResponse {
            term: 0,
            vote_granted: false,
        }
    }

    async fn handle_install_snapshot(
        &self,
        req: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        // Implementation would be similar to RaftNode::handle_install_snapshot
        InstallSnapshotResponse { term: 0 }
    }

    async fn apply_committed_entries(&self, start: u64, end: u64) {
        // Implementation would be similar to RaftNode::apply_committed_entries
    }

    async fn append_log_entry(&self, command: Command) -> Result<u64> {
        // Implementation would be similar to RaftNode::append_log_entry
        Ok(0)
    }

    async fn send_rpc(&self, target: &str, message: RpcMessage) -> Result<()> {
        // Implementation would be similar to RaftNode::send_rpc
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_serialization() {
        let entry = LogEntry {
            index: 1,
            term: 1,
            command: Command::Set {
                key: "test".to_string(),
                value: vec![1, 2, 3],
            },
            client_id: Some("client1".to_string()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let decoded: LogEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.index, entry.index);
        assert_eq!(decoded.term, entry.term);
    }

    #[test]
    fn test_raft_state_transitions() {
        assert_ne!(RaftState::Follower, RaftState::Candidate);
        assert_ne!(RaftState::Candidate, RaftState::Leader);
        assert_ne!(RaftState::Leader, RaftState::Follower);
    }
}
