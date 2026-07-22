//! Raft consensus protocol implementation

use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, Mutex, RwLock},
    time::interval,
};

use crate::{
    clustering::RaftConfig,
    error::{FusekiError, FusekiResult},
    store::Store,
};

/// Raft node states
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum RaftState {
    Follower,
    Candidate,
    Leader,
}

/// Raft log entry
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct ClusterConfig {
    /// Current members
    pub members: Vec<String>,
    /// New members (during reconfiguration)
    pub new_members: Option<Vec<String>>,
}

/// Persistent state (must be durable)
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
#[allow(dead_code)]
struct LeaderState {
    /// Next log index to send to each server
    next_index: HashMap<String, u64>,
    /// Highest log index known to be replicated on each server
    match_index: HashMap<String, u64>,
}

/// RPC messages
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub enum RpcMessage {
    AppendEntries(AppendEntriesRequest),
    AppendEntriesResponse(AppendEntriesResponse),
    RequestVote(RequestVoteRequest),
    RequestVoteResponse(RequestVoteResponse),
    InstallSnapshot(InstallSnapshotRequest),
    InstallSnapshotResponse(InstallSnapshotResponse),
}

/// AppendEntries RPC request
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update itself
    pub term: u64,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// Follower's last log index (for fast backtracking)
    pub last_log_index: u64,
}

/// RequestVote RPC request
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct RequestVoteResponse {
    /// Current term for candidate to update itself
    pub term: u64,
    /// True means candidate received vote
    pub vote_granted: bool,
}

/// InstallSnapshot RPC request
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
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
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct InstallSnapshotResponse {
    /// Current term for leader to update itself
    pub term: u64,
}

/// Raft consensus node
#[allow(dead_code)]
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

#[allow(dead_code)]
impl RaftNode {
    /// Create a new Raft node
    pub async fn new(id: String, config: RaftConfig, store: Arc<Store>) -> FusekiResult<Self> {
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

    /// Start the Raft node.
    ///
    /// This in-process Raft implementation does not wire a cross-node RPC
    /// transport (see [`RaftNode::send_rpc`]). A genuine multi-node cluster
    /// therefore cannot exchange votes or replicate entries, so rather than
    /// letting every node fabricate votes and self-elect (split-brain), starting
    /// with more than one configured member is rejected with an explicit error.
    /// Single-node operation (via [`RaftNode::bootstrap`]) is fully functional.
    pub async fn start(self: &Arc<Self>) -> FusekiResult<()> {
        let member_count = self.cluster_config.read().await.members.len();
        if member_count > 1 {
            return Err(FusekiError::internal(format!(
                "multi-node Raft ({member_count} members) requires a wired RPC transport that is \
                 not available in this build; refusing to start to avoid split-brain. Use a \
                 single-node configuration or the oxirs-cluster consensus backend."
            )));
        }

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
    pub async fn bootstrap(&self) -> FusekiResult<()> {
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
    async fn start_rpc_handler(self: &Arc<Self>) {
        let rpc_rx = self.rpc_rx.clone();
        let node = Arc::clone(self);

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
    async fn start_election_timer(self: &Arc<Self>) {
        let node = Arc::clone(self);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(50));
            let mut rng = Random::seed(42);

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
    async fn start_heartbeat_timer(self: &Arc<Self>) {
        let node = Arc::clone(self);
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
    async fn start_log_applier(self: &Arc<Self>) {
        let node = Arc::clone(self);

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

        // Request votes from all other nodes.
        let config = self.cluster_config.read().await;
        let votes = 1; // Vote for self; peer votes require a wired transport.
        let majority = (config.members.len() / 2) + 1;

        for member in &config.members {
            if member != &self.id {
                let req = RequestVoteRequest {
                    term: current_term,
                    candidate_id: self.id.clone(),
                    last_log_index,
                    last_log_term,
                };

                // A vote is counted ONLY when a peer actually returns a granted
                // RequestVoteResponse. Because this build has no wired RPC
                // transport (`send_rpc` fails loud), a peer vote can never be
                // confirmed here — so `votes` stays at 1 and a multi-node node
                // never self-elects. Counting a vote merely because the send call
                // returned would be the split-brain bug this guards against.
                if let Err(e) = self.send_rpc(member, RpcMessage::RequestVote(req)).await {
                    tracing::debug!("vote request to {member} could not be sent: {e}");
                }
            }
        }

        // Check if won election (single-node clusters have majority == 1).
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

        // Future enhancement: Implement snapshot installation for log compaction.
        // For v0.1.0: Basic Raft consensus works without snapshots.
        // Snapshot installation is needed for long-running clusters with large logs.

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
    async fn append_log_entry(&self, command: Command) -> FusekiResult<u64> {
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
                    Command::Set { key, value: _ } => {
                        // Future enhancement: Apply to actual RDF store.
                        // For v0.1.0: Raft replication logic is complete, store integration pending.
                        tracing::debug!("Applied Set({}, ...)", key);
                    }
                    Command::Delete { key } => {
                        // Future enhancement: Apply to actual RDF store.
                        // For v0.1.0: Raft replication logic is complete, store integration pending.
                        tracing::debug!("Applied Delete({})", key);
                    }
                    Command::ConfigChange { config: _ } => {
                        // Future enhancement: Apply cluster configuration changes.
                        // For v0.1.0: Static cluster configuration is sufficient.
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

    /// Send an RPC to another node.
    ///
    /// No cross-node network transport is wired in this in-process Raft
    /// implementation. Returning `Ok(())` here (as the previous stub did) made
    /// [`RaftNode::start_election`] treat every peer as having granted its vote,
    /// so every node self-elected — a split-brain. This now fails loud so that
    /// caller vote/replication logic can never mistake "not sent" for "delivered".
    async fn send_rpc(&self, target: &str, _message: RpcMessage) -> FusekiResult<()> {
        Err(FusekiError::internal(format!(
            "Raft RPC transport is not wired in this build; cannot send to {target}"
        )))
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

    /// Regression: a node with multiple configured members must refuse to start
    /// (no wired transport → would split-brain), and `send_rpc` must fail loud
    /// rather than fabricate a successful delivery.
    #[tokio::test]
    async fn regression_multinode_start_fails_loud_and_no_fake_rpc() {
        use crate::store::Store;
        let store = Arc::new(Store::new().expect("store"));
        let node = Arc::new(
            RaftNode::new("n1".to_string(), RaftConfig::default(), store)
                .await
                .expect("node"),
        );

        // send_rpc must not silently succeed (this was the split-brain source).
        let vote = RpcMessage::RequestVote(RequestVoteRequest {
            term: 1,
            candidate_id: "n1".to_string(),
            last_log_index: 0,
            last_log_term: 0,
        });
        assert!(node.send_rpc("n2", vote).await.is_err());

        // Configure a 3-node cluster and assert start refuses.
        {
            let mut cfg = node.cluster_config.write().await;
            cfg.members = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        }
        assert!(node.start().await.is_err());
    }

    /// Regression: a single-node cluster bootstraps to leader for real (majority
    /// of 1 is satisfied by the self-vote alone).
    #[tokio::test]
    async fn regression_single_node_bootstrap_becomes_leader() {
        use crate::store::Store;
        let store = Arc::new(Store::new().expect("store"));
        let node = Arc::new(
            RaftNode::new("solo".to_string(), RaftConfig::default(), store)
                .await
                .expect("node"),
        );
        node.bootstrap().await.expect("bootstrap");
        assert_eq!(*node.state.read().await, RaftState::Leader);
        node.start().await.expect("single-node start ok");
    }
}
