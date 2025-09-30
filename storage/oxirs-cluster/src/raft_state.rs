//! Raft State Machine Implementation
//!
//! This module implements the core Raft consensus algorithm state machine
//! with explicit node states (Follower, Candidate, Leader) and their transitions.

use anyhow::Result;
// MIGRATED: Using scirs2-core instead of direct rand dependency
use scirs2_core::random::{Rng, Random};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::{interval, sleep, timeout};
use tracing::{debug, error, info};

use crate::raft::{OxirsNodeId, RdfCommand};

/// Raft node state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Follower state - passive node that responds to leader
    Follower,
    /// Candidate state - actively seeking leadership
    Candidate,
    /// Leader state - actively coordinating the cluster
    Leader,
}

/// Raft term type
pub type Term = u64;

/// Log index type
pub type LogIndex = u64;

/// Log entry for Raft
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received
    pub term: Term,
    /// Position in the log
    pub index: LogIndex,
    /// Command to be applied
    pub command: RdfCommand,
}

/// Vote request RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: OxirsNodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// Vote response RPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponse {
    /// Current term for candidate to update
    pub term: Term,
    /// True means candidate received vote
    pub vote_granted: bool,
}

/// Append entries RPC (also used for heartbeats)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader ID
    pub leader_id: OxirsNodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
}

/// Append entries response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// For faster conflict resolution
    pub conflict_index: Option<LogIndex>,
    pub conflict_term: Option<Term>,
}

/// Raft state machine configuration
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Minimum election timeout in milliseconds
    pub election_timeout_min: u64,
    /// Maximum election timeout in milliseconds
    pub election_timeout_max: u64,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval: u64,
    /// Maximum batch size for log entries
    pub max_batch_size: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout_min: 150,
            election_timeout_max: 300,
            heartbeat_interval: 50,
            max_batch_size: 100,
        }
    }
}

/// Events that can trigger state transitions
#[derive(Debug, Clone, Copy)]
pub enum RaftEvent {
    /// Election timeout expired
    ElectionTimeout,
    /// Received higher term
    HigherTermDiscovered(Term),
    /// Won election
    ElectionWon,
    /// Lost election
    ElectionLost,
    /// Leader discovered
    LeaderDiscovered(OxirsNodeId),
    /// Lost leadership
    LeadershipLost,
}

/// Raft state machine
pub struct RaftStateMachine {
    /// Node ID
    node_id: OxirsNodeId,
    /// Current node state
    state: Arc<RwLock<NodeState>>,
    /// Current term
    current_term: Arc<RwLock<Term>>,
    /// Candidate ID that received vote in current term
    voted_for: Arc<RwLock<Option<OxirsNodeId>>>,
    /// Log entries
    log: Arc<RwLock<Vec<LogEntry>>>,
    /// Index of highest log entry known to be committed
    commit_index: Arc<RwLock<LogIndex>>,
    /// Index of highest log entry applied to state machine
    last_applied: Arc<RwLock<LogIndex>>,
    /// For each server, index of next log entry to send (leader only)
    next_index: Arc<RwLock<HashMap<OxirsNodeId, LogIndex>>>,
    /// For each server, index of highest log entry known to be replicated (leader only)
    match_index: Arc<RwLock<HashMap<OxirsNodeId, LogIndex>>>,
    /// Known peers
    peers: Arc<RwLock<HashSet<OxirsNodeId>>>,
    /// Current leader
    current_leader: Arc<RwLock<Option<OxirsNodeId>>>,
    /// Last heartbeat received
    last_heartbeat: Arc<Mutex<Instant>>,
    /// Configuration
    config: Arc<RaftConfig>,
    /// Channel for sending state change notifications
    state_tx: mpsc::UnboundedSender<NodeState>,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

impl RaftStateMachine {
    /// Create a new Raft state machine
    pub fn new(
        node_id: OxirsNodeId,
        peers: HashSet<OxirsNodeId>,
        config: RaftConfig,
    ) -> (Self, mpsc::UnboundedReceiver<NodeState>) {
        let (state_tx, state_rx) = mpsc::unbounded_channel();

        let state_machine = Self {
            node_id,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            current_term: Arc::new(RwLock::new(0)),
            voted_for: Arc::new(RwLock::new(None)),
            log: Arc::new(RwLock::new(Vec::new())),
            commit_index: Arc::new(RwLock::new(0)),
            last_applied: Arc::new(RwLock::new(0)),
            next_index: Arc::new(RwLock::new(HashMap::new())),
            match_index: Arc::new(RwLock::new(HashMap::new())),
            peers: Arc::new(RwLock::new(peers)),
            current_leader: Arc::new(RwLock::new(None)),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
            config: Arc::new(config),
            state_tx,
            shutdown: Arc::new(RwLock::new(false)),
        };

        (state_machine, state_rx)
    }

    /// Start the state machine
    pub async fn start(&self) {
        info!("Starting Raft state machine for node {}", self.node_id);

        // Start in follower state
        self.transition_to_follower().await;

        // Start the main state machine loop
        let state_machine = self.clone();
        tokio::spawn(async move {
            state_machine.run_state_machine().await;
        });
    }

    /// Run the main state machine loop
    async fn run_state_machine(&self) {
        loop {
            if *self.shutdown.read().await {
                info!("Shutting down Raft state machine for node {}", self.node_id);
                break;
            }

            let current_state = *self.state.read().await;

            match current_state {
                NodeState::Follower => self.run_follower().await,
                NodeState::Candidate => self.run_candidate().await,
                NodeState::Leader => self.run_leader().await,
            }
        }
    }

    /// Run follower state logic
    async fn run_follower(&self) {
        debug!("Node {} running as follower", self.node_id);

        let election_timeout = self.random_election_timeout();
        let timeout_duration = Duration::from_millis(election_timeout);

        loop {
            // Check if we should still be a follower
            if *self.state.read().await != NodeState::Follower {
                break;
            }

            // Wait for election timeout or heartbeat
            let last_heartbeat = *self.last_heartbeat.lock().await;
            let elapsed = last_heartbeat.elapsed();

            if elapsed >= timeout_duration {
                info!(
                    "Node {} election timeout expired, becoming candidate",
                    self.node_id
                );
                self.handle_event(RaftEvent::ElectionTimeout).await;
                break;
            }

            // Sleep for a short duration before checking again
            sleep(Duration::from_millis(10)).await;
        }
    }

    /// Run candidate state logic
    async fn run_candidate(&self) {
        debug!("Node {} running as candidate", self.node_id);

        // Start election
        if let Err(e) = self.start_election().await {
            error!("Election failed for node {}: {}", self.node_id, e);
            self.handle_event(RaftEvent::ElectionLost).await;
            return;
        }

        let election_timeout = self.random_election_timeout();
        let timeout_duration = Duration::from_millis(election_timeout);
        let start_time = Instant::now();

        loop {
            // Check if we should still be a candidate
            if *self.state.read().await != NodeState::Candidate {
                break;
            }

            // Check if election timeout expired
            if start_time.elapsed() >= timeout_duration {
                info!(
                    "Node {} election timeout expired, restarting election",
                    self.node_id
                );
                self.handle_event(RaftEvent::ElectionTimeout).await;
                break;
            }

            // Sleep for a short duration before checking again
            sleep(Duration::from_millis(10)).await;
        }
    }

    /// Run leader state logic
    async fn run_leader(&self) {
        info!("Node {} running as leader", self.node_id);

        // Initialize leader state
        self.initialize_leader_state().await;

        // Start heartbeat timer
        let mut heartbeat_interval =
            interval(Duration::from_millis(self.config.heartbeat_interval));

        loop {
            // Check if we should still be leader
            if *self.state.read().await != NodeState::Leader {
                break;
            }

            // Send heartbeats
            heartbeat_interval.tick().await;
            self.send_heartbeats().await;

            // Check for log entries to replicate
            self.replicate_log_entries().await;

            // Update commit index
            self.update_commit_index().await;
        }
    }

    /// Handle state machine events
    pub async fn handle_event(&self, event: RaftEvent) {
        let current_state = *self.state.read().await;

        match (current_state, event) {
            // Follower transitions
            (NodeState::Follower, RaftEvent::ElectionTimeout) => {
                self.transition_to_candidate().await;
            }
            (NodeState::Follower, RaftEvent::HigherTermDiscovered(term)) => {
                self.update_term(term).await;
            }

            // Candidate transitions
            (NodeState::Candidate, RaftEvent::ElectionWon) => {
                self.transition_to_leader().await;
            }
            (NodeState::Candidate, RaftEvent::ElectionLost) => {
                self.transition_to_follower().await;
            }
            (NodeState::Candidate, RaftEvent::LeaderDiscovered(leader_id)) => {
                *self.current_leader.write().await = Some(leader_id);
                self.transition_to_follower().await;
            }
            (NodeState::Candidate, RaftEvent::HigherTermDiscovered(term)) => {
                self.update_term(term).await;
                self.transition_to_follower().await;
            }

            // Leader transitions
            (NodeState::Leader, RaftEvent::HigherTermDiscovered(term)) => {
                self.update_term(term).await;
                self.transition_to_follower().await;
            }
            (NodeState::Leader, RaftEvent::LeadershipLost) => {
                self.transition_to_follower().await;
            }

            _ => {
                debug!("Ignoring event {:?} in state {:?}", event, current_state);
            }
        }
    }

    /// Transition to follower state
    async fn transition_to_follower(&self) {
        info!("Node {} transitioning to follower", self.node_id);

        *self.state.write().await = NodeState::Follower;
        *self.last_heartbeat.lock().await = Instant::now();

        // Notify state change
        let _ = self.state_tx.send(NodeState::Follower);
    }

    /// Transition to candidate state
    async fn transition_to_candidate(&self) {
        info!("Node {} transitioning to candidate", self.node_id);

        *self.state.write().await = NodeState::Candidate;

        // Increment current term
        let mut term = self.current_term.write().await;
        *term += 1;
        let _current_term = *term;
        drop(term);

        // Vote for self
        *self.voted_for.write().await = Some(self.node_id);

        // Reset election timer
        *self.last_heartbeat.lock().await = Instant::now();

        // Clear current leader
        *self.current_leader.write().await = None;

        // Notify state change
        let _ = self.state_tx.send(NodeState::Candidate);
    }

    /// Transition to leader state
    async fn transition_to_leader(&self) {
        info!("Node {} transitioning to leader", self.node_id);

        *self.state.write().await = NodeState::Leader;
        *self.current_leader.write().await = Some(self.node_id);

        // Notify state change
        let _ = self.state_tx.send(NodeState::Leader);
    }

    /// Initialize leader state
    async fn initialize_leader_state(&self) {
        let peers = self.peers.read().await;
        let last_log_index = self.get_last_log_index().await;

        let mut next_index = self.next_index.write().await;
        let mut match_index = self.match_index.write().await;

        for peer in peers.iter() {
            next_index.insert(*peer, last_log_index + 1);
            match_index.insert(*peer, 0);
        }
    }

    /// Start an election
    async fn start_election(&self) -> Result<()> {
        let current_term = *self.current_term.read().await;
        let last_log_index = self.get_last_log_index().await;
        let last_log_term = self.get_last_log_term().await;

        let request = VoteRequest {
            term: current_term,
            candidate_id: self.node_id,
            last_log_index,
            last_log_term,
        };

        let peers = self.peers.read().await.clone();
        let mut votes_received = 1; // Vote for self
        let votes_needed = (peers.len() + 1) / 2 + 1;

        // Request votes from all peers
        let (vote_tx, mut vote_rx) = mpsc::unbounded_channel();

        for peer in peers {
            let request = request.clone();
            let vote_tx = vote_tx.clone();

            // Generate random values outside the async block
            let delay_ms = {
                let mut random = Random::default();
                random.gen_range(10..50)
            };
            let vote_granted = {
                let mut random = Random::default();
                random.gen_bool(0.5)
            };

            // Simulate vote request (in real implementation, this would be an RPC)
            tokio::spawn(async move {
                // Simulate network delay
                sleep(Duration::from_millis(delay_ms)).await;

                // Simulate vote response
                let response = VoteResponse {
                    term: request.term,
                    vote_granted,
                };

                let _ = vote_tx.send((peer, response));
            });
        }

        // Collect votes
        let timeout_duration = Duration::from_millis(self.config.election_timeout_min);
        let deadline = Instant::now() + timeout_duration;

        while votes_received < votes_needed && Instant::now() < deadline {
            match timeout(
                deadline.saturating_duration_since(Instant::now()),
                vote_rx.recv(),
            )
            .await
            {
                Ok(Some((peer_id, response))) => {
                    if response.term > current_term {
                        self.handle_event(RaftEvent::HigherTermDiscovered(response.term))
                            .await;
                        return Err(anyhow::anyhow!("Higher term discovered"));
                    }

                    if response.vote_granted && response.term == current_term {
                        votes_received += 1;
                        debug!("Node {} received vote from {}", self.node_id, peer_id);
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        if votes_received >= votes_needed {
            info!(
                "Node {} won election with {} votes",
                self.node_id, votes_received
            );
            self.handle_event(RaftEvent::ElectionWon).await;
            Ok(())
        } else {
            info!(
                "Node {} lost election with {} votes",
                self.node_id, votes_received
            );
            self.handle_event(RaftEvent::ElectionLost).await;
            Err(anyhow::anyhow!("Insufficient votes"))
        }
    }

    /// Send heartbeats to all followers
    async fn send_heartbeats(&self) {
        let peers = self.peers.read().await.clone();
        let current_term = *self.current_term.read().await;
        let commit_index = *self.commit_index.read().await;

        for peer in peers {
            let prev_log_index = self.get_prev_log_index_for_peer(peer).await;
            let prev_log_term = self.get_prev_log_term_for_peer(peer).await;

            let _request = AppendEntriesRequest {
                term: current_term,
                leader_id: self.node_id,
                prev_log_index,
                prev_log_term,
                entries: Vec::new(), // Empty for heartbeat
                leader_commit: commit_index,
            };

            // Simulate sending heartbeat (in real implementation, this would be an RPC)
            debug!("Leader {} sending heartbeat to {}", self.node_id, peer);
        }
    }

    /// Replicate log entries to followers
    async fn replicate_log_entries(&self) {
        // This is a simplified version - in a real implementation,
        // this would send actual log entries to followers
        debug!("Leader {} replicating log entries", self.node_id);
    }

    /// Update commit index based on match indices
    async fn update_commit_index(&self) {
        let match_index = self.match_index.read().await;
        let current_term = *self.current_term.read().await;
        let log = self.log.read().await;

        // Find the highest log index that a majority of servers have replicated
        let mut indices: Vec<LogIndex> = match_index.values().cloned().collect();
        indices.push(log.len() as LogIndex); // Include leader's log
        indices.sort_unstable();

        let majority_index = indices.len() / 2;
        let new_commit_index = indices[majority_index];

        // Only commit entries from current term
        if new_commit_index > *self.commit_index.read().await {
            if let Some(entry) = log.get((new_commit_index - 1) as usize) {
                if entry.term == current_term {
                    *self.commit_index.write().await = new_commit_index;
                    debug!(
                        "Leader {} updated commit index to {}",
                        self.node_id, new_commit_index
                    );
                }
            }
        }
    }

    /// Handle vote request
    pub async fn handle_vote_request(&self, request: VoteRequest) -> VoteResponse {
        let mut current_term = self.current_term.write().await;
        let mut voted_for = self.voted_for.write().await;

        // Check if request term is higher
        if request.term > *current_term {
            *current_term = request.term;
            *voted_for = None;
            self.transition_to_follower().await;
        }

        let vote_granted = if request.term < *current_term
            || (voted_for.is_some() && *voted_for != Some(request.candidate_id))
            || !self.is_candidate_log_up_to_date(&request).await
        {
            false
        } else {
            *voted_for = Some(request.candidate_id);
            *self.last_heartbeat.lock().await = Instant::now();
            true
        };

        VoteResponse {
            term: *current_term,
            vote_granted,
        }
    }

    /// Handle append entries request
    pub async fn handle_append_entries(
        &self,
        request: AppendEntriesRequest,
    ) -> AppendEntriesResponse {
        let mut current_term = self.current_term.write().await;

        // Check if request term is higher
        if request.term > *current_term {
            *current_term = request.term;
            *self.voted_for.write().await = None;
            self.transition_to_follower().await;
        }

        if request.term < *current_term {
            return AppendEntriesResponse {
                term: *current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
            };
        }

        // Reset election timer
        *self.last_heartbeat.lock().await = Instant::now();

        // Recognize the leader
        if *self.state.read().await == NodeState::Candidate {
            self.handle_event(RaftEvent::LeaderDiscovered(request.leader_id))
                .await;
        }
        *self.current_leader.write().await = Some(request.leader_id);

        // Check log consistency
        let mut log = self.log.write().await;

        if request.prev_log_index > 0 {
            if let Some(entry) = log.get((request.prev_log_index - 1) as usize) {
                if entry.term != request.prev_log_term {
                    // Log inconsistency
                    return AppendEntriesResponse {
                        term: *current_term,
                        success: false,
                        conflict_index: Some(request.prev_log_index),
                        conflict_term: Some(entry.term),
                    };
                }
            } else {
                // Log too short
                return AppendEntriesResponse {
                    term: *current_term,
                    success: false,
                    conflict_index: Some(log.len() as LogIndex + 1),
                    conflict_term: None,
                };
            }
        }

        // Append entries
        for (i, entry) in request.entries.iter().enumerate() {
            let index = request.prev_log_index + i as LogIndex + 1;
            if let Some(existing) = log.get_mut((index - 1) as usize) {
                if existing.term != entry.term {
                    // Remove conflicting entries
                    log.truncate((index - 1) as usize);
                    log.push(entry.clone());
                }
            } else {
                log.push(entry.clone());
            }
        }

        // Update commit index
        if request.leader_commit > *self.commit_index.read().await {
            let new_commit = std::cmp::min(request.leader_commit, log.len() as LogIndex);
            *self.commit_index.write().await = new_commit;
        }

        AppendEntriesResponse {
            term: *current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
        }
    }

    /// Update term
    async fn update_term(&self, new_term: Term) {
        *self.current_term.write().await = new_term;
        *self.voted_for.write().await = None;
    }

    /// Get random election timeout
    fn random_election_timeout(&self) -> u64 {
        let mut random = Random::default();
        random.gen_range(self.config.election_timeout_min..=self.config.election_timeout_max)
    }

    /// Get last log index
    async fn get_last_log_index(&self) -> LogIndex {
        self.log.read().await.len() as LogIndex
    }

    /// Get last log term
    async fn get_last_log_term(&self) -> Term {
        self.log.read().await.last().map(|e| e.term).unwrap_or(0)
    }

    /// Get previous log index for a peer
    async fn get_prev_log_index_for_peer(&self, peer: OxirsNodeId) -> LogIndex {
        self.next_index
            .read()
            .await
            .get(&peer)
            .cloned()
            .unwrap_or(1)
            .saturating_sub(1)
    }

    /// Get previous log term for a peer
    async fn get_prev_log_term_for_peer(&self, peer: OxirsNodeId) -> Term {
        let prev_index = self.get_prev_log_index_for_peer(peer).await;
        if prev_index == 0 {
            0
        } else {
            self.log
                .read()
                .await
                .get((prev_index - 1) as usize)
                .map(|e| e.term)
                .unwrap_or(0)
        }
    }

    /// Check if candidate's log is up to date
    async fn is_candidate_log_up_to_date(&self, request: &VoteRequest) -> bool {
        let last_log_index = self.get_last_log_index().await;
        let last_log_term = self.get_last_log_term().await;

        request.last_log_term > last_log_term
            || (request.last_log_term == last_log_term && request.last_log_index >= last_log_index)
    }

    /// Shutdown the state machine
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
    }

    /// Get current state
    pub async fn get_state(&self) -> NodeState {
        *self.state.read().await
    }

    /// Get current term
    pub async fn get_current_term(&self) -> Term {
        *self.current_term.read().await
    }

    /// Get current leader
    pub async fn get_current_leader(&self) -> Option<OxirsNodeId> {
        *self.current_leader.read().await
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        *self.state.read().await == NodeState::Leader
    }

    /// Propose a new command (leader only)
    pub async fn propose_command(&self, command: RdfCommand) -> Result<()> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!("Not the leader"));
        }

        let term = *self.current_term.read().await;
        let mut log = self.log.write().await;
        let index = log.len() as LogIndex + 1;

        let entry = LogEntry {
            term,
            index,
            command,
        };

        log.push(entry);

        Ok(())
    }
}

impl Clone for RaftStateMachine {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            state: Arc::clone(&self.state),
            current_term: Arc::clone(&self.current_term),
            voted_for: Arc::clone(&self.voted_for),
            log: Arc::clone(&self.log),
            commit_index: Arc::clone(&self.commit_index),
            last_applied: Arc::clone(&self.last_applied),
            next_index: Arc::clone(&self.next_index),
            match_index: Arc::clone(&self.match_index),
            peers: Arc::clone(&self.peers),
            current_leader: Arc::clone(&self.current_leader),
            last_heartbeat: Arc::clone(&self.last_heartbeat),
            config: Arc::clone(&self.config),
            state_tx: self.state_tx.clone(),
            shutdown: Arc::clone(&self.shutdown),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_state_transitions() {
        let peers = HashSet::new();
        let config = RaftConfig::default();
        let (state_machine, mut state_rx) = RaftStateMachine::new(1, peers, config);

        // Start as follower
        assert_eq!(state_machine.get_state().await, NodeState::Follower);

        // Transition to candidate
        state_machine.handle_event(RaftEvent::ElectionTimeout).await;
        assert_eq!(state_machine.get_state().await, NodeState::Candidate);

        // Check state notification
        if let Some(new_state) = state_rx.recv().await {
            assert_eq!(new_state, NodeState::Candidate);
        }

        // Transition to leader
        state_machine.handle_event(RaftEvent::ElectionWon).await;
        assert_eq!(state_machine.get_state().await, NodeState::Leader);
        assert!(state_machine.is_leader().await);

        // Transition back to follower
        state_machine
            .handle_event(RaftEvent::HigherTermDiscovered(10))
            .await;
        assert_eq!(state_machine.get_state().await, NodeState::Follower);
        assert!(!state_machine.is_leader().await);
    }

    #[tokio::test]
    async fn test_term_updates() {
        let peers = HashSet::new();
        let config = RaftConfig::default();
        let (state_machine, _) = RaftStateMachine::new(1, peers, config);

        assert_eq!(state_machine.get_current_term().await, 0);

        // Becoming candidate increments term
        state_machine.handle_event(RaftEvent::ElectionTimeout).await;
        assert_eq!(state_machine.get_current_term().await, 1);

        // Discovering higher term updates current term
        state_machine
            .handle_event(RaftEvent::HigherTermDiscovered(5))
            .await;
        assert_eq!(state_machine.get_current_term().await, 5);
    }

    #[tokio::test]
    async fn test_vote_request_handling() {
        let peers = HashSet::new();
        let config = RaftConfig::default();
        let (state_machine, _) = RaftStateMachine::new(1, peers, config);

        let request = VoteRequest {
            term: 1,
            candidate_id: 2,
            last_log_index: 0,
            last_log_term: 0,
        };

        let response = state_machine.handle_vote_request(request).await;
        assert!(response.vote_granted);
        assert_eq!(response.term, 1);

        // Should not vote again in same term
        let request2 = VoteRequest {
            term: 1,
            candidate_id: 3,
            last_log_index: 0,
            last_log_term: 0,
        };

        let response2 = state_machine.handle_vote_request(request2).await;
        assert!(!response2.vote_granted);
    }

    #[tokio::test]
    async fn test_append_entries_handling() {
        let peers = HashSet::new();
        let config = RaftConfig::default();
        let (state_machine, _) = RaftStateMachine::new(1, peers, config);

        let request = AppendEntriesRequest {
            term: 1,
            leader_id: 2,
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                index: 1,
                command: RdfCommand::Insert {
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                },
            }],
            leader_commit: 0,
        };

        let response = state_machine.handle_append_entries(request).await;
        assert!(response.success);
        assert_eq!(response.term, 1);

        // Check log was updated
        assert_eq!(state_machine.get_last_log_index().await, 1);
        assert_eq!(state_machine.get_last_log_term().await, 1);
    }

    #[tokio::test]
    async fn test_leader_election_timeout() {
        let peers = HashSet::new();
        let config = RaftConfig {
            election_timeout_min: 50,
            election_timeout_max: 100,
            ..Default::default()
        };

        let (state_machine, _) = RaftStateMachine::new(1, peers, config);

        // Start state machine
        state_machine.start().await;

        // Wait for election timeout
        sleep(Duration::from_millis(150)).await;

        // Should have become candidate (and then leader since no peers)
        let state = state_machine.get_state().await;
        assert!(state == NodeState::Candidate || state == NodeState::Leader);

        state_machine.shutdown().await;
    }
}
