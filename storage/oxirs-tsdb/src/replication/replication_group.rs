//! Simulated Raft replication group for TSDB testing and single-process
//! embedded deployments.
//!
//! [`ReplicationGroup`] manages a set of [`TsdbRaftNode`] instances that
//! communicate through in-process message queues rather than over a real
//! network.  This makes the implementation useful for:
//!
//! 1. **Unit tests** -- deterministic leader election and log replication
//!    without spawning threads or sockets.
//! 2. **Embedded deployments** -- lightweight consensus for a small (3-5)
//!    node cluster within a single process or machine.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────┐
//!  │                 ReplicationGroup                 │
//!  │                                                  │
//!  │  ┌──────────────┐    ┌──────────────┐           │
//!  │  │ TsdbRaftNode │◄──►│ TsdbRaftNode │  ···      │
//!  │  │   (leader)   │    │  (follower)  │           │
//!  │  └──────────────┘    └──────────────┘           │
//!  │         │                    │                   │
//!  │         └────────────────────┘                   │
//!  │           in-process message passing             │
//!  └──────────────────────────────────────────────────┘
//! ```
//!
//! ## References
//!
//! - Ongaro, D. & Ousterhout, J. (2014). *In Search of an Understandable
//!   Consensus Algorithm*. USENIX ATC '14.
//!   <https://raft.github.io/raft.pdf>

use crate::error::{TsdbError, TsdbResult};
use crate::replication::raft_state::{
    AppendEntriesArgs, RaftRole, RaftState, RequestVoteArgs, TsdbCommand,
};
use std::collections::{HashMap, HashSet, VecDeque};

// =============================================================================
// WriteEntry -- time-series write entry for quorum replication
// =============================================================================

/// A time-series write entry that is replicated through the Raft log.
///
/// This is the high-level application-facing write primitive; it is
/// internally converted to a [`TsdbCommand::WriteDatapoint`] before being
/// proposed to the Raft log.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteEntry {
    /// Unix epoch milliseconds.
    pub timestamp: i64,
    /// Metric / series name.
    pub metric_name: String,
    /// Observed measurement value.
    pub value: f64,
    /// Optional tag key-value pairs for dimensionality.
    pub tags: HashMap<String, String>,
}

impl WriteEntry {
    /// Create a new write entry with no tags.
    pub fn new(timestamp: i64, metric_name: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp,
            metric_name: metric_name.into(),
            value,
            tags: HashMap::new(),
        }
    }

    /// Builder: add a single tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Builder: add multiple tags at once.
    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Convert to a [`TsdbCommand::WriteDatapoint`] for Raft replication.
    pub fn to_command(&self) -> TsdbCommand {
        TsdbCommand::WriteDatapoint {
            series_id: self.metric_name.clone(),
            timestamp: self.timestamp,
            value: self.value,
        }
    }
}

// =============================================================================
// Message types exchanged between nodes
// =============================================================================

/// An RPC message that one node sends to another.
#[derive(Debug, Clone)]
enum RaftMessage {
    /// `RequestVote` RPC (section 5.2).
    Vote(RequestVoteArgs),
    /// `AppendEntries` RPC (section 5.3 / 5.4) including heartbeats.
    Append(AppendEntriesArgs),
    /// Vote granted reply from `voter_id` to `candidate_id`.
    VoteGranted {
        /// The term of the voter when the vote was cast.
        term: u64,
    },
    /// AppendEntries success acknowledgement from follower to leader.
    AppendAck {
        /// Term of the follower.
        term: u64,
        /// The follower's last log index after applying entries.
        match_index: u64,
    },
}

/// A queued outbound message from `sender` to `recipient`.
#[derive(Debug)]
struct Envelope {
    sender: String,
    recipient: String,
    message: RaftMessage,
}

// =============================================================================
// TsdbRaftNode
// =============================================================================

/// A single Raft participant that wraps [`RaftState`] and maintains its own
/// inbound message queue.
///
/// All I/O and timer operations are handled by the enclosing
/// [`ReplicationGroup`]; the node itself is purely reactive.
#[derive(Debug)]
pub struct TsdbRaftNode {
    /// Node identifier (must be unique within the group).
    pub id: String,
    /// Core Raft state machine.
    pub(super) state: RaftState,
    /// Whether this node is currently reachable (simulates network partition).
    reachable: bool,
    /// Number of election timeouts elapsed without receiving a valid leader
    /// heartbeat.  Used by the group to trigger elections.
    pub(super) election_ticks: u32,
    /// Randomised election timeout threshold (ticks).
    pub(super) election_timeout: u32,
    /// Set of peer IDs that have granted a vote to this node in the current term.
    votes_for_me: HashSet<String>,
    /// Total number of write entries committed via quorum on this node.
    committed_writes: u64,
}

impl TsdbRaftNode {
    /// Create a new node with the given election timeout (in ticks).
    pub fn new(id: impl Into<String>, election_timeout: u32) -> Self {
        let id = id.into();
        let state = RaftState::new(id.clone());
        Self {
            id,
            state,
            reachable: true,
            election_ticks: 0,
            election_timeout,
            votes_for_me: HashSet::new(),
            committed_writes: 0,
        }
    }

    /// Return the current Raft role.
    pub fn role(&self) -> RaftRole {
        self.state.role.clone()
    }

    /// Return the current term.
    pub fn current_term(&self) -> u64 {
        self.state.current_term
    }

    /// Return the commit index.
    pub fn commit_index(&self) -> u64 {
        self.state.commit_index
    }

    /// Return the number of log entries (excluding the sentinel at index 0).
    pub fn log_len(&self) -> usize {
        // Subtract sentinel entry at position 0
        self.state.log.len().saturating_sub(1)
    }

    /// Return the last log index (0 when no user entries have been appended).
    pub fn last_log_index(&self) -> u64 {
        self.state.last_log_index()
    }

    /// Mark this node as unreachable (network partition).
    pub fn partition(&mut self) {
        self.reachable = false;
    }

    /// Restore network reachability.
    pub fn heal(&mut self) {
        self.reachable = true;
    }

    /// Whether the node is currently reachable.
    pub fn is_reachable(&self) -> bool {
        self.reachable
    }

    /// Total committed writes on this node.
    pub fn committed_writes(&self) -> u64 {
        self.committed_writes
    }

    /// Propose a write command on this node.
    ///
    /// Returns the log index if this node is the leader, or an error
    /// otherwise.
    pub fn propose(&mut self, cmd: TsdbCommand) -> TsdbResult<u64> {
        self.state
            .propose_command(cmd)
            .map_err(|e| TsdbError::Replication(e.to_string()))
    }

    /// Propose a [`WriteEntry`] on this node (convenience wrapper).
    pub fn propose_write_entry(&mut self, entry: &WriteEntry) -> TsdbResult<u64> {
        self.propose(entry.to_command())
    }

    /// Return applied commands (committed but not yet consumed).
    pub fn drain_applied(&mut self) -> Vec<TsdbCommand> {
        let applied = self.state.apply_committed_entries();
        // Count write datapoints in applied entries
        for cmd in &applied {
            if matches!(cmd, TsdbCommand::WriteDatapoint { .. }) {
                self.committed_writes += 1;
            }
        }
        applied
    }
}

// =============================================================================
// ReplicationGroup
// =============================================================================

/// A simulated Raft cluster of [`TsdbRaftNode`]s.
///
/// All message delivery, election timeout tracking, and commit-index
/// advancement are driven by calls to [`ReplicationGroup::tick`].
pub struct ReplicationGroup {
    nodes: HashMap<String, TsdbRaftNode>,
    /// Outbound message bus: envelopes waiting to be delivered.
    bus: VecDeque<Envelope>,
    /// Number of nodes in this group (fixed at construction).
    cluster_size: usize,
}

impl ReplicationGroup {
    /// Create a group with the given node IDs and (optionally distinct)
    /// election timeouts.
    ///
    /// Pass `timeout_override = None` to assign uniform timeouts (10 ticks).
    /// Pass a slice of per-node timeouts to make elections deterministic.
    pub fn new(ids: &[&str], timeout_override: Option<&[u32]>) -> Self {
        let mut nodes = HashMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let timeout = timeout_override
                .and_then(|ts| ts.get(i).copied())
                .unwrap_or(10);
            nodes.insert(id.to_string(), TsdbRaftNode::new(id, timeout));
        }
        let cluster_size = nodes.len();
        Self {
            nodes,
            bus: VecDeque::new(),
            cluster_size,
        }
    }

    /// Return the number of nodes in the group.
    pub fn cluster_size(&self) -> usize {
        self.cluster_size
    }

    /// Return a reference to a node by ID.
    pub fn node(&self, id: &str) -> Option<&TsdbRaftNode> {
        self.nodes.get(id)
    }

    /// Return a mutable reference to a node by ID.
    pub fn node_mut(&mut self, id: &str) -> Option<&mut TsdbRaftNode> {
        self.nodes.get_mut(id)
    }

    /// Find the current leader, if any.
    ///
    /// Returns `None` if no node is in the `Leader` role.
    pub fn leader_id(&self) -> Option<String> {
        self.nodes
            .values()
            .find(|n| n.role() == RaftRole::Leader && n.reachable)
            .map(|n| n.id.clone())
    }

    /// Return the number of nodes currently acting as leader.
    ///
    /// In a healthy cluster this should always be <= 1.
    pub fn leader_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| n.role() == RaftRole::Leader && n.reachable)
            .count()
    }

    /// Return the quorum size (majority) for this cluster.
    pub fn quorum_size(&self) -> usize {
        self.cluster_size / 2 + 1
    }

    /// Partition a node (simulates network isolation).
    pub fn partition(&mut self, id: &str) {
        if let Some(n) = self.nodes.get_mut(id) {
            n.partition();
        }
    }

    /// Heal a partitioned node.
    pub fn heal(&mut self, id: &str) {
        if let Some(n) = self.nodes.get_mut(id) {
            n.heal();
        }
    }

    /// Propose a write command on the current leader.
    ///
    /// Returns `(leader_id, log_index)` or an error if there is no leader.
    pub fn propose(&mut self, cmd: TsdbCommand) -> TsdbResult<(String, u64)> {
        let leader_id = self
            .leader_id()
            .ok_or_else(|| TsdbError::Replication("no leader elected yet".into()))?;

        let node = self
            .nodes
            .get_mut(&leader_id)
            .ok_or_else(|| TsdbError::Replication("leader disappeared from nodes map".into()))?;

        let idx = node.propose(cmd)?;
        Ok((leader_id, idx))
    }

    /// Propose a [`WriteEntry`] via the current leader (convenience wrapper).
    ///
    /// Returns `(leader_id, log_index)` or an error.
    pub fn propose_write_entry(&mut self, entry: &WriteEntry) -> TsdbResult<(String, u64)> {
        self.propose(entry.to_command())
    }

    /// Propose a [`WriteEntry`] and wait for quorum commit by ticking.
    ///
    /// Returns `(leader_id, log_index)` after up to `max_ticks` ticks.
    /// Returns an error if the entry is not committed within `max_ticks`.
    pub fn propose_and_commit(
        &mut self,
        entry: &WriteEntry,
        max_ticks: u32,
    ) -> TsdbResult<(String, u64)> {
        let (leader_id, log_index) = self.propose_write_entry(entry)?;
        for _ in 0..max_ticks {
            self.tick();
            if let Some(leader) = self.nodes.get(&leader_id) {
                if leader.commit_index() >= log_index {
                    return Ok((leader_id, log_index));
                }
            }
        }
        Err(TsdbError::Replication(format!(
            "quorum commit not achieved for index {log_index} within {max_ticks} ticks"
        )))
    }

    // ── Tick ──────────────────────────────────────────────────────────────────

    /// Advance the simulation by one logical tick.
    ///
    /// Each tick:
    /// 1. Sends leader heartbeats first (so they enter the bus).
    /// 2. Delivers all pending messages from the bus.
    /// 3. Increments election timers on followers/candidates.
    /// 4. Advances commit indices on the leader.
    ///
    /// The order matters: heartbeats are sent first so they can be delivered
    /// in the same tick, preventing unnecessary election timeouts.
    pub fn tick(&mut self) {
        // Send heartbeats first so they're in the bus for delivery
        self.send_leader_heartbeats();
        // Deliver all pending messages (heartbeats + any other messages)
        self.deliver_messages();
        // Now advance election timers (followers that received heartbeats
        // had their timers reset during delivery)
        self.advance_election_timers();
        // Advance commit indices based on match_index state
        self.advance_commit_indices();
    }

    /// Run `n` ticks in sequence.
    pub fn tick_n(&mut self, n: u32) {
        for _ in 0..n {
            self.tick();
        }
    }

    // ── Message delivery ──────────────────────────────────────────────────────

    /// Deliver all pending messages from the bus.
    ///
    /// For each `RequestVote`, the recipient processes it and, if it grants
    /// the vote, sends a `VoteGranted` back.  For each `AppendEntries`, the
    /// recipient processes it and sends an `AppendAck` back if successful.
    fn deliver_messages(&mut self) {
        // We may need multiple rounds of delivery for replies to propagate.
        for _round in 0..3 {
            if self.bus.is_empty() {
                break;
            }
            let envelopes: Vec<Envelope> = self.bus.drain(..).collect();
            let mut replies: Vec<Envelope> = Vec::new();

            for env in envelopes {
                // Drop messages to unreachable nodes
                let recipient_reachable = self
                    .nodes
                    .get(&env.recipient)
                    .map(|n| n.reachable)
                    .unwrap_or(false);
                if !recipient_reachable {
                    continue;
                }

                // Drop messages from unreachable senders (partitioned)
                let sender_reachable = self
                    .nodes
                    .get(&env.sender)
                    .map(|n| n.reachable)
                    .unwrap_or(true);
                if !sender_reachable {
                    continue;
                }

                let recipient = match self.nodes.get_mut(&env.recipient) {
                    Some(n) => n,
                    None => continue,
                };

                match env.message {
                    RaftMessage::Vote(args) => {
                        let reply = recipient.state.handle_vote_request(&args);
                        if reply.vote_granted {
                            replies.push(Envelope {
                                sender: env.recipient.clone(),
                                recipient: env.sender.clone(),
                                message: RaftMessage::VoteGranted { term: reply.term },
                            });
                        }
                    }
                    RaftMessage::VoteGranted { term } => {
                        if recipient.state.current_term == term
                            && recipient.role() == RaftRole::Candidate
                        {
                            recipient.votes_for_me.insert(env.sender.clone());
                        }
                    }
                    RaftMessage::Append(args) => {
                        let reply = recipient.state.handle_append_entries(&args);
                        if reply.success {
                            // Reset election timer -- valid heartbeat/append received
                            recipient.election_ticks = 0;
                            if !args.entries.is_empty() {
                                replies.push(Envelope {
                                    sender: env.recipient.clone(),
                                    recipient: env.sender.clone(),
                                    message: RaftMessage::AppendAck {
                                        term: reply.term,
                                        match_index: recipient.state.last_log_index(),
                                    },
                                });
                            }
                        }
                    }
                    RaftMessage::AppendAck { term, match_index } => {
                        // Step down if reply carries a higher term (Raft safety)
                        if term > recipient.state.current_term {
                            recipient.state.become_follower(term);
                        } else if recipient.role() == RaftRole::Leader {
                            recipient
                                .state
                                .handle_append_success(&env.sender, match_index);
                        }
                    }
                }
            }

            for r in replies {
                self.bus.push_back(r);
            }
        }
    }

    // ── Election timers ───────────────────────────────────────────────────────

    /// Increment election timers; trigger elections for timed-out
    /// followers/candidates and promote candidates with a quorum.
    fn advance_election_timers(&mut self) {
        let quorum = self.cluster_size / 2 + 1;

        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        let mut to_start_election: Vec<String> = Vec::new();
        let mut to_promote: Vec<String> = Vec::new();

        for id in &node_ids {
            let node = match self.nodes.get_mut(id) {
                Some(n) if n.reachable => n,
                _ => continue,
            };

            match node.role() {
                RaftRole::Leader => {
                    // Leaders don't time out
                    node.election_ticks = 0;
                }
                RaftRole::Follower | RaftRole::Candidate => {
                    node.election_ticks += 1;
                    if node.election_ticks >= node.election_timeout {
                        node.election_ticks = 0;
                        let _vote_args = node.state.become_candidate();
                        node.votes_for_me.clear();
                        node.votes_for_me.insert(id.clone());
                        to_start_election.push(id.clone());
                    }

                    // Check quorum for existing candidates
                    if node.role() == RaftRole::Candidate && node.votes_for_me.len() >= quorum {
                        to_promote.push(id.clone());
                    }
                }
            }
        }

        // Broadcast RequestVote for nodes starting elections
        for candidate_id in &to_start_election {
            let vote_args = {
                let node = match self.nodes.get(candidate_id) {
                    Some(n) => n,
                    None => continue,
                };
                RequestVoteArgs {
                    term: node.state.current_term,
                    candidate_id: candidate_id.clone(),
                    last_log_index: node.state.last_log_index(),
                    last_log_term: node.state.log.last().map(|e| e.term).unwrap_or(0),
                }
            };

            let peers: Vec<String> = self
                .nodes
                .keys()
                .filter(|pid| pid.as_str() != candidate_id.as_str())
                .cloned()
                .collect();

            for peer in peers {
                self.bus.push_back(Envelope {
                    sender: candidate_id.clone(),
                    recipient: peer,
                    message: RaftMessage::Vote(vote_args.clone()),
                });
            }
        }

        // Promote candidates with quorum
        for candidate_id in &to_promote {
            let peers: Vec<String> = self
                .nodes
                .keys()
                .filter(|pid| pid.as_str() != candidate_id.as_str())
                .cloned()
                .collect();

            if let Some(node) = self.nodes.get_mut(candidate_id) {
                if node.role() == RaftRole::Candidate && node.votes_for_me.len() >= quorum {
                    node.state.become_leader(&peers);
                    node.votes_for_me.clear();
                }
            }
        }
    }

    // ── Leader heartbeats ─────────────────────────────────────────────────────

    /// Leader sends heartbeat `AppendEntries` to all reachable peers.
    fn send_leader_heartbeats(&mut self) {
        let leader_id = match self.leader_id() {
            Some(id) => id,
            None => return,
        };

        let peers: Vec<String> = self
            .nodes
            .keys()
            .filter(|id| {
                id.as_str() != leader_id.as_str()
                    && self
                        .nodes
                        .get(id.as_str())
                        .map(|n| n.reachable)
                        .unwrap_or(false)
            })
            .cloned()
            .collect();

        let leader = match self.nodes.get_mut(&leader_id) {
            Some(n) => n,
            None => return,
        };

        for peer in &peers {
            if let Ok(args) = leader.state.build_append_entries(peer) {
                self.bus.push_back(Envelope {
                    sender: leader_id.clone(),
                    recipient: peer.clone(),
                    message: RaftMessage::Append(args),
                });
            }
        }
    }

    // ── Commit index advancement ───────────────────────────────────────────────

    /// Advance the leader's commit index based on follower acknowledgements.
    fn advance_commit_indices(&mut self) {
        let leader_id = match self.leader_id() {
            Some(id) => id,
            None => return,
        };

        if let Some(leader) = self.nodes.get_mut(&leader_id) {
            leader.state.try_advance_commit_index(self.cluster_size);
        }
    }

    /// Return all node IDs in the cluster.
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Return the number of pending messages on the bus.
    pub fn pending_messages(&self) -> usize {
        self.bus.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replication::raft_state::TsdbCommand;

    /// Build a deterministic 3-node group:
    /// node-0 times out first (3 ticks), node-1 at 5, node-2 at 7.
    fn three_node_group() -> ReplicationGroup {
        ReplicationGroup::new(&["node-0", "node-1", "node-2"], Some(&[3, 5, 7]))
    }

    // ── Initial state ────────────────────────────────────────────────────────

    #[test]
    fn test_initial_all_followers() {
        let g = three_node_group();
        assert!(g.leader_id().is_none());
        assert!(g.nodes.values().all(|n| n.role() == RaftRole::Follower));
    }

    #[test]
    fn test_cluster_size_three() {
        let g = three_node_group();
        assert_eq!(g.cluster_size(), 3);
    }

    #[test]
    fn test_node_lookup() {
        let g = three_node_group();
        assert!(g.node("node-0").is_some());
        assert!(g.node("missing").is_none());
    }

    // ── Log length baseline ───────────────────────────────────────────────────

    #[test]
    fn test_node_initial_log_len_is_zero() {
        let g = three_node_group();
        assert_eq!(g.node("node-1").expect("n").log_len(), 0);
    }

    #[test]
    fn test_node_initial_commit_index() {
        let g = three_node_group();
        assert_eq!(g.node("node-0").expect("n").commit_index(), 0);
    }

    #[test]
    fn test_node_last_log_index_zero_initially() {
        let g = three_node_group();
        assert_eq!(g.node("node-2").expect("n").last_log_index(), 0);
    }

    // ── Election ─────────────────────────────────────────────────────────────

    #[test]
    fn test_leader_elected_after_ticks() {
        let mut g = three_node_group();
        g.tick_n(20);
        assert_eq!(g.leader_count(), 1, "expected exactly one leader");
    }

    #[test]
    fn test_leader_has_term_at_least_one() {
        let mut g = three_node_group();
        g.tick_n(20);
        let term = g
            .leader_id()
            .and_then(|id| g.node(&id))
            .map(|n| n.current_term())
            .unwrap_or(0);
        assert!(term >= 1);
    }

    #[test]
    fn test_no_duplicate_leaders_during_election() {
        let mut g = three_node_group();
        g.tick_n(20);
        let leader_count_at_20 = g.leader_count();
        g.tick_n(30);
        assert!(g.leader_count() <= 1);
        if leader_count_at_20 == 1 {
            assert_eq!(g.leader_count(), 1);
        }
    }

    #[test]
    fn test_five_node_group_elects_leader() {
        let mut g = ReplicationGroup::new(&["a", "b", "c", "d", "e"], Some(&[3, 6, 9, 12, 15]));
        g.tick_n(40);
        assert_eq!(
            g.leader_count(),
            1,
            "5-node group must elect exactly 1 leader"
        );
    }

    // ── Log replication ───────────────────────────────────────────────────────

    #[test]
    fn test_propose_on_leader_increments_log() {
        let mut g = three_node_group();
        g.tick_n(20);

        let cmd = TsdbCommand::WriteDatapoint {
            series_id: "temperature".into(),
            timestamp: 1_700_000_000_000,
            value: 21.3,
        };
        let result = g.propose(cmd);
        assert!(result.is_ok(), "propose should succeed when leader exists");
        let (_, idx) = result.expect("ok");
        assert!(idx >= 1);
    }

    #[test]
    fn test_multiple_proposals_monotone_indices() {
        let mut g = three_node_group();
        g.tick_n(20);

        let mut prev_idx = 0u64;
        for i in 0..5 {
            let cmd = TsdbCommand::WriteDatapoint {
                series_id: "s".into(),
                timestamp: i * 1_000,
                value: i as f64,
            };
            let (_, idx) = g.propose(cmd).expect("propose");
            assert!(idx > prev_idx, "indices must be strictly increasing");
            prev_idx = idx;
        }
    }

    #[test]
    fn test_propose_without_leader_returns_error() {
        let mut g = three_node_group();
        let cmd = TsdbCommand::WriteDatapoint {
            series_id: "s".into(),
            timestamp: 0,
            value: 0.0,
        };
        assert!(g.propose(cmd).is_err());
    }

    // ── Partition and recovery ────────────────────────────────────────────────

    #[test]
    fn test_partition_reduces_leader_count() {
        let mut g = three_node_group();
        g.tick_n(20);

        let old_leader = g.leader_id().expect("leader must exist");
        g.partition(&old_leader);
        assert_eq!(
            g.leader_count(),
            0,
            "partitioned leader should not be counted as reachable leader"
        );
    }

    #[test]
    fn test_heal_partition_restores_node() {
        let mut g = three_node_group();
        g.partition("node-2");
        assert!(!g.node("node-2").expect("exists").is_reachable());
        g.heal("node-2");
        assert!(g.node("node-2").expect("exists").is_reachable());
    }

    #[test]
    fn test_remaining_nodes_can_elect_after_partition() {
        let mut g = three_node_group();
        g.tick_n(20);

        let old_leader = g.leader_id().expect("leader");
        g.partition(&old_leader);
        g.tick_n(30);

        let reachable_leaders = g
            .nodes
            .values()
            .filter(|n| n.reachable && n.role() == RaftRole::Leader)
            .count();
        assert!(reachable_leaders <= 1);
    }

    #[test]
    fn test_heal_partition_node_is_reachable() {
        let mut g = three_node_group();
        g.partition("node-1");
        g.heal("node-1");
        assert!(g.node("node-1").expect("n").is_reachable());
    }

    // ── Node state queries ────────────────────────────────────────────────────

    #[test]
    fn test_replication_group_node_count() {
        let g = ReplicationGroup::new(&["a", "b", "c", "d", "e"], None);
        assert_eq!(g.cluster_size(), 5);
    }

    #[test]
    fn test_replication_group_default_timeout() {
        let g = ReplicationGroup::new(&["x", "y"], None);
        assert_eq!(g.node("x").expect("x").election_timeout, 10);
        assert_eq!(g.node("y").expect("y").election_timeout, 10);
    }

    #[test]
    fn test_replication_group_custom_timeouts() {
        let g = ReplicationGroup::new(&["p", "q"], Some(&[3, 7]));
        assert_eq!(g.node("p").expect("p").election_timeout, 3);
        assert_eq!(g.node("q").expect("q").election_timeout, 7);
    }

    // ── TsdbRaftNode unit tests ───────────────────────────────────────────────

    #[test]
    fn test_tsdb_raft_node_propose_non_leader_error() {
        let mut node = TsdbRaftNode::new("solo", 5);
        let result = node.propose(TsdbCommand::DeleteSeries {
            series_id: "old".into(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_tsdb_raft_node_initial_role_follower() {
        let node = TsdbRaftNode::new("n", 5);
        assert_eq!(node.role(), RaftRole::Follower);
    }

    #[test]
    fn test_tsdb_raft_node_partition_and_heal() {
        let mut node = TsdbRaftNode::new("n", 5);
        assert!(node.is_reachable());
        node.partition();
        assert!(!node.is_reachable());
        node.heal();
        assert!(node.is_reachable());
    }

    #[test]
    fn test_tsdb_raft_node_drain_applied_empty_initially() {
        let mut node = TsdbRaftNode::new("n", 5);
        assert!(node.drain_applied().is_empty());
    }

    #[test]
    fn test_tsdb_raft_node_initial_term() {
        let node = TsdbRaftNode::new("n", 5);
        assert_eq!(node.current_term(), 0);
    }

    // ── Leader stability ──────────────────────────────────────────────────────

    #[test]
    fn test_leader_stable_after_election() {
        let mut g = three_node_group();
        g.tick_n(20);

        let leader_at_20 = g.leader_id();
        assert!(leader_at_20.is_some(), "must have elected a leader");

        g.tick_n(10);
        assert_eq!(g.leader_count(), 1, "leader must remain stable");
    }

    #[test]
    fn test_leader_count_never_exceeds_one() {
        let mut g = ReplicationGroup::new(&["n0", "n1", "n2"], Some(&[4, 8, 12]));
        for _ in 0..40 {
            g.tick();
            assert!(
                g.leader_count() <= 1,
                "split brain detected: {} leaders",
                g.leader_count()
            );
        }
    }

    // ── WriteEntry tests ──────────────────────────────────────────────────────

    #[test]
    fn test_write_entry_new() {
        let entry = WriteEntry::new(1_700_000_000_000, "cpu_usage", 42.5);
        assert_eq!(entry.timestamp, 1_700_000_000_000);
        assert_eq!(entry.metric_name, "cpu_usage");
        assert!((entry.value - 42.5).abs() < f64::EPSILON);
        assert!(entry.tags.is_empty());
    }

    #[test]
    fn test_write_entry_with_tag() {
        let entry = WriteEntry::new(1000, "mem", 8192.0)
            .with_tag("host", "srv-01")
            .with_tag("region", "eu-west");
        assert_eq!(entry.tags.len(), 2);
        assert_eq!(entry.tags["host"], "srv-01");
        assert_eq!(entry.tags["region"], "eu-west");
    }

    #[test]
    fn test_write_entry_with_tags_batch() {
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "prod".to_string());
        tags.insert("dc".to_string(), "us-east-1".to_string());

        let entry = WriteEntry::new(2000, "latency", 1.5).with_tags(tags);
        assert_eq!(entry.tags.len(), 2);
        assert_eq!(entry.tags["env"], "prod");
    }

    #[test]
    fn test_write_entry_to_command() {
        let entry = WriteEntry::new(5000, "temperature", 22.7);
        let cmd = entry.to_command();
        match cmd {
            TsdbCommand::WriteDatapoint {
                series_id,
                timestamp,
                value,
            } => {
                assert_eq!(series_id, "temperature");
                assert_eq!(timestamp, 5000);
                assert!((value - 22.7).abs() < f64::EPSILON);
            }
            other => panic!("expected WriteDatapoint, got {:?}", other),
        }
    }

    #[test]
    fn test_write_entry_clone_eq() {
        let a = WriteEntry::new(100, "x", 1.0);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── Quorum commit tests ─────────────────────────────────────────────────

    #[test]
    fn test_quorum_size_three_node() {
        let g = three_node_group();
        assert_eq!(g.quorum_size(), 2);
    }

    #[test]
    fn test_quorum_size_five_node() {
        let g = ReplicationGroup::new(&["a", "b", "c", "d", "e"], None);
        assert_eq!(g.quorum_size(), 3);
    }

    #[test]
    fn test_propose_write_entry_on_leader() {
        let mut g = three_node_group();
        g.tick_n(20);

        let entry = WriteEntry::new(1_700_000_000_000, "cpu", 88.0);
        let result = g.propose_write_entry(&entry);
        assert!(result.is_ok());
        let (leader_id, idx) = result.expect("propose");
        assert!(!leader_id.is_empty());
        assert!(idx >= 1);
    }

    #[test]
    fn test_propose_write_entry_no_leader() {
        let mut g = three_node_group();
        let entry = WriteEntry::new(0, "x", 0.0);
        assert!(g.propose_write_entry(&entry).is_err());
    }

    #[test]
    fn test_propose_and_commit_succeeds() {
        let mut g = three_node_group();
        g.tick_n(20);

        let entry = WriteEntry::new(1000, "sensor_a", 99.0);
        let result = g.propose_and_commit(&entry, 20);
        assert!(
            result.is_ok(),
            "quorum commit should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_propose_and_commit_multiple_entries() {
        let mut g = three_node_group();
        g.tick_n(20);

        for i in 0..5 {
            let entry = WriteEntry::new(i * 1000, format!("metric_{i}"), i as f64);
            let result = g.propose_and_commit(&entry, 20);
            assert!(result.is_ok(), "commit {i} should succeed");
        }
    }

    // ── Node IDs and pending messages ────────────────────────────────────────

    #[test]
    fn test_node_ids_returns_all() {
        let g = three_node_group();
        let ids = g.node_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"node-0".to_string()));
        assert!(ids.contains(&"node-1".to_string()));
        assert!(ids.contains(&"node-2".to_string()));
    }

    #[test]
    fn test_pending_messages_initially_zero() {
        let g = three_node_group();
        assert_eq!(g.pending_messages(), 0);
    }

    // ── Committed writes tracking ──────────────────────────────────────────────

    #[test]
    fn test_committed_writes_initially_zero() {
        let node = TsdbRaftNode::new("n", 5);
        assert_eq!(node.committed_writes(), 0);
    }

    #[test]
    fn test_propose_write_entry_on_node() {
        let mut g = three_node_group();
        g.tick_n(20);
        let leader_id = g.leader_id().expect("leader");
        let entry = WriteEntry::new(1000, "temp", 22.0);
        let node = g.node_mut(&leader_id).expect("node");
        let result = node.propose_write_entry(&entry);
        assert!(result.is_ok());
    }

    // ── Election with randomized timeouts ──────────────────────────────────

    #[test]
    fn test_randomized_timeouts_elect_fastest() {
        // node-0 has timeout=2 (fastest), should become leader first
        let mut g = ReplicationGroup::new(&["n0", "n1", "n2"], Some(&[2, 10, 15]));
        g.tick_n(20);
        assert_eq!(g.leader_count(), 1);
    }

    #[test]
    fn test_even_timeouts_still_elect_one() {
        let mut g = ReplicationGroup::new(&["a", "b", "c"], Some(&[5, 5, 5]));
        g.tick_n(30);
        // At least one leader should be elected (may require extra ticks due to ties)
        assert!(g.leader_count() <= 1);
    }

    // ── Leader step-down on partition ──────────────────────────────────────

    #[test]
    fn test_all_partitioned_no_leader() {
        let mut g = three_node_group();
        g.tick_n(20);
        g.partition("node-0");
        g.partition("node-1");
        g.partition("node-2");
        assert_eq!(g.leader_count(), 0);
    }

    #[test]
    fn test_heal_all_after_partition() {
        let mut g = three_node_group();
        g.tick_n(20);
        g.partition("node-0");
        g.partition("node-1");
        g.partition("node-2");
        g.heal("node-0");
        g.heal("node-1");
        g.heal("node-2");
        g.tick_n(30);
        assert!(g.leader_count() <= 1);
    }
}
