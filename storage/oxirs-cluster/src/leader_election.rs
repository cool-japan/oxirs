//! Bully algorithm leader election simulation.
//!
//! Implements the Bully algorithm for distributed leader election:
//! when a node notices the leader is unavailable, it starts an election.
//! The node with the highest id among those that respond wins.

use std::collections::HashMap;

/// The role of a node in the cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeRole {
    /// Not currently a candidate or leader.
    Follower,
    /// Participating in an election.
    Candidate,
    /// This node is the current leader.
    Leader,
}

/// Messages exchanged during the Bully algorithm election.
#[derive(Debug, Clone)]
pub enum ElectionMessage {
    /// "I want to start an election" (sent to higher-id nodes).
    Election {
        /// The node initiating the election.
        from_id: u64,
    },
    /// "I have won; I am the new leader."
    Victory {
        /// The winning node.
        leader_id: u64,
    },
    /// "I am alive and will handle the election."
    Alive {
        /// The responding node's id.
        from_id: u64,
    },
    /// Periodic heartbeat from the leader.
    Heartbeat {
        /// The leader's id.
        leader_id: u64,
        /// The current term.
        term: u64,
    },
}

/// A single node participating in leader election.
#[derive(Debug, Clone)]
pub struct ElectionNode {
    /// This node's unique identifier.
    pub id: u64,
    /// Current role in the cluster.
    pub role: NodeRole,
    /// The id of the node this node considers the current leader.
    pub current_leader: Option<u64>,
    /// Monotonically increasing election term.
    pub term: u64,
    /// Number of votes/alive-messages received in the current election.
    pub votes_received: u64,
    /// Whether an Alive message was received, causing this node to yield.
    pub alive_received: bool,
}

impl ElectionNode {
    /// Create a new follower node with the given id.
    pub fn new(id: u64) -> Self {
        Self {
            id,
            role: NodeRole::Follower,
            current_leader: None,
            term: 0,
            votes_received: 0,
            alive_received: false,
        }
    }
}

/// A simulated cluster for Bully algorithm leader election.
pub struct ElectionCluster {
    nodes: HashMap<u64, ElectionNode>,
}

impl ElectionCluster {
    /// Create a new empty cluster.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Add a new node with the given id to the cluster.
    ///
    /// If a node with this id already exists, this is a no-op.
    pub fn add_node(&mut self, id: u64) {
        self.nodes
            .entry(id)
            .or_insert_with(|| ElectionNode::new(id));
    }

    /// Remove a node (simulate failure).
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.remove(&id);
    }

    /// Return the total number of nodes in the cluster.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the id of the node currently acting as leader, if any.
    pub fn current_leader(&self) -> Option<u64> {
        self.nodes
            .values()
            .find(|n| n.role == NodeRole::Leader)
            .map(|n| n.id)
    }

    /// Return `true` if there is exactly one leader and all other nodes are followers.
    pub fn is_stable(&self) -> bool {
        let leader_count = self
            .nodes
            .values()
            .filter(|n| n.role == NodeRole::Leader)
            .count();
        let non_leader_count = self
            .nodes
            .values()
            .filter(|n| n.role != NodeRole::Leader && n.role != NodeRole::Follower)
            .count();
        leader_count == 1 && non_leader_count == 0
    }

    /// Directly declare a node as leader and all others as followers.
    ///
    /// This is used internally and in tests to set a known initial state.
    pub fn declare_leader(&mut self, node_id: u64) {
        for (id, node) in &mut self.nodes {
            if *id == node_id {
                node.role = NodeRole::Leader;
                node.current_leader = Some(node_id);
            } else {
                node.role = NodeRole::Follower;
                node.current_leader = Some(node_id);
            }
        }
    }

    /// Start a Bully election initiated by `initiator_id`.
    ///
    /// Returns the sequence of `ElectionMessage` values produced.
    pub fn start_election(&mut self, initiator_id: u64) -> Vec<ElectionMessage> {
        let mut messages = Vec::new();

        if !self.nodes.contains_key(&initiator_id) {
            return messages;
        }

        // Increment the initiator's term and set role to Candidate
        if let Some(node) = self.nodes.get_mut(&initiator_id) {
            node.term += 1;
            node.role = NodeRole::Candidate;
            node.alive_received = false;
            node.votes_received = 0;
        }

        // Collect all node ids with higher ids than initiator
        let higher_ids: Vec<u64> = self
            .nodes
            .keys()
            .copied()
            .filter(|&id| id > initiator_id)
            .collect();

        if higher_ids.is_empty() {
            // Initiator is the highest; it wins immediately
            messages.push(ElectionMessage::Victory {
                leader_id: initiator_id,
            });
            self.declare_leader(initiator_id);
            return messages;
        }

        // Send Election messages to all higher-id nodes
        for &higher_id in &higher_ids {
            messages.push(ElectionMessage::Election {
                from_id: initiator_id,
            });
            // Higher nodes respond with Alive and then run their own election
            if self.nodes.contains_key(&higher_id) {
                messages.push(ElectionMessage::Alive { from_id: higher_id });
                if let Some(initiator) = self.nodes.get_mut(&initiator_id) {
                    initiator.alive_received = true;
                }
            }
        }

        // All higher nodes that are present also participate; the highest wins
        let winning_id = higher_ids
            .iter()
            .copied()
            .filter(|id| self.nodes.contains_key(id))
            .max()
            .unwrap_or(initiator_id);

        messages.push(ElectionMessage::Victory {
            leader_id: winning_id,
        });
        self.declare_leader(winning_id);

        messages
    }

    /// Process a batch of messages and return any new messages generated.
    pub fn process_messages(&mut self, messages: Vec<ElectionMessage>) -> Vec<ElectionMessage> {
        let mut outgoing = Vec::new();

        for msg in messages {
            match msg {
                ElectionMessage::Election { from_id } => {
                    // All nodes with id > from_id send Alive
                    let responders: Vec<u64> = self
                        .nodes
                        .keys()
                        .copied()
                        .filter(|&id| id > from_id)
                        .collect();
                    for responder in responders {
                        outgoing.push(ElectionMessage::Alive { from_id: responder });
                    }
                }
                ElectionMessage::Alive { from_id: _ } => {
                    // Handled by start_election logic; no further messages needed here
                }
                ElectionMessage::Victory { leader_id } => {
                    self.declare_leader(leader_id);
                }
                ElectionMessage::Heartbeat { leader_id, term } => {
                    // Update all followers' view of the leader
                    for node in self.nodes.values_mut() {
                        if node.id != leader_id {
                            node.current_leader = Some(leader_id);
                            node.term = term.max(node.term);
                        }
                    }
                }
            }
        }

        outgoing
    }

    /// Simulate a full Bully election and return the winning node id.
    ///
    /// The node with the highest id among all nodes wins.
    pub fn simulate_election(&mut self) -> Option<u64> {
        if self.nodes.is_empty() {
            return None;
        }

        // In the Bully algorithm, the highest-id node that is alive always wins.
        let winner = self.nodes.keys().copied().max()?;
        self.declare_leader(winner);
        Some(winner)
    }

    /// Return an immutable reference to a node by id.
    pub fn get_node(&self, id: u64) -> Option<&ElectionNode> {
        self.nodes.get(&id)
    }

    /// Send a heartbeat from the current leader to all followers.
    ///
    /// Returns the heartbeat messages generated (one per follower).
    pub fn send_heartbeats(&mut self) -> Vec<ElectionMessage> {
        let Some(leader_id) = self.current_leader() else {
            return Vec::new();
        };
        let term = self.nodes.get(&leader_id).map(|n| n.term).unwrap_or(0);
        let follower_count = self
            .nodes
            .values()
            .filter(|n| n.role == NodeRole::Follower)
            .count();
        vec![ElectionMessage::Heartbeat { leader_id, term }; follower_count]
    }
}

impl Default for ElectionCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster_with_nodes(ids: &[u64]) -> ElectionCluster {
        let mut c = ElectionCluster::new();
        for &id in ids {
            c.add_node(id);
        }
        c
    }

    // ── add_node / remove_node ─────────────────────────────────────────────────

    #[test]
    fn test_add_node() {
        let mut c = ElectionCluster::new();
        c.add_node(1);
        assert_eq!(c.node_count(), 1);
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut c = ElectionCluster::new();
        c.add_node(1);
        c.add_node(1); // duplicate — no-op
        assert_eq!(c.node_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.remove_node(2);
        assert_eq!(c.node_count(), 2);
        assert!(c.get_node(2).is_none());
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mut c = cluster_with_nodes(&[1, 2]);
        c.remove_node(99); // no panic
        assert_eq!(c.node_count(), 2);
    }

    // ── declare_leader ────────────────────────────────────────────────────────

    #[test]
    fn test_declare_leader_sets_roles() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.declare_leader(3);
        assert_eq!(c.get_node(3).unwrap().role, NodeRole::Leader);
        assert_eq!(c.get_node(1).unwrap().role, NodeRole::Follower);
        assert_eq!(c.get_node(2).unwrap().role, NodeRole::Follower);
    }

    #[test]
    fn test_declare_leader_updates_current_leader() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.declare_leader(2);
        for id in [1u64, 2, 3] {
            assert_eq!(c.get_node(id).unwrap().current_leader, Some(2));
        }
    }

    // ── current_leader / is_stable ────────────────────────────────────────────

    #[test]
    fn test_current_leader_none_initially() {
        let c = cluster_with_nodes(&[1, 2, 3]);
        assert!(c.current_leader().is_none());
    }

    #[test]
    fn test_is_stable_true_after_declare() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.declare_leader(3);
        assert!(c.is_stable());
    }

    #[test]
    fn test_is_stable_false_when_no_leader() {
        let c = cluster_with_nodes(&[1, 2, 3]);
        assert!(!c.is_stable());
    }

    // ── start_election ────────────────────────────────────────────────────────

    #[test]
    fn test_start_election_single_node_wins() {
        let mut c = cluster_with_nodes(&[5]);
        let msgs = c.start_election(5);
        assert!(msgs
            .iter()
            .any(|m| matches!(m, ElectionMessage::Victory { leader_id: 5 })));
        assert_eq!(c.current_leader(), Some(5));
    }

    #[test]
    fn test_start_election_highest_wins() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.start_election(1);
        assert_eq!(c.current_leader(), Some(3));
    }

    #[test]
    fn test_start_election_from_middle() {
        let mut c = cluster_with_nodes(&[1, 2, 3, 4, 5]);
        c.start_election(3);
        assert_eq!(c.current_leader(), Some(5));
    }

    #[test]
    fn test_start_election_from_highest() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        let msgs = c.start_election(3);
        assert!(msgs
            .iter()
            .any(|m| matches!(m, ElectionMessage::Victory { leader_id: 3 })));
        assert_eq!(c.current_leader(), Some(3));
    }

    #[test]
    fn test_start_election_missing_initiator() {
        let mut c = cluster_with_nodes(&[1, 2]);
        let msgs = c.start_election(99); // 99 is not in cluster
        assert!(msgs.is_empty());
    }

    // ── process_messages ──────────────────────────────────────────────────────

    #[test]
    fn test_process_election_message_generates_alive() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        let outgoing = c.process_messages(vec![ElectionMessage::Election { from_id: 1 }]);
        // Nodes 2 and 3 should respond with Alive
        assert!(outgoing
            .iter()
            .any(|m| matches!(m, ElectionMessage::Alive { from_id: 2 })));
        assert!(outgoing
            .iter()
            .any(|m| matches!(m, ElectionMessage::Alive { from_id: 3 })));
    }

    #[test]
    fn test_process_victory_sets_leader() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.process_messages(vec![ElectionMessage::Victory { leader_id: 3 }]);
        assert_eq!(c.current_leader(), Some(3));
    }

    #[test]
    fn test_process_heartbeat_updates_term() {
        let mut c = cluster_with_nodes(&[1, 2]);
        c.declare_leader(2);
        c.process_messages(vec![ElectionMessage::Heartbeat {
            leader_id: 2,
            term: 10,
        }]);
        assert_eq!(c.get_node(1).unwrap().term, 10);
    }

    // ── simulate_election ─────────────────────────────────────────────────────

    #[test]
    fn test_simulate_election_empty_cluster() {
        let mut c = ElectionCluster::new();
        assert!(c.simulate_election().is_none());
    }

    #[test]
    fn test_simulate_election_single_node() {
        let mut c = cluster_with_nodes(&[42]);
        assert_eq!(c.simulate_election(), Some(42));
    }

    #[test]
    fn test_simulate_election_highest_wins() {
        let mut c = cluster_with_nodes(&[3, 7, 1, 9, 5]);
        let winner = c.simulate_election();
        assert_eq!(winner, Some(9));
        assert_eq!(c.current_leader(), Some(9));
        assert!(c.is_stable());
    }

    #[test]
    fn test_simulate_election_two_nodes() {
        let mut c = cluster_with_nodes(&[10, 20]);
        assert_eq!(c.simulate_election(), Some(20));
    }

    // ── Leader failure and re-election ────────────────────────────────────────

    #[test]
    fn test_leader_failure_triggers_re_election() {
        let mut c = cluster_with_nodes(&[1, 2, 3, 4, 5]);
        c.declare_leader(5);
        assert_eq!(c.current_leader(), Some(5));
        // Leader fails
        c.remove_node(5);
        // Remaining: 1, 2, 3, 4 — re-elect
        let new_winner = c.simulate_election();
        assert_eq!(new_winner, Some(4));
    }

    #[test]
    fn test_multiple_failures_re_election() {
        let mut c = cluster_with_nodes(&[1, 2, 3, 4, 5]);
        c.declare_leader(5);
        c.remove_node(5);
        c.remove_node(4);
        let winner = c.simulate_election();
        assert_eq!(winner, Some(3));
    }

    // ── Adding nodes mid-cluster ───────────────────────────────────────────────

    #[test]
    fn test_add_higher_node_mid_cluster() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.declare_leader(3);
        assert_eq!(c.current_leader(), Some(3));
        // Add a higher-id node
        c.add_node(10);
        // Re-elect
        let winner = c.simulate_election();
        assert_eq!(winner, Some(10));
    }

    #[test]
    fn test_add_lower_node_does_not_change_leader() {
        let mut c = cluster_with_nodes(&[5, 6, 7]);
        c.declare_leader(7);
        c.add_node(1); // lower id
        let winner = c.simulate_election();
        assert_eq!(winner, Some(7)); // still the highest
    }

    // ── send_heartbeats ───────────────────────────────────────────────────────

    #[test]
    fn test_send_heartbeats_from_leader() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        c.declare_leader(3);
        let hbs = c.send_heartbeats();
        // 2 followers → 2 heartbeat messages
        assert_eq!(hbs.len(), 2);
        assert!(hbs
            .iter()
            .all(|m| matches!(m, ElectionMessage::Heartbeat { leader_id: 3, .. })));
    }

    #[test]
    fn test_send_heartbeats_no_leader() {
        let mut c = cluster_with_nodes(&[1, 2, 3]);
        let hbs = c.send_heartbeats();
        assert!(hbs.is_empty());
    }

    // ── NodeRole / ElectionNode ───────────────────────────────────────────────

    #[test]
    fn test_node_role_clone_eq() {
        assert_eq!(NodeRole::Follower, NodeRole::Follower);
        assert_ne!(NodeRole::Follower, NodeRole::Leader);
        let r = NodeRole::Candidate;
        assert_eq!(r.clone(), NodeRole::Candidate);
    }

    #[test]
    fn test_election_node_initial_state() {
        let node = ElectionNode::new(42);
        assert_eq!(node.id, 42);
        assert_eq!(node.role, NodeRole::Follower);
        assert!(node.current_leader.is_none());
        assert_eq!(node.term, 0);
        assert_eq!(node.votes_received, 0);
        assert!(!node.alive_received);
    }

    // ── Default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_cluster_default() {
        let c = ElectionCluster::default();
        assert_eq!(c.node_count(), 0);
        assert!(c.current_leader().is_none());
    }
}
