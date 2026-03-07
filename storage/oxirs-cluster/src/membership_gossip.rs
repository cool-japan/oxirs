//! Gossip Protocol for Cluster Membership Management.
//!
//! Implements a Gossip-based failure detector and membership protocol
//! following patterns from SWIM (Scalable Weakly-consistent Infection-style
//! process group Membership protocol).
//!
//! ## Overview
//!
//! - Nodes are identified by a `String` node ID and have a string `address`.
//! - Health states: **Alive**, **Suspected** (not recently heard from), **Dead** (confirmed).
//! - On each gossip round a random subset of `fanout` peers is selected to
//!   exchange membership state with.
//! - State merges follow a "highest heartbeat wins" rule; state transitions
//!   only move forward (`Alive → Suspected → Dead`).
//!
//! ## Example
//!
//! ```rust
//! use oxirs_cluster::membership_gossip::{GossipProtocol, NodeState};
//!
//! let mut gossip = GossipProtocol::new("node-1", "127.0.0.1:7777");
//! gossip.join("node-2", "127.0.0.1:7778");
//! gossip.join("node-3", "127.0.0.1:7779");
//!
//! gossip.heartbeat("node-2");
//! let alive = gossip.alive_members();
//! assert!(alive.len() >= 2);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in gossip operations.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum GossipError {
    /// The node ID was not found in the membership list.
    #[error("Node '{0}' not found in membership list")]
    NodeNotFound(String),

    /// The node ID is already present.
    #[error("Node '{0}' is already in the membership list")]
    DuplicateNode(String),

    /// The fanout value must be at least 1.
    #[error("Fanout must be at least 1 (got {0})")]
    InvalidFanout(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeState
// ─────────────────────────────────────────────────────────────────────────────

/// The reachability state of a node in the membership list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub enum NodeState {
    /// Node is responding normally.
    Alive,
    /// Node has not been heard from recently; health is uncertain.
    Suspected,
    /// Node has been confirmed unreachable.
    Dead,
}

impl NodeState {
    /// Returns `true` if the node is in the `Alive` state.
    pub fn is_alive(self) -> bool {
        self == NodeState::Alive
    }

    /// Returns `true` if the node is in the `Suspected` state.
    pub fn is_suspected(self) -> bool {
        self == NodeState::Suspected
    }

    /// Returns `true` if the node is in the `Dead` state.
    pub fn is_dead(self) -> bool {
        self == NodeState::Dead
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeState::Alive => write!(f, "Alive"),
            NodeState::Suspected => write!(f, "Suspected"),
            NodeState::Dead => write!(f, "Dead"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GossipNode
// ─────────────────────────────────────────────────────────────────────────────

/// A member node in the gossip membership ring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipNode {
    /// Unique node identifier.
    pub node_id: String,
    /// Network address (e.g. `"127.0.0.1:7777"`).
    pub address: String,
    /// Current state.
    pub state: NodeState,
    /// Monotonically increasing heartbeat counter (incremented by the node
    /// itself on each gossip cycle).
    pub heartbeat: u64,
    /// Timestamp (milliseconds since epoch) of the last time this node was
    /// heard from.
    pub last_seen: u64,
}

impl GossipNode {
    fn new(node_id: impl Into<String>, address: impl Into<String>, now_ms: u64) -> Self {
        Self {
            node_id: node_id.into(),
            address: address.into(),
            state: NodeState::Alive,
            heartbeat: 0,
            last_seen: now_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GossipMessage
// ─────────────────────────────────────────────────────────────────────────────

/// Payload exchanged between nodes during a gossip round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// The node that generated this message.
    pub from: String,
    /// Snapshot of the sender's membership view.
    pub members: Vec<GossipNode>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic pseudo-random subset selection
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal xorshift64 PRNG used internally to avoid a `rand` dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef_cafebabe } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Select `k` unique indices from `[0, n)` using partial Fisher-Yates.
    fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k == 0 || n == 0 {
            return vec![];
        }
        let k = k.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next() as usize % (n - i));
            indices.swap(i, j);
        }
        indices[..k].to_vec()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GossipProtocol
// ─────────────────────────────────────────────────────────────────────────────

/// The gossip protocol engine for cluster membership management.
///
/// Maintains the local view of all known cluster members and drives the
/// gossip dissemination / failure-detection cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipProtocol {
    /// This node's own ID.
    pub local_node_id: String,
    /// Membership list keyed by node ID.
    members: HashMap<String, GossipNode>,
    /// Internal PRNG seed for gossip fan-out selection.
    rng_seed: u64,
    /// Round counter (used to vary the PRNG seed).
    round: u64,
}

impl GossipProtocol {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new gossip protocol instance for the given local node.
    ///
    /// The local node is automatically added as an `Alive` member.
    pub fn new(node_id: impl Into<String>, address: impl Into<String>) -> Self {
        let now_ms = monotonic_ms();
        let id_str = node_id.into();
        let mut members = HashMap::new();
        members.insert(
            id_str.clone(),
            GossipNode::new(id_str.clone(), address, now_ms),
        );
        Self {
            local_node_id: id_str,
            members,
            rng_seed: 0xcafe_babe_dead_beef,
            round: 0,
        }
    }

    /// Create with an explicit seed for the internal PRNG (useful in tests).
    pub fn with_seed(node_id: impl Into<String>, address: impl Into<String>, seed: u64) -> Self {
        let mut g = Self::new(node_id, address);
        g.rng_seed = seed;
        g
    }

    // ── Membership management ────────────────────────────────────────────────

    /// Add a new node to the membership list as `Alive`.
    ///
    /// Returns `Err(GossipError::DuplicateNode)` if the node is already known.
    pub fn join(&mut self, node_id: &str, address: &str) -> Result<(), GossipError> {
        if self.members.contains_key(node_id) {
            return Err(GossipError::DuplicateNode(node_id.to_owned()));
        }
        let now_ms = monotonic_ms();
        self.members.insert(
            node_id.to_owned(),
            GossipNode::new(node_id, address, now_ms),
        );
        Ok(())
    }

    /// Mark a node as `Suspected`.
    ///
    /// If the node is already `Dead`, the call is a no-op (state only moves
    /// forward). Returns `Err` if the node is not found.
    pub fn suspect(&mut self, node_id: &str) -> Result<(), GossipError> {
        let node = self
            .members
            .get_mut(node_id)
            .ok_or_else(|| GossipError::NodeNotFound(node_id.to_owned()))?;
        if node.state == NodeState::Alive {
            node.state = NodeState::Suspected;
        }
        Ok(())
    }

    /// Confirm that a node is `Dead`.
    ///
    /// Returns `Err` if the node is not found.
    pub fn confirm_dead(&mut self, node_id: &str) -> Result<(), GossipError> {
        let node = self
            .members
            .get_mut(node_id)
            .ok_or_else(|| GossipError::NodeNotFound(node_id.to_owned()))?;
        node.state = NodeState::Dead;
        Ok(())
    }

    /// Update the heartbeat counter and `last_seen` timestamp for a node.
    ///
    /// Bumps the heartbeat by 1 and resets the state to `Alive` if the node
    /// was `Suspected`. Returns `Err` if the node is not found.
    pub fn heartbeat(&mut self, node_id: &str) -> Result<u64, GossipError> {
        let now_ms = monotonic_ms();
        let node = self
            .members
            .get_mut(node_id)
            .ok_or_else(|| GossipError::NodeNotFound(node_id.to_owned()))?;
        node.heartbeat = node.heartbeat.saturating_add(1);
        node.last_seen = now_ms;
        if node.state == NodeState::Suspected {
            node.state = NodeState::Alive;
        }
        Ok(node.heartbeat)
    }

    /// Update heartbeat using a caller-supplied timestamp (useful in tests).
    pub fn heartbeat_at(&mut self, node_id: &str, now_ms: u64) -> Result<u64, GossipError> {
        let node = self
            .members
            .get_mut(node_id)
            .ok_or_else(|| GossipError::NodeNotFound(node_id.to_owned()))?;
        node.heartbeat = node.heartbeat.saturating_add(1);
        node.last_seen = now_ms;
        if node.state == NodeState::Suspected {
            node.state = NodeState::Alive;
        }
        Ok(node.heartbeat)
    }

    // ── Gossip dissemination ─────────────────────────────────────────────────

    /// Perform a gossip round: select up to `fanout` random peer nodes and
    /// generate a [`GossipMessage`] for each.
    ///
    /// The returned messages should be sent to the corresponding peers (by
    /// caller); each message contains the current node's full membership view.
    pub fn gossip_round(&mut self, fanout: usize) -> Result<Vec<GossipMessage>, GossipError> {
        if fanout == 0 {
            return Err(GossipError::InvalidFanout(fanout));
        }
        self.round = self.round.wrapping_add(1);
        let seed = self.rng_seed ^ (self.round.wrapping_mul(0x517cc1b727220a95));
        let mut rng = Xorshift64::new(seed);

        // Collect peers (everyone except self and Dead nodes)
        let peers: Vec<String> = self
            .members
            .iter()
            .filter(|(id, node)| {
                id.as_str() != self.local_node_id.as_str() && !node.state.is_dead()
            })
            .map(|(id, _)| id.clone())
            .collect();

        let n = peers.len();
        let selected_indices = rng.sample_indices(n, fanout.min(n));

        let snapshot: Vec<GossipNode> = self.members.values().cloned().collect();
        let from = self.local_node_id.clone();

        let messages = selected_indices
            .into_iter()
            .map(|_| GossipMessage {
                from: from.clone(),
                members: snapshot.clone(),
            })
            .collect();

        Ok(messages)
    }

    /// Merge a remote [`GossipMessage`] into the local membership view.
    ///
    /// Merging rules:
    /// - New nodes are added as received.
    /// - For existing nodes, the record with the higher heartbeat wins.
    /// - State only advances forward (`Alive → Suspected → Dead`).
    pub fn merge_state(&mut self, msg: &GossipMessage) {
        for remote in &msg.members {
            if remote.node_id == self.local_node_id {
                // Do not let remote override local node's own state
                continue;
            }
            match self.members.get_mut(&remote.node_id) {
                Some(local) => {
                    // Higher heartbeat wins
                    if remote.heartbeat > local.heartbeat {
                        local.heartbeat = remote.heartbeat;
                        local.last_seen = remote.last_seen;
                        // State can only advance (not retreat)
                        if remote.state > local.state {
                            local.state = remote.state;
                        }
                    } else if remote.heartbeat == local.heartbeat {
                        // Same heartbeat: advance state if needed
                        if remote.state > local.state {
                            local.state = remote.state;
                        }
                    }
                }
                None => {
                    // Unknown node: add it
                    self.members.insert(remote.node_id.clone(), remote.clone());
                }
            }
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Return all `Alive` member nodes.
    pub fn alive_members(&self) -> Vec<&GossipNode> {
        self.members
            .values()
            .filter(|n| n.state.is_alive())
            .collect()
    }

    /// Return all `Suspected` member nodes.
    pub fn suspected_members(&self) -> Vec<&GossipNode> {
        self.members
            .values()
            .filter(|n| n.state.is_suspected())
            .collect()
    }

    /// Return all `Dead` member nodes.
    pub fn dead_members(&self) -> Vec<&GossipNode> {
        self.members
            .values()
            .filter(|n| n.state.is_dead())
            .collect()
    }

    /// Return all members regardless of state.
    pub fn all_members(&self) -> Vec<&GossipNode> {
        self.members.values().collect()
    }

    /// Get a specific node by ID.
    pub fn get_node(&self, node_id: &str) -> Option<&GossipNode> {
        self.members.get(node_id)
    }

    /// Total number of known nodes.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    // ── Failure detection ─────────────────────────────────────────────────────

    /// Scan the membership list for nodes that have not been heard from within
    /// `timeout_ms` milliseconds and transition them to `Suspected`.
    ///
    /// `now_ms` is the current time in milliseconds since epoch.
    pub fn detect_failures(&mut self, timeout_ms: u64, now_ms: u64) {
        let local_id = self.local_node_id.clone();
        for node in self.members.values_mut() {
            if node.node_id == local_id {
                continue;
            }
            if node.state == NodeState::Alive && now_ms.saturating_sub(node.last_seen) > timeout_ms
            {
                node.state = NodeState::Suspected;
            }
        }
    }

    /// Scan the membership list for nodes that have been `Suspected` for more
    /// than `dead_timeout_ms` milliseconds and confirm them as `Dead`.
    pub fn confirm_dead_stale(&mut self, dead_timeout_ms: u64, now_ms: u64) {
        let local_id = self.local_node_id.clone();
        for node in self.members.values_mut() {
            if node.node_id == local_id {
                continue;
            }
            if node.state == NodeState::Suspected
                && now_ms.saturating_sub(node.last_seen) > dead_timeout_ms
            {
                node.state = NodeState::Dead;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn monotonic_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gossip() -> GossipProtocol {
        GossipProtocol::with_seed("n1", "127.0.0.1:7001", 12345)
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_creates_local_node() {
        let g = make_gossip();
        assert_eq!(g.member_count(), 1);
        let local = g.get_node("n1").unwrap();
        assert!(local.state.is_alive());
    }

    #[test]
    fn test_local_node_id_stored() {
        let g = make_gossip();
        assert_eq!(g.local_node_id, "n1");
    }

    // ── Join ────────────────────────────────────────────────────────────────

    #[test]
    fn test_join_adds_alive_node() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        assert_eq!(g.member_count(), 2);
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_alive());
        assert_eq!(n2.address, "127.0.0.1:7002");
    }

    #[test]
    fn test_join_duplicate_fails() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        let err = g.join("n2", "127.0.0.1:7002").unwrap_err();
        assert!(matches!(err, GossipError::DuplicateNode(_)));
    }

    #[test]
    fn test_join_multiple_nodes() {
        let mut g = make_gossip();
        for i in 2..=5 {
            g.join(&format!("n{i}"), &format!("127.0.0.1:{}", 7000 + i))
                .unwrap();
        }
        assert_eq!(g.member_count(), 5);
    }

    // ── Suspect ─────────────────────────────────────────────────────────────

    #[test]
    fn test_suspect_transitions_to_suspected() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.suspect("n2").unwrap();
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_suspected());
    }

    #[test]
    fn test_suspect_dead_node_stays_dead() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.confirm_dead("n2").unwrap();
        g.suspect("n2").unwrap(); // should be no-op
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_dead());
    }

    #[test]
    fn test_suspect_nonexistent_node_error() {
        let mut g = make_gossip();
        let err = g.suspect("nope").unwrap_err();
        assert!(matches!(err, GossipError::NodeNotFound(_)));
    }

    // ── Confirm dead ─────────────────────────────────────────────────────────

    #[test]
    fn test_confirm_dead() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.confirm_dead("n2").unwrap();
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_dead());
    }

    #[test]
    fn test_confirm_dead_nonexistent_error() {
        let mut g = make_gossip();
        let err = g.confirm_dead("nope").unwrap_err();
        assert!(matches!(err, GossipError::NodeNotFound(_)));
    }

    // ── Heartbeat ─────────────────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_increments_counter() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        let hb1 = g.heartbeat("n2").unwrap();
        let hb2 = g.heartbeat("n2").unwrap();
        assert_eq!(hb2, hb1 + 1);
    }

    #[test]
    fn test_heartbeat_resets_suspected_to_alive() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.suspect("n2").unwrap();
        assert!(g.get_node("n2").unwrap().state.is_suspected());
        g.heartbeat("n2").unwrap();
        assert!(g.get_node("n2").unwrap().state.is_alive());
    }

    #[test]
    fn test_heartbeat_nonexistent_node_error() {
        let mut g = make_gossip();
        let err = g.heartbeat("nope").unwrap_err();
        assert!(matches!(err, GossipError::NodeNotFound(_)));
    }

    // ── Gossip round ─────────────────────────────────────────────────────────

    #[test]
    fn test_gossip_round_zero_fanout_error() {
        let mut g = make_gossip();
        let err = g.gossip_round(0).unwrap_err();
        assert!(matches!(err, GossipError::InvalidFanout(0)));
    }

    #[test]
    fn test_gossip_round_no_peers_empty_messages() {
        let mut g = make_gossip();
        // Only local node; no peers to gossip with
        let msgs = g.gossip_round(3).unwrap();
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_gossip_round_selects_up_to_fanout_peers() {
        let mut g = make_gossip();
        for i in 2..=6 {
            g.join(&format!("n{i}"), &format!("127.0.0.1:{}", 7000 + i))
                .unwrap();
        }
        let msgs = g.gossip_round(3).unwrap();
        assert!(msgs.len() <= 3);
        assert!(!msgs.is_empty());
    }

    #[test]
    fn test_gossip_round_message_contains_members() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        let msgs = g.gossip_round(1).unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(!msgs[0].members.is_empty());
        assert_eq!(msgs[0].from, "n1");
    }

    #[test]
    fn test_gossip_round_fanout_larger_than_peers() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        // fanout=10, but only 1 peer
        let msgs = g.gossip_round(10).unwrap();
        assert_eq!(msgs.len(), 1);
    }

    // ── Merge state ───────────────────────────────────────────────────────────

    #[test]
    fn test_merge_adds_unknown_node() {
        let mut g = make_gossip();
        let msg = GossipMessage {
            from: "n2".to_owned(),
            members: vec![GossipNode {
                node_id: "n2".to_owned(),
                address: "127.0.0.1:7002".to_owned(),
                state: NodeState::Alive,
                heartbeat: 5,
                last_seen: 1000,
            }],
        };
        g.merge_state(&msg);
        let n2 = g.get_node("n2").unwrap();
        assert_eq!(n2.heartbeat, 5);
        assert!(n2.state.is_alive());
    }

    #[test]
    fn test_merge_higher_heartbeat_wins() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.heartbeat("n2").unwrap(); // heartbeat = 1

        let msg = GossipMessage {
            from: "n3".to_owned(),
            members: vec![GossipNode {
                node_id: "n2".to_owned(),
                address: "127.0.0.1:7002".to_owned(),
                state: NodeState::Alive,
                heartbeat: 10,
                last_seen: 9999,
            }],
        };
        g.merge_state(&msg);
        let n2 = g.get_node("n2").unwrap();
        assert_eq!(n2.heartbeat, 10);
    }

    #[test]
    fn test_merge_lower_heartbeat_ignored() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        for _ in 0..5 {
            g.heartbeat("n2").unwrap();
        }
        // heartbeat now = 5
        let msg = GossipMessage {
            from: "n3".to_owned(),
            members: vec![GossipNode {
                node_id: "n2".to_owned(),
                address: "127.0.0.1:7002".to_owned(),
                state: NodeState::Alive,
                heartbeat: 2, // lower
                last_seen: 0,
            }],
        };
        g.merge_state(&msg);
        let n2 = g.get_node("n2").unwrap();
        assert_eq!(n2.heartbeat, 5); // not overwritten
    }

    #[test]
    fn test_merge_does_not_override_local_node() {
        let mut g = make_gossip();
        let msg = GossipMessage {
            from: "n2".to_owned(),
            members: vec![GossipNode {
                node_id: "n1".to_owned(), // local node!
                address: "evil.example.com:9999".to_owned(),
                state: NodeState::Dead,
                heartbeat: 9999,
                last_seen: 9999,
            }],
        };
        g.merge_state(&msg);
        let local = g.get_node("n1").unwrap();
        // Local node should still be alive
        assert!(local.state.is_alive());
        assert_ne!(local.address, "evil.example.com:9999");
    }

    #[test]
    fn test_merge_state_advances_state() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        let msg = GossipMessage {
            from: "n3".to_owned(),
            members: vec![GossipNode {
                node_id: "n2".to_owned(),
                address: "127.0.0.1:7002".to_owned(),
                state: NodeState::Suspected,
                heartbeat: 0,
                last_seen: 0,
            }],
        };
        g.merge_state(&msg);
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_suspected());
    }

    // ── Alive members query ──────────────────────────────────────────────────

    #[test]
    fn test_alive_members_includes_local() {
        let g = make_gossip();
        let alive = g.alive_members();
        assert_eq!(alive.len(), 1);
        assert_eq!(alive[0].node_id, "n1");
    }

    #[test]
    fn test_alive_members_excludes_suspected() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.suspect("n2").unwrap();
        let alive = g.alive_members();
        assert!(!alive.iter().any(|n| n.node_id == "n2"));
    }

    #[test]
    fn test_alive_members_excludes_dead() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.confirm_dead("n2").unwrap();
        let alive = g.alive_members();
        assert!(!alive.iter().any(|n| n.node_id == "n2"));
    }

    // ── Failure detection ─────────────────────────────────────────────────────

    #[test]
    fn test_detect_failures_suspects_stale_node() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        // Set last_seen to 0 (far in the past)
        g.members.get_mut("n2").unwrap().last_seen = 0;
        // Detect failures with a 1ms timeout at time 1_000_000 ms
        g.detect_failures(1, 1_000_000);
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_suspected());
    }

    #[test]
    fn test_detect_failures_ignores_local_node() {
        let mut g = make_gossip();
        g.members.get_mut("n1").unwrap().last_seen = 0;
        g.detect_failures(1, 1_000_000);
        let local = g.get_node("n1").unwrap();
        assert!(local.state.is_alive());
    }

    #[test]
    fn test_detect_failures_recent_node_stays_alive() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        let now = monotonic_ms();
        g.members.get_mut("n2").unwrap().last_seen = now;
        g.detect_failures(60_000, now);
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_alive());
    }

    #[test]
    fn test_confirm_dead_stale_transitions_suspected_to_dead() {
        let mut g = make_gossip();
        g.join("n2", "127.0.0.1:7002").unwrap();
        g.suspect("n2").unwrap();
        g.members.get_mut("n2").unwrap().last_seen = 0;
        g.confirm_dead_stale(1, 1_000_000);
        let n2 = g.get_node("n2").unwrap();
        assert!(n2.state.is_dead());
    }

    // ── NodeState helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_node_state_is_alive() {
        assert!(NodeState::Alive.is_alive());
        assert!(!NodeState::Suspected.is_alive());
        assert!(!NodeState::Dead.is_alive());
    }

    #[test]
    fn test_node_state_is_suspected() {
        assert!(!NodeState::Alive.is_suspected());
        assert!(NodeState::Suspected.is_suspected());
        assert!(!NodeState::Dead.is_suspected());
    }

    #[test]
    fn test_node_state_is_dead() {
        assert!(!NodeState::Alive.is_dead());
        assert!(!NodeState::Suspected.is_dead());
        assert!(NodeState::Dead.is_dead());
    }

    #[test]
    fn test_node_state_display() {
        assert_eq!(format!("{}", NodeState::Alive), "Alive");
        assert_eq!(format!("{}", NodeState::Suspected), "Suspected");
        assert_eq!(format!("{}", NodeState::Dead), "Dead");
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_node_not_found() {
        let err = GossipError::NodeNotFound("x".to_owned());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn test_error_duplicate_node() {
        let err = GossipError::DuplicateNode("n2".to_owned());
        assert!(err.to_string().contains("n2"));
    }

    #[test]
    fn test_error_invalid_fanout() {
        let err = GossipError::InvalidFanout(0);
        assert!(err.to_string().contains("0"));
    }
}
