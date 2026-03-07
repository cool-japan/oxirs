//! Cluster node health monitoring with heartbeat tracking.
//!
//! A [`NodeMonitor`] tracks the liveness of cluster nodes by recording
//! incoming heartbeats and timing out nodes that have been silent for longer
//! than a configurable `timeout_ms`.

use std::collections::HashMap;

// ── Node metadata ─────────────────────────────────────────────────────────────

/// Role a node plays in the cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeRole {
    Leader,
    Follower,
    Candidate,
    Observer,
}

/// Static information about a registered node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Unique identifier for this node.
    pub id: String,
    /// Network address (e.g. `"10.0.0.1:7000"`).
    pub address: String,
    /// Role of this node when it was registered.
    pub role: NodeRole,
    /// Unix timestamp (ms) when the node joined the cluster.
    pub joined_at: u64,
}

impl NodeInfo {
    /// Create a new NodeInfo.
    pub fn new(
        id: impl Into<String>,
        address: impl Into<String>,
        role: NodeRole,
        joined_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            role,
            joined_at,
        }
    }
}

// ── Node state ────────────────────────────────────────────────────────────────

/// Current health state of a node, updated by heartbeat arrival and timeout detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeState {
    /// Heartbeats are arriving within the timeout window.
    Alive,
    /// The node has missed at least one heartbeat interval — may be slow or partitioned.
    Suspected,
    /// The node has not been heard from for longer than `timeout_ms`.
    Dead,
}

// ── Heartbeat record ──────────────────────────────────────────────────────────

/// A single heartbeat receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct HeartbeatRecord {
    /// ID of the node that sent the heartbeat.
    pub node_id: String,
    /// Unix timestamp (ms) when the heartbeat was received.
    pub received_at: u64,
    /// Round-trip latency in milliseconds.
    pub latency_ms: u64,
}

// ── Monitor ───────────────────────────────────────────────────────────────────

/// Tracks node health based on heartbeat records and configurable timeouts.
pub struct NodeMonitor {
    nodes: HashMap<String, NodeInfo>,
    states: HashMap<String, NodeState>,
    heartbeats: HashMap<String, Vec<HeartbeatRecord>>,
    timeout_ms: u64,
    max_history: usize,
}

impl NodeMonitor {
    /// Create a new monitor.
    ///
    /// - `timeout_ms` — a node is considered dead when no heartbeat has been
    ///   received for this many milliseconds.
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            nodes: HashMap::new(),
            states: HashMap::new(),
            heartbeats: HashMap::new(),
            timeout_ms,
            max_history: 100,
        }
    }

    /// Set the maximum heartbeat history to retain per node.
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }

    /// Register a new node.  Overwrites any previous registration for the same ID.
    pub fn register(&mut self, node: NodeInfo) {
        self.states.insert(node.id.clone(), NodeState::Alive);
        self.heartbeats.entry(node.id.clone()).or_default();
        self.nodes.insert(node.id.clone(), node);
    }

    /// Record a heartbeat received from a node.
    ///
    /// Sets the node's state to `Alive` and appends the record to its history.
    /// Returns `false` if the node is not registered.
    pub fn record_heartbeat(&mut self, node_id: &str, received_at: u64, latency_ms: u64) -> bool {
        if !self.nodes.contains_key(node_id) {
            return false;
        }
        self.states.insert(node_id.to_string(), NodeState::Alive);
        let records = self.heartbeats.entry(node_id.to_string()).or_default();
        records.push(HeartbeatRecord {
            node_id: node_id.to_string(),
            received_at,
            latency_ms,
        });
        // Trim to max history.
        if records.len() > self.max_history {
            let drain_count = records.len() - self.max_history;
            records.drain(..drain_count);
        }
        true
    }

    /// Scan all registered nodes and mark those whose last heartbeat is older
    /// than `timeout_ms` as `Dead`.
    ///
    /// Returns the IDs of nodes that transitioned to `Dead` in this call.
    pub fn check_timeouts(&mut self, now: u64) -> Vec<String> {
        let mut timed_out = Vec::new();
        for (id, records) in &self.heartbeats {
            let last_seen = records.last().map(|r| r.received_at).unwrap_or(0);
            let elapsed = now.saturating_sub(last_seen);
            if elapsed >= self.timeout_ms {
                if self.states.get(id) != Some(&NodeState::Dead) {
                    timed_out.push(id.clone());
                }
            }
        }
        for id in &timed_out {
            self.states.insert(id.clone(), NodeState::Dead);
        }
        timed_out
    }

    /// Return the current state of a node.
    pub fn state(&self, node_id: &str) -> Option<&NodeState> {
        self.states.get(node_id)
    }

    /// Return references to all nodes whose state is `Alive`.
    pub fn alive_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes
            .values()
            .filter(|n| self.states.get(&n.id) == Some(&NodeState::Alive))
            .collect()
    }

    /// Return references to all nodes whose state is `Dead`.
    pub fn dead_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes
            .values()
            .filter(|n| self.states.get(&n.id) == Some(&NodeState::Dead))
            .collect()
    }

    /// Compute the average heartbeat latency for a node.
    ///
    /// Returns `None` if the node has no heartbeat history.
    pub fn avg_latency(&self, node_id: &str) -> Option<f64> {
        let records = self.heartbeats.get(node_id)?;
        if records.is_empty() {
            return None;
        }
        let sum: u64 = records.iter().map(|r| r.latency_ms).sum();
        Some(sum as f64 / records.len() as f64)
    }

    /// Total number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Deregister a node.  Returns `true` if it existed.
    pub fn remove(&mut self, node_id: &str) -> bool {
        if self.nodes.remove(node_id).is_some() {
            self.states.remove(node_id);
            self.heartbeats.remove(node_id);
            true
        } else {
            false
        }
    }

    /// Return the most recent heartbeat record for a node, if any.
    pub fn last_heartbeat(&self, node_id: &str) -> Option<&HeartbeatRecord> {
        self.heartbeats.get(node_id)?.last()
    }

    /// Number of heartbeat records stored for a node.
    pub fn heartbeat_count(&self, node_id: &str) -> usize {
        self.heartbeats.get(node_id).map_or(0, |v| v.len())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn follower(id: &str) -> NodeInfo {
        NodeInfo::new(id, "127.0.0.1:7000", NodeRole::Follower, 0)
    }

    fn monitor() -> NodeMonitor {
        NodeMonitor::new(5000) // 5-second timeout
    }

    // ── register / node_count ─────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let m = monitor();
        assert_eq!(m.node_count(), 0);
    }

    #[test]
    fn test_register_increments_count() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_register_multiple_nodes() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.register(follower("n2"));
        m.register(follower("n3"));
        assert_eq!(m.node_count(), 3);
    }

    #[test]
    fn test_register_overwrites_same_id() {
        let mut m = monitor();
        m.register(NodeInfo::new("n1", "addr1", NodeRole::Follower, 0));
        m.register(NodeInfo::new("n1", "addr2", NodeRole::Leader, 100));
        assert_eq!(m.node_count(), 1);
        assert_eq!(m.nodes["n1"].address, "addr2");
    }

    #[test]
    fn test_new_node_state_is_alive() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert_eq!(m.state("n1"), Some(&NodeState::Alive));
    }

    // ── record_heartbeat ──────────────────────────────────────────────────────

    #[test]
    fn test_record_heartbeat_returns_true_for_known_node() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert!(m.record_heartbeat("n1", 1000, 5));
    }

    #[test]
    fn test_record_heartbeat_returns_false_for_unknown_node() {
        let mut m = monitor();
        assert!(!m.record_heartbeat("unknown", 1000, 5));
    }

    #[test]
    fn test_record_heartbeat_sets_alive() {
        let mut m = NodeMonitor::new(1000);
        m.register(follower("n1"));
        m.states.insert("n1".to_string(), NodeState::Dead); // manually set dead
        m.record_heartbeat("n1", 1000, 5);
        assert_eq!(m.state("n1"), Some(&NodeState::Alive));
    }

    #[test]
    fn test_record_heartbeat_increments_history() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        m.record_heartbeat("n1", 2000, 6);
        assert_eq!(m.heartbeat_count("n1"), 2);
    }

    // ── check_timeouts ────────────────────────────────────────────────────────

    #[test]
    fn test_check_timeouts_no_timeout_within_window() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        let timed_out = m.check_timeouts(5999); // 4999 ms elapsed < 5000
        assert!(timed_out.is_empty());
    }

    #[test]
    fn test_check_timeouts_marks_dead() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        let timed_out = m.check_timeouts(6001); // 5001 ms >= 5000
        assert!(timed_out.contains(&"n1".to_string()));
        assert_eq!(m.state("n1"), Some(&NodeState::Dead));
    }

    #[test]
    fn test_check_timeouts_no_heartbeat_marks_dead() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        // No heartbeat recorded; last_seen defaults to 0.
        let timed_out = m.check_timeouts(5000);
        assert!(timed_out.contains(&"n1".to_string()));
    }

    #[test]
    fn test_check_timeouts_already_dead_not_returned_again() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        m.check_timeouts(6000); // mark dead
        let timed_out2 = m.check_timeouts(7000); // already dead → not returned again
        assert!(!timed_out2.contains(&"n1".to_string()));
    }

    // ── alive_nodes / dead_nodes ──────────────────────────────────────────────

    #[test]
    fn test_alive_nodes_all_alive() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.register(follower("n2"));
        assert_eq!(m.alive_nodes().len(), 2);
    }

    #[test]
    fn test_dead_nodes_empty_initially() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert_eq!(m.dead_nodes().len(), 0);
    }

    #[test]
    fn test_alive_dead_after_timeout() {
        let mut m = NodeMonitor::new(1000);
        m.register(follower("n1"));
        m.register(follower("n2"));
        m.record_heartbeat("n1", 0, 5);
        m.record_heartbeat("n2", 0, 5);
        m.check_timeouts(1001); // both time out
        assert_eq!(m.dead_nodes().len(), 2);
        assert_eq!(m.alive_nodes().len(), 0);
    }

    // ── avg_latency ───────────────────────────────────────────────────────────

    #[test]
    fn test_avg_latency_none_for_no_history() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert!(m.avg_latency("n1").is_none());
    }

    #[test]
    fn test_avg_latency_single_record() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 10);
        assert_eq!(m.avg_latency("n1"), Some(10.0));
    }

    #[test]
    fn test_avg_latency_multiple_records() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 10);
        m.record_heartbeat("n1", 2000, 20);
        m.record_heartbeat("n1", 3000, 30);
        // avg = (10+20+30)/3 = 20
        assert!((m.avg_latency("n1").unwrap() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_avg_latency_unknown_node_none() {
        let m = monitor();
        assert!(m.avg_latency("unknown").is_none());
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_returns_true() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert!(m.remove("n1"));
    }

    #[test]
    fn test_remove_decrements_count() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.remove("n1");
        assert_eq!(m.node_count(), 0);
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut m = monitor();
        assert!(!m.remove("nobody"));
    }

    #[test]
    fn test_remove_clears_state() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.remove("n1");
        assert_eq!(m.state("n1"), None);
    }

    // ── roles ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_leader_role_preserved() {
        let mut m = monitor();
        m.register(NodeInfo::new(
            "leader",
            "10.0.0.1:7000",
            NodeRole::Leader,
            0,
        ));
        assert_eq!(m.nodes["leader"].role, NodeRole::Leader);
    }

    #[test]
    fn test_observer_role_preserved() {
        let mut m = monitor();
        m.register(NodeInfo::new("obs", "10.0.0.2:7000", NodeRole::Observer, 0));
        assert_eq!(m.nodes["obs"].role, NodeRole::Observer);
    }

    // ── last_heartbeat ────────────────────────────────────────────────────────

    #[test]
    fn test_last_heartbeat_none_if_no_history() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert!(m.last_heartbeat("n1").is_none());
    }

    #[test]
    fn test_last_heartbeat_returns_most_recent() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        m.record_heartbeat("n1", 2000, 7);
        let hb = m.last_heartbeat("n1").expect("should have record");
        assert_eq!(hb.received_at, 2000);
    }

    // ── heartbeat_count ───────────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_count_zero_initially() {
        let mut m = monitor();
        m.register(follower("n1"));
        assert_eq!(m.heartbeat_count("n1"), 0);
    }

    #[test]
    fn test_heartbeat_count_increments() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        m.record_heartbeat("n1", 2000, 5);
        assert_eq!(m.heartbeat_count("n1"), 2);
    }

    #[test]
    fn test_max_history_trims_records() {
        let mut m = NodeMonitor::new(5000).with_max_history(3);
        m.register(follower("n1"));
        for i in 0..10u64 {
            m.record_heartbeat("n1", i * 100, 5);
        }
        assert_eq!(m.heartbeat_count("n1"), 3);
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_node_info_fields() {
        let n = NodeInfo::new("id1", "10.0.0.1:7000", NodeRole::Candidate, 42);
        assert_eq!(n.id, "id1");
        assert_eq!(n.address, "10.0.0.1:7000");
        assert_eq!(n.role, NodeRole::Candidate);
        assert_eq!(n.joined_at, 42);
    }

    #[test]
    fn test_node_state_suspected() {
        let state = NodeState::Suspected;
        assert_eq!(state, NodeState::Suspected);
    }

    #[test]
    fn test_heartbeat_record_fields() {
        let hb = HeartbeatRecord {
            node_id: "n1".to_string(),
            received_at: 999,
            latency_ms: 12,
        };
        assert_eq!(hb.node_id, "n1");
        assert_eq!(hb.received_at, 999);
        assert_eq!(hb.latency_ms, 12);
    }

    #[test]
    fn test_avg_latency_zero_latency() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 0);
        assert_eq!(m.avg_latency("n1"), Some(0.0));
    }

    #[test]
    fn test_check_timeouts_at_exact_boundary() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        m.record_heartbeat("n1", 1000, 5);
        // elapsed = 6000 - 1000 = 5000 >= 5000 → times out
        let timed_out = m.check_timeouts(6000);
        assert!(timed_out.contains(&"n1".to_string()));
    }

    #[test]
    fn test_multiple_registrations_independent() {
        let mut m = monitor();
        m.register(NodeInfo::new("a", "addr_a", NodeRole::Leader, 0));
        m.register(NodeInfo::new("b", "addr_b", NodeRole::Follower, 0));
        assert_eq!(m.node_count(), 2);
        assert!(m.state("a").is_some());
        assert!(m.state("b").is_some());
    }

    #[test]
    fn test_remove_then_re_register() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.remove("n1");
        m.register(NodeInfo::new("n1", "new_addr", NodeRole::Leader, 100));
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_alive_nodes_excludes_dead() {
        let mut m = NodeMonitor::new(1000);
        m.register(follower("n1"));
        m.register(follower("n2"));
        m.record_heartbeat("n1", 0, 5);
        m.record_heartbeat("n2", 0, 5);
        m.check_timeouts(1001); // both timeout
        assert_eq!(m.alive_nodes().len(), 0);
    }

    #[test]
    fn test_check_timeouts_partial() {
        let mut m = NodeMonitor::new(5000);
        m.register(follower("n1"));
        m.register(follower("n2"));
        m.record_heartbeat("n1", 1000, 5); // will timeout
        m.record_heartbeat("n2", 5000, 5); // will not timeout
        let timed_out = m.check_timeouts(6001); // n1: 5001ms; n2: 1001ms
        assert!(timed_out.contains(&"n1".to_string()));
        assert!(!timed_out.contains(&"n2".to_string()));
    }

    #[test]
    fn test_heartbeat_count_unknown_node() {
        let m = monitor();
        assert_eq!(m.heartbeat_count("ghost"), 0);
    }

    #[test]
    fn test_state_unknown_node_none() {
        let m = monitor();
        assert_eq!(m.state("unknown"), None);
    }

    #[test]
    fn test_last_heartbeat_unknown_node_none() {
        let m = monitor();
        assert_eq!(m.last_heartbeat("unknown"), None);
    }

    #[test]
    fn test_node_monitor_new_timeout() {
        let m = NodeMonitor::new(3000);
        assert_eq!(m.timeout_ms, 3000);
    }

    #[test]
    fn test_record_heartbeat_for_multiple_nodes() {
        let mut m = monitor();
        m.register(follower("n1"));
        m.register(follower("n2"));
        assert!(m.record_heartbeat("n1", 1000, 5));
        assert!(m.record_heartbeat("n2", 2000, 10));
        assert_eq!(m.heartbeat_count("n1"), 1);
        assert_eq!(m.heartbeat_count("n2"), 1);
    }
}
