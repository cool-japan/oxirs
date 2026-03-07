//! Kafka-style stream partition management.
//!
//! Provides `PartitionManager` for distributing stream data across partitions
//! with configurable assignment strategies, leader election, in-sync replica
//! (ISR) tracking, and high-watermark advancement.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration for a partitioned topic.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of partitions for this topic.
    pub partition_count: usize,
    /// Number of replicas per partition (including leader).
    pub replication_factor: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            partition_count: 1,
            replication_factor: 1,
        }
    }
}

/// Runtime state of a single partition.
#[derive(Debug, Clone)]
pub struct PartitionState {
    /// Zero-based partition identifier.
    pub id: usize,
    /// Broker ID that is the current leader, or `None` if leaderless.
    pub leader: Option<String>,
    /// All assigned replica broker IDs.
    pub replicas: Vec<String>,
    /// In-sync replica set — replicas that are fully caught up.
    pub isr: Vec<String>,
    /// Highest offset that has been committed/acknowledged by all ISR replicas.
    pub high_watermark: u64,
    /// Next offset to be written (end of the log).
    pub log_end_offset: u64,
}

impl PartitionState {
    fn new(id: usize) -> Self {
        Self {
            id,
            leader: None,
            replicas: Vec::new(),
            isr: Vec::new(),
            high_watermark: 0,
            log_end_offset: 0,
        }
    }

    /// Lag for this partition: messages produced but not yet committed.
    pub fn lag(&self) -> u64 {
        self.log_end_offset.saturating_sub(self.high_watermark)
    }
}

/// Strategy for mapping a record key to a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentStrategy {
    /// Cycle through partitions in order, ignoring the key.
    RoundRobin,
    /// Hash the key and take modulo of the partition count.
    Hash,
    /// Sticky: always return partition 0 (simplified sticky producer).
    Sticky,
    /// Manual: always return partition 0 (caller controls partition explicitly).
    Manual,
}

/// Manages partitions for a single Kafka-style topic.
pub struct PartitionManager {
    topic: String,
    partitions: Vec<PartitionState>,
    config: PartitionConfig,
    /// Counter used internally for `RoundRobin` assignment.
    rr_counter: usize,
}

impl PartitionManager {
    /// Create a new `PartitionManager` for `topic` with the supplied config.
    ///
    /// Initialises `config.partition_count` empty `PartitionState` entries.
    pub fn new(topic: impl Into<String>, config: PartitionConfig) -> Self {
        let count = config.partition_count;
        let partitions = (0..count).map(PartitionState::new).collect();
        Self {
            topic: topic.into(),
            partitions,
            config,
            rr_counter: 0,
        }
    }

    /// Return the topic name.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Assign a partition index for `key` using the given strategy.
    ///
    /// Returns `0` when the partition count is zero (degenerate case).
    pub fn assign_partition(&mut self, key: &[u8], strategy: &AssignmentStrategy) -> usize {
        let n = self.partitions.len();
        if n == 0 {
            return 0;
        }
        match strategy {
            AssignmentStrategy::RoundRobin => {
                let idx = self.rr_counter % n;
                self.rr_counter = self.rr_counter.wrapping_add(1);
                idx
            }
            AssignmentStrategy::Hash => {
                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() as usize) % n
            }
            AssignmentStrategy::Sticky | AssignmentStrategy::Manual => 0,
        }
    }

    /// Immutable access to a partition by ID.
    pub fn get_partition(&self, id: usize) -> Option<&PartitionState> {
        self.partitions.get(id)
    }

    /// Mutable access to a partition by ID.
    pub fn get_partition_mut(&mut self, id: usize) -> Option<&mut PartitionState> {
        self.partitions.get_mut(id)
    }

    /// Number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Return the leader broker ID for `partition_id`, if any.
    pub fn leader_for(&self, partition_id: usize) -> Option<&str> {
        self.partitions
            .get(partition_id)
            .and_then(|p| p.leader.as_deref())
    }

    /// Set (or clear) the leader for `partition_id`.
    ///
    /// Returns `true` if the partition exists, `false` otherwise.
    pub fn set_leader(&mut self, partition_id: usize, leader: Option<String>) -> bool {
        match self.partitions.get_mut(partition_id) {
            Some(p) => {
                p.leader = leader;
                true
            }
            None => false,
        }
    }

    /// Replace the ISR list for `partition_id`.
    ///
    /// Returns `true` if the partition exists, `false` otherwise.
    pub fn update_isr(&mut self, partition_id: usize, isr: Vec<String>) -> bool {
        match self.partitions.get_mut(partition_id) {
            Some(p) => {
                p.isr = isr;
                true
            }
            None => false,
        }
    }

    /// Advance the `high_watermark` for `partition_id` to `offset`.
    ///
    /// The watermark will only increase; passing a lower value is a no-op.
    /// Returns `true` if the partition exists, `false` otherwise.
    pub fn advance_watermark(&mut self, partition_id: usize, offset: u64) -> bool {
        match self.partitions.get_mut(partition_id) {
            Some(p) => {
                if offset > p.high_watermark {
                    p.high_watermark = offset;
                }
                true
            }
            None => false,
        }
    }

    /// Sum of `(log_end_offset - high_watermark)` across all partitions.
    pub fn total_lag(&self) -> u64 {
        self.partitions.iter().map(|p| p.lag()).sum()
    }

    /// Distribute leader roles across `nodes` using round-robin.
    ///
    /// Also sets each partition's `replicas` list from the node pool and
    /// initialises the ISR to match the replicas (simplified).
    ///
    /// Does nothing if `nodes` is empty.
    pub fn rebalance(&mut self, nodes: &[String]) {
        if nodes.is_empty() {
            return;
        }
        let rf = self.config.replication_factor.min(nodes.len());
        for (idx, partition) in self.partitions.iter_mut().enumerate() {
            let leader_node = nodes[idx % nodes.len()].clone();
            // Assign `rf` replicas starting from the leader index.
            let replicas: Vec<String> = (0..rf)
                .map(|offset| nodes[(idx + offset) % nodes.len()].clone())
                .collect();
            partition.leader = Some(leader_node.clone());
            partition.isr = replicas.clone();
            partition.replicas = replicas;
        }
    }

    /// Return the IDs of partitions that currently have no leader.
    pub fn leaderless_partitions(&self) -> Vec<usize> {
        self.partitions
            .iter()
            .filter(|p| p.leader.is_none())
            .map(|p| p.id)
            .collect()
    }

    /// Expose an iterator over all partitions.
    pub fn partitions(&self) -> &[PartitionState] {
        &self.partitions
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(n: usize) -> PartitionManager {
        PartitionManager::new(
            "test-topic",
            PartitionConfig {
                partition_count: n,
                replication_factor: 1,
            },
        )
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_creates_correct_count() {
        let pm = make_manager(5);
        assert_eq!(pm.partition_count(), 5);
    }

    #[test]
    fn test_new_single_partition() {
        let pm = make_manager(1);
        assert_eq!(pm.partition_count(), 1);
    }

    #[test]
    fn test_new_zero_partitions() {
        let pm = make_manager(0);
        assert_eq!(pm.partition_count(), 0);
    }

    #[test]
    fn test_topic_name_stored() {
        let pm = make_manager(3);
        assert_eq!(pm.topic(), "test-topic");
    }

    #[test]
    fn test_initial_state_empty_leader() {
        let pm = make_manager(4);
        for i in 0..4 {
            assert!(pm.get_partition(i).unwrap().leader.is_none());
        }
    }

    #[test]
    fn test_initial_high_watermark_zero() {
        let pm = make_manager(3);
        for i in 0..3 {
            assert_eq!(pm.get_partition(i).unwrap().high_watermark, 0);
        }
    }

    #[test]
    fn test_initial_log_end_offset_zero() {
        let pm = make_manager(2);
        for i in 0..2 {
            assert_eq!(pm.get_partition(i).unwrap().log_end_offset, 0);
        }
    }

    // ── assign_partition RoundRobin ───────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles() {
        let mut pm = make_manager(3);
        let key = b"any-key";
        let p0 = pm.assign_partition(key, &AssignmentStrategy::RoundRobin);
        let p1 = pm.assign_partition(key, &AssignmentStrategy::RoundRobin);
        let p2 = pm.assign_partition(key, &AssignmentStrategy::RoundRobin);
        let p3 = pm.assign_partition(key, &AssignmentStrategy::RoundRobin);
        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);
        assert_eq!(p3, 0); // wraps around
    }

    #[test]
    fn test_round_robin_ignores_key() {
        let mut pm = make_manager(2);
        let a = pm.assign_partition(b"key-a", &AssignmentStrategy::RoundRobin);
        let b = pm.assign_partition(b"key-b", &AssignmentStrategy::RoundRobin);
        // First call should be 0, second 1 (irrespective of key)
        assert_ne!(a, b);
    }

    #[test]
    fn test_round_robin_single_partition() {
        let mut pm = make_manager(1);
        for _ in 0..5 {
            assert_eq!(
                pm.assign_partition(b"k", &AssignmentStrategy::RoundRobin),
                0
            );
        }
    }

    // ── assign_partition Hash ─────────────────────────────────────────────────

    #[test]
    fn test_hash_deterministic_same_key() {
        let mut pm = make_manager(4);
        let r1 = pm.assign_partition(b"consistent-key", &AssignmentStrategy::Hash);
        let r2 = pm.assign_partition(b"consistent-key", &AssignmentStrategy::Hash);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_hash_in_range() {
        let mut pm = make_manager(8);
        for key in &[b"a".as_ref(), b"bb", b"ccc", b"dddd"] {
            let p = pm.assign_partition(key, &AssignmentStrategy::Hash);
            assert!(p < 8, "partition {p} out of range");
        }
    }

    #[test]
    fn test_hash_different_keys_may_differ() {
        let mut pm = make_manager(16);
        let p1 = pm.assign_partition(b"key-alpha", &AssignmentStrategy::Hash);
        let p2 = pm.assign_partition(b"key-beta", &AssignmentStrategy::Hash);
        // Different keys should almost certainly map differently (not guaranteed, but very likely)
        // We just verify both are in range
        assert!(p1 < 16);
        assert!(p2 < 16);
    }

    // ── assign_partition Sticky / Manual ──────────────────────────────────────

    #[test]
    fn test_sticky_always_zero() {
        let mut pm = make_manager(5);
        for _ in 0..10 {
            assert_eq!(pm.assign_partition(b"k", &AssignmentStrategy::Sticky), 0);
        }
    }

    #[test]
    fn test_manual_always_zero() {
        let mut pm = make_manager(5);
        assert_eq!(pm.assign_partition(b"k", &AssignmentStrategy::Manual), 0);
    }

    // ── get_partition ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_partition_valid() {
        let pm = make_manager(3);
        let p = pm.get_partition(2);
        assert!(p.is_some());
        assert_eq!(p.unwrap().id, 2);
    }

    #[test]
    fn test_get_partition_out_of_range() {
        let pm = make_manager(3);
        assert!(pm.get_partition(10).is_none());
    }

    #[test]
    fn test_get_partition_mut_modifies_state() {
        let mut pm = make_manager(2);
        pm.get_partition_mut(0).unwrap().log_end_offset = 42;
        assert_eq!(pm.get_partition(0).unwrap().log_end_offset, 42);
    }

    // ── leader_for / set_leader ───────────────────────────────────────────────

    #[test]
    fn test_leader_for_none_initially() {
        let pm = make_manager(2);
        assert!(pm.leader_for(0).is_none());
    }

    #[test]
    fn test_set_leader_returns_true() {
        let mut pm = make_manager(3);
        assert!(pm.set_leader(1, Some("broker-1".into())));
    }

    #[test]
    fn test_set_leader_invalid_partition() {
        let mut pm = make_manager(2);
        assert!(!pm.set_leader(99, Some("broker-x".into())));
    }

    #[test]
    fn test_leader_for_after_set() {
        let mut pm = make_manager(3);
        pm.set_leader(0, Some("broker-0".into()));
        assert_eq!(pm.leader_for(0), Some("broker-0"));
    }

    #[test]
    fn test_set_leader_clear() {
        let mut pm = make_manager(2);
        pm.set_leader(0, Some("broker-0".into()));
        pm.set_leader(0, None);
        assert!(pm.leader_for(0).is_none());
    }

    // ── update_isr ────────────────────────────────────────────────────────────

    #[test]
    fn test_update_isr_returns_true() {
        let mut pm = make_manager(3);
        assert!(pm.update_isr(0, vec!["b1".into(), "b2".into()]));
    }

    #[test]
    fn test_update_isr_invalid_partition() {
        let mut pm = make_manager(2);
        assert!(!pm.update_isr(50, vec!["b1".into()]));
    }

    #[test]
    fn test_update_isr_stored() {
        let mut pm = make_manager(2);
        pm.update_isr(1, vec!["broker-a".into(), "broker-b".into()]);
        let isr = &pm.get_partition(1).unwrap().isr;
        assert_eq!(isr, &["broker-a", "broker-b"]);
    }

    // ── advance_watermark ─────────────────────────────────────────────────────

    #[test]
    fn test_advance_watermark_returns_true() {
        let mut pm = make_manager(2);
        assert!(pm.advance_watermark(0, 100));
    }

    #[test]
    fn test_advance_watermark_invalid_partition() {
        let mut pm = make_manager(1);
        assert!(!pm.advance_watermark(5, 100));
    }

    #[test]
    fn test_advance_watermark_increases() {
        let mut pm = make_manager(1);
        pm.advance_watermark(0, 50);
        assert_eq!(pm.get_partition(0).unwrap().high_watermark, 50);
        pm.advance_watermark(0, 80);
        assert_eq!(pm.get_partition(0).unwrap().high_watermark, 80);
    }

    #[test]
    fn test_advance_watermark_no_decrease() {
        let mut pm = make_manager(1);
        pm.advance_watermark(0, 100);
        pm.advance_watermark(0, 50); // should not decrease
        assert_eq!(pm.get_partition(0).unwrap().high_watermark, 100);
    }

    // ── total_lag ─────────────────────────────────────────────────────────────

    #[test]
    fn test_total_lag_zero_initially() {
        let pm = make_manager(4);
        assert_eq!(pm.total_lag(), 0);
    }

    #[test]
    fn test_total_lag_calculation() {
        let mut pm = make_manager(3);
        // Partition 0: log_end=100, hwm=80  → lag 20
        pm.get_partition_mut(0).unwrap().log_end_offset = 100;
        pm.advance_watermark(0, 80);
        // Partition 1: log_end=50, hwm=50  → lag 0
        pm.get_partition_mut(1).unwrap().log_end_offset = 50;
        pm.advance_watermark(1, 50);
        // Partition 2: log_end=200, hwm=150 → lag 50
        pm.get_partition_mut(2).unwrap().log_end_offset = 200;
        pm.advance_watermark(2, 150);
        assert_eq!(pm.total_lag(), 70);
    }

    #[test]
    fn test_total_lag_no_negative() {
        let mut pm = make_manager(1);
        // hwm > log_end_offset → saturating_sub should prevent underflow
        pm.get_partition_mut(0).unwrap().high_watermark = 100;
        pm.get_partition_mut(0).unwrap().log_end_offset = 0;
        assert_eq!(pm.total_lag(), 0);
    }

    // ── rebalance ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rebalance_assigns_leaders() {
        let mut pm = make_manager(3);
        let nodes = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        pm.rebalance(&nodes);
        for i in 0..3 {
            assert!(pm.leader_for(i).is_some());
        }
    }

    #[test]
    fn test_rebalance_distributes_round_robin() {
        let mut pm = make_manager(3);
        let nodes = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        pm.rebalance(&nodes);
        assert_eq!(pm.leader_for(0), Some("n1"));
        assert_eq!(pm.leader_for(1), Some("n2"));
        assert_eq!(pm.leader_for(2), Some("n3"));
    }

    #[test]
    fn test_rebalance_empty_nodes_no_op() {
        let mut pm = make_manager(3);
        pm.rebalance(&[]);
        assert_eq!(pm.leaderless_partitions().len(), 3);
    }

    #[test]
    fn test_rebalance_fewer_nodes_than_partitions() {
        let mut pm = make_manager(4);
        let nodes = vec!["n1".to_string(), "n2".to_string()];
        pm.rebalance(&nodes);
        // All partitions should have a leader
        assert_eq!(pm.leaderless_partitions().len(), 0);
    }

    #[test]
    fn test_rebalance_sets_isr() {
        let mut pm = make_manager(2);
        let nodes = vec!["n1".to_string(), "n2".to_string()];
        pm.rebalance(&nodes);
        for i in 0..2 {
            assert!(!pm.get_partition(i).unwrap().isr.is_empty());
        }
    }

    // ── leaderless_partitions ─────────────────────────────────────────────────

    #[test]
    fn test_leaderless_all_initially() {
        let pm = make_manager(4);
        assert_eq!(pm.leaderless_partitions().len(), 4);
    }

    #[test]
    fn test_leaderless_after_set_leader() {
        let mut pm = make_manager(3);
        pm.set_leader(0, Some("b0".into()));
        pm.set_leader(2, Some("b2".into()));
        let leaderless = pm.leaderless_partitions();
        assert_eq!(leaderless, vec![1]);
    }

    #[test]
    fn test_leaderless_none_after_rebalance() {
        let mut pm = make_manager(3);
        let nodes = vec!["n1".to_string()];
        pm.rebalance(&nodes);
        assert_eq!(pm.leaderless_partitions().len(), 0);
    }

    // ── PartitionState lag ────────────────────────────────────────────────────

    #[test]
    fn test_partition_state_lag() {
        let mut p = PartitionState::new(0);
        p.log_end_offset = 200;
        p.high_watermark = 150;
        assert_eq!(p.lag(), 50);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_partitions_slice_length() {
        let pm = make_manager(5);
        assert_eq!(pm.partitions().len(), 5);
    }

    #[test]
    fn test_config_stored() {
        let cfg = PartitionConfig {
            partition_count: 4,
            replication_factor: 3,
        };
        let pm = PartitionManager::new("t", cfg.clone());
        assert_eq!(pm.config.partition_count, 4);
        assert_eq!(pm.config.replication_factor, 3);
    }

    #[test]
    fn test_rebalance_wraps_leaders() {
        let mut pm = make_manager(4);
        // Only 2 nodes but 4 partitions; wrap around.
        let nodes = vec!["n0".to_string(), "n1".to_string()];
        pm.rebalance(&nodes);
        assert_eq!(pm.leader_for(0), Some("n0"));
        assert_eq!(pm.leader_for(1), Some("n1"));
        assert_eq!(pm.leader_for(2), Some("n0")); // wrap
        assert_eq!(pm.leader_for(3), Some("n1")); // wrap
    }

    #[test]
    fn test_partition_state_id_matches_index() {
        let pm = make_manager(4);
        for i in 0..4 {
            assert_eq!(pm.get_partition(i).unwrap().id, i);
        }
    }

    #[test]
    fn test_initial_replicas_empty() {
        let pm = make_manager(3);
        for i in 0..3 {
            assert!(pm.get_partition(i).unwrap().replicas.is_empty());
        }
    }
}
