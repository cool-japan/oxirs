//! # Distributed State Management
//!
//! Distributed state coordination for stream processors across partitions.
//! Uses consistent hashing (FNV-1a) for key-to-partition routing.

use std::collections::HashMap;

// ─── PartitionStateValue ──────────────────────────────────────────────────────

/// Value types storable in a distributed state partition.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionStateValue {
    Integer(i64),
    Float(f64),
    Bytes(Vec<u8>),
    StringVal(String),
    Counter(u64),
    Gauge { value: f64, timestamp: i64 },
}

// ─── StatePartition ───────────────────────────────────────────────────────────

/// A single partition of distributed state.
#[derive(Debug, Clone)]
pub struct StatePartition {
    pub partition_id: u32,
    pub state: HashMap<String, PartitionStateValue>,
    pub version: u64,
    pub last_checkpointed: i64,
}

impl StatePartition {
    pub fn new(partition_id: u32) -> Self {
        Self {
            partition_id,
            state: HashMap::new(),
            version: 0,
            last_checkpointed: 0,
        }
    }

    fn bump_version(&mut self) -> u64 {
        self.version += 1;
        self.version
    }
}

// ─── StateCoordinator ─────────────────────────────────────────────────────────

/// Coordinates state replication metadata across peer nodes.
#[derive(Debug, Clone)]
pub struct StateCoordinator {
    pub node_id: String,
    pub peers: Vec<String>,
}

impl StateCoordinator {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            peers: Vec::new(),
        }
    }

    pub fn add_peer(&mut self, peer: impl Into<String>) {
        self.peers.push(peer.into());
    }
}

// ─── DistributedStateStore ────────────────────────────────────────────────────

/// Distributed state store with consistent FNV-1a hashing for key-to-partition mapping.
///
/// Each key is routed to exactly one partition. Replication is handled by
/// `replicate_to` which returns a snapshot of a partition for a peer node.
pub struct DistributedStateStore {
    pub(crate) partitions: Vec<StatePartition>,
    replication_factor: usize,
    coordinator: StateCoordinator,
}

impl DistributedStateStore {
    /// Create a new store with `partition_count` partitions.
    pub fn new(partition_count: u32, replication_factor: usize) -> Self {
        let partitions = (0..partition_count).map(StatePartition::new).collect();
        Self {
            partitions,
            replication_factor,
            coordinator: StateCoordinator::new("local"),
        }
    }

    /// FNV-1a 64-bit hash for consistent partition routing.
    fn fnv_hash(key: &str) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut hash = FNV_OFFSET;
        for byte in key.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Determine which partition a key belongs to (consistent hashing).
    pub fn partition_for(&self, key: &str) -> u32 {
        let count = self.partitions.len() as u64;
        if count == 0 {
            return 0;
        }
        (Self::fnv_hash(key) % count) as u32
    }

    /// Get a value by key, or `None` if absent.
    pub fn get(&self, key: &str) -> Option<&PartitionStateValue> {
        let pid = self.partition_for(key) as usize;
        self.partitions.get(pid)?.state.get(key)
    }

    /// Set a key-value pair. Returns the new partition version number.
    pub fn set(&mut self, key: &str, value: PartitionStateValue) -> u64 {
        let pid = self.partition_for(key) as usize;
        let partition = &mut self.partitions[pid];
        partition.state.insert(key.to_string(), value);
        partition.bump_version()
    }

    /// Delete a key. Returns `true` if the key previously existed.
    pub fn delete(&mut self, key: &str) -> bool {
        let pid = self.partition_for(key) as usize;
        match self.partitions.get_mut(pid) {
            Some(partition) => partition.state.remove(key).is_some(),
            None => false,
        }
    }

    /// Return all key-value pairs from `partition_id` for replication to `peer`.
    pub fn replicate_to(
        &self,
        _peer: &str,
        partition_id: u32,
    ) -> Vec<(String, PartitionStateValue)> {
        self.partitions
            .iter()
            .find(|p| p.partition_id == partition_id)
            .map(|p| {
                p.state
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Snapshot a partition (clone it) and record `last_checkpointed` timestamp.
    pub fn checkpoint_partition(&mut self, partition_id: u32) -> StatePartition {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let partition = self
            .partitions
            .iter_mut()
            .find(|p| p.partition_id == partition_id)
            .expect("partition_id out of range");
        partition.last_checkpointed = now_ms;
        partition.clone()
    }

    /// Restore a partition from a previously-created checkpoint snapshot.
    pub fn restore_partition(&mut self, partition: StatePartition) {
        if let Some(p) = self
            .partitions
            .iter_mut()
            .find(|p| p.partition_id == partition.partition_id)
        {
            *p = partition;
        }
    }

    /// Total number of partitions in this store.
    pub fn partition_count(&self) -> u32 {
        self.partitions.len() as u32
    }

    /// Total number of keys across all partitions.
    pub fn total_keys(&self) -> usize {
        self.partitions.iter().map(|p| p.state.len()).sum()
    }

    /// The configured replication factor.
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }

    /// Read-only reference to the coordinator.
    pub fn coordinator(&self) -> &StateCoordinator {
        &self.coordinator
    }

    /// Mutable reference to the coordinator.
    pub fn coordinator_mut(&mut self) -> &mut StateCoordinator {
        &mut self.coordinator
    }
}

// ─── StateAggregator ──────────────────────────────────────────────────────────

/// High-level aggregator built on top of `DistributedStateStore`.
///
/// Provides common streaming aggregation patterns: increment counters,
/// running float sums, gauges, and windowed event counts.
pub struct StateAggregator {
    store: DistributedStateStore,
}

impl StateAggregator {
    /// Create an aggregator backed by a store with `partition_count` partitions.
    pub fn new(partition_count: u32) -> Self {
        Self {
            store: DistributedStateStore::new(partition_count, 1),
        }
    }

    /// Increment an integer counter by `by`. Returns the updated value.
    pub fn increment(&mut self, key: &str, by: i64) -> i64 {
        let current = match self.store.get(key) {
            Some(PartitionStateValue::Integer(v)) => *v,
            Some(PartitionStateValue::Counter(v)) => *v as i64,
            _ => 0,
        };
        let next = current + by;
        self.store.set(key, PartitionStateValue::Integer(next));
        next
    }

    /// Add `value` to a running float sum. Returns the updated sum.
    pub fn accumulate(&mut self, key: &str, value: f64) -> f64 {
        let current = match self.store.get(key) {
            Some(PartitionStateValue::Float(v)) => *v,
            _ => 0.0,
        };
        let next = current + value;
        self.store.set(key, PartitionStateValue::Float(next));
        next
    }

    /// Update a gauge value (timestamped float).
    pub fn update_gauge(&mut self, key: &str, value: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        self.store
            .set(key, PartitionStateValue::Gauge { value, timestamp });
    }

    /// Count events within a named window. Uses a composite key `window_key:event_key`.
    /// Returns the updated count.
    pub fn window_count(&mut self, window_key: &str, event_key: &str) -> u64 {
        let key = format!("{window_key}:{event_key}");
        let current = match self.store.get(&key) {
            Some(PartitionStateValue::Counter(v)) => *v,
            _ => 0,
        };
        let next = current + 1;
        self.store.set(&key, PartitionStateValue::Counter(next));
        next
    }

    /// Merge all state from `other` store into this aggregator's store.
    pub fn merge_from(&mut self, other: &DistributedStateStore) {
        for partition in &other.partitions {
            for (key, value) in &partition.state {
                self.store.set(key, value.clone());
            }
        }
    }

    /// Read-only access to the underlying store.
    pub fn store(&self) -> &DistributedStateStore {
        &self.store
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DistributedStateStore ──────────────────────────────────────────────────

    #[test]
    fn test_new_store_empty() {
        let store = DistributedStateStore::new(4, 1);
        assert_eq!(store.partition_count(), 4);
        assert_eq!(store.total_keys(), 0);
        assert_eq!(store.replication_factor(), 1);
    }

    #[test]
    fn test_set_and_get_string() {
        let mut store = DistributedStateStore::new(4, 1);
        store.set("hello", PartitionStateValue::StringVal("world".to_string()));
        match store.get("hello") {
            Some(PartitionStateValue::StringVal(s)) => assert_eq!(s, "world"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_set_returns_version_increases() {
        let mut store = DistributedStateStore::new(4, 1);
        let v1 = store.set("k", PartitionStateValue::Integer(1));
        let v2 = store.set("k", PartitionStateValue::Integer(2));
        assert!(v2 > v1, "version must increase on each write");
    }

    #[test]
    fn test_delete_existing_key() {
        let mut store = DistributedStateStore::new(4, 1);
        store.set("k", PartitionStateValue::Counter(10));
        assert!(
            store.delete("k"),
            "delete should return true for existing key"
        );
        assert!(store.get("k").is_none());
    }

    #[test]
    fn test_delete_missing_key() {
        let mut store = DistributedStateStore::new(4, 1);
        assert!(!store.delete("nonexistent"));
    }

    #[test]
    fn test_partition_for_deterministic() {
        let store = DistributedStateStore::new(8, 1);
        let p1 = store.partition_for("my_key");
        let p2 = store.partition_for("my_key");
        assert_eq!(p1, p2, "same key must always map to same partition");
        assert!(p1 < 8);
    }

    #[test]
    fn test_partition_for_distributes_across_partitions() {
        let store = DistributedStateStore::new(8, 1);
        let keys = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ];
        let partitions: std::collections::HashSet<u32> =
            keys.iter().map(|k| store.partition_for(k)).collect();
        assert!(
            partitions.len() >= 2,
            "8 keys over 8 partitions must use at least 2 different partitions"
        );
    }

    #[test]
    fn test_total_keys_after_operations() {
        let mut store = DistributedStateStore::new(4, 1);
        store.set("a", PartitionStateValue::Integer(1));
        store.set("b", PartitionStateValue::Integer(2));
        store.set("c", PartitionStateValue::Integer(3));
        assert_eq!(store.total_keys(), 3);
        store.delete("b");
        assert_eq!(store.total_keys(), 2);
    }

    #[test]
    fn test_replicate_to_returns_partition_contents() {
        let mut store = DistributedStateStore::new(4, 2);
        store.set("key1", PartitionStateValue::Integer(42));
        let pid = store.partition_for("key1");
        let replica = store.replicate_to("peer-node", pid);
        assert!(!replica.is_empty());
        assert!(replica.iter().any(|(k, _)| k == "key1"));
    }

    #[test]
    fn test_replicate_to_nonexistent_partition() {
        let store = DistributedStateStore::new(4, 1);
        let replica = store.replicate_to("peer", 99);
        assert!(replica.is_empty());
    }

    #[test]
    fn test_checkpoint_and_restore() {
        let mut store = DistributedStateStore::new(4, 1);
        let expected_val = 42.5_f64;
        store.set("x", PartitionStateValue::Float(expected_val));
        let pid = store.partition_for("x");

        let checkpoint = store.checkpoint_partition(pid);
        assert!(
            checkpoint.last_checkpointed > 0,
            "last_checkpointed must be set"
        );

        // Corrupt state
        store.set("x", PartitionStateValue::Float(0.0));

        // Restore
        store.restore_partition(checkpoint);
        match store.get("x") {
            Some(PartitionStateValue::Float(v)) => {
                assert!((v - expected_val).abs() < 1e-9);
            }
            other => panic!("unexpected after restore: {other:?}"),
        }
    }

    #[test]
    fn test_coordinator_default_node_id() {
        let store = DistributedStateStore::new(2, 1);
        assert_eq!(store.coordinator().node_id, "local");
        assert!(store.coordinator().peers.is_empty());
    }

    #[test]
    fn test_coordinator_add_peers() {
        let mut store = DistributedStateStore::new(2, 1);
        store.coordinator_mut().add_peer("node-2");
        store.coordinator_mut().add_peer("node-3");
        assert_eq!(store.coordinator().peers.len(), 2);
    }

    #[test]
    fn test_all_value_variants() {
        let mut store = DistributedStateStore::new(8, 1);
        store.set("int_k", PartitionStateValue::Integer(-10));
        store.set("float_k", PartitionStateValue::Float(2.5));
        store.set("bytes_k", PartitionStateValue::Bytes(vec![1, 2, 3]));
        store.set("str_k", PartitionStateValue::StringVal("hi".to_string()));
        store.set("ctr_k", PartitionStateValue::Counter(99));
        store.set(
            "gauge_k",
            PartitionStateValue::Gauge {
                value: 1.0,
                timestamp: 1000,
            },
        );
        assert_eq!(store.total_keys(), 6);
    }

    #[test]
    fn test_single_partition_all_keys_same_partition() {
        let store = DistributedStateStore::new(1, 1);
        assert_eq!(store.partition_for("anything"), 0);
        assert_eq!(store.partition_for("other_key"), 0);
    }

    #[test]
    fn test_overwrite_value() {
        let mut store = DistributedStateStore::new(4, 1);
        store.set("key", PartitionStateValue::Integer(1));
        store.set("key", PartitionStateValue::Integer(2));
        match store.get("key") {
            Some(PartitionStateValue::Integer(v)) => assert_eq!(*v, 2),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_state_partition_new() {
        let p = StatePartition::new(5);
        assert_eq!(p.partition_id, 5);
        assert_eq!(p.version, 0);
        assert!(p.state.is_empty());
        assert_eq!(p.last_checkpointed, 0);
    }

    // ── StateAggregator ────────────────────────────────────────────────────────

    #[test]
    fn test_aggregator_increment_positive() {
        let mut agg = StateAggregator::new(4);
        assert_eq!(agg.increment("counter", 5), 5);
        assert_eq!(agg.increment("counter", 3), 8);
    }

    #[test]
    fn test_aggregator_increment_negative() {
        let mut agg = StateAggregator::new(4);
        agg.increment("counter", 10);
        assert_eq!(agg.increment("counter", -2), 8);
    }

    #[test]
    fn test_aggregator_accumulate_floats() {
        let mut agg = StateAggregator::new(4);
        let v1 = agg.accumulate("sum", 1.5);
        let v2 = agg.accumulate("sum", 2.5);
        assert!((v1 - 1.5).abs() < 1e-9);
        assert!((v2 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_update_gauge() {
        let mut agg = StateAggregator::new(4);
        agg.update_gauge("temperature", 98.6);
        match agg.store().get("temperature") {
            Some(PartitionStateValue::Gauge { value, .. }) => {
                assert!((value - 98.6).abs() < 1e-9);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_aggregator_window_count_isolated() {
        let mut agg = StateAggregator::new(4);
        assert_eq!(agg.window_count("win-1", "click"), 1);
        assert_eq!(agg.window_count("win-1", "click"), 2);
        assert_eq!(agg.window_count("win-1", "view"), 1);
        assert_eq!(agg.window_count("win-2", "click"), 1);
    }

    #[test]
    fn test_aggregator_merge_from() {
        let mut store2 = DistributedStateStore::new(4, 1);
        store2.set("shared_key", PartitionStateValue::Integer(100));

        let mut agg = StateAggregator::new(4);
        agg.merge_from(&store2);

        match agg.store().get("shared_key") {
            Some(PartitionStateValue::Integer(v)) => assert_eq!(*v, 100),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_aggregator_store_accessor() {
        let agg = StateAggregator::new(4);
        assert_eq!(agg.store().partition_count(), 4);
        assert_eq!(agg.store().total_keys(), 0);
    }
}
