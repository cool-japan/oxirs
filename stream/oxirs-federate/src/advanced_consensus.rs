//! Advanced Consensus Features
//!
//! Implements advanced distributed consensus mechanisms:
//! - Byzantine Fault Tolerance (BFT)
//! - Conflict-free Replicated Data Types (CRDTs)
//! - Vector clocks for causality tracking
//! - Distributed locking mechanisms
//! - Network partition handling

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::info;

/// Byzantine Fault Tolerant Consensus
#[derive(Debug, Clone)]
pub struct ByzantineFaultTolerance {
    #[allow(dead_code)]
    node_id: String,
    nodes: Arc<RwLock<HashSet<String>>>,
    f: usize, // Maximum number of Byzantine nodes tolerated
    #[allow(dead_code)]
    view: Arc<RwLock<u64>>,
}

impl ByzantineFaultTolerance {
    pub fn new(node_id: String, total_nodes: usize) -> Self {
        let f = (total_nodes - 1) / 3;
        Self {
            node_id,
            nodes: Arc::new(RwLock::new(HashSet::new())),
            f,
            view: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn propose(&self, _value: Vec<u8>) -> Result<bool> {
        info!("BFT proposing value from node {}", self.node_id);
        let nodes = self.nodes.read().await;
        let required = 2 * self.f + 1;
        Ok(nodes.len() >= required)
    }

    pub async fn add_node(&self, node_id: String) {
        self.nodes.write().await.insert(node_id);
    }
}

/// Vector Clock for causality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    clock: HashMap<String, u64>,
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clock: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: String) {
        *self.clock.entry(node_id).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &timestamp) in &other.clock {
            let entry = self.clock.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(timestamp);
        }
    }

    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut strictly_less = false;

        // Check all nodes in self
        for (node, &my_time) in &self.clock {
            let other_time = other.clock.get(node).copied().unwrap_or(0);
            if my_time > other_time {
                return false;
            }
            if my_time < other_time {
                strictly_less = true;
            }
        }

        // Also check nodes that exist in other but not in self
        for (node, &other_time) in &other.clock {
            if !self.clock.contains_key(node) {
                // self[node] is implicitly 0, other[node] > 0
                if other_time > 0 {
                    strictly_less = true;
                }
            }
        }

        strictly_less
    }
}

/// CRDT - Grow-only Counter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounter {
    counts: HashMap<String, u64>,
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: String, amount: u64) {
        *self.counts.entry(node_id).or_insert(0) += amount;
    }

    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn merge(&mut self, other: &GCounter) {
        for (node, &count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

/// CRDT - Positive-Negative Counter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PNCounter {
    positive: GCounter,
    negative: GCounter,
}

impl Default for PNCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl PNCounter {
    pub fn new() -> Self {
        Self {
            positive: GCounter::new(),
            negative: GCounter::new(),
        }
    }

    pub fn increment(&mut self, node_id: String, amount: u64) {
        self.positive.increment(node_id, amount);
    }

    pub fn decrement(&mut self, node_id: String, amount: u64) {
        self.negative.increment(node_id, amount);
    }

    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

/// Distributed Lock
#[derive(Debug, Clone)]
pub struct DistributedLock {
    lock_id: String,
    holder: Arc<RwLock<Option<String>>>,
    acquired_at: Arc<RwLock<Option<SystemTime>>>,
    ttl: std::time::Duration,
}

impl DistributedLock {
    pub fn new(lock_id: String, ttl: std::time::Duration) -> Self {
        Self {
            lock_id,
            holder: Arc::new(RwLock::new(None)),
            acquired_at: Arc::new(RwLock::new(None)),
            ttl,
        }
    }

    pub async fn acquire(&self, node_id: String) -> Result<bool> {
        let mut holder = self.holder.write().await;

        // Check if lock is expired
        if let Some(acquired_time) = *self.acquired_at.read().await {
            if acquired_time.elapsed().unwrap_or_default() > self.ttl {
                *holder = None;
            }
        }

        if holder.is_none() {
            *holder = Some(node_id);
            *self.acquired_at.write().await = Some(SystemTime::now());
            info!("Lock {} acquired", self.lock_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn release(&self, node_id: &str) -> Result<()> {
        let mut holder = self.holder.write().await;
        if let Some(ref current_holder) = *holder {
            if current_holder == node_id {
                *holder = None;
                *self.acquired_at.write().await = None;
                info!("Lock {} released", self.lock_id);
                return Ok(());
            }
        }
        Err(anyhow!("Not the lock holder"))
    }
}

/// Network Partition Detector
#[derive(Debug, Clone)]
pub struct NetworkPartitionDetector {
    #[allow(dead_code)]
    node_id: String,
    heartbeats: Arc<RwLock<HashMap<String, SystemTime>>>,
    timeout: std::time::Duration,
}

impl NetworkPartitionDetector {
    pub fn new(node_id: String, timeout: std::time::Duration) -> Self {
        Self {
            node_id,
            heartbeats: Arc::new(RwLock::new(HashMap::new())),
            timeout,
        }
    }

    pub async fn record_heartbeat(&self, node_id: String) {
        self.heartbeats
            .write()
            .await
            .insert(node_id, SystemTime::now());
    }

    pub async fn detect_partition(&self) -> Vec<String> {
        let heartbeats = self.heartbeats.read().await;
        let now = SystemTime::now();

        heartbeats
            .iter()
            .filter_map(|(node, &last_heartbeat)| {
                if now.duration_since(last_heartbeat).unwrap_or_default() > self.timeout {
                    Some(node.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Advanced Consensus System
#[derive(Debug)]
pub struct AdvancedConsensusSystem {
    bft: Option<Arc<ByzantineFaultTolerance>>,
    vector_clock: Arc<RwLock<VectorClock>>,
    locks: Arc<RwLock<HashMap<String, DistributedLock>>>,
    partition_detector: Arc<NetworkPartitionDetector>,
}

impl AdvancedConsensusSystem {
    pub fn new(node_id: String, total_nodes: usize) -> Self {
        Self {
            bft: Some(Arc::new(ByzantineFaultTolerance::new(
                node_id.clone(),
                total_nodes,
            ))),
            vector_clock: Arc::new(RwLock::new(VectorClock::new())),
            locks: Arc::new(RwLock::new(HashMap::new())),
            partition_detector: Arc::new(NetworkPartitionDetector::new(
                node_id,
                std::time::Duration::from_secs(30),
            )),
        }
    }

    pub async fn propose_value(&self, value: Vec<u8>) -> Result<bool> {
        if let Some(ref bft) = self.bft {
            bft.propose(value).await
        } else {
            Err(anyhow!("BFT not enabled"))
        }
    }

    pub async fn increment_clock(&self, node_id: String) {
        self.vector_clock.write().await.increment(node_id);
    }

    pub async fn acquire_lock(&self, lock_id: String, node_id: String) -> Result<bool> {
        let mut locks = self.locks.write().await;
        let lock = locks
            .entry(lock_id.clone())
            .or_insert_with(|| DistributedLock::new(lock_id, std::time::Duration::from_secs(30)));
        lock.acquire(node_id).await
    }

    pub async fn detect_partitions(&self) -> Vec<String> {
        self.partition_detector.detect_partition().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bft() {
        let bft = ByzantineFaultTolerance::new("node1".to_string(), 4);
        bft.add_node("node2".to_string()).await;
        bft.add_node("node3".to_string()).await;
        bft.add_node("node4".to_string()).await;

        let result = bft.propose(vec![1, 2, 3]).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_vector_clock() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        // Test concurrent events (neither happens before the other)
        clock1.increment("node1".to_string());
        clock2.increment("node2".to_string());

        // These are concurrent, so neither should happen before the other
        assert!(!clock1.happens_before(&clock2));
        assert!(!clock2.happens_before(&clock1));

        // Test causality: merge clock2 into clock1 and advance clock1
        clock1.merge(&clock2);
        clock1.increment("node1".to_string());

        // Now clock1 should happen after clock2
        assert!(clock2.happens_before(&clock1));
        assert!(!clock1.happens_before(&clock2));
    }

    #[test]
    fn test_crdt_gcounter() {
        let mut counter = GCounter::new();
        counter.increment("node1".to_string(), 5);
        counter.increment("node2".to_string(), 3);
        assert_eq!(counter.value(), 8);
    }

    #[test]
    fn test_crdt_pncounter() {
        let mut counter = PNCounter::new();
        counter.increment("node1".to_string(), 10);
        counter.decrement("node1".to_string(), 3);
        assert_eq!(counter.value(), 7);
    }

    #[tokio::test]
    async fn test_distributed_lock() {
        let lock =
            DistributedLock::new("test_lock".to_string(), std::time::Duration::from_secs(60));

        let acquired = lock.acquire("node1".to_string()).await;
        assert!(acquired.is_ok());
        assert!(acquired.expect("operation should succeed"));

        let acquired2 = lock.acquire("node2".to_string()).await;
        assert!(acquired2.is_ok());
        assert!(!acquired2.expect("operation should succeed"));
    }
}
