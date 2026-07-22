//! Data partitioning and sharding for distributed storage

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    hash::{Hash, Hasher},
    sync::Arc,
};
use tokio::sync::RwLock;

use crate::{
    clustering::{NodeInfo, PartitionConfig, PartitionStrategy},
    error::{FusekiError, FusekiResult},
    store::Store,
};

/// Partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// Partition ID
    pub id: u32,
    /// Assigned nodes (primary and replicas)
    pub nodes: Vec<String>,
    /// Partition state
    pub state: PartitionState,
    /// Data size in bytes
    pub size: u64,
    /// Number of keys
    pub key_count: u64,
    /// Last updated timestamp
    pub updated_at: i64,
}

/// Partition state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PartitionState {
    /// Partition is active and serving requests
    Active,
    /// Partition is being migrated
    Migrating,
    /// Partition is under-replicated
    UnderReplicated,
    /// Partition is offline
    Offline,
}

/// Partition assignment
#[derive(Debug, Clone)]
pub struct PartitionAssignment {
    /// Partition to nodes mapping
    pub partitions: HashMap<u32, Vec<String>>,
    /// Node to partitions mapping
    pub nodes: HashMap<String, Vec<u32>>,
    /// Version number
    pub version: u64,
}

/// Consistent hashing ring
pub struct ConsistentHashRing {
    /// Virtual nodes on the ring
    ring: BTreeMap<u64, String>,
    /// Virtual nodes per physical node
    vnodes_per_node: u32,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(vnodes_per_node: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            vnodes_per_node,
        }
    }

    /// Add a node to the ring
    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.vnodes_per_node {
            let vnode_key = format!("{node_id}-vnode-{i}");
            let hash = self.hash_key(&vnode_key);
            self.ring.insert(hash, node_id.to_string());
        }
    }

    /// Remove a node from the ring
    pub fn remove_node(&mut self, node_id: &str) {
        self.ring.retain(|_, v| v != node_id);
    }

    /// Get the node responsible for a key
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);

        // Find the first node with hash >= key hash
        if let Some((_, node)) = self.ring.range(hash..).next() {
            Some(node.as_str())
        } else {
            // Wrap around to the first node
            self.ring.values().next().map(|s| s.as_str())
        }
    }

    /// Get N nodes for replication
    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<String> {
        if self.ring.is_empty() {
            return vec![];
        }

        let hash = self.hash_key(key);
        let mut nodes = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Start from the hash position
        let iter = self.ring.range(hash..).chain(self.ring.iter());

        for (_, node) in iter {
            if seen.insert(node.clone()) {
                nodes.push(node.clone());
                if nodes.len() >= count {
                    break;
                }
            }
        }

        nodes
    }

    /// Hash a key to a position on the ring
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Partition manager
pub struct PartitionManager {
    config: PartitionConfig,
    partitions: Arc<RwLock<HashMap<u32, Partition>>>,
    assignment: Arc<RwLock<PartitionAssignment>>,
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    #[allow(dead_code)]
    store: Arc<Store>,
}

impl PartitionManager {
    /// Create a new partition manager
    pub fn new(config: PartitionConfig, store: Arc<Store>) -> Self {
        let hash_ring = ConsistentHashRing::new(config.vnodes);

        Self {
            config,
            partitions: Arc::new(RwLock::new(HashMap::new())),
            assignment: Arc::new(RwLock::new(PartitionAssignment {
                partitions: HashMap::new(),
                nodes: HashMap::new(),
                version: 0,
            })),
            hash_ring: Arc::new(RwLock::new(hash_ring)),
            store,
        }
    }

    /// Start the partition manager
    pub async fn start(&self) -> FusekiResult<()> {
        // Initialize partitions
        self.initialize_partitions().await?;

        Ok(())
    }

    /// Initialize partitions
    async fn initialize_partitions(&self) -> FusekiResult<()> {
        let mut partitions = self.partitions.write().await;

        for id in 0..self.config.partition_count {
            partitions.insert(
                id,
                Partition {
                    id,
                    nodes: vec![],
                    state: PartitionState::Offline,
                    size: 0,
                    key_count: 0,
                    updated_at: chrono::Utc::now().timestamp_millis(),
                },
            );
        }

        Ok(())
    }

    /// Get partition for a key
    pub async fn get_partition(&self, key: &str) -> u32 {
        match self.config.strategy {
            PartitionStrategy::ConsistentHashing => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                key.hash(&mut hasher);
                (hasher.finish() % u64::from(self.config.partition_count)) as u32
            }
            PartitionStrategy::Range => {
                // Simple range partitioning based on first character
                let first_char = key.chars().next().unwrap_or('a') as u32;
                first_char % self.config.partition_count
            }
            PartitionStrategy::Custom => {
                // Default to hash partitioning
                self.hash_partition(key)
            }
        }
    }

    /// Hash-based partitioning
    fn hash_partition(&self, key: &str) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() % self.config.partition_count as u64) as u32
    }

    /// Assign partitions to nodes
    pub async fn assign_partitions(&self, nodes: Vec<NodeInfo>) -> FusekiResult<()> {
        if nodes.is_empty() {
            return Err(FusekiError::internal(
                "No nodes available for partition assignment",
            ));
        }

        let mut assignment = self.assignment.write().await;
        let mut hash_ring = self.hash_ring.write().await;

        // Clear existing assignments
        assignment.partitions.clear();
        assignment.nodes.clear();
        hash_ring.ring.clear();

        // Add nodes to hash ring
        for node in &nodes {
            if node.state == crate::clustering::NodeState::Active {
                hash_ring.add_node(&node.id);
                assignment.nodes.insert(node.id.clone(), vec![]);
            }
        }

        // Assign partitions using consistent hashing
        let mut partitions = self.partitions.write().await;

        for partition_id in 0..self.config.partition_count {
            let key = format!("partition-{partition_id}");
            let assigned_nodes =
                hash_ring.get_nodes(&key, self.config.rebalancing.max_concurrent_moves);

            if !assigned_nodes.is_empty() {
                assignment
                    .partitions
                    .insert(partition_id, assigned_nodes.clone());

                // Update node assignments
                for node_id in &assigned_nodes {
                    if let Some(node_partitions) = assignment.nodes.get_mut(node_id) {
                        node_partitions.push(partition_id);
                    }
                }

                // Update partition state
                if let Some(partition) = partitions.get_mut(&partition_id) {
                    partition.nodes = assigned_nodes;
                    partition.state = PartitionState::Active;
                    partition.updated_at = chrono::Utc::now().timestamp_millis();
                }
            }
        }

        assignment.version += 1;

        Ok(())
    }

    /// Check if rebalancing is needed
    pub async fn check_rebalancing(&self) -> FusekiResult<()> {
        if !self.config.rebalancing.enabled {
            return Ok(());
        }

        let _assignment = self.assignment.read().await;
        let partitions = self.partitions.read().await;

        // Calculate data distribution
        let mut node_sizes: HashMap<String, u64> = HashMap::new();

        for (_partition_id, partition) in partitions.iter() {
            if let Some(primary_node) = partition.nodes.first() {
                *node_sizes.entry(primary_node.clone()).or_default() += partition.size;
            }
        }

        if node_sizes.is_empty() {
            return Ok(());
        }

        // Check for imbalance
        let total_size: u64 = node_sizes.values().sum();
        let avg_size = total_size / node_sizes.len() as u64;

        for (node_id, size) in &node_sizes {
            let skew = (*size as f64 - avg_size as f64).abs() / avg_size as f64;

            if skew > self.config.rebalancing.threshold {
                tracing::info!("Node {} has {} skew, triggering rebalancing", node_id, skew);
                // TODO: Trigger rebalancing
                break;
            }
        }

        Ok(())
    }

    /// Move a partition to a new node.
    ///
    /// # Data transfer is not wired in this build
    ///
    /// `PartitionManager` tracks partition-to-node *assignment* metadata but
    /// holds a single shared [`Store`] (not one storage backend per node) and
    /// has no [`crate::clustering::node::NodeCommunication`] handle capable of
    /// streaming RDF quads to another node. There is therefore no mechanism
    /// here that can actually move a partition's data from `from_node` to
    /// `to_node`.
    ///
    /// Previously this method flipped the partition's state straight from
    /// `Migrating` to `Active` and rewrote the node assignment without
    /// transferring a single byte — any caller observing
    /// [`PartitionState::Active`] afterwards would wrongly believe the
    /// partition was fully and correctly served by `to_node`. Per the
    /// fail-loud contract, this now refuses to fabricate that success: the
    /// partition is left in [`PartitionState::Migrating`] (never silently
    /// promoted to `Active`), the node assignment is left untouched, and an
    /// explicit error is returned so callers can never mistake "not moved"
    /// for "moved". Implement a real transport-backed transfer (streaming the
    /// partition's quads to `to_node` over a wired [`NodeCommunication`]
    /// implementation) before this can honestly report success.
    ///
    /// [`NodeCommunication`]: crate::clustering::node::NodeCommunication
    pub async fn move_partition(
        &self,
        partition_id: u32,
        from_node: &str,
        to_node: &str,
    ) -> FusekiResult<()> {
        let mut partitions = self.partitions.write().await;

        let partition = partitions
            .get_mut(&partition_id)
            .ok_or_else(|| FusekiError::not_found(format!("Partition {partition_id}")))?;

        if !partition.nodes.iter().any(|n| n == from_node) {
            return Err(FusekiError::bad_request(format!(
                "Partition {partition_id} is not currently assigned to node '{from_node}'"
            )));
        }

        // Record that a migration was requested without fabricating its
        // completion: no data has moved.
        partition.state = PartitionState::Migrating;
        partition.updated_at = chrono::Utc::now().timestamp_millis();

        Err(FusekiError::service_unavailable(format!(
            "partition {partition_id} migration from '{from_node}' to '{to_node}' cannot \
             complete: no inter-node data-transfer transport is wired into PartitionManager in \
             this build. The partition has been marked Migrating and its node assignment left \
             unchanged; a real transport-backed transfer must be implemented before this \
             partition can be promoted to Active on '{to_node}'."
        )))
    }

    /// Get partition statistics
    pub async fn get_statistics(&self) -> PartitionStatistics {
        let partitions = self.partitions.read().await;
        let assignment = self.assignment.read().await;

        let total_partitions = partitions.len();
        let active_partitions = partitions
            .values()
            .filter(|p| p.state == PartitionState::Active)
            .count();
        let under_replicated = partitions
            .values()
            .filter(|p| p.state == PartitionState::UnderReplicated)
            .count();
        let offline = partitions
            .values()
            .filter(|p| p.state == PartitionState::Offline)
            .count();

        let total_size: u64 = partitions.values().map(|p| p.size).sum();
        let total_keys: u64 = partitions.values().map(|p| p.key_count).sum();

        let mut node_partition_counts = HashMap::new();
        for (node_id, node_partitions) in &assignment.nodes {
            node_partition_counts.insert(node_id.clone(), node_partitions.len());
        }

        PartitionStatistics {
            total_partitions,
            active_partitions,
            under_replicated_partitions: under_replicated,
            offline_partitions: offline,
            total_size,
            total_keys,
            node_partition_counts,
        }
    }
}

/// Partition statistics
#[derive(Debug, Clone, Serialize)]
pub struct PartitionStatistics {
    pub total_partitions: usize,
    pub active_partitions: usize,
    pub under_replicated_partitions: usize,
    pub offline_partitions: usize,
    pub total_size: u64,
    pub total_keys: u64,
    pub node_partition_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(3);

        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");

        // Test key assignment
        let node = ring.get_node("test-key").unwrap();
        assert!(["node1", "node2", "node3"].contains(&node));

        // Test replication
        let nodes = ring.get_nodes("test-key", 2);
        assert_eq!(nodes.len(), 2);
        assert_ne!(nodes[0], nodes[1]);
    }

    #[test]
    fn test_partition_assignment() {
        let assignment = PartitionAssignment {
            partitions: HashMap::new(),
            nodes: HashMap::new(),
            version: 1,
        };

        assert_eq!(assignment.version, 1);
    }

    /// Regression test for the "move_partition marks a partition migrated
    /// without moving any data" finding: the call must fail loud (no wired
    /// transport exists to actually move data), and must never leave the
    /// partition in `Active` state as if the migration had completed.
    #[tokio::test]
    async fn regression_move_partition_fails_loud_never_reports_active() {
        let store = Arc::new(Store::new().expect("store"));
        let manager = PartitionManager::new(PartitionConfig::default(), store);
        manager.start().await.expect("start");

        // Manually assign partition 0 to "node-a" (bypassing full
        // assign_partitions/hash-ring machinery, which is not the subject of
        // this test).
        {
            let mut partitions = manager.partitions.write().await;
            let partition = partitions.get_mut(&0).expect("partition 0 exists");
            partition.nodes = vec!["node-a".to_string()];
            partition.state = PartitionState::Active;
        }

        let result = manager.move_partition(0, "node-a", "node-b").await;
        assert!(
            result.is_err(),
            "move_partition must fail loud: no data-transfer transport is wired"
        );

        // The partition must be left in Migrating, never fabricated as Active.
        let partitions = manager.partitions.read().await;
        let partition = partitions.get(&0).expect("partition 0 exists");
        assert_eq!(partition.state, PartitionState::Migrating);
        // The node assignment must be left untouched (still on node-a, not
        // silently rewritten to node-b as if the transfer had happened).
        assert_eq!(partition.nodes, vec!["node-a".to_string()]);
    }

    /// move_partition on a partition not currently assigned to `from_node`
    /// must reject the request rather than silently "moving" nothing.
    #[tokio::test]
    async fn regression_move_partition_rejects_wrong_from_node() {
        let store = Arc::new(Store::new().expect("store"));
        let manager = PartitionManager::new(PartitionConfig::default(), store);
        manager.start().await.expect("start");

        {
            let mut partitions = manager.partitions.write().await;
            let partition = partitions.get_mut(&0).expect("partition 0 exists");
            partition.nodes = vec!["node-a".to_string()];
        }

        let result = manager.move_partition(0, "node-x", "node-b").await;
        assert!(result.is_err());
    }

    /// move_partition on a nonexistent partition ID must return a not-found
    /// error rather than silently succeeding.
    #[tokio::test]
    async fn regression_move_partition_unknown_partition_not_found() {
        let store = Arc::new(Store::new().expect("store"));
        let manager = PartitionManager::new(PartitionConfig::default(), store);
        manager.start().await.expect("start");

        let result = manager.move_partition(999_999, "node-a", "node-b").await;
        assert!(result.is_err());
    }
}
