//! # Advanced Partitioning Strategies
//!
//! Provides sophisticated data partitioning algorithms for distributed RDF storage
//! including consistent hashing and range-based partitioning with automatic rebalancing.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::raft::OxirsNodeId;

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Consistent hashing with virtual nodes
    ConsistentHashing,
    /// Range-based partitioning
    RangeBased,
    /// Hybrid approach
    Hybrid,
}

/// Partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Partitioning strategy
    pub strategy: PartitionStrategy,
    /// Number of virtual nodes per physical node (for consistent hashing)
    pub virtual_nodes_per_node: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Enable automatic rebalancing
    pub enable_auto_rebalancing: bool,
    /// Rebalancing threshold (0.0-1.0)
    pub rebalancing_threshold: f64,
    /// Maximum keys per partition (for range-based)
    pub max_keys_per_partition: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 150,
            replication_factor: 3,
            enable_auto_rebalancing: true,
            rebalancing_threshold: 0.15,
            max_keys_per_partition: 100_000,
        }
    }
}

/// Virtual node in consistent hashing ring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualNode {
    /// Virtual node ID
    pub id: u64,
    /// Physical node ID
    pub physical_node: OxirsNodeId,
    /// Hash position on the ring
    pub hash_position: u64,
}

/// Range partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangePartition {
    /// Partition ID
    pub id: usize,
    /// Start key (inclusive)
    pub start_key: String,
    /// End key (exclusive)
    pub end_key: String,
    /// Assigned node
    pub node_id: OxirsNodeId,
    /// Number of keys in this partition
    pub key_count: usize,
    /// Estimated size in bytes
    pub size_bytes: usize,
}

/// Partition assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    /// Primary node
    pub primary_node: OxirsNodeId,
    /// Replica nodes
    pub replica_nodes: Vec<OxirsNodeId>,
    /// Partition weight (0.0-1.0)
    pub weight: f64,
}

/// Partitioning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningStats {
    /// Total partitions
    pub total_partitions: usize,
    /// Total virtual nodes (consistent hashing only)
    pub total_virtual_nodes: usize,
    /// Average keys per partition
    pub avg_keys_per_partition: f64,
    /// Standard deviation of key distribution
    pub key_distribution_stddev: f64,
    /// Rebalancing operations performed
    pub rebalancing_ops: usize,
    /// Last rebalancing timestamp
    pub last_rebalancing: Option<std::time::SystemTime>,
}

impl Default for PartitioningStats {
    fn default() -> Self {
        Self {
            total_partitions: 0,
            total_virtual_nodes: 0,
            avg_keys_per_partition: 0.0,
            key_distribution_stddev: 0.0,
            rebalancing_ops: 0,
            last_rebalancing: None,
        }
    }
}

/// Advanced partitioning manager
pub struct AdvancedPartitioning {
    config: PartitionConfig,
    /// Consistent hashing ring (sorted by hash position)
    hash_ring: Arc<RwLock<Vec<VirtualNode>>>,
    /// Range partitions
    range_partitions: Arc<RwLock<Vec<RangePartition>>>,
    /// Node to partition mapping
    node_partitions: Arc<RwLock<BTreeMap<OxirsNodeId, Vec<usize>>>>,
    /// Partition statistics
    stats: Arc<RwLock<PartitioningStats>>,
    /// Active nodes
    active_nodes: Arc<RwLock<Vec<OxirsNodeId>>>,
}

impl AdvancedPartitioning {
    /// Create a new partitioning manager
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            hash_ring: Arc::new(RwLock::new(Vec::new())),
            range_partitions: Arc::new(RwLock::new(Vec::new())),
            node_partitions: Arc::new(RwLock::new(BTreeMap::new())),
            stats: Arc::new(RwLock::new(PartitioningStats::default())),
            active_nodes: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a node in the partitioning system
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut active_nodes = self.active_nodes.write().await;
        if !active_nodes.contains(&node_id) {
            active_nodes.push(node_id);
            info!("Registered node {} for partitioning", node_id);
        }
        drop(active_nodes);

        match self.config.strategy {
            PartitionStrategy::ConsistentHashing | PartitionStrategy::Hybrid => {
                self.add_virtual_nodes(node_id).await;
            }
            PartitionStrategy::RangeBased => {
                self.rebalance_ranges().await;
            }
        }
    }

    /// Remove a node from the partitioning system
    pub async fn unregister_node(&self, node_id: OxirsNodeId) {
        let mut active_nodes = self.active_nodes.write().await;
        active_nodes.retain(|&id| id != node_id);
        drop(active_nodes);

        match self.config.strategy {
            PartitionStrategy::ConsistentHashing | PartitionStrategy::Hybrid => {
                self.remove_virtual_nodes(node_id).await;
            }
            PartitionStrategy::RangeBased => {
                self.rebalance_ranges().await;
            }
        }

        info!("Unregistered node {} from partitioning", node_id);
    }

    /// Add virtual nodes for a physical node (consistent hashing)
    async fn add_virtual_nodes(&self, node_id: OxirsNodeId) {
        let mut hash_ring = self.hash_ring.write().await;

        for i in 0..self.config.virtual_nodes_per_node {
            let vnode_id = ((node_id as u64) << 32) | (i as u64);
            let hash_position = Self::hash_virtual_node(vnode_id);

            let vnode = VirtualNode {
                id: vnode_id,
                physical_node: node_id,
                hash_position,
            };

            hash_ring.push(vnode);
        }

        // Sort ring by hash position
        hash_ring.sort_by_key(|vnode| vnode.hash_position);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_virtual_nodes = hash_ring.len();

        info!(
            "Added {} virtual nodes for physical node {}",
            self.config.virtual_nodes_per_node, node_id
        );
    }

    /// Remove virtual nodes for a physical node
    async fn remove_virtual_nodes(&self, node_id: OxirsNodeId) {
        let mut hash_ring = self.hash_ring.write().await;
        hash_ring.retain(|vnode| vnode.physical_node != node_id);

        let mut stats = self.stats.write().await;
        stats.total_virtual_nodes = hash_ring.len();

        info!("Removed virtual nodes for physical node {}", node_id);
    }

    /// Hash a virtual node ID
    fn hash_virtual_node(vnode_id: u64) -> u64 {
        // Use FNV-1a hash for speed
        let mut hash: u64 = 0xcbf29ce484222325;
        let bytes = vnode_id.to_le_bytes();

        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Hash a key for consistent hashing
    pub fn hash_key(key: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;

        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Get partition assignment for a key (consistent hashing)
    pub async fn get_partition_assignment(&self, key: &str) -> Option<PartitionAssignment> {
        match self.config.strategy {
            PartitionStrategy::ConsistentHashing | PartitionStrategy::Hybrid => {
                self.get_consistent_hash_assignment(key).await
            }
            PartitionStrategy::RangeBased => self.get_range_based_assignment(key).await,
        }
    }

    /// Get assignment using consistent hashing
    async fn get_consistent_hash_assignment(&self, key: &str) -> Option<PartitionAssignment> {
        let hash_ring = self.hash_ring.read().await;

        if hash_ring.is_empty() {
            return None;
        }

        let key_hash = Self::hash_key(key);

        // Binary search for the first virtual node >= key_hash
        let pos = match hash_ring.binary_search_by_key(&key_hash, |vnode| vnode.hash_position) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx >= hash_ring.len() {
                    0 // Wrap around
                } else {
                    idx
                }
            }
        };

        // Get primary node
        let primary_node = hash_ring[pos].physical_node;

        // Get replica nodes (next N-1 distinct physical nodes on the ring)
        let mut replica_nodes = Vec::new();
        let mut seen = std::collections::HashSet::new();
        seen.insert(primary_node);

        let mut current_pos = (pos + 1) % hash_ring.len();
        while replica_nodes.len() < self.config.replication_factor - 1
            && seen.len() < hash_ring.len()
        {
            let physical_node = hash_ring[current_pos].physical_node;
            if !seen.contains(&physical_node) {
                replica_nodes.push(physical_node);
                seen.insert(physical_node);
            }
            current_pos = (current_pos + 1) % hash_ring.len();
        }

        Some(PartitionAssignment {
            primary_node,
            replica_nodes,
            weight: 1.0 / hash_ring.len() as f64,
        })
    }

    /// Get assignment using range-based partitioning
    async fn get_range_based_assignment(&self, key: &str) -> Option<PartitionAssignment> {
        let range_partitions = self.range_partitions.read().await;

        // Binary search for the partition containing this key
        for partition in range_partitions.iter() {
            if key >= partition.start_key.as_str() && key < partition.end_key.as_str() {
                // For range-based, replicas are next N-1 partitions
                let active_nodes = self.active_nodes.read().await;
                let mut replica_nodes = Vec::new();

                for node_id in active_nodes.iter() {
                    if *node_id != partition.node_id
                        && replica_nodes.len() < self.config.replication_factor - 1
                    {
                        replica_nodes.push(*node_id);
                    }
                }

                return Some(PartitionAssignment {
                    primary_node: partition.node_id,
                    replica_nodes,
                    weight: partition.key_count as f64 / self.config.max_keys_per_partition as f64,
                });
            }
        }

        None
    }

    /// Rebalance range partitions
    async fn rebalance_ranges(&self) {
        let active_nodes = self.active_nodes.read().await;

        if active_nodes.is_empty() {
            return;
        }

        let mut range_partitions = self.range_partitions.write().await;

        // If no partitions exist OR number of partitions doesn't match nodes, (re)create partitions
        if range_partitions.is_empty() || range_partitions.len() != active_nodes.len() {
            range_partitions.clear(); // Clear old partitions if resizing

            let nodes_count = active_nodes.len();
            for (i, &node_id) in active_nodes.iter().enumerate() {
                let partition = RangePartition {
                    id: i,
                    start_key: if i == 0 {
                        String::new()
                    } else {
                        format!("partition_{}", i)
                    },
                    end_key: if i == nodes_count - 1 {
                        String::from("\u{10ffff}") // Maximum Unicode
                    } else {
                        format!("partition_{}", i + 1)
                    },
                    node_id,
                    key_count: 0,
                    size_bytes: 0,
                };
                range_partitions.push(partition);
            }

            info!(
                "Created {} range partitions for {} nodes",
                nodes_count, nodes_count
            );
        } else {
            // Check if rebalancing is needed
            if !self.config.enable_auto_rebalancing {
                return;
            }

            let avg_keys = range_partitions.iter().map(|p| p.key_count).sum::<usize>() as f64
                / range_partitions.len() as f64;

            let mut needs_rebalancing = false;
            for partition in range_partitions.iter() {
                let deviation = (partition.key_count as f64 - avg_keys).abs() / avg_keys.max(1.0);
                if deviation > self.config.rebalancing_threshold {
                    needs_rebalancing = true;
                    break;
                }
            }

            if needs_rebalancing {
                // Simple rebalancing: redistribute partitions among nodes
                let nodes_count = active_nodes.len();
                for (i, partition) in range_partitions.iter_mut().enumerate() {
                    partition.node_id = active_nodes[i % nodes_count];
                }

                let mut stats = self.stats.write().await;
                stats.rebalancing_ops += 1;
                stats.last_rebalancing = Some(std::time::SystemTime::now());

                info!("Rebalanced {} range partitions", range_partitions.len());
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_partitions = range_partitions.len();
    }

    /// Update partition statistics (call after data operations)
    pub async fn update_partition_stats(&self, key: &str, size_delta: isize) {
        match self.config.strategy {
            PartitionStrategy::RangeBased | PartitionStrategy::Hybrid => {
                let mut range_partitions = self.range_partitions.write().await;

                for partition in range_partitions.iter_mut() {
                    if key >= partition.start_key.as_str() && key < partition.end_key.as_str() {
                        if size_delta > 0 {
                            partition.key_count += 1;
                            partition.size_bytes += size_delta as usize;
                        } else if size_delta < 0 && partition.key_count > 0 {
                            partition.key_count -= 1;
                            partition.size_bytes =
                                partition.size_bytes.saturating_sub((-size_delta) as usize);
                        }
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    /// Get all partition assignments for a node
    pub async fn get_node_partitions(&self, node_id: OxirsNodeId) -> Vec<usize> {
        let node_partitions = self.node_partitions.read().await;
        node_partitions.get(&node_id).cloned().unwrap_or_default()
    }

    /// Get partition statistics
    pub async fn get_stats(&self) -> PartitioningStats {
        let mut stats = self.stats.read().await.clone();

        match self.config.strategy {
            PartitionStrategy::RangeBased | PartitionStrategy::Hybrid => {
                let range_partitions = self.range_partitions.read().await;
                if !range_partitions.is_empty() {
                    let total_keys: usize = range_partitions.iter().map(|p| p.key_count).sum();
                    stats.avg_keys_per_partition =
                        total_keys as f64 / range_partitions.len() as f64;

                    // Calculate standard deviation
                    let variance: f64 = range_partitions
                        .iter()
                        .map(|p| {
                            let diff = p.key_count as f64 - stats.avg_keys_per_partition;
                            diff * diff
                        })
                        .sum::<f64>()
                        / range_partitions.len() as f64;

                    stats.key_distribution_stddev = variance.sqrt();
                }
            }
            _ => {}
        }

        stats
    }

    /// Get all virtual nodes (for debugging/monitoring)
    pub async fn get_virtual_nodes(&self) -> Vec<VirtualNode> {
        self.hash_ring.read().await.clone()
    }

    /// Get all range partitions
    pub async fn get_range_partitions(&self) -> Vec<RangePartition> {
        self.range_partitions.read().await.clone()
    }

    /// Check if rebalancing is needed
    pub async fn check_rebalancing_needed(&self) -> bool {
        if !self.config.enable_auto_rebalancing {
            return false;
        }

        let stats = self.get_stats().await;

        if stats.avg_keys_per_partition == 0.0 {
            return false;
        }

        stats.key_distribution_stddev / stats.avg_keys_per_partition
            > self.config.rebalancing_threshold
    }

    /// Perform rebalancing if needed
    pub async fn perform_rebalancing(&self) {
        if !self.check_rebalancing_needed().await {
            return;
        }

        match self.config.strategy {
            PartitionStrategy::RangeBased | PartitionStrategy::Hybrid => {
                self.rebalance_ranges().await;
            }
            PartitionStrategy::ConsistentHashing => {
                // Consistent hashing auto-rebalances via virtual nodes
                warn!("Consistent hashing rebalancing triggered, but not needed");
            }
        }
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.hash_ring.write().await.clear();
        self.range_partitions.write().await.clear();
        self.node_partitions.write().await.clear();
        self.active_nodes.write().await.clear();
        *self.stats.write().await = PartitioningStats::default();
    }

    /// Get load distribution (percentage of data per node)
    pub async fn get_load_distribution(&self) -> BTreeMap<OxirsNodeId, f64> {
        let mut distribution = BTreeMap::new();

        match self.config.strategy {
            PartitionStrategy::ConsistentHashing | PartitionStrategy::Hybrid => {
                let hash_ring = self.hash_ring.read().await;
                let total_vnodes = hash_ring.len() as f64;

                for vnode in hash_ring.iter() {
                    *distribution.entry(vnode.physical_node).or_insert(0.0) += 1.0 / total_vnodes;
                }
            }
            PartitionStrategy::RangeBased => {
                let range_partitions = self.range_partitions.read().await;
                let total_keys: usize = range_partitions.iter().map(|p| p.key_count).sum();

                if total_keys > 0 {
                    for partition in range_partitions.iter() {
                        *distribution.entry(partition.node_id).or_insert(0.0) +=
                            partition.key_count as f64 / total_keys as f64;
                    }
                }
            }
        }

        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_partitioning_creation() {
        let config = PartitionConfig::default();
        let partitioning = AdvancedPartitioning::new(config);

        let stats = partitioning.get_stats().await;
        assert_eq!(stats.total_partitions, 0);
        assert_eq!(stats.total_virtual_nodes, 0);
    }

    #[tokio::test]
    async fn test_register_node_consistent_hashing() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 10,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;

        let stats = partitioning.get_stats().await;
        assert_eq!(stats.total_virtual_nodes, 10);

        let vnodes = partitioning.get_virtual_nodes().await;
        assert_eq!(vnodes.len(), 10);
    }

    #[tokio::test]
    async fn test_register_multiple_nodes() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 5,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;
        partitioning.register_node(3).await;

        let stats = partitioning.get_stats().await;
        assert_eq!(stats.total_virtual_nodes, 15);
    }

    #[tokio::test]
    async fn test_unregister_node() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 10,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        partitioning.unregister_node(1).await;

        let stats = partitioning.get_stats().await;
        assert_eq!(stats.total_virtual_nodes, 10);
    }

    #[tokio::test]
    async fn test_consistent_hash_assignment() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 50,
            replication_factor: 3,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;
        partitioning.register_node(3).await;

        let assignment = partitioning.get_partition_assignment("test_key").await;
        assert!(assignment.is_some());

        let assignment = assignment.unwrap();
        assert_eq!(assignment.replica_nodes.len(), 2); // replication_factor - 1
    }

    #[tokio::test]
    async fn test_range_based_partitioning() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::RangeBased,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        let partitions = partitioning.get_range_partitions().await;
        assert_eq!(partitions.len(), 2);
    }

    #[tokio::test]
    async fn test_range_assignment() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::RangeBased,
            replication_factor: 2,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        let assignment = partitioning.get_partition_assignment("test_key").await;
        assert!(assignment.is_some());

        let assignment = assignment.unwrap();
        assert!(assignment.replica_nodes.len() <= 1);
    }

    #[tokio::test]
    async fn test_update_partition_stats() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::RangeBased,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;

        partitioning.update_partition_stats("test_key", 100).await;

        let stats = partitioning.get_stats().await;
        assert!(stats.avg_keys_per_partition > 0.0);
    }

    #[tokio::test]
    async fn test_load_distribution() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 100,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        let distribution = partitioning.get_load_distribution().await;
        assert_eq!(distribution.len(), 2);

        // Each node should have roughly 50% of virtual nodes
        for (_, load) in distribution.iter() {
            assert!(*load > 0.4 && *load < 0.6);
        }
    }

    #[tokio::test]
    async fn test_hash_key_deterministic() {
        let hash1 = AdvancedPartitioning::hash_key("test_key");
        let hash2 = AdvancedPartitioning::hash_key("test_key");
        assert_eq!(hash1, hash2);

        let hash3 = AdvancedPartitioning::hash_key("different_key");
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_rebalancing_needed() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::RangeBased,
            enable_auto_rebalancing: true,
            rebalancing_threshold: 0.1,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        // Initially balanced
        let needed = partitioning.check_rebalancing_needed().await;
        assert!(!needed);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = PartitionConfig::default();
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;
        partitioning.register_node(2).await;

        partitioning.clear().await;

        let stats = partitioning.get_stats().await;
        assert_eq!(stats.total_virtual_nodes, 0);
        assert_eq!(stats.total_partitions, 0);
    }

    #[tokio::test]
    async fn test_virtual_node_ring_sorted() {
        let config = PartitionConfig {
            strategy: PartitionStrategy::ConsistentHashing,
            virtual_nodes_per_node: 20,
            ..Default::default()
        };
        let partitioning = AdvancedPartitioning::new(config);

        partitioning.register_node(1).await;

        let vnodes = partitioning.get_virtual_nodes().await;

        // Verify sorted order
        for i in 1..vnodes.len() {
            assert!(vnodes[i].hash_position >= vnodes[i - 1].hash_position);
        }
    }
}
