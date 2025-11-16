//! Horizontal scaling support for RDF-star annotations
//!
//! This module provides foundational support for horizontal scaling of
//! RDF-star annotation processing across multiple nodes, including
//! cluster coordination, partition management, and distributed operations.

use crate::model::{StarGraph, StarTriple};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing::info;

/// Errors related to cluster scaling operations
#[derive(Error, Debug)]
pub enum ClusterError {
    #[error("Node communication failed: {0}")]
    CommunicationFailed(String),

    #[error("Partition error: {0}")]
    PartitionError(String),

    #[error("Replication failed: {0}")]
    ReplicationFailed(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Cluster consistency violation: {0}")]
    ConsistencyViolation(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Node identifier in the cluster
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Node information in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Node ID
    pub id: NodeId,

    /// Node address (host:port)
    pub address: String,

    /// Node status
    pub status: NodeStatus,

    /// Node capacity metrics
    pub capacity: NodeCapacity,

    /// Partitions assigned to this node
    pub partitions: Vec<u32>,

    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
}

/// Node status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is healthy and accepting requests
    Active,
    /// Node is draining (no new requests)
    Draining,
    /// Node is unavailable
    Unavailable,
    /// Node is starting up
    Starting,
}

/// Node capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,

    /// Memory usage (0.0-1.0)
    pub memory_usage: f64,

    /// Current annotation count
    pub annotation_count: usize,

    /// Maximum annotations this node can handle
    pub max_annotations: usize,

    /// Network bandwidth usage (bytes/sec)
    pub network_usage: u64,
}

impl NodeCapacity {
    /// Calculate node load score (0.0-1.0, higher = more loaded)
    pub fn load_score(&self) -> f64 {
        let cpu_weight = 0.4;
        let memory_weight = 0.3;
        let annotation_weight = 0.3;

        let annotation_load = if self.max_annotations > 0 {
            self.annotation_count as f64 / self.max_annotations as f64
        } else {
            0.0
        };

        (self.cpu_usage * cpu_weight)
            + (self.memory_usage * memory_weight)
            + (annotation_load * annotation_weight)
    }

    /// Check if node has capacity
    pub fn has_capacity(&self) -> bool {
        self.load_score() < 0.85
    }
}

/// Partition assignment strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Hash-based partitioning
    Hash,
    /// Range-based partitioning
    Range,
    /// Consistent hashing
    ConsistentHash,
    /// Custom partitioning logic
    Custom,
}

/// Partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// Partition ID
    pub id: u32,

    /// Primary node for this partition
    pub primary_node: NodeId,

    /// Replica nodes for this partition
    pub replica_nodes: Vec<NodeId>,

    /// Number of annotations in this partition
    pub annotation_count: usize,

    /// Partition state
    pub state: PartitionState,
}

/// Partition state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionState {
    /// Partition is active and serving requests
    Active,
    /// Partition is being migrated
    Migrating,
    /// Partition is being replicated
    Replicating,
    /// Partition is offline
    Offline,
}

/// Cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Number of partitions
    pub partition_count: u32,

    /// Replication factor
    pub replication_factor: usize,

    /// Partition strategy
    pub partition_strategy: PartitionStrategy,

    /// Enable auto-rebalancing
    pub auto_rebalance: bool,

    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u64,

    /// Node timeout (seconds)
    pub node_timeout: u64,

    /// Enable strong consistency
    pub strong_consistency: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            partition_count: 16,
            replication_factor: 3,
            partition_strategy: PartitionStrategy::ConsistentHash,
            auto_rebalance: true,
            heartbeat_interval: 5,
            node_timeout: 30,
            strong_consistency: false,
        }
    }
}

/// Cluster manager for horizontal scaling
pub struct ClusterManager {
    /// Cluster configuration
    config: ClusterConfig,

    /// Registered nodes
    nodes: Arc<Mutex<HashMap<NodeId, ClusterNode>>>,

    /// Partition assignments
    partitions: Arc<Mutex<HashMap<u32, Partition>>>,

    /// Local node ID (if this is a cluster node) - reserved for future distributed coordination
    #[allow(dead_code)]
    local_node_id: Option<NodeId>,
}

impl ClusterManager {
    /// Create a new cluster manager
    pub fn new(config: ClusterConfig) -> Self {
        let manager = Self {
            config,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            partitions: Arc::new(Mutex::new(HashMap::new())),
            local_node_id: None,
        };

        manager.initialize_partitions();
        manager
    }

    /// Initialize partition map
    fn initialize_partitions(&self) {
        let mut partitions = self.partitions.lock().unwrap();

        for partition_id in 0..self.config.partition_count {
            partitions.insert(
                partition_id,
                Partition {
                    id: partition_id,
                    primary_node: NodeId::new("unassigned"),
                    replica_nodes: Vec::new(),
                    annotation_count: 0,
                    state: PartitionState::Offline,
                },
            );
        }
    }

    /// Register a node in the cluster
    pub fn register_node(&mut self, node: ClusterNode) -> Result<(), ClusterError> {
        info!("Registering node: {:?}", node.id);

        let mut nodes = self.nodes.lock().unwrap();
        nodes.insert(node.id.clone(), node);

        // Trigger rebalancing if enabled
        if self.config.auto_rebalance {
            drop(nodes);
            self.rebalance_partitions()?;
        }

        Ok(())
    }

    /// Unregister a node from the cluster
    pub fn unregister_node(&mut self, node_id: &NodeId) -> Result<(), ClusterError> {
        info!("Unregistering node: {:?}", node_id);

        let mut nodes = self.nodes.lock().unwrap();
        nodes.remove(node_id);

        drop(nodes);

        // Reassign partitions from removed node
        self.reassign_partitions_from_node(node_id)?;

        Ok(())
    }

    /// Get partition for a triple
    pub fn get_partition_for_triple(&self, triple: &StarTriple) -> u32 {
        match self.config.partition_strategy {
            PartitionStrategy::Hash | PartitionStrategy::ConsistentHash => {
                self.hash_partition(triple)
            }
            PartitionStrategy::Range => self.range_partition(triple),
            PartitionStrategy::Custom => 0, // Default to partition 0 for custom
        }
    }

    /// Hash-based partitioning
    fn hash_partition(&self, triple: &StarTriple) -> u32 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        format!("{:?}", triple).hash(&mut hasher);
        let hash = hasher.finish();

        (hash % self.config.partition_count as u64) as u32
    }

    /// Range-based partitioning (simplified)
    fn range_partition(&self, triple: &StarTriple) -> u32 {
        // Simple range partitioning based on first character of subject
        let subject_str = format!("{:?}", triple.subject);
        let first_char = subject_str.chars().next().unwrap_or('a') as u32;

        first_char % self.config.partition_count
    }

    /// Get node for partition
    pub fn get_node_for_partition(&self, partition_id: u32) -> Option<NodeId> {
        let partitions = self.partitions.lock().unwrap();
        partitions
            .get(&partition_id)
            .map(|p| p.primary_node.clone())
    }

    /// Rebalance partitions across nodes
    pub fn rebalance_partitions(&self) -> Result<(), ClusterError> {
        info!("Rebalancing partitions");

        let active_node_ids: Vec<NodeId> = {
            let nodes = self.nodes.lock().unwrap();
            if nodes.is_empty() {
                return Err(ClusterError::InvalidConfiguration(
                    "No nodes available for rebalancing".to_string(),
                ));
            }

            nodes
                .values()
                .filter(|n| n.status == NodeStatus::Active)
                .map(|n| n.id.clone())
                .collect()
        };

        if active_node_ids.is_empty() {
            return Err(ClusterError::InvalidConfiguration(
                "No active nodes available".to_string(),
            ));
        }

        let mut partitions = self.partitions.lock().unwrap();

        // Simple round-robin assignment
        let node_count = active_node_ids.len();
        for (partition_id, partition) in partitions.iter_mut() {
            let node_index = (*partition_id as usize) % node_count;
            partition.primary_node = active_node_ids[node_index].clone();
            partition.state = PartitionState::Active;

            // Assign replicas
            partition.replica_nodes.clear();
            for i in 1..=self.config.replication_factor.min(node_count - 1) {
                let replica_index = (node_index + i) % node_count;
                partition
                    .replica_nodes
                    .push(active_node_ids[replica_index].clone());
            }
        }

        info!("Rebalancing complete");
        Ok(())
    }

    /// Reassign partitions from a removed node
    fn reassign_partitions_from_node(&self, removed_node: &NodeId) -> Result<(), ClusterError> {
        info!(
            "Reassigning partitions from removed node: {:?}",
            removed_node
        );

        let active_node_id: NodeId = {
            let nodes = self.nodes.lock().unwrap();
            let active_nodes: Vec<_> = nodes
                .values()
                .filter(|n| n.status == NodeStatus::Active && n.id != *removed_node)
                .collect();

            if active_nodes.is_empty() {
                return Err(ClusterError::InvalidConfiguration(
                    "No active nodes available for reassignment".to_string(),
                ));
            }

            active_nodes[0].id.clone()
        };

        let mut partitions = self.partitions.lock().unwrap();

        for partition in partitions.values_mut() {
            if partition.primary_node == *removed_node {
                // Promote replica if available
                if let Some(new_primary) = partition.replica_nodes.first().cloned() {
                    partition.primary_node = new_primary;
                    partition.replica_nodes.remove(0);
                } else {
                    // Assign to least loaded node
                    partition.primary_node = active_node_id.clone();
                }
            }

            // Remove from replicas
            partition.replica_nodes.retain(|id| id != removed_node);
        }

        Ok(())
    }

    /// Distribute triples across cluster
    pub fn distribute_triples(
        &self,
        graph: &StarGraph,
    ) -> Result<HashMap<NodeId, Vec<StarTriple>>, ClusterError> {
        info!("Distributing triples across cluster");

        let mut distribution: HashMap<NodeId, Vec<StarTriple>> = HashMap::new();

        // Group triples by partition
        for triple in graph.triples() {
            let partition_id = self.get_partition_for_triple(triple);
            if let Some(node_id) = self.get_node_for_partition(partition_id) {
                distribution
                    .entry(node_id)
                    .or_default()
                    .push(triple.clone());
            }
        }

        info!(
            "Distributed {} triples to {} nodes",
            graph.len(),
            distribution.len()
        );

        Ok(distribution)
    }

    /// Process triples in parallel across partitions
    pub fn parallel_process<F>(
        &self,
        graph: &StarGraph,
        processor: F,
    ) -> Result<usize, ClusterError>
    where
        F: Fn(&StarTriple) -> Result<(), String> + Send + Sync,
    {
        info!("Processing triples in parallel");

        let triples = graph.triples();
        let processor = Arc::new(processor);

        // Use rayon for parallel processing
        use rayon::prelude::*;
        let processed: usize = triples
            .par_iter()
            .filter(|triple| processor(triple).is_ok())
            .count();

        info!("Processed {} triples", processed);
        Ok(processed)
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self) -> ClusterStatistics {
        let nodes = self.nodes.lock().unwrap();
        let partitions = self.partitions.lock().unwrap();

        let total_nodes = nodes.len();
        let active_nodes = nodes
            .values()
            .filter(|n| n.status == NodeStatus::Active)
            .count();

        let total_partitions = partitions.len();
        let active_partitions = partitions
            .values()
            .filter(|p| p.state == PartitionState::Active)
            .count();

        let total_annotations: usize = partitions.values().map(|p| p.annotation_count).sum();

        let avg_load = if active_nodes > 0 {
            nodes
                .values()
                .filter(|n| n.status == NodeStatus::Active)
                .map(|n| n.capacity.load_score())
                .sum::<f64>()
                / active_nodes as f64
        } else {
            0.0
        };

        ClusterStatistics {
            total_nodes,
            active_nodes,
            total_partitions,
            active_partitions,
            total_annotations,
            avg_load,
            replication_factor: self.config.replication_factor,
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.lock().unwrap().len()
    }

    /// Get partition count
    pub fn partition_count(&self) -> usize {
        self.partitions.lock().unwrap().len()
    }
}

/// Cluster statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatistics {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub total_partitions: usize,
    pub active_partitions: usize,
    pub total_annotations: usize,
    pub avg_load: f64,
    pub replication_factor: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node(id: &str, status: NodeStatus) -> ClusterNode {
        ClusterNode {
            id: NodeId::new(id),
            address: format!("node-{}:8080", id),
            status,
            capacity: NodeCapacity {
                cpu_usage: 0.3,
                memory_usage: 0.4,
                annotation_count: 1000,
                max_annotations: 10000,
                network_usage: 1024,
            },
            partitions: Vec::new(),
            last_heartbeat: Utc::now(),
        }
    }

    #[test]
    fn test_cluster_manager_creation() {
        let config = ClusterConfig::default();
        let manager = ClusterManager::new(config);

        assert_eq!(manager.partition_count(), 16);
        assert_eq!(manager.node_count(), 0);
    }

    #[test]
    fn test_node_registration() {
        let config = ClusterConfig::default();
        let mut manager = ClusterManager::new(config);

        let node = create_test_node("node1", NodeStatus::Active);
        assert!(manager.register_node(node).is_ok());
        assert_eq!(manager.node_count(), 1);
    }

    #[test]
    fn test_node_capacity() {
        let capacity = NodeCapacity {
            cpu_usage: 0.5,
            memory_usage: 0.6,
            annotation_count: 5000,
            max_annotations: 10000,
            network_usage: 2048,
        };

        let load = capacity.load_score();
        assert!(load > 0.0 && load < 1.0);
        assert!(capacity.has_capacity());
    }

    #[test]
    fn test_partition_assignment() {
        use crate::model::StarTerm;

        let config = ClusterConfig {
            partition_count: 8,
            ..Default::default()
        };
        let manager = ClusterManager::new(config);

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let partition = manager.get_partition_for_triple(&triple);
        assert!(partition < 8);
    }

    #[test]
    fn test_cluster_statistics() {
        let config = ClusterConfig::default();
        let mut manager = ClusterManager::new(config);

        let node1 = create_test_node("node1", NodeStatus::Active);
        let node2 = create_test_node("node2", NodeStatus::Active);

        manager.register_node(node1).unwrap();
        manager.register_node(node2).unwrap();

        let stats = manager.get_cluster_stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.active_nodes, 2);
    }

    #[test]
    fn test_rebalancing() {
        let config = ClusterConfig::default();
        let mut manager = ClusterManager::new(config);

        let node = create_test_node("node1", NodeStatus::Active);
        manager.register_node(node).unwrap();

        assert!(manager.rebalance_partitions().is_ok());
    }
}
