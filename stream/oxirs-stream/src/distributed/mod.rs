//! # Distributed Stream Processing
//!
//! Provides infrastructure for coordinating stream processing across multiple nodes
//! in a cluster. This module implements consistent hashing for partition routing,
//! distributed window aggregation, and cluster-wide job distribution.
//!
//! ## Components
//!
//! - [`DistributedStreamTopology`]: Coordinates stream processing across nodes
//! - [`ConsistentHashRouter`]: Routes stream partitions to nodes via consistent hashing
//! - [`DistributedWindowAggregator`]: Aggregates windowed results across cluster nodes
//! - [`ClusterStreamCoordinator`]: Manages stream job distribution across the cluster

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, info};

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Errors that can occur in distributed stream processing
#[derive(Error, Debug)]
pub enum DistributedStreamError {
    #[error("Node not found: {node_id}")]
    NodeNotFound { node_id: String },

    #[error("No nodes available in topology")]
    NoNodesAvailable,

    #[error("Partition assignment failed: {reason}")]
    PartitionAssignmentFailed { reason: String },

    #[error("Window aggregation error: {0}")]
    WindowAggregation(String),

    #[error("Job distribution error: {0}")]
    JobDistribution(String),

    #[error("Channel send error: {0}")]
    ChannelSend(String),

    #[error("Topology inconsistency: {0}")]
    TopologyInconsistency(String),
}

/// Result type for distributed stream operations
pub type DistributedResult<T> = Result<T, DistributedStreamError>;

// ─── Node Definitions ────────────────────────────────────────────────────────

/// Represents the health status of a cluster node
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is healthy and processing
    Healthy,
    /// Node is degraded but operational
    Degraded,
    /// Node is unreachable
    Unreachable,
    /// Node is draining (removing from cluster)
    Draining,
}

/// Metadata for a node participating in distributed stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamNode {
    /// Unique identifier for this node
    pub node_id: String,
    /// Network address of the node
    pub address: String,
    /// Current health status
    pub status: NodeStatus,
    /// Number of partitions assigned to this node
    pub assigned_partitions: usize,
    /// Processing capacity (0.0 to 1.0)
    pub capacity: f64,
    /// Timestamp of last heartbeat
    pub last_heartbeat: std::time::SystemTime,
    /// Node weight for load balancing (higher = more load)
    pub weight: u32,
}

impl StreamNode {
    /// Creates a new healthy stream node
    pub fn new(node_id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            address: address.into(),
            status: NodeStatus::Healthy,
            assigned_partitions: 0,
            capacity: 1.0,
            last_heartbeat: std::time::SystemTime::now(),
            weight: 100,
        }
    }

    /// Returns true if this node can accept new partitions
    pub fn is_available(&self) -> bool {
        matches!(self.status, NodeStatus::Healthy | NodeStatus::Degraded) && self.capacity > 0.0
    }
}

// ─── Consistent Hash Router ──────────────────────────────────────────────────

/// Virtual node entry in the consistent hash ring
#[derive(Debug, Clone)]
struct VirtualNode {
    /// Hash position on the ring
    hash: u64,
    /// Owning physical node ID
    node_id: String,
}

/// Routes stream partitions to nodes using consistent hashing.
///
/// Uses a hash ring with configurable virtual nodes per physical node to
/// achieve even load distribution and minimal partition movement on topology
/// changes.
pub struct ConsistentHashRouter {
    /// Sorted ring of virtual nodes
    ring: Vec<VirtualNode>,
    /// Physical nodes indexed by ID
    nodes: HashMap<String, StreamNode>,
    /// Number of virtual nodes per physical node
    virtual_nodes_per_node: usize,
}

impl ConsistentHashRouter {
    /// Creates a new router with the specified number of virtual nodes per physical node.
    pub fn new(virtual_nodes_per_node: usize) -> Self {
        Self {
            ring: Vec::new(),
            nodes: HashMap::new(),
            virtual_nodes_per_node,
        }
    }

    /// Adds a node to the hash ring.
    pub fn add_node(&mut self, node: StreamNode) {
        let node_id = node.node_id.clone();
        for i in 0..self.virtual_nodes_per_node {
            let key = format!("{}#{}", node_id, i);
            let hash = self.fnv1a_hash(key.as_bytes());
            self.ring.push(VirtualNode {
                hash,
                node_id: node_id.clone(),
            });
        }
        self.ring.sort_by_key(|v| v.hash);
        self.nodes.insert(node_id, node);
        debug!(
            "Added node to ring, total virtual nodes: {}",
            self.ring.len()
        );
    }

    /// Removes a node from the hash ring.
    pub fn remove_node(&mut self, node_id: &str) -> DistributedResult<()> {
        if !self.nodes.contains_key(node_id) {
            return Err(DistributedStreamError::NodeNotFound {
                node_id: node_id.to_string(),
            });
        }
        self.ring.retain(|v| v.node_id != node_id);
        self.nodes.remove(node_id);
        info!("Removed node {} from ring", node_id);
        Ok(())
    }

    /// Routes a partition key to the responsible node.
    pub fn route(&self, partition_key: &str) -> DistributedResult<&StreamNode> {
        if self.ring.is_empty() {
            return Err(DistributedStreamError::NoNodesAvailable);
        }
        let hash = self.fnv1a_hash(partition_key.as_bytes());
        let start_idx = self.find_ring_position(hash);
        // Walk the ring to find an available node
        for offset in 0..self.ring.len() {
            let candidate_id = &self.ring[(start_idx + offset) % self.ring.len()].node_id;
            if let Some(node) = self.nodes.get(candidate_id) {
                if node.is_available() {
                    return Ok(node);
                }
            }
        }
        // Fallback: return the initial position even if not available
        let fallback_id = &self.ring[start_idx].node_id;
        self.nodes
            .get(fallback_id)
            .ok_or_else(|| DistributedStreamError::NodeNotFound {
                node_id: fallback_id.clone(),
            })
    }

    /// Returns an iterator over all known physical nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &StreamNode> {
        self.nodes.values()
    }

    /// Returns the number of physical nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// FNV-1a 64-bit hash
    fn fnv1a_hash(&self, data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for byte in data {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Binary search for the first ring entry >= hash; wraps to 0 if past end.
    fn find_ring_position(&self, hash: u64) -> usize {
        match self.ring.binary_search_by_key(&hash, |v| v.hash) {
            Ok(idx) => idx,
            Err(idx) => idx % self.ring.len(),
        }
    }
}

// ─── Distributed Window Aggregator ───────────────────────────────────────────

/// A partial window result from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialWindowResult {
    /// Window identifier (start timestamp ms)
    pub window_id: u64,
    /// Source node that produced this partial result
    pub source_node: String,
    /// Number of events processed in this partial window
    pub event_count: u64,
    /// Partial sum for numeric aggregations
    pub partial_sum: f64,
    /// Partial minimum value
    pub partial_min: f64,
    /// Partial maximum value
    pub partial_max: f64,
    /// Whether this partial result is complete for the node
    pub is_complete: bool,
}

/// Aggregated result across all nodes for a window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedWindowResult {
    /// Window identifier
    pub window_id: u64,
    /// Total events across all nodes
    pub total_events: u64,
    /// Global sum
    pub global_sum: f64,
    /// Global minimum
    pub global_min: f64,
    /// Global maximum
    pub global_max: f64,
    /// Global average (sum / total_events)
    pub global_avg: f64,
    /// Number of nodes that contributed
    pub contributing_nodes: usize,
    /// Whether all expected nodes contributed
    pub is_complete: bool,
}

/// Aggregates windowed results from multiple cluster nodes.
///
/// Collects partial results from each node and merges them into a global
/// aggregate once all expected contributions arrive or a force-flush is triggered.
pub struct DistributedWindowAggregator {
    /// Expected node IDs per window
    expected_nodes: HashSet<String>,
    /// Partial results keyed by (window_id, node_id)
    partial_results: Arc<RwLock<HashMap<(u64, String), PartialWindowResult>>>,
    /// Completed aggregations keyed by window_id
    completed: Arc<RwLock<BTreeMap<u64, AggregatedWindowResult>>>,
    /// Timeout before forcing finalisation of incomplete windows
    timeout: Duration,
}

impl DistributedWindowAggregator {
    /// Creates a new aggregator expecting results from the given nodes.
    pub fn new(expected_nodes: HashSet<String>, timeout: Duration) -> Self {
        Self {
            expected_nodes,
            partial_results: Arc::new(RwLock::new(HashMap::new())),
            completed: Arc::new(RwLock::new(BTreeMap::new())),
            timeout,
        }
    }

    /// Submits a partial result from a node.
    ///
    /// Returns `Some(AggregatedWindowResult)` if all expected nodes have now
    /// contributed for this window; otherwise returns `None`.
    pub fn submit_partial(
        &self,
        partial: PartialWindowResult,
    ) -> DistributedResult<Option<AggregatedWindowResult>> {
        let window_id = partial.window_id;
        {
            let mut results = self.partial_results.write();
            results.insert((window_id, partial.source_node.clone()), partial);
        }
        self.try_aggregate(window_id)
    }

    /// Forces aggregation for a window even if not all nodes have contributed.
    pub fn force_aggregate(&self, window_id: u64) -> DistributedResult<AggregatedWindowResult> {
        self.merge_partials(window_id, false)
    }

    /// Returns a completed aggregate for a specific window, if available.
    pub fn get_completed(&self, window_id: u64) -> Option<AggregatedWindowResult> {
        self.completed.read().get(&window_id).cloned()
    }

    /// Drains and returns all completed window results in ascending window order.
    pub fn drain_completed(&self) -> Vec<AggregatedWindowResult> {
        let mut completed = self.completed.write();
        let keys: Vec<u64> = completed.keys().cloned().collect();
        keys.into_iter()
            .filter_map(|k| completed.remove(&k))
            .collect()
    }

    /// Returns the configured aggregation timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn try_aggregate(&self, window_id: u64) -> DistributedResult<Option<AggregatedWindowResult>> {
        let contributing: HashSet<String> = {
            let results = self.partial_results.read();
            results
                .keys()
                .filter(|(wid, _)| *wid == window_id)
                .map(|(_, nid)| nid.clone())
                .collect()
        };
        if contributing == self.expected_nodes {
            let agg = self.merge_partials(window_id, true)?;
            Ok(Some(agg))
        } else {
            Ok(None)
        }
    }

    fn merge_partials(
        &self,
        window_id: u64,
        all_present: bool,
    ) -> DistributedResult<AggregatedWindowResult> {
        let results = self.partial_results.read();
        let partials: Vec<&PartialWindowResult> = results
            .iter()
            .filter(|((wid, _), _)| *wid == window_id)
            .map(|(_, v)| v)
            .collect();

        if partials.is_empty() {
            return Err(DistributedStreamError::WindowAggregation(format!(
                "No partial results for window {}",
                window_id
            )));
        }

        let total_events = partials.iter().map(|p| p.event_count).sum::<u64>();
        let global_sum = partials.iter().map(|p| p.partial_sum).sum::<f64>();
        let global_min = partials
            .iter()
            .map(|p| p.partial_min)
            .fold(f64::INFINITY, f64::min);
        let global_max = partials
            .iter()
            .map(|p| p.partial_max)
            .fold(f64::NEG_INFINITY, f64::max);
        let global_avg = if total_events > 0 {
            global_sum / total_events as f64
        } else {
            0.0
        };

        let agg = AggregatedWindowResult {
            window_id,
            total_events,
            global_sum,
            global_min,
            global_max,
            global_avg,
            contributing_nodes: partials.len(),
            is_complete: all_present,
        };
        self.completed.write().insert(window_id, agg.clone());
        Ok(agg)
    }
}

// ─── Distributed Stream Topology ─────────────────────────────────────────────

/// Configuration for a distributed stream topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Maximum number of partitions per node
    pub max_partitions_per_node: usize,
    /// Heartbeat interval for node health checks
    pub heartbeat_interval: Duration,
    /// How long before a silent node is considered unreachable
    pub node_timeout: Duration,
    /// Number of virtual nodes per physical node for consistent hashing
    pub virtual_nodes: usize,
    /// Replication factor for stream partitions
    pub replication_factor: usize,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            max_partitions_per_node: 64,
            heartbeat_interval: Duration::from_secs(5),
            node_timeout: Duration::from_secs(30),
            virtual_nodes: 150,
            replication_factor: 2,
        }
    }
}

/// Statistics snapshot for the topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyStats {
    /// Total number of registered nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Total partitions under management
    pub total_partitions: usize,
    /// Average partitions per node
    pub avg_partitions_per_node: f64,
    /// Timestamp of this snapshot
    pub snapshot_time: std::time::SystemTime,
}

/// A topology change event broadcast on topology changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyChange {
    /// A new node joined
    NodeAdded(String),
    /// A node was removed
    NodeRemoved(String),
    /// A partition was reassigned from one node to another
    PartitionReassigned {
        partition: u32,
        from: String,
        to: String,
    },
    /// A node changed health status
    NodeStatusChanged { node_id: String, status: NodeStatus },
}

/// Coordinates stream processing across multiple cluster nodes.
///
/// Maintains cluster membership, assigns partitions to nodes using consistent
/// hashing, and broadcasts topology change events.
pub struct DistributedStreamTopology {
    config: TopologyConfig,
    router: Arc<RwLock<ConsistentHashRouter>>,
    /// Partition assignment: partition_id to node_id
    partition_map: Arc<RwLock<HashMap<u32, String>>>,
    /// Total number of partitions managed by this topology
    total_partitions: u32,
    /// Notification channel for topology changes
    change_tx: mpsc::Sender<TopologyChange>,
    /// Receiver exposed to callers
    change_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<TopologyChange>>>,
}

impl DistributedStreamTopology {
    /// Creates a new topology with the given configuration and partition count.
    pub fn new(config: TopologyConfig, total_partitions: u32) -> Self {
        let (change_tx, change_rx) = mpsc::channel(1024);
        Self {
            config: config.clone(),
            router: Arc::new(RwLock::new(ConsistentHashRouter::new(config.virtual_nodes))),
            partition_map: Arc::new(RwLock::new(HashMap::new())),
            total_partitions,
            change_tx,
            change_rx: Arc::new(tokio::sync::Mutex::new(change_rx)),
        }
    }

    /// Adds a node to the topology and rebalances all partitions.
    pub async fn add_node(&self, node: StreamNode) -> DistributedResult<()> {
        let node_id = node.node_id.clone();
        info!("Adding node {} to topology", node_id);
        self.router.write().add_node(node);
        self.rebalance_partitions()?;
        let _ = self
            .change_tx
            .send(TopologyChange::NodeAdded(node_id))
            .await;
        Ok(())
    }

    /// Removes a node from the topology and reassigns its partitions.
    pub async fn remove_node(&self, node_id: &str) -> DistributedResult<()> {
        info!("Removing node {} from topology", node_id);
        self.router.write().remove_node(node_id)?;
        let reassigned = self.reassign_from_node(node_id)?;
        for (partition, to) in reassigned {
            let _ = self
                .change_tx
                .send(TopologyChange::PartitionReassigned {
                    partition,
                    from: node_id.to_string(),
                    to,
                })
                .await;
        }
        let _ = self
            .change_tx
            .send(TopologyChange::NodeRemoved(node_id.to_string()))
            .await;
        Ok(())
    }

    /// Routes a partition key to its responsible node, returning the node ID.
    pub fn route(&self, partition_key: &str) -> DistributedResult<String> {
        self.router
            .read()
            .route(partition_key)
            .map(|n| n.node_id.clone())
    }

    /// Returns current topology statistics.
    pub fn stats(&self) -> TopologyStats {
        let router = self.router.read();
        let total_nodes = router.node_count();
        let healthy_nodes = router
            .nodes()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        let partition_map = self.partition_map.read();
        let total_partitions = partition_map.len();
        let avg_partitions_per_node = if total_nodes > 0 {
            total_partitions as f64 / total_nodes as f64
        } else {
            0.0
        };
        TopologyStats {
            total_nodes,
            healthy_nodes,
            total_partitions,
            avg_partitions_per_node,
            snapshot_time: std::time::SystemTime::now(),
        }
    }

    /// Returns a handle to the topology change receiver.
    pub fn change_receiver(&self) -> Arc<tokio::sync::Mutex<mpsc::Receiver<TopologyChange>>> {
        Arc::clone(&self.change_rx)
    }

    fn rebalance_partitions(&self) -> DistributedResult<()> {
        let mut partition_map = self.partition_map.write();
        let router = self.router.read();
        for partition in 0..self.total_partitions {
            let key = partition.to_string();
            let node_id = router.route(&key)?.node_id.clone();
            partition_map.insert(partition, node_id);
        }
        debug!("Rebalanced {} partitions", self.total_partitions);
        Ok(())
    }

    fn reassign_from_node(&self, removed_node_id: &str) -> DistributedResult<Vec<(u32, String)>> {
        let mut partition_map = self.partition_map.write();
        let router = self.router.read();
        let mut reassigned = Vec::new();

        for (partition, node_id) in partition_map.iter_mut() {
            if node_id.as_str() == removed_node_id {
                let key = partition.to_string();
                let new_node = router.route(&key)?.node_id.clone();
                reassigned.push((*partition, new_node.clone()));
                *node_id = new_node;
            }
        }
        Ok(reassigned)
    }
}

// ─── Cluster Stream Coordinator ───────────────────────────────────────────────

/// A stream processing job that is distributed across cluster nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamJob {
    /// Unique job identifier
    pub job_id: String,
    /// Human-readable job name
    pub name: String,
    /// Partitions assigned to this job
    pub partitions: Vec<u32>,
    /// Node assignments for each partition
    pub node_assignments: HashMap<u32, String>,
    /// Job creation timestamp
    pub created_at: std::time::SystemTime,
    /// Whether this job is currently active
    pub is_active: bool,
}

impl StreamJob {
    /// Creates a new stream job
    pub fn new(job_id: impl Into<String>, name: impl Into<String>, partitions: Vec<u32>) -> Self {
        Self {
            job_id: job_id.into(),
            name: name.into(),
            partitions,
            node_assignments: HashMap::new(),
            created_at: std::time::SystemTime::now(),
            is_active: true,
        }
    }
}

/// Statistics for the cluster coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorStats {
    /// Total jobs ever submitted
    pub total_jobs: usize,
    /// Currently active jobs
    pub active_jobs: usize,
    /// Total partitions under active management
    pub total_partitions_managed: usize,
    /// Number of nodes in the cluster
    pub cluster_size: usize,
}

/// Manages stream job distribution across a cluster.
///
/// Uses the [`DistributedStreamTopology`] to place job partitions on nodes
/// and tracks running jobs.
pub struct ClusterStreamCoordinator {
    topology: Arc<DistributedStreamTopology>,
    jobs: Arc<RwLock<HashMap<String, StreamJob>>>,
    start_time: Instant,
}

impl ClusterStreamCoordinator {
    /// Creates a new coordinator backed by the given topology.
    pub fn new(topology: Arc<DistributedStreamTopology>) -> Self {
        Self {
            topology,
            jobs: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Submits a new stream job, distributing its partitions across available nodes.
    ///
    /// Returns the job ID on success.
    pub async fn submit_job(&self, mut job: StreamJob) -> DistributedResult<String> {
        let job_id = job.job_id.clone();
        info!(
            "Submitting job {} with {} partitions",
            job_id,
            job.partitions.len()
        );
        for &partition in &job.partitions {
            let key = partition.to_string();
            let node_id = self.topology.route(&key)?;
            job.node_assignments.insert(partition, node_id);
        }
        self.jobs.write().insert(job_id.clone(), job);
        Ok(job_id)
    }

    /// Cancels an active job by ID.
    pub fn cancel_job(&self, job_id: &str) -> DistributedResult<()> {
        let mut jobs = self.jobs.write();
        match jobs.get_mut(job_id) {
            Some(job) => {
                job.is_active = false;
                info!("Cancelled job {}", job_id);
                Ok(())
            }
            None => Err(DistributedStreamError::JobDistribution(format!(
                "Job {} not found",
                job_id
            ))),
        }
    }

    /// Returns a snapshot of a specific job by ID.
    pub fn get_job(&self, job_id: &str) -> Option<StreamJob> {
        self.jobs.read().get(job_id).cloned()
    }

    /// Returns coordinator statistics.
    pub fn stats(&self) -> CoordinatorStats {
        let jobs = self.jobs.read();
        let active_jobs = jobs.values().filter(|j| j.is_active).count();
        let total_partitions_managed = jobs
            .values()
            .filter(|j| j.is_active)
            .map(|j| j.partitions.len())
            .sum();
        let topology_stats = self.topology.stats();
        CoordinatorStats {
            total_jobs: jobs.len(),
            active_jobs,
            total_partitions_managed,
            cluster_size: topology_stats.total_nodes,
        }
    }

    /// Returns the uptime of this coordinator.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Rebalances all active jobs when the topology changes.
    ///
    /// Returns the number of jobs that were rebalanced.
    pub async fn rebalance_all_jobs(&self) -> DistributedResult<usize> {
        let job_ids: Vec<String> = self
            .jobs
            .read()
            .values()
            .filter(|j| j.is_active)
            .map(|j| j.job_id.clone())
            .collect();
        let count = job_ids.len();
        for job_id in job_ids {
            let mut jobs = self.jobs.write();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.node_assignments.clear();
                for &partition in &job.partitions.clone() {
                    let key = partition.to_string();
                    let node_id = self.topology.route(&key)?;
                    job.node_assignments.insert(partition, node_id);
                }
            }
        }
        info!("Rebalanced {} active jobs", count);
        Ok(count)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_router_basic() {
        let mut router = ConsistentHashRouter::new(10);
        router.add_node(StreamNode::new("node-1", "127.0.0.1:8001"));
        router.add_node(StreamNode::new("node-2", "127.0.0.1:8002"));
        router.add_node(StreamNode::new("node-3", "127.0.0.1:8003"));

        let node = router.route("partition-42").expect("route should succeed");
        assert!(!node.node_id.is_empty());
        assert_eq!(router.node_count(), 3);
    }

    #[test]
    fn test_consistent_hash_router_no_nodes() {
        let router = ConsistentHashRouter::new(10);
        let result = router.route("partition-1");
        assert!(matches!(
            result,
            Err(DistributedStreamError::NoNodesAvailable)
        ));
    }

    #[test]
    fn test_consistent_hash_router_remove_node() {
        let mut router = ConsistentHashRouter::new(10);
        router.add_node(StreamNode::new("node-1", "127.0.0.1:8001"));
        router.add_node(StreamNode::new("node-2", "127.0.0.1:8002"));

        router.remove_node("node-1").expect("remove should succeed");
        assert_eq!(router.node_count(), 1);

        let result = router.remove_node("ghost-node");
        assert!(matches!(
            result,
            Err(DistributedStreamError::NodeNotFound { .. })
        ));
    }

    #[test]
    fn test_consistent_hash_distribution() {
        let mut router = ConsistentHashRouter::new(150);
        router.add_node(StreamNode::new("node-a", "10.0.0.1:9000"));
        router.add_node(StreamNode::new("node-b", "10.0.0.2:9000"));
        router.add_node(StreamNode::new("node-c", "10.0.0.3:9000"));

        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..300u32 {
            let key = format!("partition-{}", i);
            let node = router.route(&key).expect("route should succeed");
            *counts.entry(node.node_id.clone()).or_insert(0) += 1;
        }
        // Each node should receive roughly 100 partitions (wide tolerance for small ring)
        for (node_id, count) in &counts {
            assert!(
                *count > 20,
                "distribution too skewed for {}: got {}",
                node_id,
                count
            );
        }
    }

    #[test]
    fn test_distributed_window_aggregator_full() {
        let expected: HashSet<String> = ["node-1", "node-2", "node-3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let aggregator = DistributedWindowAggregator::new(expected, Duration::from_secs(10));

        let p1 = PartialWindowResult {
            window_id: 1000,
            source_node: "node-1".to_string(),
            event_count: 100,
            partial_sum: 50.0,
            partial_min: 1.0,
            partial_max: 10.0,
            is_complete: true,
        };
        let result = aggregator
            .submit_partial(p1)
            .expect("submit should succeed");
        assert!(result.is_none(), "should not complete with only 1/3 nodes");

        let p2 = PartialWindowResult {
            window_id: 1000,
            source_node: "node-2".to_string(),
            event_count: 200,
            partial_sum: 150.0,
            partial_min: 0.5,
            partial_max: 15.0,
            is_complete: true,
        };
        let result = aggregator
            .submit_partial(p2)
            .expect("submit should succeed");
        assert!(result.is_none(), "should not complete with only 2/3 nodes");

        let p3 = PartialWindowResult {
            window_id: 1000,
            source_node: "node-3".to_string(),
            event_count: 300,
            partial_sum: 300.0,
            partial_min: 0.1,
            partial_max: 20.0,
            is_complete: true,
        };
        let result = aggregator
            .submit_partial(p3)
            .expect("submit should succeed");
        assert!(result.is_some(), "should complete with all 3 nodes");

        let agg = result.expect("aggregate must be Some");
        assert_eq!(agg.window_id, 1000);
        assert_eq!(agg.total_events, 600);
        assert!((agg.global_sum - 500.0).abs() < 1e-9);
        assert!((agg.global_min - 0.1).abs() < 1e-9);
        assert!((agg.global_max - 20.0).abs() < 1e-9);
        assert!(agg.is_complete);
    }

    #[test]
    fn test_distributed_window_force_aggregate() {
        let expected: HashSet<String> =
            ["node-1", "node-2"].iter().map(|s| s.to_string()).collect();
        let aggregator = DistributedWindowAggregator::new(expected, Duration::from_secs(10));

        let p = PartialWindowResult {
            window_id: 2000,
            source_node: "node-1".to_string(),
            event_count: 50,
            partial_sum: 25.0,
            partial_min: 2.0,
            partial_max: 8.0,
            is_complete: false,
        };
        aggregator.submit_partial(p).expect("submit should succeed");

        let agg = aggregator
            .force_aggregate(2000)
            .expect("force aggregate should succeed");
        assert_eq!(agg.window_id, 2000);
        assert_eq!(agg.total_events, 50);
        assert!(!agg.is_complete);
    }

    #[test]
    fn test_distributed_window_drain_completed() {
        let expected: HashSet<String> = ["node-x"].iter().map(|s| s.to_string()).collect();
        let aggregator = DistributedWindowAggregator::new(expected, Duration::from_secs(5));

        let p = PartialWindowResult {
            window_id: 3000,
            source_node: "node-x".to_string(),
            event_count: 10,
            partial_sum: 5.0,
            partial_min: 1.0,
            partial_max: 2.0,
            is_complete: true,
        };
        aggregator.submit_partial(p).expect("submit should succeed");

        let drained = aggregator.drain_completed();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].window_id, 3000);

        // Draining again should yield empty
        let empty = aggregator.drain_completed();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn test_distributed_stream_topology_add_remove() {
        let config = TopologyConfig::default();
        let topology = DistributedStreamTopology::new(config, 16);

        topology
            .add_node(StreamNode::new("node-a", "10.0.0.1:9000"))
            .await
            .expect("add node should succeed");
        topology
            .add_node(StreamNode::new("node-b", "10.0.0.2:9000"))
            .await
            .expect("add node should succeed");

        let stats = topology.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_partitions, 16);

        let routed = topology.route("test-key").expect("route should succeed");
        assert!(!routed.is_empty());

        topology
            .remove_node("node-a")
            .await
            .expect("remove should succeed");
        let stats = topology.stats();
        assert_eq!(stats.total_nodes, 1);
    }

    #[tokio::test]
    async fn test_cluster_stream_coordinator_submit_and_stats() {
        let config = TopologyConfig::default();
        let topology = Arc::new(DistributedStreamTopology::new(config, 32));
        topology
            .add_node(StreamNode::new("worker-1", "10.0.0.1:9000"))
            .await
            .expect("add node should succeed");
        topology
            .add_node(StreamNode::new("worker-2", "10.0.0.2:9000"))
            .await
            .expect("add node should succeed");

        let coordinator = ClusterStreamCoordinator::new(Arc::clone(&topology));
        let job = StreamJob::new("job-001", "sensor-stream", vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let job_id = coordinator
            .submit_job(job)
            .await
            .expect("submit should succeed");
        assert_eq!(job_id, "job-001");

        let retrieved = coordinator.get_job("job-001").expect("job should exist");
        assert_eq!(retrieved.node_assignments.len(), 8);

        let stats = coordinator.stats();
        assert_eq!(stats.active_jobs, 1);
        assert_eq!(stats.total_partitions_managed, 8);
        assert_eq!(stats.cluster_size, 2);
    }

    #[tokio::test]
    async fn test_cluster_stream_coordinator_cancel() {
        let config = TopologyConfig::default();
        let topology = Arc::new(DistributedStreamTopology::new(config, 8));
        topology
            .add_node(StreamNode::new("n1", "localhost:9001"))
            .await
            .expect("add node");

        let coordinator = ClusterStreamCoordinator::new(Arc::clone(&topology));
        let job = StreamJob::new("job-cancel", "cancel-test", vec![0, 1]);
        coordinator
            .submit_job(job)
            .await
            .expect("submit should succeed");

        coordinator
            .cancel_job("job-cancel")
            .expect("cancel should succeed");
        let job = coordinator
            .get_job("job-cancel")
            .expect("job should still exist after cancel");
        assert!(!job.is_active);

        let result = coordinator.cancel_job("nonexistent");
        assert!(matches!(
            result,
            Err(DistributedStreamError::JobDistribution(_))
        ));
    }

    #[tokio::test]
    async fn test_rebalance_jobs_after_node_change() {
        let config = TopologyConfig::default();
        let topology = Arc::new(DistributedStreamTopology::new(config, 16));
        topology
            .add_node(StreamNode::new("n1", "localhost:9001"))
            .await
            .expect("add node");
        topology
            .add_node(StreamNode::new("n2", "localhost:9002"))
            .await
            .expect("add node");

        let coordinator = ClusterStreamCoordinator::new(Arc::clone(&topology));
        let job = StreamJob::new("job-rebalance", "rebalance-test", (0u32..8).collect());
        coordinator
            .submit_job(job)
            .await
            .expect("submit should succeed");

        topology
            .add_node(StreamNode::new("n3", "localhost:9003"))
            .await
            .expect("add node");
        let rebalanced = coordinator
            .rebalance_all_jobs()
            .await
            .expect("rebalance should succeed");
        assert_eq!(rebalanced, 1);
    }
}
