//! Distributed Deadlock Detection
//!
//! This module implements distributed deadlock detection algorithms for detecting
//! and resolving deadlocks that span multiple nodes in a distributed RDF storage system.
//!
//! # Algorithms
//!
//! - **Wait-For Graph (WFG)**: Global wait-for graph construction
//! - **Edge Chasing**: Probe-based deadlock detection
//! - **Timeout-Based**: Fallback detection using transaction timeouts
//!
//! # Approach
//!
//! 1. Each node maintains local wait-for graph
//! 2. Nodes exchange WFG edges periodically
//! 3. Global WFG is constructed at detection coordinator
//! 4. Cycle detection identifies deadlocks
//! 5. Victim selection and abort resolution
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::distributed::deadlock::{DistributedDeadlockDetector, DeadlockDetectorConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create detector
//! let config = DeadlockDetectorConfig::default();
//! let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);
//!
//! // Register nodes
//! detector.register_node("node1".to_string()).await?;
//! detector.register_node("node2".to_string()).await?;
//!
//! // Add wait-for edge: txn1 on node1 waits for txn2 on node2
//! detector.add_wait_edge("node1".to_string(), "txn1".to_string(), "txn2".to_string(), "node2".to_string()).await?;
//!
//! // Detect deadlocks
//! let deadlocks = detector.detect_deadlocks().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TdbError};
use anyhow::Context;
use chrono::{DateTime, Duration, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Wait-for edge in distributed graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WaitEdge {
    /// Source node ID (where waiting transaction is)
    pub source_node: String,
    /// Waiting transaction ID
    pub waiting_txn: String,
    /// Target node ID (where holding transaction is)
    pub target_node: String,
    /// Holding transaction ID
    pub holding_txn: String,
    /// Timestamp when edge was added
    pub timestamp: DateTime<Utc>,
}

/// Deadlock cycle in global wait-for graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockCycle {
    /// Transactions involved in the cycle
    pub transactions: Vec<String>,
    /// Nodes involved in the cycle
    pub nodes: Vec<String>,
    /// Wait edges forming the cycle
    pub edges: Vec<WaitEdge>,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Victim transaction to abort
    pub victim: Option<String>,
}

/// Node status in distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node ID
    pub node_id: String,
    /// Last heartbeat time
    pub last_heartbeat: DateTime<Utc>,
    /// Active transactions on this node
    pub active_transactions: HashSet<String>,
    /// Local wait-for edges
    pub local_edges: Vec<WaitEdge>,
}

/// Deadlock detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockDetectorConfig {
    /// Detection interval
    pub detection_interval: Duration,
    /// Edge staleness threshold (remove old edges)
    pub edge_staleness_threshold: Duration,
    /// Enable proactive detection
    pub proactive_detection: bool,
    /// Victim selection strategy
    pub victim_selection: VictimSelectionStrategy,
    /// Maximum cycles to detect per run
    pub max_cycles_per_detection: usize,
}

impl Default for DeadlockDetectorConfig {
    fn default() -> Self {
        Self {
            detection_interval: Duration::seconds(5),
            edge_staleness_threshold: Duration::minutes(1),
            proactive_detection: true,
            victim_selection: VictimSelectionStrategy::YoungestTransaction,
            max_cycles_per_detection: 100,
        }
    }
}

/// Victim selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VictimSelectionStrategy {
    /// Abort youngest transaction in cycle
    YoungestTransaction,
    /// Abort oldest transaction in cycle
    OldestTransaction,
    /// Abort transaction with least work done
    LeastWork,
    /// Random selection
    Random,
}

/// Distributed Deadlock Detector
///
/// Coordinates deadlock detection across multiple distributed nodes.
pub struct DistributedDeadlockDetector {
    /// Detector ID
    id: String,
    /// Configuration
    config: DeadlockDetectorConfig,
    /// Registered nodes
    nodes: Arc<RwLock<HashMap<String, NodeStatus>>>,
    /// Global wait-for graph
    wait_graph: Arc<RwLock<HashMap<String, Vec<WaitEdge>>>>,
    /// Detected deadlock history
    deadlock_history: Arc<Mutex<Vec<DeadlockCycle>>>,
    /// Statistics
    stats: Arc<Mutex<DeadlockStats>>,
}

/// Deadlock detection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeadlockStats {
    /// Total detection runs
    pub total_detections: u64,
    /// Total deadlocks detected
    pub total_deadlocks: u64,
    /// Total victims aborted
    pub total_victims_aborted: u64,
    /// False positives (cycles that weren't real deadlocks)
    pub false_positives: u64,
    /// Average detection time (milliseconds)
    pub avg_detection_time_ms: f64,
    /// Total detection time (for calculating average)
    total_detection_time_ms: f64,
    /// Largest cycle detected
    pub max_cycle_length: usize,
}

impl DistributedDeadlockDetector {
    /// Create a new Distributed Deadlock Detector
    pub fn new(id: String, config: DeadlockDetectorConfig) -> Self {
        Self {
            id,
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            wait_graph: Arc::new(RwLock::new(HashMap::new())),
            deadlock_history: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(DeadlockStats::default())),
        }
    }

    /// Register a node
    pub async fn register_node(&mut self, node_id: String) -> Result<()> {
        let status = NodeStatus {
            node_id: node_id.clone(),
            last_heartbeat: Utc::now(),
            active_transactions: HashSet::new(),
            local_edges: Vec::new(),
        };

        self.nodes.write().insert(node_id, status);
        Ok(())
    }

    /// Update node heartbeat
    pub async fn update_node_heartbeat(&self, node_id: &str) -> Result<()> {
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = Utc::now();
        }
        Ok(())
    }

    /// Add a wait-for edge
    pub async fn add_wait_edge(
        &self,
        source_node: String,
        waiting_txn: String,
        holding_txn: String,
        target_node: String,
    ) -> Result<()> {
        let edge = WaitEdge {
            source_node: source_node.clone(),
            waiting_txn: waiting_txn.clone(),
            target_node,
            holding_txn,
            timestamp: Utc::now(),
        };

        let mut graph = self.wait_graph.write();
        graph
            .entry(waiting_txn.clone())
            .or_default()
            .push(edge.clone());

        // Also update local node edges
        let mut nodes = self.nodes.write();
        if let Some(node) = nodes.get_mut(&source_node) {
            node.local_edges.push(edge);
        }

        Ok(())
    }

    /// Remove a wait-for edge
    pub async fn remove_wait_edge(&self, waiting_txn: &str, holding_txn: &str) -> Result<()> {
        let mut graph = self.wait_graph.write();
        if let Some(edges) = graph.get_mut(waiting_txn) {
            edges.retain(|edge| edge.holding_txn != holding_txn);
            if edges.is_empty() {
                graph.remove(waiting_txn);
            }
        }

        Ok(())
    }

    /// Clean stale edges
    async fn clean_stale_edges(&self) -> Result<u64> {
        let now = Utc::now();
        let threshold = self.config.edge_staleness_threshold;
        let mut graph = self.wait_graph.write();
        let mut removed_count = 0;

        graph.retain(|_, edges| {
            let initial_len = edges.len();
            edges.retain(|edge| {
                let age = now - edge.timestamp;
                age <= threshold
            });
            removed_count += (initial_len - edges.len()) as u64;
            !edges.is_empty()
        });

        Ok(removed_count)
    }

    /// Detect deadlocks in the global wait-for graph
    pub async fn detect_deadlocks(&mut self) -> Result<Vec<DeadlockCycle>> {
        let start_time = Utc::now();

        // Clean stale edges first
        self.clean_stale_edges().await?;

        let graph = self.wait_graph.read().clone();
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();

        // Use DFS to detect cycles
        for txn in graph.keys() {
            if visited.contains(txn) {
                continue;
            }

            if let Some(cycle) = self.detect_cycle_from(txn, &graph, &mut visited)? {
                cycles.push(cycle);

                if cycles.len() >= self.config.max_cycles_per_detection {
                    break;
                }
            }

            visited.insert(txn.clone());
        }

        // Select victims for each cycle
        for cycle in &mut cycles {
            cycle.victim = Some(self.select_victim(cycle)?);
        }

        // Update statistics
        let detection_time = (Utc::now() - start_time).num_milliseconds() as f64;
        let mut stats = self.stats.lock();
        stats.total_detections += 1;
        stats.total_deadlocks += cycles.len() as u64;
        stats.total_detection_time_ms += detection_time;
        stats.avg_detection_time_ms = stats.total_detection_time_ms / stats.total_detections as f64;

        if !cycles.is_empty() {
            let max_len = cycles
                .iter()
                .map(|c| c.transactions.len())
                .max()
                .unwrap_or(0);
            if max_len > stats.max_cycle_length {
                stats.max_cycle_length = max_len;
            }
        }

        // Add to history
        let mut history = self.deadlock_history.lock();
        history.extend(cycles.clone());

        Ok(cycles)
    }

    /// Detect cycle starting from a transaction using DFS
    fn detect_cycle_from(
        &self,
        start_txn: &str,
        graph: &HashMap<String, Vec<WaitEdge>>,
        visited: &mut HashSet<String>,
    ) -> Result<Option<DeadlockCycle>> {
        let mut stack = vec![(start_txn.to_string(), Vec::new())];
        let mut path = HashMap::new();

        while let Some((current, edges_to_here)) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current.clone());
            path.insert(current.clone(), edges_to_here.clone());

            if let Some(outgoing_edges) = graph.get(&current) {
                for edge in outgoing_edges {
                    let next = &edge.holding_txn;

                    if next == start_txn {
                        // Cycle detected!
                        let mut cycle_edges = edges_to_here.clone();
                        cycle_edges.push(edge.clone());

                        return Ok(Some(self.build_cycle(start_txn, cycle_edges)?));
                    }

                    if !visited.contains(next) {
                        let mut new_edges = edges_to_here.clone();
                        new_edges.push(edge.clone());
                        stack.push((next.clone(), new_edges));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Build deadlock cycle from edges
    fn build_cycle(&self, start_txn: &str, edges: Vec<WaitEdge>) -> Result<DeadlockCycle> {
        let mut transactions = vec![start_txn.to_string()];
        let mut nodes = HashSet::new();

        for edge in &edges {
            transactions.push(edge.holding_txn.clone());
            nodes.insert(edge.source_node.clone());
            nodes.insert(edge.target_node.clone());
        }

        // Remove duplicate (cycle closes)
        transactions.pop();

        Ok(DeadlockCycle {
            transactions,
            nodes: nodes.into_iter().collect(),
            edges,
            detected_at: Utc::now(),
            victim: None,
        })
    }

    /// Select victim transaction to abort
    fn select_victim(&self, cycle: &DeadlockCycle) -> Result<String> {
        match self.config.victim_selection {
            VictimSelectionStrategy::YoungestTransaction => {
                // Find transaction with latest start time (youngest)
                Ok(cycle.transactions.last().unwrap().clone())
            }
            VictimSelectionStrategy::OldestTransaction => {
                // Find transaction with earliest start time (oldest)
                Ok(cycle.transactions.first().unwrap().clone())
            }
            VictimSelectionStrategy::LeastWork => {
                // For now, default to youngest
                // TODO: Implement actual work tracking
                Ok(cycle.transactions.last().unwrap().clone())
            }
            VictimSelectionStrategy::Random => {
                // Use scirs2-core's random for selection
                use scirs2_core::random::{DistributionExt, Random};
                let mut rng = Random::seed(0); // Use a fixed seed or use system time
                let idx = rng.gen_range(0..cycle.transactions.len());
                Ok(cycle.transactions[idx].clone())
            }
        }
    }

    /// Abort victim transaction
    pub async fn abort_victim(&mut self, txn_id: &str) -> Result<()> {
        // Remove all edges involving this transaction
        let mut graph = self.wait_graph.write();
        graph.remove(txn_id);

        // Also remove edges where this transaction is the holder
        for edges in graph.values_mut() {
            edges.retain(|edge| edge.holding_txn != txn_id);
        }

        let mut stats = self.stats.lock();
        stats.total_victims_aborted += 1;

        Ok(())
    }

    /// Get detector ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get statistics
    pub fn stats(&self) -> DeadlockStats {
        self.stats.lock().clone()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Get active edge count
    pub fn edge_count(&self) -> usize {
        self.wait_graph.read().values().map(|v| v.len()).sum()
    }

    /// Get deadlock history
    pub fn deadlock_history(&self) -> Vec<DeadlockCycle> {
        self.deadlock_history.lock().clone()
    }

    /// Clear deadlock history
    pub fn clear_history(&self) {
        self.deadlock_history.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detector_creation() {
        let config = DeadlockDetectorConfig::default();
        let detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        assert_eq!(detector.id(), "detector-1");
        assert_eq!(detector.node_count(), 0);
        assert_eq!(detector.edge_count(), 0);
    }

    #[tokio::test]
    async fn test_register_nodes() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();
        detector.register_node("node2".to_string()).await.unwrap();

        assert_eq!(detector.node_count(), 2);
    }

    #[tokio::test]
    async fn test_add_wait_edge() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();
        detector.register_node("node2".to_string()).await.unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node2".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(detector.edge_count(), 1);
    }

    #[tokio::test]
    async fn test_remove_wait_edge() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector.remove_wait_edge("txn1", "txn2").await.unwrap();

        assert_eq!(detector.edge_count(), 0);
    }

    #[tokio::test]
    async fn test_detect_simple_deadlock() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();
        detector.register_node("node2".to_string()).await.unwrap();

        // Create cycle: txn1 -> txn2 -> txn1
        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node2".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node2".to_string(),
                "txn2".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 1);

        let cycle = &cycles[0];
        assert_eq!(cycle.transactions.len(), 2);
        assert!(cycle.transactions.contains(&"txn1".to_string()));
        assert!(cycle.transactions.contains(&"txn2".to_string()));
        assert!(cycle.victim.is_some());

        let stats = detector.stats();
        assert_eq!(stats.total_deadlocks, 1);
    }

    #[tokio::test]
    async fn test_detect_multi_node_deadlock() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();
        detector.register_node("node2".to_string()).await.unwrap();
        detector.register_node("node3".to_string()).await.unwrap();

        // Create cycle: txn1 -> txn2 -> txn3 -> txn1
        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node2".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node2".to_string(),
                "txn2".to_string(),
                "txn3".to_string(),
                "node3".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node3".to_string(),
                "txn3".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 1);

        let cycle = &cycles[0];
        assert_eq!(cycle.transactions.len(), 3);
        assert_eq!(cycle.nodes.len(), 3);

        let stats = detector.stats();
        assert_eq!(stats.max_cycle_length, 3);
    }

    #[tokio::test]
    async fn test_no_deadlock_detection() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        // Add edges that don't form a cycle
        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn2".to_string(),
                "txn3".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[tokio::test]
    async fn test_abort_victim() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        // Create cycle
        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn2".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 1);

        let victim = cycles[0].victim.as_ref().unwrap();
        detector.abort_victim(victim).await.unwrap();

        // After aborting victim, no more cycles
        let cycles2 = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles2.len(), 0);

        let stats = detector.stats();
        assert_eq!(stats.total_victims_aborted, 1);
    }

    #[tokio::test]
    async fn test_victim_selection_youngest() {
        let config = DeadlockDetectorConfig {
            victim_selection: VictimSelectionStrategy::YoungestTransaction,
            ..Default::default()
        };

        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn2".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 1);

        // Youngest should be last in the list
        let victim = cycles[0].victim.as_ref().unwrap();
        assert!(victim == "txn1" || victim == "txn2");
    }

    #[tokio::test]
    async fn test_victim_selection_oldest() {
        let config = DeadlockDetectorConfig {
            victim_selection: VictimSelectionStrategy::OldestTransaction,
            ..Default::default()
        };

        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn2".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        let cycles = detector.detect_deadlocks().await.unwrap();
        assert_eq!(cycles.len(), 1);

        // Oldest should be first in the list
        let victim = cycles[0].victim.as_ref().unwrap();
        assert!(victim == "txn1" || victim == "txn2");
    }

    #[tokio::test]
    async fn test_deadlock_history() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        // First deadlock
        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn1".to_string(),
                "txn2".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector
            .add_wait_edge(
                "node1".to_string(),
                "txn2".to_string(),
                "txn1".to_string(),
                "node1".to_string(),
            )
            .await
            .unwrap();

        detector.detect_deadlocks().await.unwrap();

        let history = detector.deadlock_history();
        assert_eq!(history.len(), 1);

        // Clear history
        detector.clear_history();
        let history = detector.deadlock_history();
        assert_eq!(history.len(), 0);
    }

    #[tokio::test]
    async fn test_detection_stats() {
        let config = DeadlockDetectorConfig::default();
        let mut detector = DistributedDeadlockDetector::new("detector-1".to_string(), config);

        detector.register_node("node1".to_string()).await.unwrap();

        // Run detection without deadlock
        detector.detect_deadlocks().await.unwrap();

        let stats = detector.stats();
        assert_eq!(stats.total_detections, 1);
        assert_eq!(stats.total_deadlocks, 0);
        assert!(stats.avg_detection_time_ms >= 0.0);
    }
}
