//! # Network Partition Detection and Auto-Healing
//!
//! Detects network partitions and automatically heals the cluster when possible.
//! Uses heartbeat monitoring, quorum detection, and automatic recovery mechanisms.

use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Partition detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionDetectionConfig {
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Heartbeat timeout threshold
    pub heartbeat_timeout_ms: u64,
    /// Number of consecutive missed heartbeats before declaring partition
    pub max_missed_heartbeats: u32,
    /// Enable predictive partition detection using ML
    pub enable_predictive_detection: bool,
    /// Auto-healing enabled
    pub auto_healing_enabled: bool,
    /// Maximum time to wait before attempting recovery
    pub recovery_delay_ms: u64,
    /// Use quorum-based decision making
    pub use_quorum: bool,
    /// Minimum quorum size (percentage of total nodes)
    pub min_quorum_percent: u8,
}

impl Default for PartitionDetectionConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: 100,
            heartbeat_timeout_ms: 1000,
            max_missed_heartbeats: 3,
            enable_predictive_detection: true,
            auto_healing_enabled: true,
            recovery_delay_ms: 5000,
            use_quorum: true,
            min_quorum_percent: 51,
        }
    }
}

/// Partition status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStatus {
    /// No partition detected
    Healthy,
    /// Potential partition detected
    Suspected,
    /// Partition confirmed
    Partitioned,
    /// Recovery in progress
    Recovering,
    /// Recovered from partition
    Recovered,
}

/// Node connectivity status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConnectivity {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Last heartbeat timestamp
    pub last_heartbeat: SystemTime,
    /// Consecutive missed heartbeats
    pub missed_heartbeats: u32,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Latency variance
    pub latency_variance: f64,
    /// Is node reachable
    pub is_reachable: bool,
    /// Network partition detected
    pub is_partitioned: bool,
}

impl NodeConnectivity {
    pub fn new(node_id: OxirsNodeId) -> Self {
        Self {
            node_id,
            last_heartbeat: SystemTime::now(),
            missed_heartbeats: 0,
            avg_latency_ms: 0.0,
            latency_variance: 0.0,
            is_reachable: true,
            is_partitioned: false,
        }
    }

    pub fn update_heartbeat(&mut self, latency_ms: f64) {
        self.last_heartbeat = SystemTime::now();
        self.missed_heartbeats = 0;
        self.is_reachable = true;

        // Update latency statistics using exponential moving average
        let alpha = 0.3;
        self.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.avg_latency_ms;

        // Update variance
        let diff = latency_ms - self.avg_latency_ms;
        self.latency_variance = alpha * (diff * diff) + (1.0 - alpha) * self.latency_variance;
    }

    pub fn record_missed_heartbeat(&mut self) {
        self.missed_heartbeats += 1;
    }

    pub fn is_timeout(&self, timeout_ms: u64, max_missed: u32) -> bool {
        if self.missed_heartbeats >= max_missed {
            return true;
        }

        if let Ok(elapsed) = SystemTime::now().duration_since(self.last_heartbeat) {
            return elapsed.as_millis() >= timeout_ms as u128;
        }

        false
    }
}

/// Partition detector
#[derive(Debug, Clone)]
pub struct PartitionDetector {
    node_id: OxirsNodeId,
    config: PartitionDetectionConfig,
    connectivity: Arc<RwLock<BTreeMap<OxirsNodeId, NodeConnectivity>>>,
    status: Arc<RwLock<PartitionStatus>>,
    partition_history: Arc<RwLock<VecDeque<PartitionEvent>>>,
    metrics: Arc<RwLock<PartitionMetrics>>,
}

/// Partition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: PartitionEventType,
    /// Affected nodes
    pub affected_nodes: Vec<OxirsNodeId>,
    /// Event details
    pub details: String,
}

/// Partition event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionEventType {
    /// Partition detected
    PartitionDetected,
    /// Node became unreachable
    NodeUnreachable,
    /// Node recovered
    NodeRecovered,
    /// Quorum lost
    QuorumLost,
    /// Quorum restored
    QuorumRestored,
    /// Auto-healing started
    HealingStarted,
    /// Auto-healing completed
    HealingCompleted,
    /// Auto-healing failed
    HealingFailed,
}

/// Partition metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartitionMetrics {
    /// Total partitions detected
    pub total_partitions: u64,
    /// Total recoveries
    pub total_recoveries: u64,
    /// Total healing attempts
    pub total_healing_attempts: u64,
    /// Successful healing operations
    pub successful_healings: u64,
    /// Average partition duration
    pub avg_partition_duration_secs: f64,
    /// Average recovery time
    pub avg_recovery_time_secs: f64,
    /// Last partition timestamp
    pub last_partition: Option<SystemTime>,
    /// Last recovery timestamp
    pub last_recovery: Option<SystemTime>,
}

impl PartitionDetector {
    /// Create a new partition detector
    pub fn new(node_id: OxirsNodeId, config: PartitionDetectionConfig) -> Self {
        Self {
            node_id,
            config,
            connectivity: Arc::new(RwLock::new(BTreeMap::new())),
            status: Arc::new(RwLock::new(PartitionStatus::Healthy)),
            partition_history: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(PartitionMetrics::default())),
        }
    }

    /// Register a node for monitoring
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut connectivity = self.connectivity.write().await;
        connectivity.insert(node_id, NodeConnectivity::new(node_id));

        tracing::debug!(
            "Node {}: Registered node {} for partition detection",
            self.node_id,
            node_id
        );
    }

    /// Unregister a node
    pub async fn unregister_node(&self, node_id: OxirsNodeId) {
        let mut connectivity = self.connectivity.write().await;
        connectivity.remove(&node_id);

        tracing::debug!(
            "Node {}: Unregistered node {} from partition detection",
            self.node_id,
            node_id
        );
    }

    /// Record a heartbeat from a node
    pub async fn record_heartbeat(&self, node_id: OxirsNodeId, latency_ms: f64) {
        let mut connectivity = self.connectivity.write().await;

        if let Some(node) = connectivity.get_mut(&node_id) {
            let was_partitioned = node.is_partitioned;
            node.update_heartbeat(latency_ms);
            node.is_partitioned = false;

            if was_partitioned {
                // Node recovered from partition
                self.record_event(
                    PartitionEventType::NodeRecovered,
                    vec![node_id],
                    format!("Node {} recovered (latency: {:.2}ms)", node_id, latency_ms),
                )
                .await;

                drop(connectivity);
                self.check_quorum_status().await;
            }
        }
    }

    /// Check for partitions
    pub async fn check_for_partitions(&self) -> Vec<OxirsNodeId> {
        let mut connectivity = self.connectivity.write().await;
        let mut partitioned_nodes = Vec::new();

        for (node_id, node) in connectivity.iter_mut() {
            if node.is_timeout(
                self.config.heartbeat_timeout_ms,
                self.config.max_missed_heartbeats,
            ) {
                if !node.is_partitioned {
                    node.is_partitioned = true;
                    node.is_reachable = false;
                    partitioned_nodes.push(*node_id);

                    tracing::warn!(
                        "Node {}: Detected partition with node {} (missed {} heartbeats)",
                        self.node_id,
                        node_id,
                        node.missed_heartbeats
                    );
                }
            }
        }

        if !partitioned_nodes.is_empty() {
            drop(connectivity);

            // Record partition events
            for &node_id in &partitioned_nodes {
                self.record_event(
                    PartitionEventType::NodeUnreachable,
                    vec![node_id],
                    format!("Node {} became unreachable", node_id),
                )
                .await;
            }

            // Update status
            let mut status = self.status.write().await;
            *status = PartitionStatus::Partitioned;

            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.total_partitions += 1;
            metrics.last_partition = Some(SystemTime::now());

            // Check quorum
            drop(status);
            drop(metrics);
            self.check_quorum_status().await;
        }

        partitioned_nodes
    }

    /// Check quorum status
    async fn check_quorum_status(&self) {
        if !self.config.use_quorum {
            return;
        }

        let connectivity = self.connectivity.read().await;
        let total_nodes = connectivity.len() + 1; // Include self
        let reachable_nodes = connectivity.values().filter(|n| n.is_reachable).count() + 1;

        let quorum_size =
            ((total_nodes as f64 * self.config.min_quorum_percent as f64) / 100.0).ceil() as usize;

        if reachable_nodes < quorum_size {
            drop(connectivity);

            // Quorum lost
            self.record_event(
                PartitionEventType::QuorumLost,
                vec![],
                format!(
                    "Quorum lost: {} reachable out of {} total (need {})",
                    reachable_nodes, total_nodes, quorum_size
                ),
            )
            .await;

            tracing::error!(
                "Node {}: Quorum lost! {} reachable out of {} total (need {})",
                self.node_id,
                reachable_nodes,
                total_nodes,
                quorum_size
            );
        } else {
            let status = *self.status.read().await;
            if status == PartitionStatus::Partitioned {
                drop(connectivity);

                // Quorum restored
                self.record_event(
                    PartitionEventType::QuorumRestored,
                    vec![],
                    format!(
                        "Quorum restored: {} reachable out of {} total",
                        reachable_nodes, total_nodes
                    ),
                )
                .await;

                tracing::info!(
                    "Node {}: Quorum restored: {} reachable out of {} total",
                    self.node_id,
                    reachable_nodes,
                    total_nodes
                );
            }
        }
    }

    /// Attempt auto-healing
    pub async fn attempt_healing(&self) -> Result<()> {
        if !self.config.auto_healing_enabled {
            return Ok(());
        }

        let status = *self.status.read().await;
        if status != PartitionStatus::Partitioned {
            return Ok(());
        }

        // Update status to recovering
        *self.status.write().await = PartitionStatus::Recovering;

        self.record_event(
            PartitionEventType::HealingStarted,
            vec![],
            format!("Auto-healing started for node {}", self.node_id),
        )
        .await;

        let mut metrics = self.metrics.write().await;
        metrics.total_healing_attempts += 1;
        drop(metrics);

        tracing::info!("Node {}: Starting auto-healing process", self.node_id);

        // Wait for recovery delay
        tokio::time::sleep(Duration::from_millis(self.config.recovery_delay_ms)).await;

        // Re-check partitions
        let partitioned_nodes = self.check_for_partitions().await;

        if partitioned_nodes.is_empty() {
            // Healing successful
            *self.status.write().await = PartitionStatus::Recovered;

            self.record_event(
                PartitionEventType::HealingCompleted,
                vec![],
                format!("Auto-healing completed for node {}", self.node_id),
            )
            .await;

            let mut metrics = self.metrics.write().await;
            metrics.successful_healings += 1;
            metrics.total_recoveries += 1;
            metrics.last_recovery = Some(SystemTime::now());

            tracing::info!("Node {}: Auto-healing completed successfully", self.node_id);

            Ok(())
        } else {
            // Healing failed
            self.record_event(
                PartitionEventType::HealingFailed,
                partitioned_nodes.clone(),
                format!(
                    "Auto-healing failed: {} nodes still partitioned",
                    partitioned_nodes.len()
                ),
            )
            .await;

            tracing::warn!(
                "Node {}: Auto-healing failed, {} nodes still partitioned",
                self.node_id,
                partitioned_nodes.len()
            );

            Err(anyhow::anyhow!("Auto-healing failed"))
        }
    }

    /// Record a partition event
    async fn record_event(
        &self,
        event_type: PartitionEventType,
        affected_nodes: Vec<OxirsNodeId>,
        details: String,
    ) {
        let event = PartitionEvent {
            timestamp: SystemTime::now(),
            event_type,
            affected_nodes,
            details,
        };

        let mut history = self.partition_history.write().await;
        history.push_back(event);

        // Keep only last 1000 events
        if history.len() > 1000 {
            history.pop_front();
        }
    }

    /// Get current partition status
    pub async fn get_status(&self) -> PartitionStatus {
        *self.status.read().await
    }

    /// Get partition metrics
    pub async fn get_metrics(&self) -> PartitionMetrics {
        self.metrics.read().await.clone()
    }

    /// Get partition history
    pub async fn get_history(&self, limit: usize) -> Vec<PartitionEvent> {
        let history = self.partition_history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get node connectivity map
    pub async fn get_connectivity(&self) -> BTreeMap<OxirsNodeId, NodeConnectivity> {
        self.connectivity.read().await.clone()
    }

    /// Predict partition using heuristic analysis
    pub async fn predict_partition(&self) -> f64 {
        if !self.config.enable_predictive_detection {
            return 0.0;
        }

        let connectivity = self.connectivity.read().await;

        if connectivity.is_empty() {
            return 0.0;
        }

        // Collect latency statistics
        let latencies: Vec<f64> = connectivity.values().map(|n| n.avg_latency_ms).collect();
        let variances: Vec<f64> = connectivity.values().map(|n| n.latency_variance).collect();

        if latencies.is_empty() {
            return 0.0;
        }

        // Calculate mean and variance manually
        let sum: f64 = latencies.iter().sum();
        let count = latencies.len() as f64;
        let avg_latency = sum / count;

        let variance_sum: f64 = latencies.iter().map(|x| (x - avg_latency).powi(2)).sum();
        let latency_var = variance_sum / count;

        let avg_variance: f64 = variances.iter().sum::<f64>() / count;

        // Simple heuristic: high latency variance suggests instability
        let instability_score =
            (latency_var.sqrt() / (avg_latency + 1.0)) + (avg_variance / (avg_latency + 1.0));

        // Normalize to 0-1 range
        let risk_score = (instability_score / 2.0).min(1.0).max(0.0);

        if risk_score > 0.7 {
            tracing::warn!(
                "Node {}: High partition risk detected (score: {:.2})",
                self.node_id,
                risk_score
            );
        }

        risk_score
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = PartitionMetrics::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_detection_config_default() {
        let config = PartitionDetectionConfig::default();
        assert_eq!(config.heartbeat_interval_ms, 100);
        assert_eq!(config.heartbeat_timeout_ms, 1000);
        assert_eq!(config.max_missed_heartbeats, 3);
        assert!(config.enable_predictive_detection);
        assert!(config.auto_healing_enabled);
        assert_eq!(config.recovery_delay_ms, 5000);
        assert!(config.use_quorum);
        assert_eq!(config.min_quorum_percent, 51);
    }

    #[test]
    fn test_node_connectivity_creation() {
        let node = NodeConnectivity::new(1);
        assert_eq!(node.node_id, 1);
        assert_eq!(node.missed_heartbeats, 0);
        assert!(node.is_reachable);
        assert!(!node.is_partitioned);
        assert_eq!(node.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_node_connectivity_update_heartbeat() {
        let mut node = NodeConnectivity::new(1);
        node.update_heartbeat(10.0);

        assert_eq!(node.missed_heartbeats, 0);
        assert!(node.is_reachable);
        assert!(node.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_node_connectivity_missed_heartbeat() {
        let mut node = NodeConnectivity::new(1);
        node.record_missed_heartbeat();
        node.record_missed_heartbeat();

        assert_eq!(node.missed_heartbeats, 2);
    }

    #[test]
    fn test_node_connectivity_timeout() {
        let mut node = NodeConnectivity::new(1);
        node.record_missed_heartbeat();
        node.record_missed_heartbeat();
        node.record_missed_heartbeat();

        assert!(node.is_timeout(1000, 3));
    }

    #[tokio::test]
    async fn test_partition_detector_creation() {
        let config = PartitionDetectionConfig::default();
        let detector = PartitionDetector::new(1, config);

        assert_eq!(detector.node_id, 1);
        assert_eq!(detector.get_status().await, PartitionStatus::Healthy);
    }

    #[tokio::test]
    async fn test_register_and_unregister_node() {
        let config = PartitionDetectionConfig::default();
        let detector = PartitionDetector::new(1, config);

        detector.register_node(2).await;
        detector.register_node(3).await;

        let connectivity = detector.get_connectivity().await;
        assert_eq!(connectivity.len(), 2);
        assert!(connectivity.contains_key(&2));
        assert!(connectivity.contains_key(&3));

        detector.unregister_node(2).await;
        let connectivity = detector.get_connectivity().await;
        assert_eq!(connectivity.len(), 1);
        assert!(!connectivity.contains_key(&2));
    }

    #[tokio::test]
    async fn test_record_heartbeat() {
        let config = PartitionDetectionConfig::default();
        let detector = PartitionDetector::new(1, config);

        detector.register_node(2).await;
        detector.record_heartbeat(2, 15.5).await;

        let connectivity = detector.get_connectivity().await;
        let node = connectivity.get(&2).unwrap();

        assert!(node.is_reachable);
        assert_eq!(node.missed_heartbeats, 0);
        assert!(node.avg_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_partition_detection() {
        let mut config = PartitionDetectionConfig::default();
        config.heartbeat_timeout_ms = 100;
        config.max_missed_heartbeats = 1;

        let detector = PartitionDetector::new(1, config);
        detector.register_node(2).await;

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        let partitioned = detector.check_for_partitions().await;
        assert_eq!(partitioned.len(), 1);
        assert_eq!(partitioned[0], 2);

        assert_eq!(detector.get_status().await, PartitionStatus::Partitioned);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let mut config = PartitionDetectionConfig::default();
        config.heartbeat_timeout_ms = 100;
        config.max_missed_heartbeats = 1;

        let detector = PartitionDetector::new(1, config);
        detector.register_node(2).await;

        tokio::time::sleep(Duration::from_millis(150)).await;
        detector.check_for_partitions().await;

        let metrics = detector.get_metrics().await;
        assert_eq!(metrics.total_partitions, 1);
    }

    #[tokio::test]
    async fn test_partition_history() {
        let mut config = PartitionDetectionConfig::default();
        config.heartbeat_timeout_ms = 100;
        config.max_missed_heartbeats = 1;

        let detector = PartitionDetector::new(1, config);
        detector.register_node(2).await;

        tokio::time::sleep(Duration::from_millis(150)).await;
        detector.check_for_partitions().await;

        let history = detector.get_history(10).await;
        assert!(!history.is_empty());
    }
}
