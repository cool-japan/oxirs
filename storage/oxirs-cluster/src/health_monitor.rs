//! # Health Monitoring and Failure Detection
//!
//! Comprehensive health monitoring system for cluster nodes including
//! heartbeat monitoring, failure detection, and automated recovery.

use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info};

/// Health status of a cluster node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    /// Current health status
    pub status: NodeHealthLevel,
    /// System metrics
    pub system_metrics: SystemMetrics,
    /// Response time for health checks
    pub response_time: Duration,
    /// Last health check timestamp
    pub last_checked: u64,
}

impl Default for NodeHealth {
    fn default() -> Self {
        Self {
            status: NodeHealthLevel::Unknown,
            system_metrics: SystemMetrics::default(),
            response_time: Duration::from_millis(0),
            last_checked: 0,
        }
    }
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealthLevel {
    /// Node is healthy and responsive
    Healthy,
    /// Node is experiencing degraded performance
    Degraded,
    /// Node is suspected to be failed
    Suspected,
    /// Node is confirmed failed
    Failed,
    /// Node status is unknown
    Unknown,
}

/// Complete health status tracking for a node
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeHealthStatus {
    /// Node identifier
    pub node_id: OxirsNodeId,
    /// Overall health status
    pub health: NodeHealth,
    /// Last heartbeat timestamp (milliseconds since epoch)
    pub last_heartbeat: u64,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// System metrics
    pub system_metrics: SystemMetrics,
    /// Raft-specific metrics
    pub raft_metrics: Option<RaftMetrics>,
    /// Last failure timestamp
    pub last_failure: Option<u64>,
    /// Custom health check results
    pub custom_checks: HashMap<String, bool>,
}

/// System metrics for health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage (0.0-1.0)
    pub cpu_usage: f64,
    /// Memory usage percentage (0.0-1.0)
    pub memory_usage: f64,
    /// Disk I/O rate (MB/s)
    pub disk_io_rate: f64,
    /// Network throughput (MB/s)
    pub network_throughput: f64,
    /// Number of active connections
    pub connection_count: u32,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Last update timestamp
    pub timestamp: u64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_io_rate: 0.0,
            network_throughput: 0.0,
            connection_count: 0,
            error_rate: 0.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }
}

/// Raft-specific health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftMetrics {
    /// Leader election frequency (elections per hour)
    pub election_frequency: f64,
    /// Log replication lag (milliseconds)
    pub replication_lag_ms: u64,
    /// Commitment delay (milliseconds)
    pub commitment_delay_ms: u64,
    /// Number of network partitions detected
    pub partition_count: u32,
    /// Vote request rate (requests per second)
    pub vote_request_rate: f64,
    /// Heartbeat interval variance (milliseconds)
    pub heartbeat_variance_ms: u64,
    /// Last update timestamp
    pub timestamp: u64,
}

impl Default for RaftMetrics {
    fn default() -> Self {
        Self {
            election_frequency: 0.0,
            replication_lag_ms: 0,
            commitment_delay_ms: 0,
            partition_count: 0,
            vote_request_rate: 0.0,
            heartbeat_variance_ms: 0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }
}

/// Comprehensive node health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealthInfo {
    /// Node identifier
    pub node_id: OxirsNodeId,
    /// Overall health status
    pub health: NodeHealth,
    /// Last heartbeat timestamp (Unix timestamp in milliseconds)
    pub last_heartbeat: u64,
    /// System metrics
    pub system_metrics: SystemMetrics,
    /// Raft-specific metrics
    pub raft_metrics: RaftMetrics,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Last failure timestamp (Unix timestamp in milliseconds)
    pub last_failure: Option<u64>,
    /// Custom health check results
    pub custom_checks: HashMap<String, bool>,
}

impl NodeHealthStatus {
    pub fn new(node_id: OxirsNodeId) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_millis() as u64;

        Self {
            node_id,
            health: NodeHealth {
                status: NodeHealthLevel::Unknown,
                system_metrics: SystemMetrics::default(),
                response_time: Duration::from_millis(0),
                last_checked: now,
            },
            last_heartbeat: now,
            system_metrics: SystemMetrics::default(),
            raft_metrics: Some(RaftMetrics::default()),
            failure_count: 0,
            last_failure: None,
            custom_checks: HashMap::new(),
        }
    }

    /// Update health status based on current metrics
    pub fn update_health(&mut self) -> NodeHealth {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_millis() as u64;
        let heartbeat_age = Duration::from_millis(now.saturating_sub(self.last_heartbeat));

        // Check if node is unresponsive
        if heartbeat_age > Duration::from_secs(30) {
            self.health.status = NodeHealthLevel::Failed;
            self.health.last_checked = now;
            return self.health.clone();
        }

        // Check system metrics for degradation
        if self.system_metrics.cpu_usage > 0.90
            || self.system_metrics.memory_usage > 0.95
            || self.system_metrics.error_rate > 10.0
        {
            self.health.status = NodeHealthLevel::Degraded;
            self.health.last_checked = now;
            return self.health.clone();
        }

        // Check if we suspect failures
        if heartbeat_age > Duration::from_secs(10) || self.failure_count > 3 {
            self.health.status = NodeHealthLevel::Suspected;
            self.health.last_checked = now;
            return self.health.clone();
        }

        // Check custom health checks
        if self.custom_checks.values().any(|&check| !check) {
            self.health.status = NodeHealthLevel::Degraded;
            self.health.last_checked = now;
            return self.health.clone();
        }

        self.health.status = NodeHealthLevel::Healthy;
        self.health.last_checked = now;
        self.health.clone()
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Failure detection timeout
    pub failure_timeout: Duration,
    /// Number of consecutive failures before marking as failed
    pub failure_threshold: u32,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable system metrics collection
    pub enable_system_metrics: bool,
    /// Enable Raft metrics collection
    pub enable_raft_metrics: bool,
    /// Custom health check functions
    pub custom_checks: Vec<String>,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            failure_timeout: Duration::from_secs(30),
            failure_threshold: 3,
            health_check_interval: Duration::from_secs(10),
            enable_system_metrics: true,
            enable_raft_metrics: true,
            custom_checks: Vec::new(),
        }
    }
}

/// Event types for health monitoring
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// Node became healthy
    NodeHealthy(OxirsNodeId),
    /// Node became degraded
    NodeDegraded(OxirsNodeId, String),
    /// Node is suspected to be failed
    NodeSuspected(OxirsNodeId),
    /// Node failed
    NodeFailed(OxirsNodeId),
    /// Node recovered from failure
    NodeRecovered(OxirsNodeId),
    /// Cluster partition detected
    PartitionDetected(Vec<OxirsNodeId>),
    /// Cluster partition healed
    PartitionHealed,
}

/// Health monitoring and failure detection system
pub struct HealthMonitor {
    /// Configuration
    config: HealthMonitorConfig,
    /// Node health statuses
    node_statuses: Arc<RwLock<HashMap<OxirsNodeId, NodeHealthStatus>>>,
    /// Event channel sender
    event_sender: mpsc::UnboundedSender<HealthEvent>,
    /// Event channel receiver
    event_receiver: Arc<RwLock<mpsc::UnboundedReceiver<HealthEvent>>>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthMonitorConfig) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Self {
            config,
            node_statuses: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Arc::new(RwLock::new(event_receiver)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start health monitoring
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }
        *running = true;

        info!("Starting health monitor");

        // Start heartbeat monitoring task
        self.start_heartbeat_monitoring().await;

        // Start health checking task
        self.start_health_checking().await;

        // Start metrics collection if enabled
        if self.config.enable_system_metrics {
            self.start_system_metrics_collection().await;
        }

        if self.config.enable_raft_metrics {
            self.start_raft_metrics_collection().await;
        }

        Ok(())
    }

    /// Stop health monitoring
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("Health monitor stopped");
    }

    /// Register a node for monitoring
    pub async fn register_node(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        statuses.insert(node_id, NodeHealthStatus::new(node_id));
        info!("Registered node {} for health monitoring", node_id);
        Ok(())
    }

    /// Unregister a node from monitoring
    pub async fn unregister_node(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        statuses.remove(&node_id);
        info!("Unregistered node {} from health monitoring", node_id);
        Ok(())
    }

    /// Record a heartbeat from a node
    pub async fn record_heartbeat(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            let old_health = status.health.clone();
            status.last_heartbeat = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_millis() as u64;
            status.failure_count = 0;
            let new_health = status.update_health();

            // Send health event if status changed
            if old_health.status != new_health.status
                && new_health.status == NodeHealthLevel::Healthy
            {
                let _ = self.event_sender.send(HealthEvent::NodeRecovered(node_id));
            }
        }
        Ok(())
    }

    /// Update system metrics for a node
    pub async fn update_system_metrics(
        &self,
        node_id: OxirsNodeId,
        metrics: SystemMetrics,
    ) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.system_metrics = metrics;
            status.update_health();
        }
        Ok(())
    }

    /// Update Raft metrics for a node
    pub async fn update_raft_metrics(
        &self,
        node_id: OxirsNodeId,
        metrics: RaftMetrics,
    ) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.raft_metrics = Some(metrics);
        }
        Ok(())
    }

    /// Get current health status of all nodes
    pub async fn get_cluster_health(&self) -> HashMap<OxirsNodeId, NodeHealthStatus> {
        let statuses = self.node_statuses.read().await;
        statuses.clone()
    }

    /// Get health status of a specific node
    pub async fn get_node_health(&self, node_id: OxirsNodeId) -> Option<NodeHealthStatus> {
        let statuses = self.node_statuses.read().await;
        statuses.get(&node_id).cloned()
    }

    /// Get next health event
    pub async fn next_event(&self) -> Option<HealthEvent> {
        let mut receiver = self.event_receiver.write().await;
        receiver.recv().await
    }

    /// Check if cluster is healthy
    pub async fn is_cluster_healthy(&self) -> bool {
        let statuses = self.node_statuses.read().await;
        let total_nodes = statuses.len();
        if total_nodes == 0 {
            return false;
        }

        let healthy_nodes = statuses
            .values()
            .filter(|status| matches!(status.health.status, NodeHealthLevel::Healthy))
            .count();

        // Require majority of nodes to be healthy
        healthy_nodes > total_nodes / 2
    }

    /// Start heartbeat monitoring task
    async fn start_heartbeat_monitoring(&self) {
        let node_statuses = self.node_statuses.clone();
        let event_sender = self.event_sender.clone();
        let failure_timeout = self.config.failure_timeout;
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            while *running.read().await {
                interval.tick().await;

                let mut statuses = node_statuses.write().await;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;

                for (node_id, status) in statuses.iter_mut() {
                    let old_health = status.health.clone();
                    let heartbeat_age =
                        Duration::from_millis(now.saturating_sub(status.last_heartbeat));

                    if heartbeat_age > failure_timeout {
                        status.failure_count += 1;
                        if status.failure_count == 1 {
                            status.last_failure = Some(now);
                        }
                    }

                    let new_health = status.update_health();

                    // Send health events on status change
                    if old_health.status != new_health.status {
                        let event = match new_health.status {
                            NodeHealthLevel::Healthy => HealthEvent::NodeHealthy(*node_id),
                            NodeHealthLevel::Degraded => HealthEvent::NodeDegraded(
                                *node_id,
                                "System metrics degraded".to_string(),
                            ),
                            NodeHealthLevel::Suspected => HealthEvent::NodeSuspected(*node_id),
                            NodeHealthLevel::Failed => HealthEvent::NodeFailed(*node_id),
                            NodeHealthLevel::Unknown => continue,
                        };
                        let _ = event_sender.send(event);
                    }
                }
            }
        });
    }

    /// Start health checking task
    async fn start_health_checking(&self) {
        let health_check_interval = self.config.health_check_interval;
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(health_check_interval);

            while *running.read().await {
                interval.tick().await;
                // Perform custom health checks here
                debug!("Performing health checks");
            }
        });
    }

    /// Start system metrics collection task
    async fn start_system_metrics_collection(&self) {
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            while *running.read().await {
                interval.tick().await;
                // Collect system metrics here
                debug!("Collecting system metrics");
            }
        });
    }

    /// Start Raft metrics collection task
    async fn start_raft_metrics_collection(&self) {
        let running = self.running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            while *running.read().await {
                interval.tick().await;
                // Collect Raft metrics here
                debug!("Collecting Raft metrics");
            }
        });
    }

    /// Start monitoring a specific node
    pub async fn start_monitoring(&self, node_id: OxirsNodeId, _address: String) {
        let mut statuses = self.node_statuses.write().await;
        if let std::collections::hash_map::Entry::Vacant(e) = statuses.entry(node_id) {
            let status = NodeHealthStatus::new(node_id);
            e.insert(status);
            info!("Started monitoring node {}", node_id);
        }
    }

    /// Stop monitoring a specific node
    pub async fn stop_monitoring(&self, node_id: OxirsNodeId) {
        let mut statuses = self.node_statuses.write().await;
        statuses.remove(&node_id);
        info!("Stopped monitoring node {}", node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        assert!(!*monitor.running.read().await);
    }

    #[tokio::test]
    async fn test_node_registration() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);

        monitor.register_node(1).await.unwrap();
        let health = monitor.get_node_health(1).await.unwrap();
        assert_eq!(health.node_id, 1);
    }

    #[tokio::test]
    async fn test_heartbeat_recording() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);

        monitor.register_node(1).await.unwrap();
        monitor.record_heartbeat(1).await.unwrap();

        let health = monitor.get_node_health(1).await.unwrap();
        assert_eq!(health.failure_count, 0);
    }

    #[tokio::test]
    async fn test_health_status_update() {
        let mut status = NodeHealthStatus::new(1);

        // Test healthy status
        status.last_heartbeat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assert_eq!(status.update_health().status, NodeHealthLevel::Healthy);

        // Test degraded status
        status.system_metrics.cpu_usage = 0.95;
        assert_eq!(status.update_health().status, NodeHealthLevel::Degraded);

        // Test failed status
        status.last_heartbeat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            - Duration::from_secs(60).as_millis() as u64;
        assert_eq!(status.update_health().status, NodeHealthLevel::Failed);
    }

    #[tokio::test]
    async fn test_cluster_health() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);

        // Empty cluster should be unhealthy
        assert!(!monitor.is_cluster_healthy().await);

        // Add healthy nodes
        monitor.register_node(1).await.unwrap();
        monitor.register_node(2).await.unwrap();
        monitor.register_node(3).await.unwrap();

        monitor.record_heartbeat(1).await.unwrap();
        monitor.record_heartbeat(2).await.unwrap();
        monitor.record_heartbeat(3).await.unwrap();

        assert!(monitor.is_cluster_healthy().await);
    }
}
