//! # Enhanced Cluster Health Monitoring
//!
//! Comprehensive health monitoring with predictive alerts for proactive
//! cluster management. Tracks node health, resource utilization, and
//! predicts potential failures before they occur.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::raft::OxirsNodeId;

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoringConfig {
    /// Health check interval (seconds)
    pub check_interval_secs: u64,
    /// Alert threshold for CPU usage (0.0-1.0)
    pub cpu_alert_threshold: f64,
    /// Alert threshold for memory usage (0.0-1.0)
    pub memory_alert_threshold: f64,
    /// Alert threshold for disk usage (0.0-1.0)
    pub disk_alert_threshold: f64,
    /// Number of historical samples to keep
    pub history_size: usize,
    /// Prediction window size (samples)
    pub prediction_window: usize,
    /// Alert cooldown period (seconds)
    pub alert_cooldown_secs: u64,
}

impl Default for HealthMonitoringConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 30,
            cpu_alert_threshold: 0.85,
            memory_alert_threshold: 0.90,
            disk_alert_threshold: 0.85,
            history_size: 1000,
            prediction_window: 10,
            alert_cooldown_secs: 300,
        }
    }
}

/// Node health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Node is healthy
    Healthy,
    /// Node is degraded but operational
    Degraded,
    /// Node is critical and may fail
    Critical,
    /// Node is unresponsive
    Unresponsive,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,
    /// Memory usage (0.0-1.0)
    pub memory_usage: f64,
    /// Disk usage (0.0-1.0)
    pub disk_usage: f64,
    /// Network bandwidth usage (bytes/sec)
    pub network_usage: u64,
    /// Active connections count
    pub active_connections: usize,
    /// Timestamp of measurement
    pub timestamp: SystemTime,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_usage: 0,
            active_connections: 0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Node health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Current health status
    pub status: HealthStatus,
    /// Current resource metrics
    pub current_metrics: ResourceMetrics,
    /// Historical metrics
    pub metrics_history: VecDeque<ResourceMetrics>,
    /// Last health check timestamp
    pub last_check: SystemTime,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
}

/// Health alert types
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertType {
    /// High CPU usage
    HighCpu,
    /// High memory usage
    HighMemory,
    /// High disk usage
    HighDisk,
    /// Node unresponsive
    NodeUnresponsive,
    /// Predicted resource exhaustion
    PredictedFailure,
    /// Degraded performance
    DegradedPerformance,
}

/// Health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Affected node
    pub node_id: OxirsNodeId,
    /// Alert severity (0.0-1.0)
    pub severity: f64,
    /// Alert message
    pub message: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Predicted time to failure (if applicable)
    pub time_to_failure: Option<Duration>,
}

/// Health monitoring statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthMonitoringStats {
    /// Total health checks performed
    pub total_health_checks: u64,
    /// Total alerts generated
    pub total_alerts: u64,
    /// Alerts by type
    pub alerts_by_type: BTreeMap<String, u64>,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Nodes currently degraded
    pub degraded_nodes: usize,
    /// Nodes currently critical
    pub critical_nodes: usize,
}

/// Enhanced cluster health monitoring
pub struct HealthMonitoring {
    config: HealthMonitoringConfig,
    node_health: Arc<RwLock<BTreeMap<OxirsNodeId, NodeHealth>>>,
    active_alerts: Arc<RwLock<Vec<HealthAlert>>>,
    alert_history: Arc<RwLock<VecDeque<HealthAlert>>>,
    last_alert_time: Arc<RwLock<BTreeMap<(OxirsNodeId, AlertType), SystemTime>>>,
    stats: Arc<RwLock<HealthMonitoringStats>>,
}

impl HealthMonitoring {
    /// Create a new health monitoring system
    pub fn new(config: HealthMonitoringConfig) -> Self {
        Self {
            config,
            node_health: Arc::new(RwLock::new(BTreeMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            last_alert_time: Arc::new(RwLock::new(BTreeMap::new())),
            stats: Arc::new(RwLock::new(HealthMonitoringStats::default())),
        }
    }

    /// Register a node for health monitoring
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut node_health = self.node_health.write().await;
        node_health.insert(
            node_id.clone(),
            NodeHealth {
                node_id,
                status: HealthStatus::Healthy,
                current_metrics: ResourceMetrics::default(),
                metrics_history: VecDeque::with_capacity(self.config.history_size),
                last_check: SystemTime::now(),
                consecutive_failures: 0,
            },
        );
    }

    /// Unregister a node from health monitoring
    pub async fn unregister_node(&self, node_id: &OxirsNodeId) {
        let mut node_health = self.node_health.write().await;
        node_health.remove(node_id);
    }

    /// Update node health metrics
    pub async fn update_metrics(&self, node_id: &OxirsNodeId, metrics: ResourceMetrics) {
        let mut node_health = self.node_health.write().await;

        if let Some(health) = node_health.get_mut(node_id) {
            // Update current metrics
            health.current_metrics = metrics.clone();
            health.last_check = SystemTime::now();
            health.consecutive_failures = 0;

            // Add to history
            health.metrics_history.push_back(metrics);
            if health.metrics_history.len() > self.config.history_size {
                health.metrics_history.pop_front();
            }

            // Update health status
            self.update_health_status(health).await;
        }

        let mut stats = self.stats.write().await;
        stats.total_health_checks += 1;
    }

    /// Update health status based on metrics
    async fn update_health_status(&self, health: &mut NodeHealth) {
        let metrics = &health.current_metrics;

        let new_status = if metrics.cpu_usage > self.config.cpu_alert_threshold
            || metrics.memory_usage > self.config.memory_alert_threshold
            || metrics.disk_usage > self.config.disk_alert_threshold
        {
            HealthStatus::Critical
        } else if metrics.cpu_usage > self.config.cpu_alert_threshold * 0.8
            || metrics.memory_usage > self.config.memory_alert_threshold * 0.8
            || metrics.disk_usage > self.config.disk_alert_threshold * 0.8
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        if new_status != health.status {
            debug!(
                "Node {:?} health status changed from {:?} to {:?}",
                health.node_id, health.status, new_status
            );
            health.status = new_status;
        }
    }

    /// Record a health check failure
    pub async fn record_failure(&self, node_id: &OxirsNodeId) {
        let mut node_health = self.node_health.write().await;

        let should_alert = if let Some(health) = node_health.get_mut(node_id) {
            health.consecutive_failures += 1;
            health.last_check = SystemTime::now();

            if health.consecutive_failures >= 3 {
                health.status = HealthStatus::Unresponsive;
                Some(health.consecutive_failures)
            } else {
                None
            }
        } else {
            None
        };

        drop(node_health);

        if let Some(failures) = should_alert {
            self.generate_alert(
                *node_id,
                AlertType::NodeUnresponsive,
                1.0,
                format!("Node has failed {} consecutive health checks", failures),
                None,
            )
            .await;
        }
    }

    /// Check all nodes for potential issues
    pub async fn check_health(&self) -> Vec<HealthAlert> {
        let node_health = self.node_health.read().await;
        let mut alerts = Vec::new();

        for (node_id, health) in node_health.iter() {
            // Check CPU usage
            if health.current_metrics.cpu_usage > self.config.cpu_alert_threshold {
                if self
                    .should_generate_alert(node_id, &AlertType::HighCpu)
                    .await
                {
                    alerts.push(HealthAlert {
                        alert_type: AlertType::HighCpu,
                        node_id: node_id.clone(),
                        severity: health.current_metrics.cpu_usage,
                        message: format!(
                            "High CPU usage: {:.1}%",
                            health.current_metrics.cpu_usage * 100.0
                        ),
                        timestamp: SystemTime::now(),
                        time_to_failure: None,
                    });
                }
            }

            // Check memory usage
            if health.current_metrics.memory_usage > self.config.memory_alert_threshold {
                if self
                    .should_generate_alert(node_id, &AlertType::HighMemory)
                    .await
                {
                    alerts.push(HealthAlert {
                        alert_type: AlertType::HighMemory,
                        node_id: node_id.clone(),
                        severity: health.current_metrics.memory_usage,
                        message: format!(
                            "High memory usage: {:.1}%",
                            health.current_metrics.memory_usage * 100.0
                        ),
                        timestamp: SystemTime::now(),
                        time_to_failure: None,
                    });
                }
            }

            // Check disk usage
            if health.current_metrics.disk_usage > self.config.disk_alert_threshold {
                if self
                    .should_generate_alert(node_id, &AlertType::HighDisk)
                    .await
                {
                    alerts.push(HealthAlert {
                        alert_type: AlertType::HighDisk,
                        node_id: node_id.clone(),
                        severity: health.current_metrics.disk_usage,
                        message: format!(
                            "High disk usage: {:.1}%",
                            health.current_metrics.disk_usage * 100.0
                        ),
                        timestamp: SystemTime::now(),
                        time_to_failure: None,
                    });
                }
            }

            // Predictive analysis
            if let Some((alert_type, time_to_failure)) = self.predict_failure(health).await {
                if self
                    .should_generate_alert(node_id, &AlertType::PredictedFailure)
                    .await
                {
                    alerts.push(HealthAlert {
                        alert_type,
                        node_id: node_id.clone(),
                        severity: 0.8,
                        message: format!("Predicted resource exhaustion in {:?}", time_to_failure),
                        timestamp: SystemTime::now(),
                        time_to_failure: Some(time_to_failure),
                    });
                }
            }
        }

        // Store alerts
        for alert in &alerts {
            self.generate_alert(
                alert.node_id.clone(),
                alert.alert_type.clone(),
                alert.severity,
                alert.message.clone(),
                alert.time_to_failure,
            )
            .await;
        }

        alerts
    }

    /// Predict potential failures based on historical trends
    async fn predict_failure(&self, health: &NodeHealth) -> Option<(AlertType, Duration)> {
        if health.metrics_history.len() < self.config.prediction_window {
            return None;
        }

        let recent_metrics: Vec<_> = health
            .metrics_history
            .iter()
            .rev()
            .take(self.config.prediction_window)
            .collect();

        // Predict CPU exhaustion
        if let Some(ttf) = self.predict_resource_exhaustion(
            &recent_metrics,
            |m| m.cpu_usage,
            self.config.cpu_alert_threshold,
        ) {
            return Some((AlertType::PredictedFailure, ttf));
        }

        // Predict memory exhaustion
        if let Some(ttf) = self.predict_resource_exhaustion(
            &recent_metrics,
            |m| m.memory_usage,
            self.config.memory_alert_threshold,
        ) {
            return Some((AlertType::PredictedFailure, ttf));
        }

        // Predict disk exhaustion
        if let Some(ttf) = self.predict_resource_exhaustion(
            &recent_metrics,
            |m| m.disk_usage,
            self.config.disk_alert_threshold,
        ) {
            return Some((AlertType::PredictedFailure, ttf));
        }

        None
    }

    /// Predict when a resource will be exhausted based on linear regression
    fn predict_resource_exhaustion<F>(
        &self,
        metrics: &[&ResourceMetrics],
        extractor: F,
        threshold: f64,
    ) -> Option<Duration>
    where
        F: Fn(&ResourceMetrics) -> f64,
    {
        if metrics.len() < 2 {
            return None;
        }

        // Calculate trend using simple linear regression
        let values: Vec<f64> = metrics.iter().map(|m| extractor(m)).collect();

        // Calculate mean
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        // Calculate slope (rate of change per sample)
        let mut slope_sum = 0.0;
        for i in 1..values.len() {
            slope_sum += values[i] - values[i - 1];
        }
        let slope = slope_sum / (values.len() - 1) as f64;

        // If slope is negative or zero, no exhaustion predicted
        if slope <= 0.0 {
            return None;
        }

        // Calculate current value
        let current = values.last().copied().unwrap_or(mean);

        // If already above threshold, return immediately
        if current >= threshold {
            return None;
        }

        // Calculate samples until exhaustion
        let samples_to_exhaustion = ((threshold - current) / slope).ceil() as u64;

        // Assume each sample represents check_interval_secs
        let seconds_to_exhaustion =
            samples_to_exhaustion.saturating_mul(self.config.check_interval_secs);

        // Only alert if exhaustion predicted within reasonable timeframe (e.g., 1 hour)
        if seconds_to_exhaustion > 0 && seconds_to_exhaustion <= 3600 {
            Some(Duration::from_secs(seconds_to_exhaustion))
        } else {
            None
        }
    }

    /// Check if an alert should be generated (respects cooldown)
    async fn should_generate_alert(&self, node_id: &OxirsNodeId, alert_type: &AlertType) -> bool {
        let last_alert_time = self.last_alert_time.read().await;
        let key = (node_id.clone(), alert_type.clone());

        if let Some(last_time) = last_alert_time.get(&key) {
            if let Ok(elapsed) = SystemTime::now().duration_since(*last_time) {
                if elapsed.as_secs() < self.config.alert_cooldown_secs {
                    return false;
                }
            }
        }

        true
    }

    /// Generate a health alert
    async fn generate_alert(
        &self,
        node_id: OxirsNodeId,
        alert_type: AlertType,
        severity: f64,
        message: String,
        time_to_failure: Option<Duration>,
    ) {
        let alert = HealthAlert {
            alert_type: alert_type.clone(),
            node_id: node_id.clone(),
            severity,
            message: message.clone(),
            timestamp: SystemTime::now(),
            time_to_failure,
        };

        warn!("Health alert: {:?}", alert);

        // Add to active alerts
        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.push(alert.clone());

        // Add to history
        let mut alert_history = self.alert_history.write().await;
        alert_history.push_back(alert.clone());
        if alert_history.len() > self.config.history_size {
            alert_history.pop_front();
        }

        // Update last alert time
        let mut last_alert_time = self.last_alert_time.write().await;
        last_alert_time.insert((node_id, alert_type.clone()), SystemTime::now());

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_alerts += 1;
        *stats
            .alerts_by_type
            .entry(format!("{:?}", alert_type))
            .or_insert(0) += 1;

        // Update node counts
        let node_health = self.node_health.read().await;
        stats.degraded_nodes = node_health
            .values()
            .filter(|h| h.status == HealthStatus::Degraded)
            .count();
        stats.critical_nodes = node_health
            .values()
            .filter(|h| h.status == HealthStatus::Critical)
            .count();
    }

    /// Get current health status for a node
    pub async fn get_node_health(&self, node_id: &OxirsNodeId) -> Option<NodeHealth> {
        let node_health = self.node_health.read().await;
        node_health.get(node_id).cloned()
    }

    /// Get health status for all nodes
    pub async fn get_all_health(&self) -> BTreeMap<OxirsNodeId, NodeHealth> {
        self.node_health.read().await.clone()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<HealthAlert> {
        self.active_alerts.read().await.clone()
    }

    /// Clear resolved alerts
    pub async fn clear_alerts(&self, node_id: &OxirsNodeId) {
        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.retain(|alert| &alert.node_id != node_id);
    }

    /// Get alert history
    pub async fn get_alert_history(&self) -> Vec<HealthAlert> {
        self.alert_history.read().await.iter().cloned().collect()
    }

    /// Get monitoring statistics
    pub async fn get_stats(&self) -> HealthMonitoringStats {
        self.stats.read().await.clone()
    }

    /// Get overall cluster health
    pub async fn get_cluster_health(&self) -> HealthStatus {
        let node_health = self.node_health.read().await;

        let critical_count = node_health
            .values()
            .filter(|h| {
                h.status == HealthStatus::Critical || h.status == HealthStatus::Unresponsive
            })
            .count();

        let degraded_count = node_health
            .values()
            .filter(|h| h.status == HealthStatus::Degraded)
            .count();

        let total_nodes = node_health.len();

        if critical_count > 0 || (degraded_count as f64 / total_nodes as f64) > 0.5 {
            HealthStatus::Critical
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitoring_creation() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_health_checks, 0);
        assert_eq!(stats.total_alerts, 0);
    }

    #[tokio::test]
    async fn test_register_node() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let health = monitor.get_node_health(&node_id).await;
        assert!(health.is_some());

        let health = health.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_update_metrics() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.5,
            memory_usage: 0.5,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;

        let health = monitor.get_node_health(&node_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.current_metrics.cpu_usage, 0.5);
    }

    #[tokio::test]
    async fn test_high_cpu_alert() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.5,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;

        let alerts = monitor.check_health().await;
        assert!(!alerts.is_empty());
        assert!(alerts.iter().any(|a| a.alert_type == AlertType::HighCpu));
    }

    #[tokio::test]
    async fn test_health_status_degraded() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.75,
            memory_usage: 0.75,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;

        let health = monitor.get_node_health(&node_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[tokio::test]
    async fn test_health_status_critical() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.95,
            disk_usage: 0.95,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;

        let health = monitor.get_node_health(&node_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Critical);
    }

    #[tokio::test]
    async fn test_record_failure() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        monitor.record_failure(&node_id).await;
        monitor.record_failure(&node_id).await;
        monitor.record_failure(&node_id).await;

        let health = monitor.get_node_health(&node_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Unresponsive);
        assert_eq!(health.consecutive_failures, 3);
    }

    #[tokio::test]
    async fn test_alert_history() {
        let config = HealthMonitoringConfig {
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.5,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;
        monitor.check_health().await;

        let history = monitor.get_alert_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_clear_alerts() {
        let config = HealthMonitoringConfig {
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.5,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;
        monitor.check_health().await;

        let alerts_before = monitor.get_active_alerts().await;
        assert!(!alerts_before.is_empty());

        monitor.clear_alerts(&node_id).await;

        let alerts_after = monitor.get_active_alerts().await;
        assert!(alerts_after.is_empty());
    }

    #[tokio::test]
    async fn test_get_all_health() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node1: OxirsNodeId = 1;
        let node2: OxirsNodeId = 2;

        monitor.register_node(node1).await;
        monitor.register_node(node2).await;

        let all_health = monitor.get_all_health().await;
        assert_eq!(all_health.len(), 2);
    }

    #[tokio::test]
    async fn test_cluster_health() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitoring::new(config);

        let node1: OxirsNodeId = 1;
        let node2: OxirsNodeId = 2;

        monitor.register_node(node1).await;
        monitor.register_node(node2).await;

        let cluster_health = monitor.get_cluster_health().await;
        assert_eq!(cluster_health, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = HealthMonitoringConfig {
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let monitor = HealthMonitoring::new(config);

        let node_id: OxirsNodeId = 1;
        monitor.register_node(node_id).await;

        let metrics = ResourceMetrics {
            cpu_usage: 0.95,
            memory_usage: 0.5,
            disk_usage: 0.5,
            network_usage: 1000,
            active_connections: 10,
            timestamp: SystemTime::now(),
        };

        monitor.update_metrics(&node_id, metrics).await;
        monitor.check_health().await;

        let stats = monitor.get_stats().await;
        assert!(stats.total_health_checks > 0);
        assert!(stats.total_alerts > 0);
        assert_eq!(stats.critical_nodes, 1);
    }
}
