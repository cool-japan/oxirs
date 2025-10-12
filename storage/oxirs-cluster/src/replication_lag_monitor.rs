//! # Replication Lag Monitoring and Alerting
//!
//! Monitors replication lag across cluster nodes and generates alerts for
//! consistency issues. Critical for maintaining data integrity in distributed systems.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::raft::OxirsNodeId;

/// Replication lag monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationLagConfig {
    /// Warning threshold (seconds)
    pub warning_threshold_secs: u64,
    /// Critical threshold (seconds)
    pub critical_threshold_secs: u64,
    /// Sample retention window (seconds)
    pub retention_window_secs: u64,
    /// Alert cooldown (seconds)
    pub alert_cooldown_secs: u64,
    /// Enable lag prediction
    pub enable_prediction: bool,
}

impl Default for ReplicationLagConfig {
    fn default() -> Self {
        Self {
            warning_threshold_secs: 5,
            critical_threshold_secs: 30,
            retention_window_secs: 3600,
            alert_cooldown_secs: 60,
            enable_prediction: true,
        }
    }
}

/// Lag severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LagSeverity {
    /// No lag
    Normal,
    /// Warning level lag
    Warning,
    /// Critical lag
    Critical,
    /// Replication stalled
    Stalled,
}

/// Replication lag sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagSample {
    /// Leader sequence number
    pub leader_seq: u64,
    /// Follower sequence number
    pub follower_seq: u64,
    /// Lag in entries
    pub lag_entries: u64,
    /// Estimated lag time (seconds)
    pub lag_time_secs: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Node replication status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatus {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Current lag entries
    pub current_lag_entries: u64,
    /// Current lag time (seconds)
    pub current_lag_time_secs: f64,
    /// Lag severity
    pub severity: LagSeverity,
    /// Replication rate (entries/sec)
    pub replication_rate: f64,
    /// Last update time
    pub last_update: SystemTime,
    /// Lag history
    pub lag_history: VecDeque<LagSample>,
    /// Predicted lag time (if enabled)
    pub predicted_lag_secs: Option<f64>,
}

/// Lag alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagAlert {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Severity
    pub severity: LagSeverity,
    /// Lag entries
    pub lag_entries: u64,
    /// Lag time (seconds)
    pub lag_time_secs: f64,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Replication lag statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LagStatistics {
    /// Total nodes monitored
    pub total_nodes: usize,
    /// Nodes with warnings
    pub warning_nodes: usize,
    /// Nodes with critical lag
    pub critical_nodes: usize,
    /// Nodes stalled
    pub stalled_nodes: usize,
    /// Average lag (seconds)
    pub avg_lag_secs: f64,
    /// Max lag (seconds)
    pub max_lag_secs: f64,
    /// Total alerts generated
    pub total_alerts: u64,
}

/// Replication lag monitor
pub struct ReplicationLagMonitor {
    config: ReplicationLagConfig,
    /// Node replication statuses
    statuses: Arc<RwLock<BTreeMap<OxirsNodeId, ReplicationStatus>>>,
    /// Active alerts
    alerts: Arc<RwLock<Vec<LagAlert>>>,
    /// Last alert time per node
    last_alert_time: Arc<RwLock<BTreeMap<OxirsNodeId, SystemTime>>>,
    /// Statistics
    stats: Arc<RwLock<LagStatistics>>,
}

impl ReplicationLagMonitor {
    /// Create a new replication lag monitor
    pub fn new(config: ReplicationLagConfig) -> Self {
        Self {
            config,
            statuses: Arc::new(RwLock::new(BTreeMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            last_alert_time: Arc::new(RwLock::new(BTreeMap::new())),
            stats: Arc::new(RwLock::new(LagStatistics::default())),
        }
    }

    /// Register a node for monitoring
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let status = ReplicationStatus {
            node_id,
            current_lag_entries: 0,
            current_lag_time_secs: 0.0,
            severity: LagSeverity::Normal,
            replication_rate: 0.0,
            last_update: SystemTime::now(),
            lag_history: VecDeque::new(),
            predicted_lag_secs: None,
        };

        let mut statuses = self.statuses.write().await;
        statuses.insert(node_id, status);

        info!("Registered node {} for replication lag monitoring", node_id);
    }

    /// Unregister a node
    pub async fn unregister_node(&self, node_id: &OxirsNodeId) {
        let mut statuses = self.statuses.write().await;
        statuses.remove(node_id);
    }

    /// Update replication lag
    pub async fn update_lag(
        &self,
        node_id: OxirsNodeId,
        leader_seq: u64,
        follower_seq: u64,
        estimated_lag_time_secs: f64,
    ) {
        let mut statuses = self.statuses.write().await;

        let status = match statuses.get_mut(&node_id) {
            Some(s) => s,
            None => return,
        };

        let lag_entries = leader_seq.saturating_sub(follower_seq);

        // Create sample
        let sample = LagSample {
            leader_seq,
            follower_seq,
            lag_entries,
            lag_time_secs: estimated_lag_time_secs,
            timestamp: SystemTime::now(),
        };

        // Update status
        status.current_lag_entries = lag_entries;
        status.current_lag_time_secs = estimated_lag_time_secs;
        status.last_update = SystemTime::now();

        // Add to history
        status.lag_history.push_back(sample.clone());

        // Cleanup old samples
        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(self.config.retention_window_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        while let Some(first) = status.lag_history.front() {
            if first.timestamp < cutoff {
                status.lag_history.pop_front();
            } else {
                break;
            }
        }

        // Calculate replication rate
        if status.lag_history.len() >= 2 {
            let oldest = status.lag_history.front().unwrap();
            let newest = status.lag_history.back().unwrap();

            let time_diff = newest
                .timestamp
                .duration_since(oldest.timestamp)
                .unwrap_or(Duration::from_secs(1))
                .as_secs_f64();

            let entries_processed = oldest.lag_entries.saturating_sub(newest.lag_entries);
            status.replication_rate = entries_processed as f64 / time_diff;
        }

        // Determine severity
        let new_severity = if lag_entries == 0 && estimated_lag_time_secs < 1.0 {
            LagSeverity::Normal
        } else if status.replication_rate < 0.1 && lag_entries > 100 {
            LagSeverity::Stalled
        } else if estimated_lag_time_secs >= self.config.critical_threshold_secs as f64 {
            LagSeverity::Critical
        } else if estimated_lag_time_secs >= self.config.warning_threshold_secs as f64 {
            LagSeverity::Warning
        } else {
            LagSeverity::Normal
        };

        // Predict future lag
        if self.config.enable_prediction {
            status.predicted_lag_secs = self.predict_lag(status);
        }

        let severity_changed = new_severity != status.severity;
        status.severity = new_severity;

        drop(statuses);

        // Generate alert if needed
        if severity_changed && new_severity != LagSeverity::Normal {
            self.generate_alert(node_id, new_severity, lag_entries, estimated_lag_time_secs)
                .await;
        }

        self.update_stats().await;
    }

    /// Predict future lag based on historical trend
    fn predict_lag(&self, status: &ReplicationStatus) -> Option<f64> {
        if status.lag_history.len() < 5 {
            return None;
        }

        // Simple linear regression on last 10 samples
        let samples: Vec<_> = status.lag_history.iter().rev().take(10).collect();

        let n = samples.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, sample) in samples.iter().enumerate() {
            let x = i as f64;
            let y = sample.lag_time_secs;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict 60 seconds ahead
        let predicted = intercept + slope * (n + 60.0);

        Some(predicted.max(0.0))
    }

    /// Generate lag alert
    async fn generate_alert(
        &self,
        node_id: OxirsNodeId,
        severity: LagSeverity,
        lag_entries: u64,
        lag_time_secs: f64,
    ) {
        // Check cooldown
        let last_alert_time = self.last_alert_time.read().await;
        if let Some(last_time) = last_alert_time.get(&node_id) {
            if let Ok(elapsed) = SystemTime::now().duration_since(*last_time) {
                if elapsed.as_secs() < self.config.alert_cooldown_secs {
                    return;
                }
            }
        }
        drop(last_alert_time);

        let message = match severity {
            LagSeverity::Warning => {
                format!(
                    "Replication lag warning: {} entries, {:.1}s behind",
                    lag_entries, lag_time_secs
                )
            }
            LagSeverity::Critical => {
                format!(
                    "CRITICAL replication lag: {} entries, {:.1}s behind",
                    lag_entries, lag_time_secs
                )
            }
            LagSeverity::Stalled => {
                format!("Replication STALLED: {} entries backlog", lag_entries)
            }
            LagSeverity::Normal => return,
        };

        let alert = LagAlert {
            node_id,
            severity,
            lag_entries,
            lag_time_secs,
            message: message.clone(),
            timestamp: SystemTime::now(),
        };

        warn!("Replication lag alert: {}", message);

        let mut alerts = self.alerts.write().await;
        alerts.push(alert);

        let mut last_alert_time = self.last_alert_time.write().await;
        last_alert_time.insert(node_id, SystemTime::now());

        let mut stats = self.stats.write().await;
        stats.total_alerts += 1;
    }

    /// Get replication status for a node
    pub async fn get_status(&self, node_id: &OxirsNodeId) -> Option<ReplicationStatus> {
        let statuses = self.statuses.read().await;
        statuses.get(node_id).cloned()
    }

    /// Get all replication statuses
    pub async fn get_all_statuses(&self) -> BTreeMap<OxirsNodeId, ReplicationStatus> {
        self.statuses.read().await.clone()
    }

    /// Get nodes with critical lag
    pub async fn get_critical_nodes(&self) -> Vec<OxirsNodeId> {
        let statuses = self.statuses.read().await;
        statuses
            .iter()
            .filter(|(_, s)| {
                s.severity == LagSeverity::Critical || s.severity == LagSeverity::Stalled
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get active alerts
    pub async fn get_alerts(&self) -> Vec<LagAlert> {
        self.alerts.read().await.clone()
    }

    /// Clear alerts for a node
    pub async fn clear_alerts(&self, node_id: &OxirsNodeId) {
        let mut alerts = self.alerts.write().await;
        alerts.retain(|alert| &alert.node_id != node_id);
    }

    /// Get statistics
    pub async fn get_stats(&self) -> LagStatistics {
        self.stats.read().await.clone()
    }

    /// Update statistics
    async fn update_stats(&self) {
        let statuses = self.statuses.read().await;

        let mut stats = LagStatistics {
            total_nodes: statuses.len(),
            warning_nodes: 0,
            critical_nodes: 0,
            stalled_nodes: 0,
            avg_lag_secs: 0.0,
            max_lag_secs: 0.0,
            total_alerts: 0,
        };

        let mut total_lag = 0.0;

        for status in statuses.values() {
            match status.severity {
                LagSeverity::Warning => stats.warning_nodes += 1,
                LagSeverity::Critical => stats.critical_nodes += 1,
                LagSeverity::Stalled => stats.stalled_nodes += 1,
                LagSeverity::Normal => {}
            }

            total_lag += status.current_lag_time_secs;
            stats.max_lag_secs = stats.max_lag_secs.max(status.current_lag_time_secs);
        }

        if !statuses.is_empty() {
            stats.avg_lag_secs = total_lag / statuses.len() as f64;
        }

        // Preserve total alerts
        let old_stats = self.stats.read().await;
        stats.total_alerts = old_stats.total_alerts;
        drop(old_stats);

        *self.stats.write().await = stats;
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.statuses.write().await.clear();
        self.alerts.write().await.clear();
        self.last_alert_time.write().await.clear();
        *self.stats.write().await = LagStatistics::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replication_lag_monitor_creation() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_register_node() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;

        let status = monitor.get_status(&1).await;
        assert!(status.is_some());

        let status = status.unwrap();
        assert_eq!(status.severity, LagSeverity::Normal);
    }

    #[tokio::test]
    async fn test_no_lag() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 100, 0.0).await;

        let status = monitor.get_status(&1).await.unwrap();
        assert_eq!(status.current_lag_entries, 0);
        assert_eq!(status.severity, LagSeverity::Normal);
    }

    #[tokio::test]
    async fn test_warning_lag() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 90, 6.0).await; // 6 seconds lag

        let status = monitor.get_status(&1).await.unwrap();
        assert_eq!(status.current_lag_entries, 10);
        assert_eq!(status.severity, LagSeverity::Warning);
    }

    #[tokio::test]
    async fn test_critical_lag() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 50, 35.0).await; // 35 seconds lag

        let status = monitor.get_status(&1).await.unwrap();
        assert_eq!(status.current_lag_entries, 50);
        assert_eq!(status.severity, LagSeverity::Critical);
    }

    #[tokio::test]
    async fn test_get_critical_nodes() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.register_node(2).await;
        monitor.register_node(3).await;

        monitor.update_lag(1, 100, 50, 35.0).await; // Critical
        monitor.update_lag(2, 100, 95, 2.0).await; // Normal
        monitor.update_lag(3, 100, 60, 32.0).await; // Critical

        let critical = monitor.get_critical_nodes().await;
        assert_eq!(critical.len(), 2);
        assert!(critical.contains(&1));
        assert!(critical.contains(&3));
    }

    #[tokio::test]
    async fn test_stats() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.register_node(2).await;

        monitor.update_lag(1, 100, 90, 6.0).await; // Warning
        monitor.update_lag(2, 100, 50, 35.0).await; // Critical

        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.warning_nodes, 1);
        assert_eq!(stats.critical_nodes, 1);
        assert!(stats.avg_lag_secs > 0.0);
    }

    #[tokio::test]
    async fn test_alerts() {
        let config = ReplicationLagConfig {
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 50, 35.0).await;

        let alerts = monitor.get_alerts().await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, LagSeverity::Critical);
    }

    #[tokio::test]
    async fn test_clear_alerts() {
        let config = ReplicationLagConfig {
            alert_cooldown_secs: 0,
            ..Default::default()
        };
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 50, 35.0).await;

        let alerts = monitor.get_alerts().await;
        assert!(!alerts.is_empty());

        monitor.clear_alerts(&1).await;

        let alerts = monitor.get_alerts().await;
        assert!(alerts.is_empty());
    }

    #[tokio::test]
    async fn test_unregister_node() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        assert!(monitor.get_status(&1).await.is_some());

        monitor.unregister_node(&1).await;
        assert!(monitor.get_status(&1).await.is_none());
    }

    #[tokio::test]
    async fn test_clear() {
        let config = ReplicationLagConfig::default();
        let monitor = ReplicationLagMonitor::new(config);

        monitor.register_node(1).await;
        monitor.update_lag(1, 100, 50, 35.0).await;

        monitor.clear().await;

        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_nodes, 0);
    }
}
