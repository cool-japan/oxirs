//! # Adaptive Leader Election Tuning
//!
//! Dynamically adjusts leader election timeouts based on cluster health,
//! network conditions, and node performance to optimize availability.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

use crate::raft::OxirsNodeId;

/// Adaptive election configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveElectionConfig {
    /// Base timeout (milliseconds)
    pub base_timeout_ms: u64,
    /// Minimum timeout (milliseconds)
    pub min_timeout_ms: u64,
    /// Maximum timeout (milliseconds)
    pub max_timeout_ms: u64,
    /// Adjustment step (milliseconds)
    pub adjustment_step_ms: u64,
    /// Enable automatic adjustment
    pub enable_auto_adjustment: bool,
    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,
}

impl Default for AdaptiveElectionConfig {
    fn default() -> Self {
        Self {
            base_timeout_ms: 150,
            min_timeout_ms: 50,
            max_timeout_ms: 1000,
            adjustment_step_ms: 10,
            enable_auto_adjustment: true,
            health_check_interval_secs: 10,
        }
    }
}

/// Node election metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionMetrics {
    /// Total elections participated
    pub total_elections: u64,
    /// Elections won
    pub elections_won: u64,
    /// Average election duration (ms)
    pub avg_election_duration_ms: f64,
    /// Failed elections
    pub failed_elections: u64,
    /// Current timeout (ms)
    pub current_timeout_ms: u64,
    /// Network latency (ms)
    pub network_latency_ms: f64,
    /// Node health score (0.0-1.0)
    pub health_score: f64,
}

impl Default for ElectionMetrics {
    fn default() -> Self {
        Self {
            total_elections: 0,
            elections_won: 0,
            avg_election_duration_ms: 0.0,
            failed_elections: 0,
            current_timeout_ms: 150,
            network_latency_ms: 0.0,
            health_score: 1.0,
        }
    }
}

/// Adaptive leader election tuner
pub struct AdaptiveLeaderElection {
    config: AdaptiveElectionConfig,
    /// Node election metrics
    metrics: Arc<RwLock<BTreeMap<OxirsNodeId, ElectionMetrics>>>,
    /// Global timeout
    global_timeout: Arc<RwLock<u64>>,
}

impl AdaptiveLeaderElection {
    /// Create a new adaptive leader election tuner
    pub fn new(config: AdaptiveElectionConfig) -> Self {
        Self {
            global_timeout: Arc::new(RwLock::new(config.base_timeout_ms)),
            config,
            metrics: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Register a node
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut metrics = self.metrics.write().await;
        metrics.insert(
            node_id,
            ElectionMetrics {
                current_timeout_ms: self.config.base_timeout_ms,
                ..Default::default()
            },
        );
        info!("Registered node {} for adaptive election", node_id);
    }

    /// Record an election event
    pub async fn record_election(&self, node_id: OxirsNodeId, won: bool, duration_ms: u64) {
        let mut metrics = self.metrics.write().await;

        if let Some(node_metrics) = metrics.get_mut(&node_id) {
            node_metrics.total_elections += 1;
            if won {
                node_metrics.elections_won += 1;
            } else {
                node_metrics.failed_elections += 1;
            }

            // Update average
            let total = node_metrics.total_elections as f64;
            node_metrics.avg_election_duration_ms =
                (node_metrics.avg_election_duration_ms * (total - 1.0) + duration_ms as f64)
                    / total;
        }
    }

    /// Update node health score
    pub async fn update_health_score(&self, node_id: &OxirsNodeId, score: f64) {
        let mut metrics = self.metrics.write().await;
        if let Some(node_metrics) = metrics.get_mut(node_id) {
            node_metrics.health_score = score.clamp(0.0, 1.0);
        }
    }

    /// Update network latency
    pub async fn update_network_latency(&self, node_id: &OxirsNodeId, latency_ms: f64) {
        let mut metrics = self.metrics.write().await;
        if let Some(node_metrics) = metrics.get_mut(node_id) {
            node_metrics.network_latency_ms = latency_ms;
        }
    }

    /// Perform automatic timeout adjustment
    pub async fn perform_auto_adjustment(&self) {
        if !self.config.enable_auto_adjustment {
            return;
        }

        let mut metrics = self.metrics.write().await;
        let mut adjustments_made = 0;

        for (node_id, node_metrics) in metrics.iter_mut() {
            let old_timeout = node_metrics.current_timeout_ms;
            let mut new_timeout = old_timeout;

            // Adjust based on health score
            if node_metrics.health_score < 0.5 {
                // Unhealthy node: increase timeout
                new_timeout += self.config.adjustment_step_ms * 2;
            } else if node_metrics.health_score > 0.9 {
                // Very healthy node: can decrease timeout
                new_timeout = new_timeout.saturating_sub(self.config.adjustment_step_ms);
            }

            // Adjust based on network latency
            if node_metrics.network_latency_ms > 100.0 {
                new_timeout += self.config.adjustment_step_ms;
            } else if node_metrics.network_latency_ms < 10.0 {
                new_timeout = new_timeout.saturating_sub(self.config.adjustment_step_ms);
            }

            // Adjust based on election success rate
            if node_metrics.total_elections > 10 {
                let success_rate =
                    node_metrics.elections_won as f64 / node_metrics.total_elections as f64;
                if success_rate < 0.3 {
                    // Low success rate: increase timeout
                    new_timeout += self.config.adjustment_step_ms;
                } else if success_rate > 0.7 {
                    // High success rate: can be more aggressive
                    new_timeout = new_timeout.saturating_sub(self.config.adjustment_step_ms / 2);
                }
            }

            // Clamp to min/max
            new_timeout = new_timeout.clamp(self.config.min_timeout_ms, self.config.max_timeout_ms);

            if new_timeout != old_timeout {
                node_metrics.current_timeout_ms = new_timeout;
                adjustments_made += 1;
                info!(
                    "Adjusted election timeout for node {}: {}ms -> {}ms",
                    node_id, old_timeout, new_timeout
                );
            }
        }

        // Update global timeout (average of all nodes)
        if !metrics.is_empty() {
            let avg_timeout: u64 =
                metrics.values().map(|m| m.current_timeout_ms).sum::<u64>() / metrics.len() as u64;
            *self.global_timeout.write().await = avg_timeout;
        }

        if adjustments_made > 0 {
            info!("Adjusted timeouts for {} nodes", adjustments_made);
        }
    }

    /// Get recommended timeout for a node
    pub async fn get_timeout(&self, node_id: &OxirsNodeId) -> Duration {
        let metrics = self.metrics.read().await;
        let timeout_ms = metrics
            .get(node_id)
            .map(|m| m.current_timeout_ms)
            .unwrap_or(self.config.base_timeout_ms);

        Duration::from_millis(timeout_ms)
    }

    /// Get global timeout
    pub async fn get_global_timeout(&self) -> Duration {
        Duration::from_millis(*self.global_timeout.read().await)
    }

    /// Get metrics for a node
    pub async fn get_metrics(&self, node_id: &OxirsNodeId) -> Option<ElectionMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(node_id).cloned()
    }

    /// Get all metrics
    pub async fn get_all_metrics(&self) -> BTreeMap<OxirsNodeId, ElectionMetrics> {
        self.metrics.read().await.clone()
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.metrics.write().await.clear();
        *self.global_timeout.write().await = self.config.base_timeout_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adaptive_election_creation() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        let timeout = election.get_global_timeout().await;
        assert_eq!(timeout.as_millis(), 150);
    }

    #[tokio::test]
    async fn test_register_node() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;

        let metrics = election.get_metrics(&1).await;
        assert!(metrics.is_some());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.current_timeout_ms, 150);
    }

    #[tokio::test]
    async fn test_record_election() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;

        election.record_election(1, true, 100).await;
        election.record_election(1, false, 150).await;

        let metrics = election.get_metrics(&1).await.unwrap();
        assert_eq!(metrics.total_elections, 2);
        assert_eq!(metrics.elections_won, 1);
        assert_eq!(metrics.failed_elections, 1);
    }

    #[tokio::test]
    async fn test_health_score_update() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.update_health_score(&1, 0.75).await;

        let metrics = election.get_metrics(&1).await.unwrap();
        assert_eq!(metrics.health_score, 0.75);
    }

    #[tokio::test]
    async fn test_network_latency_update() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.update_network_latency(&1, 50.0).await;

        let metrics = election.get_metrics(&1).await.unwrap();
        assert_eq!(metrics.network_latency_ms, 50.0);
    }

    #[tokio::test]
    async fn test_auto_adjustment_unhealthy() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.update_health_score(&1, 0.3).await; // Unhealthy

        let timeout_before = election.get_timeout(&1).await;

        election.perform_auto_adjustment().await;

        let timeout_after = election.get_timeout(&1).await;
        assert!(timeout_after > timeout_before);
    }

    #[tokio::test]
    async fn test_auto_adjustment_high_latency() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.update_health_score(&1, 0.5).await; // Neutral health to isolate latency effect
        election.update_network_latency(&1, 150.0).await; // High latency

        let timeout_before = election.get_timeout(&1).await;

        election.perform_auto_adjustment().await;

        let timeout_after = election.get_timeout(&1).await;
        assert!(timeout_after > timeout_before);
    }

    #[tokio::test]
    async fn test_timeout_bounds() {
        let config = AdaptiveElectionConfig {
            min_timeout_ms: 50,
            max_timeout_ms: 200,
            ..Default::default()
        };
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;

        // Set very unhealthy to try to exceed max
        election.update_health_score(&1, 0.0).await;
        for _ in 0..20 {
            election.perform_auto_adjustment().await;
        }

        let timeout = election.get_timeout(&1).await;
        assert!(timeout.as_millis() <= 200);
        assert!(timeout.as_millis() >= 50);
    }

    #[tokio::test]
    async fn test_global_timeout() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.register_node(2).await;

        election.update_health_score(&1, 0.3).await;
        election.update_health_score(&2, 0.9).await;

        election.perform_auto_adjustment().await;

        let global_timeout = election.get_global_timeout().await;
        assert!(global_timeout.as_millis() > 0);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = AdaptiveElectionConfig::default();
        let election = AdaptiveLeaderElection::new(config);

        election.register_node(1).await;
        election.register_node(2).await;

        election.clear().await;

        let metrics = election.get_all_metrics().await;
        assert!(metrics.is_empty());
    }
}
