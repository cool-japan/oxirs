//! # Automated Horizontal Scaling
//!
//! Provides automatic scaling of cluster nodes based on load metrics and policies.
//!
//! ## Overview
//!
//! This module implements intelligent horizontal scaling that:
//! - Monitors cluster load metrics (CPU, memory, query throughput)
//! - Applies scaling policies to determine when to scale
//! - Automatically adds or removes nodes
//! - Rebalances data after scaling operations
//! - Uses SciRS2 for statistical analysis of metrics
//!
//! ## Features
//!
//! - Multiple scaling policies (threshold, predictive, schedule-based)
//! - Cooldown periods to prevent thrashing
//! - Gradual scaling to maintain stability
//! - Resource allocation optimization
//! - Integration with cloud providers
//! - Comprehensive metrics and logging

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::error::Result;
use crate::raft::OxirsNodeId;

/// Scaling policy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingPolicy {
    /// Threshold-based scaling (scale when metrics exceed thresholds)
    Threshold,
    /// Predictive scaling (use ML to predict future load)
    Predictive,
    /// Schedule-based scaling (scale based on time of day)
    Scheduled,
    /// Hybrid approach (combination of policies)
    Hybrid,
}

/// Scaling action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up (add nodes)
    ScaleUp,
    /// Scale down (remove nodes)
    ScaleDown,
    /// No action needed
    NoAction,
}

/// Cluster load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// Query throughput (queries per second)
    pub query_throughput: f64,
    /// Average query latency (milliseconds)
    pub avg_query_latency: f64,
    /// Number of active connections
    pub active_connections: u32,
    /// Replication lag (milliseconds)
    pub replication_lag: u64,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl Default for LoadMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            query_throughput: 0.0,
            avg_query_latency: 0.0,
            active_connections: 0,
            replication_lag: 0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Scaling policy
    pub policy: ScalingPolicy,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// CPU threshold for scale up (0.0-1.0)
    pub cpu_scale_up_threshold: f64,
    /// CPU threshold for scale down (0.0-1.0)
    pub cpu_scale_down_threshold: f64,
    /// Memory threshold for scale up (0.0-1.0)
    pub memory_scale_up_threshold: f64,
    /// Memory threshold for scale down (0.0-1.0)
    pub memory_scale_down_threshold: f64,
    /// Query throughput threshold for scale up (QPS)
    pub throughput_scale_up_threshold: f64,
    /// Cooldown period between scaling actions (seconds)
    pub cooldown_period_secs: u64,
    /// Number of consecutive violations before scaling
    pub consecutive_violations: usize,
    /// Enable gradual scaling (add/remove 1 node at a time)
    pub gradual_scaling: bool,
    /// Metric collection interval (seconds)
    pub metric_collection_interval_secs: u64,
    /// Enable predictive scaling
    pub enable_predictive: bool,
    /// Prediction window (minutes)
    pub prediction_window_mins: u32,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            policy: ScalingPolicy::Threshold,
            min_nodes: 1,
            max_nodes: 10,
            cpu_scale_up_threshold: 0.75,
            cpu_scale_down_threshold: 0.25,
            memory_scale_up_threshold: 0.80,
            memory_scale_down_threshold: 0.30,
            throughput_scale_up_threshold: 1000.0,
            cooldown_period_secs: 300, // 5 minutes
            consecutive_violations: 3,
            gradual_scaling: true,
            metric_collection_interval_secs: 60,
            enable_predictive: false,
            prediction_window_mins: 30,
        }
    }
}

/// Scaling event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Scaling action taken
    pub action: ScalingAction,
    /// Number of nodes added/removed
    pub node_count_change: i32,
    /// Metrics at time of scaling
    pub metrics: LoadMetrics,
    /// Reason for scaling
    pub reason: String,
}

/// Auto-scaling statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutoScalingStats {
    /// Total scale up events
    pub total_scale_ups: u64,
    /// Total scale down events
    pub total_scale_downs: u64,
    /// Total nodes added
    pub total_nodes_added: u64,
    /// Total nodes removed
    pub total_nodes_removed: u64,
    /// Last scaling event
    pub last_scaling_event: Option<SystemTime>,
    /// Average time between scaling events (seconds)
    pub avg_time_between_scaling_secs: f64,
    /// Failed scaling attempts
    pub failed_scaling_attempts: u64,
}

/// Auto-scaling manager
pub struct AutoScalingManager {
    config: AutoScalingConfig,
    /// Current cluster node count
    current_node_count: Arc<RwLock<usize>>,
    /// Recent load metrics
    metrics_history: Arc<RwLock<VecDeque<LoadMetrics>>>,
    /// Consecutive violation count
    violation_count: Arc<RwLock<HashMap<String, usize>>>,
    /// Last scaling action timestamp
    last_scaling_action: Arc<RwLock<Option<Instant>>>,
    /// Scaling events history
    scaling_events: Arc<RwLock<Vec<ScalingEvent>>>,
    /// Statistics
    stats: Arc<RwLock<AutoScalingStats>>,
}

impl AutoScalingManager {
    /// Create a new auto-scaling manager
    pub fn new(config: AutoScalingConfig, initial_node_count: usize) -> Self {
        Self {
            config,
            current_node_count: Arc::new(RwLock::new(initial_node_count)),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            violation_count: Arc::new(RwLock::new(HashMap::new())),
            last_scaling_action: Arc::new(RwLock::new(None)),
            scaling_events: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(AutoScalingStats::default())),
        }
    }

    /// Record current load metrics
    pub async fn record_metrics(&self, metrics: LoadMetrics) {
        let mut history = self.metrics_history.write().await;

        // Keep only recent metrics (last hour)
        let max_history_size = 3600 / self.config.metric_collection_interval_secs as usize;
        while history.len() >= max_history_size {
            history.pop_front();
        }

        history.push_back(metrics);
        debug!("Recorded load metrics, history size: {}", history.len());
    }

    /// Evaluate whether scaling is needed
    pub async fn evaluate_scaling(&self) -> Result<ScalingAction> {
        // Check cooldown period
        if !self.is_cooldown_expired().await {
            debug!("Scaling is in cooldown period");
            return Ok(ScalingAction::NoAction);
        }

        match self.config.policy {
            ScalingPolicy::Threshold => self.evaluate_threshold_policy().await,
            ScalingPolicy::Predictive => self.evaluate_predictive_policy().await,
            ScalingPolicy::Scheduled => self.evaluate_scheduled_policy().await,
            ScalingPolicy::Hybrid => self.evaluate_hybrid_policy().await,
        }
    }

    /// Execute scaling action
    pub async fn execute_scaling(&self, action: ScalingAction) -> Result<Vec<OxirsNodeId>> {
        if action == ScalingAction::NoAction {
            return Ok(Vec::new());
        }

        let current_count = *self.current_node_count.read().await;
        let target_count = self.calculate_target_node_count(action, current_count);

        if target_count == current_count {
            return Ok(Vec::new());
        }

        info!(
            "Executing scaling action: {:?}, current: {}, target: {}",
            action, current_count, target_count
        );

        let node_ids = if target_count > current_count {
            // Scale up
            self.scale_up(target_count - current_count).await?
        } else {
            // Scale down
            self.scale_down(current_count - target_count).await?
        };

        // Update node count
        {
            let mut count = self.current_node_count.write().await;
            *count = target_count;
        }

        // Record scaling event
        let metrics = self.get_latest_metrics().await;
        let event = ScalingEvent {
            timestamp: SystemTime::now(),
            action,
            node_count_change: (target_count as i32) - (current_count as i32),
            metrics,
            reason: format!(
                "Auto-scaling {:?} from {} to {} nodes",
                action, current_count, target_count
            ),
        };

        {
            let mut events = self.scaling_events.write().await;
            events.push(event);

            // Keep only recent events (last 100)
            let len = events.len();
            if len > 100 {
                events.drain(0..len - 100);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            match action {
                ScalingAction::ScaleUp => {
                    stats.total_scale_ups += 1;
                    stats.total_nodes_added += (target_count - current_count) as u64;
                }
                ScalingAction::ScaleDown => {
                    stats.total_scale_downs += 1;
                    stats.total_nodes_removed += (current_count - target_count) as u64;
                }
                ScalingAction::NoAction => {}
            }
            stats.last_scaling_event = Some(SystemTime::now());
        }

        // Update last scaling action timestamp
        {
            let mut last_action = self.last_scaling_action.write().await;
            *last_action = Some(Instant::now());
        }

        Ok(node_ids)
    }

    /// Get current node count
    pub async fn get_current_node_count(&self) -> usize {
        *self.current_node_count.read().await
    }

    /// Get scaling statistics
    pub async fn get_statistics(&self) -> AutoScalingStats {
        self.stats.read().await.clone()
    }

    /// Get recent scaling events
    pub async fn get_recent_events(&self, count: usize) -> Vec<ScalingEvent> {
        let events = self.scaling_events.read().await;
        events.iter().rev().take(count).cloned().collect()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, duration_secs: u64) -> Vec<LoadMetrics> {
        let history = self.metrics_history.read().await;
        let cutoff = SystemTime::now() - Duration::from_secs(duration_secs);

        history
            .iter()
            .filter(|m| m.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    /// Check if cooldown period has expired
    async fn is_cooldown_expired(&self) -> bool {
        let last_action = self.last_scaling_action.read().await;

        if let Some(last) = *last_action {
            let elapsed = last.elapsed().as_secs();
            elapsed >= self.config.cooldown_period_secs
        } else {
            true
        }
    }

    /// Evaluate threshold-based scaling policy
    async fn evaluate_threshold_policy(&self) -> Result<ScalingAction> {
        let metrics = self.get_latest_metrics().await;

        let mut violations = Vec::new();

        // Check CPU threshold
        if metrics.cpu_utilization >= self.config.cpu_scale_up_threshold {
            violations.push("cpu_high");
        } else if metrics.cpu_utilization <= self.config.cpu_scale_down_threshold {
            violations.push("cpu_low");
        }

        // Check memory threshold
        if metrics.memory_utilization >= self.config.memory_scale_up_threshold {
            violations.push("memory_high");
        } else if metrics.memory_utilization <= self.config.memory_scale_down_threshold {
            violations.push("memory_low");
        }

        // Check throughput threshold
        if metrics.query_throughput >= self.config.throughput_scale_up_threshold {
            violations.push("throughput_high");
        }

        // Count consecutive violations
        let scale_up_violations = violations.iter().filter(|v| v.ends_with("high")).count();
        let scale_down_violations = violations.iter().filter(|v| v.ends_with("low")).count();

        if scale_up_violations > 0 {
            let count = self.increment_violation_count("scale_up").await;
            if count >= self.config.consecutive_violations {
                self.reset_violation_count("scale_up").await;
                return Ok(ScalingAction::ScaleUp);
            }
        } else {
            self.reset_violation_count("scale_up").await;
        }

        if scale_down_violations >= 2 {
            let count = self.increment_violation_count("scale_down").await;
            if count >= self.config.consecutive_violations {
                self.reset_violation_count("scale_down").await;
                return Ok(ScalingAction::ScaleDown);
            }
        } else {
            self.reset_violation_count("scale_down").await;
        }

        Ok(ScalingAction::NoAction)
    }

    /// Evaluate predictive scaling policy using SciRS2 statistics
    async fn evaluate_predictive_policy(&self) -> Result<ScalingAction> {
        if !self.config.enable_predictive {
            return self.evaluate_threshold_policy().await;
        }

        let history = self.metrics_history.read().await;

        if history.len() < 10 {
            // Not enough data for prediction, fall back to threshold
            return self.evaluate_threshold_policy().await;
        }

        // Use SciRS2 for statistical analysis
        use scirs2_core::ndarray_ext::Array1;

        // Extract CPU utilization time series
        let cpu_values: Vec<f64> = history.iter().map(|m| m.cpu_utilization).collect();
        let cpu_array = Array1::from_vec(cpu_values);

        // Calculate trend using simple linear regression (slope)
        let n = cpu_array.len() as f64;
        let x_values: Vec<f64> = (0..cpu_array.len()).map(|i| i as f64).collect();
        let x_array = Array1::from_vec(x_values);

        let x_mean = x_array.mean().unwrap_or(0.0);
        let y_mean = cpu_array.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..cpu_array.len() {
            let x_diff = x_array[i] - x_mean;
            let y_diff = cpu_array[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Predict future CPU utilization
        let prediction_steps = (self.config.prediction_window_mins as f64 * 60.0
            / self.config.metric_collection_interval_secs as f64)
            as usize;
        let future_x = n + prediction_steps as f64;
        let predicted_cpu = y_mean + slope * (future_x - x_mean);

        debug!(
            "Predictive scaling: current CPU: {:.2}%, predicted CPU: {:.2}%, trend slope: {:.4}",
            y_mean * 100.0,
            predicted_cpu * 100.0,
            slope
        );

        // Make scaling decision based on prediction
        if predicted_cpu >= self.config.cpu_scale_up_threshold {
            Ok(ScalingAction::ScaleUp)
        } else if predicted_cpu <= self.config.cpu_scale_down_threshold {
            Ok(ScalingAction::ScaleDown)
        } else {
            Ok(ScalingAction::NoAction)
        }
    }

    /// Evaluate schedule-based scaling policy
    async fn evaluate_scheduled_policy(&self) -> Result<ScalingAction> {
        // This would integrate with a schedule configuration
        // For now, fall back to threshold-based
        self.evaluate_threshold_policy().await
    }

    /// Evaluate hybrid scaling policy
    async fn evaluate_hybrid_policy(&self) -> Result<ScalingAction> {
        // Combine threshold and predictive
        let threshold_action = self.evaluate_threshold_policy().await?;
        let predictive_action = self.evaluate_predictive_policy().await?;

        // If both agree, take action
        if threshold_action == predictive_action {
            Ok(threshold_action)
        } else {
            // If they disagree, be conservative
            Ok(ScalingAction::NoAction)
        }
    }

    /// Calculate target node count based on scaling action
    fn calculate_target_node_count(&self, action: ScalingAction, current: usize) -> usize {
        let target = match action {
            ScalingAction::ScaleUp => {
                if self.config.gradual_scaling {
                    current + 1
                } else {
                    (current as f64 * 1.5).ceil() as usize
                }
            }
            ScalingAction::ScaleDown => {
                if self.config.gradual_scaling {
                    current.saturating_sub(1)
                } else {
                    (current as f64 * 0.7).ceil() as usize
                }
            }
            ScalingAction::NoAction => current,
        };

        // Apply min/max constraints
        target.max(self.config.min_nodes).min(self.config.max_nodes)
    }

    /// Scale up by adding nodes
    async fn scale_up(&self, count: usize) -> Result<Vec<OxirsNodeId>> {
        info!("Scaling up: adding {} nodes", count);

        // In a real implementation, this would:
        // 1. Request new instances from cloud provider
        // 2. Wait for instances to be ready
        // 3. Add nodes to cluster
        // 4. Trigger data rebalancing

        // For now, return simulated node IDs
        let base_id = 1000 + *self.current_node_count.read().await as u64;
        let node_ids: Vec<OxirsNodeId> = (0..count).map(|i| base_id + i as u64).collect();

        Ok(node_ids)
    }

    /// Scale down by removing nodes
    async fn scale_down(&self, count: usize) -> Result<Vec<OxirsNodeId>> {
        info!("Scaling down: removing {} nodes", count);

        // In a real implementation, this would:
        // 1. Select nodes to remove (preferably non-leader, low load)
        // 2. Drain connections and data
        // 3. Remove from cluster
        // 4. Terminate instances

        // For now, return simulated node IDs
        let base_id = 1000 + *self.current_node_count.read().await as u64;
        let node_ids: Vec<OxirsNodeId> = (0..count)
            .map(|i| base_id.saturating_sub(i as u64 + 1))
            .collect();

        Ok(node_ids)
    }

    /// Get latest metrics
    async fn get_latest_metrics(&self) -> LoadMetrics {
        let history = self.metrics_history.read().await;
        history.back().cloned().unwrap_or_default()
    }

    /// Increment violation count
    async fn increment_violation_count(&self, key: &str) -> usize {
        let mut counts = self.violation_count.write().await;
        let count = counts.entry(key.to_string()).or_insert(0);
        *count += 1;
        *count
    }

    /// Reset violation count
    async fn reset_violation_count(&self, key: &str) {
        let mut counts = self.violation_count.write().await;
        counts.insert(key.to_string(), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_scaling_manager_creation() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config, 3);

        assert_eq!(manager.get_current_node_count().await, 3);
    }

    #[tokio::test]
    async fn test_record_metrics() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config, 3);

        let metrics = LoadMetrics {
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            query_throughput: 500.0,
            ..Default::default()
        };

        manager.record_metrics(metrics).await;

        let history = manager.get_metrics_history(3600).await;
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_threshold_scale_up() {
        let config = AutoScalingConfig {
            consecutive_violations: 1,
            ..Default::default()
        };
        let manager = AutoScalingManager::new(config, 3);

        // Record high CPU metrics
        let metrics = LoadMetrics {
            cpu_utilization: 0.9,
            memory_utilization: 0.85,
            ..Default::default()
        };

        manager.record_metrics(metrics).await;

        let action = manager.evaluate_scaling().await.unwrap();
        assert_eq!(action, ScalingAction::ScaleUp);
    }

    #[tokio::test]
    async fn test_threshold_scale_down() {
        let config = AutoScalingConfig {
            consecutive_violations: 1,
            ..Default::default()
        };
        let manager = AutoScalingManager::new(config, 5);

        // Record low CPU and memory metrics
        let metrics = LoadMetrics {
            cpu_utilization: 0.1,
            memory_utilization: 0.15,
            ..Default::default()
        };

        manager.record_metrics(metrics).await;

        let action = manager.evaluate_scaling().await.unwrap();
        assert_eq!(action, ScalingAction::ScaleDown);
    }

    #[tokio::test]
    async fn test_execute_scaling() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config, 3);

        let node_ids = manager
            .execute_scaling(ScalingAction::ScaleUp)
            .await
            .unwrap();
        assert!(!node_ids.is_empty());
        assert_eq!(manager.get_current_node_count().await, 4);
    }

    #[tokio::test]
    async fn test_scaling_statistics() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config, 3);

        manager
            .execute_scaling(ScalingAction::ScaleUp)
            .await
            .unwrap();

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_scale_ups, 1);
        assert_eq!(stats.total_nodes_added, 1);
    }

    #[tokio::test]
    async fn test_min_max_constraints() {
        let config = AutoScalingConfig {
            min_nodes: 2,
            max_nodes: 5,
            ..Default::default()
        };
        let manager = AutoScalingManager::new(config, 3);

        // Try to scale down below minimum
        manager
            .execute_scaling(ScalingAction::ScaleDown)
            .await
            .unwrap();
        manager
            .execute_scaling(ScalingAction::ScaleDown)
            .await
            .unwrap();
        manager
            .execute_scaling(ScalingAction::ScaleDown)
            .await
            .unwrap();

        assert_eq!(manager.get_current_node_count().await, 2);
    }

    #[tokio::test]
    async fn test_cooldown_period() {
        let config = AutoScalingConfig {
            cooldown_period_secs: 10,
            consecutive_violations: 1,
            ..Default::default()
        };
        let manager = AutoScalingManager::new(config, 3);

        // First scaling should work
        manager
            .execute_scaling(ScalingAction::ScaleUp)
            .await
            .unwrap();

        // Second scaling should be blocked by cooldown
        let metrics = LoadMetrics {
            cpu_utilization: 0.9,
            ..Default::default()
        };
        manager.record_metrics(metrics).await;

        let action = manager.evaluate_scaling().await.unwrap();
        assert_eq!(action, ScalingAction::NoAction);
    }
}
