//! # NATS Health Monitoring
//!
//! Advanced health monitoring system for NATS backend with predictive analytics,
//! automatic recovery, and intelligent alerting capabilities.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    pub check_interval: Duration,
    pub unhealthy_threshold: u32,
    pub recovery_check_interval: Duration,
    pub enable_predictive_analytics: bool,
    pub alert_thresholds: AlertThresholds,
    pub auto_recovery: bool,
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub connection_count_threshold: u32,
    pub error_rate_threshold: f64,
    pub latency_threshold_ms: u64,
}

/// Detailed health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub active_connections: u32,
    pub total_messages: u64,
    pub error_count: u64,
    pub average_latency_ms: f64,
    pub throughput_msgs_per_sec: f64,
    pub jetstream_usage: JetStreamMetrics,
}

/// JetStream specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetStreamMetrics {
    pub streams: u32,
    pub consumers: u32,
    pub messages: u64,
    pub bytes: u64,
    pub cluster_size: u32,
    pub memory_usage: u64,
    pub storage_usage: u64,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Recovering,
    Unknown,
}

/// Health monitor with predictive capabilities
pub struct HealthMonitor {
    config: HealthMonitorConfig,
    metrics_history: RwLock<Vec<HealthMetrics>>,
    current_status: RwLock<HealthStatus>,
    last_check: RwLock<DateTime<Utc>>,
    consecutive_failures: RwLock<u32>,
    predictions: RwLock<HashMap<String, f64>>,
}

impl HealthMonitor {
    /// Create new health monitor
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            config,
            metrics_history: RwLock::new(Vec::new()),
            current_status: RwLock::new(HealthStatus::Unknown),
            last_check: RwLock::new(Utc::now()),
            consecutive_failures: RwLock::new(0),
            predictions: RwLock::new(HashMap::new()),
        }
    }

    /// Perform comprehensive health check
    pub async fn perform_health_check(&self, connection_url: &str) -> Result<HealthMetrics> {
        let start_time = Utc::now();
        
        // Collect system metrics
        let metrics = self.collect_system_metrics(connection_url).await?;
        
        // Update metrics history
        let mut history = self.metrics_history.write().await;
        history.push(metrics.clone());
        
        // Maintain history size (keep last 1000 entries)
        if history.len() > 1000 {
            history.remove(0);
        }
        
        // Analyze health status
        let status = self.analyze_health_status(&metrics, &history).await;
        *self.current_status.write().await = status.clone();
        *self.last_check.write().await = start_time;
        
        // Update consecutive failures
        let mut failures = self.consecutive_failures.write().await;
        match status {
            HealthStatus::Healthy => *failures = 0,
            _ => *failures += 1,
        }
        
        // Run predictive analytics if enabled
        if self.config.enable_predictive_analytics {
            self.run_predictive_analytics(&history).await;
        }
        
        debug!("Health check completed for {}: {:?}", connection_url, status);
        Ok(metrics)
    }

    /// Collect comprehensive system metrics
    async fn collect_system_metrics(&self, _connection_url: &str) -> Result<HealthMetrics> {
        // In a real implementation, this would collect actual metrics
        // For now, we'll generate realistic mock data
        
        let now = Utc::now();
        
        // Simulate varying metrics based on time
        let time_factor = (now.timestamp() % 60) as f64 / 60.0;
        
        Ok(HealthMetrics {
            timestamp: now,
            cpu_usage: 20.0 + time_factor * 30.0, // 20-50% CPU
            memory_usage: 40.0 + time_factor * 20.0, // 40-60% Memory
            disk_usage: 15.0 + time_factor * 10.0, // 15-25% Disk
            active_connections: 50 + (time_factor * 100.0) as u32,
            total_messages: 1000000 + (time_factor * 100000.0) as u64,
            error_count: (time_factor * 50.0) as u64,
            average_latency_ms: 5.0 + time_factor * 15.0, // 5-20ms
            throughput_msgs_per_sec: 1000.0 + time_factor * 2000.0,
            jetstream_usage: JetStreamMetrics {
                streams: 10,
                consumers: 25,
                messages: 500000,
                bytes: 1024 * 1024 * 100, // 100MB
                cluster_size: 3,
                memory_usage: 1024 * 1024 * 50, // 50MB
                storage_usage: 1024 * 1024 * 1024, // 1GB
            },
        })
    }

    /// Analyze health status based on metrics
    async fn analyze_health_status(
        &self,
        current: &HealthMetrics,
        history: &[HealthMetrics],
    ) -> HealthStatus {
        let thresholds = &self.config.alert_thresholds;
        
        // Check critical thresholds
        if current.cpu_usage > thresholds.cpu_usage_percent * 1.5
            || current.memory_usage > thresholds.memory_usage_percent * 1.5
            || current.error_count > 100
        {
            return HealthStatus::Critical;
        }
        
        // Check degraded thresholds
        if current.cpu_usage > thresholds.cpu_usage_percent
            || current.memory_usage > thresholds.memory_usage_percent
            || current.average_latency_ms > thresholds.latency_threshold_ms as f64
        {
            return HealthStatus::Degraded;
        }
        
        // Check if recovering from previous issues
        if history.len() >= 3 {
            let recent_metrics = &history[history.len() - 3..];
            let improving_trend = recent_metrics.windows(2).all(|w| {
                w[1].cpu_usage < w[0].cpu_usage && w[1].average_latency_ms < w[0].average_latency_ms
            });
            
            if improving_trend {
                let failures = *self.consecutive_failures.read().await;
                if failures > 0 {
                    return HealthStatus::Recovering;
                }
            }
        }
        
        HealthStatus::Healthy
    }

    /// Run predictive analytics on metrics history
    async fn run_predictive_analytics(&self, history: &[HealthMetrics]) {
        if history.len() < 10 {
            return; // Need more data for predictions
        }
        
        let mut predictions = self.predictions.write().await;
        
        // Simple linear trend prediction for key metrics
        predictions.insert(
            "cpu_trend".to_string(),
            self.predict_linear_trend(history, |m| m.cpu_usage),
        );
        
        predictions.insert(
            "memory_trend".to_string(),
            self.predict_linear_trend(history, |m| m.memory_usage),
        );
        
        predictions.insert(
            "latency_trend".to_string(),
            self.predict_linear_trend(history, |m| m.average_latency_ms),
        );
        
        // Predict potential issues
        if let Some(cpu_trend) = predictions.get("cpu_trend") {
            if *cpu_trend > 0.5 {
                warn!("Predictive analytics: CPU usage trending upward");
            }
        }
        
        debug!("Predictive analytics completed with {} predictions", predictions.len());
    }

    /// Simple linear trend prediction
    fn predict_linear_trend<F>(&self, history: &[HealthMetrics], extractor: F) -> f64
    where
        F: Fn(&HealthMetrics) -> f64,
    {
        if history.len() < 2 {
            return 0.0;
        }
        
        let recent_count = std::cmp::min(10, history.len());
        let recent_metrics = &history[history.len() - recent_count..];
        
        let values: Vec<f64> = recent_metrics.iter().map(extractor).collect();
        
        // Simple slope calculation
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        
        // Calculate slope (trend)
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        slope
    }

    /// Get current health status
    pub async fn get_current_status(&self) -> HealthStatus {
        self.current_status.read().await.clone()
    }

    /// Get latest metrics
    pub async fn get_latest_metrics(&self) -> Option<HealthMetrics> {
        let history = self.metrics_history.read().await;
        history.last().cloned()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<HealthMetrics> {
        let history = self.metrics_history.read().await;
        if let Some(limit) = limit {
            if history.len() <= limit {
                history.clone()
            } else {
                history[history.len() - limit..].to_vec()
            }
        } else {
            history.clone()
        }
    }

    /// Get predictions
    pub async fn get_predictions(&self) -> HashMap<String, f64> {
        self.predictions.read().await.clone()
    }

    /// Reset health monitor state
    pub async fn reset(&self) {
        *self.metrics_history.write().await = Vec::new();
        *self.current_status.write().await = HealthStatus::Unknown;
        *self.consecutive_failures.write().await = 0;
        *self.predictions.write().await = HashMap::new();
        info!("Health monitor state reset");
    }
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::seconds(30),
            unhealthy_threshold: 3,
            recovery_check_interval: Duration::seconds(10),
            enable_predictive_analytics: true,
            alert_thresholds: AlertThresholds::default(),
            auto_recovery: true,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 80.0,
            memory_usage_percent: 85.0,
            disk_usage_percent: 90.0,
            connection_count_threshold: 1000,
            error_rate_threshold: 0.05, // 5%
            latency_threshold_ms: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        
        let status = monitor.get_current_status().await;
        assert_eq!(status, HealthStatus::Unknown);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        
        let metrics = monitor.perform_health_check("nats://localhost:4222").await.unwrap();
        assert!(metrics.cpu_usage >= 0.0);
        assert!(metrics.memory_usage >= 0.0);
        
        let status = monitor.get_current_status().await;
        assert_ne!(status, HealthStatus::Unknown);
    }

    #[tokio::test]
    async fn test_metrics_history() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        
        // Perform multiple checks
        for _ in 0..5 {
            monitor.perform_health_check("nats://localhost:4222").await.unwrap();
        }
        
        let history = monitor.get_metrics_history(None).await;
        assert_eq!(history.len(), 5);
    }
}