//! Real-time metrics collection for streaming adaptation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Real-time metrics collector
#[derive(Debug)]
pub struct RealTimeMetricsCollector {
    pub total_adaptations: u64,
    pub adaptation_rate: f64,
    pub average_response_time: Duration,
    pub current_throughput: f64,
    pub last_adaptation_time: Option<SystemTime>,
    metrics_history: VecDeque<RealTimeMetrics>,
}

impl RealTimeMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            total_adaptations: 0,
            adaptation_rate: 0.0,
            average_response_time: Duration::from_millis(0),
            current_throughput: 0.0,
            last_adaptation_time: None,
            metrics_history: VecDeque::new(),
        }
    }

    /// Collect current metrics snapshot
    pub async fn collect_current_metrics(&mut self) -> Result<RealTimeMetrics> {
        let metrics = RealTimeMetrics {
            timestamp: SystemTime::now(),
            cpu_usage: self.get_cpu_usage().await?,
            memory_usage: self.get_memory_usage().await?,
            throughput: self.current_throughput,
            latency: self.average_response_time,
            error_rate: self.calculate_error_rate().await?,
        };

        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        Ok(metrics)
    }

    /// Collect and store metrics
    pub async fn collect_metrics(&mut self) -> Result<()> {
        let _ = self.collect_current_metrics().await?;
        Ok(())
    }

    /// Update throughput measurement
    pub fn update_throughput(&mut self, new_throughput: f64) {
        self.current_throughput = new_throughput;
    }

    /// Record adaptation event
    pub fn record_adaptation(&mut self) {
        self.total_adaptations += 1;
        self.last_adaptation_time = Some(SystemTime::now());
        self.update_adaptation_rate();
    }

    /// Update response time
    pub fn update_response_time(&mut self, response_time: Duration) {
        // Simple moving average
        self.average_response_time = Duration::from_millis(
            (self.average_response_time.as_millis() as u64 + response_time.as_millis() as u64) / 2
        );
    }

    /// Get adaptation statistics
    pub fn get_adaptation_stats(&self) -> RealTimeAdaptationStats {
        RealTimeAdaptationStats {
            active_streams: 0, // Would be tracked separately
            total_adaptations: self.total_adaptations,
            adaptation_rate: self.adaptation_rate,
            average_response_time: self.average_response_time,
            current_throughput: self.current_throughput,
            stream_health: self.calculate_stream_health(),
            last_adaptation: self.last_adaptation_time,
        }
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> &VecDeque<RealTimeMetrics> {
        &self.metrics_history
    }

    /// Get latest metrics
    pub fn get_latest_metrics(&self) -> Option<&RealTimeMetrics> {
        self.metrics_history.back()
    }

    // Private helper methods
    async fn get_cpu_usage(&self) -> Result<f64> {
        // In a real implementation, this would get actual CPU usage
        // For now, return a simulated value
        Ok(0.5)
    }

    async fn get_memory_usage(&self) -> Result<f64> {
        // In a real implementation, this would get actual memory usage
        // For now, return a simulated value
        Ok(0.6)
    }

    async fn calculate_error_rate(&self) -> Result<f64> {
        // Calculate error rate based on recent metrics
        if self.metrics_history.len() < 10 {
            return Ok(0.01);
        }

        let recent_metrics: Vec<_> = self.metrics_history
            .iter()
            .rev()
            .take(10)
            .collect();

        let avg_error_rate = recent_metrics
            .iter()
            .map(|m| m.error_rate)
            .sum::<f64>() / recent_metrics.len() as f64;

        Ok(avg_error_rate)
    }

    fn update_adaptation_rate(&mut self) {
        if let Some(last_time) = self.last_adaptation_time {
            if let Ok(duration) = last_time.elapsed() {
                let hours = duration.as_secs_f64() / 3600.0;
                if hours > 0.0 {
                    self.adaptation_rate = self.total_adaptations as f64 / hours;
                }
            }
        }
    }

    fn calculate_stream_health(&self) -> f64 {
        if self.metrics_history.is_empty() {
            return 1.0;
        }

        let recent_metrics: Vec<_> = self.metrics_history
            .iter()
            .rev()
            .take(5)
            .collect();

        let avg_error_rate = recent_metrics
            .iter()
            .map(|m| m.error_rate)
            .sum::<f64>() / recent_metrics.len() as f64;

        let avg_cpu_usage = recent_metrics
            .iter()
            .map(|m| m.cpu_usage)
            .sum::<f64>() / recent_metrics.len() as f64;

        let avg_memory_usage = recent_metrics
            .iter()
            .map(|m| m.memory_usage)
            .sum::<f64>() / recent_metrics.len() as f64;

        // Calculate health score (lower is better for error rate, resource usage)
        let health = 1.0 - (avg_error_rate + (avg_cpu_usage * 0.5) + (avg_memory_usage * 0.3));
        health.max(0.0).min(1.0)
    }
}

/// Real-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    pub timestamp: SystemTime,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub throughput: f64,
    pub latency: Duration,
    pub error_rate: f64,
}

/// Real-time adaptation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeAdaptationStats {
    pub active_streams: usize,
    pub total_adaptations: u64,
    pub adaptation_rate: f64,
    pub average_response_time: Duration,
    pub current_throughput: f64,
    pub stream_health: f64,
    pub last_adaptation: Option<SystemTime>,
}

/// Performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics_collector: RealTimeMetricsCollector,
    performance_thresholds: PerformanceThresholds,
    alert_handlers: Vec<Box<dyn AlertHandler>>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            metrics_collector: RealTimeMetricsCollector::new(),
            performance_thresholds: PerformanceThresholds::default(),
            alert_handlers: Vec::new(),
        }
    }

    /// Add alert handler
    pub fn add_alert_handler(&mut self, handler: Box<dyn AlertHandler>) {
        self.alert_handlers.push(handler);
    }

    /// Monitor performance and trigger alerts
    pub async fn monitor_performance(&mut self) -> Result<()> {
        let metrics = self.metrics_collector.collect_current_metrics().await?;
        
        // Check thresholds
        if metrics.cpu_usage > self.performance_thresholds.max_cpu_usage {
            self.trigger_alert(AlertType::HighCpuUsage, metrics.cpu_usage).await?;
        }

        if metrics.memory_usage > self.performance_thresholds.max_memory_usage {
            self.trigger_alert(AlertType::HighMemoryUsage, metrics.memory_usage).await?;
        }

        if metrics.error_rate > self.performance_thresholds.max_error_rate {
            self.trigger_alert(AlertType::HighErrorRate, metrics.error_rate).await?;
        }

        if metrics.latency > self.performance_thresholds.max_latency {
            self.trigger_alert(AlertType::HighLatency, metrics.latency.as_millis() as f64).await?;
        }

        Ok(())
    }

    /// Get metrics collector
    pub fn metrics_collector_mut(&mut self) -> &mut RealTimeMetricsCollector {
        &mut self.metrics_collector
    }

    /// Get current metrics
    pub async fn get_current_metrics(&mut self) -> Result<RealTimeMetrics> {
        self.metrics_collector.collect_current_metrics().await
    }

    // Private helper methods
    async fn trigger_alert(&self, alert_type: AlertType, value: f64) -> Result<()> {
        let alert = Alert {
            alert_id: Uuid::new_v4(),
            alert_type: alert_type.clone(),
            timestamp: SystemTime::now(),
            value,
            message: format!("Performance threshold exceeded: {:?} = {}", alert_type, value),
        };

        for handler in &self.alert_handlers {
            handler.handle_alert(&alert).await?;
        }

        Ok(())
    }
}

/// Performance thresholds for alerting
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_cpu_usage: f64,
    pub max_memory_usage: f64,
    pub max_error_rate: f64,
    pub max_latency: Duration,
    pub min_throughput: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_cpu_usage: 0.8,
            max_memory_usage: 0.9,
            max_error_rate: 0.1,
            max_latency: Duration::from_millis(1000),
            min_throughput: 100.0,
        }
    }
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighErrorRate,
    HighLatency,
    LowThroughput,
}

/// Alert structure
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_id: Uuid,
    pub alert_type: AlertType,
    pub timestamp: SystemTime,
    pub value: f64,
    pub message: String,
}

/// Alert handler trait
#[async_trait::async_trait]
pub trait AlertHandler: Send + Sync + std::fmt::Debug {
    async fn handle_alert(&self, alert: &Alert) -> Result<()>;
}

/// Simple logging alert handler
#[derive(Debug)]
pub struct LoggingAlertHandler;

#[async_trait::async_trait]
impl AlertHandler for LoggingAlertHandler {
    async fn handle_alert(&self, alert: &Alert) -> Result<()> {
        tracing::warn!("Alert triggered: {} - {}", alert.alert_id, alert.message);
        Ok(())
    }
}

impl Default for RealTimeMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}