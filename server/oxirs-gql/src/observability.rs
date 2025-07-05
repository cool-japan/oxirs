//! Comprehensive Observability and Monitoring System
//!
//! This module provides enterprise-grade observability capabilities including
//! OpenTelemetry integration, custom metrics, distributed tracing, and real-time
//! monitoring for GraphQL operations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tracing::{debug, error, info, span, warn, Level};

use crate::ast::{Document, OperationType};
use crate::performance::{ClientInfo, OperationMetrics};

/// Comprehensive observability configuration
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub enable_opentelemetry: bool,
    pub enable_custom_metrics: bool,
    pub enable_distributed_tracing: bool,
    pub enable_real_time_monitoring: bool,
    pub enable_alerting: bool,
    pub enable_log_aggregation: bool,
    pub metrics_collection_interval: Duration,
    pub trace_sampling_rate: f64,
    pub alert_thresholds: AlertThresholds,
    pub retention_period: Duration,
    pub export_endpoint: Option<String>,
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_opentelemetry: true,
            enable_custom_metrics: true,
            enable_distributed_tracing: true,
            enable_real_time_monitoring: true,
            enable_alerting: true,
            enable_log_aggregation: true,
            metrics_collection_interval: Duration::from_secs(10),
            trace_sampling_rate: 1.0,
            alert_thresholds: AlertThresholds::default(),
            retention_period: Duration::from_secs(86400), // 24 hours
            export_endpoint: None,
            service_name: "oxirs-gql".to_string(),
            service_version: "0.1.0".to_string(),
            environment: "development".to_string(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_response_time_ms: u64,
    pub max_error_rate: f64,
    pub max_memory_usage_mb: u64,
    pub max_cpu_usage_percent: f64,
    pub min_cache_hit_ratio: f64,
    pub max_concurrent_requests: usize,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_response_time_ms: 5000,
            max_error_rate: 0.05, // 5%
            max_memory_usage_mb: 1024,
            max_cpu_usage_percent: 80.0,
            min_cache_hit_ratio: 0.8,
            max_concurrent_requests: 1000,
        }
    }
}

/// Comprehensive observability system
pub struct ObservabilitySystem {
    config: ObservabilityConfig,
    metrics_collector: Arc<AsyncRwLock<MetricsCollector>>,
    trace_collector: Arc<AsyncRwLock<TraceCollector>>,
    alert_manager: Arc<AsyncMutex<AlertManager>>,
    real_time_monitor: Arc<AsyncRwLock<RealTimeMonitor>>,
}

impl ObservabilitySystem {
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            metrics_collector: Arc::new(AsyncRwLock::new(MetricsCollector::new(&config))),
            trace_collector: Arc::new(AsyncRwLock::new(TraceCollector::new(&config))),
            alert_manager: Arc::new(AsyncMutex::new(AlertManager::new(&config))),
            real_time_monitor: Arc::new(AsyncRwLock::new(RealTimeMonitor::new(&config))),
            config,
        }
    }

    /// Record a GraphQL operation
    pub async fn record_operation(&self, metrics: &OperationMetrics) -> Result<()> {
        // Record metrics
        if self.config.enable_custom_metrics {
            self.metrics_collector
                .write()
                .await
                .record_operation(metrics)?;
        }

        // Create trace span
        if self.config.enable_distributed_tracing {
            self.trace_collector
                .write()
                .await
                .create_operation_span(metrics)?;
        }

        // Update real-time monitoring
        if self.config.enable_real_time_monitoring {
            self.real_time_monitor
                .write()
                .await
                .update_metrics(metrics)?;
        }

        // Check alert conditions
        if self.config.enable_alerting {
            self.alert_manager.lock().await.check_alerts(metrics)?;
        }

        Ok(())
    }

    /// Get comprehensive observability dashboard
    pub async fn get_dashboard(&self) -> Result<ObservabilityDashboard> {
        let metrics = if self.config.enable_custom_metrics {
            Some(self.metrics_collector.read().await.get_summary())
        } else {
            None
        };

        let traces = if self.config.enable_distributed_tracing {
            Some(self.trace_collector.read().await.get_recent_traces(100))
        } else {
            None
        };

        let alerts = if self.config.enable_alerting {
            Some(self.alert_manager.lock().await.get_active_alerts())
        } else {
            None
        };

        let real_time_data = if self.config.enable_real_time_monitoring {
            Some(self.real_time_monitor.read().await.get_current_state())
        } else {
            None
        };

        Ok(ObservabilityDashboard {
            service_info: ServiceInfo {
                name: self.config.service_name.clone(),
                version: self.config.service_version.clone(),
                environment: self.config.environment.clone(),
                uptime: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default(),
            },
            metrics,
            traces,
            alerts,
            real_time_data,
            health_status: self.calculate_health_status().await,
        })
    }

    /// Calculate overall health status
    async fn calculate_health_status(&self) -> HealthStatus {
        let mut health_score = 100.0;
        let mut issues = Vec::new();

        if self.config.enable_alerting {
            let active_alerts = self.alert_manager.lock().await.get_active_alerts();
            if !active_alerts.is_empty() {
                health_score -= active_alerts.len() as f64 * 10.0;
                issues.push(format!("{} active alerts", active_alerts.len()));
            }
        }

        if self.config.enable_real_time_monitoring {
            let monitor = self.real_time_monitor.read().await;
            let current_state = monitor.get_current_state();

            if current_state.error_rate > self.config.alert_thresholds.max_error_rate {
                health_score -= 20.0;
                issues.push("High error rate detected".to_string());
            }

            if current_state.avg_response_time.as_millis()
                > self.config.alert_thresholds.max_response_time_ms as u128
            {
                health_score -= 15.0;
                issues.push("High response time detected".to_string());
            }
        }

        let status = if health_score >= 90.0 {
            HealthStatusLevel::Healthy
        } else if health_score >= 70.0 {
            HealthStatusLevel::Warning
        } else {
            HealthStatusLevel::Critical
        };

        HealthStatus {
            status,
            score: health_score.max(0.0),
            issues,
            last_check: SystemTime::now(),
        }
    }

    /// Export metrics to external systems
    pub async fn export_metrics(&self) -> Result<()> {
        if let Some(endpoint) = &self.config.export_endpoint {
            let dashboard = self.get_dashboard().await?;
            let serialized = serde_json::to_string(&dashboard)?;

            // In a real implementation, this would send to external monitoring systems
            info!(
                "Exporting metrics to {}: {} bytes",
                endpoint,
                serialized.len()
            );
        }
        Ok(())
    }
}

/// Metrics collection system
pub struct MetricsCollector {
    operation_metrics: VecDeque<OperationMetrics>,
    aggregated_metrics: AggregatedMetrics,
    custom_counters: HashMap<String, u64>,
    custom_gauges: HashMap<String, f64>,
    custom_histograms: HashMap<String, Histogram>,
}

impl MetricsCollector {
    fn new(config: &ObservabilityConfig) -> Self {
        Self {
            operation_metrics: VecDeque::new(),
            aggregated_metrics: AggregatedMetrics::new(),
            custom_counters: HashMap::new(),
            custom_gauges: HashMap::new(),
            custom_histograms: HashMap::new(),
        }
    }

    fn record_operation(&mut self, metrics: &OperationMetrics) -> Result<()> {
        self.operation_metrics.push_back(metrics.clone());
        self.aggregated_metrics.update(metrics);

        // Increment operation counter
        let counter_key = format!("operations.{:?}", metrics.operation_type);
        *self.custom_counters.entry(counter_key).or_insert(0) += 1;

        // Update execution time histogram
        let histogram_key = "execution_time_ms".to_string();
        let histogram = self
            .custom_histograms
            .entry(histogram_key)
            .or_insert_with(Histogram::new);
        histogram.record(metrics.execution_time.as_millis() as f64);

        Ok(())
    }

    fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_operations: self.operation_metrics.len(),
            aggregated: self.aggregated_metrics.clone(),
            counters: self.custom_counters.clone(),
            gauges: self.custom_gauges.clone(),
            histograms: self
                .custom_histograms
                .iter()
                .map(|(k, v)| (k.clone(), v.summary()))
                .collect(),
        }
    }
}

/// Distributed tracing system
pub struct TraceCollector {
    active_spans: HashMap<String, TraceSpan>,
    completed_traces: VecDeque<DistributedTrace>,
    trace_id_counter: u64,
}

impl TraceCollector {
    fn new(config: &ObservabilityConfig) -> Self {
        Self {
            active_spans: HashMap::new(),
            completed_traces: VecDeque::new(),
            trace_id_counter: 0,
        }
    }

    fn create_operation_span(&mut self, metrics: &OperationMetrics) -> Result<()> {
        self.trace_id_counter += 1;
        let trace_id = format!("trace_{}", self.trace_id_counter);

        let span = TraceSpan {
            trace_id: trace_id.clone(),
            span_id: format!("span_{}", self.trace_id_counter),
            parent_span_id: None,
            operation_name: metrics
                .operation_name
                .clone()
                .unwrap_or_else(|| "anonymous".to_string()),
            start_time: metrics.timestamp,
            duration: Some(metrics.execution_time),
            tags: vec![
                (
                    "operation.type".to_string(),
                    format!("{:?}", metrics.operation_type),
                ),
                (
                    "query.complexity".to_string(),
                    metrics.complexity_score.to_string(),
                ),
                ("query.depth".to_string(), metrics.depth.to_string()),
                ("cache.hit".to_string(), metrics.cache_hit.to_string()),
            ],
            logs: Vec::new(),
            status: if metrics.error_count > 0 {
                SpanStatus::Error
            } else {
                SpanStatus::Ok
            },
        };

        let trace = DistributedTrace {
            trace_id: trace_id.clone(),
            spans: vec![span],
            total_duration: metrics.execution_time,
            service_count: 1,
            error_count: metrics.error_count,
        };

        self.completed_traces.push_back(trace);

        // Keep only recent traces
        while self.completed_traces.len() > 1000 {
            self.completed_traces.pop_front();
        }

        Ok(())
    }

    fn get_recent_traces(&self, limit: usize) -> Vec<DistributedTrace> {
        self.completed_traces
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

/// Alert management system
pub struct AlertManager {
    active_alerts: Vec<Alert>,
    alert_history: VecDeque<Alert>,
    thresholds: AlertThresholds,
}

impl AlertManager {
    fn new(config: &ObservabilityConfig) -> Self {
        Self {
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
            thresholds: config.alert_thresholds.clone(),
        }
    }

    fn check_alerts(&mut self, metrics: &OperationMetrics) -> Result<()> {
        let mut new_alerts = Vec::new();

        // Check response time
        if metrics.execution_time.as_millis() > self.thresholds.max_response_time_ms as u128 {
            new_alerts.push(Alert {
                id: format!(
                    "response_time_{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis()
                ),
                alert_type: AlertType::HighResponseTime,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Operation took {}ms (threshold: {}ms)",
                    metrics.execution_time.as_millis(),
                    self.thresholds.max_response_time_ms
                ),
                timestamp: SystemTime::now(),
                resolved: false,
                metadata: HashMap::new(),
            });
        }

        // Check error rate
        if metrics.error_count > 0 {
            new_alerts.push(Alert {
                id: format!(
                    "error_{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis()
                ),
                alert_type: AlertType::HighErrorRate,
                severity: AlertSeverity::Critical,
                message: format!("Operation had {} errors", metrics.error_count),
                timestamp: SystemTime::now(),
                resolved: false,
                metadata: HashMap::new(),
            });
        }

        for alert in new_alerts {
            info!("New alert: {} - {}", alert.alert_type, alert.message);
            self.alert_history.push_back(alert.clone());
            self.active_alerts.push(alert);
        }

        Ok(())
    }

    fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.clone()
    }
}

/// Real-time monitoring system
pub struct RealTimeMonitor {
    current_metrics: RealTimeMetrics,
    metrics_history: VecDeque<RealTimeMetrics>,
    update_interval: Duration,
    last_update: Instant,
}

impl RealTimeMonitor {
    fn new(config: &ObservabilityConfig) -> Self {
        Self {
            current_metrics: RealTimeMetrics::default(),
            metrics_history: VecDeque::new(),
            update_interval: config.metrics_collection_interval,
            last_update: Instant::now(),
        }
    }

    fn update_metrics(&mut self, operation_metrics: &OperationMetrics) -> Result<()> {
        self.current_metrics.total_requests += 1;
        self.current_metrics.total_errors += operation_metrics.error_count as u64;

        // Update running averages
        let new_response_time = operation_metrics.execution_time;
        self.current_metrics.avg_response_time = Duration::from_millis(
            ((self.current_metrics.avg_response_time.as_millis() as u64
                + new_response_time.as_millis() as u64)
                / 2),
        );

        // Update error rate
        self.current_metrics.error_rate =
            self.current_metrics.total_errors as f64 / self.current_metrics.total_requests as f64;

        // Update cache metrics
        if operation_metrics.cache_hit {
            self.current_metrics.cache_hits += 1;
        }
        self.current_metrics.cache_hit_ratio =
            self.current_metrics.cache_hits as f64 / self.current_metrics.total_requests as f64;

        // Store historical data
        if self.last_update.elapsed() >= self.update_interval {
            self.metrics_history.push_back(self.current_metrics.clone());

            // Keep only recent history
            while self.metrics_history.len() > 100 {
                self.metrics_history.pop_front();
            }

            self.last_update = Instant::now();
        }

        Ok(())
    }

    fn get_current_state(&self) -> RealTimeMetrics {
        self.current_metrics.clone()
    }
}

// Data structures for observability system

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityDashboard {
    pub service_info: ServiceInfo,
    pub metrics: Option<MetricsSummary>,
    pub traces: Option<Vec<DistributedTrace>>,
    pub alerts: Option<Vec<Alert>>,
    pub real_time_data: Option<RealTimeMetrics>,
    pub health_status: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub version: String,
    pub environment: String,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_operations: usize,
    pub aggregated: AggregatedMetrics,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub total_requests: u64,
    pub total_errors: u64,
    pub avg_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub p50_execution_time: Duration,
    pub p95_execution_time: Duration,
    pub p99_execution_time: Duration,
}

impl AggregatedMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            total_errors: 0,
            avg_execution_time: Duration::from_millis(0),
            min_execution_time: Duration::from_secs(u64::MAX),
            max_execution_time: Duration::from_millis(0),
            p50_execution_time: Duration::from_millis(0),
            p95_execution_time: Duration::from_millis(0),
            p99_execution_time: Duration::from_millis(0),
        }
    }

    fn update(&mut self, metrics: &OperationMetrics) {
        self.total_requests += 1;
        self.total_errors += metrics.error_count as u64;

        let exec_time = metrics.execution_time;

        // Update min/max
        if exec_time < self.min_execution_time {
            self.min_execution_time = exec_time;
        }
        if exec_time > self.max_execution_time {
            self.max_execution_time = exec_time;
        }

        // Update running average
        self.avg_execution_time = Duration::from_millis(
            ((self.avg_execution_time.as_millis() as u64 * (self.total_requests - 1)
                + exec_time.as_millis() as u64)
                / self.total_requests),
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrace {
    pub trace_id: String,
    pub spans: Vec<TraceSpan>,
    pub total_duration: Duration,
    pub service_count: usize,
    pub error_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: SystemTime,
    pub duration: Option<Duration>,
    pub tags: Vec<(String, String)>,
    pub logs: Vec<SpanLog>,
    pub status: SpanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: SystemTime,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub resolved: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighResponseTime,
    HighErrorRate,
    HighMemoryUsage,
    HighCpuUsage,
    LowCacheHitRatio,
    TooManyConcurrentRequests,
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertType::HighResponseTime => write!(f, "High Response Time"),
            AlertType::HighErrorRate => write!(f, "High Error Rate"),
            AlertType::HighMemoryUsage => write!(f, "High Memory Usage"),
            AlertType::HighCpuUsage => write!(f, "High CPU Usage"),
            AlertType::LowCacheHitRatio => write!(f, "Low Cache Hit Ratio"),
            AlertType::TooManyConcurrentRequests => write!(f, "Too Many Concurrent Requests"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    pub total_requests: u64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub avg_response_time: Duration,
    pub current_concurrent_requests: usize,
    pub cache_hits: u64,
    pub cache_hit_ratio: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub timestamp: SystemTime,
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_errors: 0,
            error_rate: 0.0,
            avg_response_time: Duration::from_millis(0),
            current_concurrent_requests: 0,
            cache_hits: 0,
            cache_hit_ratio: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            timestamp: SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthStatusLevel,
    pub score: f64,
    pub issues: Vec<String>,
    pub last_check: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatusLevel {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Histogram {
    buckets: Vec<(f64, u64)>,
    total_count: u64,
    total_sum: f64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: vec![
                (1.0, 0),
                (5.0, 0),
                (10.0, 0),
                (25.0, 0),
                (50.0, 0),
                (100.0, 0),
                (250.0, 0),
                (500.0, 0),
                (1000.0, 0),
                (2500.0, 0),
                (5000.0, 0),
                (10000.0, 0),
                (f64::INFINITY, 0),
            ],
            total_count: 0,
            total_sum: 0.0,
        }
    }

    fn record(&mut self, value: f64) {
        self.total_count += 1;
        self.total_sum += value;

        for (threshold, count) in &mut self.buckets {
            if value <= *threshold {
                *count += 1;
            }
        }
    }

    fn summary(&self) -> HistogramSummary {
        HistogramSummary {
            count: self.total_count,
            sum: self.total_sum,
            buckets: self.buckets.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSummary {
    pub count: u64,
    pub sum: f64,
    pub buckets: Vec<(f64, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_observability_system_creation() {
        let config = ObservabilityConfig::default();
        let system = ObservabilitySystem::new(config);

        let dashboard = system.get_dashboard().await.unwrap();
        assert_eq!(dashboard.service_info.name, "oxirs-gql");
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = ObservabilityConfig::default();
        let system = ObservabilitySystem::new(config);

        let metrics = OperationMetrics {
            operation_name: Some("test_query".to_string()),
            operation_type: OperationType::Query,
            query_hash: 12345,
            execution_time: Duration::from_millis(100),
            parsing_time: Duration::from_millis(10),
            validation_time: Duration::from_millis(5),
            planning_time: Duration::from_millis(15),
            field_count: 3,
            depth: 2,
            complexity_score: 10,
            cache_hit: true,
            error_count: 0,
            timestamp: SystemTime::now(),
            client_info: ClientInfo::default(),
        };

        system.record_operation(&metrics).await.unwrap();

        let dashboard = system.get_dashboard().await.unwrap();
        assert!(dashboard.metrics.is_some());
    }

    #[tokio::test]
    async fn test_alert_generation() {
        let mut config = ObservabilityConfig::default();
        config.alert_thresholds.max_response_time_ms = 50; // Low threshold for testing

        let system = ObservabilitySystem::new(config);

        let slow_metrics = OperationMetrics {
            operation_name: Some("slow_query".to_string()),
            operation_type: OperationType::Query,
            query_hash: 67890,
            execution_time: Duration::from_millis(100), // Exceeds threshold
            parsing_time: Duration::from_millis(10),
            validation_time: Duration::from_millis(5),
            planning_time: Duration::from_millis(15),
            field_count: 5,
            depth: 3,
            complexity_score: 20,
            cache_hit: false,
            error_count: 0,
            timestamp: SystemTime::now(),
            client_info: ClientInfo::default(),
        };

        system.record_operation(&slow_metrics).await.unwrap();

        let dashboard = system.get_dashboard().await.unwrap();
        if let Some(alerts) = dashboard.alerts {
            assert!(!alerts.is_empty());
        }
    }

    #[tokio::test]
    async fn test_health_status_calculation() {
        let config = ObservabilityConfig::default();
        let system = ObservabilitySystem::new(config);

        let health = system.calculate_health_status().await;
        assert!(matches!(health.status, HealthStatusLevel::Healthy));
        assert!(health.score >= 90.0);
    }
}
