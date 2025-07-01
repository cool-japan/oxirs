//! # Monitoring and Observability
//!
//! Comprehensive monitoring, metrics collection, and observability features
//! for the OxiRS streaming platform.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use crate::backend_optimizer::ResourceUsage;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub metrics_interval: Duration,
    pub health_check_interval: Duration,
    pub enable_profiling: bool,
    pub prometheus_endpoint: Option<String>,
    pub jaeger_endpoint: Option<String>,
    pub log_level: String,
}

/// Metrics collector for streaming operations
pub struct MetricsCollector {
    config: MonitoringConfig,
    metrics: Arc<RwLock<StreamingMetrics>>,
    health_checker: Arc<HealthChecker>,
    profiler: Option<Profiler>,
}

/// Comprehensive streaming metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    // Producer metrics
    pub producer_events_published: u64,
    pub producer_events_failed: u64,
    pub producer_bytes_sent: u64,
    pub producer_batches_sent: u64,
    pub producer_average_latency_ms: f64,
    pub producer_throughput_eps: f64,

    // Consumer metrics
    pub consumer_events_consumed: u64,
    pub consumer_events_processed: u64,
    pub consumer_events_filtered: u64,
    pub consumer_events_failed: u64,
    pub consumer_bytes_received: u64,
    pub consumer_batches_received: u64,
    pub consumer_average_processing_time_ms: f64,
    pub consumer_throughput_eps: f64,
    pub consumer_lag_ms: Option<f64>,

    // System metrics
    pub system_memory_usage_bytes: u64,
    pub system_cpu_usage_percent: f64,
    pub system_network_bytes_in: u64,
    pub system_network_bytes_out: u64,
    pub system_gc_collections: u64,
    pub system_gc_time_ms: u64,

    // Backend metrics
    pub backend_connections_active: u32,
    pub backend_connections_idle: u32,
    pub backend_connection_errors: u64,
    pub backend_circuit_breaker_trips: u64,
    pub backend_retry_attempts: u64,

    // Stream processing metrics
    pub window_operations_count: u64,
    pub aggregation_operations_count: u64,
    pub pattern_matches_found: u64,
    pub state_store_operations: u64,
    pub subscriptions_active: u32,

    // Quality metrics
    pub message_loss_rate: f64,
    pub duplicate_rate: f64,
    pub out_of_order_rate: f64,
    pub error_rate: f64,
    pub success_rate: f64,
    pub availability: f64,

    // Timestamps
    pub last_updated: DateTime<Utc>,
    pub collection_start_time: DateTime<Utc>,
}

/// Health checker for system health monitoring
pub struct HealthChecker {
    config: MonitoringConfig,
    health_status: Arc<RwLock<SystemHealth>>,
    component_checkers: Vec<Box<dyn ComponentHealthChecker>>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub component_health: HashMap<String, ComponentHealth>,
    pub last_check: DateTime<Utc>,
    pub uptime: Duration,
    pub alerts: Vec<HealthAlert>,
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Warning => write!(f, "warning"),
            HealthStatus::Critical => write!(f, "critical"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub message: String,
    pub last_check: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
    pub dependencies: Vec<String>,
}

/// Health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub id: String,
    pub component: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Profiler for performance analysis
pub struct Profiler {
    enabled: bool,
    traces: Arc<RwLock<Vec<PerformanceTrace>>>,
    sampling_rate: f64,
}

/// Performance trace data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrace {
    pub operation: String,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub metadata: HashMap<String, String>,
    pub call_stack: Vec<String>,
}

/// Component health checker trait
pub trait ComponentHealthChecker: Send + Sync {
    fn component_name(&self) -> &str;
    fn check_health(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ComponentHealth> + Send + '_>>;
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MonitoringConfig) -> Self {
        let health_checker = Arc::new(HealthChecker::new(config.clone()));
        let profiler = if config.enable_profiling {
            Some(Profiler::new(0.1)) // 10% sampling rate
        } else {
            None
        };

        Self {
            config,
            metrics: Arc::new(RwLock::new(StreamingMetrics::default())),
            health_checker,
            profiler,
        }
    }

    /// Start metrics collection
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting metrics collection with interval: {:?}",
            self.config.metrics_interval
        );

        // Start metrics collection task
        self.start_metrics_collection().await;

        // Start health checking task
        self.start_health_checking().await;

        // Start system metrics collection
        self.start_system_metrics_collection().await;

        Ok(())
    }

    /// Update producer metrics
    pub async fn update_producer_metrics(&self, metrics: ProducerMetricsUpdate) {
        let mut current_metrics = self.metrics.write().await;

        current_metrics.producer_events_published += metrics.events_published;
        current_metrics.producer_events_failed += metrics.events_failed;
        current_metrics.producer_bytes_sent += metrics.bytes_sent;
        current_metrics.producer_batches_sent += metrics.batches_sent;

        if metrics.latency_ms > 0.0 {
            current_metrics.producer_average_latency_ms =
                (current_metrics.producer_average_latency_ms + metrics.latency_ms) / 2.0;
        }

        current_metrics.producer_throughput_eps = metrics.throughput_eps;
        current_metrics.last_updated = Utc::now();
    }

    /// Update consumer metrics
    pub async fn update_consumer_metrics(&self, metrics: ConsumerMetricsUpdate) {
        let mut current_metrics = self.metrics.write().await;

        current_metrics.consumer_events_consumed += metrics.events_consumed;
        current_metrics.consumer_events_processed += metrics.events_processed;
        current_metrics.consumer_events_filtered += metrics.events_filtered;
        current_metrics.consumer_events_failed += metrics.events_failed;
        current_metrics.consumer_bytes_received += metrics.bytes_received;
        current_metrics.consumer_batches_received += metrics.batches_received;
        
        // Enhanced health assessment based on metrics trends
        let _health = self.health_checker.get_health().await;

        if metrics.processing_time_ms > 0.0 {
            current_metrics.consumer_average_processing_time_ms =
                (current_metrics.consumer_average_processing_time_ms + metrics.processing_time_ms)
                    / 2.0;
        }

        current_metrics.consumer_throughput_eps = metrics.throughput_eps;
        current_metrics.consumer_lag_ms = metrics.lag_ms;
        current_metrics.last_updated = Utc::now();
    }

    /// Update backend metrics
    pub async fn update_backend_metrics(&self, metrics: BackendMetricsUpdate) {
        let mut current_metrics = self.metrics.write().await;

        current_metrics.backend_connections_active = metrics.connections_active;
        current_metrics.backend_connections_idle = metrics.connections_idle;
        current_metrics.backend_connection_errors += metrics.connection_errors;
        current_metrics.backend_circuit_breaker_trips += metrics.circuit_breaker_trips;
        current_metrics.backend_retry_attempts += metrics.retry_attempts;
        current_metrics.last_updated = Utc::now();
    }

    /// Get current metrics snapshot
    pub async fn get_metrics(&self) -> StreamingMetrics {
        self.metrics.read().await.clone()
    }

    /// Start metrics collection task
    async fn start_metrics_collection(&self) {
        let metrics = self.metrics.clone();
        let interval = self.config.metrics_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Collect system metrics
                let mut current_metrics = metrics.write().await;

                // Calculate rates and derived metrics
                let elapsed = current_metrics
                    .last_updated
                    .signed_duration_since(current_metrics.collection_start_time)
                    .num_seconds() as f64;

                if elapsed > 0.0 {
                    // Calculate error rate
                    let total_events = current_metrics.producer_events_published
                        + current_metrics.producer_events_failed;
                    if total_events > 0 {
                        current_metrics.error_rate =
                            current_metrics.producer_events_failed as f64 / total_events as f64;
                        current_metrics.success_rate = 1.0 - current_metrics.error_rate;
                    }

                    // Calculate availability (simplified)
                    current_metrics.availability = if current_metrics.error_rate < 0.01 {
                        99.9 + (1.0 - current_metrics.error_rate) * 0.1
                    } else {
                        100.0 - (current_metrics.error_rate * 100.0)
                    };
                }

                debug!(
                    "Updated metrics: throughput={:.2} eps, error_rate={:.4}",
                    current_metrics.producer_throughput_eps, current_metrics.error_rate
                );
            }
        });
    }

    /// Start health checking task
    async fn start_health_checking(&self) {
        let health_checker = self.health_checker.clone();
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = health_checker.check_all_components().await {
                    error!("Health check failed: {}", e);
                }
            }
        });
    }

    /// Start system metrics collection
    async fn start_system_metrics_collection(&self) {
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                let mut current_metrics = metrics.write().await;

                // Collect system metrics (simplified - would use actual system APIs)
                current_metrics.system_memory_usage_bytes = Self::get_memory_usage();
                current_metrics.system_cpu_usage_percent = Self::get_cpu_usage();
                current_metrics.system_network_bytes_in += Self::get_network_bytes_in();
                current_metrics.system_network_bytes_out += Self::get_network_bytes_out();
            }
        });
    }

    /// Get system memory usage (simplified)
    fn get_memory_usage() -> u64 {
        // Would use actual system APIs like sysinfo crate
        1024 * 1024 * 512 // 512MB placeholder
    }

    /// Get CPU usage (simplified)
    fn get_cpu_usage() -> f64 {
        // Would use actual system APIs
        25.0 // 25% placeholder
    }

    /// Get network bytes in (simplified)
    fn get_network_bytes_in() -> u64 {
        // Would use actual network statistics
        1024 // 1KB placeholder
    }

    /// Get network bytes out (simplified)
    fn get_network_bytes_out() -> u64 {
        // Would use actual network statistics
        1024 // 1KB placeholder
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.metrics.read().await;

        format!(
            r#"# HELP oxirs_producer_events_published_total Total number of events published by producers
# TYPE oxirs_producer_events_published_total counter
oxirs_producer_events_published_total {}

# HELP oxirs_producer_events_failed_total Total number of failed events in producers
# TYPE oxirs_producer_events_failed_total counter
oxirs_producer_events_failed_total {}

# HELP oxirs_producer_throughput_eps Current producer throughput in events per second
# TYPE oxirs_producer_throughput_eps gauge
oxirs_producer_throughput_eps {}

# HELP oxirs_consumer_events_consumed_total Total number of events consumed
# TYPE oxirs_consumer_events_consumed_total counter
oxirs_consumer_events_consumed_total {}

# HELP oxirs_consumer_throughput_eps Current consumer throughput in events per second
# TYPE oxirs_consumer_throughput_eps gauge
oxirs_consumer_throughput_eps {}

# HELP oxirs_error_rate Current error rate
# TYPE oxirs_error_rate gauge
oxirs_error_rate {}

# HELP oxirs_availability Current system availability percentage
# TYPE oxirs_availability gauge
oxirs_availability {}
"#,
            metrics.producer_events_published,
            metrics.producer_events_failed,
            metrics.producer_throughput_eps,
            metrics.consumer_events_consumed,
            metrics.consumer_throughput_eps,
            metrics.error_rate,
            metrics.availability
        )
    }

    /// Get health status
    pub async fn get_health(&self) -> SystemHealth {
        self.health_checker.get_health().await
    }
}

impl HealthChecker {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            health_status: Arc::new(RwLock::new(SystemHealth::default())),
            component_checkers: Vec::new(),
        }
    }

    /// Add a component health checker
    pub fn add_component_checker(&mut self, checker: Box<dyn ComponentHealthChecker>) {
        self.component_checkers.push(checker);
    }

    /// Check health of all components
    pub async fn check_all_components(&self) -> Result<()> {
        let mut component_health = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;
        let mut alerts = Vec::new();

        for checker in &self.component_checkers {
            let health = checker.check_health().await;
            let component_name = checker.component_name().to_string();

            match health.status {
                HealthStatus::Warning => {
                    if overall_status == HealthStatus::Healthy {
                        overall_status = HealthStatus::Warning;
                    }
                    alerts.push(HealthAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        component: component_name.clone(),
                        severity: AlertSeverity::Warning,
                        message: health.message.clone(),
                        timestamp: Utc::now(),
                        resolved: false,
                    });
                }
                HealthStatus::Critical => {
                    overall_status = HealthStatus::Critical;
                    alerts.push(HealthAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        component: component_name.clone(),
                        severity: AlertSeverity::Critical,
                        message: health.message.clone(),
                        timestamp: Utc::now(),
                        resolved: false,
                    });
                }
                _ => {}
            }

            component_health.insert(component_name, health);
        }

        let mut health_status = self.health_status.write().await;
        health_status.overall_status = overall_status;
        health_status.component_health = component_health;
        health_status.last_check = Utc::now();
        health_status.alerts.extend(alerts);

        Ok(())
    }

    /// Get current health status
    pub async fn get_health(&self) -> SystemHealth {
        self.health_status.read().await.clone()
    }

    /// Assess system health based on current metrics trends
    async fn assess_system_health(&self, metrics: &StreamingMetrics) {
        let mut health_alerts = Vec::new();
        let now = Utc::now();
        let mut alert_id = 1;
        
        // Producer health assessment
        if metrics.producer_events_failed > 0 {
            let failure_rate = metrics.producer_events_failed as f64 / 
                (metrics.producer_events_published + metrics.producer_events_failed) as f64;
            if failure_rate > 0.05 {
                health_alerts.push(HealthAlert {
                    id: format!("producer_failure_{}", alert_id),
                    component: "producer".to_string(),
                    severity: AlertSeverity::Warning,
                    message: format!("High producer failure rate: {:.2}%", failure_rate * 100.0),
                    timestamp: now,
                    resolved: false,
                });
                alert_id += 1;
            }
        }
        
        // Consumer health assessment  
        if metrics.consumer_events_failed > 0 {
            let failure_rate = metrics.consumer_events_failed as f64 / 
                metrics.consumer_events_consumed as f64;
            if failure_rate > 0.05 {
                health_alerts.push(HealthAlert {
                    id: format!("consumer_failure_{}", alert_id),
                    component: "consumer".to_string(),
                    severity: AlertSeverity::Warning,
                    message: format!("High consumer failure rate: {:.2}%", failure_rate * 100.0),
                    timestamp: now,
                    resolved: false,
                });
                alert_id += 1;
            }
        }
        
        // Performance health assessment
        if metrics.producer_average_latency_ms > 1000.0 {
            health_alerts.push(HealthAlert {
                id: format!("producer_latency_{}", alert_id),
                component: "producer".to_string(),
                severity: AlertSeverity::Critical,
                message: format!("High producer latency: {:.2}ms", metrics.producer_average_latency_ms),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        }
        
        if metrics.consumer_average_processing_time_ms > 500.0 {
            health_alerts.push(HealthAlert {
                id: format!("consumer_processing_{}", alert_id),
                component: "consumer".to_string(),
                severity: AlertSeverity::Critical,
                message: format!("High consumer processing time: {:.2}ms", metrics.consumer_average_processing_time_ms),
                timestamp: now,
                resolved: false,
            });
        }
        
        // Update health status based on assessments
        let health_status = if health_alerts.is_empty() {
            HealthStatus::Healthy
        } else if health_alerts.len() <= 2 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        };
        
        if !health_alerts.is_empty() {
            warn!("System health alerts: {:?}", health_alerts);
        }
        
        // Update health status
        let system_health = SystemHealth {
            overall_status: health_status,
            component_health: HashMap::new(),
            last_check: now,
            uptime: Duration::from_secs(0), // Default uptime
            alerts: health_alerts,
        };
        
        *self.health_status.write().await = system_health;
    }
}

impl Profiler {
    fn new(sampling_rate: f64) -> Self {
        Self {
            enabled: true,
            traces: Arc::new(RwLock::new(Vec::new())),
            sampling_rate,
        }
    }

    /// Start a performance trace
    pub async fn start_trace(&self, operation: String) -> Option<TraceHandle> {
        if !self.enabled || fastrand::f64() > self.sampling_rate {
            return None;
        }

        Some(TraceHandle {
            operation,
            start_time: Instant::now(),
            timestamp: Utc::now(),
            traces: self.traces.clone(),
        })
    }
}

/// Handle for performance tracing
pub struct TraceHandle {
    operation: String,
    start_time: Instant,
    timestamp: DateTime<Utc>,
    traces: Arc<RwLock<Vec<PerformanceTrace>>>,
}

impl Drop for TraceHandle {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let trace = PerformanceTrace {
            operation: self.operation.clone(),
            start_time: self.timestamp,
            duration,
            metadata: HashMap::new(),
            call_stack: Vec::new(), // Would be populated with actual call stack
        };

        let traces = self.traces.clone();
        tokio::spawn(async move {
            traces.write().await.push(trace);
        });
    }
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            overall_status: HealthStatus::Unknown,
            component_health: HashMap::new(),
            last_check: Utc::now(),
            uptime: Duration::from_secs(0),
            alerts: Vec::new(),
        }
    }
}

/// Metrics update structures
#[derive(Debug, Clone)]
pub struct ProducerMetricsUpdate {
    pub events_published: u64,
    pub events_failed: u64,
    pub bytes_sent: u64,
    pub batches_sent: u64,
    pub latency_ms: f64,
    pub throughput_eps: f64,
}

#[derive(Debug, Clone)]
pub struct ConsumerMetricsUpdate {
    pub events_consumed: u64,
    pub events_processed: u64,
    pub events_filtered: u64,
    pub events_failed: u64,
    pub bytes_received: u64,
    pub batches_received: u64,
    pub processing_time_ms: f64,
    pub throughput_eps: f64,
    pub lag_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BackendMetricsUpdate {
    pub connections_active: u32,
    pub connections_idle: u32,
    pub connection_errors: u64,
    pub circuit_breaker_trips: u64,
    pub retry_attempts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: true,
            metrics_interval: Duration::from_millis(100),
            health_check_interval: Duration::from_millis(100),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let collector = MetricsCollector::new(config);

        // Update some metrics
        collector
            .update_producer_metrics(ProducerMetricsUpdate {
                events_published: 100,
                events_failed: 5,
                bytes_sent: 1024,
                batches_sent: 10,
                latency_ms: 5.0,
                throughput_eps: 1000.0,
            })
            .await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.producer_events_published, 100);
        assert_eq!(metrics.producer_events_failed, 5);
        assert_eq!(metrics.producer_throughput_eps, 1000.0);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let collector = MetricsCollector::new(config);

        collector
            .update_producer_metrics(ProducerMetricsUpdate {
                events_published: 500,
                events_failed: 10,
                bytes_sent: 2048,
                batches_sent: 50,
                latency_ms: 3.0,
                throughput_eps: 2000.0,
            })
            .await;

        let prometheus_output = collector.export_prometheus().await;
        assert!(prometheus_output.contains("oxirs_producer_events_published_total 500"));
        assert!(prometheus_output.contains("oxirs_producer_events_failed_total 10"));
    }

    #[tokio::test]
    async fn test_health_checking() {
        let config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let mut health_checker = HealthChecker::new(config);

        // Add a mock component checker
        struct MockChecker;

        impl ComponentHealthChecker for MockChecker {
            fn component_name(&self) -> &str {
                "mock_component"
            }

            fn check_health(
                &self,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ComponentHealth> + Send + '_>>
            {
                Box::pin(async move {
                    ComponentHealth {
                        status: HealthStatus::Healthy,
                        message: "Component is healthy".to_string(),
                        last_check: Utc::now(),
                        metrics: HashMap::new(),
                        dependencies: vec!["database".to_string()],
                    }
                })
            }
        }

        health_checker.add_component_checker(Box::new(MockChecker));
        health_checker.check_all_components().await.unwrap();

        let health = health_checker.get_health().await;
        assert_eq!(health.overall_status, HealthStatus::Healthy);
        assert!(health.component_health.contains_key("mock_component"));
    }

    #[tokio::test]
    async fn test_consumer_metrics_update() {
        let config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let collector = MetricsCollector::new(config);

        collector
            .update_consumer_metrics(ConsumerMetricsUpdate {
                events_consumed: 1000,
                events_processed: 950,
                events_filtered: 50,
                events_failed: 10,
                bytes_received: 4096,
                batches_received: 100,
                processing_time_ms: 2.5,
                throughput_eps: 1500.0,
                lag_ms: Some(100.0),
            })
            .await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.consumer_events_consumed, 1000);
        assert_eq!(metrics.consumer_events_processed, 950);
        assert_eq!(metrics.consumer_throughput_eps, 1500.0);
        assert_eq!(metrics.consumer_lag_ms, Some(100.0));
    }

    #[tokio::test]
    async fn test_backend_metrics_update() {
        let config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let collector = MetricsCollector::new(config);

        collector
            .update_backend_metrics(BackendMetricsUpdate {
                connections_active: 5,
                connections_idle: 3,
                connection_errors: 2,
                circuit_breaker_trips: 1,
                retry_attempts: 5,
            })
            .await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.backend_connections_active, 5);
        assert_eq!(metrics.backend_connections_idle, 3);
        assert_eq!(metrics.backend_connection_errors, 2);
    }

    #[test]
    fn test_health_status_serialization() {
        let health = SystemHealth {
            overall_status: HealthStatus::Warning,
            component_health: {
                let mut health_map = HashMap::new();
                health_map.insert(
                    "database".to_string(),
                    ComponentHealth {
                        status: HealthStatus::Warning,
                        message: "High latency detected".to_string(),
                        last_check: Utc::now(),
                        metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("latency_ms".to_string(), 150.0);
                            metrics
                        },
                        dependencies: vec!["network".to_string()],
                    },
                );
                health_map
            },
            last_check: Utc::now(),
            uptime: Duration::from_secs(3600),
            alerts: vec![HealthAlert {
                id: "alert-1".to_string(),
                component: "database".to_string(),
                severity: AlertSeverity::Warning,
                message: "High latency detected".to_string(),
                timestamp: Utc::now(),
                resolved: false,
            }],
        };

        let serialized = serde_json::to_string(&health).unwrap();
        let deserialized: SystemHealth = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.overall_status, HealthStatus::Warning);
        assert_eq!(deserialized.component_health.len(), 1);
        assert_eq!(deserialized.alerts.len(), 1);
    }

    #[tokio::test]
    async fn test_profiler() {
        let profiler = Profiler::new(1.0); // 100% sampling for testing

        {
            let _trace = profiler.start_trace("test_operation".to_string()).await;
            // Simulate some work
            tokio::time::sleep(Duration::from_millis(10)).await;
        } // TraceHandle dropped here, trace should be recorded

        // Give some time for async trace recording
        tokio::time::sleep(Duration::from_millis(50)).await;

        let traces = profiler.traces.read().await;
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].operation, "test_operation");
        assert!(traces[0].duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_metrics_update_structures() {
        let producer_update = ProducerMetricsUpdate {
            events_published: 100,
            events_failed: 2,
            bytes_sent: 1024,
            batches_sent: 10,
            latency_ms: 5.5,
            throughput_eps: 200.0,
        };

        assert_eq!(producer_update.events_published, 100);
        assert_eq!(producer_update.latency_ms, 5.5);

        let consumer_update = ConsumerMetricsUpdate {
            events_consumed: 95,
            events_processed: 90,
            events_filtered: 5,
            events_failed: 1,
            bytes_received: 950,
            batches_received: 9,
            processing_time_ms: 2.0,
            throughput_eps: 190.0,
            lag_ms: Some(50.0),
        };

        assert_eq!(consumer_update.events_consumed, 95);
        assert_eq!(consumer_update.lag_ms, Some(50.0));

        let backend_update = BackendMetricsUpdate {
            connections_active: 3,
            connections_idle: 2,
            connection_errors: 1,
            circuit_breaker_trips: 0,
            retry_attempts: 2,
        };

        assert_eq!(backend_update.connections_active, 3);
        assert_eq!(backend_update.retry_attempts, 2);
    }
}
