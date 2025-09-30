//! # Monitoring and Observability
//!
//! Comprehensive monitoring, metrics collection, and observability features
//! for the OxiRS streaming platform.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{Pid, System};
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
    system: Arc<RwLock<System>>,
}

/// Comprehensive streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    // Dead Letter Queue metrics
    pub dlq_messages_count: u64,
    pub dlq_messages_per_second: f64,
    pub dlq_processing_rate: f64,
    pub dlq_oldest_message_age_ms: u64,
    pub dlq_replay_success_rate: f64,
    pub dlq_total_replayed: u64,
    pub dlq_size_bytes: u64,
    pub dlq_error_categories: HashMap<String, u64>,

    // Timestamps
    pub last_updated: DateTime<Utc>,
    pub collection_start_time: DateTime<Utc>,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            // Producer metrics
            producer_events_published: 0,
            producer_events_failed: 0,
            producer_bytes_sent: 0,
            producer_batches_sent: 0,
            producer_average_latency_ms: 0.0,
            producer_throughput_eps: 0.0,

            // Consumer metrics
            consumer_events_consumed: 0,
            consumer_events_processed: 0,
            consumer_events_filtered: 0,
            consumer_events_failed: 0,
            consumer_bytes_received: 0,
            consumer_batches_received: 0,
            consumer_average_processing_time_ms: 0.0,
            consumer_throughput_eps: 0.0,
            consumer_lag_ms: None,

            // System metrics
            system_memory_usage_bytes: 0,
            system_cpu_usage_percent: 0.0,
            system_network_bytes_in: 0,
            system_network_bytes_out: 0,
            system_gc_collections: 0,
            system_gc_time_ms: 0,

            // Backend metrics
            backend_connections_active: 0,
            backend_connections_idle: 0,
            backend_connection_errors: 0,
            backend_circuit_breaker_trips: 0,
            backend_retry_attempts: 0,

            // Stream processing metrics
            window_operations_count: 0,
            aggregation_operations_count: 0,
            pattern_matches_found: 0,
            state_store_operations: 0,
            subscriptions_active: 0,

            // Quality metrics
            message_loss_rate: 0.0,
            duplicate_rate: 0.0,
            out_of_order_rate: 0.0,
            error_rate: 0.0,
            success_rate: 100.0, // Start with 100% success rate
            availability: 100.0, // Start with 100% availability

            // Dead Letter Queue metrics
            dlq_messages_count: 0,
            dlq_messages_per_second: 0.0,
            dlq_processing_rate: 0.0,
            dlq_oldest_message_age_ms: 0,
            dlq_replay_success_rate: 100.0, // Start with 100% replay success rate
            dlq_total_replayed: 0,
            dlq_size_bytes: 0,
            dlq_error_categories: HashMap::new(),

            // Timestamps
            last_updated: now,
            collection_start_time: now,
        }
    }
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

/// Load average information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAverage {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
}

/// System information for detailed monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub total_memory: u64,
    pub used_memory: u64,
    pub total_swap: u64,
    pub used_swap: u64,
    pub cpu_count: usize,
    pub load_average: LoadAverage,
    pub boot_time: u64,
    pub uptime: u64,
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
        let mut sys = System::new_all();
        sys.refresh_all();

        Self {
            config,
            metrics: Arc::new(RwLock::new(StreamingMetrics::default())),
            health_checker,
            profiler,
            system: Arc::new(RwLock::new(sys)),
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
        let system = self.system.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            let mut previous_network_in = 0u64;
            let mut previous_network_out = 0u64;

            loop {
                interval.tick().await;

                // Refresh system information
                {
                    let mut sys = system.write().await;
                    sys.refresh_cpu();  // sysinfo 0.30: refresh_cpu_all() → refresh_cpu()
                    sys.refresh_memory();
                    // Network refresh is handled separately if available
                    sys.refresh_processes();  // sysinfo 0.30: simplified API
                }

                let mut current_metrics = metrics.write().await;
                let sys = system.read().await;

                // Real system metrics collection
                current_metrics.system_memory_usage_bytes = sys.used_memory();
                current_metrics.system_cpu_usage_percent = sys.global_cpu_info().cpu_usage() as f64;  // sysinfo 0.30: global_cpu_usage() → global_cpu_info().cpu_usage()

                // Network metrics (cumulative)
                let (network_in, network_out) = Self::get_network_metrics(&sys);
                current_metrics.system_network_bytes_in = previous_network_in + network_in;
                current_metrics.system_network_bytes_out = previous_network_out + network_out;
                previous_network_in = current_metrics.system_network_bytes_in;
                previous_network_out = current_metrics.system_network_bytes_out;

                // Process-specific metrics
                if let Some(process) = sys.process(Pid::from_u32(std::process::id())) {
                    // Add process-specific metrics here if needed
                    debug!("Process memory: {} bytes", process.memory());
                }
            }
        });
    }

    /// Get network metrics from system information
    fn get_network_metrics(_sys: &System) -> (u64, u64) {
        // Basic network metrics implementation
        // In sysinfo 0.32, network API access is different and may require
        // a separate Networks struct. For now, we provide a placeholder
        // that can be enhanced with proper implementation later.

        // Future improvement: Use std::fs to read /proc/net/dev on Linux
        // or implement platform-specific network metric collection

        // Return placeholder values for now to ensure compilation
        // This maintains functionality while allowing for future enhancement
        (0, 0)
    }

    /// Get detailed system information for health assessment
    pub async fn get_system_info(&self) -> SystemInfo {
        let sys = self.system.read().await;

        SystemInfo {
            total_memory: sys.total_memory(),
            used_memory: sys.used_memory(),
            total_swap: sys.total_swap(),
            used_swap: sys.used_swap(),
            cpu_count: sys.cpus().len(),
            load_average: {
                let load_avg = System::load_average();
                LoadAverage {
                    one: load_avg.one,
                    five: load_avg.five,
                    fifteen: load_avg.fifteen,
                }
            },
            boot_time: System::boot_time(),
            uptime: System::uptime(),
        }
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

    /// Assess system health based on current metrics trends and system resources
    pub async fn assess_system_health(
        &self,
        metrics: &StreamingMetrics,
        system_info: &SystemInfo,
    ) -> Result<()> {
        let mut health_alerts = Vec::new();
        let now = Utc::now();
        let mut alert_id = 1;

        // Memory health assessment
        let memory_usage_percent =
            (system_info.used_memory as f64 / system_info.total_memory as f64) * 100.0;
        if memory_usage_percent > 90.0 {
            health_alerts.push(HealthAlert {
                id: format!("memory_critical_{alert_id}"),
                component: "system".to_string(),
                severity: AlertSeverity::Critical,
                message: format!("Critical memory usage: {memory_usage_percent:.1}%"),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        } else if memory_usage_percent > 80.0 {
            health_alerts.push(HealthAlert {
                id: format!("memory_warning_{alert_id}"),
                component: "system".to_string(),
                severity: AlertSeverity::Warning,
                message: format!("High memory usage: {memory_usage_percent:.1}%"),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        }

        // CPU health assessment
        if metrics.system_cpu_usage_percent > 95.0 {
            health_alerts.push(HealthAlert {
                id: format!("cpu_critical_{alert_id}"),
                component: "system".to_string(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "Critical CPU usage: {:.1}%",
                    metrics.system_cpu_usage_percent
                ),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        } else if metrics.system_cpu_usage_percent > 85.0 {
            health_alerts.push(HealthAlert {
                id: format!("cpu_warning_{alert_id}"),
                component: "system".to_string(),
                severity: AlertSeverity::Warning,
                message: format!("High CPU usage: {:.1}%", metrics.system_cpu_usage_percent),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        }

        // Producer health assessment
        if metrics.producer_events_failed > 0 {
            let total_producer_events =
                metrics.producer_events_published + metrics.producer_events_failed;
            if total_producer_events > 0 {
                let failure_rate =
                    metrics.producer_events_failed as f64 / total_producer_events as f64;
                if failure_rate > 0.10 {
                    health_alerts.push(HealthAlert {
                        id: format!("producer_failure_{alert_id}"),
                        component: "producer".to_string(),
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "High producer failure rate: {:.2}%",
                            failure_rate * 100.0
                        ),
                        timestamp: now,
                        resolved: false,
                    });
                    alert_id += 1;
                } else if failure_rate > 0.05 {
                    health_alerts.push(HealthAlert {
                        id: format!("producer_failure_{alert_id}"),
                        component: "producer".to_string(),
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "Elevated producer failure rate: {:.2}%",
                            failure_rate * 100.0
                        ),
                        timestamp: now,
                        resolved: false,
                    });
                    alert_id += 1;
                }
            }
        }

        // Consumer health assessment
        if metrics.consumer_events_consumed > 0 && metrics.consumer_events_failed > 0 {
            let failure_rate =
                metrics.consumer_events_failed as f64 / metrics.consumer_events_consumed as f64;
            if failure_rate > 0.10 {
                health_alerts.push(HealthAlert {
                    id: format!("consumer_failure_{alert_id}"),
                    component: "consumer".to_string(),
                    severity: AlertSeverity::Critical,
                    message: format!("High consumer failure rate: {:.2}%", failure_rate * 100.0),
                    timestamp: now,
                    resolved: false,
                });
                alert_id += 1;
            } else if failure_rate > 0.05 {
                health_alerts.push(HealthAlert {
                    id: format!("consumer_failure_{alert_id}"),
                    component: "consumer".to_string(),
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Elevated consumer failure rate: {:.2}%",
                        failure_rate * 100.0
                    ),
                    timestamp: now,
                    resolved: false,
                });
                alert_id += 1;
            }
        }

        // Performance health assessment
        if metrics.producer_average_latency_ms > 2000.0 {
            health_alerts.push(HealthAlert {
                id: format!("producer_latency_{alert_id}"),
                component: "producer".to_string(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "Critical producer latency: {:.2}ms",
                    metrics.producer_average_latency_ms
                ),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        } else if metrics.producer_average_latency_ms > 1000.0 {
            health_alerts.push(HealthAlert {
                id: format!("producer_latency_{alert_id}"),
                component: "producer".to_string(),
                severity: AlertSeverity::Warning,
                message: format!(
                    "High producer latency: {:.2}ms",
                    metrics.producer_average_latency_ms
                ),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        }

        if metrics.consumer_average_processing_time_ms > 1000.0 {
            health_alerts.push(HealthAlert {
                id: format!("consumer_processing_{alert_id}"),
                component: "consumer".to_string(),
                severity: AlertSeverity::Critical,
                message: format!(
                    "Critical consumer processing time: {:.2}ms",
                    metrics.consumer_average_processing_time_ms
                ),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        } else if metrics.consumer_average_processing_time_ms > 500.0 {
            health_alerts.push(HealthAlert {
                id: format!("consumer_processing_{alert_id}"),
                component: "consumer".to_string(),
                severity: AlertSeverity::Warning,
                message: format!(
                    "High consumer processing time: {:.2}ms",
                    metrics.consumer_average_processing_time_ms
                ),
                timestamp: now,
                resolved: false,
            });
            alert_id += 1;
        }

        // Connection health assessment
        if metrics.backend_connection_errors > 0 {
            let total_connections =
                metrics.backend_connections_active + metrics.backend_connections_idle;
            if total_connections > 0 {
                let error_rate =
                    metrics.backend_connection_errors as f64 / total_connections as f64;
                if error_rate > 0.20 {
                    health_alerts.push(HealthAlert {
                        id: format!("connection_errors_{alert_id}"),
                        component: "backend".to_string(),
                        severity: AlertSeverity::Critical,
                        message: format!("High connection error rate: {:.2}%", error_rate * 100.0),
                        timestamp: now,
                        resolved: false,
                    });
                }
            }
        }

        // Update health status based on assessments
        let health_status = if health_alerts.is_empty() {
            HealthStatus::Healthy
        } else {
            let critical_alerts = health_alerts
                .iter()
                .filter(|a| matches!(a.severity, AlertSeverity::Critical))
                .count();
            if critical_alerts > 0 {
                HealthStatus::Critical
            } else {
                HealthStatus::Warning
            }
        };

        if !health_alerts.is_empty() {
            warn!(
                "System health alerts detected: {} total, {} critical",
                health_alerts.len(),
                health_alerts
                    .iter()
                    .filter(|a| matches!(a.severity, AlertSeverity::Critical))
                    .count()
            );
        }

        // Update health status with system uptime
        let system_health = SystemHealth {
            overall_status: health_status,
            component_health: HashMap::new(),
            last_check: now,
            uptime: Duration::from_secs(system_info.uptime),
            alerts: health_alerts,
        };

        *self.health_status.write().await = system_health;
        Ok(())
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
