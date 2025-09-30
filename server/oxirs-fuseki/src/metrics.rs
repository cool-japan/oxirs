//! Comprehensive monitoring, metrics, and observability system

use crate::config::MonitoringConfig;
use crate::error::{FusekiError, FusekiResult};
use axum::{extract::State, response::IntoResponse, routing::get, Json, Router};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[cfg(feature = "metrics")]
use prometheus::{Registry, TextEncoder};

/// Metrics service for collecting and exposing application metrics
#[derive(Clone)]
pub struct MetricsService {
    config: MonitoringConfig,
    registry: Arc<RwLock<MetricsRegistry>>,
    start_time: Instant,
    #[cfg(feature = "metrics")]
    prometheus_registry: Registry,
}

/// Internal metrics registry
#[derive(Default)]
pub struct MetricsRegistry {
    counters: HashMap<String, u64>,
    gauges: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
    system_metrics: SystemMetrics,
}

/// System-level metrics
#[derive(Debug, Clone, Default, Serialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub memory_total_bytes: u64,
    pub disk_usage_bytes: u64,
    pub disk_total_bytes: u64,
    pub open_file_descriptors: u64,
    pub network_connections: u64,
}

/// Application metrics summary
#[derive(Debug, Serialize)]
pub struct MetricsSummary {
    pub uptime_seconds: u64,
    pub requests_total: u64,
    pub requests_per_second: f64,
    pub sparql_queries_total: u64,
    pub sparql_updates_total: u64,
    pub active_connections: u64,
    pub cache_hit_ratio: f64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub system: SystemMetrics,
    pub custom_metrics: HashMap<String, serde_json::Value>,
}

/// Health check status
#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub version: String,
    pub uptime: String,
    pub timestamp: String,
    pub checks: HashMap<String, CheckResult>,
}

/// Health check state
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual health check result
#[derive(Debug, Serialize)]
pub struct CheckResult {
    pub status: HealthState,
    pub message: String,
    pub duration_ms: u64,
    pub timestamp: String,
}

/// Request metrics middleware data
#[derive(Debug)]
pub struct RequestMetrics {
    pub method: String,
    pub path: String,
    pub status: u16,
    pub duration: Duration,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl MetricsService {
    /// Create a new metrics service
    pub fn new(config: MonitoringConfig) -> FusekiResult<Self> {
        let registry = Arc::new(RwLock::new(MetricsRegistry::default()));

        #[cfg(feature = "metrics")]
        let prometheus_registry = Registry::new();

        let service = Self {
            config,
            registry,
            start_time: Instant::now(),
            #[cfg(feature = "metrics")]
            prometheus_registry,
        };

        // Initialize core metrics
        service.initialize_metrics()?;

        // Start background tasks if enabled
        if service.config.metrics.enabled {
            service.start_background_tasks();
        }

        Ok(service)
    }

    /// Initialize core application metrics
    fn initialize_metrics(&self) -> FusekiResult<()> {
        // Register core counters
        describe_counter!("http_requests_total", "Total number of HTTP requests");
        describe_counter!("sparql_queries_total", "Total number of SPARQL queries");
        describe_counter!("sparql_updates_total", "Total number of SPARQL updates");
        describe_counter!(
            "authentication_attempts_total",
            "Total authentication attempts"
        );
        describe_counter!(
            "authentication_failures_total",
            "Total authentication failures"
        );
        describe_counter!("cache_hits_total", "Total cache hits");
        describe_counter!("cache_misses_total", "Total cache misses");
        describe_counter!("errors_total", "Total errors by type");

        // Register core gauges
        describe_gauge!("active_connections", "Number of active connections");
        describe_gauge!("active_sessions", "Number of active user sessions");
        describe_gauge!("memory_usage_bytes", "Memory usage in bytes");
        describe_gauge!("cpu_usage_percent", "CPU usage percentage");
        describe_gauge!("cache_size_bytes", "Cache size in bytes");
        describe_gauge!("database_connections", "Number of database connections");

        // Register core histograms
        describe_histogram!("http_request_duration_seconds", "HTTP request duration");
        describe_histogram!(
            "sparql_query_duration_seconds",
            "SPARQL query execution time"
        );
        describe_histogram!(
            "sparql_update_duration_seconds",
            "SPARQL update execution time"
        );
        describe_histogram!(
            "cache_operation_duration_seconds",
            "Cache operation duration"
        );
        describe_histogram!(
            "database_operation_duration_seconds",
            "Database operation duration"
        );

        info!("Metrics registry initialized with core metrics");
        Ok(())
    }

    /// Start background monitoring tasks
    fn start_background_tasks(&self) {
        let config = self.config.clone();

        // System metrics collection task
        if config.metrics.collect_system_metrics {
            let registry = Arc::clone(&self.registry);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));

                loop {
                    interval.tick().await;

                    if let Ok(system_metrics) = collect_system_metrics().await {
                        let mut registry = registry.write().await;
                        registry.system_metrics = system_metrics;

                        // Update Prometheus metrics
                        gauge!("memory_usage_bytes")
                            .set(registry.system_metrics.memory_usage_bytes as f64);
                        gauge!("cpu_usage_percent").set(registry.system_metrics.cpu_usage_percent);
                    }
                }
            });
        }

        // Health check task
        if config.health_checks.enabled {
            let _registry_clone = Arc::clone(&self.registry);
            let health_config = config.health_checks.clone();

            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_secs(health_config.interval_secs));

                loop {
                    interval.tick().await;

                    // Run health checks (implementation would go here)
                    debug!("Running health checks");
                }
            });
        }
    }

    /// Record HTTP request metrics
    pub async fn record_request(&self, metrics: RequestMetrics) {
        let mut registry = self.registry.write().await;

        // Update counters
        *registry
            .counters
            .entry("http_requests_total".to_string())
            .or_insert(0) += 1;

        let status_key = format!("http_requests_status_{}", metrics.status);
        *registry.counters.entry(status_key).or_insert(0) += 1;

        // Update histograms
        registry
            .histograms
            .entry("http_request_duration_seconds".to_string())
            .or_insert_with(Vec::new)
            .push(metrics.duration.as_secs_f64());

        // Update Prometheus metrics
        let status_str = metrics.status.to_string();
        counter!("http_requests_total", "method" => metrics.method.clone(), "status" => status_str)
            .increment(1);
        histogram!("http_request_duration_seconds", "method" => metrics.method.clone())
            .record(metrics.duration.as_secs_f64());

        debug!(
            method = %metrics.method,
            path = %metrics.path,
            status = metrics.status,
            duration_ms = metrics.duration.as_millis(),
            "HTTP request recorded"
        );
    }

    /// Record SPARQL query metrics
    pub async fn record_sparql_query(&self, duration: Duration, success: bool, query_type: &str) {
        let mut registry = self.registry.write().await;

        *registry
            .counters
            .entry("sparql_queries_total".to_string())
            .or_insert(0) += 1;

        if success {
            *registry
                .counters
                .entry("sparql_queries_success".to_string())
                .or_insert(0) += 1;
        } else {
            *registry
                .counters
                .entry("sparql_queries_failed".to_string())
                .or_insert(0) += 1;
        }

        registry
            .histograms
            .entry("sparql_query_duration_seconds".to_string())
            .or_insert_with(Vec::new)
            .push(duration.as_secs_f64());

        // Update Prometheus metrics
        let query_type_str = query_type.to_string();
        let success_str = success.to_string();
        counter!("sparql_queries_total", "type" => query_type_str.clone(), "success" => success_str)
            .increment(1);
        histogram!("sparql_query_duration_seconds", "type" => query_type_str)
            .record(duration.as_secs_f64());

        info!(
            query_type = query_type,
            duration_ms = duration.as_millis(),
            success = success,
            "SPARQL query recorded"
        );
    }

    /// Record SPARQL update metrics
    pub async fn record_sparql_update(
        &self,
        duration: Duration,
        success: bool,
        operation_type: &str,
    ) {
        let mut registry = self.registry.write().await;

        *registry
            .counters
            .entry("sparql_updates_total".to_string())
            .or_insert(0) += 1;

        if success {
            *registry
                .counters
                .entry("sparql_updates_success".to_string())
                .or_insert(0) += 1;
        } else {
            *registry
                .counters
                .entry("sparql_updates_failed".to_string())
                .or_insert(0) += 1;
        }

        registry
            .histograms
            .entry("sparql_update_duration_seconds".to_string())
            .or_insert_with(Vec::new)
            .push(duration.as_secs_f64());

        // Update Prometheus metrics
        let operation_type_str = operation_type.to_string();
        let success_str = success.to_string();
        counter!("sparql_updates_total", "operation" => operation_type_str.clone(), "success" => success_str).increment(1);
        histogram!("sparql_update_duration_seconds", "operation" => operation_type_str)
            .record(duration.as_secs_f64());

        info!(
            operation_type = operation_type,
            duration_ms = duration.as_millis(),
            success = success,
            "SPARQL update recorded"
        );
    }

    /// Record authentication metrics
    pub async fn record_authentication(&self, success: bool, method: &str) {
        let mut registry = self.registry.write().await;

        *registry
            .counters
            .entry("authentication_attempts_total".to_string())
            .or_insert(0) += 1;

        if success {
            *registry
                .counters
                .entry("authentication_success_total".to_string())
                .or_insert(0) += 1;
        } else {
            *registry
                .counters
                .entry("authentication_failures_total".to_string())
                .or_insert(0) += 1;
        }

        // Update Prometheus metrics
        let method_str = method.to_string();
        let success_str = success.to_string();
        counter!("authentication_attempts_total", "method" => method_str, "success" => success_str)
            .increment(1);

        debug!(
            method = method,
            success = success,
            "Authentication attempt recorded"
        );
    }

    /// Record cache metrics
    pub async fn record_cache_operation(&self, hit: bool, operation: &str, duration: Duration) {
        let mut registry = self.registry.write().await;

        if hit {
            *registry
                .counters
                .entry("cache_hits_total".to_string())
                .or_insert(0) += 1;
        } else {
            *registry
                .counters
                .entry("cache_misses_total".to_string())
                .or_insert(0) += 1;
        }

        registry
            .histograms
            .entry("cache_operation_duration_seconds".to_string())
            .or_insert_with(Vec::new)
            .push(duration.as_secs_f64());

        // Update Prometheus metrics
        let operation_str = operation.to_string();
        let hit_str = hit.to_string();
        counter!("cache_operations_total", "operation" => operation_str.clone(), "hit" => hit_str)
            .increment(1);
        histogram!("cache_operation_duration_seconds", "operation" => operation_str)
            .record(duration.as_secs_f64());
    }

    /// Update gauge metric
    pub async fn set_gauge(&self, name: &str, value: f64) {
        let mut registry = self.registry.write().await;
        registry.gauges.insert(name.to_string(), value);

        // Update Prometheus gauge
        let name_str = name.to_string();
        gauge!(name_str).set(value);
    }

    /// Increment counter metric
    pub async fn increment_counter(&self, name: &str, value: u64) {
        let mut registry = self.registry.write().await;
        *registry.counters.entry(name.to_string()).or_insert(0) += value;

        // Update Prometheus counter
        let name_str = name.to_string();
        counter!(name_str).increment(value);
    }

    /// Get metrics summary
    pub async fn get_summary(&self) -> MetricsSummary {
        let registry = self.registry.read().await;

        let uptime_seconds = self.start_time.elapsed().as_secs();
        let requests_total = registry
            .counters
            .get("http_requests_total")
            .copied()
            .unwrap_or(0);
        let requests_per_second = if uptime_seconds > 0 {
            requests_total as f64 / uptime_seconds as f64
        } else {
            0.0
        };

        let sparql_queries_total = registry
            .counters
            .get("sparql_queries_total")
            .copied()
            .unwrap_or(0);
        let sparql_updates_total = registry
            .counters
            .get("sparql_updates_total")
            .copied()
            .unwrap_or(0);

        // Calculate cache hit ratio
        let cache_hits = registry
            .counters
            .get("cache_hits_total")
            .copied()
            .unwrap_or(0);
        let cache_misses = registry
            .counters
            .get("cache_misses_total")
            .copied()
            .unwrap_or(0);
        let cache_hit_ratio = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64 * 100.0
        } else {
            0.0
        };

        // Calculate average response time
        let response_times = registry
            .histograms
            .get("http_request_duration_seconds")
            .cloned()
            .unwrap_or_default();
        let average_response_time_ms = if !response_times.is_empty() {
            response_times.iter().sum::<f64>() / response_times.len() as f64 * 1000.0
        } else {
            0.0
        };

        // Calculate error rate
        let successful_requests = registry
            .counters
            .get("sparql_queries_success")
            .copied()
            .unwrap_or(0)
            + registry
                .counters
                .get("sparql_updates_success")
                .copied()
                .unwrap_or(0);
        let failed_requests = registry
            .counters
            .get("sparql_queries_failed")
            .copied()
            .unwrap_or(0)
            + registry
                .counters
                .get("sparql_updates_failed")
                .copied()
                .unwrap_or(0);
        let total_sparql_operations = successful_requests + failed_requests;
        let error_rate_percent = if total_sparql_operations > 0 {
            failed_requests as f64 / total_sparql_operations as f64 * 100.0
        } else {
            0.0
        };

        MetricsSummary {
            uptime_seconds,
            requests_total,
            requests_per_second,
            sparql_queries_total,
            sparql_updates_total,
            active_connections: registry
                .gauges
                .get("active_connections")
                .copied()
                .unwrap_or(0.0) as u64,
            cache_hit_ratio,
            average_response_time_ms,
            error_rate_percent,
            system: registry.system_metrics.clone(),
            custom_metrics: HashMap::new(), // Could be populated with additional custom metrics
        }
    }

    /// Get health status
    pub async fn get_health_status(&self) -> HealthStatus {
        let mut checks = HashMap::new();
        let start_time = Instant::now();

        // Basic service health check
        checks.insert(
            "service".to_string(),
            CheckResult {
                status: HealthState::Healthy,
                message: "Service is running".to_string(),
                duration_ms: start_time.elapsed().as_millis() as u64,
                timestamp: current_timestamp(),
            },
        );

        // Memory health check
        let registry = self.registry.read().await;
        let memory_usage = registry.system_metrics.memory_usage_bytes;
        let memory_total = registry.system_metrics.memory_total_bytes;
        let memory_percent = if memory_total > 0 {
            (memory_usage as f64 / memory_total as f64) * 100.0
        } else {
            0.0
        };

        let memory_status = if memory_percent > 90.0 {
            HealthState::Unhealthy
        } else if memory_percent > 80.0 {
            HealthState::Degraded
        } else {
            HealthState::Healthy
        };

        checks.insert(
            "memory".to_string(),
            CheckResult {
                status: memory_status,
                message: format!("Memory usage: {memory_percent:.1}%"),
                duration_ms: start_time.elapsed().as_millis() as u64,
                timestamp: current_timestamp(),
            },
        );

        // Overall status determination
        let overall_status = if checks.values().any(|c| c.status == HealthState::Unhealthy) {
            HealthState::Unhealthy
        } else if checks.values().any(|c| c.status == HealthState::Degraded) {
            HealthState::Degraded
        } else {
            HealthState::Healthy
        };

        HealthStatus {
            status: overall_status,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime: format_duration(self.start_time.elapsed()),
            timestamp: current_timestamp(),
            checks,
        }
    }

    /// Get Prometheus metrics (if enabled)
    #[cfg(feature = "metrics")]
    pub async fn get_prometheus_metrics(&self) -> FusekiResult<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.prometheus_registry.gather();

        encoder
            .encode_to_string(&metric_families)
            .map_err(|e| FusekiError::internal(format!("Failed to encode Prometheus metrics: {e}")))
    }

    /// Create metrics router for HTTP endpoints
    pub fn create_router(&self) -> Router<Arc<Self>> {
        Router::new()
            .route("/metrics", get(prometheus_metrics_handler))
            .route("/metrics/summary", get(metrics_summary_handler))
            .route("/health", get(health_handler))
            .route("/health/live", get(liveness_handler))
            .route("/health/ready", get(readiness_handler))
    }
}

/// Collect system-level metrics
async fn collect_system_metrics() -> FusekiResult<SystemMetrics> {
    // This is a simplified implementation
    // In a real-world scenario, you'd use a proper system monitoring library

    use std::fs;

    let mut metrics = SystemMetrics::default();

    // Try to read memory info on Linux
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(value) = parse_meminfo_value(line) {
                    metrics.memory_total_bytes = value * 1024; // Convert KB to bytes
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(value) = parse_meminfo_value(line) {
                    let available = value * 1024; // Convert KB to bytes
                    metrics.memory_usage_bytes =
                        metrics.memory_total_bytes.saturating_sub(available);
                }
            }
        }
    }

    // Try to read CPU info (simplified)
    if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
        if let Some(load) = loadavg.split_whitespace().next() {
            if let Ok(load_value) = load.parse::<f64>() {
                // Very simplified CPU usage estimation
                metrics.cpu_usage_percent = (load_value * 100.0).min(100.0);
            }
        }
    }

    // Get process file descriptor count
    if let Ok(entries) = fs::read_dir("/proc/self/fd") {
        metrics.open_file_descriptors = entries.count() as u64;
    }

    Ok(metrics)
}

/// Parse memory info value from /proc/meminfo
fn parse_meminfo_value(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Format duration as human-readable string
fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

/// Get current timestamp as ISO 8601 string
fn current_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .unwrap_or_else(chrono::Utc::now)
                .to_rfc3339()
        })
        .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339())
}

// HTTP handlers for metrics endpoints

/// Prometheus metrics endpoint handler
#[cfg(feature = "metrics")]
async fn prometheus_metrics_handler(
    State(metrics): State<Arc<MetricsService>>,
) -> impl IntoResponse {
    match metrics.get_prometheus_metrics().await {
        Ok(metrics_text) => (
            [("content-type", "text/plain; charset=utf-8")],
            metrics_text,
        )
            .into_response(),
        Err(e) => {
            error!("Failed to get Prometheus metrics: {}", e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to generate metrics",
            )
                .into_response()
        }
    }
}

/// Fallback metrics handler when Prometheus is not enabled
#[cfg(not(feature = "metrics"))]
async fn prometheus_metrics_handler(
    State(metrics): State<Arc<MetricsService>>,
) -> impl IntoResponse {
    let summary = metrics.get_summary().await;
    Json(summary)
}

/// Metrics summary endpoint handler
async fn metrics_summary_handler(State(metrics): State<Arc<MetricsService>>) -> impl IntoResponse {
    let summary = metrics.get_summary().await;
    Json(summary)
}

/// Health check endpoint handler
async fn health_handler(State(metrics): State<Arc<MetricsService>>) -> impl IntoResponse {
    let health = metrics.get_health_status().await;
    let status_code = match health.status {
        HealthState::Healthy => axum::http::StatusCode::OK,
        HealthState::Degraded => axum::http::StatusCode::OK, // Still OK but with warnings
        HealthState::Unhealthy => axum::http::StatusCode::SERVICE_UNAVAILABLE,
    };

    (status_code, Json(health))
}

/// Liveness probe handler (for Kubernetes)
async fn liveness_handler() -> impl IntoResponse {
    // Simple liveness check - if we can respond, we're alive
    axum::http::StatusCode::OK
}

/// Readiness probe handler (for Kubernetes)
async fn readiness_handler(State(metrics): State<Arc<MetricsService>>) -> impl IntoResponse {
    let health = metrics.get_health_status().await;
    match health.status {
        HealthState::Healthy => axum::http::StatusCode::OK,
        _ => axum::http::StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HealthCheckConfig, MetricsConfig, TracingConfig, TracingOutput};

    fn create_test_metrics_service() -> MetricsService {
        let config = MonitoringConfig {
            metrics: MetricsConfig {
                enabled: true,
                endpoint: "/metrics".to_string(),
                port: None,
                namespace: "test".to_string(),
                collect_system_metrics: false,
                histogram_buckets: vec![0.1, 1.0, 10.0],
            },
            health_checks: HealthCheckConfig {
                enabled: true,
                interval_secs: 30,
                timeout_secs: 5,
                checks: vec!["service".to_string()],
            },
            tracing: TracingConfig {
                enabled: false,
                endpoint: None,
                service_name: "test".to_string(),
                sample_rate: 1.0,
                output: TracingOutput::Stdout,
            },
            prometheus: None,
        };

        MetricsService::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let metrics = create_test_metrics_service();

        // Record a request
        let request_metrics = RequestMetrics {
            method: "GET".to_string(),
            path: "/sparql".to_string(),
            status: 200,
            duration: Duration::from_millis(150),
            bytes_sent: 1024,
            bytes_received: 512,
        };

        metrics.record_request(request_metrics).await;

        // Record a SPARQL query
        metrics
            .record_sparql_query(Duration::from_millis(250), true, "SELECT")
            .await;

        // Get summary
        let summary = metrics.get_summary().await;
        assert_eq!(summary.requests_total, 1);
        assert_eq!(summary.sparql_queries_total, 1);
    }

    #[tokio::test]
    async fn test_health_status() {
        let metrics = create_test_metrics_service();

        let health = metrics.get_health_status().await;
        assert_eq!(health.status, HealthState::Healthy);
        assert!(!health.version.is_empty());
        assert!(health.checks.contains_key("service"));
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let metrics = create_test_metrics_service();

        // Record cache operations
        metrics
            .record_cache_operation(true, "get", Duration::from_millis(5))
            .await;
        metrics
            .record_cache_operation(false, "get", Duration::from_millis(50))
            .await;
        metrics
            .record_cache_operation(true, "set", Duration::from_millis(10))
            .await;

        let summary = metrics.get_summary().await;
        assert!((summary.cache_hit_ratio - 66.66666666666667).abs() < 1e-10); // 2 hits out of 3 operations (approximately)
    }

    #[tokio::test]
    async fn test_authentication_metrics() {
        let metrics = create_test_metrics_service();

        // Record authentication attempts
        metrics.record_authentication(true, "jwt").await;
        metrics.record_authentication(false, "basic").await;
        metrics.record_authentication(true, "jwt").await;

        // Check that metrics were recorded
        let registry = metrics.registry.read().await;
        assert_eq!(
            registry.counters.get("authentication_attempts_total"),
            Some(&3)
        );
        assert_eq!(
            registry.counters.get("authentication_success_total"),
            Some(&2)
        );
        assert_eq!(
            registry.counters.get("authentication_failures_total"),
            Some(&1)
        );
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }

    #[test]
    fn test_parse_meminfo_value() {
        assert_eq!(
            parse_meminfo_value("MemTotal:        8147484 kB"),
            Some(8147484)
        );
        assert_eq!(
            parse_meminfo_value("MemAvailable:    4567890 kB"),
            Some(4567890)
        );
        assert_eq!(parse_meminfo_value("Invalid line"), None);
    }

    #[tokio::test]
    async fn test_gauge_operations() {
        let metrics = create_test_metrics_service();

        metrics.set_gauge("test_gauge", 42.5).await;
        metrics.increment_counter("test_counter", 10).await;

        let registry = metrics.registry.read().await;
        assert_eq!(registry.gauges.get("test_gauge"), Some(&42.5));
        assert_eq!(registry.counters.get("test_counter"), Some(&10));
    }
}
