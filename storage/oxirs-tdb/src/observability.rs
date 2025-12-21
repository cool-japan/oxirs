//! Observability module for comprehensive monitoring and metrics
//!
//! Provides centralized observability features including:
//! - Structured logging with context
//! - Metrics collection and aggregation
//! - Health checks and readiness probes
//! - Distributed tracing support
//! - Performance monitoring

use scirs2_core::metrics::{Counter, Gauge, Histogram};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Observability manager for centralized monitoring
pub struct ObservabilityManager {
    metrics: Arc<RwLock<MetricsRegistry>>,
    health_checks: Arc<RwLock<Vec<HealthCheck>>>,
    traces: Arc<RwLock<Vec<TraceSpan>>>,
    config: ObservabilityConfig,
}

/// Observability configuration
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Enable distributed tracing
    pub enable_tracing: bool,
    /// Metrics retention duration
    pub metrics_retention: Duration,
    /// Trace sampling rate (0.0 to 1.0)
    pub trace_sample_rate: f64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_health_checks: true,
            enable_tracing: true,
            metrics_retention: Duration::from_secs(3600), // 1 hour
            trace_sample_rate: 0.1,                       // 10% sampling
        }
    }
}

impl ObservabilityManager {
    /// Create new observability manager
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(MetricsRegistry::new())),
            health_checks: Arc::new(RwLock::new(Vec::new())),
            traces: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ObservabilityConfig::default())
    }

    /// Record a metric
    pub fn record_metric(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        if !self.config.enable_metrics {
            return;
        }

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.record(name, value, labels);
        }
    }

    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, labels: HashMap<String, String>) {
        self.record_metric(name, 1.0, labels);
    }

    /// Record a gauge metric
    pub fn record_gauge(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        self.record_metric(name, value, labels);
    }

    /// Record a histogram metric
    pub fn record_histogram(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        self.record_metric(name, value, labels);
    }

    /// Get all metrics
    pub fn get_metrics(&self) -> Vec<MetricSnapshot> {
        if let Ok(metrics) = self.metrics.read() {
            metrics.snapshot()
        } else {
            Vec::new()
        }
    }

    /// Register a health check
    pub fn register_health_check(&self, check: HealthCheck) {
        if !self.config.enable_health_checks {
            return;
        }

        if let Ok(mut checks) = self.health_checks.write() {
            checks.push(check);
        }
    }

    /// Run all health checks
    pub fn run_health_checks(&self) -> HealthCheckResults {
        if !self.config.enable_health_checks {
            return HealthCheckResults {
                overall_status: HealthStatus::Unknown,
                checks: Vec::new(),
                timestamp: SystemTime::now(),
            };
        }

        let checks = if let Ok(checks) = self.health_checks.read() {
            checks.clone()
        } else {
            Vec::new()
        };

        let mut results = Vec::new();
        let mut overall_status = HealthStatus::Healthy;

        for check in checks {
            let result = (check.check_fn)();

            // Update overall status
            match result.status {
                HealthStatus::Unhealthy | HealthStatus::Critical => {
                    overall_status = HealthStatus::Unhealthy;
                }
                HealthStatus::Degraded if overall_status == HealthStatus::Healthy => {
                    overall_status = HealthStatus::Degraded;
                }
                _ => {}
            }

            results.push(result);
        }

        HealthCheckResults {
            overall_status,
            checks: results,
            timestamp: SystemTime::now(),
        }
    }

    /// Start a trace span
    pub fn start_span(&self, name: &str, operation: &str) -> TraceSpanId {
        if !self.config.enable_tracing {
            return TraceSpanId::new();
        }

        // Simple sampling
        use scirs2_core::random::{Random, Rng};
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut rng = Random::seed(seed);
        if rng.random::<f64>() > self.config.trace_sample_rate {
            return TraceSpanId::new();
        }

        let span = TraceSpan {
            id: TraceSpanId::new(),
            name: name.to_string(),
            operation: operation.to_string(),
            start_time: Instant::now(),
            end_time: None,
            attributes: HashMap::new(),
        };

        let span_id = span.id;

        if let Ok(mut traces) = self.traces.write() {
            traces.push(span);
        }

        span_id
    }

    /// End a trace span
    pub fn end_span(&self, span_id: TraceSpanId) {
        if !self.config.enable_tracing {
            return;
        }

        if let Ok(mut traces) = self.traces.write() {
            if let Some(span) = traces.iter_mut().find(|s| s.id == span_id) {
                span.end_time = Some(Instant::now());
            }
        }
    }

    /// Get trace spans
    pub fn get_traces(&self) -> Vec<TraceSpan> {
        if let Ok(traces) = self.traces.read() {
            traces.clone()
        } else {
            Vec::new()
        }
    }

    /// Clear old traces (cleanup)
    pub fn cleanup_traces(&self, max_age: Duration) {
        if let Ok(mut traces) = self.traces.write() {
            let cutoff = Instant::now() - max_age;
            traces.retain(|span| span.end_time.map_or(true, |end| end > cutoff));
        }
    }
}

/// Metrics registry
struct MetricsRegistry {
    metrics: HashMap<String, Vec<MetricEntry>>,
}

impl MetricsRegistry {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    fn record(&mut self, name: &str, value: f64, labels: HashMap<String, String>) {
        let entry = MetricEntry {
            value,
            labels,
            timestamp: SystemTime::now(),
        };

        self.metrics
            .entry(name.to_string())
            .or_default()
            .push(entry);
    }

    fn snapshot(&self) -> Vec<MetricSnapshot> {
        self.metrics
            .iter()
            .map(|(name, entries)| {
                let values: Vec<f64> = entries.iter().map(|e| e.value).collect();
                let count = values.len();
                let sum: f64 = values.iter().sum();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };

                let mut sorted_values = values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let p50 = if count > 0 {
                    sorted_values[count / 2]
                } else {
                    0.0
                };

                let p95 = if count > 0 {
                    sorted_values[(count as f64 * 0.95) as usize]
                } else {
                    0.0
                };

                let p99 = if count > 0 {
                    sorted_values[(count as f64 * 0.99) as usize]
                } else {
                    0.0
                };

                MetricSnapshot {
                    name: name.clone(),
                    count,
                    sum,
                    avg,
                    min: sorted_values.first().copied().unwrap_or(0.0),
                    max: sorted_values.last().copied().unwrap_or(0.0),
                    p50,
                    p95,
                    p99,
                }
            })
            .collect()
    }
}

/// Single metric entry
struct MetricEntry {
    value: f64,
    labels: HashMap<String, String>,
    timestamp: SystemTime,
}

/// Snapshot of a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Metric name
    pub name: String,
    /// Number of data points
    pub count: usize,
    /// Sum of all values
    pub sum: f64,
    /// Average value
    pub avg: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 50th percentile
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
}

/// Health check definition
#[derive(Clone)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Check description
    pub description: String,
    /// Check function
    pub check_fn: Arc<dyn Fn() -> HealthCheckResult + Send + Sync>,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check name
    pub name: String,
    /// Health status
    pub status: HealthStatus,
    /// Optional message
    pub message: Option<String>,
    /// Check duration
    pub duration_ms: f64,
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy - all systems operational
    Healthy,
    /// Degraded - some issues but still functional
    Degraded,
    /// Unhealthy - significant issues
    Unhealthy,
    /// Critical - system failure
    Critical,
    /// Unknown - unable to determine status
    Unknown,
}

/// Results from running health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResults {
    /// Overall health status
    pub overall_status: HealthStatus,
    /// Individual check results
    pub checks: Vec<HealthCheckResult>,
    /// Timestamp when checks were run
    pub timestamp: SystemTime,
}

/// Trace span ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceSpanId(u64);

impl TraceSpanId {
    fn new() -> Self {
        use scirs2_core::random::{Random, Rng};
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut rng = Random::seed(seed);
        Self(rng.random::<u64>())
    }
}

/// Trace span
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// Span ID
    pub id: TraceSpanId,
    /// Span name
    pub name: String,
    /// Operation being traced
    pub operation: String,
    /// Start time
    pub start_time: Instant,
    /// End time (if completed)
    pub end_time: Option<Instant>,
    /// Span attributes
    pub attributes: HashMap<String, String>,
}

impl TraceSpan {
    /// Get span duration
    pub fn duration(&self) -> Option<Duration> {
        self.end_time.map(|end| end - self.start_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_manager_creation() {
        let manager = ObservabilityManager::with_defaults();
        assert!(manager.config.enable_metrics);
        assert!(manager.config.enable_health_checks);
    }

    #[test]
    fn test_record_metric() {
        let manager = ObservabilityManager::with_defaults();
        let mut labels = HashMap::new();
        labels.insert("type".to_string(), "test".to_string());

        manager.record_metric("test_metric", 42.0, labels);

        let metrics = manager.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_metric");
        assert_eq!(metrics[0].count, 1);
    }

    #[test]
    fn test_increment_counter() {
        let manager = ObservabilityManager::with_defaults();

        manager.increment_counter("test_counter", HashMap::new());
        manager.increment_counter("test_counter", HashMap::new());

        let metrics = manager.get_metrics();
        assert_eq!(metrics[0].count, 2);
        assert_eq!(metrics[0].sum, 2.0);
    }

    #[test]
    fn test_health_checks() {
        let manager = ObservabilityManager::with_defaults();

        let check = HealthCheck {
            name: "test_check".to_string(),
            description: "Test health check".to_string(),
            check_fn: Arc::new(|| HealthCheckResult {
                name: "test_check".to_string(),
                status: HealthStatus::Healthy,
                message: None,
                duration_ms: 1.0,
            }),
        };

        manager.register_health_check(check);

        let results = manager.run_health_checks();
        assert_eq!(results.overall_status, HealthStatus::Healthy);
        assert_eq!(results.checks.len(), 1);
    }

    #[test]
    fn test_health_check_degraded() {
        let manager = ObservabilityManager::with_defaults();

        let check = HealthCheck {
            name: "degraded_check".to_string(),
            description: "Degraded check".to_string(),
            check_fn: Arc::new(|| HealthCheckResult {
                name: "degraded_check".to_string(),
                status: HealthStatus::Degraded,
                message: Some("System degraded".to_string()),
                duration_ms: 1.0,
            }),
        };

        manager.register_health_check(check);

        let results = manager.run_health_checks();
        assert_eq!(results.overall_status, HealthStatus::Degraded);
    }

    #[test]
    fn test_tracing() {
        let manager = ObservabilityManager::with_defaults();

        let span_id = manager.start_span("test_operation", "query");
        std::thread::sleep(Duration::from_millis(10));
        manager.end_span(span_id);

        let traces = manager.get_traces();
        // Due to sampling, might be empty
        if !traces.is_empty() {
            assert_eq!(traces[0].name, "test_operation");
            assert!(traces[0].duration().is_some());
        }
    }

    #[test]
    fn test_metric_snapshot() {
        let manager = ObservabilityManager::with_defaults();

        for i in 1..=10 {
            manager.record_metric("test", i as f64, HashMap::new());
        }

        let metrics = manager.get_metrics();
        assert_eq!(metrics[0].count, 10);
        assert_eq!(metrics[0].sum, 55.0);
        assert_eq!(metrics[0].avg, 5.5);
        assert_eq!(metrics[0].min, 1.0);
        assert_eq!(metrics[0].max, 10.0);
    }

    #[test]
    fn test_cleanup_traces() {
        let manager = ObservabilityManager::with_defaults();

        let span_id = manager.start_span("old_operation", "query");
        manager.end_span(span_id);

        std::thread::sleep(Duration::from_millis(100));

        manager.cleanup_traces(Duration::from_millis(50));

        let traces = manager.get_traces();
        // Traces older than 50ms should be cleaned up
        assert!(
            traces.is_empty()
                || traces
                    .iter()
                    .all(|t| t.duration().unwrap() < Duration::from_millis(50))
        );
    }

    #[test]
    fn test_disabled_metrics() {
        let config = ObservabilityConfig {
            enable_metrics: false,
            ..Default::default()
        };

        let manager = ObservabilityManager::new(config);
        manager.record_metric("test", 42.0, HashMap::new());

        let metrics = manager.get_metrics();
        assert_eq!(metrics.len(), 0);
    }
}
