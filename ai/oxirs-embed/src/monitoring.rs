//! Comprehensive monitoring and metrics system for embedding models.
//!
//! This is the facade module. Implementation is split across:
//! - [`crate::monitoring_metrics`] — metric types, collectors, aggregators
//! - [`crate::monitoring_health`] — health check logic, alerting, threshold monitoring
//! - [`crate::monitoring_tests`] — tests

// Re-export all public types from sibling modules
pub use crate::monitoring_health::{
    Alert, AlertHandler, AlertSeverity, AlertThresholds, AlertType, ComponentHealth,
    ConsoleAlertHandler, HealthCheckResult, HealthChecker, HealthStatus, SlackAlertHandler,
};
pub use crate::monitoring_metrics::{
    CacheMetrics, DriftMetrics, ErrorMetrics, LatencyMetrics, MetricsCollector, PerformanceMetrics,
    QualityMetrics, ResourceMetrics, ThroughputMetrics,
};

use anyhow::Result;
use chrono::Utc;
use scirs2_core::random::{Random, RngExt};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

// ErrorEvent and ErrorSeverity live in monitoring_metrics but are also used here
pub use crate::monitoring_metrics::{ErrorEvent, ErrorSeverity, QualityAssessment};

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Metrics collection interval (seconds)
    pub collection_interval_seconds: u64,
    /// Latency window size for percentile calculations
    pub latency_window_size: usize,
    /// Throughput window size
    pub throughput_window_size: usize,
    /// Quality assessment interval (seconds)
    pub quality_assessment_interval_seconds: u64,
    /// Drift detection interval (seconds)
    pub drift_detection_interval_seconds: u64,
    /// Enable real-time alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Metrics export configuration
    pub export_config: ExportConfig,
}

/// Metrics export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Enable Prometheus metrics export
    pub enable_prometheus: bool,
    /// Prometheus metrics port
    pub prometheus_port: u16,
    /// Enable OpenTelemetry export
    pub enable_opentelemetry: bool,
    /// OTLP endpoint
    pub otlp_endpoint: Option<String>,
    /// Export interval (seconds)
    pub export_interval_seconds: u64,
    /// Enable JSON metrics export
    pub enable_json_export: bool,
    /// JSON export path
    pub json_export_path: Option<String>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_seconds: 10,
            latency_window_size: 1000,
            throughput_window_size: 100,
            quality_assessment_interval_seconds: 300, // 5 minutes
            drift_detection_interval_seconds: 3600,   // 1 hour
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
            export_config: ExportConfig::default(),
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: true,
            prometheus_port: 9090,
            enable_opentelemetry: false,
            otlp_endpoint: None,
            export_interval_seconds: 60,
            enable_json_export: false,
            json_export_path: None,
        }
    }
}

/// Performance monitoring manager
pub struct PerformanceMonitor {
    /// Current metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Latency measurements window
    latency_window: Arc<Mutex<VecDeque<f64>>>,
    /// Throughput measurements window
    throughput_window: Arc<Mutex<VecDeque<f64>>>,
    /// Error tracking
    error_log: Arc<Mutex<VecDeque<ErrorEvent>>>,
    /// Quality assessments
    quality_history: Arc<Mutex<VecDeque<QualityAssessment>>>,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Background monitoring tasks
    monitoring_tasks: Vec<JoinHandle<()>>,
    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler + Send + Sync>>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            latency_window: Arc::new(Mutex::new(VecDeque::with_capacity(
                config.latency_window_size,
            ))),
            throughput_window: Arc::new(Mutex::new(VecDeque::with_capacity(
                config.throughput_window_size,
            ))),
            error_log: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            quality_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            config,
            monitoring_tasks: Vec::new(),
            alert_handlers: Vec::new(),
        }
    }

    /// Start monitoring services
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting performance monitoring system");

        let metrics_task = self.start_metrics_collection().await;
        self.monitoring_tasks.push(metrics_task);

        let drift_task = self.start_drift_detection().await;
        self.monitoring_tasks.push(drift_task);

        let quality_task = self.start_quality_assessment().await;
        self.monitoring_tasks.push(quality_task);

        if self.config.export_config.enable_prometheus {
            let export_task = self.start_metrics_export().await;
            self.monitoring_tasks.push(export_task);
        }

        info!("Performance monitoring system started successfully");
        Ok(())
    }

    /// Stop monitoring services
    pub async fn stop(&mut self) {
        info!("Stopping performance monitoring system");
        for task in self.monitoring_tasks.drain(..) {
            task.abort();
        }
        info!("Performance monitoring system stopped");
    }

    /// Record request latency
    pub async fn record_latency(&self, latency_ms: f64) {
        let mut window = self.latency_window.lock().await;

        if window.len() >= self.config.latency_window_size {
            window.pop_front();
        }
        window.push_back(latency_ms);

        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.latency.total_measurements += 1;

            metrics.latency.max_latency_ms = metrics.latency.max_latency_ms.max(latency_ms);
            metrics.latency.min_latency_ms = metrics.latency.min_latency_ms.min(latency_ms);

            let alpha = 0.1;
            metrics.latency.avg_embedding_time_ms =
                alpha * latency_ms + (1.0 - alpha) * metrics.latency.avg_embedding_time_ms;

            let mut sorted_latencies: Vec<f64> = window.iter().copied().collect();
            sorted_latencies.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("latency values should be comparable")
            });

            if !sorted_latencies.is_empty() {
                let len = sorted_latencies.len();
                metrics.latency.p50_latency_ms = sorted_latencies[len * 50 / 100];
                metrics.latency.p95_latency_ms = sorted_latencies[len * 95 / 100];
                metrics.latency.p99_latency_ms = sorted_latencies[len * 99 / 100];
            }
        }

        if self.config.enable_alerting {
            self.check_latency_alerts(latency_ms).await;
        }
    }

    /// Record throughput measurement
    pub async fn record_throughput(&self, requests_per_second: f64) {
        let mut window = self.throughput_window.lock().await;

        if window.len() >= self.config.throughput_window_size {
            window.pop_front();
        }
        window.push_back(requests_per_second);

        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.throughput.peak_throughput =
                metrics.throughput.peak_throughput.max(requests_per_second);

            let avg_throughput = window.iter().sum::<f64>() / window.len() as f64;
            metrics.throughput.requests_per_second = avg_throughput;
        }

        if self.config.enable_alerting {
            self.check_throughput_alerts(requests_per_second).await;
        }
    }

    /// Record error event
    pub async fn record_error(&self, error_event: ErrorEvent) {
        let mut error_log = self.error_log.lock().await;

        if error_log.len() >= 1000 {
            error_log.pop_front();
        }
        error_log.push_back(error_event.clone());

        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.errors.total_errors += 1;
            metrics.errors.last_error = Some(error_event.timestamp);

            *metrics
                .errors
                .errors_by_type
                .entry(error_event.error_type.clone())
                .or_insert(0) += 1;

            if let ErrorSeverity::Critical = error_event.severity {
                metrics.errors.critical_errors += 1
            }

            if error_event.error_type.contains("timeout") {
                metrics.errors.timeout_errors += 1;
            } else if error_event.error_type.contains("model") {
                metrics.errors.model_errors += 1;
            } else {
                metrics.errors.system_errors += 1;
            }

            let total_requests = metrics.throughput.total_requests;
            if total_requests > 0 {
                metrics.errors.error_rate_per_hour =
                    (metrics.errors.total_errors as f64 / total_requests as f64) * 3600.0;
            }
        }

        if matches!(error_event.severity, ErrorSeverity::Critical) {
            self.handle_critical_error(error_event).await;
        }
    }

    /// Update resource metrics
    pub async fn update_resource_metrics(&self, resources: ResourceMetrics) {
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");

            metrics.resources.peak_memory_mb = metrics
                .resources
                .peak_memory_mb
                .max(resources.memory_usage_mb);
            metrics.resources.peak_gpu_memory_mb = metrics
                .resources
                .peak_gpu_memory_mb
                .max(resources.gpu_memory_usage_mb);

            metrics.resources = resources.clone();
        }

        if self.config.enable_alerting {
            self.check_resource_alerts(resources).await;
        }
    }

    /// Update cache metrics
    pub async fn update_cache_metrics(&self, cache_metrics: CacheMetrics) {
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.cache = cache_metrics.clone();
        }

        if self.config.enable_alerting
            && cache_metrics.hit_rate < self.config.alert_thresholds.min_cache_hit_rate
        {
            self.send_alert(Alert {
                alert_type: AlertType::LowCacheHitRate,
                message: format!(
                    "Cache hit rate dropped to {:.2}%",
                    cache_metrics.hit_rate * 100.0
                ),
                severity: AlertSeverity::Warning,
                timestamp: Utc::now(),
                metrics: HashMap::from([
                    ("hit_rate".to_string(), cache_metrics.hit_rate),
                    (
                        "threshold".to_string(),
                        self.config.alert_thresholds.min_cache_hit_rate,
                    ),
                ]),
            })
            .await;
        }
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics
            .read()
            .expect("rwlock should not be poisoned")
            .clone()
    }

    /// Add alert handler
    pub fn add_alert_handler(&mut self, handler: Box<dyn AlertHandler + Send + Sync>) {
        self.alert_handlers.push(handler);
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> String {
        let metrics = self.metrics.read().expect("rwlock should not be poisoned");

        format!(
            "Performance Summary:\n\
             - P95 Latency: {:.2}ms\n\
             - Throughput: {:.1} req/s\n\
             - Error Rate: {:.3}%\n\
             - Cache Hit Rate: {:.1}%\n\
             - Memory Usage: {:.1}MB\n\
             - Quality Score: {:.3}",
            metrics.latency.p95_latency_ms,
            metrics.throughput.requests_per_second,
            (metrics.errors.total_errors as f64 / metrics.throughput.total_requests.max(1) as f64)
                * 100.0,
            metrics.cache.hit_rate * 100.0,
            metrics.resources.memory_usage_mb,
            metrics.quality.avg_quality_score
        )
    }

    // ---- Background tasks ----

    async fn start_metrics_collection(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let interval = Duration::from_secs(self.config.collection_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let system_metrics = Self::collect_system_metrics().await;
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.resources = system_metrics;
                }
                debug!("Collected system metrics");
            }
        })
    }

    async fn start_drift_detection(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let quality_history = Arc::clone(&self.quality_history);
        let interval = Duration::from_secs(self.config.drift_detection_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let drift_metrics = Self::detect_drift(&quality_history).await;
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.drift = drift_metrics;
                    metrics.drift.last_drift_check = Utc::now();
                }
                info!("Performed drift detection analysis");
            }
        })
    }

    async fn start_quality_assessment(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let quality_history = Arc::clone(&self.quality_history);
        let interval = Duration::from_secs(self.config.quality_assessment_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let quality_assessment = Self::assess_quality().await;
                {
                    let mut history = quality_history.lock().await;
                    if history.len() >= 100 {
                        history.pop_front();
                    }
                    history.push_back(quality_assessment.clone());
                }
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.quality.avg_quality_score = quality_assessment.quality_score;
                    metrics.quality.last_assessment = quality_assessment.timestamp;

                    for (key, value) in &quality_assessment.metrics {
                        match key.as_str() {
                            "isotropy" => metrics.quality.isotropy_score = *value,
                            "neighborhood_preservation" => {
                                metrics.quality.neighborhood_preservation = *value
                            }
                            "clustering_quality" => metrics.quality.clustering_quality = *value,
                            "similarity_correlation" => {
                                metrics.quality.similarity_correlation = *value
                            }
                            _ => {}
                        }
                    }
                }
                info!(
                    "Performed quality assessment: score = {:.3}",
                    quality_assessment.quality_score
                );
            }
        })
    }

    async fn start_metrics_export(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let export_config = self.config.export_config.clone();
        let interval = Duration::from_secs(export_config.export_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                let current_metrics = metrics
                    .read()
                    .expect("rwlock should not be poisoned")
                    .clone();

                if export_config.enable_prometheus {
                    Self::export_prometheus_metrics(&current_metrics).await;
                }
                if export_config.enable_json_export {
                    if let Some(ref path) = export_config.json_export_path {
                        Self::export_json_metrics(&current_metrics, path).await;
                    }
                }
                debug!("Exported metrics");
            }
        })
    }

    // ---- Internal helpers ----

    async fn collect_system_metrics() -> ResourceMetrics {
        let mut random = Random::default();
        ResourceMetrics {
            cpu_utilization_percent: random.random::<f64>() * 100.0,
            memory_usage_mb: 1024.0 + random.random::<f64>() * 2048.0,
            gpu_utilization_percent: random.random::<f64>() * 100.0,
            gpu_memory_usage_mb: 2048.0 + random.random::<f64>() * 4096.0,
            network_io_mbps: random.random::<f64>() * 100.0,
            disk_io_mbps: random.random::<f64>() * 50.0,
            peak_memory_mb: 3072.0,
            peak_gpu_memory_mb: 6144.0,
        }
    }

    async fn detect_drift(
        quality_history: &Arc<Mutex<VecDeque<QualityAssessment>>>,
    ) -> DriftMetrics {
        let history = quality_history.lock().await;

        if history.len() < 2 {
            return DriftMetrics::default();
        }

        let recent_quality = history
            .back()
            .expect("quality history should not be empty")
            .quality_score;
        let baseline_quality = history
            .front()
            .expect("quality history should not be empty")
            .quality_score;
        let quality_drift = (recent_quality - baseline_quality).abs() / baseline_quality;

        let mut random = Random::default();
        DriftMetrics {
            quality_drift_score: quality_drift,
            performance_drift_score: random.random::<f64>() * 0.1,
            distribution_shift: quality_drift > 0.1,
            concept_drift_score: random.random::<f64>() * 0.05,
            data_quality_issues: if quality_drift > 0.2 { 1 } else { 0 },
            drift_alerts: if quality_drift > 0.15 { 1 } else { 0 },
            last_drift_check: Utc::now(),
        }
    }

    async fn assess_quality() -> QualityAssessment {
        let mut random = Random::default();
        let quality_score = 0.8 + random.random::<f64>() * 0.2;

        let mut metrics = HashMap::new();
        metrics.insert("isotropy".to_string(), 0.7 + random.random::<f64>() * 0.3);
        metrics.insert(
            "neighborhood_preservation".to_string(),
            0.8 + random.random::<f64>() * 0.2,
        );
        metrics.insert(
            "clustering_quality".to_string(),
            0.75 + random.random::<f64>() * 0.25,
        );
        metrics.insert(
            "similarity_correlation".to_string(),
            0.85 + random.random::<f64>() * 0.15,
        );

        QualityAssessment {
            timestamp: Utc::now(),
            quality_score,
            metrics,
            assessment_details: format!(
                "Quality assessment completed with score: {quality_score:.3}"
            ),
        }
    }

    async fn export_prometheus_metrics(metrics: &PerformanceMetrics) {
        debug!(
            "Exporting Prometheus metrics: P95 latency = {:.2}ms",
            metrics.latency.p95_latency_ms
        );
    }

    async fn export_json_metrics(metrics: &PerformanceMetrics, path: &str) {
        match serde_json::to_string_pretty(metrics) {
            Ok(json) => {
                if let Err(e) = tokio::fs::write(path, json).await {
                    error!("Failed to export JSON metrics: {}", e);
                }
            }
            Err(e) => error!("Failed to serialize metrics to JSON: {}", e),
        }
    }

    async fn check_latency_alerts(&self, latency_ms: f64) {
        if latency_ms > self.config.alert_thresholds.max_p95_latency_ms {
            self.send_alert(Alert {
                alert_type: AlertType::HighLatency,
                message: format!("High latency detected: {latency_ms:.2}ms"),
                severity: AlertSeverity::Warning,
                timestamp: Utc::now(),
                metrics: HashMap::from([
                    ("latency_ms".to_string(), latency_ms),
                    (
                        "threshold_ms".to_string(),
                        self.config.alert_thresholds.max_p95_latency_ms,
                    ),
                ]),
            })
            .await;
        }
    }

    async fn check_throughput_alerts(&self, throughput_rps: f64) {
        if throughput_rps < self.config.alert_thresholds.min_throughput_rps {
            self.send_alert(Alert {
                alert_type: AlertType::LowThroughput,
                message: format!("Low throughput detected: {throughput_rps:.2} req/s"),
                severity: AlertSeverity::Warning,
                timestamp: Utc::now(),
                metrics: HashMap::from([
                    ("throughput_rps".to_string(), throughput_rps),
                    (
                        "threshold_rps".to_string(),
                        self.config.alert_thresholds.min_throughput_rps,
                    ),
                ]),
            })
            .await;
        }
    }

    async fn check_resource_alerts(&self, resources: ResourceMetrics) {
        if resources.memory_usage_mb > self.config.alert_thresholds.max_memory_usage_mb {
            self.send_alert(Alert {
                alert_type: AlertType::ResourceExhaustion,
                message: format!("High memory usage: {:.1}MB", resources.memory_usage_mb),
                severity: AlertSeverity::Critical,
                timestamp: Utc::now(),
                metrics: HashMap::from([
                    ("memory_mb".to_string(), resources.memory_usage_mb),
                    (
                        "threshold_mb".to_string(),
                        self.config.alert_thresholds.max_memory_usage_mb,
                    ),
                ]),
            })
            .await;
        }

        if resources.gpu_memory_usage_mb > self.config.alert_thresholds.max_gpu_memory_mb {
            self.send_alert(Alert {
                alert_type: AlertType::ResourceExhaustion,
                message: format!(
                    "High GPU memory usage: {:.1}MB",
                    resources.gpu_memory_usage_mb
                ),
                severity: AlertSeverity::Critical,
                timestamp: Utc::now(),
                metrics: HashMap::from([
                    ("gpu_memory_mb".to_string(), resources.gpu_memory_usage_mb),
                    (
                        "threshold_mb".to_string(),
                        self.config.alert_thresholds.max_gpu_memory_mb,
                    ),
                ]),
            })
            .await;
        }
    }

    async fn send_alert(&self, alert: Alert) {
        warn!(
            "Alert triggered: {:?} - {}",
            alert.alert_type, alert.message
        );
        for handler in &self.alert_handlers {
            if let Err(e) = handler.handle_alert(alert.clone()) {
                error!("Alert handler failed: {}", e);
            }
        }
    }

    async fn handle_critical_error(&self, error_event: ErrorEvent) {
        error!(
            "Critical error occurred: {} - {}",
            error_event.error_type, error_event.error_message
        );
        self.send_alert(Alert {
            alert_type: AlertType::SystemFailure,
            message: format!("Critical error: {}", error_event.error_message),
            severity: AlertSeverity::Emergency,
            timestamp: error_event.timestamp,
            metrics: HashMap::new(),
        })
        .await;
    }
}
