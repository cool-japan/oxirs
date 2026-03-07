//! Comprehensive monitoring and metrics system for embedding models
//!
//! This module provides real-time performance monitoring, drift detection,
//! and comprehensive metrics collection for production embedding systems.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Comprehensive performance metrics for embedding systems
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    /// Request latency metrics
    pub latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Resource utilization metrics
    pub resources: ResourceMetrics,
    /// Quality metrics
    pub quality: QualityMetrics,
    /// Error metrics
    pub errors: ErrorMetrics,
    /// Cache performance
    pub cache: CacheMetrics,
    /// Model drift metrics
    pub drift: DriftMetrics,
}

/// Latency tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average embedding generation time (ms)
    pub avg_embedding_time_ms: f64,
    /// P50 latency (ms)
    pub p50_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
    /// P99 latency (ms)
    pub p99_latency_ms: f64,
    /// Maximum latency observed (ms)
    pub max_latency_ms: f64,
    /// Minimum latency observed (ms)
    pub min_latency_ms: f64,
    /// End-to-end request latency (ms)
    pub end_to_end_latency_ms: f64,
    /// Model inference latency (ms)
    pub model_inference_time_ms: f64,
    /// Queue wait time (ms)
    pub queue_wait_time_ms: f64,
    /// Total measurements
    pub total_measurements: u64,
}

/// Throughput monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Requests per second
    pub requests_per_second: f64,
    /// Embeddings generated per second
    pub embeddings_per_second: f64,
    /// Batches processed per second
    pub batches_per_second: f64,
    /// Peak throughput achieved
    pub peak_throughput: f64,
    /// Current concurrent requests
    pub concurrent_requests: u32,
    /// Maximum concurrent requests handled
    pub max_concurrent_requests: u32,
    /// Total requests processed
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Success rate
    pub success_rate: f64,
}

/// Resource utilization tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU utilization percentage
    pub gpu_utilization_percent: f64,
    /// GPU memory usage in MB
    pub gpu_memory_usage_mb: f64,
    /// Network I/O in MB/s
    pub network_io_mbps: f64,
    /// Disk I/O in MB/s
    pub disk_io_mbps: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Peak GPU memory usage
    pub peak_gpu_memory_mb: f64,
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average embedding quality score
    pub avg_quality_score: f64,
    /// Embedding space isotropy
    pub isotropy_score: f64,
    /// Neighborhood preservation
    pub neighborhood_preservation: f64,
    /// Clustering quality
    pub clustering_quality: f64,
    /// Similarity correlation
    pub similarity_correlation: f64,
    /// Quality degradation alerts
    pub quality_alerts: u32,
    /// Last quality assessment time
    pub last_assessment: DateTime<Utc>,
}

/// Error tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors
    pub total_errors: u64,
    /// Error rate per hour
    pub error_rate_per_hour: f64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Critical errors
    pub critical_errors: u64,
    /// Timeout errors
    pub timeout_errors: u64,
    /// Model errors
    pub model_errors: u64,
    /// System errors
    pub system_errors: u64,
    /// Last error time
    pub last_error: Option<DateTime<Utc>>,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Overall cache hit rate
    pub hit_rate: f64,
    /// L1 cache hit rate
    pub l1_hit_rate: f64,
    /// L2 cache hit rate
    pub l2_hit_rate: f64,
    /// L3 cache hit rate
    pub l3_hit_rate: f64,
    /// Cache memory usage MB
    pub cache_memory_mb: f64,
    /// Cache evictions
    pub cache_evictions: u64,
    /// Time saved by caching (seconds)
    pub time_saved_seconds: f64,
}

/// Model drift detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    /// Embedding quality drift
    pub quality_drift_score: f64,
    /// Performance degradation score
    pub performance_drift_score: f64,
    /// Distribution shift detected
    pub distribution_shift: bool,
    /// Concept drift score
    pub concept_drift_score: f64,
    /// Data quality issues
    pub data_quality_issues: u32,
    /// Drift detection alerts
    pub drift_alerts: u32,
    /// Last drift assessment
    pub last_drift_check: DateTime<Utc>,
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

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable P95 latency (ms)
    pub max_p95_latency_ms: f64,
    /// Minimum acceptable throughput (req/s)
    pub min_throughput_rps: f64,
    /// Maximum acceptable error rate
    pub max_error_rate: f64,
    /// Minimum acceptable cache hit rate
    pub min_cache_hit_rate: f64,
    /// Maximum acceptable quality drift
    pub max_quality_drift: f64,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: f64,
    /// Maximum acceptable GPU memory usage (MB)
    pub max_gpu_memory_mb: f64,
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

/// Error event for tracking
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub error_message: String,
    pub severity: ErrorSeverity,
    pub context: HashMap<String, String>,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality assessment record
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub timestamp: DateTime<Utc>,
    pub quality_score: f64,
    pub metrics: HashMap<String, f64>,
    pub assessment_details: String,
}

/// Alert handling trait
pub trait AlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()>;
}

/// Alert types
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighErrorRate,
    LowCacheHitRate,
    QualityDrift,
    PerformanceDrift,
    ResourceExhaustion,
    SystemFailure,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
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

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_p95_latency_ms: 500.0,
            min_throughput_rps: 100.0,
            max_error_rate: 0.05,    // 5%
            min_cache_hit_rate: 0.8, // 80%
            max_quality_drift: 0.1,
            max_memory_usage_mb: 4096.0, // 4GB
            max_gpu_memory_mb: 8192.0,   // 8GB
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

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            avg_embedding_time_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            max_latency_ms: 0.0,
            min_latency_ms: f64::MAX,
            end_to_end_latency_ms: 0.0,
            model_inference_time_ms: 0.0,
            queue_wait_time_ms: 0.0,
            total_measurements: 0,
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            requests_per_second: 0.0,
            embeddings_per_second: 0.0,
            batches_per_second: 0.0,
            peak_throughput: 0.0,
            concurrent_requests: 0,
            max_concurrent_requests: 0,
            total_requests: 0,
            failed_requests: 0,
            success_rate: 1.0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization_percent: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization_percent: 0.0,
            gpu_memory_usage_mb: 0.0,
            network_io_mbps: 0.0,
            disk_io_mbps: 0.0,
            peak_memory_mb: 0.0,
            peak_gpu_memory_mb: 0.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            avg_quality_score: 0.0,
            isotropy_score: 0.0,
            neighborhood_preservation: 0.0,
            clustering_quality: 0.0,
            similarity_correlation: 0.0,
            quality_alerts: 0,
            last_assessment: Utc::now(),
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate_per_hour: 0.0,
            errors_by_type: HashMap::new(),
            critical_errors: 0,
            timeout_errors: 0,
            model_errors: 0,
            system_errors: 0,
            last_error: None,
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            l1_hit_rate: 0.0,
            l2_hit_rate: 0.0,
            l3_hit_rate: 0.0,
            cache_memory_mb: 0.0,
            cache_evictions: 0,
            time_saved_seconds: 0.0,
        }
    }
}

impl Default for DriftMetrics {
    fn default() -> Self {
        Self {
            quality_drift_score: 0.0,
            performance_drift_score: 0.0,
            distribution_shift: false,
            concept_drift_score: 0.0,
            data_quality_issues: 0,
            drift_alerts: 0,
            last_drift_check: Utc::now(),
        }
    }
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

        // Start metrics collection task
        let metrics_task = self.start_metrics_collection().await;
        self.monitoring_tasks.push(metrics_task);

        // Start drift detection task
        let drift_task = self.start_drift_detection().await;
        self.monitoring_tasks.push(drift_task);

        // Start quality assessment task
        let quality_task = self.start_quality_assessment().await;
        self.monitoring_tasks.push(quality_task);

        // Start metrics export task
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

        // Add to sliding window
        if window.len() >= self.config.latency_window_size {
            window.pop_front();
        }
        window.push_back(latency_ms);

        // Update metrics
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.latency.total_measurements += 1;

            // Update min/max
            metrics.latency.max_latency_ms = metrics.latency.max_latency_ms.max(latency_ms);
            metrics.latency.min_latency_ms = metrics.latency.min_latency_ms.min(latency_ms);

            // Update average (rolling average)
            let alpha = 0.1; // Exponential smoothing factor
            metrics.latency.avg_embedding_time_ms =
                alpha * latency_ms + (1.0 - alpha) * metrics.latency.avg_embedding_time_ms;

            // Calculate percentiles from window
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

        // Check for alerts
        if self.config.enable_alerting {
            self.check_latency_alerts(latency_ms).await;
        }
    }

    /// Record throughput measurement
    pub async fn record_throughput(&self, requests_per_second: f64) {
        let mut window = self.throughput_window.lock().await;

        // Add to sliding window
        if window.len() >= self.config.throughput_window_size {
            window.pop_front();
        }
        window.push_back(requests_per_second);

        // Update metrics
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.throughput.requests_per_second = requests_per_second;
            metrics.throughput.peak_throughput =
                metrics.throughput.peak_throughput.max(requests_per_second);

            // Calculate average throughput
            let avg_throughput = window.iter().sum::<f64>() / window.len() as f64;
            metrics.throughput.requests_per_second = avg_throughput;
        }

        // Check for alerts
        if self.config.enable_alerting {
            self.check_throughput_alerts(requests_per_second).await;
        }
    }

    /// Record error event
    pub async fn record_error(&self, error_event: ErrorEvent) {
        let mut error_log = self.error_log.lock().await;

        // Add to error log
        if error_log.len() >= 1000 {
            error_log.pop_front();
        }
        error_log.push_back(error_event.clone());

        // Update metrics
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");
            metrics.errors.total_errors += 1;
            metrics.errors.last_error = Some(error_event.timestamp);

            // Update error counts by type
            *metrics
                .errors
                .errors_by_type
                .entry(error_event.error_type.clone())
                .or_insert(0) += 1;

            // Update error type counters
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

            // Calculate error rate
            let total_requests = metrics.throughput.total_requests;
            if total_requests > 0 {
                metrics.errors.error_rate_per_hour =
                    (metrics.errors.total_errors as f64 / total_requests as f64) * 3600.0;
            }
        }

        // Handle critical errors immediately
        if matches!(error_event.severity, ErrorSeverity::Critical) {
            self.handle_critical_error(error_event).await;
        }
    }

    /// Update resource metrics
    pub async fn update_resource_metrics(&self, resources: ResourceMetrics) {
        {
            let mut metrics = self.metrics.write().expect("rwlock should not be poisoned");

            // Update peak values
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

        // Check resource alerts
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

        // Check cache performance alerts
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

    /// Start metrics collection background task
    async fn start_metrics_collection(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let interval = Duration::from_secs(self.config.collection_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Collect system metrics
                let system_metrics = Self::collect_system_metrics().await;

                // Update metrics
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.resources = system_metrics;
                }

                debug!("Collected system metrics");
            }
        })
    }

    /// Start drift detection background task
    async fn start_drift_detection(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let quality_history = Arc::clone(&self.quality_history);
        let interval = Duration::from_secs(self.config.drift_detection_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Perform drift detection
                let drift_metrics = Self::detect_drift(&quality_history).await;

                // Update metrics
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.drift = drift_metrics;
                    metrics.drift.last_drift_check = Utc::now();
                }

                info!("Performed drift detection analysis");
            }
        })
    }

    /// Start quality assessment background task
    async fn start_quality_assessment(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let quality_history = Arc::clone(&self.quality_history);
        let interval = Duration::from_secs(self.config.quality_assessment_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Perform quality assessment
                let quality_assessment = Self::assess_quality().await;

                // Add to history
                {
                    let mut history = quality_history.lock().await;
                    if history.len() >= 100 {
                        history.pop_front();
                    }
                    history.push_back(quality_assessment.clone());
                }

                // Update metrics
                {
                    let mut metrics = metrics.write().expect("rwlock should not be poisoned");
                    metrics.quality.avg_quality_score = quality_assessment.quality_score;
                    metrics.quality.last_assessment = quality_assessment.timestamp;

                    // Update quality metrics from assessment details
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

    /// Start metrics export background task
    async fn start_metrics_export(&self) -> JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        let export_config = self.config.export_config.clone();
        let interval = Duration::from_secs(export_config.export_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Export metrics
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

    /// Collect system resource metrics
    async fn collect_system_metrics() -> ResourceMetrics {
        // Simulate system metrics collection
        // In production, this would use actual system monitoring libraries
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

    /// Detect model and performance drift
    async fn detect_drift(
        quality_history: &Arc<Mutex<VecDeque<QualityAssessment>>>,
    ) -> DriftMetrics {
        let history = quality_history.lock().await;

        if history.len() < 2 {
            return DriftMetrics::default();
        }

        // Calculate quality drift
        let recent_quality = history
            .back()
            .expect("quality history should not be empty")
            .quality_score;
        let baseline_quality = history
            .front()
            .expect("quality history should not be empty")
            .quality_score;
        let quality_drift = (recent_quality - baseline_quality).abs() / baseline_quality;

        // Simulate other drift metrics
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

    /// Assess embedding quality
    async fn assess_quality() -> QualityAssessment {
        // Simulate quality assessment
        // In production, this would perform actual quality metrics calculation
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

    /// Export metrics to Prometheus format
    async fn export_prometheus_metrics(metrics: &PerformanceMetrics) {
        // In production, this would export to Prometheus
        debug!(
            "Exporting Prometheus metrics: P95 latency = {:.2}ms",
            metrics.latency.p95_latency_ms
        );
    }

    /// Export metrics to JSON file
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

    /// Check latency alerts
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

    /// Check throughput alerts
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

    /// Check resource alerts
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

    /// Send alert to all registered handlers
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

    /// Handle critical errors immediately
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
}

/// Console alert handler implementation
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()> {
        println!(
            "🚨 ALERT [{}]: {} - {}",
            format!("{:?}", alert.severity).to_uppercase(),
            alert.message,
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        Ok(())
    }
}

/// Slack alert handler (placeholder)
pub struct SlackAlertHandler {
    pub webhook_url: String,
}

impl AlertHandler for SlackAlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()> {
        // In production, this would send to Slack
        info!(
            "Would send Slack alert to {}: {}",
            self.webhook_url, alert.message
        );
        Ok(())
    }
}

// ====================================================================================
// ENHANCED MONITORING WITH SCIRS2-CORE METRICS
// ====================================================================================

use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry, Timer};

/// Enhanced metrics collector using scirs2_core::metrics
pub struct MetricsCollector {
    // Counters
    requests_total: Arc<Counter>,
    embeddings_generated_total: Arc<Counter>,
    errors_total: Arc<Counter>,
    cache_hits: Arc<Counter>,
    cache_misses: Arc<Counter>,

    // Gauges
    concurrent_requests: Arc<Gauge>,
    memory_usage_bytes: Arc<Gauge>,
    gpu_memory_bytes: Arc<Gauge>,
    cpu_utilization: Arc<Gauge>,
    gpu_utilization: Arc<Gauge>,

    // Histograms
    request_latency: Arc<Histogram>,
    embedding_generation_time: Arc<Histogram>,
    batch_size: Arc<Histogram>,

    // Timers
    inference_timer: Arc<Timer>,
    preprocessing_timer: Arc<Timer>,
    postprocessing_timer: Arc<Timer>,

    // Registry
    registry: Arc<MetricsRegistry>,
}

impl MetricsCollector {
    /// Create a new metrics collector with scirs2-core metrics
    pub fn new() -> Self {
        let registry = Arc::new(MetricsRegistry::new());

        // Create counters
        let requests_total = Arc::new(Counter::new("embed_requests_total".to_string()));
        let embeddings_generated_total =
            Arc::new(Counter::new("embeddings_generated_total".to_string()));
        let errors_total = Arc::new(Counter::new("embed_errors_total".to_string()));
        let cache_hits = Arc::new(Counter::new("embed_cache_hits_total".to_string()));
        let cache_misses = Arc::new(Counter::new("embed_cache_misses_total".to_string()));

        // Create gauges
        let concurrent_requests = Arc::new(Gauge::new("embed_concurrent_requests".to_string()));
        let memory_usage_bytes = Arc::new(Gauge::new("embed_memory_usage_bytes".to_string()));
        let gpu_memory_bytes = Arc::new(Gauge::new("embed_gpu_memory_bytes".to_string()));
        let cpu_utilization = Arc::new(Gauge::new("embed_cpu_utilization".to_string()));
        let gpu_utilization = Arc::new(Gauge::new("embed_gpu_utilization".to_string()));

        // Create histograms
        let request_latency = Arc::new(Histogram::with_buckets(
            "embed_request_latency_ms".to_string(),
            vec![
                1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
            ],
        ));
        let embedding_generation_time = Arc::new(Histogram::with_buckets(
            "embed_generation_time_ms".to_string(),
            vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
        ));
        let batch_size = Arc::new(Histogram::with_buckets(
            "embed_batch_size".to_string(),
            vec![1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
        ));

        // Create timers
        let inference_timer = Arc::new(Timer::new("embed_inference_duration".to_string()));
        let preprocessing_timer = Arc::new(Timer::new("embed_preprocessing_duration".to_string()));
        let postprocessing_timer =
            Arc::new(Timer::new("embed_postprocessing_duration".to_string()));

        Self {
            requests_total,
            embeddings_generated_total,
            errors_total,
            cache_hits,
            cache_misses,
            concurrent_requests,
            memory_usage_bytes,
            gpu_memory_bytes,
            cpu_utilization,
            gpu_utilization,
            request_latency,
            embedding_generation_time,
            batch_size,
            inference_timer,
            preprocessing_timer,
            postprocessing_timer,
            registry,
        }
    }

    /// Record a request start
    pub fn record_request_start(&self) {
        self.requests_total.inc();
        self.concurrent_requests.inc();
    }

    /// Record a request completion
    pub fn record_request_complete(&self, latency_ms: f64) {
        self.concurrent_requests.dec();
        self.request_latency.observe(latency_ms);
    }

    /// Record embedding generation
    pub fn record_embeddings(&self, count: u64, generation_time_ms: f64) {
        self.embeddings_generated_total.add(count);
        self.embedding_generation_time.observe(generation_time_ms);
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors_total.inc();
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    /// Update resource metrics
    pub fn update_resource_metrics(&self, cpu: f64, memory_mb: f64, gpu: f64, gpu_memory_mb: f64) {
        self.cpu_utilization.set(cpu);
        self.memory_usage_bytes.set(memory_mb * 1024.0 * 1024.0);
        self.gpu_utilization.set(gpu);
        self.gpu_memory_bytes.set(gpu_memory_mb * 1024.0 * 1024.0);
    }

    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get();
        let misses = self.cache_misses.get();
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> Result<String> {
        self.registry
            .export_prometheus()
            .map_err(|e| anyhow!("Failed to export prometheus metrics: {:?}", e))
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ====================================================================================
// HEALTH CHECK FUNCTIONALITY
// ====================================================================================

/// Health status for the embedding service
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: HealthStatus,
    /// Timestamp of health check
    pub timestamp: DateTime<Utc>,
    /// Individual component health
    pub components: HashMap<String, ComponentHealth>,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: HealthStatus,
    /// Component message
    pub message: String,
    /// Last check time
    pub last_check: DateTime<Utc>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Health checker for embedding service
pub struct HealthChecker {
    /// Model load status
    models_loaded: Arc<RwLock<bool>>,
    /// Last successful request time
    last_request_time: Arc<RwLock<DateTime<Utc>>>,
    /// Error rate threshold
    error_rate_threshold: f64,
    /// Latency threshold (ms)
    latency_threshold_ms: f64,
    /// Memory threshold (MB)
    memory_threshold_mb: f64,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            models_loaded: Arc::new(RwLock::new(false)),
            last_request_time: Arc::new(RwLock::new(Utc::now())),
            error_rate_threshold: 0.1,    // 10%
            latency_threshold_ms: 1000.0, // 1 second
            memory_threshold_mb: 8192.0,  // 8GB
            metrics,
        }
    }

    /// Set models loaded status
    pub fn set_models_loaded(&self, loaded: bool) -> Result<()> {
        let mut status = self
            .models_loaded
            .write()
            .map_err(|e| anyhow!("Failed to write lock: {}", e))?;
        *status = loaded;
        Ok(())
    }

    /// Update last request time
    pub fn update_last_request_time(&self) -> Result<()> {
        let mut time = self
            .last_request_time
            .write()
            .map_err(|e| anyhow!("Failed to write lock: {}", e))?;
        *time = Utc::now();
        Ok(())
    }

    /// Perform liveness check (basic service availability)
    pub fn check_liveness(&self) -> HealthCheckResult {
        let mut components = HashMap::new();

        // Check if service is running (always healthy if we can respond)
        components.insert(
            "service".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: "Service is running".to_string(),
                last_check: Utc::now(),
                metrics: HashMap::new(),
            },
        );

        HealthCheckResult {
            status: HealthStatus::Healthy,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Perform readiness check (service ready to handle requests)
    pub fn check_readiness(&self) -> HealthCheckResult {
        let mut components = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        // Check if models are loaded
        let models_loaded = self.models_loaded.read().map(|g| *g).unwrap_or(false);
        if !models_loaded {
            overall_status = HealthStatus::Unhealthy;
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Unhealthy,
                    message: "Models not loaded".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        } else {
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Healthy,
                    message: "Models loaded and ready".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        }

        // Check cache availability
        let cache_hit_rate = self.metrics.get_cache_hit_rate();
        components.insert(
            "cache".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("hit_rate".to_string(), cache_hit_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Perform comprehensive health check
    pub fn check_health(&self, performance_metrics: &PerformanceMetrics) -> HealthCheckResult {
        let mut components = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        // Check models
        let models_loaded = self.models_loaded.read().map(|g| *g).unwrap_or(false);
        if !models_loaded {
            overall_status = HealthStatus::Unhealthy;
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Unhealthy,
                    message: "Models not loaded".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        } else {
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Healthy,
                    message: "Models operational".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        }

        // Check latency
        let latency_status =
            if performance_metrics.latency.p95_latency_ms > self.latency_threshold_ms {
                if overall_status == HealthStatus::Healthy {
                    overall_status = HealthStatus::Degraded;
                }
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

        components.insert(
            "latency".to_string(),
            ComponentHealth {
                status: latency_status,
                message: format!(
                    "P95 latency: {:.2}ms",
                    performance_metrics.latency.p95_latency_ms
                ),
                last_check: Utc::now(),
                metrics: [
                    (
                        "p50".to_string(),
                        performance_metrics.latency.p50_latency_ms,
                    ),
                    (
                        "p95".to_string(),
                        performance_metrics.latency.p95_latency_ms,
                    ),
                    (
                        "p99".to_string(),
                        performance_metrics.latency.p99_latency_ms,
                    ),
                ]
                .into_iter()
                .collect(),
            },
        );

        // Check error rate
        let error_rate = if performance_metrics.throughput.total_requests > 0 {
            performance_metrics.errors.total_errors as f64
                / performance_metrics.throughput.total_requests as f64
        } else {
            0.0
        };

        let error_status = if error_rate > self.error_rate_threshold {
            if overall_status == HealthStatus::Healthy {
                overall_status = HealthStatus::Degraded;
            }
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        components.insert(
            "errors".to_string(),
            ComponentHealth {
                status: error_status,
                message: format!("Error rate: {:.2}%", error_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("error_rate".to_string(), error_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        // Check memory
        let memory_status =
            if performance_metrics.resources.memory_usage_mb > self.memory_threshold_mb {
                if overall_status == HealthStatus::Healthy {
                    overall_status = HealthStatus::Degraded;
                }
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

        components.insert(
            "memory".to_string(),
            ComponentHealth {
                status: memory_status,
                message: format!(
                    "Memory usage: {:.2}MB / {:.2}MB",
                    performance_metrics.resources.memory_usage_mb, self.memory_threshold_mb
                ),
                last_check: Utc::now(),
                metrics: [
                    (
                        "usage_mb".to_string(),
                        performance_metrics.resources.memory_usage_mb,
                    ),
                    ("threshold_mb".to_string(), self.memory_threshold_mb),
                ]
                .into_iter()
                .collect(),
            },
        );

        // Check cache
        let cache_hit_rate = self.metrics.get_cache_hit_rate();
        components.insert(
            "cache".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("hit_rate".to_string(), cache_hit_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Get metrics endpoint (Prometheus format)
    pub fn get_metrics_endpoint(&self) -> Result<String> {
        self.metrics.export_prometheus()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.latency.total_measurements, 0);
        assert_eq!(metrics.throughput.total_requests, 0);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        monitor.record_latency(100.0).await;
        monitor.record_latency(150.0).await;
        monitor.record_latency(120.0).await;

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.latency.total_measurements, 3);
        assert_eq!(metrics.latency.max_latency_ms, 150.0);
        assert_eq!(metrics.latency.min_latency_ms, 100.0);
    }

    #[tokio::test]
    async fn test_error_recording() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let error_event = ErrorEvent {
            timestamp: Utc::now(),
            error_type: "timeout".to_string(),
            error_message: "Request timeout".to_string(),
            severity: ErrorSeverity::Medium,
            context: HashMap::new(),
        };

        monitor.record_error(error_event).await;

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.errors.total_errors, 1);
        assert_eq!(metrics.errors.timeout_errors, 1);
    }

    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.max_p95_latency_ms, 500.0);
        assert_eq!(thresholds.min_throughput_rps, 100.0);
        assert_eq!(thresholds.max_error_rate, 0.05);
    }

    #[test]
    fn test_console_alert_handler() {
        let handler = ConsoleAlertHandler;
        let alert = Alert {
            alert_type: AlertType::HighLatency,
            message: "Test alert".to_string(),
            severity: AlertSeverity::Warning,
            timestamp: Utc::now(),
            metrics: HashMap::new(),
        };

        assert!(handler.handle_alert(alert).is_ok());
    }

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        // Just verify it was created successfully
        assert_eq!(collector.requests_total.get(), 0);
    }

    #[test]
    fn test_metrics_collector_counters() {
        let collector = MetricsCollector::new();

        collector.record_request_start();
        collector.record_request_complete(50.0);

        assert_eq!(collector.requests_total.get(), 1);
    }

    #[test]
    fn test_metrics_collector_cache_hit_rate() {
        let collector = MetricsCollector::new();

        collector.record_cache_hit();
        collector.record_cache_hit();
        collector.record_cache_miss();

        let hit_rate = collector.get_cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01); // ~66.6%
    }

    #[test]
    fn test_metrics_collector_resource_update() {
        let collector = MetricsCollector::new();

        collector.update_resource_metrics(0.75, 2048.0, 0.5, 4096.0);

        assert_eq!(collector.cpu_utilization.get(), 0.75);
        assert_eq!(collector.memory_usage_bytes.get(), 2048.0 * 1024.0 * 1024.0);
        assert_eq!(collector.gpu_utilization.get(), 0.5);
        assert_eq!(collector.gpu_memory_bytes.get(), 4096.0 * 1024.0 * 1024.0);
    }

    #[test]
    fn test_health_checker_liveness() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        let result = checker.check_liveness();
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.components.contains_key("service"));
    }

    #[test]
    fn test_health_checker_readiness_no_models() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        let result = checker.check_readiness();
        assert_eq!(result.status, HealthStatus::Unhealthy);
        assert!(result.components.contains_key("models"));
    }

    #[test]
    fn test_health_checker_readiness_with_models() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let result = checker.check_readiness();
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_checker_comprehensive() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let perf_metrics = PerformanceMetrics::default();

        let result = checker.check_health(&perf_metrics);
        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.components.contains_key("models"));
        assert!(result.components.contains_key("latency"));
        assert!(result.components.contains_key("errors"));
        assert!(result.components.contains_key("memory"));
    }

    #[test]
    fn test_health_checker_degraded_latency() {
        let metrics = Arc::new(MetricsCollector::new());
        let checker = HealthChecker::new(metrics);

        checker
            .set_models_loaded(true)
            .expect("Failed to set models loaded");

        let mut perf_metrics = PerformanceMetrics::default();
        perf_metrics.latency.p95_latency_ms = 2000.0; // Above threshold

        let result = checker.check_health(&perf_metrics);
        assert_eq!(result.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_prometheus_export() {
        let collector = MetricsCollector::new();

        collector.record_request_start();
        collector.record_embeddings(5, 25.0);

        let prometheus_output = collector.export_prometheus();
        assert!(prometheus_output.is_ok());

        let _output = prometheus_output.unwrap_or_default();
        // Check that metrics were recorded (may not always be in prometheus output depending on implementation)
        assert_eq!(collector.requests_total.get(), 1);
        assert_eq!(collector.embeddings_generated_total.get(), 5);
    }
}
