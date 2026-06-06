//! Metric types, collectors, and aggregators for embedding monitoring.
//!
//! This module contains all metric struct definitions and the enhanced
//! MetricsCollector backed by scirs2_core::metrics.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
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

/// Quality assessment record
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub timestamp: DateTime<Utc>,
    pub quality_score: f64,
    pub metrics: HashMap<String, f64>,
    pub assessment_details: String,
}

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

// ====================================================================================
// ENHANCED MONITORING WITH SCIRS2-CORE METRICS
// ====================================================================================

/// Enhanced metrics collector using scirs2_core::metrics
pub struct MetricsCollector {
    // Counters
    pub(crate) requests_total: Arc<Counter>,
    pub(crate) embeddings_generated_total: Arc<Counter>,
    pub(crate) errors_total: Arc<Counter>,
    pub(crate) cache_hits: Arc<Counter>,
    pub(crate) cache_misses: Arc<Counter>,

    // Gauges
    pub(crate) concurrent_requests: Arc<Gauge>,
    pub(crate) memory_usage_bytes: Arc<Gauge>,
    pub(crate) gpu_memory_bytes: Arc<Gauge>,
    pub(crate) cpu_utilization: Arc<Gauge>,
    pub(crate) gpu_utilization: Arc<Gauge>,

    // Histograms
    pub(crate) request_latency: Arc<Histogram>,
    pub(crate) embedding_generation_time: Arc<Histogram>,
    pub(crate) batch_size: Arc<Histogram>,

    // Timers
    pub(crate) inference_timer: Arc<Timer>,
    pub(crate) preprocessing_timer: Arc<Timer>,
    pub(crate) postprocessing_timer: Arc<Timer>,

    // Registry
    pub(crate) registry: Arc<MetricsRegistry>,
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
