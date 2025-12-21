//! # Performance Profiler and Optimizer
//!
//! This module provides comprehensive performance profiling and optimization
//! capabilities for stream processing applications.
//!
//! ## Features
//! - Real-time performance monitoring
//! - Bottleneck detection and analysis
//! - CPU and memory profiling
//! - Latency distribution tracking
//! - Throughput analysis
//! - Optimization recommendations
//! - Performance reports
//!
//! ## Usage
//! ```rust,ignore
//! let profiler = PerformanceProfiler::builder()
//!     .with_cpu_profiling()
//!     .with_memory_tracking()
//!     .build()
//!     .await?;
//!
//! profiler.start().await?;
//! // ... run stream processing ...
//! let report = profiler.generate_report().await;
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

/// Configuration for the performance profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable latency tracking
    pub enable_latency_tracking: bool,
    /// Enable throughput tracking
    pub enable_throughput_tracking: bool,
    /// Sampling interval
    pub sampling_interval: Duration,
    /// History size (number of samples to keep)
    pub history_size: usize,
    /// Enable automatic optimization recommendations
    pub enable_recommendations: bool,
    /// Latency percentiles to track
    pub percentiles: Vec<f64>,
    /// Warning thresholds
    pub warning_thresholds: WarningThresholds,
    /// Enable flame graph generation
    pub enable_flame_graph: bool,
    /// Maximum span depth
    pub max_span_depth: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_latency_tracking: true,
            enable_throughput_tracking: true,
            sampling_interval: Duration::from_secs(1),
            history_size: 3600, // 1 hour of per-second samples
            enable_recommendations: true,
            percentiles: vec![50.0, 90.0, 95.0, 99.0, 99.9],
            warning_thresholds: WarningThresholds::default(),
            enable_flame_graph: false,
            max_span_depth: 100,
        }
    }
}

/// Warning thresholds for performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_usage_percent: f64,
    /// Memory usage threshold (percentage)
    pub memory_usage_percent: f64,
    /// P99 latency threshold (microseconds)
    pub p99_latency_us: u64,
    /// Minimum throughput threshold (events/sec)
    pub min_throughput: f64,
    /// Buffer usage threshold (percentage)
    pub buffer_usage_percent: f64,
}

impl Default for WarningThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 80.0,
            memory_usage_percent: 85.0,
            p99_latency_us: 10000, // 10ms
            min_throughput: 1000.0,
            buffer_usage_percent: 90.0,
        }
    }
}

/// Performance span for tracking operations
#[derive(Debug, Clone)]
pub struct Span {
    /// Span name
    pub name: String,
    /// Start time
    pub start: Instant,
    /// End time
    pub end: Option<Instant>,
    /// Parent span ID
    pub parent_id: Option<u64>,
    /// Span ID
    pub id: u64,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Child spans
    pub children: Vec<u64>,
}

impl Span {
    /// Create a new span
    pub fn new(name: &str, id: u64) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            end: None,
            parent_id: None,
            id,
            tags: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// End the span
    pub fn finish(&mut self) {
        self.end = Some(Instant::now());
    }

    /// Get duration
    pub fn duration(&self) -> Duration {
        if let Some(end) = self.end {
            end.duration_since(self.start)
        } else {
            self.start.elapsed()
        }
    }

    /// Add a tag
    pub fn tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }
}

/// Latency histogram for tracking distribution
pub struct LatencyHistogram {
    /// Buckets for latency distribution
    buckets: Vec<(u64, AtomicU64)>, // (upper_bound_us, count)
    /// Total count
    total: AtomicU64,
    /// Sum for mean calculation
    sum: AtomicU64,
    /// Maximum value
    max: AtomicU64,
    /// Minimum value
    min: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new histogram with default buckets
    pub fn new() -> Self {
        // Buckets: 1us, 10us, 50us, 100us, 500us, 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, inf
        let bucket_bounds = vec![
            1,
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
            500000,
            1000000,
            u64::MAX,
        ];

        let buckets = bucket_bounds
            .into_iter()
            .map(|b| (b, AtomicU64::new(0)))
            .collect();

        Self {
            buckets,
            total: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            max: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
        }
    }

    /// Record a latency value
    pub fn record(&self, latency_us: u64) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(latency_us, Ordering::Relaxed);

        // Update max
        let mut current_max = self.max.load(Ordering::Relaxed);
        while latency_us > current_max {
            match self.max.compare_exchange_weak(
                current_max,
                latency_us,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }

        // Update min
        let mut current_min = self.min.load(Ordering::Relaxed);
        while latency_us < current_min {
            match self.min.compare_exchange_weak(
                current_min,
                latency_us,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_min = v,
            }
        }

        // Find bucket
        for (bound, count) in &self.buckets {
            if latency_us <= *bound {
                count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }

        let target = ((total as f64) * p / 100.0) as u64;
        let mut cumulative = 0u64;

        for (bound, count) in &self.buckets {
            cumulative += count.load(Ordering::Relaxed);
            if cumulative >= target {
                return *bound;
            }
        }

        self.max.load(Ordering::Relaxed)
    }

    /// Get mean latency
    pub fn mean(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.sum.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get statistics
    pub fn stats(&self) -> HistogramStats {
        HistogramStats {
            count: self.total.load(Ordering::Relaxed),
            mean: self.mean(),
            min: self.min.load(Ordering::Relaxed),
            max: self.max.load(Ordering::Relaxed),
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
            p999: self.percentile(99.9),
        }
    }

    /// Reset the histogram
    pub fn reset(&self) {
        self.total.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        for (_, count) in &self.buckets {
            count.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: u64,
    pub mean: f64,
    pub min: u64,
    pub max: u64,
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
}

/// Performance sample at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage bytes
    pub memory_usage_bytes: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// Events processed per second
    pub events_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: u64,
    /// Active operations
    pub active_operations: u64,
    /// P99 latency
    pub p99_latency_us: u64,
    /// Buffer usage percentage
    pub buffer_usage_percent: f64,
}

/// Operation timer for measuring specific operations
pub struct OperationTimer {
    /// Operation name
    name: String,
    /// Start time
    start: Instant,
    /// Tags
    tags: HashMap<String, String>,
}

impl OperationTimer {
    /// Create a new operation timer
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            tags: HashMap::new(),
        }
    }

    /// Add a tag
    pub fn tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Performance profiler
pub struct PerformanceProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Latency histogram
    latency_histogram: Arc<LatencyHistogram>,
    /// Active spans
    spans: Arc<RwLock<HashMap<u64, Span>>>,
    /// Performance samples
    samples: Arc<RwLock<VecDeque<PerformanceSample>>>,
    /// Warnings
    warnings: Arc<RwLock<Vec<PerformanceWarning>>>,
    /// Recommendations
    recommendations: Arc<RwLock<Vec<Recommendation>>>,
    /// Statistics
    stats: Arc<RwLock<ProfilerStats>>,
    /// Next span ID
    next_span_id: AtomicU64,
    /// Start time
    start_time: Arc<RwLock<Option<Instant>>>,
    /// Events counter
    events_counter: AtomicU64,
    /// Bytes counter
    bytes_counter: AtomicU64,
}

/// Performance warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceWarning {
    /// Warning type
    pub warning_type: WarningType,
    /// Message
    pub message: String,
    /// Severity
    pub severity: WarningSeverity,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Current value
    pub current_value: f64,
    /// Threshold
    pub threshold: f64,
}

/// Warning types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WarningType {
    HighCpuUsage,
    HighMemoryUsage,
    HighLatency,
    LowThroughput,
    BufferOverflow,
    MemoryLeak,
    GarbageCollection,
    ThreadContention,
}

/// Warning severity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Warning,
    Critical,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Category
    pub category: RecommendationCategory,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Impact
    pub impact: RecommendationImpact,
    /// Effort
    pub effort: RecommendationEffort,
    /// Priority score
    pub priority: u8,
}

/// Recommendation categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    BatchSize,
    BufferSize,
    Parallelism,
    MemoryManagement,
    CpuOptimization,
    NetworkOptimization,
    QueryOptimization,
    Configuration,
}

/// Recommendation impact
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationImpact {
    Low,
    Medium,
    High,
}

/// Recommendation effort
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecommendationEffort {
    Low,
    Medium,
    High,
}

/// Profiler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilerStats {
    /// Total events processed
    pub total_events: u64,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Total duration
    pub total_duration_secs: f64,
    /// Average throughput
    pub avg_throughput: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Warnings generated
    pub warnings_generated: u64,
    /// Spans recorded
    pub spans_recorded: u64,
    /// Samples collected
    pub samples_collected: u64,
}

impl PerformanceProfiler {
    /// Create a new profiler builder
    pub fn builder() -> ProfilerBuilder {
        ProfilerBuilder::new()
    }

    /// Create a new profiler with config
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            latency_histogram: Arc::new(LatencyHistogram::new()),
            spans: Arc::new(RwLock::new(HashMap::new())),
            samples: Arc::new(RwLock::new(VecDeque::new())),
            warnings: Arc::new(RwLock::new(Vec::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(ProfilerStats::default())),
            next_span_id: AtomicU64::new(0),
            start_time: Arc::new(RwLock::new(None)),
            events_counter: AtomicU64::new(0),
            bytes_counter: AtomicU64::new(0),
        }
    }

    /// Start profiling
    pub async fn start(&self) -> Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(anyhow!("Profiler already running"));
        }

        self.running.store(true, Ordering::Release);
        *self.start_time.write().await = Some(Instant::now());

        info!("Performance profiler started");
        Ok(())
    }

    /// Stop profiling
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);

        // Update final stats
        if let Some(start) = *self.start_time.read().await {
            let duration = start.elapsed();
            let mut stats = self.stats.write().await;
            stats.total_duration_secs = duration.as_secs_f64();
            stats.total_events = self.events_counter.load(Ordering::Relaxed);
            stats.total_bytes = self.bytes_counter.load(Ordering::Relaxed);

            if duration.as_secs_f64() > 0.0 {
                stats.avg_throughput = stats.total_events as f64 / duration.as_secs_f64();
            }
        }

        info!("Performance profiler stopped");
        Ok(())
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Record an event
    pub fn record_event(&self, bytes: u64) {
        self.events_counter.fetch_add(1, Ordering::Relaxed);
        self.bytes_counter.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record latency
    pub fn record_latency(&self, latency: Duration) {
        self.latency_histogram.record(latency.as_micros() as u64);
    }

    /// Start a new span
    pub async fn start_span(&self, name: &str) -> u64 {
        let id = self.next_span_id.fetch_add(1, Ordering::SeqCst);
        let span = Span::new(name, id);

        let mut spans = self.spans.write().await;
        spans.insert(id, span);

        let mut stats = self.stats.write().await;
        stats.spans_recorded += 1;

        id
    }

    /// End a span
    pub async fn end_span(&self, id: u64) -> Option<Duration> {
        let mut spans = self.spans.write().await;
        if let Some(span) = spans.get_mut(&id) {
            span.finish();
            let duration = span.duration();

            // Record latency if latency tracking is enabled
            if self.config.enable_latency_tracking {
                self.record_latency(duration);
            }

            Some(duration)
        } else {
            None
        }
    }

    /// Create an operation timer
    pub fn time_operation(&self, name: &str) -> OperationTimer {
        OperationTimer::new(name)
    }

    /// Record operation completion
    pub fn record_operation(&self, timer: OperationTimer) {
        let duration = timer.elapsed();
        self.record_latency(duration);
    }

    /// Collect a performance sample
    pub async fn collect_sample(&self) -> PerformanceSample {
        let now = Utc::now();

        // Calculate throughput
        let events = self.events_counter.load(Ordering::Relaxed);
        let bytes = self.bytes_counter.load(Ordering::Relaxed);

        let (events_per_second, bytes_per_second) =
            if let Some(start) = *self.start_time.read().await {
                let duration = start.elapsed().as_secs_f64();
                if duration > 0.0 {
                    (events as f64 / duration, (bytes as f64 / duration) as u64)
                } else {
                    (0.0, 0)
                }
            } else {
                (0.0, 0)
            };

        let latency_stats = self.latency_histogram.stats();

        let sample = PerformanceSample {
            timestamp: now,
            cpu_usage_percent: 0.0, // Would need system API
            memory_usage_bytes: 0,  // Would need system API
            memory_usage_percent: 0.0,
            events_per_second,
            bytes_per_second,
            active_operations: self.spans.read().await.len() as u64,
            p99_latency_us: latency_stats.p99,
            buffer_usage_percent: 0.0,
        };

        // Store sample
        let mut samples = self.samples.write().await;
        samples.push_back(sample.clone());
        while samples.len() > self.config.history_size {
            samples.pop_front();
        }
        drop(samples); // Release samples lock

        let mut stats = self.stats.write().await;
        stats.samples_collected += 1;
        drop(stats); // Release stats lock before calling check_warnings

        // Check for warnings (needs to acquire stats lock)
        self.check_warnings(&sample).await;

        sample
    }

    /// Check for performance warnings
    async fn check_warnings(&self, sample: &PerformanceSample) {
        let mut warnings = self.warnings.write().await;

        // Check CPU usage
        if sample.cpu_usage_percent > self.config.warning_thresholds.cpu_usage_percent {
            warnings.push(PerformanceWarning {
                warning_type: WarningType::HighCpuUsage,
                message: format!(
                    "CPU usage {}% exceeds threshold {}%",
                    sample.cpu_usage_percent, self.config.warning_thresholds.cpu_usage_percent
                ),
                severity: if sample.cpu_usage_percent > 95.0 {
                    WarningSeverity::Critical
                } else {
                    WarningSeverity::Warning
                },
                timestamp: sample.timestamp,
                current_value: sample.cpu_usage_percent,
                threshold: self.config.warning_thresholds.cpu_usage_percent,
            });
        }

        // Check latency
        if sample.p99_latency_us > self.config.warning_thresholds.p99_latency_us {
            warnings.push(PerformanceWarning {
                warning_type: WarningType::HighLatency,
                message: format!(
                    "P99 latency {}us exceeds threshold {}us",
                    sample.p99_latency_us, self.config.warning_thresholds.p99_latency_us
                ),
                severity: if sample.p99_latency_us
                    > self.config.warning_thresholds.p99_latency_us * 2
                {
                    WarningSeverity::Critical
                } else {
                    WarningSeverity::Warning
                },
                timestamp: sample.timestamp,
                current_value: sample.p99_latency_us as f64,
                threshold: self.config.warning_thresholds.p99_latency_us as f64,
            });
        }

        // Check throughput
        if sample.events_per_second < self.config.warning_thresholds.min_throughput {
            warnings.push(PerformanceWarning {
                warning_type: WarningType::LowThroughput,
                message: format!(
                    "Throughput {:.2} events/sec below threshold {:.2}",
                    sample.events_per_second, self.config.warning_thresholds.min_throughput
                ),
                severity: WarningSeverity::Warning,
                timestamp: sample.timestamp,
                current_value: sample.events_per_second,
                threshold: self.config.warning_thresholds.min_throughput,
            });
        }

        let mut stats = self.stats.write().await;
        stats.warnings_generated = warnings.len() as u64;
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();
        let latency_stats = self.latency_histogram.stats();
        let stats = self.stats.read().await;

        // Check latency
        if latency_stats.p99 > 10000 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::BatchSize,
                title: "Increase batch size".to_string(),
                description: "High P99 latency detected. Consider increasing batch size to amortize overhead.".to_string(),
                impact: RecommendationImpact::High,
                effort: RecommendationEffort::Low,
                priority: 9,
            });
        }

        // Check throughput
        if stats.avg_throughput < 1000.0 && stats.total_events > 100 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Parallelism,
                title: "Increase parallelism".to_string(),
                description:
                    "Low throughput detected. Consider increasing worker threads or partitions."
                        .to_string(),
                impact: RecommendationImpact::High,
                effort: RecommendationEffort::Medium,
                priority: 8,
            });
        }

        // Check variance
        if latency_stats.max > latency_stats.p99 * 10 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::MemoryManagement,
                title: "Investigate latency spikes".to_string(),
                description: "Large variance in latency detected. May indicate GC pressure or resource contention.".to_string(),
                impact: RecommendationImpact::Medium,
                effort: RecommendationEffort::High,
                priority: 7,
            });
        }

        // Store recommendations
        *self.recommendations.write().await = recommendations.clone();

        recommendations
    }

    /// Get latency statistics
    pub fn get_latency_stats(&self) -> HistogramStats {
        self.latency_histogram.stats()
    }

    /// Get all warnings
    pub async fn get_warnings(&self) -> Vec<PerformanceWarning> {
        self.warnings.read().await.clone()
    }

    /// Get samples
    pub async fn get_samples(&self) -> Vec<PerformanceSample> {
        self.samples.read().await.iter().cloned().collect()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> ProfilerStats {
        self.stats.read().await.clone()
    }

    /// Generate performance report
    pub async fn generate_report(&self) -> PerformanceReport {
        let stats = self.stats.read().await.clone();
        let latency_stats = self.latency_histogram.stats();
        let warnings = self.warnings.read().await.clone();
        let recommendations = self.generate_recommendations().await;
        let samples = self.samples.read().await.iter().cloned().collect();
        let summary = self.generate_summary(&stats, &latency_stats).await;

        PerformanceReport {
            generated_at: Utc::now(),
            duration_secs: stats.total_duration_secs,
            total_events: stats.total_events,
            total_bytes: stats.total_bytes,
            avg_throughput: stats.avg_throughput,
            peak_throughput: stats.peak_throughput,
            latency_stats,
            warnings,
            recommendations,
            samples,
            summary,
        }
    }

    /// Generate summary
    async fn generate_summary(&self, stats: &ProfilerStats, latency: &HistogramStats) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Performance Summary\n{}\n", "=".repeat(50)));
        summary.push_str(&format!("Duration: {:.2}s\n", stats.total_duration_secs));
        summary.push_str(&format!("Events processed: {}\n", stats.total_events));
        summary.push_str(&format!(
            "Throughput: {:.2} events/sec\n",
            stats.avg_throughput
        ));
        summary.push_str(&format!(
            "Latency P50/P99/Max: {}us / {}us / {}us\n",
            latency.p50, latency.p99, latency.max
        ));
        summary.push_str(&format!("Warnings: {}\n", stats.warnings_generated));

        summary
    }

    /// Reset profiler
    pub async fn reset(&self) {
        self.latency_histogram.reset();
        self.spans.write().await.clear();
        self.samples.write().await.clear();
        self.warnings.write().await.clear();
        self.recommendations.write().await.clear();
        *self.stats.write().await = ProfilerStats::default();
        self.events_counter.store(0, Ordering::Relaxed);
        self.bytes_counter.store(0, Ordering::Relaxed);
        *self.start_time.write().await = None;

        info!("Performance profiler reset");
    }
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Generation time
    pub generated_at: DateTime<Utc>,
    /// Duration
    pub duration_secs: f64,
    /// Total events
    pub total_events: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Average throughput
    pub avg_throughput: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Latency statistics
    pub latency_stats: HistogramStats,
    /// Warnings
    pub warnings: Vec<PerformanceWarning>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
    /// Samples
    pub samples: Vec<PerformanceSample>,
    /// Summary
    pub summary: String,
}

impl PerformanceReport {
    /// Convert to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("JSON error: {}", e))
    }

    /// Print report
    pub fn print(&self) {
        println!("{}", self.summary);

        if !self.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &self.warnings {
                println!("  [{:?}] {}", warning.severity, warning.message);
            }
        }

        if !self.recommendations.is_empty() {
            println!("\nRecommendations:");
            for rec in &self.recommendations {
                println!(
                    "  [Priority {}] {} - {}",
                    rec.priority, rec.title, rec.description
                );
            }
        }
    }
}

/// Profiler builder
pub struct ProfilerBuilder {
    config: ProfilerConfig,
}

impl ProfilerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ProfilerConfig::default(),
        }
    }

    /// Enable CPU profiling
    pub fn with_cpu_profiling(mut self) -> Self {
        self.config.enable_cpu_profiling = true;
        self
    }

    /// Enable memory tracking
    pub fn with_memory_tracking(mut self) -> Self {
        self.config.enable_memory_profiling = true;
        self
    }

    /// Set sampling interval
    pub fn sampling_interval(mut self, interval: Duration) -> Self {
        self.config.sampling_interval = interval;
        self
    }

    /// Set history size
    pub fn history_size(mut self, size: usize) -> Self {
        self.config.history_size = size;
        self
    }

    /// Set warning thresholds
    pub fn warning_thresholds(mut self, thresholds: WarningThresholds) -> Self {
        self.config.warning_thresholds = thresholds;
        self
    }

    /// Build the profiler
    pub fn build(self) -> PerformanceProfiler {
        PerformanceProfiler::new(self.config)
    }
}

impl Default for ProfilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profiler_creation() {
        let profiler = PerformanceProfiler::builder().build();
        assert!(!profiler.is_running());
    }

    #[tokio::test]
    async fn test_start_stop() {
        let profiler = PerformanceProfiler::builder().build();

        profiler.start().await.unwrap();
        assert!(profiler.is_running());

        profiler.stop().await.unwrap();
        assert!(!profiler.is_running());
    }

    #[tokio::test]
    async fn test_record_event() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        profiler.record_event(100);
        profiler.record_event(200);

        profiler.stop().await.unwrap();
        let stats = profiler.get_stats().await;
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.total_bytes, 300);
    }

    #[tokio::test]
    async fn test_latency_histogram() {
        let histogram = LatencyHistogram::new();

        histogram.record(100);
        histogram.record(200);
        histogram.record(1000);
        histogram.record(5000);
        histogram.record(10000);

        let stats = histogram.stats();
        assert_eq!(stats.count, 5);
        assert!(stats.min <= 100);
        assert!(stats.max >= 10000);
    }

    #[tokio::test]
    async fn test_spans() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        let span_id = profiler.start_span("test_operation").await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let duration = profiler.end_span(span_id).await;

        assert!(duration.is_some());
        assert!(duration.unwrap() >= Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_operation_timer() {
        let profiler = PerformanceProfiler::builder().build();

        let timer = profiler.time_operation("test");
        tokio::time::sleep(Duration::from_millis(5)).await;
        profiler.record_operation(timer);

        let stats = profiler.get_latency_stats();
        assert!(stats.count > 0);
    }

    #[tokio::test]
    async fn test_collect_sample() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        profiler.record_event(100);
        let sample = profiler.collect_sample().await;

        assert!(sample.events_per_second >= 0.0);
    }

    #[tokio::test]
    async fn test_recommendations() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        // Record some high latency operations
        for _ in 0..100 {
            profiler.record_latency(Duration::from_millis(50));
        }

        let recommendations = profiler.generate_recommendations().await;
        assert!(!recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_generate_report() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        for _ in 0..10 {
            profiler.record_event(100);
            profiler.record_latency(Duration::from_micros(500));
        }

        profiler.stop().await.unwrap();
        let report = profiler.generate_report().await;

        assert_eq!(report.total_events, 10);
        assert!(!report.summary.is_empty());
    }

    #[tokio::test]
    async fn test_warnings() {
        let thresholds = WarningThresholds {
            min_throughput: 10000.0, // Very high threshold
            ..Default::default()
        };

        let profiler = PerformanceProfiler::builder()
            .warning_thresholds(thresholds)
            .build();

        profiler.start().await.unwrap();

        // Wait a bit to ensure time passes for throughput calculation
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        profiler.record_event(100);

        // Collect sample which checks warnings
        profiler.collect_sample().await;

        let warnings = profiler.get_warnings().await;
        // Should have low throughput warning
        assert!(warnings
            .iter()
            .any(|w| w.warning_type == WarningType::LowThroughput));
    }

    #[tokio::test]
    async fn test_reset() {
        let profiler = PerformanceProfiler::builder().build();
        profiler.start().await.unwrap();

        profiler.record_event(100);
        profiler.record_latency(Duration::from_micros(100));

        profiler.reset().await;

        let stats = profiler.get_stats().await;
        assert_eq!(stats.total_events, 0);

        let latency = profiler.get_latency_stats();
        assert_eq!(latency.count, 0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let histogram = LatencyHistogram::new();

        // Add 100 samples
        for i in 1..=100 {
            histogram.record(i * 10);
        }

        let p50 = histogram.percentile(50.0);
        let p99 = histogram.percentile(99.0);

        assert!(p50 <= p99);
    }
}
