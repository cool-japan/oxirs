//! # Raft Consensus Profiling Integration
//!
//! Comprehensive performance profiling for Raft consensus operations using SciRS2-Core.
//! Tracks append_entries, snapshot operations, network round-trips, and query execution.
//!
//! ## Features
//! - Real-time performance metrics with SciRS2-Core profiling
//! - Histogram-based latency distributions
//! - Operation counters and timers
//! - Automatic percentile calculations (p50, p95, p99)
//! - Memory profiling with leak detection
//! - Performance regression detection
//! - SciRS2-Core Profiler integration for advanced bottleneck analysis
//! - SciRS2-Core Histogram for accurate latency tracking
//! - SciRS2-Core LeakDetector for memory safety validation

use crate::raft::OxirsNodeId;
use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry, Timer};
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[cfg(feature = "leak_detection")]
use scirs2_core::memory::{LeakDetectionConfig, LeakDetector};

/// Raft profiling categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaftOperation {
    /// Append entries operation
    AppendEntries,
    /// Request vote operation
    RequestVote,
    /// Install snapshot operation
    InstallSnapshot,
    /// Create snapshot operation
    CreateSnapshot,
    /// Restore snapshot operation
    RestoreSnapshot,
    /// Log compaction operation
    LogCompaction,
    /// Network round-trip
    NetworkRoundTrip,
    /// Query execution
    QueryExecution,
    /// Batch processing
    BatchProcessing,
    /// Log replication
    LogReplication,
}

impl RaftOperation {
    /// Get operation name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AppendEntries => "append_entries",
            Self::RequestVote => "request_vote",
            Self::InstallSnapshot => "install_snapshot",
            Self::CreateSnapshot => "create_snapshot",
            Self::RestoreSnapshot => "restore_snapshot",
            Self::LogCompaction => "log_compaction",
            Self::NetworkRoundTrip => "network_roundtrip",
            Self::QueryExecution => "query_execution",
            Self::BatchProcessing => "batch_processing",
            Self::LogReplication => "log_replication",
        }
    }

    /// Get profiling label for this operation
    pub fn profiling_label(&self) -> String {
        format!("raft_{}", self.as_str())
    }
}

/// Latency statistics for an operation
#[derive(Debug, Clone, Default)]
struct LatencyStats {
    /// All recorded latencies in microseconds
    latencies: Vec<f64>,
    /// Total operation count
    count: u64,
    /// Sum of all latencies
    sum_micros: f64,
    /// Minimum latency
    min_micros: f64,
    /// Maximum latency
    max_micros: f64,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            latencies: Vec::new(),
            count: 0,
            sum_micros: 0.0,
            min_micros: f64::MAX,
            max_micros: 0.0,
        }
    }

    fn record(&mut self, micros: f64) {
        self.latencies.push(micros);
        self.count += 1;
        self.sum_micros += micros;
        self.min_micros = self.min_micros.min(micros);
        self.max_micros = self.max_micros.max(micros);
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_micros / self.count as f64
        }
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let mean = self.mean();
        let variance = self
            .latencies
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.count as f64);

        variance.sqrt()
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

/// Comprehensive Raft profiling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftProfilingMetrics {
    /// Operation name
    pub operation: String,
    /// Node ID performing the operation
    pub node_id: OxirsNodeId,
    /// Total operation count
    pub operation_count: u64,
    /// Mean latency in microseconds
    pub mean_latency_micros: f64,
    /// Standard deviation in microseconds
    pub std_dev_micros: f64,
    /// 50th percentile (median) in microseconds
    pub p50_micros: f64,
    /// 95th percentile in microseconds
    pub p95_micros: f64,
    /// 99th percentile in microseconds
    pub p99_micros: f64,
    /// Minimum latency in microseconds
    pub min_micros: f64,
    /// Maximum latency in microseconds
    pub max_micros: f64,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Operations per second (throughput)
    pub ops_per_second: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Last update timestamp
    pub last_update: String,
}

/// Raft profiling manager with SciRS2-Core integration
#[derive(Clone)]
pub struct RaftProfiler {
    /// Node ID
    node_id: OxirsNodeId,
    /// Operation statistics
    stats: Arc<RwLock<HashMap<String, LatencyStats>>>,
    /// SciRS2-Core profiler for bottleneck analysis
    profiler: Arc<RwLock<Profiler>>,
    /// SciRS2-Core metrics registry
    #[allow(dead_code)]
    metrics_registry: Arc<MetricsRegistry>,
    /// SciRS2-Core histograms for latency distributions
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    /// SciRS2-Core counters for operation counts
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    /// Memory usage tracking
    memory_usage: Arc<RwLock<HashMap<String, u64>>>,
    /// Profiling enabled flag
    enabled: Arc<RwLock<bool>>,
    /// Memory leak detector (optional feature)
    #[cfg(feature = "leak_detection")]
    leak_detector: Arc<RwLock<LeakDetector>>,
}

impl std::fmt::Debug for RaftProfiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaftProfiler")
            .field("node_id", &self.node_id)
            .field("enabled", &"Arc<RwLock<bool>>")
            .field("stats", &"Arc<RwLock<HashMap>>")
            .field("profiler", &"Arc<RwLock<Profiler>>")
            .field("metrics_registry", &"Arc<MetricsRegistry>")
            .field("histograms", &"Arc<RwLock<HashMap>>")
            .field("counters", &"Arc<RwLock<HashMap>>")
            .field("memory_usage", &"Arc<RwLock<HashMap>>")
            .finish()
    }
}

impl RaftProfiler {
    /// Create a new Raft profiler
    pub fn new(node_id: OxirsNodeId) -> Self {
        let profiler = Profiler::new();
        let metrics_registry = Arc::new(MetricsRegistry::new());

        Self {
            node_id,
            stats: Arc::new(RwLock::new(HashMap::new())),
            profiler: Arc::new(RwLock::new(profiler)),
            metrics_registry,
            histograms: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            memory_usage: Arc::new(RwLock::new(HashMap::new())),
            enabled: Arc::new(RwLock::new(true)),
            #[cfg(feature = "leak_detection")]
            leak_detector: Arc::new(RwLock::new(
                LeakDetector::new(LeakDetectionConfig::default())
                    .expect("Failed to initialize leak detector"),
            )),
        }
    }

    /// Enable profiling
    pub async fn enable(&self) {
        let mut enabled = self.enabled.write().await;
        *enabled = true;
        info!("Raft profiling enabled for node {}", self.node_id);
    }

    /// Disable profiling
    pub async fn disable(&self) {
        let mut enabled = self.enabled.write().await;
        *enabled = false;
        info!("Raft profiling disabled for node {}", self.node_id);
    }

    /// Check if profiling is enabled
    pub async fn is_enabled(&self) -> bool {
        *self.enabled.read().await
    }

    /// Start profiling an operation
    pub async fn start_operation(&self, operation: RaftOperation) -> OperationProfiler {
        if !*self.enabled.read().await {
            return OperationProfiler::disabled();
        }

        let label = operation.profiling_label();

        // Initialize histogram and counter if not exists
        {
            let mut histograms = self.histograms.write().await;
            if !histograms.contains_key(&label) {
                histograms.insert(
                    label.clone(),
                    Histogram::new(format!("{}_latency_us", label)),
                );
            }

            let mut counters = self.counters.write().await;
            if !counters.contains_key(&label) {
                counters.insert(label.clone(), Counter::new(format!("{}_count", label)));
            }
        }

        // Start SciRS2-Core profiling
        {
            let mut profiler = self.profiler.write().await;
            profiler.start();
        }

        debug!(
            "Started profiling {} for node {}",
            operation.as_str(),
            self.node_id
        );

        OperationProfiler {
            operation,
            label,
            start_time: Instant::now(),
            stats: Arc::clone(&self.stats),
            profiler: Arc::clone(&self.profiler),
            histograms: Arc::clone(&self.histograms),
            counters: Arc::clone(&self.counters),
            enabled: true,
        }
    }

    /// Record network round-trip time
    pub async fn record_network_roundtrip(&self, target_node: OxirsNodeId, duration: Duration) {
        if !*self.enabled.read().await {
            return;
        }

        let label = format!("network_roundtrip_to_{}", target_node);
        let micros = duration.as_micros() as f64;

        let mut stats = self.stats.write().await;
        let entry = stats.entry(label).or_insert_with(LatencyStats::new);
        entry.record(micros);

        debug!(
            "Network round-trip to node {}: {:.2}ms",
            target_node,
            duration.as_secs_f64() * 1000.0
        );
    }

    /// Record query execution time
    pub async fn record_query_execution(&self, query_id: &str, duration: Duration) {
        if !*self.enabled.read().await {
            return;
        }

        let label = "query_execution".to_string();
        let micros = duration.as_micros() as f64;

        let mut stats = self.stats.write().await;
        let entry = stats.entry(label).or_insert_with(LatencyStats::new);
        entry.record(micros);

        debug!(
            "Query {} execution time: {:.2}ms",
            query_id,
            duration.as_secs_f64() * 1000.0
        );
    }

    /// Record memory usage
    pub async fn record_memory_usage(&self, operation: &str, bytes: u64) {
        if !*self.enabled.read().await {
            return;
        }

        let mut memory = self.memory_usage.write().await;
        memory.insert(operation.to_string(), bytes);
    }

    /// Get profiling metrics for a specific operation
    pub async fn get_metrics(&self, operation: RaftOperation) -> Option<RaftProfilingMetrics> {
        let label = operation.profiling_label();
        let stats = self.stats.read().await;
        let stat = stats.get(&label)?;

        let memory = self.memory_usage.read().await;
        let memory_bytes = memory.get(operation.as_str()).copied().unwrap_or(0);

        Some(RaftProfilingMetrics {
            operation: operation.as_str().to_string(),
            node_id: self.node_id,
            operation_count: stat.count,
            mean_latency_micros: stat.mean(),
            std_dev_micros: stat.std_dev(),
            p50_micros: stat.percentile(50.0),
            p95_micros: stat.percentile(95.0),
            p99_micros: stat.percentile(99.0),
            min_micros: if stat.min_micros == f64::MAX {
                0.0
            } else {
                stat.min_micros
            },
            max_micros: stat.max_micros,
            total_time_ms: stat.sum_micros / 1000.0,
            ops_per_second: if stat.mean() > 0.0 {
                1_000_000.0 / stat.mean()
            } else {
                0.0
            },
            memory_bytes,
            last_update: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Get all profiling metrics
    pub async fn get_all_metrics(&self) -> Vec<RaftProfilingMetrics> {
        let mut all_metrics = Vec::new();

        for operation in [
            RaftOperation::AppendEntries,
            RaftOperation::RequestVote,
            RaftOperation::InstallSnapshot,
            RaftOperation::CreateSnapshot,
            RaftOperation::RestoreSnapshot,
            RaftOperation::LogCompaction,
            RaftOperation::NetworkRoundTrip,
            RaftOperation::QueryExecution,
            RaftOperation::BatchProcessing,
            RaftOperation::LogReplication,
        ] {
            if let Some(metrics) = self.get_metrics(operation).await {
                all_metrics.push(metrics);
            }
        }

        all_metrics
    }

    /// Reset all metrics (useful for testing)
    pub async fn reset(&self) {
        let mut stats = self.stats.write().await;
        stats.clear();

        let mut memory = self.memory_usage.write().await;
        memory.clear();

        info!("Reset all profiling metrics for node {}", self.node_id);
    }

    /// Generate performance report
    pub async fn generate_report(&self) -> String {
        let metrics = self.get_all_metrics().await;

        let mut report = format!("=== Raft Profiling Report (Node {}) ===\n\n", self.node_id);

        for metric in metrics {
            report.push_str(&format!(
                "Operation: {}\n\
                 - Count: {} operations\n\
                 - Mean: {:.2}ms (±{:.2}ms)\n\
                 - Percentiles: p50={:.2}ms, p95={:.2}ms, p99={:.2}ms\n\
                 - Range: [{:.2}ms - {:.2}ms]\n\
                 - Throughput: {:.2} ops/sec\n\
                 - Memory: {} bytes\n\
                 - Last Update: {}\n\n",
                metric.operation,
                metric.operation_count,
                metric.mean_latency_micros / 1000.0,
                metric.std_dev_micros / 1000.0,
                metric.p50_micros / 1000.0,
                metric.p95_micros / 1000.0,
                metric.p99_micros / 1000.0,
                metric.min_micros / 1000.0,
                metric.max_micros / 1000.0,
                metric.ops_per_second,
                metric.memory_bytes,
                metric.last_update,
            ));
        }

        report
    }

    /// Get SciRS2-Core profiler bottleneck analysis
    pub async fn analyze_bottlenecks(&self) -> Vec<(String, f64)> {
        let profiler = self.profiler.read().await;
        let timings = profiler.timings();

        let mut bottlenecks: Vec<(String, f64)> = timings
            .iter()
            .map(|(label, entry)| {
                let duration_ms = entry.average_duration().as_secs_f64() * 1000.0;
                (label.clone(), duration_ms)
            })
            .collect();

        // Sort by duration descending (slowest first)
        bottlenecks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        bottlenecks
    }

    /// Get SciRS2-Core profiler report
    pub async fn get_profiler_report(&self) -> String {
        let profiler = self.profiler.read().await;
        profiler.get_report()
    }

    /// Get histogram statistics for an operation
    pub async fn get_histogram_stats(
        &self,
        operation: RaftOperation,
    ) -> Option<scirs2_core::metrics::HistogramStats> {
        let label = operation.profiling_label();
        let histograms = self.histograms.read().await;
        histograms.get(&label).map(|h| h.get_stats())
    }

    /// Get operation counter
    pub async fn get_operation_count(&self, operation: RaftOperation) -> u64 {
        let label = operation.profiling_label();
        let counters = self.counters.read().await;
        counters.get(&label).map(|c| c.get()).unwrap_or(0)
    }

    /// Check for memory leaks (requires leak_detection feature)
    #[cfg(feature = "leak_detection")]
    pub async fn check_memory_leaks(&self) -> Result<(), String> {
        let detector = self.leak_detector.read().await;
        // Get all reports to check if there are any leaks
        let reports = detector
            .get_reports()
            .map_err(|e| format!("Failed to get leak reports: {:?}", e))?;

        if reports.iter().any(|r| r.has_leaks()) {
            Err(format!("Memory leaks detected: {} reports", reports.len()))
        } else {
            Ok(())
        }
    }

    /// Take a memory snapshot (requires leak_detection feature)
    #[cfg(feature = "leak_detection")]
    pub async fn take_memory_snapshot(&self, label: &str) {
        let detector = self.leak_detector.read().await;
        match detector.create_checkpoint(label) {
            Ok(checkpoint) => {
                info!(
                    "Took memory snapshot: {} ({} bytes)",
                    label, checkpoint.memory_usage.rss_bytes
                );
            }
            Err(e) => {
                warn!("Failed to create memory checkpoint {}: {:?}", label, e);
            }
        }
    }

    /// Export metrics to Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // Export counters
        {
            let counters = self.counters.read().await;
            for (label, counter) in counters.iter() {
                output.push_str(&format!(
                    "# HELP oxirs_cluster_{} Total count of {} operations\n",
                    label, label
                ));
                output.push_str(&format!("# TYPE oxirs_cluster_{} counter\n", label));
                output.push_str(&format!(
                    "oxirs_cluster_{}{{node_id=\"{}\"}} {}\n",
                    label,
                    self.node_id,
                    counter.get()
                ));
            }
        }

        // Export histograms
        {
            let histograms = self.histograms.read().await;
            for (label, histogram) in histograms.iter() {
                let stats = histogram.get_stats();
                output.push_str(&format!(
                    "# HELP oxirs_cluster_{}_latency Latency distribution for {}\n",
                    label, label
                ));
                output.push_str(&format!("# TYPE oxirs_cluster_{}_latency summary\n", label));
                // Note: HistogramStats has count, sum, mean, buckets - not percentiles
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency{{node_id=\"{}\",stat=\"mean\"}} {}\n",
                    label, self.node_id, stats.mean
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_sum{{node_id=\"{}\"}} {}\n",
                    label, self.node_id, stats.sum
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_count{{node_id=\"{}\"}} {}\n",
                    label, self.node_id, stats.count
                ));
            }
        }

        output
    }
}

/// Operation profiler for tracking individual operation execution
pub struct OperationProfiler {
    operation: RaftOperation,
    label: String,
    start_time: Instant,
    stats: Arc<RwLock<HashMap<String, LatencyStats>>>,
    profiler: Arc<RwLock<Profiler>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    enabled: bool,
}

impl OperationProfiler {
    /// Create a disabled profiler (no-op)
    fn disabled() -> Self {
        Self {
            operation: RaftOperation::AppendEntries,
            label: String::new(),
            start_time: Instant::now(),
            stats: Arc::new(RwLock::new(HashMap::new())),
            profiler: Arc::new(RwLock::new(Profiler::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            enabled: false,
        }
    }

    /// Complete the operation and record metrics
    pub async fn complete(self) {
        if !self.enabled {
            return;
        }

        let duration = self.start_time.elapsed();
        let micros = duration.as_micros() as f64;

        // Update statistics
        let mut stats = self.stats.write().await;
        let entry = stats
            .entry(self.label.clone())
            .or_insert_with(LatencyStats::new);
        entry.record(micros);

        // Update SciRS2-Core metrics
        {
            let mut profiler = self.profiler.write().await;
            profiler.stop();
        }

        {
            let histograms = self.histograms.read().await;
            if let Some(histogram) = histograms.get(&self.label) {
                histogram.observe(micros);
            }
        }

        {
            let counters = self.counters.read().await;
            if let Some(counter) = counters.get(&self.label) {
                counter.inc();
            }
        }

        debug!(
            "Completed {} in {:.2}ms",
            self.operation.as_str(),
            duration.as_secs_f64() * 1000.0
        );
    }

    /// Complete with error
    pub async fn complete_with_error(self, error: &str) {
        if !self.enabled {
            return;
        }

        let duration = self.start_time.elapsed();
        warn!(
            "Operation {} failed after {:.2}ms: {}",
            self.operation.as_str(),
            duration.as_secs_f64() * 1000.0,
            error
        );

        // Still record the time
        self.complete().await;
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline_metrics: HashMap<String, RaftProfilingMetrics>,
    threshold_percentage: f64,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(threshold_percentage: f64) -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            threshold_percentage,
        }
    }

    /// Set baseline metrics
    pub fn set_baseline(&mut self, metrics: Vec<RaftProfilingMetrics>) {
        self.baseline_metrics.clear();
        for metric in metrics {
            self.baseline_metrics
                .insert(metric.operation.clone(), metric);
        }
    }

    /// Detect regressions compared to baseline
    pub fn detect_regressions(
        &self,
        current_metrics: Vec<RaftProfilingMetrics>,
    ) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();

        for current in current_metrics {
            if let Some(baseline) = self.baseline_metrics.get(&current.operation) {
                // Check mean latency regression
                let mean_change_pct = ((current.mean_latency_micros
                    - baseline.mean_latency_micros)
                    / baseline.mean_latency_micros)
                    * 100.0;

                if mean_change_pct > self.threshold_percentage {
                    regressions.push(PerformanceRegression {
                        operation: current.operation.clone(),
                        metric_name: "mean_latency".to_string(),
                        baseline_value: baseline.mean_latency_micros,
                        current_value: current.mean_latency_micros,
                        change_percentage: mean_change_pct,
                        severity: if mean_change_pct > 50.0 {
                            RegressionSeverity::Critical
                        } else if mean_change_pct > 25.0 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                    });
                }

                // Check p99 latency regression
                let p99_change_pct =
                    ((current.p99_micros - baseline.p99_micros) / baseline.p99_micros) * 100.0;

                if p99_change_pct > self.threshold_percentage {
                    regressions.push(PerformanceRegression {
                        operation: current.operation.clone(),
                        metric_name: "p99_latency".to_string(),
                        baseline_value: baseline.p99_micros,
                        current_value: current.p99_micros,
                        change_percentage: p99_change_pct,
                        severity: if p99_change_pct > 50.0 {
                            RegressionSeverity::Critical
                        } else if p99_change_pct > 25.0 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                    });
                }
            }
        }

        regressions
    }
}

/// Performance regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub operation: String,
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_percentage: f64,
    pub severity: RegressionSeverity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Resource utilization gauges for Raft operations
///
/// Tracks real-time resource metrics using SciRS2-Core Gauge
#[allow(dead_code)]
pub struct ResourceGauges {
    /// Active append_entries operations
    pub active_append_entries: Gauge,
    /// Active request_vote operations
    pub active_request_votes: Gauge,
    /// Active snapshot operations
    pub active_snapshots: Gauge,
    /// Queue size for pending operations
    pub pending_operations_queue: Gauge,
    /// Memory usage in bytes
    pub memory_usage_bytes: Gauge,
    /// Network connections count
    pub active_connections: Gauge,
    /// Log entries count
    pub log_entries_count: Gauge,
    /// Committed entries count
    pub committed_entries_count: Gauge,
    /// Metrics registry
    registry: Arc<MetricsRegistry>,
}

impl ResourceGauges {
    /// Create new resource gauges
    pub fn new(node_id: OxirsNodeId) -> Self {
        let registry = Arc::new(MetricsRegistry::new());

        Self {
            active_append_entries: Gauge::new(format!("node_{}_active_append_entries", node_id)),
            active_request_votes: Gauge::new(format!("node_{}_active_request_votes", node_id)),
            active_snapshots: Gauge::new(format!("node_{}_active_snapshots", node_id)),
            pending_operations_queue: Gauge::new(format!(
                "node_{}_pending_operations_queue",
                node_id
            )),
            memory_usage_bytes: Gauge::new(format!("node_{}_memory_usage_bytes", node_id)),
            active_connections: Gauge::new(format!("node_{}_active_connections", node_id)),
            log_entries_count: Gauge::new(format!("node_{}_log_entries_count", node_id)),
            committed_entries_count: Gauge::new(format!(
                "node_{}_committed_entries_count",
                node_id
            )),
            registry,
        }
    }

    /// Increment active append_entries count
    pub fn inc_append_entries(&self) {
        self.active_append_entries.inc();
    }

    /// Decrement active append_entries count
    pub fn dec_append_entries(&self) {
        self.active_append_entries.dec();
    }

    /// Increment active request_vote count
    pub fn inc_request_votes(&self) {
        self.active_request_votes.inc();
    }

    /// Decrement active request_vote count
    pub fn dec_request_votes(&self) {
        self.active_request_votes.dec();
    }

    /// Increment active snapshot count
    pub fn inc_snapshots(&self) {
        self.active_snapshots.inc();
    }

    /// Decrement active snapshot count
    pub fn dec_snapshots(&self) {
        self.active_snapshots.dec();
    }

    /// Set pending operations queue size
    pub fn set_queue_size(&self, size: usize) {
        self.pending_operations_queue.set(size as f64);
    }

    /// Set memory usage in bytes
    pub fn set_memory_usage(&self, bytes: u64) {
        self.memory_usage_bytes.set(bytes as f64);
    }

    /// Increment active connections
    pub fn inc_connections(&self) {
        self.active_connections.inc();
    }

    /// Decrement active connections
    pub fn dec_connections(&self) {
        self.active_connections.dec();
    }

    /// Set log entries count
    pub fn set_log_entries(&self, count: u64) {
        self.log_entries_count.set(count as f64);
    }

    /// Set committed entries count
    pub fn set_committed_entries(&self, count: u64) {
        self.committed_entries_count.set(count as f64);
    }

    /// Get current gauge values as a summary
    pub fn get_summary(&self) -> ResourceGaugeSummary {
        ResourceGaugeSummary {
            active_append_entries: self.active_append_entries.get() as u64,
            active_request_votes: self.active_request_votes.get() as u64,
            active_snapshots: self.active_snapshots.get() as u64,
            pending_operations_queue: self.pending_operations_queue.get() as u64,
            memory_usage_bytes: self.memory_usage_bytes.get() as u64,
            active_connections: self.active_connections.get() as u64,
            log_entries_count: self.log_entries_count.get() as u64,
            committed_entries_count: self.committed_entries_count.get() as u64,
        }
    }
}

/// Resource gauge values summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceGaugeSummary {
    pub active_append_entries: u64,
    pub active_request_votes: u64,
    pub active_snapshots: u64,
    pub pending_operations_queue: u64,
    pub memory_usage_bytes: u64,
    pub active_connections: u64,
    pub log_entries_count: u64,
    pub committed_entries_count: u64,
}

/// Automatic timer for Raft operations
///
/// Uses SciRS2-Core Timer for precise timing measurements
#[allow(dead_code)]
pub struct MetricsTimer {
    timer: Timer,
    operation_label: String,
    start_time: Instant,
}

impl MetricsTimer {
    /// Create a new metrics timer
    pub fn new(operation: RaftOperation) -> Self {
        let label = operation.profiling_label();
        let timer = Timer::new(format!("{}_duration_ms", label));

        Self {
            timer,
            operation_label: label,
            start_time: Instant::now(),
        }
    }

    /// Stop the timer and record the elapsed time
    pub fn stop(self) -> Duration {
        let elapsed = self.start_time.elapsed();
        // Note: Timer automatically records on drop
        debug!(
            "Operation {} took {} ms",
            self.operation_label,
            elapsed.as_millis()
        );
        elapsed
    }

    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Enhanced metrics export for Prometheus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMetricsExport {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Timestamp
    pub timestamp: String,
    /// Latency metrics per operation
    pub latency_metrics: HashMap<String, RaftProfilingMetrics>,
    /// Resource gauge values
    pub resource_gauges: ResourceGaugeSummary,
    /// Counter values
    pub counter_values: HashMap<String, u64>,
    /// Histogram summaries
    pub histogram_summaries: HashMap<String, HistogramSummary>,
}

/// Histogram summary for Prometheus export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSummary {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub bucket_count: usize,
}

impl RaftProfiler {
    /// Create resource gauges for this profiler
    pub fn create_resource_gauges(&self) -> ResourceGauges {
        ResourceGauges::new(self.node_id)
    }

    /// Export enhanced metrics for Prometheus
    pub async fn export_enhanced_metrics(&self, gauges: &ResourceGauges) -> EnhancedMetricsExport {
        let mut latency_metrics = HashMap::new();
        let mut counter_values = HashMap::new();
        let mut histogram_summaries = HashMap::new();

        // Collect all operation metrics
        for operation in [
            RaftOperation::AppendEntries,
            RaftOperation::RequestVote,
            RaftOperation::InstallSnapshot,
            RaftOperation::CreateSnapshot,
            RaftOperation::RestoreSnapshot,
            RaftOperation::LogCompaction,
            RaftOperation::NetworkRoundTrip,
            RaftOperation::QueryExecution,
            RaftOperation::BatchProcessing,
            RaftOperation::LogReplication,
        ] {
            if let Some(metrics) = self.get_metrics(operation).await {
                latency_metrics.insert(operation.as_str().to_string(), metrics);
            }

            // Get counter values
            let label = operation.profiling_label();
            let counters = self.counters.read().await;
            if let Some(counter) = counters.get(&label) {
                counter_values.insert(label.clone(), counter.get());
            }

            // Get histogram summaries
            if let Some(hist_stats) = self.get_histogram_stats(operation).await {
                histogram_summaries.insert(
                    label,
                    HistogramSummary {
                        count: hist_stats.count,
                        sum: hist_stats.sum,
                        mean: hist_stats.mean,
                        bucket_count: hist_stats.buckets.len(),
                    },
                );
            }
        }

        EnhancedMetricsExport {
            node_id: self.node_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            latency_metrics,
            resource_gauges: gauges.get_summary(),
            counter_values,
            histogram_summaries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_raft_profiler_creation() {
        let profiler = RaftProfiler::new(1);
        assert!(profiler.is_enabled().await);
    }

    #[tokio::test]
    async fn test_operation_profiling() {
        let profiler = RaftProfiler::new(1);

        // Start and complete an operation
        let op = profiler.start_operation(RaftOperation::AppendEntries).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        op.complete().await;

        // Get metrics
        let metrics = profiler.get_metrics(RaftOperation::AppendEntries).await;
        assert!(metrics.is_some());

        let metrics = metrics.unwrap();
        assert_eq!(metrics.operation_count, 1);
        assert!(metrics.mean_latency_micros >= 10_000.0); // At least 10ms
    }

    #[tokio::test]
    async fn test_network_roundtrip_recording() {
        let profiler = RaftProfiler::new(1);

        profiler
            .record_network_roundtrip(2, Duration::from_millis(5))
            .await;

        // Verify stats were updated
        let stats = profiler.stats.read().await;
        assert!(stats.contains_key("network_roundtrip_to_2"));
    }

    #[tokio::test]
    async fn test_query_execution_recording() {
        let profiler = RaftProfiler::new(1);

        profiler
            .record_query_execution("test_query_1", Duration::from_millis(20))
            .await;

        // Verify stats were updated
        let stats = profiler.stats.read().await;
        assert!(stats.contains_key("query_execution"));
    }

    #[tokio::test]
    async fn test_memory_tracking() {
        let profiler = RaftProfiler::new(1);

        profiler.record_memory_usage("snapshot", 1024 * 1024).await;

        // Verify memory was tracked
        let memory = profiler.memory_usage.read().await;
        assert_eq!(memory.get("snapshot"), Some(&(1024 * 1024)));
    }

    #[tokio::test]
    async fn test_enable_disable() {
        let profiler = RaftProfiler::new(1);

        assert!(profiler.is_enabled().await);

        profiler.disable().await;
        assert!(!profiler.is_enabled().await);

        profiler.enable().await;
        assert!(profiler.is_enabled().await);
    }

    #[tokio::test]
    async fn test_regression_detection() {
        let mut detector = RegressionDetector::new(10.0);

        let baseline = vec![RaftProfilingMetrics {
            operation: "append_entries".to_string(),
            node_id: 1,
            operation_count: 100,
            mean_latency_micros: 1000.0,
            std_dev_micros: 100.0,
            p50_micros: 950.0,
            p95_micros: 1500.0,
            p99_micros: 2000.0,
            min_micros: 500.0,
            max_micros: 3000.0,
            total_time_ms: 100.0,
            ops_per_second: 1000.0,
            memory_bytes: 1024,
            last_update: chrono::Utc::now().to_rfc3339(),
        }];

        detector.set_baseline(baseline);

        // Create current metrics with regression
        let current = vec![RaftProfilingMetrics {
            operation: "append_entries".to_string(),
            node_id: 1,
            operation_count: 100,
            mean_latency_micros: 1500.0, // 50% slower
            std_dev_micros: 150.0,
            p50_micros: 1400.0,
            p95_micros: 2000.0,
            p99_micros: 3000.0, // 50% slower
            min_micros: 700.0,
            max_micros: 4000.0,
            total_time_ms: 150.0,
            ops_per_second: 666.0,
            memory_bytes: 2048,
            last_update: chrono::Utc::now().to_rfc3339(),
        }];

        let regressions = detector.detect_regressions(current);
        assert!(!regressions.is_empty());
        assert!(regressions.iter().any(|r| r.metric_name == "mean_latency"));
    }

    #[tokio::test]
    async fn test_report_generation() {
        let profiler = RaftProfiler::new(1);

        // Record some operations
        for _ in 0..5 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            op.complete().await;
        }

        let report = profiler.generate_report().await;
        assert!(report.contains("Raft Profiling Report"));
        assert!(report.contains("append_entries"));
    }

    #[tokio::test]
    async fn test_multiple_operations() {
        let profiler = RaftProfiler::new(1);

        // Test different operations
        let ops = vec![
            RaftOperation::AppendEntries,
            RaftOperation::CreateSnapshot,
            RaftOperation::QueryExecution,
        ];

        for op in ops {
            let profiler_op = profiler.start_operation(op).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            profiler_op.complete().await;
        }

        let all_metrics = profiler.get_all_metrics().await;
        assert!(all_metrics.len() >= 3);
    }

    #[tokio::test]
    async fn test_resource_gauges_creation() {
        let gauges = ResourceGauges::new(1);
        let summary = gauges.get_summary();

        assert_eq!(summary.active_append_entries, 0);
        assert_eq!(summary.active_request_votes, 0);
        assert_eq!(summary.active_snapshots, 0);
        assert_eq!(summary.pending_operations_queue, 0);
        assert_eq!(summary.memory_usage_bytes, 0);
        assert_eq!(summary.active_connections, 0);
    }

    #[tokio::test]
    async fn test_resource_gauges_increment_decrement() {
        let gauges = ResourceGauges::new(1);

        // Test append_entries
        gauges.inc_append_entries();
        gauges.inc_append_entries();
        assert_eq!(gauges.get_summary().active_append_entries, 2);
        gauges.dec_append_entries();
        assert_eq!(gauges.get_summary().active_append_entries, 1);

        // Test request_votes
        gauges.inc_request_votes();
        assert_eq!(gauges.get_summary().active_request_votes, 1);
        gauges.dec_request_votes();
        assert_eq!(gauges.get_summary().active_request_votes, 0);

        // Test snapshots
        gauges.inc_snapshots();
        gauges.inc_snapshots();
        gauges.inc_snapshots();
        assert_eq!(gauges.get_summary().active_snapshots, 3);
        gauges.dec_snapshots();
        assert_eq!(gauges.get_summary().active_snapshots, 2);
    }

    #[tokio::test]
    async fn test_resource_gauges_setters() {
        let gauges = ResourceGauges::new(1);

        gauges.set_queue_size(42);
        gauges.set_memory_usage(1024);
        gauges.set_log_entries(1000);
        gauges.set_committed_entries(950);

        let summary = gauges.get_summary();
        assert_eq!(summary.pending_operations_queue, 42);
        assert_eq!(summary.memory_usage_bytes, 1024);
        assert_eq!(summary.log_entries_count, 1000);
        assert_eq!(summary.committed_entries_count, 950);
    }

    #[tokio::test]
    async fn test_resource_gauges_connections() {
        let gauges = ResourceGauges::new(1);

        gauges.inc_connections();
        gauges.inc_connections();
        gauges.inc_connections();
        assert_eq!(gauges.get_summary().active_connections, 3);

        gauges.dec_connections();
        assert_eq!(gauges.get_summary().active_connections, 2);
    }

    #[tokio::test]
    async fn test_metrics_timer_creation() {
        let timer = MetricsTimer::new(RaftOperation::AppendEntries);

        // Let some time pass
        tokio::time::sleep(Duration::from_millis(10)).await;

        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[tokio::test]
    async fn test_metrics_timer_stop() {
        let timer = MetricsTimer::new(RaftOperation::RequestVote);

        // Let some time pass
        tokio::time::sleep(Duration::from_millis(5)).await;

        let elapsed = timer.stop();
        assert!(elapsed.as_millis() >= 5);
    }

    #[tokio::test]
    async fn test_enhanced_metrics_export() {
        let profiler = RaftProfiler::new(1);
        let gauges = profiler.create_resource_gauges();

        // Set up some gauge values
        gauges.inc_append_entries();
        gauges.set_queue_size(10);
        gauges.set_memory_usage(2048);

        // Record some operations
        let op = profiler.start_operation(RaftOperation::AppendEntries).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        op.complete().await;

        // Export metrics
        let export = profiler.export_enhanced_metrics(&gauges).await;

        assert_eq!(export.node_id, 1);
        assert_eq!(export.resource_gauges.active_append_entries, 1);
        assert_eq!(export.resource_gauges.pending_operations_queue, 10);
        assert_eq!(export.resource_gauges.memory_usage_bytes, 2048);
        assert!(!export.timestamp.is_empty());
    }

    #[tokio::test]
    async fn test_enhanced_metrics_export_multiple_operations() {
        let profiler = RaftProfiler::new(1);
        let gauges = profiler.create_resource_gauges();

        // Record different operations
        for operation in [
            RaftOperation::AppendEntries,
            RaftOperation::RequestVote,
            RaftOperation::CreateSnapshot,
        ] {
            let op = profiler.start_operation(operation).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            op.complete().await;
        }

        // Export metrics
        let export = profiler.export_enhanced_metrics(&gauges).await;

        // Should have metrics for all operations
        assert!(export.latency_metrics.contains_key("append_entries"));
        assert!(export.latency_metrics.contains_key("request_vote"));
        assert!(export.latency_metrics.contains_key("create_snapshot"));

        // Should have counter values
        assert!(export.counter_values.len() >= 3);

        // Should have histogram summaries
        assert!(export.histogram_summaries.len() >= 3);
    }

    #[tokio::test]
    async fn test_histogram_summary_fields() {
        let profiler = RaftProfiler::new(1);
        let gauges = profiler.create_resource_gauges();

        // Record multiple operations to build histogram data
        for _ in 0..10 {
            let op = profiler.start_operation(RaftOperation::AppendEntries).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            op.complete().await;
        }

        let export = profiler.export_enhanced_metrics(&gauges).await;

        let hist_summary = export
            .histogram_summaries
            .get("raft_append_entries")
            .expect("Should have append_entries histogram");

        assert_eq!(hist_summary.count, 10);
        assert!(hist_summary.sum > 0.0);
        assert!(hist_summary.mean > 0.0);
        assert!(hist_summary.bucket_count > 0);
    }

    #[tokio::test]
    async fn test_profiler_create_resource_gauges() {
        let profiler = RaftProfiler::new(42);
        let gauges = profiler.create_resource_gauges();

        // Gauges should be created for the correct node ID
        let summary = gauges.get_summary();
        assert_eq!(summary.active_append_entries, 0);
    }

    #[tokio::test]
    async fn test_resource_gauge_summary_serialization() {
        use serde_json;

        let summary = ResourceGaugeSummary {
            active_append_entries: 5,
            active_request_votes: 3,
            active_snapshots: 2,
            pending_operations_queue: 10,
            memory_usage_bytes: 4096,
            active_connections: 7,
            log_entries_count: 1000,
            committed_entries_count: 950,
        };

        // Should be serializable
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("active_append_entries"));
        assert!(json.contains("5"));

        // Should be deserializable
        let deserialized: ResourceGaugeSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.active_append_entries, 5);
        assert_eq!(deserialized.memory_usage_bytes, 4096);
    }
}
