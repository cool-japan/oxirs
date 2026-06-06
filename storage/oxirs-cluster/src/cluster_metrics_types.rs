//! Supporting types for cluster metrics: operation timer, result records, and reports.
//!
//! This module defines:
//! - [`OperationTimer`]: an RAII-style timer that records latency, histogram, and
//!   counter samples when an operation completes
//! - [`OperationMetrics`]: a comprehensive snapshot of an operation's statistics
//! - [`BenchmarkResultRecord`] and [`BenchmarkComparison`]: benchmarking outputs
//! - [`OperationBaseline`], [`PerformanceRegression`], and [`RegressionSeverity`]:
//!   regression-detection types

use crate::cluster_metrics_stats::{ClusterOperation, EnhancedLatencyStats};
use scirs2_core::metrics::{Counter, Histogram};
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Stored benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResultRecord {
    /// Benchmark name
    pub name: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Mean time in nanoseconds
    pub mean_ns: f64,
    /// Standard deviation
    pub std_dev_ns: f64,
    /// Iterations
    pub iterations: u64,
    /// Throughput (ops/sec)
    pub throughput: f64,
}

/// Operation baseline for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationBaseline {
    /// Operation name
    pub operation: String,
    /// Baseline mean latency
    pub mean_micros: f64,
    /// Baseline standard deviation
    pub std_dev_micros: f64,
    /// Baseline p50
    pub p50_micros: f64,
    /// Baseline p95
    pub p95_micros: f64,
    /// Baseline p99
    pub p99_micros: f64,
    /// Sample size used for baseline
    pub sample_size: usize,
    /// Timestamp when baseline was established
    pub timestamp: SystemTime,
}

/// Timer for tracking operation duration
pub struct OperationTimer {
    pub(crate) operation: ClusterOperation,
    pub(crate) label: String,
    pub(crate) start_time: Instant,
    pub(crate) latency_stats: Arc<RwLock<HashMap<String, EnhancedLatencyStats>>>,
    pub(crate) profiler: Arc<RwLock<Profiler>>,
    pub(crate) histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    pub(crate) counters: Arc<RwLock<HashMap<String, Counter>>>,
    pub(crate) enabled: bool,
}

impl OperationTimer {
    /// Create a disabled timer
    pub(crate) fn disabled() -> Self {
        Self {
            operation: ClusterOperation::AppendEntries,
            label: String::new(),
            start_time: Instant::now(),
            latency_stats: Arc::new(RwLock::new(HashMap::new())),
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

        // Update enhanced stats
        {
            let mut stats = self.latency_stats.write().await;
            if let Some(stat) = stats.get_mut(&self.label) {
                stat.record(micros);
            }
        }

        // Update SciRS2 metrics
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

        self.complete().await;
    }

    /// Get elapsed time without completing
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Comprehensive operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Operation name
    pub operation: String,
    /// Node ID
    pub node_id: u64,
    /// Total count
    pub count: u64,
    /// Mean latency in microseconds
    pub mean_micros: f64,
    /// Standard deviation
    pub std_dev_micros: f64,
    /// Variance
    pub variance_micros: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// 50th percentile
    pub p50_micros: f64,
    /// 75th percentile
    pub p75_micros: f64,
    /// 90th percentile
    pub p90_micros: f64,
    /// 95th percentile
    pub p95_micros: f64,
    /// 99th percentile
    pub p99_micros: f64,
    /// 99.9th percentile
    pub p999_micros: f64,
    /// Minimum
    pub min_micros: f64,
    /// Maximum
    pub max_micros: f64,
    /// Interquartile range
    pub iqr_micros: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Trend (slope)
    pub trend: f64,
    /// Exponential moving average
    pub ema_micros: f64,
    /// Operations per second
    pub rate_ops_per_sec: f64,
    /// Timestamp
    pub timestamp: String,
}

/// Performance regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Operation name
    pub operation: String,
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Change percentage
    pub change_percentage: f64,
    /// P-value from statistical test
    pub p_value: f64,
    /// T-statistic
    pub t_statistic: f64,
    /// Severity level
    pub severity: RegressionSeverity,
    /// Detection method used
    pub detection_method: String,
}

/// Regression severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RegressionSeverity {
    /// Minor regression (<25%)
    Low,
    /// Moderate regression (25-50%)
    Medium,
    /// Significant regression (50-100%)
    High,
    /// Critical regression (>100%)
    Critical,
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Baseline benchmark name
    pub baseline_name: String,
    /// Current benchmark name
    pub current_name: String,
    /// Baseline mean time
    pub baseline_mean_ns: f64,
    /// Current mean time
    pub current_mean_ns: f64,
    /// Speedup factor
    pub speedup: f64,
    /// Throughput improvement percentage
    pub throughput_improvement: f64,
    /// Whether performance improved
    pub is_improved: bool,
}
