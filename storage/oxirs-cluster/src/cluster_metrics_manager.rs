//! The comprehensive cluster metrics manager.
//!
//! This module implements [`ClusterMetricsManager`], which coordinates SciRS2-Core
//! metric primitives (counters, gauges, histograms, timers, profiler) together with
//! [`EnhancedLatencyStats`] per operation. It provides timing, gauge/counter helpers,
//! Prometheus export, report generation, baseline establishment, statistical
//! regression detection, and a synthetic benchmarking suite.

use crate::cluster_metrics_stats::{ClusterOperation, EnhancedLatencyStats};
use crate::cluster_metrics_types::{
    BenchmarkComparison, BenchmarkResultRecord, OperationBaseline, OperationMetrics,
    OperationTimer, PerformanceRegression, RegressionSeverity,
};
use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry, Timer};
use scirs2_core::profiling::Profiler;
use scirs2_stats::distributions::StudentT;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Comprehensive cluster metrics manager
pub struct ClusterMetricsManager {
    /// Node ID
    node_id: u64,
    /// SciRS2-Core metrics registry (reserved for future use)
    #[allow(dead_code)]
    registry: Arc<MetricsRegistry>,
    /// SciRS2-Core profiler
    profiler: Arc<RwLock<Profiler>>,
    /// Enhanced latency statistics per operation
    latency_stats: Arc<RwLock<HashMap<String, EnhancedLatencyStats>>>,
    /// SciRS2-Core histograms
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    /// SciRS2-Core counters
    counters: Arc<RwLock<HashMap<String, Counter>>>,
    /// SciRS2-Core gauges
    gauges: Arc<RwLock<HashMap<String, Gauge>>>,
    /// SciRS2-Core timers (reserved for future use)
    #[allow(dead_code)]
    timers: Arc<RwLock<HashMap<String, Timer>>>,
    /// Enabled flag
    enabled: Arc<RwLock<bool>>,
    /// Rolling window size for statistics
    window_size: usize,
    /// Benchmark results storage
    benchmark_results: Arc<RwLock<Vec<BenchmarkResultRecord>>>,
    /// Regression baselines
    baselines: Arc<RwLock<HashMap<String, OperationBaseline>>>,
}

impl ClusterMetricsManager {
    /// Create a new cluster metrics manager
    pub fn new(node_id: u64, window_size: usize) -> Self {
        let registry = Arc::new(MetricsRegistry::new());
        let profiler = Arc::new(RwLock::new(Profiler::new()));

        Self {
            node_id,
            registry,
            profiler,
            latency_stats: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
            enabled: Arc::new(RwLock::new(true)),
            window_size,
            benchmark_results: Arc::new(RwLock::new(Vec::new())),
            baselines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enable metrics collection
    pub async fn enable(&self) {
        let mut enabled = self.enabled.write().await;
        *enabled = true;
        info!("Cluster metrics enabled for node {}", self.node_id);
    }

    /// Disable metrics collection
    pub async fn disable(&self) {
        let mut enabled = self.enabled.write().await;
        *enabled = false;
        info!("Cluster metrics disabled for node {}", self.node_id);
    }

    /// Check if metrics are enabled
    pub async fn is_enabled(&self) -> bool {
        *self.enabled.read().await
    }

    /// Start timing an operation
    pub async fn start_operation(&self, operation: ClusterOperation) -> OperationTimer {
        if !*self.enabled.read().await {
            return OperationTimer::disabled();
        }

        let label = operation.as_str().to_string();

        // Initialize metrics if not exists
        {
            let mut stats = self.latency_stats.write().await;
            if !stats.contains_key(&label) {
                stats.insert(label.clone(), EnhancedLatencyStats::new(self.window_size));
            }

            let mut histograms = self.histograms.write().await;
            if !histograms.contains_key(&label) {
                histograms.insert(
                    label.clone(),
                    Histogram::new(format!("cluster_{}_latency_us", label)),
                );
            }

            let mut counters = self.counters.write().await;
            if !counters.contains_key(&label) {
                counters.insert(
                    label.clone(),
                    Counter::new(format!("cluster_{}_count", label)),
                );
            }
        }

        // Start profiling
        {
            let mut profiler = self.profiler.write().await;
            profiler.start();
        }

        debug!(
            "Started timing {} for node {}",
            operation.as_str(),
            self.node_id
        );

        OperationTimer {
            operation,
            label,
            start_time: Instant::now(),
            latency_stats: Arc::clone(&self.latency_stats),
            profiler: Arc::clone(&self.profiler),
            histograms: Arc::clone(&self.histograms),
            counters: Arc::clone(&self.counters),
            enabled: true,
        }
    }

    /// Set a gauge value
    pub async fn set_gauge(&self, name: &str, value: f64) {
        if !*self.enabled.read().await {
            return;
        }

        let mut gauges = self.gauges.write().await;
        let gauge = gauges
            .entry(name.to_string())
            .or_insert_with(|| Gauge::new(format!("cluster_{}", name)));
        gauge.set(value);
    }

    /// Increment a gauge
    pub async fn inc_gauge(&self, name: &str) {
        if !*self.enabled.read().await {
            return;
        }

        let mut gauges = self.gauges.write().await;
        let gauge = gauges
            .entry(name.to_string())
            .or_insert_with(|| Gauge::new(format!("cluster_{}", name)));
        gauge.inc();
    }

    /// Decrement a gauge
    pub async fn dec_gauge(&self, name: &str) {
        if !*self.enabled.read().await {
            return;
        }

        let mut gauges = self.gauges.write().await;
        let gauge = gauges
            .entry(name.to_string())
            .or_insert_with(|| Gauge::new(format!("cluster_{}", name)));
        gauge.dec();
    }

    /// Get gauge value
    pub async fn get_gauge(&self, name: &str) -> f64 {
        let gauges = self.gauges.read().await;
        gauges.get(name).map(|g| g.get()).unwrap_or(0.0)
    }

    /// Increment a counter
    pub async fn inc_counter(&self, name: &str) {
        if !*self.enabled.read().await {
            return;
        }

        let mut counters = self.counters.write().await;
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(|| Counter::new(format!("cluster_{}", name)));
        counter.inc();
    }

    /// Increment counter by value
    pub async fn inc_counter_by(&self, name: &str, value: u64) {
        if !*self.enabled.read().await {
            return;
        }

        let mut counters = self.counters.write().await;
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(|| Counter::new(format!("cluster_{}", name)));
        counter.add(value);
    }

    /// Get counter value
    pub async fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.read().await;
        counters.get(name).map(|c| c.get()).unwrap_or(0)
    }

    /// Get operation metrics
    pub async fn get_operation_metrics(
        &self,
        operation: ClusterOperation,
    ) -> Option<OperationMetrics> {
        let label = operation.as_str().to_string();
        let stats = self.latency_stats.read().await;
        let stat = stats.get(&label)?;

        let counters = self.counters.read().await;
        let count = counters.get(&label).map(|c| c.get()).unwrap_or(0);

        Some(OperationMetrics {
            operation: operation.as_str().to_string(),
            node_id: self.node_id,
            count,
            mean_micros: stat.mean(),
            std_dev_micros: stat.std_dev(),
            variance_micros: stat.variance(),
            cv: stat.coefficient_of_variation(),
            p50_micros: stat.percentile(50.0),
            p75_micros: stat.percentile(75.0),
            p90_micros: stat.percentile(90.0),
            p95_micros: stat.percentile(95.0),
            p99_micros: stat.percentile(99.0),
            p999_micros: stat.percentile(99.9),
            min_micros: if stat.min == f64::MAX { 0.0 } else { stat.min },
            max_micros: stat.max,
            iqr_micros: stat.iqr(),
            skewness: stat.skewness(),
            kurtosis: stat.kurtosis(),
            trend: stat.trend(),
            ema_micros: stat.ema(),
            rate_ops_per_sec: stat.rate(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Get all operation metrics
    pub async fn get_all_metrics(&self) -> Vec<OperationMetrics> {
        let mut all_metrics = Vec::new();

        for operation in ClusterOperation::all() {
            if let Some(metrics) = self.get_operation_metrics(operation).await {
                all_metrics.push(metrics);
            }
        }

        all_metrics
    }

    /// Reset all metrics
    pub async fn reset(&self) {
        let mut stats = self.latency_stats.write().await;
        stats.clear();

        let mut counters = self.counters.write().await;
        counters.clear();

        let mut gauges = self.gauges.write().await;
        gauges.clear();

        let mut histograms = self.histograms.write().await;
        histograms.clear();

        info!("Reset all cluster metrics for node {}", self.node_id);
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // Export counters
        {
            let counters = self.counters.read().await;
            for (label, counter) in counters.iter() {
                output.push_str(&format!(
                    "# HELP oxirs_cluster_{}_total Total count of {} operations\n",
                    label, label
                ));
                output.push_str(&format!("# TYPE oxirs_cluster_{}_total counter\n", label));
                output.push_str(&format!(
                    "oxirs_cluster_{}_total{{node_id=\"{}\"}} {}\n",
                    label,
                    self.node_id,
                    counter.get()
                ));
            }
        }

        // Export gauges
        {
            let gauges = self.gauges.read().await;
            for (label, gauge) in gauges.iter() {
                output.push_str(&format!(
                    "# HELP oxirs_cluster_{} Current value of {}\n",
                    label, label
                ));
                output.push_str(&format!("# TYPE oxirs_cluster_{} gauge\n", label));
                output.push_str(&format!(
                    "oxirs_cluster_{}{{node_id=\"{}\"}} {}\n",
                    label,
                    self.node_id,
                    gauge.get()
                ));
            }
        }

        // Export latency statistics
        {
            let stats = self.latency_stats.read().await;
            for (label, stat) in stats.iter() {
                output.push_str(&format!(
                    "# HELP oxirs_cluster_{}_latency_us Latency in microseconds\n",
                    label
                ));
                output.push_str(&format!(
                    "# TYPE oxirs_cluster_{}_latency_us summary\n",
                    label
                ));

                // Quantiles
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us{{node_id=\"{}\",quantile=\"0.5\"}} {}\n",
                    label,
                    self.node_id,
                    stat.percentile(50.0)
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us{{node_id=\"{}\",quantile=\"0.9\"}} {}\n",
                    label,
                    self.node_id,
                    stat.percentile(90.0)
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us{{node_id=\"{}\",quantile=\"0.95\"}} {}\n",
                    label,
                    self.node_id,
                    stat.percentile(95.0)
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us{{node_id=\"{}\",quantile=\"0.99\"}} {}\n",
                    label,
                    self.node_id,
                    stat.percentile(99.0)
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us_sum{{node_id=\"{}\"}} {}\n",
                    label, self.node_id, stat.sum
                ));
                output.push_str(&format!(
                    "oxirs_cluster_{}_latency_us_count{{node_id=\"{}\"}} {}\n",
                    label, self.node_id, stat.count
                ));
            }
        }

        output
    }

    /// Generate comprehensive metrics report
    pub async fn generate_report(&self) -> String {
        let metrics = self.get_all_metrics().await;

        let mut report = format!("=== Cluster Metrics Report (Node {}) ===\n", self.node_id);
        report.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Utc::now().to_rfc3339()
        ));

        for metric in metrics {
            report.push_str(&format!(
                "Operation: {}\n\
                 - Count: {} operations\n\
                 - Mean: {:.2}ms (±{:.2}ms, CV={:.2}%)\n\
                 - Percentiles: p50={:.2}ms, p90={:.2}ms, p95={:.2}ms, p99={:.2}ms\n\
                 - Range: [{:.2}ms - {:.2}ms], IQR={:.2}ms\n\
                 - Distribution: skewness={:.3}, kurtosis={:.3}\n\
                 - Trend: {:.4}μs/sample, EMA={:.2}ms\n\
                 - Throughput: {:.2} ops/sec\n\n",
                metric.operation,
                metric.count,
                metric.mean_micros / 1000.0,
                metric.std_dev_micros / 1000.0,
                metric.cv * 100.0,
                metric.p50_micros / 1000.0,
                metric.p90_micros / 1000.0,
                metric.p95_micros / 1000.0,
                metric.p99_micros / 1000.0,
                metric.min_micros / 1000.0,
                metric.max_micros / 1000.0,
                metric.iqr_micros / 1000.0,
                metric.skewness,
                metric.kurtosis,
                metric.trend,
                metric.ema_micros / 1000.0,
                metric.rate_ops_per_sec,
            ));
        }

        report
    }

    /// Establish baseline for an operation
    pub async fn establish_baseline(&self, operation: ClusterOperation) -> Result<(), String> {
        let label = operation.as_str().to_string();
        let stats = self.latency_stats.read().await;

        let stat = stats
            .get(&label)
            .ok_or_else(|| format!("No metrics available for operation {}", operation.as_str()))?;

        if stat.values.len() < 30 {
            return Err(format!(
                "Insufficient samples for baseline (need 30, have {})",
                stat.values.len()
            ));
        }

        let baseline = OperationBaseline {
            operation: label.clone(),
            mean_micros: stat.mean(),
            std_dev_micros: stat.std_dev(),
            p50_micros: stat.percentile(50.0),
            p95_micros: stat.percentile(95.0),
            p99_micros: stat.percentile(99.0),
            sample_size: stat.values.len(),
            timestamp: SystemTime::now(),
        };

        let mut baselines = self.baselines.write().await;
        baselines.insert(label, baseline);

        info!(
            "Established baseline for {} on node {}",
            operation.as_str(),
            self.node_id
        );

        Ok(())
    }

    /// Detect performance regressions
    pub async fn detect_regressions(&self) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();
        let stats = self.latency_stats.read().await;
        let baselines = self.baselines.read().await;

        for (label, baseline) in baselines.iter() {
            if let Some(current) = stats.get(label) {
                // Perform statistical tests for regression detection

                // 1. Mean comparison with t-test
                if current.values.len() >= 10 {
                    let _current_values: &[f64] = &current.values;
                    let baseline_mean = baseline.mean_micros;
                    let baseline_std = baseline.std_dev_micros;

                    // Calculate t-statistic
                    let current_mean = current.mean();
                    let current_std = current.std_dev();
                    let n = current.values.len() as f64;

                    // Welch's t-test for unequal variances
                    let se = ((current_std.powi(2) / n)
                        + (baseline_std.powi(2) / baseline.sample_size as f64))
                        .sqrt();

                    if se > 0.0 {
                        let t_stat = (current_mean - baseline_mean) / se;

                        // Use SciRS2 for statistical testing
                        let df = n - 1.0;
                        if df > 0.0 {
                            // Calculate p-value using Student's t distribution
                            let t_dist = StudentT::new(0.0, 1.0, df);
                            let p_value = if let Ok(dist) = t_dist {
                                // One-tailed test for regression (slower)
                                // Use StudentT's cdf method directly
                                1.0 - dist.cdf(t_stat)
                            } else {
                                1.0 // Assume no regression if distribution fails
                            };

                            let change_pct =
                                ((current_mean - baseline_mean) / baseline_mean) * 100.0;

                            // Significant regression if p < 0.05 and >10% slower
                            if p_value < 0.05 && change_pct > 10.0 {
                                regressions.push(PerformanceRegression {
                                    operation: label.clone(),
                                    metric_name: "mean_latency".to_string(),
                                    baseline_value: baseline_mean,
                                    current_value: current_mean,
                                    change_percentage: change_pct,
                                    p_value,
                                    t_statistic: t_stat,
                                    severity: if change_pct > 100.0 {
                                        RegressionSeverity::Critical
                                    } else if change_pct > 50.0 {
                                        RegressionSeverity::High
                                    } else if change_pct > 25.0 {
                                        RegressionSeverity::Medium
                                    } else {
                                        RegressionSeverity::Low
                                    },
                                    detection_method: "Welch's t-test".to_string(),
                                });
                            }
                        }
                    }
                }

                // 2. Check p99 latency regression
                let current_p99 = current.percentile(99.0);
                let p99_change =
                    ((current_p99 - baseline.p99_micros) / baseline.p99_micros) * 100.0;

                if p99_change > 25.0 {
                    regressions.push(PerformanceRegression {
                        operation: label.clone(),
                        metric_name: "p99_latency".to_string(),
                        baseline_value: baseline.p99_micros,
                        current_value: current_p99,
                        change_percentage: p99_change,
                        p_value: 0.0,
                        t_statistic: 0.0,
                        severity: if p99_change > 100.0 {
                            RegressionSeverity::Critical
                        } else if p99_change > 50.0 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                        detection_method: "Percentile comparison".to_string(),
                    });
                }

                // 3. Check trend (increasing latency over time)
                let trend = current.trend();
                if trend > baseline.std_dev_micros * 0.1 {
                    regressions.push(PerformanceRegression {
                        operation: label.clone(),
                        metric_name: "latency_trend".to_string(),
                        baseline_value: 0.0,
                        current_value: trend,
                        change_percentage: (trend / baseline.mean_micros) * 100.0,
                        p_value: 0.0,
                        t_statistic: 0.0,
                        severity: RegressionSeverity::Low,
                        detection_method: "Trend analysis".to_string(),
                    });
                }
            }
        }

        if !regressions.is_empty() {
            warn!(
                "Detected {} performance regressions on node {}",
                regressions.len(),
                self.node_id
            );
        }

        regressions
    }

    /// Run benchmarks for cluster operations
    pub async fn run_benchmarks(&self) -> Vec<BenchmarkResultRecord> {
        let mut results = Vec::new();

        info!("Running cluster benchmarks on node {}", self.node_id);

        // Benchmark various operations with synthetic workloads
        let operations_to_benchmark = vec![
            ClusterOperation::AppendEntries,
            ClusterOperation::QueryExecution,
            ClusterOperation::BatchProcessing,
            ClusterOperation::DataReplication,
            ClusterOperation::MerkleVerification,
        ];

        for operation in operations_to_benchmark {
            let result = self.benchmark_operation(operation, 1000).await;
            results.push(result);
        }

        // Store results
        let mut stored = self.benchmark_results.write().await;
        stored.extend(results.clone());

        info!(
            "Completed {} benchmarks on node {}",
            results.len(),
            self.node_id
        );

        results
    }

    /// Benchmark a specific operation
    async fn benchmark_operation(
        &self,
        operation: ClusterOperation,
        iterations: u64,
    ) -> BenchmarkResultRecord {
        let mut latencies = Vec::with_capacity(iterations as usize);

        for _ in 0..iterations {
            let start = Instant::now();
            // Simulate operation workload
            std::hint::black_box(self.simulate_operation_workload(operation));
            latencies.push(start.elapsed().as_nanos() as f64);
        }

        let sum: f64 = latencies.iter().sum();
        let mean = sum / iterations as f64;
        let variance: f64 =
            latencies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / iterations as f64;
        let std_dev = variance.sqrt();

        BenchmarkResultRecord {
            name: format!("cluster_{}", operation.as_str()),
            timestamp: SystemTime::now(),
            mean_ns: mean,
            std_dev_ns: std_dev,
            iterations,
            throughput: 1_000_000_000.0 / mean,
        }
    }

    /// Simulate operation workload for benchmarking
    fn simulate_operation_workload(&self, operation: ClusterOperation) -> u64 {
        // Simulate CPU-bound work based on operation type
        match operation {
            ClusterOperation::AppendEntries => {
                // Simulate log entry serialization
                let mut sum: u64 = 0;
                for i in 0..100 {
                    sum = sum.wrapping_add(i * i);
                }
                sum
            }
            ClusterOperation::QueryExecution => {
                // Simulate query parsing and planning
                let mut sum: u64 = 0;
                for i in 0..500 {
                    sum = sum.wrapping_add(i * i * i);
                }
                sum
            }
            ClusterOperation::BatchProcessing => {
                // Simulate batch aggregation
                let mut sum: u64 = 0;
                for i in 0..200 {
                    sum = sum.wrapping_add(i);
                }
                sum
            }
            ClusterOperation::DataReplication => {
                // Simulate data serialization
                let mut sum: u64 = 0;
                for i in 0..300 {
                    sum = sum.wrapping_add(i * 7);
                }
                sum
            }
            ClusterOperation::MerkleVerification => {
                // Simulate hash computation
                let mut sum: u64 = 0;
                for i in 0u64..150 {
                    sum = sum.wrapping_add(i.wrapping_mul(i));
                }
                sum
            }
            _ => 0,
        }
    }

    /// Get benchmark history
    pub async fn get_benchmark_history(&self) -> Vec<BenchmarkResultRecord> {
        self.benchmark_results.read().await.clone()
    }

    /// Compare benchmarks between runs
    pub async fn compare_benchmarks(
        &self,
        baseline_name: &str,
        current_name: &str,
    ) -> Option<BenchmarkComparison> {
        let results = self.benchmark_results.read().await;

        let baseline = results.iter().find(|r| r.name == baseline_name)?;
        let current = results.iter().rev().find(|r| r.name == current_name)?;

        let speedup = baseline.mean_ns / current.mean_ns;
        let throughput_improvement =
            ((current.throughput - baseline.throughput) / baseline.throughput) * 100.0;

        Some(BenchmarkComparison {
            baseline_name: baseline_name.to_string(),
            current_name: current_name.to_string(),
            baseline_mean_ns: baseline.mean_ns,
            current_mean_ns: current.mean_ns,
            speedup,
            throughput_improvement,
            is_improved: current.mean_ns < baseline.mean_ns,
        })
    }
}
