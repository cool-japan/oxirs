//! Performance Benchmarking Framework Module
//!
//! This module provides comprehensive performance benchmarking capabilities for the SHACL-AI
//! system, including micro-benchmarks, macro-benchmarks, scalability testing, regression
//! detection, and automated performance analysis.

pub mod benchmarks;
pub mod config;
pub mod types;

// Re-export key types and structs for easier access
pub use benchmarks::{Benchmark, BenchmarkBuilder, BenchmarkSuite, BenchmarkSuiteStatistics};
pub use config::{
    AnalysisConfig, BenchmarkConfig, BenchmarkSuiteConfig, ProfilerConfig, RegressionConfig,
    ReportingConfig, ScalabilityConfig,
};
pub use types::{
    AccessPattern, BenchmarkExecutionContext, BenchmarkResult, BenchmarkStatus, BenchmarkType,
    CacheBehavior, CpuUsageStats, DataDistribution, ExecutionSummary, IoUsageStats,
    MeasurementConfig, MemoryUsageStats, PerformanceCounters, PrecisionLevel, ResourceUsageSummary,
    RunningBenchmark, SuccessCriteria, TargetComponent, ThroughputSummary, WorkloadConfig,
};

use crate::{Result, ShaclAiError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Performance benchmarking framework
#[derive(Debug)]
pub struct PerformanceBenchmarkFramework {
    config: BenchmarkConfig,
    benchmark_runner: Arc<Mutex<BenchmarkRunner>>,
    result_collector: Arc<Mutex<BenchmarkResultCollector>>,
}

impl PerformanceBenchmarkFramework {
    /// Create a new performance benchmark framework
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config: config.clone(),
            benchmark_runner: Arc::new(Mutex::new(BenchmarkRunner::new(config.clone()))),
            result_collector: Arc::new(Mutex::new(BenchmarkResultCollector::new())),
        }
    }

    /// Execute a benchmark suite
    pub fn execute_suite(&self, suite: &BenchmarkSuite) -> Result<ExecutionSummary> {
        let mut runner = self.benchmark_runner.lock().map_err(|e| {
            ShaclAiError::Benchmark(format!("Failed to lock benchmark runner: {e}"))
        })?;

        let execution_order = suite.get_execution_order()?;
        let mut results = Vec::new();
        let start_time = Instant::now();

        for benchmark in execution_order {
            let result = runner.execute_benchmark(benchmark)?;
            results.push(result);
        }

        let total_execution_time = start_time.elapsed();
        let summary = self.create_execution_summary(results, total_execution_time);

        // Store results
        let mut collector = self.result_collector.lock().map_err(|e| {
            ShaclAiError::Benchmark(format!("Failed to lock result collector: {e}"))
        })?;
        collector.store_execution_summary(&summary);

        Ok(summary)
    }

    /// Get framework configuration
    pub fn get_config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Create execution summary from benchmark results
    fn create_execution_summary(
        &self,
        results: Vec<BenchmarkResult>,
        total_time: Duration,
    ) -> ExecutionSummary {
        let total_benchmarks = results.len();
        let successful_benchmarks = results.iter().filter(|r| r.success).count();
        let failed_benchmarks = total_benchmarks - successful_benchmarks;

        let average_execution_time = if total_benchmarks > 0 {
            Duration::from_millis(
                results
                    .iter()
                    .map(|r| r.execution_time.as_millis() as u64)
                    .sum::<u64>()
                    / total_benchmarks as u64,
            )
        } else {
            Duration::from_millis(0)
        };

        let success_rate_percent = if total_benchmarks > 0 {
            (successful_benchmarks as f64 / total_benchmarks as f64) * 100.0
        } else {
            0.0
        };

        ExecutionSummary {
            total_benchmarks,
            successful_benchmarks,
            failed_benchmarks,
            total_execution_time: total_time,
            average_execution_time,
            throughput_summary: self.calculate_throughput_summary(&results),
            resource_usage_summary: self.calculate_resource_usage_summary(&results),
            success_rate_percent,
        }
    }

    /// Calculate throughput summary
    fn calculate_throughput_summary(&self, results: &[BenchmarkResult]) -> ThroughputSummary {
        let throughputs: Vec<f64> = results.iter().map(|r| r.throughput_ops_per_sec).collect();

        if throughputs.is_empty() {
            return ThroughputSummary {
                min_throughput_ops_per_sec: 0.0,
                max_throughput_ops_per_sec: 0.0,
                average_throughput_ops_per_sec: 0.0,
                median_throughput_ops_per_sec: 0.0,
                throughput_std_dev: 0.0,
            };
        }

        let min_throughput = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_throughput = throughputs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let average_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        let mut sorted_throughputs = throughputs.clone();
        sorted_throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_throughput = sorted_throughputs[sorted_throughputs.len() / 2];

        let variance = throughputs
            .iter()
            .map(|&x| (x - average_throughput).powi(2))
            .sum::<f64>()
            / throughputs.len() as f64;
        let std_dev = variance.sqrt();

        ThroughputSummary {
            min_throughput_ops_per_sec: min_throughput,
            max_throughput_ops_per_sec: max_throughput,
            average_throughput_ops_per_sec: average_throughput,
            median_throughput_ops_per_sec: median_throughput,
            throughput_std_dev: std_dev,
        }
    }

    /// Calculate resource usage summary
    fn calculate_resource_usage_summary(
        &self,
        results: &[BenchmarkResult],
    ) -> ResourceUsageSummary {
        if results.is_empty() {
            return ResourceUsageSummary {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                peak_cpu_percent: 0.0,
                average_cpu_percent: 0.0,
                total_io_mb: 0.0,
                average_io_latency: Duration::from_millis(0),
            };
        }

        let peak_memory = results
            .iter()
            .map(|r| r.memory_usage_stats.peak_memory_mb)
            .fold(0.0, f64::max);

        let average_memory = results
            .iter()
            .map(|r| r.memory_usage_stats.average_memory_mb)
            .sum::<f64>()
            / results.len() as f64;

        let peak_cpu = results
            .iter()
            .map(|r| r.cpu_usage_stats.peak_cpu_percent)
            .fold(0.0, f64::max);

        let average_cpu = results
            .iter()
            .map(|r| r.cpu_usage_stats.average_cpu_percent)
            .sum::<f64>()
            / results.len() as f64;

        let total_io = results
            .iter()
            .map(|r| {
                (r.io_usage_stats.total_bytes_read + r.io_usage_stats.total_bytes_written) as f64
                    / 1024.0
                    / 1024.0
            })
            .sum::<f64>();

        let average_io_latency_ms = results
            .iter()
            .map(|r| r.io_usage_stats.average_io_latency.as_millis() as u64)
            .sum::<u64>()
            / results.len() as u64;

        ResourceUsageSummary {
            peak_memory_mb: peak_memory,
            average_memory_mb: average_memory,
            peak_cpu_percent: peak_cpu,
            average_cpu_percent: average_cpu,
            total_io_mb: total_io,
            average_io_latency: Duration::from_millis(average_io_latency_ms),
        }
    }
}

/// Benchmark runner for executing individual benchmarks
#[derive(Debug)]
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    running_benchmarks: HashMap<Uuid, RunningBenchmark>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            running_benchmarks: HashMap::new(),
        }
    }

    /// Execute a single benchmark
    pub fn execute_benchmark(&mut self, benchmark: &Benchmark) -> Result<BenchmarkResult> {
        let start_time = Instant::now();

        // Create running benchmark state
        let running_benchmark = RunningBenchmark {
            benchmark_id: benchmark.id,
            name: benchmark.name.clone(),
            status: BenchmarkStatus::Running,
            start_time,
            current_iteration: 0,
            total_iterations: benchmark.measurement_config.iterations,
            progress_percentage: 0.0,
            estimated_time_remaining: Some(benchmark.timeout),
        };

        self.running_benchmarks
            .insert(benchmark.id, running_benchmark);

        // Execute the benchmark
        let result = benchmark.execute();

        // Remove from running benchmarks
        self.running_benchmarks.remove(&benchmark.id);

        result
    }

    /// Get currently running benchmarks
    pub fn get_running_benchmarks(&self) -> Vec<&RunningBenchmark> {
        self.running_benchmarks.values().collect()
    }
}

/// Benchmark result collector for storing and managing results
#[derive(Debug)]
pub struct BenchmarkResultCollector {
    execution_summaries: Vec<ExecutionSummary>,
    benchmark_results: HashMap<Uuid, BenchmarkResult>,
}

impl BenchmarkResultCollector {
    pub fn new() -> Self {
        Self {
            execution_summaries: Vec::new(),
            benchmark_results: HashMap::new(),
        }
    }

    /// Store an execution summary
    pub fn store_execution_summary(&mut self, summary: &ExecutionSummary) {
        self.execution_summaries.push(summary.clone());
    }

    /// Store a benchmark result
    pub fn store_benchmark_result(&mut self, result: BenchmarkResult) {
        self.benchmark_results.insert(result.benchmark_id, result);
    }

    /// Get all execution summaries
    pub fn get_execution_summaries(&self) -> &[ExecutionSummary] {
        &self.execution_summaries
    }

    /// Get benchmark result by ID
    pub fn get_benchmark_result(&self, benchmark_id: &Uuid) -> Option<&BenchmarkResult> {
        self.benchmark_results.get(benchmark_id)
    }
}

impl Default for BenchmarkResultCollector {
    fn default() -> Self {
        Self::new()
    }
}
