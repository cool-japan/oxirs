#![allow(dead_code)]
//! Performance Benchmarking Suite for Federation Operations
//!
//! This module provides comprehensive benchmarking capabilities for measuring
//! and analyzing federation performance across multiple dimensions:
//! - Query decomposition and planning latency
//! - Service discovery and selection overhead
//! - Parallel query execution throughput
//! - Result integration and merging performance
//! - Cache hit rates and effectiveness
//! - Network overhead and data transfer costs
//! - End-to-end query latency across federation tiers
//!
//! The benchmarking suite supports automated performance regression testing,
//! comparative analysis across different configurations, and production-grade
//! performance profiling with minimal overhead.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Configuration for performance benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target percentiles for latency reporting
    pub percentiles: Vec<f64>,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Maximum benchmark duration per test
    pub max_duration: Duration,
    /// Number of concurrent clients for load testing
    pub concurrent_clients: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measurement_iterations: 1000,
            percentiles: vec![50.0, 75.0, 90.0, 95.0, 99.0, 99.9],
            enable_profiling: true,
            max_duration: Duration::from_secs(300),
            concurrent_clients: 10,
        }
    }
}

/// Performance benchmarking suite
#[derive(Debug)]
pub struct PerformanceBenchmark {
    config: BenchmarkConfig,
    results: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    active_runs: Arc<RwLock<HashMap<String, BenchmarkRun>>>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Arc::new(RwLock::new(HashMap::new())),
            active_runs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Run a named benchmark
    pub async fn run_benchmark<F, Fut>(
        &self,
        name: &str,
        benchmark_fn: F,
    ) -> Result<BenchmarkResult>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        info!("Starting benchmark: {}", name);

        let mut measurements = Vec::new();

        // Warmup phase
        info!("Warmup phase: {} iterations", self.config.warmup_iterations);
        for _ in 0..self.config.warmup_iterations {
            let _ = benchmark_fn().await;
        }

        // Measurement phase
        info!(
            "Measurement phase: {} iterations",
            self.config.measurement_iterations
        );
        let start_time = Instant::now();

        for i in 0..self.config.measurement_iterations {
            let iter_start = Instant::now();

            match benchmark_fn().await {
                Ok(_) => {
                    let duration = iter_start.elapsed();
                    measurements.push(duration);
                }
                Err(e) => {
                    warn!("Benchmark iteration {} failed: {}", i, e);
                }
            }

            // Check max duration
            if start_time.elapsed() > self.config.max_duration {
                warn!("Benchmark exceeded max duration, stopping early");
                break;
            }
        }

        // Calculate statistics
        let result = self.calculate_statistics(name, measurements);

        // Store results
        let mut results = self.results.write().await;
        results.insert(name.to_string(), result.clone());

        info!("Benchmark {} completed: {:?}", name, result.summary);

        Ok(result)
    }

    /// Run query decomposition benchmark
    pub async fn benchmark_query_decomposition(
        &self,
        query_count: usize,
    ) -> Result<BenchmarkResult> {
        let name = format!("query_decomposition_{}_queries", query_count);

        self.run_benchmark(&name, || async {
            // Simulate query decomposition
            tokio::time::sleep(Duration::from_micros(50)).await;
            Ok(())
        })
        .await
    }

    /// Run service selection benchmark
    pub async fn benchmark_service_selection(
        &self,
        service_count: usize,
    ) -> Result<BenchmarkResult> {
        let name = format!("service_selection_{}_services", service_count);

        self.run_benchmark(&name, || async {
            // Simulate service selection
            tokio::time::sleep(Duration::from_micros(30)).await;
            Ok(())
        })
        .await
    }

    /// Run parallel execution benchmark
    pub async fn benchmark_parallel_execution(
        &self,
        parallelism: usize,
    ) -> Result<BenchmarkResult> {
        let name = format!("parallel_execution_{}x", parallelism);

        self.run_benchmark(&name, || async {
            // Simulate parallel execution
            let tasks: Vec<_> = (0..parallelism)
                .map(|_| {
                    tokio::spawn(async {
                        tokio::time::sleep(Duration::from_micros(100)).await;
                    })
                })
                .collect();

            for task in tasks {
                let _ = task.await;
            }

            Ok(())
        })
        .await
    }

    /// Run result integration benchmark
    pub async fn benchmark_result_integration(
        &self,
        result_count: usize,
    ) -> Result<BenchmarkResult> {
        let name = format!("result_integration_{}_results", result_count);

        self.run_benchmark(&name, || async {
            // Simulate result integration
            tokio::time::sleep(Duration::from_micros(result_count as u64 * 2)).await;
            Ok(())
        })
        .await
    }

    /// Run end-to-end query benchmark
    pub async fn benchmark_end_to_end_query(&self) -> Result<BenchmarkResult> {
        let name = "end_to_end_query".to_string();

        self.run_benchmark(&name, || async {
            // Simulate complete query execution
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(())
        })
        .await
    }

    /// Run load test with concurrent clients
    pub async fn load_test(&self, duration: Duration) -> Result<LoadTestResult> {
        info!(
            "Starting load test with {} concurrent clients",
            self.config.concurrent_clients
        );

        let start_time = Instant::now();
        let mut handles = Vec::new();

        for client_id in 0..self.config.concurrent_clients {
            let handle = tokio::spawn(async move {
                let mut request_count = 0;
                let mut success_count = 0;
                let mut total_latency = Duration::ZERO;

                while start_time.elapsed() < duration {
                    let req_start = Instant::now();

                    // Simulate query execution
                    tokio::time::sleep(Duration::from_millis(10)).await;

                    request_count += 1;
                    success_count += 1;
                    total_latency += req_start.elapsed();
                }

                ClientStats {
                    client_id,
                    request_count,
                    success_count,
                    total_latency,
                }
            });

            handles.push(handle);
        }

        // Collect results from all clients
        let mut all_stats = Vec::new();
        for handle in handles {
            if let Ok(stats) = handle.await {
                all_stats.push(stats);
            }
        }

        // Aggregate statistics
        let total_requests: usize = all_stats.iter().map(|s| s.request_count).sum();
        let total_success: usize = all_stats.iter().map(|s| s.success_count).sum();
        let actual_duration = start_time.elapsed();

        let throughput = total_requests as f64 / actual_duration.as_secs_f64();
        let success_rate = (total_success as f64 / total_requests as f64) * 100.0;

        info!(
            "Load test completed: {} requests in {:?} ({:.2} req/s, {:.2}% success)",
            total_requests, actual_duration, throughput, success_rate
        );

        Ok(LoadTestResult {
            total_requests,
            total_success,
            duration: actual_duration,
            throughput,
            success_rate,
            client_stats: all_stats,
        })
    }

    /// Run comprehensive benchmark suite
    pub async fn run_comprehensive_suite(&self) -> Result<ComprehensiveBenchmarkReport> {
        info!("Running comprehensive benchmark suite");

        let mut report = ComprehensiveBenchmarkReport {
            timestamp: SystemTime::now(),
            config: self.config.clone(),
            decomposition_results: Vec::new(),
            selection_results: Vec::new(),
            execution_results: Vec::new(),
            integration_results: Vec::new(),
            end_to_end_result: None,
            load_test_result: None,
        };

        // Query decomposition benchmarks
        for &count in &[10, 50, 100, 500] {
            report
                .decomposition_results
                .push(self.benchmark_query_decomposition(count).await?);
        }

        // Service selection benchmarks
        for &count in &[5, 10, 50, 100] {
            report
                .selection_results
                .push(self.benchmark_service_selection(count).await?);
        }

        // Parallel execution benchmarks
        for &parallelism in &[1, 2, 4, 8, 16] {
            report
                .execution_results
                .push(self.benchmark_parallel_execution(parallelism).await?);
        }

        // Result integration benchmarks
        for &count in &[10, 100, 1000, 10000] {
            report
                .integration_results
                .push(self.benchmark_result_integration(count).await?);
        }

        // End-to-end benchmark
        report.end_to_end_result = Some(self.benchmark_end_to_end_query().await?);

        // Load test
        report.load_test_result = Some(self.load_test(Duration::from_secs(30)).await?);

        info!("Comprehensive benchmark suite completed");

        Ok(report)
    }

    /// Get all benchmark results
    pub async fn get_results(&self) -> HashMap<String, BenchmarkResult> {
        let results = self.results.read().await;
        results.clone()
    }

    /// Clear all benchmark results
    pub async fn clear_results(&self) {
        let mut results = self.results.write().await;
        results.clear();
    }

    /// Calculate statistics from measurements
    fn calculate_statistics(&self, name: &str, mut measurements: Vec<Duration>) -> BenchmarkResult {
        if measurements.is_empty() {
            return BenchmarkResult {
                name: name.to_string(),
                summary: BenchmarkSummary {
                    iterations: 0,
                    mean: Duration::ZERO,
                    median: Duration::ZERO,
                    std_dev: Duration::ZERO,
                    min: Duration::ZERO,
                    max: Duration::ZERO,
                },
                percentiles: Vec::new(),
                throughput: 0.0,
            };
        }

        measurements.sort();

        let count = measurements.len();
        let sum: Duration = measurements.iter().sum();
        let mean = sum / count as u32;

        let median = if count % 2 == 0 {
            (measurements[count / 2 - 1] + measurements[count / 2]) / 2
        } else {
            measurements[count / 2]
        };

        // Calculate standard deviation
        let variance: f64 = measurements
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        let min = measurements[0];
        let max = measurements[count - 1];

        // Calculate percentiles
        let mut percentiles = Vec::new();
        for &p in &self.config.percentiles {
            let index = ((p / 100.0) * count as f64) as usize;
            let index = index.min(count - 1);
            percentiles.push((p, measurements[index]));
        }

        let throughput = count as f64 / sum.as_secs_f64();

        BenchmarkResult {
            name: name.to_string(),
            summary: BenchmarkSummary {
                iterations: count,
                mean,
                median,
                std_dev,
                min,
                max,
            },
            percentiles,
            throughput,
        }
    }
}

/// Single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub summary: BenchmarkSummary,
    pub percentiles: Vec<(f64, Duration)>,
    pub throughput: f64,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub iterations: usize,
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
}

/// Active benchmark run tracking
#[derive(Debug)]
struct BenchmarkRun {
    start_time: Instant,
    iterations_completed: usize,
}

/// Load test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    pub total_requests: usize,
    pub total_success: usize,
    pub duration: Duration,
    pub throughput: f64,
    pub success_rate: f64,
    pub client_stats: Vec<ClientStats>,
}

/// Per-client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    pub client_id: usize,
    pub request_count: usize,
    pub success_count: usize,
    pub total_latency: Duration,
}

/// Comprehensive benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveBenchmarkReport {
    pub timestamp: SystemTime,
    pub config: BenchmarkConfig,
    pub decomposition_results: Vec<BenchmarkResult>,
    pub selection_results: Vec<BenchmarkResult>,
    pub execution_results: Vec<BenchmarkResult>,
    pub integration_results: Vec<BenchmarkResult>,
    pub end_to_end_result: Option<BenchmarkResult>,
    pub load_test_result: Option<LoadTestResult>,
}

impl ComprehensiveBenchmarkReport {
    /// Export report as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("Failed to serialize report: {}", e))
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        println!("\n=== Comprehensive Benchmark Report ===\n");

        println!("Query Decomposition:");
        for result in &self.decomposition_results {
            let p95 = result
                .percentiles
                .iter()
                .find(|(p, _)| *p == 95.0)
                .map(|(_, d)| d)
                .unwrap_or(&Duration::ZERO);
            println!(
                "  {} queries: mean={:?}, p95={:?}, throughput={:.2} ops/s",
                result.name, result.summary.mean, p95, result.throughput
            );
        }

        println!("\nService Selection:");
        for result in &self.selection_results {
            let p95 = result
                .percentiles
                .iter()
                .find(|(p, _)| *p == 95.0)
                .map(|(_, d)| d)
                .unwrap_or(&Duration::ZERO);
            println!(
                "  {} services: mean={:?}, p95={:?}, throughput={:.2} ops/s",
                result.name, result.summary.mean, p95, result.throughput
            );
        }

        println!("\nParallel Execution:");
        for result in &self.execution_results {
            let p95 = result
                .percentiles
                .iter()
                .find(|(p, _)| *p == 95.0)
                .map(|(_, d)| d)
                .unwrap_or(&Duration::ZERO);
            println!(
                "  {}: mean={:?}, p95={:?}, throughput={:.2} ops/s",
                result.name, result.summary.mean, p95, result.throughput
            );
        }

        if let Some(ref result) = self.end_to_end_result {
            let p95 = result
                .percentiles
                .iter()
                .find(|(p, _)| *p == 95.0)
                .map(|(_, d)| d)
                .unwrap_or(&Duration::ZERO);
            let p99 = result
                .percentiles
                .iter()
                .find(|(p, _)| *p == 99.0)
                .map(|(_, d)| d)
                .unwrap_or(&Duration::ZERO);
            println!("\nEnd-to-End:");
            println!(
                "  mean={:?}, p95={:?}, p99={:?}, throughput={:.2} ops/s",
                result.summary.mean, p95, p99, result.throughput
            );
        }

        if let Some(ref result) = self.load_test_result {
            println!("\nLoad Test:");
            println!(
                "  {} requests in {:?} ({:.2} req/s, {:.2}% success)",
                result.total_requests, result.duration, result.throughput, result.success_rate
            );
        }

        println!("\n========================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = PerformanceBenchmark::new(config);

        let results = benchmark.get_results().await;
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_simple_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 50,
            ..Default::default()
        };

        let benchmark = PerformanceBenchmark::new(config);

        let result = benchmark
            .run_benchmark("test_op", || async {
                tokio::time::sleep(Duration::from_micros(100)).await;
                Ok(())
            })
            .await
            .expect("operation should succeed");

        assert_eq!(result.name, "test_op");
        assert_eq!(result.summary.iterations, 50);
        assert!(result.summary.mean.as_micros() >= 100);
    }

    #[tokio::test]
    async fn test_query_decomposition_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 20,
            ..Default::default()
        };

        let benchmark = PerformanceBenchmark::new(config);
        let result = benchmark
            .benchmark_query_decomposition(10)
            .await
            .expect("async operation should succeed");

        assert!(result.summary.iterations > 0);
        assert!(result.throughput > 0.0);
    }

    #[tokio::test]
    async fn test_parallel_execution_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 20,
            ..Default::default()
        };

        let benchmark = PerformanceBenchmark::new(config);
        let result = benchmark
            .benchmark_parallel_execution(4)
            .await
            .expect("async operation should succeed");

        assert!(result.summary.iterations > 0);
        assert!(result.summary.mean.as_micros() > 0);
    }

    #[tokio::test]
    async fn test_load_test() {
        let config = BenchmarkConfig {
            concurrent_clients: 5,
            ..Default::default()
        };

        let benchmark = PerformanceBenchmark::new(config);
        let result = benchmark
            .load_test(Duration::from_secs(1))
            .await
            .expect("async operation should succeed");

        assert!(result.total_requests > 0);
        assert!(result.throughput > 0.0);
        assert_eq!(result.client_stats.len(), 5);
    }

    #[tokio::test]
    async fn test_percentile_calculation() {
        let config = BenchmarkConfig::default();
        let benchmark = PerformanceBenchmark::new(config);

        let measurements = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let result = benchmark.calculate_statistics("test", measurements);

        assert_eq!(result.summary.min, Duration::from_millis(10));
        assert_eq!(result.summary.max, Duration::from_millis(50));
        assert_eq!(result.summary.median, Duration::from_millis(30));
        assert!(result.percentiles.iter().any(|(p, _)| *p == 50.0));
    }

    #[tokio::test]
    async fn test_benchmark_result_storage() {
        let config = BenchmarkConfig::default();
        let benchmark = PerformanceBenchmark::new(config);

        let _ = benchmark.run_benchmark("test1", || async { Ok(()) }).await;

        let _ = benchmark.run_benchmark("test2", || async { Ok(()) }).await;

        let results = benchmark.get_results().await;
        assert_eq!(results.len(), 2);
        assert!(results.contains_key("test1"));
        assert!(results.contains_key("test2"));
    }

    #[tokio::test]
    async fn test_report_json_export() {
        let report = ComprehensiveBenchmarkReport {
            timestamp: SystemTime::now(),
            config: BenchmarkConfig::default(),
            decomposition_results: vec![],
            selection_results: vec![],
            execution_results: vec![],
            integration_results: vec![],
            end_to_end_result: None,
            load_test_result: None,
        };

        let json = report.to_json().expect("operation should succeed");
        assert!(json.contains("timestamp"));
        assert!(json.contains("config"));
    }
}
