//! Comprehensive GPU vs CPU benchmarking for vector operations
//!
//! This module provides detailed benchmarks comparing GPU and CPU performance
//! across all supported distance metrics, different vector dimensions, and dataset sizes.

use crate::gpu::{GpuConfig, GpuVectorIndex};
use crate::similarity::SimilarityMetric;
use crate::Vector;
use anyhow::Result;
use scirs2_core::random;
use scirs2_core::Rng;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct GpuBenchmarkConfig {
    /// Number of vectors in the database
    pub database_size: usize,
    /// Number of query vectors
    pub query_count: usize,
    /// Vector dimensions to test
    pub dimensions: Vec<usize>,
    /// Distance metrics to benchmark
    pub metrics: Vec<SimilarityMetric>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Enable CPU baseline comparison
    pub compare_cpu: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

impl Default for GpuBenchmarkConfig {
    fn default() -> Self {
        Self {
            database_size: 10_000,
            query_count: 100,
            dimensions: vec![128, 256, 512, 768, 1024],
            metrics: vec![
                SimilarityMetric::Cosine,
                SimilarityMetric::Euclidean,
                SimilarityMetric::Manhattan,
                SimilarityMetric::Pearson,
                SimilarityMetric::Jaccard,
                SimilarityMetric::Angular,
            ],
            warmup_iterations: 3,
            measurement_iterations: 10,
            compare_cpu: true,
            enable_gpu: true,
        }
    }
}

/// Benchmark results for a single metric and dimension
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub metric: SimilarityMetric,
    pub dimension: usize,
    pub database_size: usize,
    pub query_count: usize,
    pub cpu_time_ms: Option<f64>,
    pub gpu_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub throughput_qps: f64,
    pub memory_usage_mb: f64,
}

impl BenchmarkResult {
    /// Calculate speedup factor (GPU vs CPU)
    fn calculate_speedup(&mut self) {
        if let (Some(cpu_time), Some(gpu_time)) = (self.cpu_time_ms, self.gpu_time_ms) {
            if gpu_time > 0.0 {
                self.speedup = Some(cpu_time / gpu_time);
            }
        }
    }

    /// Calculate queries per second
    fn calculate_throughput(&mut self) {
        let time_ms = self.gpu_time_ms.or(self.cpu_time_ms).unwrap_or(1.0);
        if time_ms > 0.0 {
            self.throughput_qps = (self.query_count as f64 / time_ms) * 1000.0;
        }
    }
}

/// Comprehensive GPU benchmark suite
pub struct GpuBenchmarkSuite {
    config: GpuBenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl GpuBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: GpuBenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all benchmarks
    pub fn run(&mut self) -> Result<&[BenchmarkResult]> {
        tracing::info!(
            "Starting GPU benchmark suite with {} metrics, {} dimensions",
            self.config.metrics.len(),
            self.config.dimensions.len()
        );

        for &dim in &self.config.dimensions {
            for metric in &self.config.metrics {
                tracing::info!(
                    "Benchmarking {} metric with dimension {}",
                    format!("{:?}", metric),
                    dim
                );

                let result = self.benchmark_metric(*metric, dim)?;
                self.results.push(result);
            }
        }

        Ok(&self.results)
    }

    /// Benchmark a single metric and dimension
    fn benchmark_metric(&self, metric: SimilarityMetric, dim: usize) -> Result<BenchmarkResult> {
        // Generate test data
        let (database, queries) = self.generate_test_data(dim)?;

        let mut result = BenchmarkResult {
            metric,
            dimension: dim,
            database_size: self.config.database_size,
            query_count: self.config.query_count,
            cpu_time_ms: None,
            gpu_time_ms: None,
            speedup: None,
            throughput_qps: 0.0,
            memory_usage_mb: self.estimate_memory_usage(dim),
        };

        // CPU baseline
        if self.config.compare_cpu {
            result.cpu_time_ms = Some(self.benchmark_cpu(&database, &queries, metric)?);
        }

        // GPU benchmark
        if self.config.enable_gpu {
            match self.benchmark_gpu(&database, &queries, metric, dim) {
                Ok(time) => result.gpu_time_ms = Some(time),
                Err(e) => {
                    tracing::warn!("GPU benchmark failed: {}, falling back to CPU-only", e);
                }
            }
        }

        result.calculate_speedup();
        result.calculate_throughput();

        Ok(result)
    }

    /// Generate synthetic test data
    fn generate_test_data(&self, dim: usize) -> Result<(Vec<Vector>, Vec<Vector>)> {
        let mut rng = random::rng();

        let mut database = Vec::with_capacity(self.config.database_size);
        for _i in 0..self.config.database_size {
            let values: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();
            database.push(Vector::new(values));
        }

        let mut queries = Vec::with_capacity(self.config.query_count);
        for _i in 0..self.config.query_count {
            let values: Vec<f32> = (0..dim).map(|_| rng.random_range(0.0..1.0)).collect();
            queries.push(Vector::new(values));
        }

        Ok((database, queries))
    }

    /// Benchmark CPU implementation
    fn benchmark_cpu(
        &self,
        database: &[Vector],
        queries: &[Vector],
        metric: SimilarityMetric,
    ) -> Result<f64> {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in queries.iter().take(5) {
                for db_vec in database.iter().take(100) {
                    let _ = metric.compute(query, db_vec)?;
                }
            }
        }

        // Measurement
        let mut total_time = Duration::ZERO;
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            for query in queries {
                for db_vec in database {
                    let _ = metric.compute(query, db_vec)?;
                }
            }
            total_time += start.elapsed();
        }

        let avg_time_ms =
            total_time.as_secs_f64() * 1000.0 / self.config.measurement_iterations as f64;
        Ok(avg_time_ms)
    }

    /// Benchmark GPU implementation
    fn benchmark_gpu(
        &self,
        database: &[Vector],
        queries: &[Vector],
        metric: SimilarityMetric,
        _dim: usize,
    ) -> Result<f64> {
        let gpu_config = GpuConfig {
            device_id: 0,
            enable_tensor_cores: true,
            enable_mixed_precision: true,
            memory_pool_size: 1 << 30, // 1GB
            stream_count: 4,
            ..Default::default()
        };

        let mut gpu_index = GpuVectorIndex::new(gpu_config)?;
        gpu_index.add_vectors(database.to_vec())?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            for query in queries.iter().take(5) {
                let _ = gpu_index.search(query, 10, metric)?;
            }
        }

        // Measurement
        let mut total_time = Duration::ZERO;
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            for query in queries {
                let _ = gpu_index.search(query, 10, metric)?;
            }
            total_time += start.elapsed();
        }

        let avg_time_ms =
            total_time.as_secs_f64() * 1000.0 / self.config.measurement_iterations as f64;
        Ok(avg_time_ms)
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, dim: usize) -> f64 {
        let vector_size_bytes = dim * std::mem::size_of::<f32>();
        let total_vectors = self.config.database_size + self.config.query_count;
        let total_bytes = total_vectors * vector_size_bytes;
        total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU Benchmark Report ===\n\n");

        report.push_str(&format!(
            "Configuration:\n  Database size: {}\n  Query count: {}\n  Dimensions tested: {:?}\n\n",
            self.config.database_size, self.config.query_count, self.config.dimensions
        ));

        report.push_str("Results:\n");
        report.push_str(&format!(
            "{:<20} {:<10} {:<12} {:<12} {:<10} {:<12}\n",
            "Metric", "Dimension", "CPU (ms)", "GPU (ms)", "Speedup", "QPS"
        ));
        report.push_str(&"-".repeat(90));
        report.push('\n');

        for result in &self.results {
            let cpu_time = result
                .cpu_time_ms
                .map(|t| format!("{:.2}", t))
                .unwrap_or_else(|| "N/A".to_string());
            let gpu_time = result
                .gpu_time_ms
                .map(|t| format!("{:.2}", t))
                .unwrap_or_else(|| "N/A".to_string());
            let speedup = result
                .speedup
                .map(|s| format!("{:.2}x", s))
                .unwrap_or_else(|| "N/A".to_string());

            report.push_str(&format!(
                "{:<20} {:<10} {:<12} {:<12} {:<10} {:<12.0}\n",
                format!("{:?}", result.metric),
                result.dimension,
                cpu_time,
                gpu_time,
                speedup,
                result.throughput_qps
            ));
        }

        report.push('\n');
        self.add_summary_statistics(&mut report);

        report
    }

    /// Add summary statistics to report
    fn add_summary_statistics(&self, report: &mut String) {
        if self.results.is_empty() {
            return;
        }

        report.push_str("Summary Statistics:\n");

        // Calculate average speedup
        let speedups: Vec<f64> = self.results.iter().filter_map(|r| r.speedup).collect();

        if !speedups.is_empty() {
            let avg_speedup: f64 = speedups.iter().sum::<f64>() / speedups.len() as f64;
            let max_speedup = speedups
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            report.push_str(&format!("  Average speedup: {:.2}x\n", avg_speedup));
            report.push_str(&format!("  Maximum speedup: {:.2}x\n", max_speedup));
        }

        // Calculate total throughput
        let total_qps: f64 = self.results.iter().map(|r| r.throughput_qps).sum();
        report.push_str(&format!(
            "  Total throughput: {:.0} queries/sec\n",
            total_qps / self.results.len() as f64
        ));

        // Memory usage
        let total_memory: f64 = self.results.iter().map(|r| r.memory_usage_mb).sum();
        report.push_str(&format!(
            "  Estimated memory: {:.2} MB\n",
            total_memory / self.results.len() as f64
        ));
    }

    /// Export results to JSON
    pub fn export_json(&self) -> Result<String> {
        #[derive(serde::Serialize)]
        struct JsonResult {
            metric: String,
            dimension: usize,
            database_size: usize,
            query_count: usize,
            cpu_time_ms: Option<f64>,
            gpu_time_ms: Option<f64>,
            speedup: Option<f64>,
            throughput_qps: f64,
            memory_usage_mb: f64,
        }

        let json_results: Vec<JsonResult> = self
            .results
            .iter()
            .map(|r| JsonResult {
                metric: format!("{:?}", r.metric),
                dimension: r.dimension,
                database_size: r.database_size,
                query_count: r.query_count,
                cpu_time_ms: r.cpu_time_ms,
                gpu_time_ms: r.gpu_time_ms,
                speedup: r.speedup,
                throughput_qps: r.throughput_qps,
                memory_usage_mb: r.memory_usage_mb,
            })
            .collect();

        Ok(serde_json::to_string_pretty(&json_results)?)
    }

    /// Get benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = GpuBenchmarkConfig::default();
        assert_eq!(config.database_size, 10_000);
        assert_eq!(config.query_count, 100);
        assert!(!config.dimensions.is_empty());
        assert!(!config.metrics.is_empty());
    }

    #[test]
    fn test_memory_estimation() {
        let config = GpuBenchmarkConfig::default();
        let suite = GpuBenchmarkSuite::new(config);
        let memory_mb = suite.estimate_memory_usage(256);
        assert!(memory_mb > 0.0);
    }

    #[test]
    fn test_benchmark_result_calculation() {
        let mut result = BenchmarkResult {
            metric: SimilarityMetric::Cosine,
            dimension: 128,
            database_size: 1000,
            query_count: 100,
            cpu_time_ms: Some(100.0),
            gpu_time_ms: Some(10.0),
            speedup: None,
            throughput_qps: 0.0,
            memory_usage_mb: 10.0,
        };

        result.calculate_speedup();
        assert_eq!(result.speedup, Some(10.0));

        result.calculate_throughput();
        assert!(result.throughput_qps > 0.0);
    }

    #[test]
    fn test_generate_test_data() {
        let config = GpuBenchmarkConfig {
            database_size: 100,
            query_count: 10,
            ..Default::default()
        };

        let suite = GpuBenchmarkSuite::new(config);
        let result = suite.generate_test_data(128);
        assert!(result.is_ok());

        let (database, queries) = result.unwrap();
        assert_eq!(database.len(), 100);
        assert_eq!(queries.len(), 10);
    }
}
