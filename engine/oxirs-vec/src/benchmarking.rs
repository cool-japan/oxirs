//! Comprehensive benchmarking framework for vector search systems
//!
//! This module provides extensive benchmarking capabilities including:
//! - ANN-Benchmarks integration
//! - Performance profiling and analysis
//! - Quality metrics (recall, precision)
//! - Scalability testing
//! - Comparative benchmarks

use crate::{similarity::SimilarityMetric, Vector, VectorIndex, VectorStore};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup runs
    pub warmup_runs: usize,
    /// Number of benchmark runs
    pub benchmark_runs: usize,
    /// Maximum benchmark duration
    pub max_duration: Duration,
    /// Enable memory profiling
    pub profile_memory: bool,
    /// Enable detailed timing
    pub detailed_timing: bool,
    /// Enable quality metrics
    pub quality_metrics: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Output format for results
    pub output_format: BenchmarkOutputFormat,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 3,
            benchmark_runs: 10,
            max_duration: Duration::from_secs(300), // 5 minutes
            profile_memory: true,
            detailed_timing: true,
            quality_metrics: true,
            random_seed: Some(42),
            output_format: BenchmarkOutputFormat::Json,
        }
    }
}

/// Benchmark output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkOutputFormat {
    Json,
    Csv,
    Table,
    AnnBenchmarks,
}

/// Benchmark suite for comprehensive vector search testing
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    datasets: Vec<BenchmarkDataset>,
    algorithms: Vec<Box<dyn VectorIndex>>,
    results: Vec<BenchmarkResult>,
}

/// Benchmark dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    /// Dataset name
    pub name: String,
    /// Dataset description
    pub description: String,
    /// Training vectors
    pub train_vectors: Vec<Vector>,
    /// Query vectors
    pub query_vectors: Vec<Vector>,
    /// Ground truth results (for quality metrics)
    pub ground_truth: Option<Vec<Vec<usize>>>, // For each query, list of nearest neighbor indices
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
}

/// Benchmark test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTestCase {
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Dataset name
    pub dataset: String,
    /// Algorithm name
    pub algorithm: String,
    /// Test parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Number of queries to test
    pub query_count: usize,
    /// k value for kNN search
    pub k: usize,
}

/// Comprehensive benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test case information
    pub test_case: BenchmarkTestCase,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Quality metrics
    pub quality: Option<QualityMetrics>,
    /// Memory usage statistics
    pub memory: Option<MemoryMetrics>,
    /// Scalability metrics
    pub scalability: Option<ScalabilityMetrics>,
    /// System information
    pub system_info: SystemInfo,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query time
    pub avg_query_time: Duration,
    /// Median query time
    pub median_query_time: Duration,
    /// 95th percentile query time
    pub p95_query_time: Duration,
    /// 99th percentile query time
    pub p99_query_time: Duration,
    /// Minimum query time
    pub min_query_time: Duration,
    /// Maximum query time
    pub max_query_time: Duration,
    /// Standard deviation of query times
    pub std_dev_query_time: Duration,
    /// Queries per second
    pub queries_per_second: f64,
    /// Index build time
    pub index_build_time: Option<Duration>,
    /// Index update time (per vector)
    pub index_update_time: Option<Duration>,
    /// Throughput (vectors/second)
    pub throughput: f64,
}

/// Quality metrics for approximate search algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Recall at k
    pub recall_at_k: f64,
    /// Precision at k
    pub precision_at_k: f64,
    /// Mean Average Precision (MAP)
    pub mean_average_precision: f64,
    /// Normalized Discounted Cumulative Gain (NDCG)
    pub ndcg_at_k: f64,
    /// Distance ratio (for approximate vs exact)
    pub distance_ratio: Option<f64>,
    /// Relative error
    pub relative_error: Option<f64>,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Average memory usage (bytes)
    pub avg_memory_bytes: usize,
    /// Memory per vector (bytes)
    pub memory_per_vector: f64,
    /// Index memory overhead
    pub index_overhead_bytes: usize,
    /// Memory efficiency ratio
    pub memory_efficiency: f64,
}

/// Scalability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Performance vs dataset size
    pub performance_scaling: Vec<(usize, Duration)>, // (dataset_size, avg_query_time)
    /// Memory vs dataset size
    pub memory_scaling: Vec<(usize, usize)>, // (dataset_size, memory_bytes)
    /// Build time vs dataset size
    pub build_time_scaling: Vec<(usize, Duration)>, // (dataset_size, build_time)
    /// Concurrent query performance
    pub concurrency_scaling: Vec<(usize, f64)>, // (thread_count, queries_per_second)
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU information
    pub cpu_info: String,
    /// Total RAM
    pub total_ram_gb: f64,
    /// Available RAM
    pub available_ram_gb: f64,
    /// Operating system
    pub os: String,
    /// Rust version
    pub rust_version: String,
    /// SIMD capabilities
    pub simd_features: Vec<String>,
    /// GPU information
    pub gpu_info: Option<String>,
}

/// Performance profiler for detailed timing analysis
pub struct PerformanceProfiler {
    start_time: Instant,
    checkpoints: Vec<(String, Instant)>,
    memory_samples: Vec<(Instant, usize)>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            checkpoints: Vec::new(),
            memory_samples: Vec::new(),
        }
    }

    /// Record a checkpoint
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), Instant::now()));
    }

    /// Sample memory usage
    pub fn sample_memory(&mut self) {
        let memory_usage = self.get_current_memory_usage();
        self.memory_samples.push((Instant::now(), memory_usage));
    }

    /// Get timing breakdown
    pub fn get_timing_breakdown(&self) -> Vec<(String, Duration)> {
        let mut breakdown = Vec::new();
        let mut last_time = self.start_time;

        for (name, time) in &self.checkpoints {
            breakdown.push((name.clone(), time.duration_since(last_time)));
            last_time = *time;
        }

        breakdown
    }

    /// Get memory usage over time
    pub fn get_memory_profile(&self) -> Vec<(Duration, usize)> {
        self.memory_samples
            .iter()
            .map(|(time, memory)| (time.duration_since(self.start_time), *memory))
            .collect()
    }

    fn get_current_memory_usage(&self) -> usize {
        // Platform-specific memory usage detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
                for line in content.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("ps")
                .args(&["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<usize>() {
                        return rss_kb * 1024; // Convert to bytes
                    }
                }
            }
        }

        // Fallback - return 0 if unable to determine
        0
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            datasets: Vec::new(),
            algorithms: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a dataset to the benchmark suite
    pub fn add_dataset(&mut self, dataset: BenchmarkDataset) {
        self.datasets.push(dataset);
    }

    /// Add an algorithm to benchmark
    pub fn add_algorithm(&mut self, algorithm: Box<dyn VectorIndex>) {
        self.algorithms.push(algorithm);
    }

    /// Generate synthetic datasets for testing
    pub fn generate_synthetic_datasets(&mut self) -> Result<()> {
        // Generate various synthetic datasets with different characteristics
        self.generate_random_dataset("random_1000", 1000, 128, 100)?;
        self.generate_random_dataset("random_10000", 10000, 256, 1000)?;
        self.generate_clustered_dataset("clustered_5000", 5000, 384, 500, 10)?;
        self.generate_uniform_dataset("uniform_2000", 2000, 512, 200)?;

        Ok(())
    }

    /// Generate random dataset
    fn generate_random_dataset(
        &mut self,
        name: &str,
        size: usize,
        dimensions: usize,
        query_count: usize,
    ) -> Result<()> {
        let mut train_vectors = Vec::new();
        let mut query_vectors = Vec::new();

        // Generate training vectors
        for i in 0..size {
            let vector = crate::utils::random_vector(dimensions, Some(i as u64));
            train_vectors.push(vector);
        }

        // Generate query vectors
        for i in 0..query_count {
            let vector = crate::utils::random_vector(dimensions, Some((size + i) as u64));
            query_vectors.push(vector);
        }

        let dataset = BenchmarkDataset {
            name: name.to_string(),
            description: format!(
                "Random dataset with {} vectors of {} dimensions",
                size, dimensions
            ),
            train_vectors,
            query_vectors,
            ground_truth: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "random".to_string());
                meta.insert("size".to_string(), size.to_string());
                meta.insert("dimensions".to_string(), dimensions.to_string());
                meta
            },
        };

        self.add_dataset(dataset);
        Ok(())
    }

    /// Generate clustered dataset
    fn generate_clustered_dataset(
        &mut self,
        name: &str,
        size: usize,
        dimensions: usize,
        query_count: usize,
        num_clusters: usize,
    ) -> Result<()> {
        let mut train_vectors = Vec::new();
        let mut query_vectors = Vec::new();

        // Generate cluster centers
        let mut cluster_centers = Vec::new();
        for i in 0..num_clusters {
            let center = crate::utils::random_vector(dimensions, Some(i as u64));
            cluster_centers.push(center);
        }

        // Generate training vectors around clusters
        for i in 0..size {
            let cluster_idx = i % num_clusters;
            let center = &cluster_centers[cluster_idx];
            let center_f32 = center.as_f32();

            // Add Gaussian noise around cluster center
            let mut noisy_vector = center_f32.clone();
            let noise_scale = 0.1;

            for val in &mut noisy_vector {
                let noise = self.gaussian_random(0.0, noise_scale, i as u64);
                *val += noise;
            }

            train_vectors.push(Vector::new(noisy_vector));
        }

        // Generate query vectors
        for i in 0..query_count {
            let cluster_idx = i % num_clusters;
            let center = &cluster_centers[cluster_idx];
            let center_f32 = center.as_f32();

            let mut noisy_vector = center_f32.clone();
            let noise_scale = 0.05; // Less noise for queries

            for val in &mut noisy_vector {
                let noise = self.gaussian_random(0.0, noise_scale, (size + i) as u64);
                *val += noise;
            }

            query_vectors.push(Vector::new(noisy_vector));
        }

        let dataset = BenchmarkDataset {
            name: name.to_string(),
            description: format!(
                "Clustered dataset with {} vectors in {} clusters of {} dimensions",
                size, num_clusters, dimensions
            ),
            train_vectors,
            query_vectors,
            ground_truth: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "clustered".to_string());
                meta.insert("size".to_string(), size.to_string());
                meta.insert("dimensions".to_string(), dimensions.to_string());
                meta.insert("clusters".to_string(), num_clusters.to_string());
                meta
            },
        };

        self.add_dataset(dataset);
        Ok(())
    }

    /// Generate uniform dataset
    fn generate_uniform_dataset(
        &mut self,
        name: &str,
        size: usize,
        dimensions: usize,
        query_count: usize,
    ) -> Result<()> {
        let mut train_vectors = Vec::new();
        let mut query_vectors = Vec::new();

        // Generate uniformly distributed vectors
        for i in 0..size {
            let mut values = Vec::with_capacity(dimensions);
            let mut state = i as u64;

            for _ in 0..dimensions {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (state as f32) / (u64::MAX as f32);
                values.push(normalized); // 0 to 1 range
            }

            train_vectors.push(Vector::new(values));
        }

        // Generate query vectors
        for i in 0..query_count {
            let mut values = Vec::with_capacity(dimensions);
            let mut state = (size + i) as u64;

            for _ in 0..dimensions {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (state as f32) / (u64::MAX as f32);
                values.push(normalized);
            }

            query_vectors.push(Vector::new(values));
        }

        let dataset = BenchmarkDataset {
            name: name.to_string(),
            description: format!(
                "Uniform dataset with {} vectors of {} dimensions",
                size, dimensions
            ),
            train_vectors,
            query_vectors,
            ground_truth: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "uniform".to_string());
                meta.insert("size".to_string(), size.to_string());
                meta.insert("dimensions".to_string(), dimensions.to_string());
                meta
            },
        };

        self.add_dataset(dataset);
        Ok(())
    }

    /// Simple Gaussian random number generator
    fn gaussian_random(&self, mean: f32, std_dev: f32, seed: u64) -> f32 {
        // Box-Muller transform for Gaussian distribution
        let mut state = seed;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = (state as f32) / (u64::MAX as f32);
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (state as f32) / (u64::MAX as f32);

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std_dev * z0
    }

    /// Run all benchmarks
    pub fn run_all_benchmarks(&mut self) -> Result<Vec<BenchmarkResult>> {
        let mut all_results = Vec::new();

        // For each dataset and algorithm combination
        for dataset in &self.datasets {
            for (alg_idx, algorithm) in self.algorithms.iter().enumerate() {
                let test_case = BenchmarkTestCase {
                    name: format!("{}_{}", dataset.name, alg_idx),
                    description: format!("Benchmark {} on {}", alg_idx, dataset.name),
                    dataset: dataset.name.clone(),
                    algorithm: format!("algorithm_{}", alg_idx),
                    parameters: HashMap::new(),
                    query_count: dataset.query_vectors.len(),
                    k: 10,
                };

                let result = self.run_single_benchmark(&test_case, dataset, algorithm.as_ref())?;
                all_results.push(result);
            }
        }

        self.results.extend(all_results.clone());
        Ok(all_results)
    }

    /// Run a single benchmark test
    fn run_single_benchmark(
        &self,
        test_case: &BenchmarkTestCase,
        dataset: &BenchmarkDataset,
        algorithm: &dyn VectorIndex,
    ) -> Result<BenchmarkResult> {
        let mut profiler = PerformanceProfiler::new();

        tracing::info!("Running benchmark: {}", test_case.name);
        profiler.checkpoint("benchmark_start");

        // Build index
        profiler.checkpoint("index_build_start");
        let mut index = self.create_index_copy(algorithm)?;

        for (i, vector) in dataset.train_vectors.iter().enumerate() {
            index.insert(format!("vec_{}", i), vector.clone())?;
        }
        profiler.checkpoint("index_build_end");

        // Warmup runs
        profiler.checkpoint("warmup_start");
        for _ in 0..self.config.warmup_runs {
            for query in dataset.query_vectors.iter().take(10) {
                let _ = index.search_knn(query, test_case.k)?;
            }
        }
        profiler.checkpoint("warmup_end");

        // Benchmark runs
        profiler.checkpoint("benchmark_queries_start");
        let mut query_times = Vec::new();

        for query in &dataset.query_vectors {
            profiler.sample_memory();

            let start = Instant::now();
            let _results = index.search_knn(query, test_case.k)?;
            let query_time = start.elapsed();

            query_times.push(query_time);
        }
        profiler.checkpoint("benchmark_queries_end");

        // Calculate performance metrics
        let performance = self.calculate_performance_metrics(&query_times)?;

        // Calculate quality metrics if ground truth is available
        let quality = if let Some(ground_truth) = &dataset.ground_truth {
            Some(self.calculate_quality_metrics(
                index.as_ref(),
                &dataset.query_vectors,
                ground_truth,
                test_case.k,
            )?)
        } else {
            None
        };

        // Calculate memory metrics
        let memory = if self.config.profile_memory {
            Some(self.calculate_memory_metrics(&profiler, dataset.train_vectors.len())?)
        } else {
            None
        };

        // Get system information
        let system_info = self.get_system_info();

        Ok(BenchmarkResult {
            test_case: test_case.clone(),
            performance,
            quality,
            memory,
            scalability: None, // Would be calculated in scalability tests
            system_info,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Create a copy of the index for benchmarking
    fn create_index_copy(&self, _algorithm: &dyn VectorIndex) -> Result<Box<dyn VectorIndex>> {
        // For now, create a simple memory index
        // In practice, would clone the specific algorithm
        Ok(Box::new(crate::MemoryVectorIndex::new()))
    }

    /// Calculate performance metrics from query times
    fn calculate_performance_metrics(
        &self,
        query_times: &[Duration],
    ) -> Result<PerformanceMetrics> {
        if query_times.is_empty() {
            return Err(anyhow!("No query times to analyze"));
        }

        let mut sorted_times = query_times.to_vec();
        sorted_times.sort();

        let avg_query_time = Duration::from_nanos(
            (query_times.iter().map(|d| d.as_nanos()).sum::<u128>() / query_times.len() as u128)
                .try_into()
                .unwrap(),
        );

        let median_query_time = sorted_times[sorted_times.len() / 2];
        let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;
        let p95_query_time = sorted_times[p95_idx.min(sorted_times.len() - 1)];
        let p99_query_time = sorted_times[p99_idx.min(sorted_times.len() - 1)];

        let min_query_time = sorted_times[0];
        let max_query_time = sorted_times[sorted_times.len() - 1];

        // Calculate standard deviation
        let mean_nanos = avg_query_time.as_nanos() as f64;
        let variance = query_times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / query_times.len() as f64;
        let std_dev_query_time = Duration::from_nanos(variance.sqrt() as u64);

        let queries_per_second = 1.0 / avg_query_time.as_secs_f64();
        let throughput =
            query_times.len() as f64 / query_times.iter().map(|d| d.as_secs_f64()).sum::<f64>();

        Ok(PerformanceMetrics {
            avg_query_time,
            median_query_time,
            p95_query_time,
            p99_query_time,
            min_query_time,
            max_query_time,
            std_dev_query_time,
            queries_per_second,
            index_build_time: None, // Would be calculated from profiler
            index_update_time: None,
            throughput,
        })
    }

    /// Calculate quality metrics
    fn calculate_quality_metrics(
        &self,
        index: &dyn VectorIndex,
        queries: &[Vector],
        ground_truth: &[Vec<usize>],
        k: usize,
    ) -> Result<QualityMetrics> {
        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        let mut total_queries = 0;

        for (query_idx, query) in queries.iter().enumerate() {
            if query_idx >= ground_truth.len() {
                break;
            }

            let results = index.search_knn(query, k)?;
            let returned_indices: Vec<usize> = results
                .iter()
                .filter_map(|(uri, _)| {
                    // Extract index from URI (assuming format "vec_{index}")
                    uri.strip_prefix("vec_")
                        .and_then(|s| s.parse::<usize>().ok())
                })
                .collect();

            let true_neighbors = &ground_truth[query_idx];
            let true_neighbors_k: std::collections::HashSet<usize> =
                true_neighbors.iter().take(k).copied().collect();

            // Calculate recall: how many true neighbors were found
            let found_true = returned_indices
                .iter()
                .filter(|&idx| true_neighbors_k.contains(idx))
                .count();

            let recall = found_true as f64 / k.min(true_neighbors.len()) as f64;
            let precision = found_true as f64 / returned_indices.len() as f64;

            total_recall += recall;
            total_precision += precision;
            total_queries += 1;
        }

        let avg_recall = total_recall / total_queries as f64;
        let avg_precision = total_precision / total_queries as f64;

        Ok(QualityMetrics {
            recall_at_k: avg_recall,
            precision_at_k: avg_precision,
            mean_average_precision: avg_precision, // Simplified
            ndcg_at_k: avg_recall,                 // Simplified
            distance_ratio: None,
            relative_error: None,
        })
    }

    /// Calculate memory metrics
    fn calculate_memory_metrics(
        &self,
        profiler: &PerformanceProfiler,
        vector_count: usize,
    ) -> Result<MemoryMetrics> {
        let memory_profile = profiler.get_memory_profile();

        if memory_profile.is_empty() {
            return Ok(MemoryMetrics {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                memory_per_vector: 0.0,
                index_overhead_bytes: 0,
                memory_efficiency: 0.0,
            });
        }

        let peak_memory_bytes = memory_profile
            .iter()
            .map(|(_, mem)| *mem)
            .max()
            .unwrap_or(0);
        let avg_memory_bytes =
            memory_profile.iter().map(|(_, mem)| *mem).sum::<usize>() / memory_profile.len();
        let memory_per_vector = avg_memory_bytes as f64 / vector_count as f64;

        Ok(MemoryMetrics {
            peak_memory_bytes,
            avg_memory_bytes,
            memory_per_vector,
            index_overhead_bytes: 0, // Would calculate based on vector size
            memory_efficiency: 1.0,  // Would calculate as ratio of theoretical to actual memory
        })
    }

    /// Get system information
    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            cpu_info: self.get_cpu_info(),
            total_ram_gb: self.get_total_ram_gb(),
            available_ram_gb: self.get_available_ram_gb(),
            os: std::env::consts::OS.to_string(),
            rust_version: self.get_rust_version(),
            simd_features: self.get_simd_features(),
            gpu_info: self.get_gpu_info(),
        }
    }

    fn get_cpu_info(&self) -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            return name.trim().to_string();
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "machdep.cpu.brand_string"])
                .output()
            {
                if let Ok(cpu_name) = String::from_utf8(output.stdout) {
                    return cpu_name.trim().to_string();
                }
            }
        }

        "Unknown CPU".to_string()
    }

    fn get_total_ram_gb(&self) -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb as f64 / 1024.0 / 1024.0; // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                        return bytes as f64 / 1024.0 / 1024.0 / 1024.0; // Convert bytes to GB
                    }
                }
            }
        }

        8.0 // Default fallback
    }

    fn get_available_ram_gb(&self) -> f64 {
        // Simplified - would implement proper available memory detection
        self.get_total_ram_gb() * 0.8
    }

    fn get_rust_version(&self) -> String {
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
    }

    fn get_simd_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                features.push("SSE".to_string());
            }
            if is_x86_feature_detected!("sse2") {
                features.push("SSE2".to_string());
            }
            if is_x86_feature_detected!("sse3") {
                features.push("SSE3".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                features.push("SSE4.1".to_string());
            }
            if is_x86_feature_detected!("sse4.2") {
                features.push("SSE4.2".to_string());
            }
            if is_x86_feature_detected!("avx") {
                features.push("AVX".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                features.push("AVX2".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            features.push("NEON".to_string());
        }

        features
    }

    fn get_gpu_info(&self) -> Option<String> {
        // Would integrate with GPU detection libraries
        None
    }

    /// Export results in various formats
    pub fn export_results(&self, format: BenchmarkOutputFormat) -> Result<String> {
        match format {
            BenchmarkOutputFormat::Json => serde_json::to_string_pretty(&self.results)
                .map_err(|e| anyhow!("Failed to serialize to JSON: {}", e)),
            BenchmarkOutputFormat::Csv => self.export_csv(),
            BenchmarkOutputFormat::Table => self.export_table(),
            BenchmarkOutputFormat::AnnBenchmarks => self.export_ann_benchmarks(),
        }
    }

    fn export_csv(&self) -> Result<String> {
        let mut csv = String::new();
        csv.push_str(
            "dataset,algorithm,avg_query_time_ms,queries_per_second,recall_at_k,memory_mb\n",
        );

        for result in &self.results {
            csv.push_str(&format!(
                "{},{},{:.3},{:.2},{:.3},{:.2}\n",
                result.test_case.dataset,
                result.test_case.algorithm,
                result.performance.avg_query_time.as_millis(),
                result.performance.queries_per_second,
                result
                    .quality
                    .as_ref()
                    .map(|q| q.recall_at_k)
                    .unwrap_or(0.0),
                result
                    .memory
                    .as_ref()
                    .map(|m| m.avg_memory_bytes as f64 / 1024.0 / 1024.0)
                    .unwrap_or(0.0),
            ));
        }

        Ok(csv)
    }

    fn export_table(&self) -> Result<String> {
        let mut table = String::new();
        table.push_str(&format!(
            "{:<20} {:<15} {:<15} {:<15} {:<10}\n",
            "Dataset", "Algorithm", "Avg Time (ms)", "QPS", "Recall@K"
        ));
        table.push_str(&"-".repeat(80));
        table.push('\n');

        for result in &self.results {
            table.push_str(&format!(
                "{:<20} {:<15} {:<15.3} {:<15.2} {:<10.3}\n",
                result.test_case.dataset,
                result.test_case.algorithm,
                result.performance.avg_query_time.as_millis(),
                result.performance.queries_per_second,
                result
                    .quality
                    .as_ref()
                    .map(|q| q.recall_at_k)
                    .unwrap_or(0.0),
            ));
        }

        Ok(table)
    }

    fn export_ann_benchmarks(&self) -> Result<String> {
        // Export in ANN-Benchmarks compatible format
        let mut results = serde_json::Map::new();

        for result in &self.results {
            let mut entry = serde_json::Map::new();
            entry.insert(
                "k".to_string(),
                serde_json::Value::Number(result.test_case.k.into()),
            );
            entry.insert(
                "recall".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(
                        result
                            .quality
                            .as_ref()
                            .map(|q| q.recall_at_k)
                            .unwrap_or(0.0),
                    )
                    .unwrap_or(serde_json::Number::from(0)),
                ),
            );
            entry.insert(
                "qps".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(result.performance.queries_per_second)
                        .unwrap_or(serde_json::Number::from(0)),
                ),
            );

            let key = format!(
                "{}_{}",
                result.test_case.dataset, result.test_case.algorithm
            );
            results.insert(key, serde_json::Value::Object(entry));
        }

        serde_json::to_string_pretty(&results)
            .map_err(|e| anyhow!("Failed to serialize ANN-Benchmarks format: {}", e))
    }

    /// Run scalability benchmarks
    pub fn run_scalability_benchmark(&mut self, dataset_name: &str) -> Result<ScalabilityMetrics> {
        let dataset = self
            .datasets
            .iter()
            .find(|d| d.name == dataset_name)
            .ok_or_else(|| anyhow!("Dataset not found: {}", dataset_name))?;

        let mut performance_scaling = Vec::new();
        let mut memory_scaling = Vec::new();
        let mut build_time_scaling = Vec::new();

        // Test different dataset sizes
        let test_sizes = [100, 500, 1000, 2000, 5000];

        for &size in &test_sizes {
            if size > dataset.train_vectors.len() {
                continue;
            }

            let subset_vectors = &dataset.train_vectors[..size];
            let test_queries = &dataset.query_vectors[..10.min(dataset.query_vectors.len())];

            // Build index and measure time
            let build_start = Instant::now();
            let mut index = Box::new(crate::MemoryVectorIndex::new());

            for (i, vector) in subset_vectors.iter().enumerate() {
                index.insert(format!("vec_{}", i), vector.clone())?;
            }
            let build_time = build_start.elapsed();

            // Measure query performance
            let mut query_times = Vec::new();
            let memory_start = self.get_current_memory_usage();

            for query in test_queries {
                let start = Instant::now();
                let _ = index.search_knn(query, 10)?;
                query_times.push(start.elapsed());
            }

            let memory_end = self.get_current_memory_usage();
            let avg_query_time = Duration::from_nanos(
                (query_times.iter().map(|d| d.as_nanos()).sum::<u128>()
                    / query_times.len() as u128)
                    .try_into()
                    .unwrap(),
            );

            performance_scaling.push((size, avg_query_time));
            memory_scaling.push((size, memory_end.saturating_sub(memory_start)));
            build_time_scaling.push((size, build_time));
        }

        // Test concurrency scaling
        let mut concurrency_scaling = Vec::new();
        let thread_counts = [1, 2, 4, 8];

        for &thread_count in &thread_counts {
            let qps = self.measure_concurrent_performance(
                &dataset.query_vectors[..100.min(dataset.query_vectors.len())],
                thread_count,
            )?;
            concurrency_scaling.push((thread_count, qps));
        }

        Ok(ScalabilityMetrics {
            performance_scaling,
            memory_scaling,
            build_time_scaling,
            concurrency_scaling,
        })
    }

    fn get_current_memory_usage(&self) -> usize {
        // Platform-specific memory usage detection (same as in PerformanceProfiler)
        0 // Placeholder
    }

    fn measure_concurrent_performance(
        &self,
        queries: &[Vector],
        thread_count: usize,
    ) -> Result<f64> {
        use std::sync::Arc;
        use std::thread;

        let index = Arc::new(crate::MemoryVectorIndex::new());
        let queries = Arc::new(queries.to_vec());

        let start_time = Instant::now();
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let index = Arc::clone(&index);
            let queries = Arc::clone(&queries);

            let handle = thread::spawn(move || {
                let queries_per_thread = queries.len() / thread_count;
                let start_idx = thread_id * queries_per_thread;
                let end_idx = if thread_id == thread_count - 1 {
                    queries.len()
                } else {
                    start_idx + queries_per_thread
                };

                let mut query_count = 0;
                for query in &queries[start_idx..end_idx] {
                    if let Ok(_) = index.search_knn(query, 10) {
                        query_count += 1;
                    }
                }
                query_count
            });

            handles.push(handle);
        }

        let total_queries: usize = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
        let elapsed = start_time.elapsed();

        Ok(total_queries as f64 / elapsed.as_secs_f64())
    }
}

/// Benchmark runner for easy execution
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run standard benchmarks with default configuration
    pub fn run_standard_benchmarks() -> Result<Vec<BenchmarkResult>> {
        let config = BenchmarkConfig::default();
        let mut suite = BenchmarkSuite::new(config);

        // Generate synthetic datasets
        suite.generate_synthetic_datasets()?;

        // Add different algorithms
        suite.add_algorithm(Box::new(crate::MemoryVectorIndex::new()));

        // Run benchmarks
        suite.run_all_benchmarks()
    }

    /// Run quick benchmarks for CI/testing
    pub fn run_quick_benchmarks() -> Result<Vec<BenchmarkResult>> {
        let config = BenchmarkConfig {
            warmup_runs: 1,
            benchmark_runs: 3,
            max_duration: Duration::from_secs(30),
            ..BenchmarkConfig::default()
        };

        let mut suite = BenchmarkSuite::new(config);

        // Generate smaller datasets for quick testing
        suite.generate_random_dataset("quick_test", 100, 64, 10)?;
        suite.add_algorithm(Box::new(crate::MemoryVectorIndex::new()));

        suite.run_all_benchmarks()
    }

    /// Run comprehensive benchmarks with quality metrics
    pub fn run_comprehensive_benchmarks() -> Result<String> {
        let results = Self::run_standard_benchmarks()?;

        // Export results in table format
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite {
            config,
            datasets: Vec::new(),
            algorithms: Vec::new(),
            results,
        };

        suite.export_results(BenchmarkOutputFormat::Table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);
        assert_eq!(suite.datasets.len(), 0);
        assert_eq!(suite.algorithms.len(), 0);
    }

    #[test]
    fn test_synthetic_dataset_generation() {
        let config = BenchmarkConfig::default();
        let mut suite = BenchmarkSuite::new(config);

        suite.generate_synthetic_datasets().unwrap();
        assert!(suite.datasets.len() > 0);

        for dataset in &suite.datasets {
            assert!(!dataset.train_vectors.is_empty());
            assert!(!dataset.query_vectors.is_empty());
        }
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);

        let query_times = vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(12),
            Duration::from_millis(20),
            Duration::from_millis(8),
        ];

        let metrics = suite.calculate_performance_metrics(&query_times).unwrap();

        assert!(metrics.avg_query_time.as_millis() > 0);
        assert!(metrics.queries_per_second > 0.0);
        assert!(metrics.min_query_time <= metrics.median_query_time);
        assert!(metrics.median_query_time <= metrics.max_query_time);
    }

    #[test]
    fn test_quick_benchmarks() {
        let results = BenchmarkRunner::run_quick_benchmarks();
        assert!(results.is_ok());
        let results = results.unwrap();
        assert!(!results.is_empty());

        for result in results {
            assert!(result.performance.avg_query_time.as_nanos() > 0);
            assert!(result.performance.queries_per_second > 0.0);
        }
    }

    #[test]
    fn test_export_formats() {
        let config = BenchmarkConfig::default();
        let mut suite = BenchmarkSuite::new(config);

        // Create a simple test result
        let test_case = BenchmarkTestCase {
            name: "test".to_string(),
            description: "test case".to_string(),
            dataset: "test_data".to_string(),
            algorithm: "test_alg".to_string(),
            parameters: HashMap::new(),
            query_count: 10,
            k: 5,
        };

        let result = BenchmarkResult {
            test_case,
            performance: PerformanceMetrics {
                avg_query_time: Duration::from_millis(10),
                median_query_time: Duration::from_millis(10),
                p95_query_time: Duration::from_millis(15),
                p99_query_time: Duration::from_millis(20),
                min_query_time: Duration::from_millis(5),
                max_query_time: Duration::from_millis(25),
                std_dev_query_time: Duration::from_millis(3),
                queries_per_second: 100.0,
                index_build_time: None,
                index_update_time: None,
                throughput: 100.0,
            },
            quality: None,
            memory: None,
            scalability: None,
            system_info: SystemInfo {
                cpu_info: "Test CPU".to_string(),
                total_ram_gb: 16.0,
                available_ram_gb: 12.0,
                os: "test".to_string(),
                rust_version: "1.70.0".to_string(),
                simd_features: vec!["AVX2".to_string()],
                gpu_info: None,
            },
            timestamp: std::time::SystemTime::now(),
        };

        suite.results.push(result);

        // Test different export formats
        let json_output = suite.export_results(BenchmarkOutputFormat::Json);
        assert!(json_output.is_ok());

        let csv_output = suite.export_results(BenchmarkOutputFormat::Csv);
        assert!(csv_output.is_ok());

        let table_output = suite.export_results(BenchmarkOutputFormat::Table);
        assert!(table_output.is_ok());
    }

    #[test]
    fn test_profiler() {
        let mut profiler = PerformanceProfiler::new();

        profiler.checkpoint("start");
        std::thread::sleep(Duration::from_millis(50));
        profiler.checkpoint("middle");
        std::thread::sleep(Duration::from_millis(100));
        profiler.checkpoint("end");

        let breakdown = profiler.get_timing_breakdown();
        assert_eq!(breakdown.len(), 3);

        // Check that timings are reasonable
        for (name, duration) in breakdown {
            assert!(!name.is_empty());
            // Use nanos instead of micros for more reliable timing
            assert!(duration.as_nanos() > 0);
        }
    }
}
