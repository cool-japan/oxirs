//! Numeric/vector math utilities: dot products, norms, distances, dataset statistics,
//! embedding analysis, graph analysis, progress tracking, and performance utilities.

use crate::utils_types::{
    BenchmarkComparison, BenchmarkConfig, BenchmarkResult, BenchmarkSummary, DatasetStatistics,
    EmbeddingDistributionStats, GraphMetrics, MemoryStats, RegressionAnalysis, SimilarityStats,
};
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

/// Compute dataset statistics
pub fn compute_dataset_statistics(triples: &[(String, String, String)]) -> DatasetStatistics {
    let mut entities = HashSet::new();
    let mut relations = HashSet::new();
    let mut entity_frequency = HashMap::new();
    let mut relation_frequency = HashMap::new();

    for (subject, predicate, object) in triples {
        entities.insert(subject.clone());
        entities.insert(object.clone());
        relations.insert(predicate.clone());

        *entity_frequency.entry(subject.clone()).or_insert(0) += 1;
        *entity_frequency.entry(object.clone()).or_insert(0) += 1;
        *relation_frequency.entry(predicate.clone()).or_insert(0) += 1;
    }

    let num_entities = entities.len();
    let num_relations = relations.len();
    let num_triples = triples.len();

    let avg_degree = if num_entities > 0 {
        (num_triples * 2) as f64 / num_entities as f64
    } else {
        0.0
    };

    let max_possible_edges = num_entities * num_entities;
    let density = if max_possible_edges > 0 {
        num_triples as f64 / max_possible_edges as f64
    } else {
        0.0
    };

    DatasetStatistics {
        num_triples,
        num_entities,
        num_relations,
        entity_frequency,
        relation_frequency,
        avg_degree,
        density,
    }
}

/// Embedding dimension analysis utilities
pub mod embedding_analysis {
    use super::*;

    /// Analyze embedding distribution
    pub fn analyze_embedding_distribution(embeddings: &Array2<f64>) -> EmbeddingDistributionStats {
        let flat_values: Vec<f64> = embeddings.iter().cloned().collect();

        let mean = flat_values.iter().sum::<f64>() / flat_values.len() as f64;
        let variance =
            flat_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / flat_values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = flat_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted_values[0];
        let max_val = sorted_values[sorted_values.len() - 1];
        let median = sorted_values[sorted_values.len() / 2];

        EmbeddingDistributionStats {
            mean,
            std_dev,
            variance,
            min: min_val,
            max: max_val,
            median,
            num_parameters: embeddings.len(),
        }
    }

    /// Compute embedding norms
    pub fn compute_embedding_norms(embeddings: &Array2<f64>) -> Vec<f64> {
        embeddings
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect()
    }

    /// Analyze embedding similarities
    pub fn analyze_embedding_similarities(
        embeddings: &Array2<f64>,
        sample_size: usize,
    ) -> SimilarityStats {
        let num_embeddings = embeddings.nrows();
        let mut similarities = Vec::new();

        let sample_size = sample_size.min(num_embeddings * (num_embeddings - 1) / 2);
        let mut rng = Random::default();

        for _ in 0..sample_size {
            let i = rng.random_range(0..num_embeddings);
            let j = rng.random_range(0..num_embeddings);

            if i != j {
                let emb_i = embeddings.row(i);
                let emb_j = embeddings.row(j);
                let similarity = cosine_similarity_array(&emb_i.to_owned(), &emb_j.to_owned());
                similarities.push(similarity);
            }
        }

        similarities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let min_similarity = similarities[0];
        let max_similarity = similarities[similarities.len() - 1];
        let median_similarity = similarities[similarities.len() / 2];

        SimilarityStats {
            mean_similarity,
            min_similarity,
            max_similarity,
            median_similarity,
            num_comparisons: similarities.len(),
        }
    }

    /// Cosine similarity between two ndarray vectors
    fn cosine_similarity_array(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Graph analysis utilities
pub mod graph_analysis {
    use super::*;

    /// Compute graph metrics for knowledge graph
    pub fn compute_graph_metrics(triples: &[(String, String, String)]) -> GraphMetrics {
        let estimated_entities = triples.len();
        let estimated_relations = triples.len() / 10;

        let mut entity_degrees: HashMap<String, usize> = HashMap::with_capacity(estimated_entities);
        let mut relation_counts: HashMap<String, usize> =
            HashMap::with_capacity(estimated_relations);
        let mut entities = HashSet::with_capacity(estimated_entities);

        for (subject, predicate, object) in triples {
            entities.insert(subject.clone());
            entities.insert(object.clone());

            *entity_degrees.entry(subject.clone()).or_insert(0) += 1;
            *entity_degrees.entry(object.clone()).or_insert(0) += 1;
            *relation_counts.entry(predicate.clone()).or_insert(0) += 1;
        }

        let num_entities = entities.len();
        let num_relations = relation_counts.len();
        let num_triples = triples.len();

        let degrees: Vec<usize> = entity_degrees.values().cloned().collect();
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        let max_degree = degrees.iter().max().cloned().unwrap_or(0);
        let min_degree = degrees.iter().min().cloned().unwrap_or(0);

        GraphMetrics {
            num_entities,
            num_relations,
            num_triples,
            avg_degree,
            max_degree,
            min_degree,
            density: num_triples as f64 / (num_entities * num_entities) as f64,
        }
    }
}

/// Progress tracking utility
#[derive(Debug)]
pub struct ProgressTracker {
    total: usize,
    current: usize,
    start_time: Instant,
    last_update: Instant,
    update_interval: Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        let now = Instant::now();
        Self {
            total,
            current: 0,
            start_time: now,
            last_update: now,
            update_interval: Duration::from_secs(1),
        }
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
        let now = Instant::now();
        if now.duration_since(self.last_update) >= self.update_interval {
            self.print_progress();
            self.last_update = now;
        }
    }

    fn print_progress(&self) {
        let percentage = (self.current as f64 / self.total as f64) * 100.0;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;
        println!(
            "Progress: {}/{} ({:.1}%) - {:.1} items/sec",
            self.current, self.total, percentage, rate
        );
    }

    /// Finish and print final statistics
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.total as f64 / elapsed;
        println!(
            "Completed: {} items in {:.2}s ({:.1} items/sec)",
            self.total, elapsed, rate
        );
    }
}

/// High-precision timer for micro-benchmarking
pub struct PrecisionTimer {
    start_time: Instant,
    lap_times: Vec<Duration>,
}

impl Default for PrecisionTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecisionTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            lap_times: Vec::new(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.lap_times.clear();
    }

    /// Record a lap time
    pub fn lap(&mut self) -> Duration {
        let lap_duration = self.start_time.elapsed();
        self.lap_times.push(lap_duration);
        lap_duration
    }

    /// Stop timing and return final duration
    pub fn stop(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get all recorded lap times
    pub fn lap_times(&self) -> &[Duration] {
        &self.lap_times
    }
}

/// Benchmarking framework for embedding operations
pub struct EmbeddingBenchmark {
    config: BenchmarkConfig,
    results: BTreeMap<String, BenchmarkResult>,
}

impl EmbeddingBenchmark {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: BTreeMap::new(),
        }
    }

    /// Benchmark a function with comprehensive timing and memory analysis
    pub fn benchmark<F, T>(&mut self, name: &str, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let _ = operation()?;
        }

        let mut durations = Vec::with_capacity(self.config.measurement_iterations);
        let mut memory_snapshots = Vec::new();
        let mut result = None;

        for i in 0..self.config.measurement_iterations {
            let memory_before = self.get_memory_usage();
            let start = Instant::now();

            let op_result = operation()?;

            let duration = start.elapsed();
            let memory_after = self.get_memory_usage();

            durations.push(duration);

            if self.config.enable_memory_profiling {
                memory_snapshots.push((memory_before, memory_after));
            }

            if i == 0 {
                result = Some(op_result);
            }
        }

        let total_duration: Duration = durations.iter().sum();
        let avg_duration = total_duration / durations.len() as u32;
        let min_duration = *durations
            .iter()
            .min()
            .expect("durations should not be empty");
        let max_duration = *durations
            .iter()
            .max()
            .expect("durations should not be empty");

        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - avg_duration.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;
        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);

        let ops_per_second = 1_000_000_000.0 / avg_duration.as_nanos() as f64;

        let memory_stats = if self.config.enable_memory_profiling && !memory_snapshots.is_empty() {
            let peak_memory = memory_snapshots
                .iter()
                .map(|(_, after)| after.peak_memory_bytes)
                .max()
                .unwrap_or(0);

            let avg_memory = memory_snapshots
                .iter()
                .map(|(before, after)| (before.avg_memory_bytes + after.avg_memory_bytes) / 2)
                .sum::<usize>()
                / memory_snapshots.len();

            MemoryStats {
                peak_memory_bytes: peak_memory,
                avg_memory_bytes: avg_memory,
                allocations: memory_snapshots.len(),
                deallocations: 0,
            }
        } else {
            MemoryStats {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                allocations: 0,
                deallocations: 0,
            }
        };

        let benchmark_result = BenchmarkResult {
            operation: name.to_string(),
            iterations: self.config.measurement_iterations,
            total_duration,
            avg_duration,
            min_duration,
            max_duration,
            std_deviation,
            ops_per_second,
            memory_stats,
            custom_metrics: HashMap::new(),
        };

        self.results.insert(name.to_string(), benchmark_result);
        result.ok_or_else(|| anyhow!("Failed to capture benchmark result"))
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> BenchmarkSuite {
        let total_duration = self.results.values().map(|r| r.total_duration).sum();
        let total_operations = self.results.len();
        let overall_throughput =
            self.results.values().map(|r| r.ops_per_second).sum::<f64>() / total_operations as f64;
        let efficiency_score = self.calculate_efficiency_score();
        let bottlenecks = self.identify_bottlenecks();

        let summary = BenchmarkSummary {
            total_duration,
            total_operations,
            overall_throughput,
            efficiency_score,
            bottlenecks,
        };

        BenchmarkSuite {
            results: self.results.clone(),
            summary,
            config: self.config.clone(),
        }
    }

    fn calculate_efficiency_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let consistency_scores: Vec<f64> = self
            .results
            .values()
            .map(|result| {
                let cv =
                    result.std_deviation.as_nanos() as f64 / result.avg_duration.as_nanos() as f64;
                1.0 / (1.0 + cv)
            })
            .collect();
        consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
    }

    fn identify_bottlenecks(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        for (name, result) in &self.results {
            let cv = result.std_deviation.as_nanos() as f64 / result.avg_duration.as_nanos() as f64;
            if cv > 0.2 {
                bottlenecks.push(format!("High variance in {}: {:.2}% CV", name, cv * 100.0));
            }
        }

        let avg_throughput = self.results.values().map(|r| r.ops_per_second).sum::<f64>()
            / self.results.len() as f64;

        for (name, result) in &self.results {
            if result.ops_per_second < avg_throughput * 0.5 {
                bottlenecks.push(format!(
                    "Slow operation {}: {:.0} ops/sec",
                    name, result.ops_per_second
                ));
            }
        }
        bottlenecks
    }

    fn get_memory_usage(&self) -> MemoryStats {
        MemoryStats {
            peak_memory_bytes: 0,
            avg_memory_bytes: 0,
            allocations: 0,
            deallocations: 0,
        }
    }
}

/// Benchmark suite result
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub results: BTreeMap<String, BenchmarkResult>,
    pub summary: BenchmarkSummary,
    pub config: BenchmarkConfig,
}

/// Utility functions for performance analysis
pub mod analysis {
    use super::*;

    /// Compare two benchmark results
    pub fn compare_benchmarks(
        baseline: &BenchmarkResult,
        comparison: &BenchmarkResult,
    ) -> BenchmarkComparison {
        let throughput_improvement =
            (comparison.ops_per_second - baseline.ops_per_second) / baseline.ops_per_second;

        let latency_improvement = (baseline.avg_duration.as_nanos() as f64
            - comparison.avg_duration.as_nanos() as f64)
            / baseline.avg_duration.as_nanos() as f64;

        let consistency_improvement = {
            let baseline_cv =
                baseline.std_deviation.as_nanos() as f64 / baseline.avg_duration.as_nanos() as f64;
            let comparison_cv = comparison.std_deviation.as_nanos() as f64
                / comparison.avg_duration.as_nanos() as f64;
            (baseline_cv - comparison_cv) / baseline_cv
        };

        BenchmarkComparison {
            baseline_name: baseline.operation.clone(),
            comparison_name: comparison.operation.clone(),
            throughput_improvement,
            latency_improvement,
            consistency_improvement,
            is_improvement: throughput_improvement > 0.0 && latency_improvement > 0.0,
        }
    }

    /// Generate performance regression analysis
    pub fn analyze_regression(
        historical_results: &[BenchmarkResult],
        current_result: &BenchmarkResult,
    ) -> RegressionAnalysis {
        if historical_results.is_empty() {
            return RegressionAnalysis::default();
        }

        let historical_avg_throughput = historical_results
            .iter()
            .map(|r| r.ops_per_second)
            .sum::<f64>()
            / historical_results.len() as f64;

        let throughput_change =
            (current_result.ops_per_second - historical_avg_throughput) / historical_avg_throughput;
        let is_regression = throughput_change < -0.05;

        RegressionAnalysis {
            throughput_change,
            is_regression,
            confidence_level: 0.95,
            analysis_notes: if is_regression {
                vec!["Performance regression detected".to_string()]
            } else {
                vec!["Performance within expected range".to_string()]
            },
        }
    }
}

/// Type alias for batch processor function
type ProcessorFn<T> = Box<dyn Fn(&[T]) -> Result<()> + Send + Sync>;

/// Memory-efficient batch processor for large datasets
pub struct BatchProcessor<T> {
    batch_size: usize,
    current_batch: Vec<T>,
    processor_fn: ProcessorFn<T>,
}

impl<T> BatchProcessor<T> {
    pub fn new<F>(batch_size: usize, processor_fn: F) -> Self
    where
        F: Fn(&[T]) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            batch_size,
            current_batch: Vec::with_capacity(batch_size),
            processor_fn: Box::new(processor_fn),
        }
    }

    pub fn add(&mut self, item: T) -> Result<()> {
        self.current_batch.push(item);
        if self.current_batch.len() >= self.batch_size {
            return self.flush();
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.current_batch.is_empty() {
            (self.processor_fn)(&self.current_batch)?;
            self.current_batch.clear();
        }
        Ok(())
    }
}

/// Enhanced memory monitoring for embedding operations
#[derive(Debug, Clone)]
pub struct MemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
    allocations: usize,
    deallocations: usize,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }

    pub fn record_allocation(&mut self, size: usize) {
        self.current_usage += size;
        self.allocations += 1;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.deallocations += 1;
    }

    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}
