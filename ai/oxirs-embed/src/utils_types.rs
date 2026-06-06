//! Common types, enums, and config structs for embedding utilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Dataset split result
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    pub train: Vec<(String, String, String)>,
    pub validation: Vec<(String, String, String)>,
    pub test: Vec<(String, String, String)>,
}

/// Statistics about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub num_triples: usize,
    pub num_entities: usize,
    pub num_relations: usize,
    pub entity_frequency: HashMap<String, usize>,
    pub relation_frequency: HashMap<String, usize>,
    pub avg_degree: f64,
    pub density: f64,
}

/// Embedding distribution statistics
#[derive(Debug, Clone)]
pub struct EmbeddingDistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub num_parameters: usize,
}

/// Similarity statistics
#[derive(Debug, Clone)]
pub struct SimilarityStats {
    pub mean_similarity: f64,
    pub min_similarity: f64,
    pub max_similarity: f64,
    pub median_similarity: f64,
    pub num_comparisons: usize,
}

/// Graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub num_entities: usize,
    pub num_relations: usize,
    pub num_triples: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
    pub density: f64,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target confidence level (0.0-1.0)
    pub confidence_level: f64,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable detailed timing analysis
    pub enable_detailed_timing: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measurement_iterations: 1000,
            confidence_level: 0.95,
            enable_memory_profiling: true,
            enable_detailed_timing: true,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Average memory usage (bytes)
    pub avg_memory_bytes: usize,
    /// Memory allocations count
    pub allocations: usize,
    /// Memory deallocations count
    pub deallocations: usize,
}

/// Individual benchmark result for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Total number of iterations
    pub iterations: usize,
    /// Total elapsed time
    pub total_duration: Duration,
    /// Average time per operation
    pub avg_duration: Duration,
    /// Minimum time observed
    pub min_duration: Duration,
    /// Maximum time observed
    pub max_duration: Duration,
    /// Standard deviation of durations
    pub std_deviation: Duration,
    /// Operations per second
    pub ops_per_second: f64,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Additional metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Overall benchmark summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Total benchmark duration
    pub total_duration: Duration,
    /// Number of operations benchmarked
    pub total_operations: usize,
    /// Overall throughput (ops/sec)
    pub overall_throughput: f64,
    /// Performance efficiency score (0.0-1.0)
    pub efficiency_score: f64,
    /// Bottleneck analysis
    pub bottlenecks: Vec<String>,
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_name: String,
    pub comparison_name: String,
    pub throughput_improvement: f64,
    pub latency_improvement: f64,
    pub consistency_improvement: f64,
    pub is_improvement: bool,
}

/// Performance regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub throughput_change: f64,
    pub is_regression: bool,
    pub confidence_level: f64,
    pub analysis_notes: Vec<String>,
}

impl Default for RegressionAnalysis {
    fn default() -> Self {
        Self {
            throughput_change: 0.0,
            is_regression: false,
            confidence_level: 0.0,
            analysis_notes: vec!["No historical data available".to_string()],
        }
    }
}
