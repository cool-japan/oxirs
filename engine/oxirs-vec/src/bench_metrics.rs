//! Benchmark metric types and data structures for the advanced benchmarking framework.

use crate::benchmarking::{BenchmarkConfig, BenchmarkDataset, BenchmarkOutputFormat};
use crate::VectorIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Advanced benchmarking configuration with comprehensive options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedBenchmarkConfig {
    /// Base configuration
    pub base_config: BenchmarkConfig,
    /// Statistical confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Minimum number of runs for statistical significance
    pub min_runs: usize,
    /// Maximum coefficient of variation allowed
    pub max_cv: f64,
    /// Enable memory profiling
    pub memory_profiling: bool,
    /// Enable latency distribution analysis
    pub latency_distribution: bool,
    /// Enable throughput testing
    pub throughput_testing: bool,
    /// Enable quality degradation analysis
    pub quality_degradation: bool,
    /// Enable hyperparameter optimization
    pub hyperparameter_optimization: bool,
    /// Enable comparative analysis
    pub comparative_analysis: bool,
    /// ANN-Benchmarks compatibility mode
    pub ann_benchmarks_mode: bool,
    /// Export detailed traces
    pub export_traces: bool,
    /// Parallel execution configuration
    pub parallel_config: ParallelBenchmarkConfig,
}

/// Configuration for parallel benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelBenchmarkConfig {
    /// Number of concurrent threads
    pub num_threads: usize,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// Thread affinity settings
    pub thread_affinity: bool,
    /// Memory bandwidth testing
    pub memory_bandwidth_test: bool,
}

/// Enhanced dataset with comprehensive metadata and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedBenchmarkDataset {
    /// Base dataset information
    pub base_dataset: BenchmarkDataset,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
    /// Quality metrics
    pub quality_metrics: DatasetQualityMetrics,
    /// Intrinsic dimensionality
    pub intrinsic_dimensionality: f32,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
    /// Hubness score
    pub hubness_score: f32,
    /// Local intrinsic dimensionality
    pub local_id: Vec<f32>,
}

/// Comprehensive dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Number of vectors
    pub vector_count: usize,
    /// Dimensionality
    pub dimensions: usize,
    /// Mean vector magnitude
    pub mean_magnitude: f32,
    /// Standard deviation of magnitudes
    pub std_magnitude: f32,
    /// Inter-vector distance statistics
    pub distance_stats: DistanceStatistics,
    /// Nearest neighbor distribution
    pub nn_distribution: Vec<f32>,
    /// Sparsity ratio (if applicable)
    pub sparsity_ratio: Option<f32>,
}

/// Distance statistics for dataset analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceStatistics {
    /// Mean pairwise distance
    pub mean_distance: f32,
    /// Standard deviation of distances
    pub std_distance: f32,
    /// Minimum distance
    pub min_distance: f32,
    /// Maximum distance
    pub max_distance: f32,
    /// Distance distribution percentiles
    pub percentiles: Vec<(f32, f32)>, // (percentile, value)
}

/// Quality metrics for dataset characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetQualityMetrics {
    /// Effective dimensionality
    pub effective_dimensionality: f32,
    /// Concentration measure
    pub concentration_measure: f32,
    /// Outlier ratio
    pub outlier_ratio: f32,
    /// Cluster quality (silhouette score)
    pub cluster_quality: f32,
    /// Manifold quality
    pub manifold_quality: f32,
}

/// Algorithm wrapper for benchmarking
pub struct BenchmarkAlgorithm {
    pub name: String,
    pub description: String,
    pub index: Box<dyn VectorIndex>,
    pub parameters: AlgorithmParameters,
    pub build_time: Option<Duration>,
    pub memory_usage: Option<usize>,
}

/// Algorithm-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParameters {
    /// Parameter map
    pub params: HashMap<String, ParameterValue>,
    /// Search parameters
    pub search_params: HashMap<String, ParameterValue>,
    /// Build parameters
    pub build_params: HashMap<String, ParameterValue>,
}

/// Parameter values with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    IntegerRange(i64, i64, i64), // min, max, step
    FloatRange(f64, f64, f64),   // min, max, step
}

/// Advanced benchmark result with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedBenchmarkResult {
    /// Basic result information
    pub algorithm_name: String,
    pub dataset_name: String,
    pub timestamp: std::time::SystemTime,

    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Quality metrics
    pub quality: QualityMetrics,
    /// Scalability metrics
    pub scalability: ScalabilityMetrics,
    /// Memory metrics
    pub memory: MemoryMetrics,
    /// Statistical analysis
    pub statistics: StatisticalMetrics,

    /// Detailed traces
    pub traces: Option<BenchmarkTraces>,
    /// Error information
    pub errors: Vec<String>,
}

impl Default for AdvancedBenchmarkResult {
    fn default() -> Self {
        Self {
            algorithm_name: String::new(),
            dataset_name: String::new(),
            timestamp: std::time::SystemTime::now(),
            performance: PerformanceMetrics {
                latency: LatencyMetrics {
                    mean_ms: 0.0,
                    std_ms: 0.0,
                    percentiles: std::collections::HashMap::new(),
                    distribution: Vec::new(),
                    max_ms: 0.0,
                    min_ms: 0.0,
                },
                throughput: ThroughputMetrics {
                    qps: 0.0,
                    batch_qps: std::collections::HashMap::new(),
                    concurrent_qps: std::collections::HashMap::new(),
                    saturation_qps: 0.0,
                },
                build_time: BuildTimeMetrics {
                    total_seconds: 0.0,
                    per_vector_ms: 0.0,
                    allocation_seconds: 0.0,
                    construction_seconds: 0.0,
                    optimization_seconds: 0.0,
                },
                index_size: IndexSizeMetrics {
                    total_bytes: 0,
                    per_vector_bytes: 0.0,
                    overhead_ratio: 0.0,
                    compression_ratio: 0.0,
                    serialized_bytes: 0,
                },
            },
            quality: QualityMetrics {
                recall_at_k: std::collections::HashMap::new(),
                precision_at_k: std::collections::HashMap::new(),
                mean_average_precision: 0.0,
                ndcg_at_k: std::collections::HashMap::new(),
                f1_at_k: std::collections::HashMap::new(),
                mean_reciprocal_rank: 0.0,
                quality_degradation: QualityDegradation {
                    recall_latency_tradeoff: Vec::new(),
                    quality_size_tradeoff: Vec::new(),
                    quality_buildtime_tradeoff: Vec::new(),
                },
            },
            scalability: ScalabilityMetrics {
                latency_scaling: Vec::new(),
                memory_scaling: Vec::new(),
                buildtime_scaling: Vec::new(),
                throughput_scaling: Vec::new(),
                scaling_efficiency: 0.0,
            },
            memory: MemoryMetrics {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                allocation_patterns: Vec::new(),
                fragmentation_ratio: 0.0,
                cache_metrics: CacheMetrics {
                    l1_hit_ratio: 0.0,
                    l2_hit_ratio: 0.0,
                    l3_hit_ratio: 0.0,
                    memory_bandwidth_util: 0.0,
                },
            },
            statistics: StatisticalMetrics {
                sample_size: 0,
                confidence_intervals: std::collections::HashMap::new(),
                significance_tests: std::collections::HashMap::new(),
                effect_sizes: std::collections::HashMap::new(),
                power_analysis: PowerAnalysis {
                    power: 0.0,
                    effect_size: 0.0,
                    required_sample_size: 0,
                },
            },
            traces: None,
            errors: Vec::new(),
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Query latency statistics
    pub latency: LatencyMetrics,
    /// Throughput measurements
    pub throughput: ThroughputMetrics,
    /// Build time metrics
    pub build_time: BuildTimeMetrics,
    /// Index size metrics
    pub index_size: IndexSizeMetrics,
}

/// Detailed latency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Mean latency
    pub mean_ms: f64,
    /// Standard deviation
    pub std_ms: f64,
    /// Percentile latencies
    pub percentiles: HashMap<String, f64>, // P50, P95, P99, P99.9
    /// Latency distribution
    pub distribution: Vec<f64>,
    /// Worst-case latency
    pub max_ms: f64,
    /// Best-case latency
    pub min_ms: f64,
}

/// Throughput analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Queries per second
    pub qps: f64,
    /// Batched QPS
    pub batch_qps: HashMap<usize, f64>, // batch_size -> qps
    /// Concurrent QPS
    pub concurrent_qps: HashMap<usize, f64>, // thread_count -> qps
    /// Saturation point
    pub saturation_qps: f64,
}

/// Build time analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTimeMetrics {
    /// Total build time
    pub total_seconds: f64,
    /// Build time per vector
    pub per_vector_ms: f64,
    /// Memory allocation time
    pub allocation_seconds: f64,
    /// Index construction time
    pub construction_seconds: f64,
    /// Optimization time
    pub optimization_seconds: f64,
}

/// Index size analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSizeMetrics {
    /// Total memory usage (bytes)
    pub total_bytes: usize,
    /// Memory per vector (bytes)
    pub per_vector_bytes: f64,
    /// Overhead ratio
    pub overhead_ratio: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Serialized size
    pub serialized_bytes: usize,
}

/// Quality metrics for search accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Recall at different k values
    pub recall_at_k: HashMap<usize, f64>,
    /// Precision at different k values
    pub precision_at_k: HashMap<usize, f64>,
    /// Mean Average Precision
    pub mean_average_precision: f64,
    /// Normalized Discounted Cumulative Gain
    pub ndcg_at_k: HashMap<usize, f64>,
    /// F1 scores
    pub f1_at_k: HashMap<usize, f64>,
    /// Mean Reciprocal Rank
    pub mean_reciprocal_rank: f64,
    /// Quality degradation analysis
    pub quality_degradation: QualityDegradation,
}

/// Quality degradation under different conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDegradation {
    /// Recall vs latency trade-off
    pub recall_latency_tradeoff: Vec<(f64, f64)>, // (recall, latency_ms)
    /// Quality vs index size trade-off
    pub quality_size_tradeoff: Vec<(f64, usize)>, // (recall, size_bytes)
    /// Quality vs build time trade-off
    pub quality_buildtime_tradeoff: Vec<(f64, f64)>, // (recall, build_seconds)
}

/// Scalability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Latency vs dataset size
    pub latency_scaling: Vec<(usize, f64)>, // (dataset_size, latency_ms)
    /// Memory vs dataset size
    pub memory_scaling: Vec<(usize, usize)>, // (dataset_size, memory_bytes)
    /// Build time vs dataset size
    pub buildtime_scaling: Vec<(usize, f64)>, // (dataset_size, build_seconds)
    /// Throughput vs concurrent users
    pub throughput_scaling: Vec<(usize, f64)>, // (concurrent_users, qps)
    /// Scaling efficiency
    pub scaling_efficiency: f64,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Average memory usage
    pub average_memory_mb: f64,
    /// Memory allocation patterns
    pub allocation_patterns: Vec<MemoryAllocation>,
    /// Memory fragmentation
    pub fragmentation_ratio: f64,
    /// Cache hit ratios
    pub cache_metrics: CacheMetrics,
}

/// Memory allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Timestamp
    pub timestamp_ms: u64,
    /// Allocated bytes
    pub allocated_bytes: usize,
    /// Allocation type
    pub allocation_type: String,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// L1 cache hit ratio
    pub l1_hit_ratio: f64,
    /// L2 cache hit ratio
    pub l2_hit_ratio: f64,
    /// L3 cache hit ratio
    pub l3_hit_ratio: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f64,
}

/// Statistical analysis of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetrics {
    /// Sample size
    pub sample_size: usize,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>, // metric -> (lower, upper)
    /// Statistical significance tests
    pub significance_tests: HashMap<String, StatisticalTest>,
    /// Effect sizes
    pub effect_sizes: HashMap<String, f64>,
    /// Power analysis
    pub power_analysis: PowerAnalysis,
}

/// Statistical test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test type (t-test, Mann-Whitney U, etc.)
    pub test_type: String,
    /// P-value
    pub p_value: f64,
    /// Test statistic
    pub test_statistic: f64,
    /// Significant at alpha=0.05
    pub is_significant: bool,
}

/// Power analysis for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Statistical power
    pub power: f64,
    /// Effect size
    pub effect_size: f64,
    /// Required sample size
    pub required_sample_size: usize,
}

/// Detailed benchmark traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTraces {
    /// Query-level traces
    pub query_traces: Vec<QueryTrace>,
    /// System-level traces
    pub system_traces: Vec<SystemTrace>,
    /// Memory traces
    pub memory_traces: Vec<MemoryTrace>,
}

/// Individual query trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTrace {
    /// Query ID
    pub query_id: usize,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
    /// Results returned
    pub results_count: usize,
    /// Distance computations
    pub distance_computations: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Memory allocations
    pub memory_allocations: usize,
}

/// System-level trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTrace {
    /// Timestamp
    pub timestamp: u64,
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// IO operations
    pub io_operations: usize,
    /// Context switches
    pub context_switches: usize,
}

/// Memory trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// Timestamp
    pub timestamp: u64,
    /// Heap usage
    pub heap_usage: usize,
    /// Stack usage
    pub stack_usage: usize,
    /// Page faults
    pub page_faults: usize,
    /// Memory bandwidth
    pub memory_bandwidth: f64,
}

/// Optimization strategies for hyperparameter tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    EvolutionaryOptimization,
    MultiObjective,
}

/// Parameter search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    pub parameter_type: ParameterType,
    pub constraints: Vec<ParameterConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Categorical(Vec<String>),
    Continuous { min: f64, max: f64 },
    Integer { min: i64, max: i64 },
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    GreaterThan(f64),
    LessThan(f64),
    Conditional {
        if_param: String,
        if_value: String,
        then_constraint: Box<ParameterConstraint>,
    },
}

/// Objective function for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveFunction {
    Recall { k: usize, weight: f64 },
    Latency { percentile: f64, weight: f64 },
    Throughput { weight: f64 },
    MemoryUsage { weight: f64 },
    Composite { objectives: Vec<ObjectiveFunction> },
    Pareto { objectives: Vec<ObjectiveFunction> },
}

impl Default for AdvancedBenchmarkConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedBenchmarkConfig {
    pub fn new() -> Self {
        Self {
            base_config: BenchmarkConfig::default(),
            confidence_level: 0.95,
            min_runs: 10,
            max_cv: 0.05, // 5% coefficient of variation
            memory_profiling: true,
            latency_distribution: true,
            throughput_testing: true,
            quality_degradation: true,
            hyperparameter_optimization: false,
            comparative_analysis: true,
            ann_benchmarks_mode: false,
            export_traces: false,
            parallel_config: ParallelBenchmarkConfig {
                num_threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
                numa_aware: false,
                thread_affinity: false,
                memory_bandwidth_test: false,
            },
        }
    }

    pub fn ann_benchmarks_compatible() -> Self {
        let mut config = Self::new();
        config.ann_benchmarks_mode = true;
        config.base_config.output_format = BenchmarkOutputFormat::AnnBenchmarks;
        config.base_config.quality_metrics = true;
        config.comparative_analysis = false;
        config
    }
}
