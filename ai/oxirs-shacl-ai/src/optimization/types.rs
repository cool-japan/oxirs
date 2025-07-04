//! Type definitions for optimization engine
//!
//! This module contains all shared data structures, enums, and type aliases
//! used across the optimization engine modules.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use oxirs_shacl::{Shape, ShapeId, ValidationReport};
use crate::shape::{AiShape, PropertyConstraint};
use crate::shape_management::{OptimizationOpportunity, PerformanceProfile};

/// Optimized validation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedValidationStrategy {
    /// Graph analysis results
    pub graph_analysis: Option<GraphAnalysis>,

    /// Optimized shape execution order
    pub shape_execution_order: Vec<ShapeExecutionPlan>,

    /// Parallel execution strategy
    pub parallel_execution: Option<ParallelExecutionStrategy>,

    /// Cache optimization strategy
    pub cache_strategy: Option<CacheStrategy>,

    /// Memory optimization strategy
    pub memory_optimization: Option<MemoryOptimization>,

    /// Expected performance improvements
    pub performance_improvements: PerformanceImprovements,
}

impl OptimizedValidationStrategy {
    pub fn new() -> Self {
        Self {
            graph_analysis: None,
            shape_execution_order: Vec::new(),
            parallel_execution: None,
            cache_strategy: None,
            memory_optimization: None,
            performance_improvements: PerformanceImprovements::new(),
        }
    }
}

impl Default for OptimizedValidationStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Shape execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeExecutionPlan {
    /// Shape identifier
    pub shape_id: ShapeId,

    /// Execution order
    pub execution_order: usize,

    /// Estimated complexity
    pub estimated_complexity: u32,

    /// Estimated selectivity
    pub estimated_selectivity: f64,

    /// Shape dependencies
    pub dependencies: Vec<ShapeId>,

    /// Whether this shape can be executed in parallel
    pub parallel_eligible: bool,
}

/// Parallel execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionStrategy {
    /// Groups of shapes that can be executed in parallel
    pub parallel_groups: Vec<Vec<ShapeId>>,

    /// Recommended thread count
    pub recommended_thread_count: u32,

    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,

    /// Synchronization points
    pub synchronization_points: Vec<SynchronizationPoint>,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    CostBased,
}

/// Synchronization point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationPoint {
    /// Point in execution where synchronization is needed
    pub execution_point: String,

    /// Shapes that need to wait
    pub waiting_shapes: Vec<ShapeId>,

    /// Shapes that must complete first
    pub required_shapes: Vec<ShapeId>,
}

/// Cache optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStrategy {
    /// Recommended cache size (MB)
    pub cache_size_mb: u64,

    /// Cache replacement policy
    pub replacement_policy: CacheReplacementPolicy,

    /// Cache partitioning strategy
    pub partitioning_strategy: CachePartitioningStrategy,

    /// TTL for cached results (seconds)
    pub cache_ttl_seconds: u64,
}

/// Cache replacement policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheReplacementPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    Adaptive,
}

/// Cache partitioning strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachePartitioningStrategy {
    None,
    ByShape,
    ByQuery,
    Hybrid,
}

/// Memory optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Recommended heap size (MB)
    pub heap_size_mb: u64,

    /// Memory pooling configuration
    pub memory_pools: Vec<MemoryPool>,

    /// Garbage collection strategy
    pub gc_strategy: GcStrategy,

    /// Streaming threshold (MB)
    pub streaming_threshold_mb: u64,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPool {
    /// Pool type
    pub pool_type: PoolType,

    /// Pool size (MB)
    pub size_mb: u64,

    /// Object size (bytes)
    pub object_size_bytes: u64,
}

/// Pool types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolType {
    SmallObjects,
    LargeObjects,
    Constraints,
    Results,
}

/// Garbage collection strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GcStrategy {
    Generational,
    Concurrent,
    LowLatency,
    Throughput,
}

/// Performance improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovements {
    /// Expected execution time improvement (percentage)
    pub execution_time_improvement: f64,

    /// Expected memory usage reduction (percentage)
    pub memory_usage_reduction: f64,

    /// Expected throughput increase (percentage)
    pub throughput_increase: f64,

    /// Expected latency reduction (percentage)
    pub latency_reduction: f64,
}

impl PerformanceImprovements {
    pub fn new() -> Self {
        Self {
            execution_time_improvement: 0.0,
            memory_usage_reduction: 0.0,
            throughput_increase: 0.0,
            latency_reduction: 0.0,
        }
    }
}

impl Default for PerformanceImprovements {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub recommendation_type: OptimizationRecommendationType,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Description
    pub description: String,

    /// Estimated benefit (0.0 - 1.0)
    pub estimated_benefit: f64,

    /// Implementation effort
    pub implementation_effort: ImplementationEffort,

    /// Affected components
    pub affected_components: Vec<String>,
}

/// Optimization recommendation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationRecommendationType {
    ConstraintReordering,
    ShapeMerging,
    ParallelExecution,
    Caching,
    MemoryOptimization,
    IndexOptimization,
}

/// Recommendation priority (Higher values = Higher priority)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl PartialOrd for RecommendationPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecommendationPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (RecommendationPriority::Critical, RecommendationPriority::Critical) => {
                std::cmp::Ordering::Equal
            }
            (RecommendationPriority::Critical, _) => std::cmp::Ordering::Greater,
            (_, RecommendationPriority::Critical) => std::cmp::Ordering::Less,
            (RecommendationPriority::High, RecommendationPriority::High) => {
                std::cmp::Ordering::Equal
            }
            (
                RecommendationPriority::High,
                RecommendationPriority::Low | RecommendationPriority::Medium,
            ) => std::cmp::Ordering::Greater,
            (
                RecommendationPriority::Low | RecommendationPriority::Medium,
                RecommendationPriority::High,
            ) => std::cmp::Ordering::Less,
            (RecommendationPriority::Medium, RecommendationPriority::Medium) => {
                std::cmp::Ordering::Equal
            }
            (RecommendationPriority::Medium, RecommendationPriority::Low) => {
                std::cmp::Ordering::Greater
            }
            (RecommendationPriority::Low, RecommendationPriority::Medium) => {
                std::cmp::Ordering::Less
            }
            (RecommendationPriority::Low, RecommendationPriority::Low) => std::cmp::Ordering::Equal,
        }
    }
}

/// Implementation effort
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,

    /// Description
    pub description: String,

    /// Severity
    pub severity: BottleneckSeverity,

    /// Impact score (0.0 - 1.0)
    pub impact_score: f64,

    /// Affected operations
    pub affected_operations: Vec<String>,
}

/// Bottleneck types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    ExecutionTime,
    Memory,
    Cpu,
    Io,
    Network,
    Constraint,
}

/// Bottleneck severity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Graph analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysis {
    /// Basic graph statistics
    pub statistics: GraphStatistics,

    /// Connectivity analysis
    pub connectivity_analysis: ConnectivityAnalysis,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,

    /// Analysis time
    pub analysis_time: Duration,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of triples
    pub triple_count: u64,

    /// Number of unique subjects
    pub unique_subjects: u64,

    /// Number of unique predicates
    pub unique_predicates: u64,

    /// Number of unique objects
    pub unique_objects: u64,

    /// Graph density
    pub density: f64,

    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Connectivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityAnalysis {
    /// Number of connected components
    pub connected_components: u32,

    /// Size of largest component
    pub largest_component_size: u32,

    /// Average degree
    pub average_degree: f64,

    /// Graph diameter
    pub diameter: u32,
}

/// Optimization opportunity for validation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOptimizationOpportunity {
    /// Type of opportunity
    pub opportunity_type: OpportunityType,

    /// Description
    pub description: String,

    /// Estimated benefit (0.0 - 1.0)
    pub estimated_benefit: f64,

    /// Implementation complexity
    pub implementation_complexity: ComplexityLevel,
}

/// Opportunity types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpportunityType {
    Parallelization,
    SparseOptimization,
    Indexing,
    Caching,
    Streaming,
}

/// Complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Merge candidate
#[derive(Debug, Clone)]
pub(super) struct MergeCandidate {
    pub shape1_id: ShapeId,
    pub shape2_id: ShapeId,
    pub similarity_score: f64,
    pub merge_strategy: MergeStrategy,
}

/// Merge strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum MergeStrategy {
    Union,
    Intersection,
    Custom,
}

/// Cache analysis
#[derive(Debug, Clone)]
pub(super) struct CacheAnalysis {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub average_access_time: Duration,
    pub memory_usage_mb: u64,
    pub eviction_rate: f64,
}

/// Memory analysis
#[derive(Debug, Clone)]
pub(super) struct MemoryAnalysis {
    pub current_usage_mb: u64,
    pub peak_usage_mb: u64,
    pub gc_frequency: Duration,
    pub allocation_rate_mb_per_sec: f64,
    pub fragmentation_ratio: f64,
}

/// Optimization model state
#[derive(Debug)]
pub(super) struct OptimizationModelState {
    pub version: String,
    pub accuracy: f64,
    pub loss: f64,
    pub training_epochs: usize,
    pub last_training: Option<chrono::DateTime<chrono::Utc>>,
}

impl OptimizationModelState {
    pub fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            accuracy: 0.8,
            loss: 0.2,
            training_epochs: 0,
            last_training: None,
        }
    }
}

/// Cached optimization result
#[derive(Debug, Clone)]
pub(super) struct CachedOptimization {
    pub result: OptimizationResult,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: Duration,
}

impl CachedOptimization {
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let expiry = self.timestamp + chrono::Duration::from_std(self.ttl).unwrap_or_default();
        now > expiry
    }
}

/// Optimization result types
#[derive(Debug, Clone)]
pub(super) enum OptimizationResult {
    OptimizedShapes(Vec<Shape>),
    OptimizedStrategy(OptimizedValidationStrategy),
    OptimizationRecommendations(Vec<OptimizationRecommendation>),
}

/// Optimization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    pub total_optimizations: usize,
    pub shape_optimizations: usize,
    pub strategy_optimizations: usize,
    pub total_optimization_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub model_trained: bool,
}

/// Training data for optimization models
#[derive(Debug, Clone)]
pub struct OptimizationTrainingData {
    pub examples: Vec<OptimizationExample>,
    pub validation_examples: Vec<OptimizationExample>,
}

/// Training example for optimization
#[derive(Debug, Clone)]
pub struct OptimizationExample {
    pub shapes: Vec<Shape>,
    pub validation_data: Vec<ValidationReport>,
    pub expected_optimization: OptimizedValidationStrategy,
    pub context_metadata: HashMap<String, String>,
}
