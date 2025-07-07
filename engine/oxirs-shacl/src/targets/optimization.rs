//! Target optimization types and structures
//!
//! This module contains optimization-related types for SHACL target selection.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for target selection optimization
#[derive(Debug, Clone)]
pub struct TargetOptimizationConfig {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Enable query plan optimization
    pub enable_query_optimization: bool,
    /// Index hint threshold (use index if selectivity < threshold)
    pub index_hint_threshold: f64,
    /// Parallel execution threshold (execute in parallel if cardinality > threshold)
    pub parallel_threshold: usize,
    /// Enable adaptive optimization based on statistics
    pub enable_adaptive_optimization: bool,
    /// Use UNION optimization for batch queries
    pub use_union_optimization: bool,
}

impl Default for TargetOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: 300, // 5 minutes
            max_cache_size: 1000,
            enable_query_optimization: true,
            index_hint_threshold: 0.1,
            parallel_threshold: 10000,
            enable_adaptive_optimization: true,
            use_union_optimization: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of hits
    pub hits: usize,
    /// Number of misses
    pub misses: usize,
    /// Average query time
    pub avg_query_time: Duration,
}

/// Target selection statistics
#[derive(Debug, Clone)]
pub struct TargetSelectionStats {
    /// Total number of target evaluations
    pub total_evaluations: usize,
    /// Total time spent on target selection
    pub total_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average evaluation time
    pub avg_evaluation_time: Duration,
    /// Index usage statistics
    pub index_usage_rate: f64,
    /// Parallel execution rate
    pub parallel_execution_rate: f64,
}

impl Default for TargetSelectionStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_time: Duration::from_millis(0),
            cache_hit_rate: 0.0,
            avg_evaluation_time: Duration::from_millis(0),
            index_usage_rate: 0.0,
            parallel_execution_rate: 0.0,
        }
    }
}

/// Query plan for SPARQL targets
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Optimized SPARQL query
    pub optimized_query: String,
    /// Estimated cardinality
    pub estimated_cardinality: usize,
    /// Index usage recommendations
    pub index_hints: Vec<IndexHint>,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Plan creation time
    pub created_at: Instant,
}

/// Index usage hint
#[derive(Debug, Clone)]
pub struct IndexHint {
    /// Index type
    pub index_type: String,
    /// Estimated selectivity
    pub selectivity: f64,
    /// Cost benefit
    pub cost_benefit: f64,
}

/// Execution strategy for target selection
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Index-driven execution
    IndexDriven,
    /// Hybrid approach
    Hybrid,
}

/// Index usage statistics
#[derive(Debug, Clone)]
pub struct IndexUsageStats {
    /// Number of times used
    pub usage_count: usize,
    /// Average performance gain
    pub avg_performance_gain: f64,
    /// Last used timestamp
    pub last_used: Instant,
}

/// Target cache statistics
#[derive(Debug, Clone)]
pub struct TargetCacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Cache size
    pub cache_size: usize,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

/// Query optimization options
#[derive(Debug, Clone)]
pub struct QueryOptimizationOptions {
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Ensure deterministic ordering of results
    pub deterministic_results: bool,
    /// Use index hints in generated queries
    pub use_index_hints: bool,
    /// Include performance monitoring hints
    pub include_performance_hints: bool,
    /// Use UNION optimization for batch queries
    pub use_union_optimization: bool,
    /// Custom optimization parameters
    pub custom_params: HashMap<String, String>,
}

impl Default for QueryOptimizationOptions {
    fn default() -> Self {
        Self {
            limit: None,
            deterministic_results: false,
            use_index_hints: true,
            include_performance_hints: false,
            use_union_optimization: true,
            custom_params: HashMap::new(),
        }
    }
}

/// Optimized query result
#[derive(Debug, Clone)]
pub struct OptimizedQuery {
    /// The optimized SPARQL query
    pub sparql: String,
    /// Estimated result cardinality
    pub estimated_cardinality: usize,
    /// Recommended execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Index usage hints
    pub index_hints: Vec<IndexHint>,
    /// Time spent on optimization
    pub optimization_time: Duration,
}

/// Execution plan for target selection
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Recommended execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Index usage hints
    pub index_hints: Vec<IndexHint>,
    /// Estimated cardinality
    pub estimated_cardinality: usize,
}

/// Batch query result
#[derive(Debug, Clone)]
pub struct BatchQueryResult {
    /// Individual optimized queries
    pub individual_queries: Vec<OptimizedQuery>,
    /// Optional union query combining all targets
    pub union_query: Option<String>,
    /// Total estimated cardinality across all queries
    pub total_estimated_cardinality: usize,
    /// Time spent on batch optimization
    pub batch_optimization_time: Duration,
}
