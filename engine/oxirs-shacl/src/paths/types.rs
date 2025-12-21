//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;
use oxirs_core::model::{NamedNode, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Query plan for path evaluation with caching
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PathQueryPlan {
    /// The optimized query
    pub query: String,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// When this plan was created (skip serialization for Instant)
    #[cfg_attr(feature = "serde", serde(skip, default = "std::time::Instant::now"))]
    pub created_at: std::time::Instant,
    /// How many times this plan has been used
    pub usage_count: usize,
    /// Average execution time in nanoseconds
    pub average_execution_time_nanos: u64,
    /// Whether this plan uses SPARQL property paths
    pub uses_native_sparql: bool,
}
/// Optimization strategies for property path evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathOptimizationStrategy {
    /// Use native SPARQL property path queries
    SparqlPath,
    /// Use programmatic evaluation
    Programmatic,
    /// Use hybrid approach (SPARQL for simple parts, programmatic for complex)
    Hybrid,
}
/// Statistics about property path evaluation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCacheStats {
    pub entries: usize,
    pub total_values: usize,
}
/// Query plan for property path evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPathQueryPlan {
    /// Generated SPARQL query
    pub query: String,
    /// Execution strategy to use
    pub execution_strategy: PathExecutionStrategy,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Estimated complexity
    pub estimated_complexity: usize,
    /// Optimization hints applied
    pub optimization_hints: PathOptimizationHints,
    /// Cache key for this plan
    pub cache_key: String,
}
/// Cached path evaluation result with metadata
#[derive(Debug, Clone)]
pub struct CachedPathResult {
    /// The cached result values
    pub values: Vec<Term>,
    /// When this result was cached
    pub cached_at: std::time::Instant,
    /// How many times this result has been accessed
    pub access_count: usize,
    /// Last access time
    pub last_accessed: std::time::Instant,
    /// Estimated freshness of the result
    pub freshness_score: f64,
    /// Size estimate for memory management
    pub estimated_size_bytes: usize,
}
impl CachedPathResult {
    pub fn new(values: Vec<Term>) -> Self {
        let now = std::time::Instant::now();
        let estimated_size = values.len() * 64;
        Self {
            values,
            cached_at: now,
            access_count: 1,
            last_accessed: now,
            freshness_score: 1.0,
            estimated_size_bytes: estimated_size,
        }
    }
    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::Instant::now();
        let age_seconds = self.cached_at.elapsed().as_secs() as f64;
        let recency_factor = 1.0 / (1.0 + age_seconds / 3600.0);
        let popularity_factor = (self.access_count as f64).ln() / 10.0;
        self.freshness_score = recency_factor * 0.7 + popularity_factor.min(1.0) * 0.3;
    }
    pub fn is_fresh(&self, max_age: std::time::Duration) -> bool {
        self.cached_at.elapsed() < max_age
    }
}
/// Property path optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptimizationHints {
    /// Cache simple predicate path results
    pub cache_simple_paths: bool,
    /// Cache complex path results
    pub cache_complex_paths: bool,
    /// Maximum cache size for path results
    pub max_cache_size: usize,
    /// Parallel evaluation threshold
    pub parallel_threshold: usize,
    /// Maximum recursion depth for cyclic paths
    pub max_recursion_depth: usize,
    /// Maximum intermediate results
    pub max_intermediate_results: usize,
}
/// Performance statistics for property path evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPathStats {
    pub cache_entries: usize,
    pub total_cached_results: usize,
}
/// SHACL property path types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Simple property path (single predicate)
    Predicate(NamedNode),
    /// Inverse property path (^predicate)
    Inverse(Box<PropertyPath>),
    /// Sequence path (path1 / path2 / ...)
    Sequence(Vec<PropertyPath>),
    /// Alternative path (path1 | path2 | ...)
    Alternative(Vec<PropertyPath>),
    /// Zero-or-more path (path*)
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more path (path+)
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one path (path?)
    ZeroOrOne(Box<PropertyPath>),
}
impl PropertyPath {
    /// Create a simple predicate path
    pub fn predicate(predicate: NamedNode) -> Self {
        PropertyPath::Predicate(predicate)
    }
    /// Create an inverse path
    pub fn inverse(path: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(path))
    }
    /// Create a sequence path
    pub fn sequence(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Sequence(paths)
    }
    /// Create an alternative path
    pub fn alternative(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Alternative(paths)
    }
    /// Create a zero-or-more path
    pub fn zero_or_more(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(path))
    }
    /// Create a one-or-more path
    pub fn one_or_more(path: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(path))
    }
    /// Create a zero-or-one path
    pub fn zero_or_one(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(path))
    }
    /// Check if this is a simple predicate path
    pub fn is_predicate(&self) -> bool {
        matches!(self, PropertyPath::Predicate(_))
    }
    /// Get the predicate if this is a simple predicate path
    pub fn as_predicate(&self) -> Option<&NamedNode> {
        match self {
            PropertyPath::Predicate(p) => Some(p),
            _ => None,
        }
    }
    /// Check if this path involves complex operations (non-predicate)
    pub fn is_complex(&self) -> bool {
        !self.is_predicate()
    }
    /// Generate SPARQL property path syntax for this path
    pub fn to_sparql_path(&self) -> Result<String> {
        match self {
            PropertyPath::Predicate(predicate) => Ok(format!("<{}>", predicate.as_str())),
            PropertyPath::Inverse(inner_path) => Ok(format!("^({})", inner_path.to_sparql_path()?)),
            PropertyPath::Sequence(paths) => {
                let path_strs: Result<Vec<String>> =
                    paths.iter().map(|p| p.to_sparql_path()).collect();
                Ok(path_strs?.join(" / "))
            }
            PropertyPath::Alternative(paths) => {
                let path_strs: Result<Vec<String>> =
                    paths.iter().map(|p| p.to_sparql_path()).collect();
                Ok(format!("({})", path_strs?.join(" | ")))
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                Ok(format!("({})*", inner_path.to_sparql_path()?))
            }
            PropertyPath::OneOrMore(inner_path) => {
                Ok(format!("({})+", inner_path.to_sparql_path()?))
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                Ok(format!("({})?", inner_path.to_sparql_path()?))
            }
        }
    }
    /// Check if this path can be efficiently represented as a SPARQL property path
    pub fn can_use_sparql_path(&self) -> bool {
        match self {
            PropertyPath::Predicate(_) => true,
            PropertyPath::Inverse(inner) => inner.can_use_sparql_path(),
            PropertyPath::Sequence(paths) => {
                paths.len() <= 5 && paths.iter().all(|p| p.can_use_sparql_path())
            }
            PropertyPath::Alternative(paths) => {
                paths.len() <= 10 && paths.iter().all(|p| p.can_use_sparql_path())
            }
            PropertyPath::ZeroOrMore(_) => true,
            PropertyPath::OneOrMore(_) => true,
            PropertyPath::ZeroOrOne(inner) => inner.can_use_sparql_path(),
        }
    }
    /// Estimate the complexity of this path for optimization
    pub fn complexity(&self) -> usize {
        match self {
            PropertyPath::Predicate(_) => 1,
            PropertyPath::Inverse(path) => path.complexity() + 1,
            PropertyPath::Sequence(paths) => {
                paths.iter().map(|p| p.complexity()).sum::<usize>() + 1
            }
            PropertyPath::Alternative(paths) => {
                paths.iter().map(|p| p.complexity()).max().unwrap_or(0) + 1
            }
            PropertyPath::ZeroOrMore(path) => path.complexity() * 10,
            PropertyPath::OneOrMore(path) => path.complexity() * 8,
            PropertyPath::ZeroOrOne(path) => path.complexity() + 1,
        }
    }
}
/// Execution strategy for property path evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathExecutionStrategy {
    /// Execute using direct SPARQL property path queries
    DirectSparql,
    /// Execute using programmatic evaluation
    Programmatic,
    /// Execute using hybrid approach
    HybridExecution,
}
/// An optimized property path with analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPropertyPath {
    pub original_path: PropertyPath,
    pub optimization_strategy: PathOptimizationStrategy,
    pub estimated_complexity: usize,
    pub estimated_cost: f64,
    pub can_use_sparql_path: bool,
}
/// Property path validation context
#[derive(Debug, Clone)]
pub struct PathValidationContext {
    /// Current recursion depth
    pub depth: usize,
    /// Visited nodes (for cycle detection)
    pub visited: HashSet<Term>,
    /// Path being evaluated
    pub current_path: PropertyPath,
    /// Performance statistics
    pub stats: PathEvaluationStats,
}
impl PathValidationContext {
    pub fn new(path: PropertyPath) -> Self {
        Self {
            depth: 0,
            visited: HashSet::new(),
            current_path: path,
            stats: PathEvaluationStats::default(),
        }
    }
}
/// Property path evaluation performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathEvaluationStats {
    pub total_evaluations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_values_found: usize,
    pub avg_values_per_evaluation: f64,
    pub max_recursion_depth_reached: usize,
    pub query_plan_cache_hits: usize,
    pub query_plan_cache_misses: usize,
    pub average_query_execution_time: std::time::Duration,
    pub total_query_execution_time: std::time::Duration,
}
impl PathEvaluationStats {
    pub fn record_evaluation(&mut self, values_found: usize, cache_hit: bool, depth: usize) {
        self.total_evaluations += 1;
        self.total_values_found += values_found;
        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        if depth > self.max_recursion_depth_reached {
            self.max_recursion_depth_reached = depth;
        }
        self.avg_values_per_evaluation =
            self.total_values_found as f64 / self.total_evaluations as f64;
    }
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_evaluations as f64
        }
    }
}
/// Validation result for property paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathValidationResult {
    /// Whether the path is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Path complexity
    pub complexity: usize,
    /// Whether path can use SPARQL
    pub can_use_sparql: bool,
    /// Estimated cost
    pub estimated_cost: f64,
}
/// Configuration for path result caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCacheConfig {
    /// Maximum number of cached results
    pub max_cache_entries: usize,
    /// Maximum time to keep results in cache
    pub max_cache_age: std::time::Duration,
    /// Whether to enable intelligent cache eviction
    pub intelligent_eviction: bool,
    /// Minimum access count to keep in cache during pressure
    pub min_access_threshold: usize,
    /// Whether to cache negative results (empty results)
    pub cache_negative_results: bool,
}
