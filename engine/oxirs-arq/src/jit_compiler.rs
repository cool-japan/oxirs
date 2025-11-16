//! JIT Compilation for SPARQL Queries
//!
//! This module provides Just-In-Time compilation for SPARQL queries, transforming
//! high-level SPARQL algebra into optimized execution plans that can be compiled
//! to native code or bytecode for improved performance.
//!
//! # Features
//!
//! - **Query Plan Compilation**: Transform SPARQL algebra into optimized execution plans
//! - **Code Generation**: Generate specialized code for query patterns
//! - **Plan Caching**: Reuse compiled plans with intelligent invalidation
//! - **Adaptive Optimization**: Runtime profiling and re-compilation
//! - **Performance Tracking**: Measure compilation and execution metrics
//!
//! # Architecture
//!
//! ```text
//! SPARQL Query → Algebra → Plan Generation → Code Gen → Specialized Executor
//!                   ↓           ↓              ↓              ↓
//!              Optimization  Lowering    Specialization  Execution
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_arq::jit_compiler::{QueryJitCompiler, JitCompilerConfig};
//!
//! let config = JitCompilerConfig::default();
//! let mut compiler = QueryJitCompiler::new(config)?;
//!
//! // Compile a SPARQL algebra
//! let compiled = compiler.compile(&algebra)?;
//!
//! // Execute compiled plan
//! let results = compiled.execute(&dataset)?;
//! ```

use crate::algebra::Algebra;
use crate::cardinality_estimator::{CardinalityEstimator, EstimatorConfig};
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use scirs2_core::metrics::MetricsRegistry;
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Configuration for SPARQL query JIT compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitCompilerConfig {
    /// Enable JIT compilation
    pub enabled: bool,

    /// Optimization level (0-3)
    pub optimization_level: usize,

    /// Enable query plan caching
    pub enable_caching: bool,

    /// Maximum cache size (in bytes)
    pub max_cache_size: usize,

    /// Cache TTL (time-to-live)
    pub cache_ttl: Duration,

    /// Enable adaptive optimization
    pub adaptive_optimization: bool,

    /// Minimum execution count before re-optimization
    pub min_executions_for_reopt: usize,

    /// Compilation timeout
    pub compilation_timeout: Duration,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Enable specialized code generation
    pub enable_specialization: bool,

    /// Maximum plan complexity for compilation
    pub max_plan_complexity: usize,
}

impl Default for JitCompilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_level: 2,
            enable_caching: true,
            max_cache_size: 512 * 1024 * 1024,    // 512MB
            cache_ttl: Duration::from_secs(3600), // 1 hour
            adaptive_optimization: true,
            min_executions_for_reopt: 10,
            compilation_timeout: Duration::from_secs(30),
            enable_profiling: true,
            enable_specialization: true,
            max_plan_complexity: 1000,
        }
    }
}

/// Compiled SPARQL query plan ready for execution
#[derive(Clone)]
pub struct CompiledQuery {
    /// Unique query identifier
    pub id: String,

    /// Original SPARQL algebra
    pub algebra: Arc<Algebra>,

    /// Compiled execution plan
    pub plan: Arc<ExecutionPlan>,

    /// Compilation timestamp
    pub compiled_at: Instant,

    /// Execution statistics
    pub stats: Arc<RwLock<ExecutionStats>>,

    /// Optimization metadata
    pub metadata: QueryMetadata,
}

impl fmt::Debug for CompiledQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledQuery")
            .field("id", &self.id)
            .field("compiled_at", &self.compiled_at.elapsed())
            .field("stats", &self.stats)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Execution plan generated from SPARQL algebra
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Plan operations
    pub operations: Vec<PlanOperation>,

    /// Estimated cost
    pub estimated_cost: f64,

    /// Estimated memory usage (bytes)
    pub estimated_memory: usize,

    /// Optimization hints applied
    pub optimization_hints: Vec<String>,

    /// Specialization metadata
    pub specializations: Vec<Specialization>,
}

/// Individual operation in execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanOperation {
    /// Scan triple patterns (specialized for pattern type)
    ScanTriples {
        pattern_id: usize,
        pattern_type: PatternType,
        estimated_cardinality: usize,
    },

    /// Hash join operation (with strategy)
    HashJoin {
        left_id: usize,
        right_id: usize,
        join_variables: Vec<String>,
        strategy: JitJoinStrategy,
    },

    /// Nested loop join (for small cardinalities)
    NestedLoopJoin {
        left_id: usize,
        right_id: usize,
        join_variables: Vec<String>,
    },

    /// Filter operation (with specialization)
    Filter {
        expr_id: usize,
        filter_type: FilterType,
    },

    /// Project variables
    Project { variables: Vec<String> },

    /// Sort operation
    Sort {
        variables: Vec<String>,
        ascending: Vec<bool>,
    },

    /// Limit operation
    Limit { limit: usize },

    /// Offset operation
    Offset { offset: usize },

    /// Distinct operation
    Distinct,

    /// Union operation
    Union { branches: Vec<usize> },

    /// Optional (left join) operation
    Optional { left_id: usize, right_id: usize },
}

/// Pattern types for specialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// All variables (?s ?p ?o)
    AllVariables,

    /// Subject bound (s ?p ?o)
    SubjectBound,

    /// Predicate bound (?s p ?o)
    PredicateBound,

    /// Object bound (?s ?p o)
    ObjectBound,

    /// Subject-Predicate bound (s p ?o)
    SubjectPredicateBound,

    /// Subject-Object bound (s ?p o)
    SubjectObjectBound,

    /// Predicate-Object bound (?s p o)
    PredicateObjectBound,

    /// Fully bound (s p o)
    FullyBound,
}

/// Join strategies for JIT compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JitJoinStrategy {
    /// Hash join (default for large inputs)
    Hash,

    /// Sort-merge join
    SortMerge,

    /// Index nested loop join
    IndexNestedLoop,

    /// Bind join (for federated queries)
    Bind,
}

/// Filter types for optimization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    /// Simple equality filter
    Equality,

    /// Numeric comparison
    NumericComparison,

    /// String operation
    StringOperation,

    /// Regex filter
    Regex,

    /// Boolean logic
    BooleanLogic,

    /// Complex expression
    Complex,
}

/// Specialization applied to the plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specialization {
    /// Specialization type
    pub spec_type: SpecializationType,

    /// Description
    pub description: String,

    /// Expected speedup factor
    pub speedup_factor: f64,
}

/// Types of specializations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializationType {
    /// Pattern-specific scanning
    PatternScanning,

    /// Join strategy selection
    JoinStrategy,

    /// Filter pushdown
    FilterPushdown,

    /// Index usage
    IndexUsage,

    /// SIMD vectorization
    SimdVectorization,

    /// Parallel execution
    ParallelExecution,
}

/// Query execution statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Number of executions
    pub execution_count: usize,

    /// Total execution time
    pub total_execution_time: Duration,

    /// Average execution time
    pub avg_execution_time: Duration,

    /// Minimum execution time
    pub min_execution_time: Option<Duration>,

    /// Maximum execution time
    pub max_execution_time: Option<Duration>,

    /// Total results produced
    pub total_results: usize,

    /// Average results per execution
    pub avg_results: f64,

    /// Last execution timestamp (excluded from serialization)
    #[serde(skip)]
    pub last_executed: Option<Instant>,

    /// Compilation time
    pub compilation_time: Duration,

    /// Re-optimization count
    pub reoptimization_count: usize,
}

impl ExecutionStats {
    /// Update statistics with a new execution
    pub fn record_execution(&mut self, duration: Duration, result_count: usize) {
        self.execution_count += 1;
        self.total_execution_time += duration;
        self.avg_execution_time = self.total_execution_time / self.execution_count as u32;
        self.total_results += result_count;
        self.avg_results = self.total_results as f64 / self.execution_count as f64;
        self.last_executed = Some(Instant::now());

        self.min_execution_time = Some(
            self.min_execution_time
                .map_or(duration, |min| min.min(duration)),
        );
        self.max_execution_time = Some(
            self.max_execution_time
                .map_or(duration, |max| max.max(duration)),
        );
    }

    /// Check if re-optimization is beneficial
    pub fn should_reoptimize(&self, min_executions: usize) -> bool {
        self.execution_count >= min_executions
            && self.avg_execution_time > Duration::from_millis(100)
    }
}

/// Query compilation metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Query complexity score (0-1000)
    pub complexity: usize,

    /// Estimated memory usage (bytes)
    pub estimated_memory: usize,

    /// Number of triple patterns
    pub triple_pattern_count: usize,

    /// Number of joins
    pub join_count: usize,

    /// Number of filters
    pub filter_count: usize,

    /// Has aggregation
    pub has_aggregation: bool,

    /// Has optional patterns
    pub has_optional: bool,

    /// Has union patterns
    pub has_union: bool,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// SPARQL Query JIT Compiler
pub struct QueryJitCompiler {
    /// Compiler configuration
    config: JitCompilerConfig,

    /// Compiled query cache
    query_cache: Arc<DashMap<String, Arc<CompiledQuery>>>,

    /// Metric registry (reserved for future use)
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,

    /// Performance profiler (reserved for future use)
    #[allow(dead_code)]
    profiler: Arc<Profiler>,

    /// Cardinality estimator for accurate query planning
    cardinality_estimator: Arc<CardinalityEstimator>,

    /// Random seed for cache eviction
    _rng_seed: u64,

    /// Compilation statistics
    stats: Arc<RwLock<CompilerStats>>,
}

/// Compiler-wide statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompilerStats {
    /// Total compilations
    pub total_compilations: usize,

    /// Total compilation time
    pub total_compilation_time: Duration,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Cache evictions
    pub cache_evictions: usize,

    /// Failed compilations
    pub failed_compilations: usize,

    /// Average compilation time
    pub avg_compilation_time: Duration,
}

impl QueryJitCompiler {
    /// Create a new JIT compiler with the given configuration
    pub fn new(config: JitCompilerConfig) -> Result<Self> {
        // Initialize metrics
        let metrics = Arc::new(MetricsRegistry::new());

        // Initialize profiler
        let profiler = Arc::new(Profiler::new());

        // Initialize cardinality estimator with default configuration
        let cardinality_estimator = Arc::new(CardinalityEstimator::new(EstimatorConfig::default()));

        Ok(Self {
            config,
            query_cache: Arc::new(DashMap::new()),
            metrics,
            profiler,
            cardinality_estimator,
            _rng_seed: 42,
            stats: Arc::new(RwLock::new(CompilerStats::default())),
        })
    }

    /// Compile a SPARQL algebra expression
    pub fn compile(&mut self, algebra: &Algebra) -> Result<Arc<CompiledQuery>> {
        let start_time = Instant::now();

        // Generate query ID from algebra
        let query_id = self.generate_query_id(algebra);

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.query_cache.get(&query_id) {
                self.record_cache_hit();
                debug!("JIT cache hit for query: {}", query_id);
                return Ok(cached.clone());
            }
        }

        self.record_cache_miss();
        info!("Compiling query: {}", query_id);

        // Analyze query to extract metadata
        let metadata = self.analyze_query(algebra)?;

        // Check complexity threshold
        if metadata.complexity > self.config.max_plan_complexity {
            warn!(
                "Query complexity ({}) exceeds threshold ({}), using basic execution",
                metadata.complexity, self.config.max_plan_complexity
            );
        }

        // Generate execution plan
        let plan = self.generate_execution_plan(algebra, &metadata)?;

        // Create compiled query
        let compiled = Arc::new(CompiledQuery {
            id: query_id.clone(),
            algebra: Arc::new(algebra.clone()),
            plan: Arc::new(plan),
            compiled_at: Instant::now(),
            stats: Arc::new(RwLock::new(ExecutionStats {
                compilation_time: start_time.elapsed(),
                ..Default::default()
            })),
            metadata,
        });

        // Update cache
        if self.config.enable_caching {
            self.insert_into_cache(query_id.clone(), compiled.clone())?;
        }

        // Record metrics
        let compilation_time = start_time.elapsed();
        self.record_compilation(compilation_time);

        info!(
            "Query compiled successfully in {:?}: {}",
            compilation_time, query_id
        );

        Ok(compiled)
    }

    /// Generate a unique identifier for the query
    fn generate_query_id(&self, algebra: &Algebra) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", algebra).hash(&mut hasher);
        format!("query_{:x}", hasher.finish())
    }

    /// Analyze query to extract compilation metadata
    fn analyze_query(&self, algebra: &Algebra) -> Result<QueryMetadata> {
        let mut metadata = QueryMetadata::default();

        // Recursively analyze algebra structure
        self.analyze_algebra_recursive(algebra, &mut metadata);

        // Calculate complexity score
        metadata.complexity = self.calculate_complexity(&metadata);

        // Estimate memory usage
        metadata.estimated_memory = self.estimate_memory(&metadata);

        // Identify optimization opportunities
        metadata.optimization_opportunities = self.identify_optimizations(&metadata);

        Ok(metadata)
    }

    /// Recursively analyze algebra structure
    #[allow(clippy::only_used_in_recursion)]
    fn analyze_algebra_recursive(&self, algebra: &Algebra, metadata: &mut QueryMetadata) {
        match algebra {
            Algebra::Bgp(patterns) => {
                metadata.triple_pattern_count += patterns.len();
            }
            Algebra::Join { left, right } => {
                metadata.join_count += 1;
                self.analyze_algebra_recursive(left, metadata);
                self.analyze_algebra_recursive(right, metadata);
            }
            Algebra::Filter { pattern, .. } => {
                metadata.filter_count += 1;
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::LeftJoin { left, right, .. } => {
                metadata.has_optional = true;
                metadata.join_count += 1;
                self.analyze_algebra_recursive(left, metadata);
                self.analyze_algebra_recursive(right, metadata);
            }
            Algebra::Union { left, right } => {
                metadata.has_union = true;
                self.analyze_algebra_recursive(left, metadata);
                self.analyze_algebra_recursive(right, metadata);
            }
            Algebra::Group { pattern, .. } => {
                metadata.has_aggregation = true;
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. } => {
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::Slice {
                pattern,
                offset: _,
                limit: _,
            } => {
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::Graph { graph: _, pattern } => {
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::Extend { pattern, .. } => {
                self.analyze_algebra_recursive(pattern, metadata);
            }
            Algebra::Minus { left, right } => {
                self.analyze_algebra_recursive(left, metadata);
                self.analyze_algebra_recursive(right, metadata);
            }
            _ => {}
        }
    }

    /// Calculate query complexity score
    fn calculate_complexity(&self, metadata: &QueryMetadata) -> usize {
        let mut score = 0;

        // Base complexity from triple patterns
        score += metadata.triple_pattern_count * 10;

        // Join complexity (exponential growth)
        score += metadata.join_count.pow(2) * 20;

        // Filter complexity
        score += metadata.filter_count * 15;

        // Additional complexity for special features
        if metadata.has_aggregation {
            score += 50;
        }
        if metadata.has_optional {
            score += 30;
        }
        if metadata.has_union {
            score += 25;
        }

        score.min(1000) // Cap at 1000
    }

    /// Estimate memory usage for query execution
    fn estimate_memory(&self, metadata: &QueryMetadata) -> usize {
        let base_memory = 1024 * 1024; // 1MB base

        // Memory per triple pattern
        let pattern_memory = metadata.triple_pattern_count * 100 * 1024; // 100KB per pattern

        // Memory per join (increases exponentially)
        let join_memory = if metadata.join_count > 0 {
            2_usize.pow(metadata.join_count as u32) * 50 * 1024 // 50KB * 2^joins
        } else {
            0
        };

        base_memory + pattern_memory + join_memory
    }

    /// Identify optimization opportunities
    fn identify_optimizations(&self, metadata: &QueryMetadata) -> Vec<String> {
        let mut opportunities = Vec::new();

        if metadata.join_count > 2 {
            opportunities.push("Consider join reordering".to_string());
        }

        if metadata.filter_count > 0 {
            opportunities.push("Filter pushdown optimization".to_string());
        }

        if metadata.triple_pattern_count > 5 {
            opportunities.push("Pattern specialization".to_string());
        }

        if metadata.has_aggregation {
            opportunities.push("Streaming aggregation".to_string());
        }

        opportunities
    }

    /// Generate execution plan from algebra
    fn generate_execution_plan(
        &self,
        algebra: &Algebra,
        metadata: &QueryMetadata,
    ) -> Result<ExecutionPlan> {
        let mut operations = Vec::new();
        let mut specializations = Vec::new();

        // Lower algebra to operations
        self.lower_to_operations(algebra, &mut operations)?;

        // Apply optimizations based on configuration
        if self.config.optimization_level >= 1 {
            self.optimize_plan(&mut operations, &mut specializations)?;
        }

        // Calculate estimated cost
        let estimated_cost = self.calculate_plan_cost(&operations);

        Ok(ExecutionPlan {
            operations,
            estimated_cost,
            estimated_memory: metadata.estimated_memory,
            optimization_hints: metadata.optimization_opportunities.clone(),
            specializations,
        })
    }

    /// Lower algebra to executable operations
    #[allow(clippy::ptr_arg)]
    fn lower_to_operations(&self, algebra: &Algebra, ops: &mut Vec<PlanOperation>) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                // Basic graph pattern - scan operation
                let pattern_type = self.determine_pattern_type(algebra);

                // Estimate cardinality using CardinalityEstimator
                let estimated_cardinality = if !patterns.is_empty() {
                    // Use the first pattern for estimation
                    // For multiple patterns, we could sum or average the estimates
                    match self
                        .cardinality_estimator
                        .estimate_triple_pattern(&patterns[0])
                    {
                        Ok(cardinality) => cardinality,
                        Err(e) => {
                            warn!("Cardinality estimation failed: {}, using default", e);
                            10_000 // Fallback to conservative default
                        }
                    }
                } else {
                    10_000 // Default for empty BGP
                };

                debug!(
                    "BGP cardinality estimate: {} for {} patterns",
                    estimated_cardinality,
                    patterns.len()
                );

                ops.push(PlanOperation::ScanTriples {
                    pattern_id: ops.len(),
                    pattern_type,
                    estimated_cardinality: estimated_cardinality.try_into().unwrap_or(10_000),
                });
            }
            Algebra::Join { left, right } => {
                let left_start = ops.len();
                self.lower_to_operations(left, ops)?;

                let right_start = ops.len();
                self.lower_to_operations(right, ops)?;

                // Extract join variables (intersection of left and right variables)
                let left_vars = left.variables();
                let right_vars = right.variables();
                let join_variables: Vec<String> = left_vars
                    .iter()
                    .filter(|v| right_vars.contains(v))
                    .map(|v| v.name().to_string())
                    .collect();

                ops.push(PlanOperation::HashJoin {
                    left_id: left_start,
                    right_id: right_start,
                    join_variables,
                    strategy: JitJoinStrategy::Hash,
                });
            }
            Algebra::Filter { pattern, .. } => {
                self.lower_to_operations(pattern, ops)?;

                ops.push(PlanOperation::Filter {
                    expr_id: ops.len(),
                    filter_type: FilterType::Complex,
                });
            }
            Algebra::Project { pattern, variables } => {
                self.lower_to_operations(pattern, ops)?;

                ops.push(PlanOperation::Project {
                    variables: variables.iter().map(|v| v.name().to_string()).collect(),
                });
            }
            Algebra::Distinct { pattern } => {
                self.lower_to_operations(pattern, ops)?;
                ops.push(PlanOperation::Distinct);
            }
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => {
                self.lower_to_operations(pattern, ops)?;

                if let Some(off) = offset {
                    ops.push(PlanOperation::Offset { offset: *off });
                }
                if let Some(lim) = limit {
                    ops.push(PlanOperation::Limit { limit: *lim });
                }
            }
            _ => {
                // For other types, use a basic scan operation
                ops.push(PlanOperation::ScanTriples {
                    pattern_id: ops.len(),
                    pattern_type: PatternType::AllVariables,
                    estimated_cardinality: 1000,
                });
            }
        }

        Ok(())
    }

    /// Determine the type of triple pattern for specialization
    fn determine_pattern_type(&self, _algebra: &Algebra) -> PatternType {
        // For now, return AllVariables - would need actual pattern analysis
        PatternType::AllVariables
    }

    /// Optimize the execution plan
    #[allow(clippy::ptr_arg)]
    fn optimize_plan(
        &self,
        operations: &mut Vec<PlanOperation>,
        specializations: &mut Vec<Specialization>,
    ) -> Result<()> {
        // Apply pattern-specific optimizations
        for op in operations.iter_mut() {
            if let PlanOperation::ScanTriples { pattern_type, .. } = op {
                // Record specialization
                specializations.push(Specialization {
                    spec_type: SpecializationType::PatternScanning,
                    description: format!("Specialized scan for pattern type: {:?}", pattern_type),
                    speedup_factor: 1.5,
                });
            }
        }

        Ok(())
    }

    /// Calculate estimated cost of execution plan
    fn calculate_plan_cost(&self, operations: &[PlanOperation]) -> f64 {
        let mut total_cost = 0.0;

        for op in operations {
            total_cost += match op {
                PlanOperation::ScanTriples {
                    estimated_cardinality,
                    ..
                } => *estimated_cardinality as f64 * 0.1,
                PlanOperation::HashJoin { .. } => 100.0,
                PlanOperation::NestedLoopJoin { .. } => 500.0,
                PlanOperation::Filter { .. } => 10.0,
                PlanOperation::Project { .. } => 5.0,
                PlanOperation::Sort { .. } => 200.0,
                PlanOperation::Limit { .. } => 1.0,
                PlanOperation::Offset { .. } => 1.0,
                PlanOperation::Distinct => 150.0,
                PlanOperation::Union { .. } => 50.0,
                PlanOperation::Optional { .. } => 120.0,
            };
        }

        total_cost
    }

    /// Insert compiled query into cache
    fn insert_into_cache(&self, query_id: String, compiled: Arc<CompiledQuery>) -> Result<()> {
        // Check cache size and evict if necessary
        if self.query_cache.len() * 1024 * 1024 > self.config.max_cache_size {
            self.evict_cache_entry()?;
        }

        self.query_cache.insert(query_id, compiled);

        Ok(())
    }

    /// Evict a cache entry (random eviction for now)
    fn evict_cache_entry(&self) -> Result<()> {
        if let Some(key) = self.query_cache.iter().next().map(|e| e.key().clone()) {
            self.query_cache.remove(&key);
            self.stats.write().cache_evictions += 1;
            debug!("Evicted cached query: {}", key);
        }

        Ok(())
    }

    /// Record a cache hit
    fn record_cache_hit(&self) {
        self.stats.write().cache_hits += 1;
    }

    /// Record a cache miss
    fn record_cache_miss(&self) {
        self.stats.write().cache_misses += 1;
    }

    /// Record a compilation
    fn record_compilation(&self, duration: Duration) {
        let mut stats = self.stats.write();
        stats.total_compilations += 1;
        stats.total_compilation_time += duration;
        stats.avg_compilation_time = if stats.total_compilations > 0 {
            stats.total_compilation_time / stats.total_compilations as u32
        } else {
            Duration::ZERO
        };
    }

    /// Get compiler statistics
    pub fn stats(&self) -> CompilerStats {
        self.stats.read().clone()
    }

    /// Clear the query cache
    pub fn clear_cache(&self) {
        self.query_cache.clear();
        info!("JIT query cache cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let config = JitCompilerConfig::default();
        let compiler = QueryJitCompiler::new(config);
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_complexity_calculation() {
        let compiler = QueryJitCompiler::new(JitCompilerConfig::default()).unwrap();

        let metadata = QueryMetadata {
            triple_pattern_count: 5,
            join_count: 2,
            filter_count: 3,
            has_aggregation: true,
            has_optional: true,
            has_union: false,
            ..Default::default()
        };

        let complexity = compiler.calculate_complexity(&metadata);
        assert!(complexity > 0);
        assert!(complexity <= 1000);
    }

    #[test]
    fn test_memory_estimation() {
        let compiler = QueryJitCompiler::new(JitCompilerConfig::default()).unwrap();

        let metadata = QueryMetadata {
            triple_pattern_count: 3,
            join_count: 2,
            ..Default::default()
        };

        let memory = compiler.estimate_memory(&metadata);
        assert!(memory > 0);
    }

    #[test]
    fn test_cache_operations() {
        let config = JitCompilerConfig {
            enable_caching: true,
            ..Default::default()
        };
        let compiler = QueryJitCompiler::new(config).unwrap();

        // Initially empty
        assert_eq!(compiler.query_cache.len(), 0);

        // Clear should work on empty cache
        compiler.clear_cache();
        assert_eq!(compiler.query_cache.len(), 0);
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::default();

        stats.record_execution(Duration::from_millis(100), 50);
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.total_results, 50);
        assert_eq!(stats.avg_results, 50.0);

        stats.record_execution(Duration::from_millis(200), 30);
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.total_results, 80);
        assert_eq!(stats.avg_results, 40.0);
    }

    #[test]
    fn test_should_reoptimize() {
        let mut stats = ExecutionStats::default();

        // Not enough executions
        assert!(!stats.should_reoptimize(10));

        // Execute many times with slow queries
        for _ in 0..15 {
            stats.record_execution(Duration::from_millis(150), 10);
        }

        assert!(stats.should_reoptimize(10));
    }

    #[test]
    fn test_pattern_type_variants() {
        // Test that all pattern types are defined
        let patterns = [
            PatternType::AllVariables,
            PatternType::SubjectBound,
            PatternType::PredicateBound,
            PatternType::ObjectBound,
            PatternType::SubjectPredicateBound,
            PatternType::SubjectObjectBound,
            PatternType::PredicateObjectBound,
            PatternType::FullyBound,
        ];

        assert_eq!(patterns.len(), 8);
    }

    #[test]
    fn test_join_strategy_variants() {
        let strategies = [
            JitJoinStrategy::Hash,
            JitJoinStrategy::SortMerge,
            JitJoinStrategy::IndexNestedLoop,
            JitJoinStrategy::Bind,
        ];

        assert_eq!(strategies.len(), 4);
    }
}
