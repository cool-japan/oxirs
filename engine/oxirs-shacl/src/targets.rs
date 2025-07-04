//! SHACL target selection implementation
//!
//! This module handles target node selection according to SHACL specification.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use oxirs_core::{
    model::{BlankNode, NamedNode, RdfTerm, Term, Triple},
    ConcreteStore, OxirsError, Store,
};

use crate::{Result, ShaclError, SHACL_VOCAB};

/// SHACL target types for selecting nodes to validate
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// sh:targetClass - selects all instances of a class
    Class(NamedNode),

    /// sh:targetNode - selects specific nodes
    Node(Term),

    /// sh:targetObjectsOf - selects objects of a property
    ObjectsOf(NamedNode),

    /// sh:targetSubjectsOf - selects subjects of a property  
    SubjectsOf(NamedNode),

    /// sh:target with SPARQL SELECT query
    Sparql(SparqlTarget),

    /// Implicit target (shape IRI used as class)
    Implicit(NamedNode),
}

/// SPARQL-based target definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlTarget {
    /// SPARQL SELECT query that returns target nodes
    pub query: String,

    /// Optional prefixes for the query
    pub prefixes: Option<String>,
}

impl Target {
    /// Create a class target
    pub fn class(class_iri: NamedNode) -> Self {
        Target::Class(class_iri)
    }

    /// Create a node target
    pub fn node(node: Term) -> Self {
        Target::Node(node)
    }

    /// Create an objects-of target
    pub fn objects_of(property: NamedNode) -> Self {
        Target::ObjectsOf(property)
    }

    /// Create a subjects-of target
    pub fn subjects_of(property: NamedNode) -> Self {
        Target::SubjectsOf(property)
    }

    /// Create a SPARQL target
    pub fn sparql(query: String, prefixes: Option<String>) -> Self {
        Target::Sparql(SparqlTarget { query, prefixes })
    }

    /// Create an implicit target
    pub fn implicit(class_iri: NamedNode) -> Self {
        Target::Implicit(class_iri)
    }
}

/// Target selector for finding nodes that match target definitions
#[derive(Debug)]
pub struct TargetSelector {
    /// Cache for target results to improve performance
    cache: std::collections::HashMap<String, CachedTargetResult>,

    /// Optimization settings
    optimization_config: TargetOptimizationConfig,

    /// Performance statistics
    stats: TargetSelectionStats,

    /// Query plan cache for SPARQL targets
    query_plan_cache: std::collections::HashMap<String, QueryPlan>,

    /// Index usage statistics for adaptive optimization
    index_usage_stats: std::collections::HashMap<String, IndexUsageStats>,
}

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
        }
    }
}

/// Cached target result
#[derive(Debug, Clone)]
struct CachedTargetResult {
    /// Target nodes
    nodes: HashSet<Term>,
    /// Cache timestamp
    cached_at: std::time::Instant,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone)]
struct CacheStats {
    /// Number of hits
    hits: usize,
    /// Number of misses
    misses: usize,
    /// Average query time
    avg_query_time: std::time::Duration,
}

/// Target selection statistics
#[derive(Debug, Clone)]
pub struct TargetSelectionStats {
    /// Total number of target evaluations
    pub total_evaluations: usize,
    /// Total time spent on target selection
    pub total_time: std::time::Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average evaluation time
    pub avg_evaluation_time: std::time::Duration,
    /// Index usage statistics
    pub index_usage_rate: f64,
    /// Parallel execution rate
    pub parallel_execution_rate: f64,
}

/// Query plan for SPARQL targets
#[derive(Debug, Clone)]
struct QueryPlan {
    /// Optimized SPARQL query
    optimized_query: String,
    /// Estimated cardinality
    estimated_cardinality: usize,
    /// Index usage recommendations
    index_hints: Vec<IndexHint>,
    /// Execution strategy
    execution_strategy: ExecutionStrategy,
    /// Plan creation time
    created_at: std::time::Instant,
}

/// Index usage hint
#[derive(Debug, Clone)]
struct IndexHint {
    /// Index type
    index_type: String,
    /// Estimated selectivity
    selectivity: f64,
    /// Cost benefit
    cost_benefit: f64,
}

/// Execution strategy for target selection
#[derive(Debug, Clone)]
enum ExecutionStrategy {
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
struct IndexUsageStats {
    /// Number of times used
    usage_count: usize,
    /// Average performance gain
    avg_performance_gain: f64,
    /// Last used timestamp
    last_used: std::time::Instant,
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
    pub custom_params: std::collections::HashMap<String, String>,
}

impl Default for QueryOptimizationOptions {
    fn default() -> Self {
        Self {
            limit: None,
            deterministic_results: false,
            use_index_hints: true,
            include_performance_hints: false,
            use_union_optimization: true,
            custom_params: std::collections::HashMap::new(),
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
    pub optimization_time: std::time::Duration,
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
    pub batch_optimization_time: std::time::Duration,
}

impl Default for TargetSelectionStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_time: std::time::Duration::from_millis(0),
            cache_hit_rate: 0.0,
            avg_evaluation_time: std::time::Duration::from_millis(0),
            index_usage_rate: 0.0,
            parallel_execution_rate: 0.0,
        }
    }
}

impl TargetSelector {
    /// Create a new target selector
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: TargetOptimizationConfig::default(),
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
        }
    }

    /// Create a new target selector with custom optimization config
    pub fn with_config(config: TargetOptimizationConfig) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: config,
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
        }
    }

    /// Generate efficient SPARQL query for target selection
    pub fn generate_target_query(
        &self,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        match target {
            Target::Class(class_iri) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }
            Target::Node(node) => {
                // For specific nodes, just return the node
                match node {
                    Term::NamedNode(nn) => {
                        Ok(format!("SELECT (<{}> AS ?target) WHERE {{ }}", nn.as_str()))
                    }
                    Term::BlankNode(bn) => {
                        Ok(format!("SELECT (?{} AS ?target) WHERE {{ }}", bn.as_str()))
                    }
                    Term::Literal(lit) => Ok(format!("SELECT ({} AS ?target) WHERE {{ }}", lit)),
                    _ => Err(ShaclError::ValidationEngine(
                        "Unsupported term type for node target".to_string(),
                    )),
                }
            }
            Target::ObjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?s <{}> ?target .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::SubjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <{}> ?o .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::Sparql(sparql_target) => {
                // Return the user-provided SPARQL query with optional prefixes
                let mut query = String::new();
                if let Some(prefixes) = &sparql_target.prefixes {
                    query.push_str(prefixes);
                    query.push('\n');
                }
                query.push_str(&sparql_target.query);
                Ok(query)
            }
            Target::Implicit(class_iri) => {
                // Same as class target
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }
        }
    }

    /// Optimize SPARQL query for target selection with index hints
    pub fn optimize_target_query(&mut self, query: &str, target: &Target) -> Result<String> {
        if !self.optimization_config.enable_query_optimization {
            return Ok(query.to_string());
        }

        let mut optimized_query = query.to_string();

        // Add index hints based on target type
        match target {
            Target::Class(_) | Target::Implicit(_) => {
                // Add hint for type index if available
                if self.should_use_index_hint("type_index") {
                    optimized_query = format!("# Use type index\n{}", optimized_query);
                }
            }
            Target::ObjectsOf(_) => {
                // Add hint for predicate index
                if self.should_use_index_hint("predicate_index") {
                    optimized_query = format!("# Use predicate index\n{}", optimized_query);
                }
            }
            Target::SubjectsOf(_) => {
                // Add hint for subject index
                if self.should_use_index_hint("subject_index") {
                    optimized_query = format!("# Use subject index\n{}", optimized_query);
                }
            }
            _ => {} // No specific optimization for other types
        }

        // Add LIMIT clause if max_results is configured
        if self.optimization_config.parallel_threshold > 0 {
            if !optimized_query.to_uppercase().contains("LIMIT") {
                optimized_query = format!(
                    "{} LIMIT {}",
                    optimized_query, self.optimization_config.parallel_threshold
                );
            }
        }

        Ok(optimized_query)
    }

    /// Check if index hint should be used based on statistics
    fn should_use_index_hint(&self, index_type: &str) -> bool {
        if let Some(stats) = self.index_usage_stats.get(index_type) {
            stats.avg_performance_gain > self.optimization_config.index_hint_threshold
        } else {
            true // Default to using index hints if no statistics available
        }
    }

    /// Select all target nodes for a given target definition
    pub fn select_targets(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let start_time = std::time::Instant::now();
        let cache_key = self.create_cache_key(target, graph_name);

        // Check cache first
        if self.optimization_config.enable_caching {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                let cache_age = cached_result.cached_at.elapsed();
                if cache_age.as_secs() <= self.optimization_config.cache_ttl {
                    // Clone the nodes before updating statistics to avoid borrow conflict
                    let nodes: Vec<_> = cached_result.nodes.iter().cloned().collect();

                    // Update statistics
                    self.stats.total_evaluations += 1;
                    self.record_cache_hit();

                    return Ok(nodes);
                }
            }
        }

        // Record cache miss
        self.record_cache_miss();

        // Choose execution strategy based on target type and optimization settings
        let result = match target {
            Target::Sparql(_) => {
                // Use advanced SPARQL optimization for SPARQL targets
                self.execute_sparql_target_optimized(store, target, graph_name)?
            }
            _ => {
                // Use direct store access for other target types
                self.execute_target_selection_direct(store, target, graph_name)?
            }
        };

        // Cache the result if beneficial
        if self.optimization_config.enable_caching && self.should_cache_result(&result) {
            let cached_result = CachedTargetResult {
                nodes: result.iter().cloned().collect(),
                cached_at: std::time::Instant::now(),
                stats: CacheStats {
                    hits: 0,
                    misses: 1,
                    avg_query_time: start_time.elapsed(),
                },
            };

            // Manage cache size with intelligent eviction
            self.manage_cache_size(&cache_key, cached_result);
        }

        // Update statistics
        self.update_execution_statistics(start_time.elapsed(), result.len());

        Ok(result)
    }

    /// Execute SPARQL target with advanced optimizations
    fn execute_sparql_target_optimized(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        if let Target::Sparql(sparql_target) = target {
            // Get or create optimized query plan
            let query_plan = self.get_or_create_query_plan(&sparql_target.query)?;

            // Choose execution strategy based on estimated cardinality
            match query_plan.execution_strategy {
                ExecutionStrategy::Sequential => {
                    self.execute_sparql_sequential(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::Parallel => {
                    self.execute_sparql_parallel(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::IndexDriven => {
                    self.execute_sparql_index_driven(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::Hybrid => {
                    self.execute_sparql_hybrid(store, &query_plan.optimized_query, graph_name)
                }
            }
        } else {
            Err(ShaclError::TargetSelection(
                "Expected SPARQL target".to_string(),
            ))
        }
    }

    /// Get or create optimized query plan for SPARQL query
    fn get_or_create_query_plan(&mut self, query: &str) -> Result<QueryPlan> {
        // Check cache first
        if let Some(plan) = self.query_plan_cache.get(query) {
            // Check if plan is still fresh (older than 1 hour)
            if plan.created_at.elapsed().as_secs() < 3600 {
                return Ok(plan.clone());
            }
        }

        // Create new optimized query plan
        let plan = self.create_optimized_query_plan(query)?;
        self.query_plan_cache
            .insert(query.to_string(), plan.clone());

        Ok(plan)
    }

    /// Create optimized query plan with cost estimation
    fn create_optimized_query_plan(&self, query: &str) -> Result<QueryPlan> {
        let mut optimized_query = query.to_string();
        let mut index_hints = Vec::new();

        // Analyze query patterns for optimization opportunities
        let estimated_cardinality = self.estimate_query_cardinality(query);

        // Add DISTINCT if not present to avoid duplicates
        if !query.to_uppercase().contains("DISTINCT") && query.to_uppercase().contains("SELECT") {
            optimized_query = optimized_query.replace("SELECT", "SELECT DISTINCT");
        }

        // Add index hints based on query patterns
        if query.contains("rdf:type") || query.contains("a ") {
            index_hints.push(IndexHint {
                index_type: "type_index".to_string(),
                selectivity: 0.1,
                cost_benefit: 0.8,
            });
        }

        // Determine execution strategy
        let execution_strategy =
            if estimated_cardinality > self.optimization_config.parallel_threshold {
                if self.optimization_config.enable_adaptive_optimization {
                    ExecutionStrategy::Hybrid
                } else {
                    ExecutionStrategy::Parallel
                }
            } else if !index_hints.is_empty() {
                ExecutionStrategy::IndexDriven
            } else {
                ExecutionStrategy::Sequential
            };

        // Add performance hints
        if self.optimization_config.enable_adaptive_optimization {
            optimized_query = format!(
                "# Query plan: {:?}, Est. cardinality: {}\n{}",
                execution_strategy, estimated_cardinality, optimized_query
            );
        }

        Ok(QueryPlan {
            optimized_query,
            estimated_cardinality,
            index_hints,
            execution_strategy,
            created_at: std::time::Instant::now(),
        })
    }

    /// Estimate query cardinality based on patterns
    fn estimate_query_cardinality(&self, query: &str) -> usize {
        // Simple heuristic-based estimation
        let mut estimated_cardinality = 1000; // Default estimate

        // Adjust based on query patterns
        if query.contains("rdf:type") {
            estimated_cardinality *= 10; // Type queries tend to be larger
        }

        if query.contains("OPTIONAL") {
            estimated_cardinality = (estimated_cardinality as f64 * 1.5) as usize;
        }

        if query.contains("UNION") {
            estimated_cardinality *= 2;
        }

        // Limit-based adjustment
        if let Some(limit_match) = query.to_uppercase().find("LIMIT") {
            let limit_part = &query[limit_match + 5..];
            if let Some(number) = limit_part.split_whitespace().next() {
                if let Ok(limit) = number.parse::<usize>() {
                    estimated_cardinality = estimated_cardinality.min(limit);
                }
            }
        }

        estimated_cardinality
    }

    /// Execute SPARQL query with sequential strategy
    fn execute_sparql_sequential(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as SPARQL execution would require
        // integration with oxirs-arq or similar SPARQL engine
        tracing::debug!("Executing SPARQL target query sequentially: {}", query);
        Ok(Vec::new())
    }

    /// Execute SPARQL query with parallel strategy
    fn execute_sparql_parallel(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as parallel SPARQL execution would require
        // advanced integration with the SPARQL engine
        tracing::debug!("Executing SPARQL target query in parallel: {}", query);
        Ok(Vec::new())
    }

    /// Execute SPARQL query with index-driven strategy
    fn execute_sparql_index_driven(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as index-driven execution would require
        // specific index optimization in the SPARQL engine
        tracing::debug!(
            "Executing SPARQL target query with index optimization: {}",
            query
        );
        Ok(Vec::new())
    }

    /// Execute SPARQL query with hybrid strategy
    fn execute_sparql_hybrid(
        &self,
        store: &dyn Store,
        query: &str,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // Hybrid strategy: start with index-driven, fall back to parallel if needed
        let start_time = std::time::Instant::now();

        // Try index-driven first
        let result = self.execute_sparql_index_driven(store, query, graph_name)?;

        // If taking too long or no results, try parallel approach
        if start_time.elapsed().as_millis() > 1000 && result.is_empty() {
            tracing::debug!("Falling back to parallel execution for SPARQL target");
            return self.execute_sparql_parallel(store, query, graph_name);
        }

        Ok(result)
    }

    /// Check if result should be cached based on size and performance characteristics
    fn should_cache_result(&self, result: &[Term]) -> bool {
        // Don't cache very large results that might consume too much memory
        if result.len() > 10000 {
            return false;
        }

        // Don't cache empty results unless configuration allows it
        if result.is_empty() {
            return false;
        }

        // Cache results of medium size that are likely to be reused
        result.len() > 10 && result.len() < 1000
    }

    /// Manage cache size with intelligent eviction strategies
    fn manage_cache_size(&mut self, new_key: &str, new_result: CachedTargetResult) {
        if self.cache.len() >= self.optimization_config.max_cache_size {
            // Advanced eviction strategy: remove least valuable entries
            let mut removal_candidates = Vec::new();

            for (key, cached) in &self.cache {
                let age = cached.cached_at.elapsed().as_secs();
                let hit_rate = if cached.stats.hits + cached.stats.misses > 0 {
                    cached.stats.hits as f64 / (cached.stats.hits + cached.stats.misses) as f64
                } else {
                    0.0
                };

                // Score based on age, hit rate, and result size
                let score = (age as f64) * 0.5
                    + (1.0 - hit_rate) * 0.3
                    + (cached.nodes.len() as f64 * 0.01);
                removal_candidates.push((key.clone(), score));
            }

            // Sort by score (higher score = more likely to remove)
            removal_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove the worst candidate
            if let Some((key_to_remove, _)) = removal_candidates.first() {
                self.cache.remove(key_to_remove);
            }
        }

        self.cache.insert(new_key.to_string(), new_result);
    }

    /// Update execution statistics
    fn update_execution_statistics(&mut self, duration: std::time::Duration, result_count: usize) {
        self.stats.total_evaluations += 1;
        self.stats.total_time += duration;

        // Update average evaluation time
        if self.stats.total_evaluations > 0 {
            self.stats.avg_evaluation_time =
                self.stats.total_time / self.stats.total_evaluations as u32;
        }

        // Update adaptive thresholds based on performance
        if self.optimization_config.enable_adaptive_optimization {
            self.update_adaptive_thresholds(duration, result_count);
        }
    }

    /// Update adaptive optimization thresholds based on performance data
    fn update_adaptive_thresholds(&mut self, duration: std::time::Duration, result_count: usize) {
        // Adjust parallel threshold based on performance
        if duration.as_millis() > 500
            && result_count > self.optimization_config.parallel_threshold / 2
        {
            // Lower threshold if we're seeing slow performance on medium-sized results
            self.optimization_config.parallel_threshold =
                (self.optimization_config.parallel_threshold * 8) / 10;
        } else if duration.as_millis() < 100
            && result_count > self.optimization_config.parallel_threshold
        {
            // Raise threshold if we're handling large results quickly
            self.optimization_config.parallel_threshold =
                (self.optimization_config.parallel_threshold * 12) / 10;
        }

        // Adjust cache TTL based on hit patterns
        if self.stats.cache_hit_rate > 0.8 {
            // High hit rate, can afford longer TTL
            self.optimization_config.cache_ttl = (self.optimization_config.cache_ttl * 12) / 10;
        } else if self.stats.cache_hit_rate < 0.3 {
            // Low hit rate, reduce TTL to freshen cache more often
            self.optimization_config.cache_ttl = (self.optimization_config.cache_ttl * 8) / 10;
        }
    }

    /// Record cache hit
    fn record_cache_hit(&mut self) {
        self.update_cache_hit_rate();
    }

    /// Record cache miss  
    fn record_cache_miss(&mut self) {
        self.update_cache_hit_rate();
    }

    /// Execute target selection using direct store access instead of SPARQL due to oxirs-core limitations
    fn execute_target_query(
        &self,
        store: &dyn Store,
        _query: &str,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // We'll implement direct store access based on the target type
        // This bypasses the SPARQL parser limitations with long IRIs
        Ok(Vec::new()) // Placeholder - will be implemented per target type
    }

    /// Execute target selection using direct store operations
    fn execute_target_selection_direct(
        &self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, NamedNode as CoreNamedNode, Object, Predicate, Quad};

        match target {
            Target::Class(class_iri) => {
                // Find all subjects that have rdf:type = class_iri
                let rdf_type = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").map_err(
                        |e| ShaclError::TargetSelection(format!("Invalid RDF type IRI: {}", e)),
                    )?,
                );

                let mut target_nodes = Vec::new();

                // Iterate through all quads in the store
                for quad in store.find_quads(None, None, None, None)? {
                    // Check if this is a type triple and matches our target class
                    if quad.predicate() == &rdf_type
                        && matches!(quad.object(), Object::NamedNode(obj) if obj.as_str() == class_iri.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        // Add the subject as a target node
                        match quad.subject() {
                            oxirs_core::model::Subject::NamedNode(node) => {
                                target_nodes.push(Term::NamedNode(node.clone()));
                            }
                            oxirs_core::model::Subject::BlankNode(blank) => {
                                target_nodes.push(Term::BlankNode(blank.clone()));
                            }
                            oxirs_core::model::Subject::Variable(var) => {
                                target_nodes.push(Term::Variable(var.clone()));
                            }
                            oxirs_core::model::Subject::QuotedTriple(qt) => {
                                target_nodes.push(Term::QuotedTriple(qt.clone()));
                            }
                        }
                    }
                }

                // Remove duplicates and sort for deterministic results
                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for class {}",
                    target_nodes.len(),
                    class_iri.as_str()
                );
                Ok(target_nodes)
            }
            Target::Node(node) => {
                // For specific nodes, just return the node itself
                Ok(vec![node.clone()])
            }
            Target::ObjectsOf(property) => {
                let mut target_nodes = Vec::new();

                // Find all objects of the specified property
                for quad in store.find_quads(None, None, None, None)? {
                    if matches!(quad.predicate(), Predicate::NamedNode(pred) if pred.as_str() == property.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        target_nodes.push(quad.object().clone().into());
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for objectsOf {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::SubjectsOf(property) => {
                let mut target_nodes = Vec::new();

                // Find all subjects of the specified property
                for quad in store.find_quads(None, None, None, None)? {
                    if matches!(quad.predicate(), Predicate::NamedNode(pred) if pred.as_str() == property.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        // Add the subject as a target node
                        match quad.subject() {
                            oxirs_core::model::Subject::NamedNode(node) => {
                                target_nodes.push(Term::NamedNode(node.clone()));
                            }
                            oxirs_core::model::Subject::BlankNode(blank) => {
                                target_nodes.push(Term::BlankNode(blank.clone()));
                            }
                            oxirs_core::model::Subject::Variable(var) => {
                                target_nodes.push(Term::Variable(var.clone()));
                            }
                            oxirs_core::model::Subject::QuotedTriple(qt) => {
                                target_nodes.push(Term::QuotedTriple(qt.clone()));
                            }
                        }
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for subjectsOf {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::Sparql(_sparql_target) => {
                // For SPARQL targets, we still need to use the query engine
                // This might fail due to oxirs-core limitations, but let's try
                let query = self.generate_target_query(target, graph_name)?;
                self.execute_sparql_target_query(store, &query)
            }
            Target::Implicit(class_iri) => {
                // Same as class target
                let class_target = Target::Class(class_iri.clone());
                self.execute_target_selection_direct(store, &class_target, graph_name)
            }
        }
    }

    /// Execute SPARQL target query (may fail due to oxirs-core limitations)
    fn execute_sparql_target_query(&self, store: &dyn Store, query: &str) -> Result<Vec<Term>> {
        use oxirs_core::query::{QueryEngine, QueryResult};

        tracing::info!("Executing SPARQL target query: '{}'", query);

        let query_engine = QueryEngine::new();

        match query_engine.query(query, store) {
            Ok(QueryResult::Select {
                variables: _,
                bindings,
            }) => {
                let mut target_nodes = Vec::new();

                for binding in bindings {
                    // Look for ?this variable or first variable
                    if let Some(term) = binding.get("this").or_else(|| binding.values().next()) {
                        target_nodes.push(term.clone());
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);
                Ok(target_nodes)
            }
            Ok(_) => Err(ShaclError::SparqlExecution(
                "SPARQL target query must return SELECT results".to_string(),
            )),
            Err(e) => {
                tracing::error!("SPARQL target query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(format!(
                    "SPARQL target query execution failed: {}",
                    e
                )))
            }
        }
    }

    /// Sort and remove duplicate target nodes
    fn sort_and_dedupe_targets(&self, target_nodes: &mut Vec<Term>) {
        target_nodes.sort_by(|a, b| match (a, b) {
            (Term::NamedNode(a_node), Term::NamedNode(b_node)) => {
                a_node.as_str().cmp(b_node.as_str())
            }
            (Term::BlankNode(a_blank), Term::BlankNode(b_blank)) => {
                a_blank.as_str().cmp(b_blank.as_str())
            }
            (Term::Literal(a_lit), Term::Literal(b_lit)) => a_lit.as_str().cmp(b_lit.as_str()),
            (Term::NamedNode(_), _) => std::cmp::Ordering::Less,
            (Term::BlankNode(_), Term::NamedNode(_)) => std::cmp::Ordering::Greater,
            (Term::BlankNode(_), _) => std::cmp::Ordering::Less,
            (Term::Literal(_), Term::NamedNode(_)) => std::cmp::Ordering::Greater,
            (Term::Literal(_), Term::BlankNode(_)) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        });
        target_nodes.dedup();
    }

    /// Create cache key for target and graph combination
    fn create_cache_key(&self, target: &Target, graph_name: Option<&str>) -> String {
        match target {
            Target::Class(class_iri) => {
                format!(
                    "class:{}:{}",
                    class_iri.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::Node(node) => {
                format!(
                    "node:{}:{}",
                    format!("{:?}", node),
                    graph_name.unwrap_or("default")
                )
            }
            Target::ObjectsOf(property) => {
                format!(
                    "objects_of:{}:{}",
                    property.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::SubjectsOf(property) => {
                format!(
                    "subjects_of:{}:{}",
                    property.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::Sparql(sparql_target) => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                sparql_target.query.hash(&mut hasher);
                let query_hash = hasher.finish();

                format!("sparql:{}:{}", query_hash, graph_name.unwrap_or("default"))
            }
            Target::Implicit(class_iri) => {
                format!(
                    "implicit:{}:{}",
                    class_iri.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
        }
    }

    /// Find the oldest cache entry for eviction
    fn find_oldest_cache_entry(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, cached_result)| cached_result.cached_at)
            .map(|(key, _)| key.clone())
    }

    /// Update cache hit rate statistics
    fn update_cache_hit_rate(&mut self) {
        if self.stats.total_evaluations > 0 {
            let total_cache_operations = self
                .cache
                .values()
                .map(|cached_result| cached_result.stats.hits + cached_result.stats.misses)
                .sum::<usize>();

            let total_hits = self
                .cache
                .values()
                .map(|cached_result| cached_result.stats.hits)
                .sum::<usize>();

            if total_cache_operations > 0 {
                self.stats.cache_hit_rate = total_hits as f64 / total_cache_operations as f64;
            }
        }
    }

    /// Get current target selection statistics
    pub fn get_statistics(&self) -> &TargetSelectionStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> TargetCacheStats {
        let total_hits = self
            .cache
            .values()
            .map(|cached_result| cached_result.stats.hits)
            .sum::<usize>();

        let total_operations = self
            .cache
            .values()
            .map(|cached_result| cached_result.stats.hits + cached_result.stats.misses)
            .sum::<usize>();

        let hit_rate = if total_operations > 0 {
            total_hits as f64 / total_operations as f64
        } else {
            0.0
        };

        // Estimate memory usage (rough calculation)
        let memory_usage = self.cache.len() * 1024; // Rough estimate: 1KB per cache entry

        TargetCacheStats {
            hits: total_hits,
            misses: total_operations - total_hits,
            hit_rate,
            cache_size: self.cache.len(),
            memory_usage_bytes: memory_usage,
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Update index usage statistics for adaptive optimization
    pub fn update_index_usage_stats(&mut self, index_type: &str, performance_gain: f64) {
        let stats = self
            .index_usage_stats
            .entry(index_type.to_string())
            .or_insert_with(|| IndexUsageStats {
                usage_count: 0,
                avg_performance_gain: 0.0,
                last_used: std::time::Instant::now(),
            });

        stats.usage_count += 1;
        stats.avg_performance_gain = (stats.avg_performance_gain * (stats.usage_count - 1) as f64
            + performance_gain)
            / stats.usage_count as f64;
        stats.last_used = std::time::Instant::now();
    }

    /// Generate optimized SPARQL query with advanced features
    pub fn generate_optimized_target_query(
        &mut self,
        target: &Target,
        graph_name: Option<&str>,
        query_options: &QueryOptimizationOptions,
    ) -> Result<OptimizedQuery> {
        let start_time = std::time::Instant::now();

        // Check if we have a cached query plan
        let cache_key = self.create_query_plan_cache_key(target, graph_name, query_options);
        if let Some(cached_plan) = self.query_plan_cache.get(&cache_key) {
            if cached_plan.created_at.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(OptimizedQuery {
                    sparql: cached_plan.optimized_query.clone(),
                    estimated_cardinality: cached_plan.estimated_cardinality,
                    execution_strategy: cached_plan.execution_strategy.clone(),
                    index_hints: cached_plan.index_hints.clone(),
                    optimization_time: start_time.elapsed(),
                });
            }
        }

        // Generate base query
        let base_query = self.generate_target_query(target, graph_name)?;

        // Apply optimizations
        let optimized_query = self.apply_query_optimizations(&base_query, target, query_options)?;

        // Create execution plan
        let execution_plan = self.create_execution_plan(target, query_options)?;

        // Estimate cardinality
        let estimated_cardinality = self.estimate_target_cardinality(target, query_options);

        // Cache the query plan
        let query_plan = QueryPlan {
            optimized_query: optimized_query.clone(),
            estimated_cardinality,
            index_hints: execution_plan.index_hints.clone(),
            execution_strategy: execution_plan.execution_strategy.clone(),
            created_at: std::time::Instant::now(),
        };

        if self.query_plan_cache.len() < 100 {
            self.query_plan_cache.insert(cache_key, query_plan);
        }

        Ok(OptimizedQuery {
            sparql: optimized_query,
            estimated_cardinality,
            execution_strategy: execution_plan.execution_strategy,
            index_hints: execution_plan.index_hints,
            optimization_time: start_time.elapsed(),
        })
    }

    /// Apply query optimizations based on target type and options
    fn apply_query_optimizations(
        &self,
        base_query: &str,
        target: &Target,
        options: &QueryOptimizationOptions,
    ) -> Result<String> {
        let mut optimized_query = base_query.to_string();

        // Add LIMIT if specified
        if let Some(limit) = options.limit {
            if !optimized_query.contains("LIMIT") {
                optimized_query.push_str(&format!(" LIMIT {}", limit));
            }
        }

        // Add ORDER BY for deterministic results if requested
        if options.deterministic_results {
            if !optimized_query.contains("ORDER BY") {
                optimized_query =
                    optimized_query.replace("SELECT DISTINCT ?target", "SELECT DISTINCT ?target");
                optimized_query.push_str(" ORDER BY ?target");
            }
        }

        // Add index hints based on target type
        if options.use_index_hints {
            optimized_query = self.add_index_hints(&optimized_query, target)?;
        }

        // Add performance monitoring hints if enabled
        if options.include_performance_hints {
            optimized_query = format!(
                "# Generated: {}\n{}",
                chrono::Utc::now().to_rfc3339(),
                optimized_query
            );
        }

        Ok(optimized_query)
    }

    /// Add index hints to the query based on target type
    fn add_index_hints(&self, query: &str, target: &Target) -> Result<String> {
        let mut optimized_query = query.to_string();

        match target {
            Target::Class(_) | Target::Implicit(_) => {
                if self.should_use_index_hint("type_index") {
                    optimized_query = format!(
                        "# HINT: Use type index for rdf:type lookups\n{}",
                        optimized_query
                    );
                }
            }
            Target::ObjectsOf(property) => {
                if self.should_use_index_hint("object_index") {
                    optimized_query = format!(
                        "# HINT: Use object index for property <{}>\n{}",
                        property.as_str(),
                        optimized_query
                    );
                }
            }
            Target::SubjectsOf(property) => {
                if self.should_use_index_hint("subject_index") {
                    optimized_query = format!(
                        "# HINT: Use subject index for property <{}>\n{}",
                        property.as_str(),
                        optimized_query
                    );
                }
            }
            _ => {}
        }

        Ok(optimized_query)
    }

    /// Create execution plan for target selection
    fn create_execution_plan(
        &self,
        target: &Target,
        options: &QueryOptimizationOptions,
    ) -> Result<ExecutionPlan> {
        let estimated_cardinality = self.estimate_target_cardinality(target, options);

        // Choose execution strategy based on estimated cardinality
        let execution_strategy =
            if estimated_cardinality > self.optimization_config.parallel_threshold {
                ExecutionStrategy::Parallel
            } else if self.should_use_index_strategy(target) {
                ExecutionStrategy::IndexDriven
            } else {
                ExecutionStrategy::Sequential
            };

        // Generate index hints
        let index_hints = self.generate_index_hints(target)?;

        Ok(ExecutionPlan {
            execution_strategy,
            index_hints,
            estimated_cardinality,
        })
    }

    /// Estimate cardinality for target selection
    fn estimate_target_cardinality(
        &self,
        target: &Target,
        _options: &QueryOptimizationOptions,
    ) -> usize {
        match target {
            Target::Node(_) => 1, // Single node
            Target::Class(_) | Target::Implicit(_) => {
                // Estimate based on typical class sizes
                1000 // Default estimate
            }
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => {
                // Estimate based on property usage
                500 // Default estimate
            }
            Target::Sparql(_) => {
                // Can't estimate SPARQL queries easily
                100 // Conservative estimate
            }
        }
    }

    /// Generate index hints for target type
    fn generate_index_hints(&self, target: &Target) -> Result<Vec<IndexHint>> {
        let mut hints = Vec::new();

        match target {
            Target::Class(_) | Target::Implicit(_) => {
                hints.push(IndexHint {
                    index_type: "type_index".to_string(),
                    selectivity: 0.1, // Assume 10% selectivity
                    cost_benefit: 0.8,
                });
            }
            Target::ObjectsOf(property) => {
                hints.push(IndexHint {
                    index_type: format!("object_index_{}", property.as_str()),
                    selectivity: 0.05,
                    cost_benefit: 0.9,
                });
            }
            Target::SubjectsOf(property) => {
                hints.push(IndexHint {
                    index_type: format!("subject_index_{}", property.as_str()),
                    selectivity: 0.05,
                    cost_benefit: 0.9,
                });
            }
            _ => {}
        }

        Ok(hints)
    }

    /// Check if index strategy should be used
    fn should_use_index_strategy(&self, target: &Target) -> bool {
        match target {
            Target::Class(_) | Target::Implicit(_) => self.should_use_index_hint("type_index"),
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => {
                self.should_use_index_hint("property_index")
            }
            _ => false,
        }
    }

    /// Create cache key for query plans
    fn create_query_plan_cache_key(
        &self,
        target: &Target,
        graph_name: Option<&str>,
        options: &QueryOptimizationOptions,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash target
        match target {
            Target::Class(class_iri) => {
                "class".hash(&mut hasher);
                class_iri.as_str().hash(&mut hasher);
            }
            Target::Node(node) => {
                "node".hash(&mut hasher);
                format!("{:?}", node).hash(&mut hasher);
            }
            Target::ObjectsOf(property) => {
                "objects_of".hash(&mut hasher);
                property.as_str().hash(&mut hasher);
            }
            Target::SubjectsOf(property) => {
                "subjects_of".hash(&mut hasher);
                property.as_str().hash(&mut hasher);
            }
            Target::Sparql(sparql_target) => {
                "sparql".hash(&mut hasher);
                sparql_target.query.hash(&mut hasher);
            }
            Target::Implicit(class_iri) => {
                "implicit".hash(&mut hasher);
                class_iri.as_str().hash(&mut hasher);
            }
        }

        // Hash graph name and options
        graph_name.hash(&mut hasher);
        options.limit.hash(&mut hasher);
        options.deterministic_results.hash(&mut hasher);
        options.use_index_hints.hash(&mut hasher);

        format!("plan_{:x}", hasher.finish())
    }

    /// Generate batch target queries for multiple targets
    pub fn generate_batch_target_queries(
        &mut self,
        targets: &[Target],
        graph_name: Option<&str>,
        options: &QueryOptimizationOptions,
    ) -> Result<BatchQueryResult> {
        let start_time = std::time::Instant::now();
        let mut optimized_queries = Vec::new();
        let mut total_estimated_cardinality = 0;

        for target in targets {
            let optimized_query =
                self.generate_optimized_target_query(target, graph_name, options)?;
            total_estimated_cardinality += optimized_query.estimated_cardinality;
            optimized_queries.push(optimized_query);
        }

        // Create union query if beneficial
        let union_query = if optimized_queries.len() > 1 && options.use_union_optimization {
            Some(self.create_union_query(&optimized_queries)?)
        } else {
            None
        };

        Ok(BatchQueryResult {
            individual_queries: optimized_queries,
            union_query,
            total_estimated_cardinality,
            batch_optimization_time: start_time.elapsed(),
        })
    }

    /// Create union query from multiple target queries
    fn create_union_query(&self, queries: &[OptimizedQuery]) -> Result<String> {
        let mut union_parts = Vec::new();

        for query in queries {
            // Extract the WHERE clause from each query
            if let Some(where_start) = query.sparql.find("WHERE {") {
                let where_clause = &query.sparql[where_start + 7..];
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = &where_clause[..where_end].trim();
                    union_parts.push(format!("{{ {} }}", where_content));
                }
            }
        }

        if union_parts.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "No valid WHERE clauses found for union query".to_string(),
            ));
        }

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n}}",
            union_parts.join("\n  UNION\n  ")
        ))
    }
}

impl Default for TargetSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_creation() {
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri.clone());

        match target {
            Target::Class(iri) => assert_eq!(iri, class_iri),
            _ => panic!("Expected class target"),
        }
    }

    #[test]
    fn test_node_target() {
        let node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let target = Target::node(node.clone());

        match target {
            Target::Node(n) => assert_eq!(n, node),
            _ => panic!("Expected node target"),
        }
    }

    #[test]
    fn test_sparql_target() {
        let query = "SELECT ?this WHERE { ?this a ex:Person }".to_string();
        let prefixes = Some("PREFIX ex: <http://example.org/>".to_string());
        let target = Target::sparql(query.clone(), prefixes.clone());

        match target {
            Target::Sparql(sparql_target) => {
                assert_eq!(sparql_target.query, query);
                assert_eq!(sparql_target.prefixes, prefixes);
            }
            _ => panic!("Expected SPARQL target"),
        }
    }

    #[test]
    fn debug_target_query_generation() {
        let selector = TargetSelector::new();
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri);

        let query = selector.generate_target_query(&target, None).unwrap();
        println!("Generated query: '{}'", query);
        eprintln!("Generated query: '{}'", query);

        assert!(query.contains("SELECT"));
        assert!(query.contains("?target"));
        assert!(query.contains("http://example.org/Person"));
    }

    #[test]
    fn debug_direct_sparql_execution() {
        use oxirs_core::model::{GraphName, Quad};
        use oxirs_core::{query::QueryEngine, Store};

        let mut store = ConcreteStore::new().unwrap();

        // Test simple query first
        let simple_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        println!("Testing simple query: '{}'", simple_query);

        let query_engine = QueryEngine::new();
        let result = query_engine.query(simple_query, &store);

        match result {
            Ok(result) => println!("Simple query executed successfully: {:?}", result),
            Err(e) => {
                println!("Simple query failed: {}", e);
                eprintln!("Simple query failed: {}", e);
            }
        }

        // Test with simpler IRI first
        let simple_iri_query = "SELECT DISTINCT ?target WHERE { ?target <http://example.org/type> <http://example.org/Person> . }";
        println!("Testing simple IRI query: '{}'", simple_iri_query);

        let result2 = query_engine.query(simple_iri_query, &store);

        match result2 {
            Ok(result) => println!("Simple IRI query executed successfully: {:?}", result),
            Err(e) => {
                println!("Simple IRI query failed: {}", e);
                eprintln!("Simple IRI query failed: {}", e);
            }
        }

        // Test our specific query with long RDF type IRI
        let query = "SELECT DISTINCT ?target WHERE { ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . }";
        println!("Testing RDF type query: '{}'", query);

        let result3 = query_engine.query(query, &store);

        match result3 {
            Ok(result) => println!("RDF type query executed successfully: {:?}", result),
            Err(e) => {
                println!("RDF type query failed: {}", e);
                eprintln!("RDF type query failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cache_key_generation() {
        let selector = TargetSelector::new();
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri);

        let key1 = selector.create_cache_key(&target, None);
        let key2 = selector.create_cache_key(&target, None);
        let key3 = selector.create_cache_key(&target, Some("graph1"));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_target_optimization_config() {
        let config = TargetOptimizationConfig::default();

        assert!(config.enable_caching);
        assert_eq!(config.cache_ttl, 300);
        assert_eq!(config.max_cache_size, 1000);
        assert!(config.enable_query_optimization);
    }

    #[test]
    fn test_target_selector_with_config() {
        let mut config = TargetOptimizationConfig::default();
        config.cache_ttl = 600;
        config.enable_caching = false;

        let selector = TargetSelector::with_config(config.clone());
        assert_eq!(selector.optimization_config.cache_ttl, 600);
        assert!(!selector.optimization_config.enable_caching);
    }
}
