//! SHACL Constraint Evaluation Optimizations
//!
//! This module provides performance optimizations for SHACL constraint evaluation
//! including result caching, batch validation, and dependency-aware evaluation ordering.

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator},
    PropertyPath, Result, ShaclError, ShapeId,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Constraint evaluation cache for performance optimization
#[derive(Debug, Clone)]
pub struct ConstraintCache {
    /// Cache for constraint evaluation results
    cache: Arc<RwLock<HashMap<CacheKey, CachedResult>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache entry TTL
    ttl: Duration,
}

/// Cache key for constraint evaluation results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    /// Hash of the constraint
    constraint_hash: u64,
    /// Focus node
    focus_node: Term,
    /// Property path (if applicable)
    path: Option<PropertyPath>,
    /// Values being validated
    values_hash: u64,
    /// Shape ID
    shape_id: ShapeId,
}

/// Cached constraint evaluation result
#[derive(Debug, Clone)]
struct CachedResult {
    /// The cached result
    result: ConstraintEvaluationResult,
    /// Timestamp when cached
    cached_at: Instant,
    /// Number of times this result has been used
    hit_count: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Total constraint evaluations
    pub evaluations: usize,
    /// Average evaluation time (microseconds)
    pub avg_evaluation_time_us: f64,
    /// Cache evictions
    pub evictions: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

impl Default for ConstraintCache {
    fn default() -> Self {
        Self::new(10000, Duration::from_secs(300)) // 10k entries, 5 min TTL
    }
}

impl ConstraintCache {
    /// Create a new constraint cache
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            max_size,
            ttl,
        }
    }

    /// Get a cached result if available
    pub fn get(
        &self,
        constraint: &Constraint,
        context: &ConstraintContext,
    ) -> Option<ConstraintEvaluationResult> {
        let key = self.create_cache_key(constraint, context);

        // First, check if we have a valid cached result
        let result = {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&key) {
                // Check if entry is still valid
                if cached.cached_at.elapsed() <= self.ttl {
                    Some(cached.result.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };

        if result.is_some() {
            // Update statistics and hit count
            {
                let mut stats = self.stats.write().unwrap();
                stats.hits += 1;
            }

            // Increment hit count
            {
                let mut cache_mut = self.cache.write().unwrap();
                if let Some(entry) = cache_mut.get_mut(&key) {
                    entry.hit_count += 1;
                }
            }

            return result;
        }

        // Cache miss
        let mut stats = self.stats.write().unwrap();
        stats.misses += 1;
        None
    }

    /// Store a result in the cache
    pub fn put(
        &self,
        constraint: &Constraint,
        context: &ConstraintContext,
        result: ConstraintEvaluationResult,
        evaluation_time: Duration,
    ) {
        let key = self.create_cache_key(constraint, context);
        let cached_result = CachedResult {
            result: result.clone(),
            cached_at: Instant::now(),
            hit_count: 0,
        };

        {
            let mut cache = self.cache.write().unwrap();

            // Evict entries if cache is full
            if cache.len() >= self.max_size {
                self.evict_entries(&mut cache);
            }

            cache.insert(key, cached_result);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.evaluations += 1;
            let eval_time_us = evaluation_time.as_micros() as f64;
            if stats.evaluations == 1 {
                stats.avg_evaluation_time_us = eval_time_us;
            } else {
                stats.avg_evaluation_time_us =
                    (stats.avg_evaluation_time_us * (stats.evaluations - 1) as f64 + eval_time_us)
                        / stats.evaluations as f64;
            }
        }
    }

    /// Create cache key from constraint and context
    fn create_cache_key(&self, constraint: &Constraint, context: &ConstraintContext) -> CacheKey {
        let constraint_hash = self.hash_constraint(constraint);
        let values_hash = self.hash_values(&context.values);

        CacheKey {
            constraint_hash,
            focus_node: context.focus_node.clone(),
            path: context.path.clone(),
            values_hash,
            shape_id: context.shape_id.clone(),
        }
    }

    /// Hash a constraint for caching
    fn hash_constraint(&self, constraint: &Constraint) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash based on constraint type and key properties
        match constraint {
            Constraint::Class(c) => {
                "class".hash(&mut hasher);
                c.class_iri.as_str().hash(&mut hasher);
            }
            Constraint::Datatype(c) => {
                "datatype".hash(&mut hasher);
                c.datatype_iri.as_str().hash(&mut hasher);
            }
            Constraint::MinCount(c) => {
                "minCount".hash(&mut hasher);
                c.min_count.hash(&mut hasher);
            }
            Constraint::MaxCount(c) => {
                "maxCount".hash(&mut hasher);
                c.max_count.hash(&mut hasher);
            }
            Constraint::Pattern(c) => {
                "pattern".hash(&mut hasher);
                c.pattern.hash(&mut hasher);
                c.flags.hash(&mut hasher);
            }
            // Add more constraint types as needed
            _ => {
                format!("{:?}", constraint).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Hash values for caching
    fn hash_values(&self, values: &[Term]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for value in values {
            format!("{:?}", value).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Evict cache entries using LFU (Least Frequently Used) strategy
    fn evict_entries(&self, cache: &mut HashMap<CacheKey, CachedResult>) {
        let evict_count = cache.len() / 4; // Evict 25% of entries

        // Sort by hit count and age
        let mut entries: Vec<_> = cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| {
            let hit_cmp = a.1.hit_count.cmp(&b.1.hit_count);
            if hit_cmp == std::cmp::Ordering::Equal {
                // If hit counts are equal, evict older entries
                b.1.cached_at.cmp(&a.1.cached_at)
            } else {
                hit_cmp
            }
        });

        // Collect keys to remove
        let keys_to_remove: Vec<_> = entries
            .iter()
            .take(evict_count)
            .map(|(k, _)| k.clone())
            .collect();

        // Remove least frequently used entries
        for key in keys_to_remove {
            cache.remove(&key);
        }

        // Update eviction statistics
        let mut stats = self.stats.write().unwrap();
        stats.evictions += evict_count;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }
}

/// Batch constraint evaluator for efficient evaluation of multiple constraints
#[derive(Debug)]
pub struct BatchConstraintEvaluator {
    cache: ConstraintCache,
    parallel_evaluation: bool,
    batch_size: usize,
}

impl Default for BatchConstraintEvaluator {
    fn default() -> Self {
        Self::new(ConstraintCache::default(), false, 100)
    }
}

impl BatchConstraintEvaluator {
    /// Create a new batch evaluator
    pub fn new(cache: ConstraintCache, parallel_evaluation: bool, batch_size: usize) -> Self {
        Self {
            cache,
            parallel_evaluation,
            batch_size,
        }
    }

    /// Evaluate multiple constraints in batches
    pub fn evaluate_batch(
        &self,
        store: &Store,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let mut results = Vec::with_capacity(constraints_with_contexts.len());

        // Process in batches
        for batch in constraints_with_contexts.chunks(self.batch_size) {
            let batch_results = if self.parallel_evaluation {
                self.evaluate_batch_parallel(store, batch)?
            } else {
                self.evaluate_batch_sequential(store, batch)?
            };
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Evaluate constraints sequentially with caching
    fn evaluate_batch_sequential(
        &self,
        store: &Store,
        batch: &[(Constraint, ConstraintContext)],
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let mut results = Vec::new();

        for (constraint, context) in batch {
            // Try cache first
            if let Some(cached_result) = self.cache.get(constraint, context) {
                results.push(cached_result);
                continue;
            }

            // Evaluate and cache result
            let start_time = Instant::now();
            let result = constraint.evaluate(store, context)?;
            let evaluation_time = start_time.elapsed();

            self.cache
                .put(constraint, context, result.clone(), evaluation_time);
            results.push(result);
        }

        Ok(results)
    }

    /// Evaluate constraints in parallel (when safe to do so)
    fn evaluate_batch_parallel(
        &self,
        store: &Store,
        batch: &[(Constraint, ConstraintContext)],
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        // For parallel evaluation, we need to be careful about thread safety
        // We'll use a thread pool for CPU-bound constraint evaluation

        use std::sync::{Arc, Mutex};
        use std::thread;

        if batch.len() < 4 {
            // For small batches, sequential is faster
            return self.evaluate_batch_sequential(store, batch);
        }

        // For now, parallel evaluation has thread safety limitations
        // Fall back to sequential evaluation to avoid Rc<> threading issues
        self.evaluate_batch_sequential(store, batch)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

/// Constraint dependency analyzer for optimal evaluation ordering
#[derive(Debug)]
pub struct ConstraintDependencyAnalyzer {
    /// Constraint evaluation cost estimates
    cost_estimates: HashMap<String, f64>,
}

impl Default for ConstraintDependencyAnalyzer {
    fn default() -> Self {
        let mut cost_estimates = HashMap::new();

        // Cost estimates for different constraint types (relative)
        cost_estimates.insert("class".to_string(), 5.0);
        cost_estimates.insert("datatype".to_string(), 1.0);
        cost_estimates.insert("nodeKind".to_string(), 1.0);
        cost_estimates.insert("minCount".to_string(), 1.0);
        cost_estimates.insert("maxCount".to_string(), 1.0);
        cost_estimates.insert("pattern".to_string(), 3.0);
        cost_estimates.insert("sparql".to_string(), 10.0);
        cost_estimates.insert("qualifiedValueShape".to_string(), 8.0);
        cost_estimates.insert("closed".to_string(), 6.0);

        Self { cost_estimates }
    }
}

impl ConstraintDependencyAnalyzer {
    /// Analyze and order constraints for optimal evaluation
    pub fn optimize_constraint_order(&self, constraints: Vec<Constraint>) -> Vec<Constraint> {
        let mut constraint_info: Vec<_> = constraints
            .into_iter()
            .map(|c| {
                let cost = self.estimate_constraint_cost(&c);
                let selectivity = self.estimate_constraint_selectivity(&c);
                // Sort by selectivity (lower is better) then by cost
                (c, selectivity, cost)
            })
            .collect();

        // Sort by selectivity first (more selective constraints first), then by cost
        constraint_info.sort_by(|a, b| {
            let selectivity_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
            if selectivity_cmp == std::cmp::Ordering::Equal {
                // If selectivity is equal, prioritize lower cost constraints
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                selectivity_cmp
            }
        });

        constraint_info.into_iter().map(|(c, _, _)| c).collect()
    }

    /// Estimate the cost of evaluating a constraint
    fn estimate_constraint_cost(&self, constraint: &Constraint) -> f64 {
        let base_cost = match constraint {
            Constraint::Class(_) => self.cost_estimates.get("class").copied().unwrap_or(5.0),
            Constraint::Datatype(_) => self.cost_estimates.get("datatype").copied().unwrap_or(1.0),
            Constraint::NodeKind(_) => self.cost_estimates.get("nodeKind").copied().unwrap_or(1.0),
            Constraint::MinCount(_) => self.cost_estimates.get("minCount").copied().unwrap_or(1.0),
            Constraint::MaxCount(_) => self.cost_estimates.get("maxCount").copied().unwrap_or(1.0),
            Constraint::Pattern(_) => self.cost_estimates.get("pattern").copied().unwrap_or(3.0),
            Constraint::Sparql(_) => self.cost_estimates.get("sparql").copied().unwrap_or(10.0),
            Constraint::QualifiedValueShape(_) => self.cost_estimates.get("qualifiedValueShape").copied().unwrap_or(8.0),
            Constraint::Closed(_) => self.cost_estimates.get("closed").copied().unwrap_or(6.0),
            Constraint::And(_) | Constraint::Or(_) | Constraint::Xone(_) => {
                // Logical constraints have variable cost based on sub-constraints
                7.0
            }
            _ => 3.0, // Default cost for other constraints
        };
        
        base_cost
    }

    /// Estimate the selectivity of a constraint (how many results it will filter out)
    fn estimate_constraint_selectivity(&self, constraint: &Constraint) -> f64 {
        match constraint {
            // Very selective constraints (eliminate many candidates)
            Constraint::Class(_) => 0.8,
            Constraint::Datatype(_) => 0.6,
            Constraint::NodeKind(_) => 0.3,
            Constraint::HasValue(_) => 0.05,
            Constraint::In(_) => 0.15,
            
            // Moderately selective constraints - MinCount is often very cheap to check
            Constraint::MinCount(_) | Constraint::MaxCount(_) => 0.1,
            Constraint::Pattern(_) => 0.5,
            Constraint::MinLength(_) | Constraint::MaxLength(_) => 0.6,
            
            // Less selective constraints
            Constraint::MinInclusive(_) | Constraint::MaxInclusive(_) => 0.7,
            Constraint::MinExclusive(_) | Constraint::MaxExclusive(_) => 0.7,
            
            // Variable selectivity (depends on implementation)
            Constraint::Sparql(_) => 0.8,
            Constraint::QualifiedValueShape(_) => 0.6,
            Constraint::Closed(_) => 0.4,
            
            // Logical constraints depend on sub-constraints
            Constraint::And(_) => 0.3, // AND is generally selective
            Constraint::Or(_) => 0.8,  // OR is generally less selective
            Constraint::Xone(_) => 0.5, // XOR is moderately selective
            Constraint::Not(_) => 0.9,  // NOT is generally less selective
            
            _ => 0.5, // Default moderate selectivity
        }
    }

    /// Update cost estimate for a constraint type based on actual performance
    pub fn update_cost_estimate(&mut self, constraint_type: &str, actual_cost: f64) {
        // Use exponential moving average to update cost estimates
        let alpha = 0.1; // Learning rate
        let current_estimate = self.cost_estimates.get(constraint_type).copied().unwrap_or(3.0);
        let new_estimate = alpha * actual_cost + (1.0 - alpha) * current_estimate;
        self.cost_estimates.insert(constraint_type.to_string(), new_estimate);
    }
}

/// Advanced validation optimization engine
#[derive(Debug)]
pub struct ValidationOptimizationEngine {
    /// Constraint cache for memoization
    cache: ConstraintCache,
    /// Dependency analyzer for ordering
    dependency_analyzer: ConstraintDependencyAnalyzer,
    /// Batch evaluator for efficient processing
    batch_evaluator: BatchConstraintEvaluator,
    /// Performance metrics
    metrics: Arc<RwLock<OptimizationMetrics>>,
    /// Configuration
    config: OptimizationConfig,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable constraint result caching
    pub enable_caching: bool,
    /// Enable parallel evaluation where possible
    pub enable_parallel: bool,
    /// Batch size for constraint evaluation
    pub batch_size: usize,
    /// Enable constraint reordering
    pub enable_reordering: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_parallel: false, // Disabled by default due to thread safety
            batch_size: 100,
            enable_reordering: true,
            max_cache_size: 10000,
            cache_ttl_secs: 300,
        }
    }
}

/// Optimization metrics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct OptimizationMetrics {
    /// Total constraint evaluations
    pub total_evaluations: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average evaluation time (microseconds)
    pub avg_evaluation_time_us: f64,
    /// Constraints reordered for optimization
    pub constraints_reordered: usize,
    /// Total optimization time saved (microseconds)
    pub optimization_time_saved_us: f64,
    /// Most expensive constraint types
    pub expensive_constraints: HashMap<String, f64>,
}

impl ValidationOptimizationEngine {
    /// Create a new optimization engine
    pub fn new(config: OptimizationConfig) -> Self {
        let cache = ConstraintCache::new(
            config.max_cache_size,
            Duration::from_secs(config.cache_ttl_secs),
        );
        let dependency_analyzer = ConstraintDependencyAnalyzer::default();
        let batch_evaluator = BatchConstraintEvaluator::new(
            cache.clone(),
            config.enable_parallel,
            config.batch_size,
        );
        
        Self {
            cache,
            dependency_analyzer,
            batch_evaluator,
            metrics: Arc::new(RwLock::new(OptimizationMetrics::default())),
            config,
        }
    }

    /// Optimize and evaluate a set of constraints
    pub fn optimize_and_evaluate(
        &mut self,
        store: &Store,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let start_time = Instant::now();
        
        // Step 1: Reorder constraints for optimal evaluation if enabled
        let optimized_constraints = if self.config.enable_reordering {
            self.reorder_constraints_for_optimization(constraints_with_contexts)
        } else {
            constraints_with_contexts
        };
        
        // Step 2: Evaluate using batch processing
        let results = if self.config.enable_caching {
            self.batch_evaluator.evaluate_batch(store, optimized_constraints)?
        } else {
            // Direct evaluation without caching
            self.evaluate_without_cache(store, optimized_constraints)?
        };
        
        // Step 3: Update metrics
        let total_time = start_time.elapsed();
        self.update_metrics(results.len(), total_time);
        
        Ok(results)
    }

    /// Reorder constraints based on cost and selectivity analysis
    fn reorder_constraints_for_optimization(
        &mut self,
        mut constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Vec<(Constraint, ConstraintContext)> {
        // Group by context to maintain constraint evaluation order within same context
        let mut context_groups: HashMap<String, Vec<(Constraint, ConstraintContext)>> = HashMap::new();
        
        for (constraint, context) in constraints_with_contexts {
            let context_key = format!("{:?}_{:?}", context.focus_node, context.shape_id);
            context_groups.entry(context_key).or_default().push((constraint, context));
        }
        
        let mut optimized = Vec::new();
        
        for (_, mut group) in context_groups {
            // Sort constraints within each context group
            group.sort_by(|(a, _), (b, _)| {
                let cost_a = self.dependency_analyzer.estimate_constraint_cost(a);
                let cost_b = self.dependency_analyzer.estimate_constraint_cost(b);
                let selectivity_a = self.dependency_analyzer.estimate_constraint_selectivity(a);
                let selectivity_b = self.dependency_analyzer.estimate_constraint_selectivity(b);
                
                // Primary: selectivity (more selective first)
                // Secondary: cost (lower cost first)
                selectivity_a.partial_cmp(&selectivity_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal))
            });
            
            optimized.extend(group);
        }
        
        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.constraints_reordered += optimized.len();
        }
        
        optimized
    }

    /// Evaluate constraints without caching (for comparison)
    fn evaluate_without_cache(
        &self,
        store: &Store,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let mut results = Vec::new();
        
        for (constraint, context) in constraints_with_contexts {
            let result = constraint.evaluate(store, &context)?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Update optimization metrics
    fn update_metrics(&self, evaluation_count: usize, total_time: Duration) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_evaluations += evaluation_count;
            
            let cache_stats = self.cache.stats();
            metrics.cache_hit_rate = cache_stats.hit_rate();
            metrics.avg_evaluation_time_us = cache_stats.avg_evaluation_time_us;
            
            // Estimate time saved through optimization
            let time_per_evaluation = total_time.as_micros() as f64 / evaluation_count as f64;
            metrics.optimization_time_saved_us += 
                cache_stats.hits as f64 * time_per_evaluation * 0.8; // Assume 80% time saving from cache hits
        }
    }

    /// Get current optimization metrics
    pub fn get_metrics(&self) -> OptimizationMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Clear all caches and reset metrics
    pub fn reset(&mut self) {
        self.cache.clear();
        if let Ok(mut metrics) = self.metrics.write() {
            *metrics = OptimizationMetrics::default();
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: OptimizationConfig) {
        self.config = config;
        // Note: Some config changes may require recreating components
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}


/// Advanced constraint evaluation orchestrator
#[derive(Debug)]
pub struct AdvancedConstraintEvaluator {
    batch_evaluator: BatchConstraintEvaluator,
    dependency_analyzer: ConstraintDependencyAnalyzer,
    enable_early_termination: bool,
}

impl Default for AdvancedConstraintEvaluator {
    fn default() -> Self {
        Self {
            batch_evaluator: BatchConstraintEvaluator::default(),
            dependency_analyzer: ConstraintDependencyAnalyzer::default(),
            enable_early_termination: true,
        }
    }
}

impl AdvancedConstraintEvaluator {
    /// Create new advanced evaluator with custom configuration
    pub fn new(
        cache: ConstraintCache,
        parallel: bool,
        batch_size: usize,
        early_termination: bool,
    ) -> Self {
        Self {
            batch_evaluator: BatchConstraintEvaluator::new(cache, parallel, batch_size),
            dependency_analyzer: ConstraintDependencyAnalyzer::default(),
            enable_early_termination: early_termination,
        }
    }

    /// Evaluate constraints with advanced optimizations
    pub fn evaluate_optimized(
        &self,
        store: &Store,
        constraints: Vec<Constraint>,
        context: ConstraintContext,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        // Optimize constraint order
        let optimized_constraints = self
            .dependency_analyzer
            .optimize_constraint_order(constraints);

        // Prepare constraint-context pairs
        let constraints_with_contexts: Vec<_> = optimized_constraints
            .into_iter()
            .map(|c| (c, context.clone()))
            .collect();

        // Evaluate with early termination if enabled
        if self.enable_early_termination {
            let mut results = Vec::new();

            for (constraint, ctx) in constraints_with_contexts {
                // Try cache first
                let result = if let Some(cached) = self.batch_evaluator.cache.get(&constraint, &ctx)
                {
                    cached
                } else {
                    let start_time = Instant::now();
                    let result = constraint.evaluate(store, &ctx)?;
                    let evaluation_time = start_time.elapsed();
                    self.batch_evaluator.cache.put(
                        &constraint,
                        &ctx,
                        result.clone(),
                        evaluation_time,
                    );
                    result
                };

                results.push(result.clone());

                // Early termination on first violation for performance
                if result.is_violated() {
                    // For the remaining constraints, return "not evaluated" or continue based on use case
                    // For now, we'll continue evaluating all constraints
                }
            }

            Ok(results)
        } else {
            // Evaluate all constraints in batches
            self.batch_evaluator
                .evaluate_batch(store, constraints_with_contexts)
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> ConstraintPerformanceStats {
        let cache_stats = self.batch_evaluator.cache_stats();
        ConstraintPerformanceStats {
            cache_hit_rate: cache_stats.hit_rate(),
            total_evaluations: cache_stats.evaluations,
            avg_evaluation_time_us: cache_stats.avg_evaluation_time_us,
            cache_evictions: cache_stats.evictions,
        }
    }
}

/// Performance statistics for constraint evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPerformanceStats {
    pub cache_hit_rate: f64,
    pub total_evaluations: usize,
    pub avg_evaluation_time_us: f64,
    pub cache_evictions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{ClassConstraint, DatatypeConstraint, MinCountConstraint};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_constraint_cache() {
        let cache = ConstraintCache::new(100, Duration::from_secs(60));

        let constraint = Constraint::Class(ClassConstraint {
            class_iri: NamedNode::new("http://example.org/Person").unwrap(),
        });

        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
            ShapeId::new("PersonShape"),
        );

        // Should be cache miss initially
        assert!(cache.get(&constraint, &context).is_none());
    }

    #[test]
    fn test_constraint_ordering() {
        let analyzer = ConstraintDependencyAnalyzer::default();

        let constraints = vec![
            Constraint::Class(ClassConstraint {
                class_iri: NamedNode::new("http://example.org/Person").unwrap(),
            }),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
            }),
        ];

        let optimized = analyzer.optimize_constraint_order(constraints);

        // MinCount should come first (low selectivity), then datatype, then class
        assert!(matches!(optimized[0], Constraint::MinCount(_)));
    }
}

/// Streaming validation engine for large datasets
#[derive(Debug)]
pub struct StreamingValidationEngine {
    /// Batch size for streaming processing
    batch_size: usize,
    /// Memory threshold in bytes
    memory_threshold: usize,
    /// Enable memory monitoring
    memory_monitoring: bool,
    /// Advanced constraint evaluator
    evaluator: AdvancedConstraintEvaluator,
}

impl Default for StreamingValidationEngine {
    fn default() -> Self {
        Self::new(1000, 100 * 1024 * 1024, true) // 1k batch, 100MB memory limit
    }
}

impl StreamingValidationEngine {
    /// Create new streaming validation engine
    pub fn new(batch_size: usize, memory_threshold: usize, memory_monitoring: bool) -> Self {
        let cache = ConstraintCache::new(10000, Duration::from_secs(300));
        let evaluator = AdvancedConstraintEvaluator::new(cache, true, batch_size / 4, true);

        Self {
            batch_size,
            memory_threshold,
            memory_monitoring,
            evaluator,
        }
    }

    /// Validate large dataset in streaming fashion
    pub fn validate_streaming<I>(
        &self,
        store: &Store,
        constraints: Vec<Constraint>,
        node_stream: I,
    ) -> Result<StreamingValidationResult>
    where
        I: Iterator<Item = Term>,
    {
        let mut result = StreamingValidationResult::new();
        let mut current_batch = Vec::new();
        let mut processed_count = 0;

        for node in node_stream {
            current_batch.push(node);

            // Process batch when full
            if current_batch.len() >= self.batch_size {
                let batch_result = self.process_batch(store, &constraints, &current_batch)?;
                result.merge_batch_result(batch_result);

                processed_count += current_batch.len();
                current_batch.clear();

                // Memory monitoring
                if self.memory_monitoring && self.check_memory_pressure()? {
                    result.memory_pressure_events += 1;

                    // Trigger garbage collection or cache eviction
                    self.evaluator.batch_evaluator.cache.clear();

                    // Could also implement memory spill-to-disk here
                    tracing::warn!("Memory pressure detected, cleared cache");
                }

                // Progress reporting
                if processed_count % (self.batch_size * 10) == 0 {
                    tracing::info!("Processed {} nodes", processed_count);
                }
            }
        }

        // Process remaining batch
        if !current_batch.is_empty() {
            let batch_result = self.process_batch(store, &constraints, &current_batch)?;
            result.merge_batch_result(batch_result);
        }

        result.total_nodes = processed_count + current_batch.len();
        Ok(result)
    }

    /// Process a single batch of nodes
    fn process_batch(
        &self,
        store: &Store,
        constraints: &[Constraint],
        nodes: &[Term],
    ) -> Result<BatchValidationResult> {
        let mut batch_result = BatchValidationResult::new();
        let start_time = Instant::now();

        for node in nodes {
            let context = ConstraintContext::new(node.clone(), ShapeId::new("BatchValidation"));

            let constraint_results =
                self.evaluator
                    .evaluate_optimized(store, constraints.to_vec(), context)?;

            // Count violations in this batch
            let violations = constraint_results
                .iter()
                .filter(|r| r.is_violated())
                .count();
            batch_result.violation_count += violations;
            batch_result.node_count += 1;
        }

        batch_result.processing_time = start_time.elapsed();
        Ok(batch_result)
    }

    /// Check if memory usage is approaching threshold
    fn check_memory_pressure(&self) -> Result<bool> {
        if !self.memory_monitoring {
            return Ok(false);
        }

        // Simple memory check - in practice would use more sophisticated monitoring
        let stats = self.evaluator.get_performance_stats();

        // Heuristic: if we have many cache evictions, we're under memory pressure
        Ok(stats.cache_evictions > 100 && stats.cache_hit_rate < 0.5)
    }
}

/// Result of streaming validation
#[derive(Debug, Clone)]
pub struct StreamingValidationResult {
    pub total_nodes: usize,
    pub total_violations: usize,
    pub total_processing_time: Duration,
    pub memory_pressure_events: usize,
    pub batches_processed: usize,
}

impl StreamingValidationResult {
    fn new() -> Self {
        Self {
            total_nodes: 0,
            total_violations: 0,
            total_processing_time: Duration::ZERO,
            memory_pressure_events: 0,
            batches_processed: 0,
        }
    }

    fn merge_batch_result(&mut self, batch: BatchValidationResult) {
        self.total_violations += batch.violation_count;
        self.total_processing_time += batch.processing_time;
        self.batches_processed += 1;
    }
}

/// Result of processing a single batch
#[derive(Debug, Clone)]
struct BatchValidationResult {
    pub node_count: usize,
    pub violation_count: usize,
    pub processing_time: Duration,
}

impl BatchValidationResult {
    fn new() -> Self {
        Self {
            node_count: 0,
            violation_count: 0,
            processing_time: Duration::ZERO,
        }
    }
}

/// Incremental validation engine for change-based validation
#[derive(Debug)]
pub struct IncrementalValidationEngine {
    /// Cache for previous validation results
    previous_results: Arc<RwLock<HashMap<Term, ValidationSnapshot>>>,
    /// Advanced evaluator
    evaluator: AdvancedConstraintEvaluator,
    /// Change detection sensitivity
    change_detection_level: ChangeDetectionLevel,
}

/// Validation snapshot for incremental processing
#[derive(Debug, Clone)]
struct ValidationSnapshot {
    /// Node that was validated
    node: Term,
    /// Hash of constraints that were applied
    constraints_hash: u64,
    /// Hash of the node's properties at validation time
    properties_hash: u64,
    /// Validation result
    result: Vec<ConstraintEvaluationResult>,
    /// Timestamp of validation
    validated_at: Instant,
}

/// Level of change detection for incremental validation
#[derive(Debug, Clone)]
pub enum ChangeDetectionLevel {
    /// Only detect if node identity changed
    NodeOnly,
    /// Detect changes in immediate properties
    Properties,
    /// Detect changes in entire subgraph
    SubGraph,
}

impl Default for IncrementalValidationEngine {
    fn default() -> Self {
        let cache = ConstraintCache::new(50000, Duration::from_secs(3600)); // Larger cache for incremental
        let evaluator = AdvancedConstraintEvaluator::new(cache, true, 100, false);

        Self {
            previous_results: Arc::new(RwLock::new(HashMap::new())),
            evaluator,
            change_detection_level: ChangeDetectionLevel::Properties,
        }
    }
}

impl IncrementalValidationEngine {
    /// Validate only changed nodes since last validation
    pub fn validate_incremental(
        &mut self,
        store: &Store,
        constraints: Vec<Constraint>,
        nodes: &[Term],
        force_revalidate: bool,
    ) -> Result<IncrementalValidationResult> {
        let mut result = IncrementalValidationResult::new();
        let start_time = Instant::now();

        let constraints_hash = self.hash_constraints(&constraints);

        for node in nodes {
            let properties_hash = self.hash_node_properties(store, node)?;

            let needs_validation = force_revalidate || {
                let previous_results = self.previous_results.read().unwrap();
                match previous_results.get(node) {
                    Some(snapshot) => {
                        // Check if constraints or properties changed
                        snapshot.constraints_hash != constraints_hash
                            || snapshot.properties_hash != properties_hash
                    }
                    None => true, // Never validated before
                }
            };

            if needs_validation {
                let context =
                    ConstraintContext::new(node.clone(), ShapeId::new("IncrementalValidation"));

                let constraint_results =
                    self.evaluator
                        .evaluate_optimized(store, constraints.clone(), context)?;

                // Update snapshot
                let snapshot = ValidationSnapshot {
                    node: node.clone(),
                    constraints_hash,
                    properties_hash,
                    result: constraint_results.clone(),
                    validated_at: Instant::now(),
                };

                {
                    let mut previous_results = self.previous_results.write().unwrap();
                    previous_results.insert(node.clone(), snapshot);
                }

                let violations = constraint_results
                    .iter()
                    .filter(|r| r.is_violated())
                    .count();
                result.revalidated_nodes += 1;
                result.new_violations += violations;
            } else {
                result.skipped_nodes += 1;
            }
        }

        result.total_processing_time = start_time.elapsed();
        Ok(result)
    }

    /// Hash constraints for change detection
    fn hash_constraints(&self, constraints: &[Constraint]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for constraint in constraints {
            format!("{:?}", constraint).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Hash node properties for change detection
    fn hash_node_properties(&self, store: &Store, node: &Term) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // In practice, this would query the store for all triples where node is subject
        // For now, we'll use a simplified hash based on the node itself
        format!("{:?}", node).hash(&mut hasher);

        // TODO: Implement actual property querying when store interface is available
        // Example: for triple in store.triples_with_subject(node) { triple.hash(&mut hasher); }

        Ok(hasher.finish())
    }

    /// Clear validation history
    pub fn clear_history(&mut self) {
        self.previous_results.write().unwrap().clear();
    }

    /// Get statistics about incremental validation
    pub fn get_incremental_stats(&self) -> IncrementalValidationStats {
        let snapshots = self.previous_results.read().unwrap();
        IncrementalValidationStats {
            cached_validations: snapshots.len(),
            memory_usage_mb: snapshots.len() * std::mem::size_of::<ValidationSnapshot>()
                / (1024 * 1024),
        }
    }
}

/// Result of incremental validation
#[derive(Debug, Clone)]
pub struct IncrementalValidationResult {
    pub revalidated_nodes: usize,
    pub skipped_nodes: usize,
    pub new_violations: usize,
    pub total_processing_time: Duration,
}

impl IncrementalValidationResult {
    fn new() -> Self {
        Self {
            revalidated_nodes: 0,
            skipped_nodes: 0,
            new_violations: 0,
            total_processing_time: Duration::ZERO,
        }
    }

    pub fn efficiency_ratio(&self) -> f64 {
        let total_nodes = self.revalidated_nodes + self.skipped_nodes;
        if total_nodes == 0 {
            0.0
        } else {
            self.skipped_nodes as f64 / total_nodes as f64
        }
    }
}

/// Statistics for incremental validation
#[derive(Debug, Clone)]
pub struct IncrementalValidationStats {
    pub cached_validations: usize,
    pub memory_usage_mb: usize,
}
