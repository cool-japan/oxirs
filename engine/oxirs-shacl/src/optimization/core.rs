//! SHACL Constraint Evaluation Optimizations
//!
//! This module provides performance optimizations for SHACL constraint evaluation
//! including result caching, batch validation, and dependency-aware evaluation ordering.

#![allow(dead_code)]

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult},
    PropertyPath, Result, ShaclError, ShapeId,
};
use oxirs_core::{model::Term, RdfTerm, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
            let cache = self
                .cache
                .read()
                .expect("cache lock should not be poisoned");
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
                let mut stats = self
                    .stats
                    .write()
                    .expect("stats lock should not be poisoned");
                stats.hits += 1;
            }

            // Increment hit count
            {
                let mut cache_mut = self
                    .cache
                    .write()
                    .expect("cache lock should not be poisoned");
                if let Some(entry) = cache_mut.get_mut(&key) {
                    entry.hit_count += 1;
                }
            }

            return result;
        }

        // Cache miss
        let mut stats = self
            .stats
            .write()
            .expect("stats lock should not be poisoned");
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
            let mut cache = self
                .cache
                .write()
                .expect("cache lock should not be poisoned");

            // Evict entries if cache is full
            if cache.len() >= self.max_size {
                self.evict_entries(&mut cache);
            }

            cache.insert(key, cached_result);
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .expect("stats lock should not be poisoned");
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
                format!("{constraint:?}").hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Hash values for caching
    fn hash_values(&self, values: &[Term]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for value in values {
            format!("{value:?}").hash(&mut hasher);
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
        let mut stats = self
            .stats
            .write()
            .expect("stats lock should not be poisoned");
        stats.evictions += evict_count;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats
            .read()
            .expect("stats lock should not be poisoned")
            .clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache
            .write()
            .expect("cache lock should not be poisoned")
            .clear();
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
        store: &dyn Store,
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
        store: &dyn Store,
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
        store: &dyn Store,
        batch: &[(Constraint, ConstraintContext)],
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        // For parallel evaluation, we need to be careful about thread safety
        // We'll use a thread pool for CPU-bound constraint evaluation

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
    pub fn estimate_constraint_cost(&self, constraint: &Constraint) -> f64 {
        let base_cost = match constraint {
            Constraint::Class(_) => self.cost_estimates.get("class").copied().unwrap_or(5.0),
            Constraint::Datatype(_) => self.cost_estimates.get("datatype").copied().unwrap_or(1.0),
            Constraint::NodeKind(_) => self.cost_estimates.get("nodeKind").copied().unwrap_or(1.0),
            Constraint::MinCount(_) => self.cost_estimates.get("minCount").copied().unwrap_or(1.0),
            Constraint::MaxCount(_) => self.cost_estimates.get("maxCount").copied().unwrap_or(1.0),
            Constraint::Pattern(_) => self.cost_estimates.get("pattern").copied().unwrap_or(3.0),
            Constraint::Sparql(_) => self.cost_estimates.get("sparql").copied().unwrap_or(10.0),
            Constraint::QualifiedValueShape(_) => self
                .cost_estimates
                .get("qualifiedValueShape")
                .copied()
                .unwrap_or(8.0),
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
    pub fn estimate_constraint_selectivity(&self, constraint: &Constraint) -> f64 {
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
            Constraint::And(_) => 0.3,  // AND is generally selective
            Constraint::Or(_) => 0.8,   // OR is generally less selective
            Constraint::Xone(_) => 0.5, // XOR is moderately selective
            Constraint::Not(_) => 0.9,  // NOT is generally less selective

            _ => 0.5, // Default moderate selectivity
        }
    }

    /// Update cost estimate for a constraint type based on actual performance
    pub fn update_cost_estimate(&mut self, constraint_type: &str, actual_cost: f64) {
        // Use exponential moving average to update cost estimates
        let alpha = 0.1; // Learning rate
        let current_estimate = self
            .cost_estimates
            .get(constraint_type)
            .copied()
            .unwrap_or(3.0);
        let new_estimate = alpha * actual_cost + (1.0 - alpha) * current_estimate;
        self.cost_estimates
            .insert(constraint_type.to_string(), new_estimate);
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
        let batch_evaluator =
            BatchConstraintEvaluator::new(cache.clone(), config.enable_parallel, config.batch_size);

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
        store: &dyn Store,
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
            self.batch_evaluator
                .evaluate_batch(store, optimized_constraints)?
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
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Vec<(Constraint, ConstraintContext)> {
        // Group by context to maintain constraint evaluation order within same context
        let mut context_groups: HashMap<String, Vec<(Constraint, ConstraintContext)>> =
            HashMap::new();

        for (constraint, context) in constraints_with_contexts {
            let context_key = format!("{:?}_{:?}", context.focus_node, context.shape_id);
            context_groups
                .entry(context_key)
                .or_default()
                .push((constraint, context));
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
                selectivity_a
                    .partial_cmp(&selectivity_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        cost_a
                            .partial_cmp(&cost_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
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
        store: &dyn Store,
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
        self.metrics
            .read()
            .expect("metrics lock should not be poisoned")
            .clone()
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
        store: &dyn Store,
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
            class_iri: NamedNode::new("http://example.org/Person").expect("valid IRI"),
        });

        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").expect("valid IRI")),
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
                class_iri: NamedNode::new("http://example.org/Person").expect("valid IRI"),
            }),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string")
                    .expect("valid IRI"),
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
        store: &dyn Store,
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
        store: &dyn Store,
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
        store: &dyn Store,
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
                let previous_results = self
                    .previous_results
                    .read()
                    .expect("read lock should not be poisoned");
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
                    let mut previous_results = self
                        .previous_results
                        .write()
                        .expect("write lock should not be poisoned");
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
            format!("{constraint:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Hash node properties for change detection with comprehensive RDF triple analysis
    fn hash_node_properties(&self, store: &dyn Store, node: &Term) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        let mut hasher = DefaultHasher::new();

        // Hash all triples where this node is the subject
        match self.query_node_triples_as_subject(store, node) {
            Ok(subject_triples) => {
                for triple in subject_triples {
                    triple.subject().as_str().hash(&mut hasher);
                    triple.predicate().as_str().hash(&mut hasher);
                    triple.object().as_str().hash(&mut hasher);
                }
            }
            Err(_) => {
                // Fallback to node-only hash if store query fails
                node.as_str().hash(&mut hasher);
            }
        }

        // Hash all triples where this node is the object (to detect incoming references)
        match self.query_node_triples_as_object(store, node) {
            Ok(object_triples) => {
                for triple in object_triples {
                    triple.subject().as_str().hash(&mut hasher);
                    triple.predicate().as_str().hash(&mut hasher);
                    triple.object().as_str().hash(&mut hasher);
                }
            }
            Err(_) => {
                // Continue without incoming reference detection if query fails
            }
        }

        Ok(hasher.finish())
    }

    /// Query store for all triples where the node is the subject
    fn query_node_triples_as_subject(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<oxirs_core::model::Triple>> {
        let mut triples = Vec::new();

        // Create a pattern to match triples with this node as subject
        let subject = match node {
            Term::NamedNode(nn) => Some(oxirs_core::model::Subject::NamedNode(nn.clone())),
            Term::BlankNode(bn) => Some(oxirs_core::model::Subject::BlankNode(bn.clone())),
            Term::Variable(v) => Some(oxirs_core::model::Subject::Variable(v.clone())),
            _ => None,
        };
        let quads = match subject {
            Some(s) => store.find_quads(Some(&s), None, None, None)?,
            None => Vec::new(),
        };
        for quad in quads {
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Query store for all triples where the node is the object
    fn query_node_triples_as_object(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<oxirs_core::model::Triple>> {
        let mut triples = Vec::new();

        // Create a pattern to match triples with this node as object
        let object = match node {
            Term::NamedNode(nn) => Some(oxirs_core::model::Object::NamedNode(nn.clone())),
            Term::BlankNode(bn) => Some(oxirs_core::model::Object::BlankNode(bn.clone())),
            Term::Literal(lit) => Some(oxirs_core::model::Object::Literal(lit.clone())),
            Term::Variable(v) => Some(oxirs_core::model::Object::Variable(v.clone())),
            _ => None,
        };
        let quads = match object {
            Some(o) => store.find_quads(None, None, Some(&o), None)?,
            None => Vec::new(),
        };
        for quad in quads {
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Clear validation history
    pub fn clear_history(&mut self) {
        self.previous_results
            .write()
            .expect("write lock should not be poisoned")
            .clear();
    }

    /// Get statistics about incremental validation
    pub fn get_incremental_stats(&self) -> IncrementalValidationStats {
        let snapshots = self
            .previous_results
            .read()
            .expect("read lock should not be poisoned");
        IncrementalValidationStats {
            cached_validations: snapshots.len(),
            memory_usage_mb: snapshots.len() * std::mem::size_of::<ValidationSnapshot>()
                / (1024 * 1024),
        }
    }

    /// Compute detailed change delta between current state and cached snapshots
    pub fn compute_change_delta(
        &self,
        store: &dyn Store,
        current_constraints: &[Constraint],
        nodes: &[Term],
    ) -> Result<ChangesDelta> {
        let mut delta = ChangesDelta::new();
        let snapshots = self
            .previous_results
            .read()
            .expect("read lock should not be poisoned");

        for node in nodes {
            // Check if we have a previous snapshot for this node
            if let Some(previous_snapshot) = snapshots.get(node) {
                // Compute current hashes
                let current_property_hash = self.hash_node_properties(store, node)?;
                let current_constraint_hash = self.hash_constraints(current_constraints);

                // Check for property changes
                if current_property_hash != previous_snapshot.properties_hash {
                    delta.nodes_with_property_changes.push(NodePropertyChange {
                        node: node.clone(),
                        previous_hash: previous_snapshot.properties_hash,
                        current_hash: current_property_hash,
                        property_changes: self.compute_property_changes(store, node)?,
                        detected_at: std::time::SystemTime::now(),
                    });
                }

                // Check for constraint changes
                if current_constraint_hash != previous_snapshot.constraints_hash {
                    delta
                        .nodes_with_constraint_changes
                        .push(NodeConstraintChange {
                            node: node.clone(),
                            previous_constraints_hash: previous_snapshot.constraints_hash,
                            current_constraints_hash: current_constraint_hash,
                            changed_shapes: vec![], // Could be enhanced to track specific shape changes
                            detected_at: std::time::SystemTime::now(),
                        });
                }
            } else {
                // New node detected
                delta.new_nodes.push(node.clone());
            }
        }

        // Detect deleted nodes (in snapshots but not in current nodes)
        let current_nodes: std::collections::HashSet<&Term> = nodes.iter().collect();

        for snapshot_node in snapshots.keys() {
            if !current_nodes.contains(snapshot_node) {
                delta.deleted_nodes.push(snapshot_node.clone());
            }
        }

        Ok(delta)
    }

    /// Generate change events for external system integration
    pub fn generate_change_events(
        &self,
        delta: &ChangesDelta,
        _validation_results: &[crate::constraints::ConstraintEvaluationResult],
    ) -> Vec<ChangeEvent> {
        let mut events = Vec::new();
        let timestamp = std::time::SystemTime::now();

        // Generate events for property changes
        for property_change in &delta.nodes_with_property_changes {
            let event_id = format!(
                "prop_change_{}_{}",
                property_change.node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": property_change.node.as_str(),
                "previous_hash": property_change.previous_hash,
                "current_hash": property_change.current_hash,
                "detected_at": property_change.detected_at
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodePropertiesChanged,
                node: property_change.node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        // Generate events for constraint changes
        for constraint_change in &delta.nodes_with_constraint_changes {
            let event_id = format!(
                "constraint_change_{}_{}",
                constraint_change.node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": constraint_change.node.as_str(),
                "previous_constraints_hash": constraint_change.previous_constraints_hash,
                "current_constraints_hash": constraint_change.current_constraints_hash,
                "changed_shapes": constraint_change.changed_shapes
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::ShapeConstraintsChanged,
                node: constraint_change.node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        // Generate events for new nodes
        for new_node in &delta.new_nodes {
            let event_id = format!(
                "node_added_{}_{}",
                new_node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": new_node.as_str(),
                "detected_at": timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodeAdded,
                node: new_node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        // Generate events for deleted nodes
        for deleted_node in &delta.deleted_nodes {
            let event_id = format!(
                "node_removed_{}_{}",
                deleted_node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": deleted_node.as_str(),
                "detected_at": timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodeRemoved,
                node: deleted_node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        events
    }

    /// Compute specific property changes for a node (simplified implementation)
    fn compute_property_changes(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<PropertyChange>> {
        // This is a simplified implementation that could be enhanced with:
        // 1. Actual before/after triple comparison
        // 2. Property-specific change detection
        // 3. Integration with store change logs

        let mut changes = Vec::new();
        let current_triples = self.query_node_triples_as_subject(store, node)?;

        // For each triple, we could compare with cached previous state
        // For now, we'll create a placeholder change indicating some property changed
        if !current_triples.is_empty() {
            for triple in current_triples.iter().take(5) {
                // Limit for performance
                // Convert predicate to NamedNode and object to Term
                if let oxirs_core::model::Predicate::NamedNode(predicate_nn) = triple.predicate() {
                    changes.push(PropertyChange {
                        subject: node.clone(),
                        property: predicate_nn.clone(),
                        change_type: PropertyChangeType::Modified, // Simplified assumption
                        old_value: None, // Would need previous state comparison
                        new_value: Some(match triple.object() {
                            oxirs_core::model::Object::NamedNode(nn) => Term::NamedNode(nn.clone()),
                            oxirs_core::model::Object::BlankNode(bn) => Term::BlankNode(bn.clone()),
                            oxirs_core::model::Object::Literal(lit) => Term::Literal(lit.clone()),
                            oxirs_core::model::Object::Variable(v) => Term::Variable(v.clone()),
                            _ => continue, // Skip unsupported object types
                        }),
                        timestamp: std::time::SystemTime::now(),
                    });
                }
            }
        }

        Ok(changes)
    }

    /// Reconstruct term from string key (simplified approach)
    fn reconstruct_term_from_key(&self, key: &str) -> Result<Term> {
        // This is a simplified implementation that could be enhanced
        // In practice, you'd want a more robust term serialization/deserialization
        if key.starts_with("NamedNode(") {
            let iri = key
                .trim_start_matches("NamedNode(\"")
                .trim_end_matches("\")");
            oxirs_core::model::NamedNode::new(iri)
                .map(Term::NamedNode)
                .map_err(|e| ShaclError::ValidationEngine(format!("Invalid IRI: {e}")))
        } else {
            Err(ShaclError::ValidationEngine(format!(
                "Unsupported term key format: {key}"
            )))
        }
    }
}

/// Enhanced result of incremental validation with delta processing
#[derive(Debug, Clone)]
pub struct IncrementalValidationResult {
    /// Nodes that were revalidated due to changes
    pub revalidated_nodes: usize,
    /// Nodes that were skipped (no changes detected)
    pub skipped_nodes: usize,
    /// New violations found during incremental validation
    pub new_violations: usize,
    /// Total processing time for incremental validation
    pub total_processing_time: Duration,
    /// Detailed change delta information
    pub change_delta: ChangesDelta,
    /// Specific violations that were resolved (no longer violations)
    pub resolved_violations: Vec<crate::validation::ValidationViolation>,
    /// Specific new violations found
    pub new_violation_details: Vec<crate::validation::ValidationViolation>,
    /// Change events for external systems
    pub change_events: Vec<ChangeEvent>,
}

impl IncrementalValidationResult {
    fn new() -> Self {
        Self {
            revalidated_nodes: 0,
            skipped_nodes: 0,
            new_violations: 0,
            total_processing_time: Duration::ZERO,
            change_delta: ChangesDelta::new(),
            resolved_violations: Vec::new(),
            new_violation_details: Vec::new(),
            change_events: Vec::new(),
        }
    }

    /// Get the efficiency ratio (percentage of nodes skipped)
    pub fn efficiency_ratio(&self) -> f64 {
        let total_nodes = self.revalidated_nodes + self.skipped_nodes;
        if total_nodes == 0 {
            0.0
        } else {
            self.skipped_nodes as f64 / total_nodes as f64
        }
    }

    /// Get the net change in violations (positive = more violations, negative = fewer)
    pub fn net_violation_change(&self) -> i32 {
        self.new_violation_details.len() as i32 - self.resolved_violations.len() as i32
    }

    /// Check if this incremental validation improved overall conformance
    pub fn improved_conformance(&self) -> bool {
        self.resolved_violations.len() > self.new_violation_details.len()
    }

    /// Get summary of changes for reporting
    pub fn change_summary(&self) -> String {
        format!(
            "Incremental validation: {} nodes revalidated, {} skipped ({}% efficiency), {} net violation change",
            self.revalidated_nodes,
            self.skipped_nodes,
            (self.efficiency_ratio() * 100.0) as u32,
            self.net_violation_change()
        )
    }
}

/// Comprehensive delta information about detected changes
#[derive(Debug, Clone)]
pub struct ChangesDelta {
    /// Nodes that had property changes
    pub nodes_with_property_changes: Vec<NodePropertyChange>,
    /// Nodes that had constraint changes (shape modifications)
    pub nodes_with_constraint_changes: Vec<NodeConstraintChange>,
    /// New nodes detected
    pub new_nodes: Vec<oxirs_core::model::Term>,
    /// Deleted nodes detected
    pub deleted_nodes: Vec<oxirs_core::model::Term>,
    /// Property-level changes
    pub property_changes: Vec<PropertyChange>,
}

impl ChangesDelta {
    fn new() -> Self {
        Self {
            nodes_with_property_changes: Vec::new(),
            nodes_with_constraint_changes: Vec::new(),
            new_nodes: Vec::new(),
            deleted_nodes: Vec::new(),
            property_changes: Vec::new(),
        }
    }

    /// Check if any significant changes were detected
    pub fn has_changes(&self) -> bool {
        !self.nodes_with_property_changes.is_empty()
            || !self.nodes_with_constraint_changes.is_empty()
            || !self.new_nodes.is_empty()
            || !self.deleted_nodes.is_empty()
            || !self.property_changes.is_empty()
    }

    /// Get total number of changed entities
    pub fn total_changes(&self) -> usize {
        self.nodes_with_property_changes.len()
            + self.nodes_with_constraint_changes.len()
            + self.new_nodes.len()
            + self.deleted_nodes.len()
            + self.property_changes.len()
    }
}

/// Details about property changes for a specific node
#[derive(Debug, Clone)]
pub struct NodePropertyChange {
    /// The node that changed
    pub node: oxirs_core::model::Term,
    /// Hash before the change
    pub previous_hash: u64,
    /// Hash after the change
    pub current_hash: u64,
    /// Specific property changes (if available)
    pub property_changes: Vec<PropertyChange>,
    /// Timestamp of change detection
    pub detected_at: std::time::SystemTime,
}

/// Details about constraint changes affecting a node
#[derive(Debug, Clone)]
pub struct NodeConstraintChange {
    /// The node affected by constraint changes
    pub node: oxirs_core::model::Term,
    /// Hash of constraints before the change
    pub previous_constraints_hash: u64,
    /// Hash of constraints after the change
    pub current_constraints_hash: u64,
    /// Shape IDs that changed
    pub changed_shapes: Vec<crate::ShapeId>,
    /// Timestamp of change detection
    pub detected_at: std::time::SystemTime,
}

/// Specific property-level change information
#[derive(Debug, Clone)]
pub struct PropertyChange {
    /// Subject of the changed triple
    pub subject: oxirs_core::model::Term,
    /// Property/predicate that changed
    pub property: oxirs_core::model::NamedNode,
    /// Change type
    pub change_type: PropertyChangeType,
    /// Old value (for modifications and deletions)
    pub old_value: Option<oxirs_core::model::Term>,
    /// New value (for modifications and additions)
    pub new_value: Option<oxirs_core::model::Term>,
    /// Timestamp of the change
    pub timestamp: std::time::SystemTime,
}

/// Type of property change
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyChangeType {
    /// Property value was added
    Added,
    /// Property value was modified
    Modified,
    /// Property value was deleted
    Deleted,
}

/// Change events for external system integration
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// Event type
    pub event_type: ChangeEventType,
    /// Node affected by the change
    pub node: oxirs_core::model::Term,
    /// Shape context (if applicable)
    pub shape_context: Option<crate::ShapeId>,
    /// Event payload with detailed information
    pub payload: serde_json::Value,
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
    /// Event ID for tracking
    pub event_id: String,
}

/// Types of change events
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeEventType {
    /// Node validation status changed
    ValidationStatusChanged,
    /// New violation detected
    ViolationAdded,
    /// Violation resolved
    ViolationResolved,
    /// Node properties changed
    NodePropertiesChanged,
    /// Shape constraints changed
    ShapeConstraintsChanged,
    /// New node added to validation scope
    NodeAdded,
    /// Node removed from validation scope
    NodeRemoved,
}

/// Statistics for incremental validation
#[derive(Debug, Clone)]
pub struct IncrementalValidationStats {
    pub cached_validations: usize,
    pub memory_usage_mb: usize,
}
