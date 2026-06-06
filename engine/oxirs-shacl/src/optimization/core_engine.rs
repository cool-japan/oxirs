//! SHACL Optimization Engine — Cache and Batch Evaluation
//!
//! Provides `ConstraintCache`, `BatchConstraintEvaluator`, and
//! `ValidationOptimizationEngine` with their supporting configuration and metrics.

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult},
    PropertyPath, Result, ShapeId,
};
use oxirs_core::{model::Term, Store};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::core_strategies::ConstraintDependencyAnalyzer;

// ─── Cache internals ────────────────────────────────────────────────────────

/// Cache key for constraint evaluation results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct CacheKey {
    constraint_hash: u64,
    focus_node: Term,
    path: Option<PropertyPath>,
    values_hash: u64,
    shape_id: ShapeId,
}

/// Cached constraint evaluation result
#[derive(Debug, Clone)]
pub(crate) struct CachedResult {
    pub(crate) result: ConstraintEvaluationResult,
    pub(crate) cached_at: Instant,
    pub(crate) hit_count: usize,
}

// ─── CacheStats ─────────────────────────────────────────────────────────────

/// Cache statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evaluations: usize,
    pub avg_evaluation_time_us: f64,
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

// ─── ConstraintCache ─────────────────────────────────────────────────────────

/// Constraint evaluation cache for performance optimization
#[derive(Debug, Clone)]
pub struct ConstraintCache {
    cache: Arc<RwLock<HashMap<CacheKey, CachedResult>>>,
    stats: Arc<RwLock<CacheStats>>,
    max_size: usize,
    ttl: Duration,
}

impl Default for ConstraintCache {
    fn default() -> Self {
        Self::new(10000, Duration::from_secs(300))
    }
}

impl ConstraintCache {
    /// Create a new constraint cache with the given capacity and TTL.
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            max_size,
            ttl,
        }
    }

    /// Get a cached result if available and not expired.
    pub fn get(
        &self,
        constraint: &Constraint,
        context: &ConstraintContext,
    ) -> Option<ConstraintEvaluationResult> {
        let key = self.create_cache_key(constraint, context);

        let result = {
            let cache = self
                .cache
                .read()
                .expect("cache lock should not be poisoned");
            if let Some(cached) = cache.get(&key) {
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
            {
                let mut stats = self
                    .stats
                    .write()
                    .expect("stats lock should not be poisoned");
                stats.hits += 1;
            }
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

        let mut stats = self
            .stats
            .write()
            .expect("stats lock should not be poisoned");
        stats.misses += 1;
        None
    }

    /// Store a result in the cache, evicting entries if at capacity.
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

            if cache.len() >= self.max_size {
                self.evict_entries(&mut cache);
            }

            cache.insert(key, cached_result);
        }

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

    fn hash_constraint(&self, constraint: &Constraint) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

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
            _ => {
                format!("{constraint:?}").hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    fn hash_values(&self, values: &[Term]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for value in values {
            format!("{value:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Evict 25% of entries using LFU (Least Frequently Used) strategy.
    fn evict_entries(&self, cache: &mut HashMap<CacheKey, CachedResult>) {
        let evict_count = cache.len() / 4;

        let mut entries: Vec<_> = cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| {
            let hit_cmp = a.1.hit_count.cmp(&b.1.hit_count);
            if hit_cmp == std::cmp::Ordering::Equal {
                b.1.cached_at.cmp(&a.1.cached_at)
            } else {
                hit_cmp
            }
        });

        let keys_to_remove: Vec<_> = entries
            .iter()
            .take(evict_count)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            cache.remove(&key);
        }

        let mut stats = self
            .stats
            .write()
            .expect("stats lock should not be poisoned");
        stats.evictions += evict_count;
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats
            .read()
            .expect("stats lock should not be poisoned")
            .clone()
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.cache
            .write()
            .expect("cache lock should not be poisoned")
            .clear();
    }
}

// ─── BatchConstraintEvaluator ────────────────────────────────────────────────

/// Batch constraint evaluator for efficient evaluation of multiple constraints.
#[derive(Debug)]
pub struct BatchConstraintEvaluator {
    pub(crate) cache: ConstraintCache,
    parallel_evaluation: bool,
    batch_size: usize,
}

impl Default for BatchConstraintEvaluator {
    fn default() -> Self {
        Self::new(ConstraintCache::default(), false, 100)
    }
}

impl BatchConstraintEvaluator {
    /// Create a new batch evaluator.
    pub fn new(cache: ConstraintCache, parallel_evaluation: bool, batch_size: usize) -> Self {
        Self {
            cache,
            parallel_evaluation,
            batch_size,
        }
    }

    /// Evaluate multiple constraints in batches.
    pub fn evaluate_batch(
        &self,
        store: &dyn Store,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let mut results = Vec::with_capacity(constraints_with_contexts.len());

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

    fn evaluate_batch_sequential(
        &self,
        store: &dyn Store,
        batch: &[(Constraint, ConstraintContext)],
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let mut results = Vec::new();

        for (constraint, context) in batch {
            if let Some(cached_result) = self.cache.get(constraint, context) {
                results.push(cached_result);
                continue;
            }

            let start_time = Instant::now();
            let result = constraint.evaluate(store, context)?;
            let evaluation_time = start_time.elapsed();

            self.cache
                .put(constraint, context, result.clone(), evaluation_time);
            results.push(result);
        }

        Ok(results)
    }

    fn evaluate_batch_parallel(
        &self,
        store: &dyn Store,
        batch: &[(Constraint, ConstraintContext)],
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        if batch.len() < 4 {
            return self.evaluate_batch_sequential(store, batch);
        }
        // Fall back to sequential to avoid Rc<> threading issues
        self.evaluate_batch_sequential(store, batch)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

// ─── OptimizationConfig ──────────────────────────────────────────────────────

/// Optimization configuration for the validation engine
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_caching: bool,
    pub enable_parallel: bool,
    pub batch_size: usize,
    pub enable_reordering: bool,
    pub max_cache_size: usize,
    pub cache_ttl_secs: u64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_parallel: false,
            batch_size: 100,
            enable_reordering: true,
            max_cache_size: 10000,
            cache_ttl_secs: 300,
        }
    }
}

// ─── OptimizationMetrics ─────────────────────────────────────────────────────

/// Optimization metrics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct OptimizationMetrics {
    pub total_evaluations: usize,
    pub cache_hit_rate: f64,
    pub avg_evaluation_time_us: f64,
    pub constraints_reordered: usize,
    pub optimization_time_saved_us: f64,
    pub expensive_constraints: HashMap<String, f64>,
}

// ─── ValidationOptimizationEngine ────────────────────────────────────────────

/// Advanced validation optimization engine
#[derive(Debug)]
pub struct ValidationOptimizationEngine {
    cache: ConstraintCache,
    dependency_analyzer: ConstraintDependencyAnalyzer,
    batch_evaluator: BatchConstraintEvaluator,
    metrics: Arc<RwLock<OptimizationMetrics>>,
    config: OptimizationConfig,
}

impl ValidationOptimizationEngine {
    /// Create a new optimization engine with the given configuration.
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

    /// Optimize and evaluate a set of constraints, updating internal metrics.
    pub fn optimize_and_evaluate(
        &mut self,
        store: &dyn Store,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let start_time = Instant::now();

        let optimized_constraints = if self.config.enable_reordering {
            self.reorder_constraints_for_optimization(constraints_with_contexts)
        } else {
            constraints_with_contexts
        };

        let results = if self.config.enable_caching {
            self.batch_evaluator
                .evaluate_batch(store, optimized_constraints)?
        } else {
            self.evaluate_without_cache(store, optimized_constraints)?
        };

        let total_time = start_time.elapsed();
        self.update_metrics(results.len(), total_time);

        Ok(results)
    }

    fn reorder_constraints_for_optimization(
        &mut self,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Vec<(Constraint, ConstraintContext)> {
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
            group.sort_by(|(a, _), (b, _)| {
                let cost_a = self.dependency_analyzer.estimate_constraint_cost(a);
                let cost_b = self.dependency_analyzer.estimate_constraint_cost(b);
                let selectivity_a = self.dependency_analyzer.estimate_constraint_selectivity(a);
                let selectivity_b = self.dependency_analyzer.estimate_constraint_selectivity(b);

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

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.constraints_reordered += optimized.len();
        }

        optimized
    }

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

    fn update_metrics(&self, evaluation_count: usize, total_time: Duration) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_evaluations += evaluation_count;

            let cache_stats = self.cache.stats();
            metrics.cache_hit_rate = cache_stats.hit_rate();
            metrics.avg_evaluation_time_us = cache_stats.avg_evaluation_time_us;

            let time_per_evaluation = if evaluation_count > 0 {
                total_time.as_micros() as f64 / evaluation_count as f64
            } else {
                0.0
            };
            metrics.optimization_time_saved_us +=
                cache_stats.hits as f64 * time_per_evaluation * 0.8;
        }
    }

    /// Get current optimization metrics.
    pub fn get_metrics(&self) -> OptimizationMetrics {
        self.metrics
            .read()
            .expect("metrics lock should not be poisoned")
            .clone()
    }

    /// Clear all caches and reset metrics.
    pub fn reset(&mut self) {
        self.cache.clear();
        if let Ok(mut metrics) = self.metrics.write() {
            *metrics = OptimizationMetrics::default();
        }
    }

    /// Update configuration (some changes may require recreating components).
    pub fn update_config(&mut self, config: OptimizationConfig) {
        self.config = config;
    }

    /// Get cache statistics.
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}
