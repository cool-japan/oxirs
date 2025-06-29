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
        // Note: In practice, Store would need to be thread-safe for this to work
        // For now, we'll fall back to sequential evaluation
        // TODO: Implement proper parallel evaluation when Store is thread-safe
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
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                selectivity_cmp
            }
        });

        constraint_info.into_iter().map(|(c, _, _)| c).collect()
    }

    /// Estimate the computational cost of evaluating a constraint
    fn estimate_constraint_cost(&self, constraint: &Constraint) -> f64 {
        let constraint_type = match constraint {
            Constraint::Class(_) => "class",
            Constraint::Datatype(_) => "datatype",
            Constraint::NodeKind(_) => "nodeKind",
            Constraint::MinCount(_) => "minCount",
            Constraint::MaxCount(_) => "maxCount",
            Constraint::Pattern(_) => "pattern",
            Constraint::Sparql(_) => "sparql",
            Constraint::QualifiedValueShape(_) => "qualifiedValueShape",
            Constraint::Closed(_) => "closed",
            _ => "default",
        };

        self.cost_estimates
            .get(constraint_type)
            .copied()
            .unwrap_or(2.0)
    }

    /// Estimate the selectivity of a constraint (probability it will fail)
    /// Lower values indicate more selective constraints (should be evaluated first)
    fn estimate_constraint_selectivity(&self, constraint: &Constraint) -> f64 {
        match constraint {
            // These constraints are very selective (low probability of failing but cheap to check)
            Constraint::MinCount(_) => 0.1,
            Constraint::MaxCount(_) => 0.1,
            Constraint::NodeKind(_) => 0.2,
            Constraint::Datatype(_) => 0.3,

            // These are moderately selective
            Constraint::Pattern(_) => 0.4,
            Constraint::Class(_) => 0.5,

            // Complex constraints are less selective but expensive (evaluate later)
            Constraint::Sparql(_) => 0.8,
            Constraint::QualifiedValueShape(_) => 0.7,
            Constraint::Closed(_) => 0.6,

            _ => 0.5, // Default selectivity
        }
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

        // Cache a result
        let result = ConstraintEvaluationResult::satisfied();
        cache.put(
            &constraint,
            &context,
            result.clone(),
            Duration::from_millis(5),
        );

        // Should be cache hit now
        let cached_result = cache.get(&constraint, &context);
        assert!(cached_result.is_some());
        assert!(cached_result.unwrap().is_satisfied());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_dependency_analyzer() {
        let analyzer = ConstraintDependencyAnalyzer::default();

        let constraints = vec![
            Constraint::Sparql(crate::sparql::SparqlConstraint::ask(
                "ASK { ?s ?p ?o }".to_string(),
            )),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
            }),
        ];

        let optimized = analyzer.optimize_constraint_order(constraints);

        // MinCount should come first (most selective), then Datatype, then SPARQL
        assert!(matches!(optimized[0], Constraint::MinCount(_)));
        assert!(matches!(optimized[1], Constraint::Datatype(_)));
        assert!(matches!(optimized[2], Constraint::Sparql(_)));
    }

    #[test]
    fn test_cache_key_generation() {
        let cache = ConstraintCache::default();

        let constraint1 = Constraint::Class(ClassConstraint {
            class_iri: NamedNode::new("http://example.org/Person").unwrap(),
        });

        let constraint2 = Constraint::Class(ClassConstraint {
            class_iri: NamedNode::new("http://example.org/Organization").unwrap(),
        });

        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
            ShapeId::new("PersonShape"),
        );

        let key1 = cache.create_cache_key(&constraint1, &context);
        let key2 = cache.create_cache_key(&constraint2, &context);

        // Different constraints should have different cache keys
        assert_ne!(key1, key2);
    }
}
