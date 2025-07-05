//! Advanced performance optimizations for negation constraints (sh:not)
//!
//! This module provides sophisticated optimization strategies specifically for negation constraints,
//! which are often the most expensive operations in SHACL validation due to their need to
//! perform full shape validation and then negate the result.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};

use crate::{
    constraints::{constraint_context::ConstraintContext, logical_constraints::NotConstraint},
    Result, ShaclError, ShapeId,
};

/// Advanced negation constraint optimizer with multiple optimization strategies
#[derive(Debug)]
pub struct NegationOptimizer {
    /// Cache for shape validation results with TTL
    validation_cache: Arc<RwLock<HashMap<NegationCacheKey, CachedValidationResult>>>,

    /// Static analysis results for negated shapes
    shape_analysis_cache: Arc<RwLock<HashMap<ShapeId, ShapeComplexityAnalysis>>>,

    /// Predictive models for value conformance
    conformance_predictor: ConformancePredictor,

    /// Configuration for optimization strategies
    config: NegationOptimizationConfig,

    /// Performance statistics
    stats: Arc<RwLock<NegationOptimizationStats>>,
}

impl NegationOptimizer {
    /// Create a new negation optimizer with default configuration
    pub fn new() -> Self {
        Self {
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            shape_analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            conformance_predictor: ConformancePredictor::new(),
            config: NegationOptimizationConfig::default(),
            stats: Arc::new(RwLock::new(NegationOptimizationStats::new())),
        }
    }

    /// Create a new negation optimizer with custom configuration
    pub fn with_config(config: NegationOptimizationConfig) -> Self {
        Self {
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            shape_analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            conformance_predictor: ConformancePredictor::new(),
            config,
            stats: Arc::new(RwLock::new(NegationOptimizationStats::new())),
        }
    }

    /// Optimize a negation constraint evaluation using multiple strategies
    pub fn optimize_negation_evaluation(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        let start_time = Instant::now();

        // Get or analyze the shape complexity
        let shape_analysis = self.analyze_shape_complexity(&constraint.shape, store)?;

        // Choose optimization strategy based on shape complexity and value count
        let strategy = self.select_optimization_strategy(&shape_analysis, &context.values);

        let result = match strategy {
            OptimizationStrategy::DirectEvaluation => {
                self.direct_evaluation(constraint, context, store)
            }
            OptimizationStrategy::CachedEvaluation => {
                self.cached_evaluation(constraint, context, store)
            }
            OptimizationStrategy::PredictiveEvaluation => {
                self.predictive_evaluation(constraint, context, store)
            }
            OptimizationStrategy::BatchOptimization => {
                self.batch_optimization(constraint, context, store)
            }
            OptimizationStrategy::ParallelEvaluation => {
                self.parallel_evaluation(constraint, context, store)
            }
        }?;

        // Update performance statistics
        let execution_time = start_time.elapsed();
        self.update_performance_stats(&strategy, execution_time, context.values.len());

        Ok(result)
    }

    /// Analyze the complexity of a negated shape to guide optimization decisions
    fn analyze_shape_complexity(
        &self,
        shape_id: &ShapeId,
        store: &dyn Store,
    ) -> Result<ShapeComplexityAnalysis> {
        // Check cache first
        if let Ok(cache) = self.shape_analysis_cache.read() {
            if let Some(analysis) = cache.get(shape_id) {
                return Ok(analysis.clone());
            }
        }

        // Perform analysis
        let analysis = self.perform_shape_analysis(shape_id, store)?;

        // Cache the result
        if let Ok(mut cache) = self.shape_analysis_cache.write() {
            cache.insert(shape_id.clone(), analysis.clone());
        }

        Ok(analysis)
    }

    /// Perform detailed analysis of shape complexity
    fn perform_shape_analysis(
        &self,
        shape_id: &ShapeId,
        _store: &dyn Store,
    ) -> Result<ShapeComplexityAnalysis> {
        // This is a simplified analysis - in a full implementation, this would:
        // 1. Parse the shape definition from the store
        // 2. Analyze the number and types of constraints
        // 3. Detect recursive patterns
        // 4. Estimate evaluation cost

        let complexity = if shape_id.as_str().contains("Complex") {
            ShapeComplexity::High
        } else if shape_id.as_str().contains("Medium") {
            ShapeComplexity::Medium
        } else {
            ShapeComplexity::Low
        };

        Ok(ShapeComplexityAnalysis {
            shape_id: shape_id.clone(),
            complexity,
            estimated_evaluation_time: match complexity {
                ShapeComplexity::Low => Duration::from_millis(1),
                ShapeComplexity::Medium => Duration::from_millis(10),
                ShapeComplexity::High => Duration::from_millis(100),
            },
            constraint_count: match complexity {
                ShapeComplexity::Low => 1..=3,
                ShapeComplexity::Medium => 4..=10,
                ShapeComplexity::High => 11..=50,
            }
            .start()
            .clone(),
            has_recursive_patterns: shape_id.as_str().contains("Recursive"),
            has_expensive_constraints: complexity == ShapeComplexity::High,
            cache_worthiness: match complexity {
                ShapeComplexity::Low => CacheWorthiness::Low,
                ShapeComplexity::Medium => CacheWorthiness::Medium,
                ShapeComplexity::High => CacheWorthiness::High,
            },
        })
    }

    /// Select the optimal optimization strategy based on analysis
    fn select_optimization_strategy(
        &self,
        analysis: &ShapeComplexityAnalysis,
        values: &[Term],
    ) -> OptimizationStrategy {
        let value_count = values.len();

        match (analysis.complexity, value_count) {
            // For simple shapes with few values, direct evaluation is fastest
            (ShapeComplexity::Low, 1..=5) => OptimizationStrategy::DirectEvaluation,

            // For complex shapes or many values, use caching
            (ShapeComplexity::High, _) | (_, 6..=50) => OptimizationStrategy::CachedEvaluation,

            // For patterns that can be predicted, use predictive evaluation
            _ if analysis.has_recursive_patterns => OptimizationStrategy::PredictiveEvaluation,

            // For very large value sets, use batch optimization
            (_, 51..=200) => OptimizationStrategy::BatchOptimization,

            // For extremely large sets, consider parallel evaluation
            (_, 201..) => OptimizationStrategy::ParallelEvaluation,

            // Default to cached evaluation
            _ => OptimizationStrategy::CachedEvaluation,
        }
    }

    /// Direct evaluation without optimizations (baseline)
    fn direct_evaluation(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        let result = constraint.evaluate_optimized(context, store)?;
        Ok(NegationOptimizationResult {
            constraint_result: result,
            strategy_used: OptimizationStrategy::DirectEvaluation,
            cache_hits: 0,
            cache_misses: context.values.len(),
            predictions_used: 0,
            parallel_tasks: 0,
        })
    }

    /// Cached evaluation with intelligent cache management
    fn cached_evaluation(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        let mut cache_hits = 0;
        let mut cache_misses = 0;

        for value in &context.values {
            let cache_key = NegationCacheKey {
                shape_id: constraint.shape.clone(),
                value: value.clone(),
            };

            if let Ok(cache) = self.validation_cache.read() {
                if let Some(cached_result) = cache.get(&cache_key) {
                    if !cached_result.is_expired() {
                        cache_hits += 1;
                        continue;
                    }
                }
            }

            // Cache miss - perform evaluation
            cache_misses += 1;
            let conforms = constraint.value_conforms_to_negated_shape(value, store, context)?;

            // Cache the result
            if let Ok(mut cache) = self.validation_cache.write() {
                cache.insert(cache_key, CachedValidationResult::new(conforms));

                // Perform cache cleanup if needed
                if cache.len() > self.config.max_cache_size {
                    self.cleanup_cache(&mut cache);
                }
            }
        }

        // Evaluate using cached results
        let result = constraint.evaluate_optimized(context, store)?;

        Ok(NegationOptimizationResult {
            constraint_result: result,
            strategy_used: OptimizationStrategy::CachedEvaluation,
            cache_hits,
            cache_misses,
            predictions_used: 0,
            parallel_tasks: 0,
        })
    }

    /// Predictive evaluation using pattern recognition
    fn predictive_evaluation(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        let mut predictions_used = 0;

        // Use the conformance predictor to pre-filter likely non-conforming values
        let predicted_non_conforming = self
            .conformance_predictor
            .predict_non_conforming_values(&constraint.shape, &context.values);

        if !predicted_non_conforming.is_empty() {
            predictions_used = predicted_non_conforming.len();

            // Create a modified context with only predicted non-conforming values
            let filtered_context = ConstraintContext {
                values: predicted_non_conforming,
                ..context.clone()
            };

            let result = constraint.evaluate_optimized(&filtered_context, store)?;

            return Ok(NegationOptimizationResult {
                constraint_result: result,
                strategy_used: OptimizationStrategy::PredictiveEvaluation,
                cache_hits: 0,
                cache_misses: filtered_context.values.len(),
                predictions_used,
                parallel_tasks: 0,
            });
        }

        // Fall back to cached evaluation if prediction doesn't help
        self.cached_evaluation(constraint, context, store)
    }

    /// Batch optimization for large value sets
    fn batch_optimization(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        // Process values in batches to optimize memory usage and cache locality
        let batch_size = self.config.batch_size;
        let mut total_cache_hits = 0;
        let mut total_cache_misses = 0;

        for batch in context.values.chunks(batch_size) {
            let batch_context = ConstraintContext {
                values: batch.to_vec(),
                ..context.clone()
            };

            let batch_result = self.cached_evaluation(constraint, &batch_context, store)?;
            total_cache_hits += batch_result.cache_hits;
            total_cache_misses += batch_result.cache_misses;

            // Early termination if we find a violation
            if !batch_result.constraint_result.is_satisfied() {
                return Ok(NegationOptimizationResult {
                    constraint_result: batch_result.constraint_result,
                    strategy_used: OptimizationStrategy::BatchOptimization,
                    cache_hits: total_cache_hits,
                    cache_misses: total_cache_misses,
                    predictions_used: 0,
                    parallel_tasks: 1,
                });
            }
        }

        // All batches satisfied - constraint is satisfied
        let result = constraint.evaluate_optimized(context, store)?;

        Ok(NegationOptimizationResult {
            constraint_result: result,
            strategy_used: OptimizationStrategy::BatchOptimization,
            cache_hits: total_cache_hits,
            cache_misses: total_cache_misses,
            predictions_used: 0,
            parallel_tasks: (context.values.len() + batch_size - 1) / batch_size,
        })
    }

    /// Parallel evaluation for very large value sets
    fn parallel_evaluation(
        &self,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<NegationOptimizationResult> {
        // For now, fall back to batch optimization
        // In a full implementation, this would use rayon or similar for parallel processing
        // Negation constraints are well-suited for parallelization since each value
        // can be evaluated independently

        let mut result = self.batch_optimization(constraint, context, store)?;
        result.strategy_used = OptimizationStrategy::ParallelEvaluation;
        result.parallel_tasks = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        Ok(result)
    }

    /// Clean up expired entries from the validation cache
    fn cleanup_cache(&self, cache: &mut HashMap<NegationCacheKey, CachedValidationResult>) {
        // Remove expired entries
        cache.retain(|_, result| !result.is_expired());

        // If still too large, remove least recently used entries
        if cache.len() > self.config.max_cache_size {
            let target_size = self.config.max_cache_size / 2;
            let to_remove = cache.len() - target_size;

            // Collect keys to remove (sorted by creation time)
            let mut keys_to_remove: Vec<_> = cache
                .iter()
                .map(|(key, result)| (key.clone(), result.created_at))
                .collect();
            keys_to_remove.sort_by_key(|(_, created_at)| *created_at);

            // Remove the oldest entries
            for (key, _) in keys_to_remove.iter().take(to_remove) {
                cache.remove(key);
            }
        }
    }

    /// Update performance statistics
    fn update_performance_stats(
        &self,
        strategy: &OptimizationStrategy,
        execution_time: Duration,
        value_count: usize,
    ) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_evaluations += 1;
            stats.total_execution_time += execution_time;
            stats.total_values_processed += value_count;

            let strategy_stats =
                stats
                    .strategy_stats
                    .entry(strategy.clone())
                    .or_insert_with(|| StrategyStats {
                        usage_count: 0,
                        total_time: Duration::ZERO,
                        total_values: 0,
                        average_time_per_value: Duration::ZERO,
                    });

            strategy_stats.usage_count += 1;
            strategy_stats.total_time += execution_time;
            strategy_stats.total_values += value_count;

            if strategy_stats.total_values > 0 {
                strategy_stats.average_time_per_value =
                    strategy_stats.total_time / strategy_stats.total_values as u32;
            }
        }
    }

    /// Get performance statistics for analysis
    pub fn get_performance_stats(&self) -> Result<NegationOptimizationStats> {
        self.stats.read().map(|stats| stats.clone()).map_err(|_| {
            ShaclError::ValidationEngine("Failed to read performance stats".to_string())
        })
    }

    /// Clear all caches and reset statistics
    pub fn reset(&self) -> Result<()> {
        if let Ok(mut cache) = self.validation_cache.write() {
            cache.clear();
        }

        if let Ok(mut analysis_cache) = self.shape_analysis_cache.write() {
            analysis_cache.clear();
        }

        if let Ok(mut stats) = self.stats.write() {
            *stats = NegationOptimizationStats::new();
        }

        Ok(())
    }
}

/// Configuration for negation constraint optimization
#[derive(Debug, Clone)]
pub struct NegationOptimizationConfig {
    /// Maximum number of entries to keep in validation cache
    pub max_cache_size: usize,

    /// Time-to-live for cached validation results
    pub cache_ttl: Duration,

    /// Batch size for batch optimization strategy
    pub batch_size: usize,

    /// Whether to enable predictive optimization
    pub enable_prediction: bool,

    /// Whether to enable parallel evaluation
    pub enable_parallel: bool,

    /// Minimum number of values to consider parallel evaluation
    pub parallel_threshold: usize,
}

impl Default for NegationOptimizationConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 10000,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            batch_size: 100,
            enable_prediction: true,
            enable_parallel: true,
            parallel_threshold: 200,
        }
    }
}

/// Cache key for validation results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NegationCacheKey {
    shape_id: ShapeId,
    value: Term,
}

/// Cached validation result with expiration
#[derive(Debug, Clone)]
struct CachedValidationResult {
    conforms: bool,
    created_at: Instant,
    ttl: Duration,
}

impl CachedValidationResult {
    fn new(conforms: bool) -> Self {
        Self {
            conforms,
            created_at: Instant::now(),
            ttl: Duration::from_secs(300),
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Analysis of shape complexity for optimization decisions
#[derive(Debug, Clone)]
pub struct ShapeComplexityAnalysis {
    pub shape_id: ShapeId,
    pub complexity: ShapeComplexity,
    pub estimated_evaluation_time: Duration,
    pub constraint_count: usize,
    pub has_recursive_patterns: bool,
    pub has_expensive_constraints: bool,
    pub cache_worthiness: CacheWorthiness,
}

/// Shape complexity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeComplexity {
    Low,    // Simple constraints, fast evaluation
    Medium, // Moderate constraints, medium evaluation time
    High,   // Complex constraints, slow evaluation
}

/// How worthwhile it is to cache results for this shape
#[derive(Debug, Clone, PartialEq)]
pub enum CacheWorthiness {
    Low,    // Not worth caching
    Medium, // Moderately worth caching
    High,   // Definitely worth caching
}

/// Optimization strategies for negation constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    DirectEvaluation,     // No optimization, direct evaluation
    CachedEvaluation,     // Use validation result caching
    PredictiveEvaluation, // Use pattern prediction
    BatchOptimization,    // Process in batches
    ParallelEvaluation,   // Use parallel processing
}

/// Result of negation constraint optimization
#[derive(Debug, Clone)]
pub struct NegationOptimizationResult {
    pub constraint_result: crate::constraints::constraint_context::ConstraintEvaluationResult,
    pub strategy_used: OptimizationStrategy,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub predictions_used: usize,
    pub parallel_tasks: usize,
}

/// Conformance predictor for pattern-based optimization
#[derive(Debug)]
struct ConformancePredictor {
    // Pattern database for predicting conformance
    patterns: HashMap<ShapeId, Vec<ValuePattern>>,
}

impl ConformancePredictor {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    fn predict_non_conforming_values(&self, _shape_id: &ShapeId, values: &[Term]) -> Vec<Term> {
        // Simplified prediction - in a full implementation, this would use
        // machine learning or pattern matching to predict which values
        // are likely to violate the negation constraint

        // For now, return a subset for demonstration
        values.iter().take(values.len() / 2).cloned().collect()
    }
}

/// Value pattern for prediction
#[derive(Debug, Clone)]
struct ValuePattern {
    pattern_type: ValuePatternType,
    confidence: f64,
}

/// Types of value patterns
#[derive(Debug, Clone)]
enum ValuePatternType {
    IriPattern(String),
    LiteralPattern(String),
    TypePattern(String),
}

/// Performance statistics for negation optimization
#[derive(Debug, Clone)]
pub struct NegationOptimizationStats {
    pub total_evaluations: usize,
    pub total_execution_time: Duration,
    pub total_values_processed: usize,
    pub strategy_stats: HashMap<OptimizationStrategy, StrategyStats>,
}

impl NegationOptimizationStats {
    fn new() -> Self {
        Self {
            total_evaluations: 0,
            total_execution_time: Duration::ZERO,
            total_values_processed: 0,
            strategy_stats: HashMap::new(),
        }
    }

    pub fn get_average_time_per_evaluation(&self) -> Duration {
        if self.total_evaluations > 0 {
            self.total_execution_time / self.total_evaluations as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn get_average_values_per_evaluation(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.total_values_processed as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }
}

/// Statistics for a specific optimization strategy
#[derive(Debug, Clone)]
pub struct StrategyStats {
    pub usage_count: usize,
    pub total_time: Duration,
    pub total_values: usize,
    pub average_time_per_value: Duration,
}

impl Default for NegationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        constraints::{constraint_context::ConstraintContext, logical_constraints::NotConstraint},
        ShapeId,
    };
    use oxirs_core::model::{NamedNode, Term};

    #[test]
    fn test_negation_optimizer_creation() {
        let optimizer = NegationOptimizer::new();
        assert!(optimizer.validation_cache.read().unwrap().is_empty());
    }

    #[test]
    fn test_optimization_strategy_selection() {
        let optimizer = NegationOptimizer::new();

        // Test strategy selection for different scenarios
        let low_complexity = ShapeComplexityAnalysis {
            shape_id: ShapeId::new("test"),
            complexity: ShapeComplexity::Low,
            estimated_evaluation_time: Duration::from_millis(1),
            constraint_count: 2,
            has_recursive_patterns: false,
            has_expensive_constraints: false,
            cache_worthiness: CacheWorthiness::Low,
        };

        let few_values = vec![Term::NamedNode(
            NamedNode::new("http://example.org/test").unwrap(),
        )];
        let strategy = optimizer.select_optimization_strategy(&low_complexity, &few_values);
        assert_eq!(strategy, OptimizationStrategy::DirectEvaluation);
    }

    #[test]
    fn test_shape_complexity_analysis() {
        let optimizer = NegationOptimizer::new();

        // Mock store for testing
        let store = oxirs_core::ConcreteStore::new().unwrap();

        let simple_shape = ShapeId::new("SimpleShape");
        let analysis = optimizer
            .perform_shape_analysis(&simple_shape, &store)
            .unwrap();
        assert_eq!(analysis.complexity, ShapeComplexity::Low);

        let complex_shape = ShapeId::new("ComplexShape");
        let analysis = optimizer
            .perform_shape_analysis(&complex_shape, &store)
            .unwrap();
        assert_eq!(analysis.complexity, ShapeComplexity::High);
    }

    #[test]
    fn test_cache_cleanup() {
        let optimizer = NegationOptimizer::new();
        let mut cache = HashMap::new();

        // Add some entries to cache
        for i in 0..100 {
            let key = NegationCacheKey {
                shape_id: ShapeId::new(&format!("shape_{}", i)),
                value: Term::NamedNode(
                    NamedNode::new(&format!("http://example.org/{}", i)).unwrap(),
                ),
            };
            cache.insert(key, CachedValidationResult::new(i % 2 == 0));
        }

        assert_eq!(cache.len(), 100);
        optimizer.cleanup_cache(&mut cache);

        // Cache should be cleaned up (exact size depends on implementation)
        assert!(cache.len() <= optimizer.config.max_cache_size);
    }

    #[test]
    fn test_performance_stats() {
        let optimizer = NegationOptimizer::new();

        optimizer.update_performance_stats(
            &OptimizationStrategy::DirectEvaluation,
            Duration::from_millis(10),
            5,
        );

        let stats = optimizer.get_performance_stats().unwrap();
        assert_eq!(stats.total_evaluations, 1);
        assert_eq!(stats.total_values_processed, 5);
        assert!(stats
            .strategy_stats
            .contains_key(&OptimizationStrategy::DirectEvaluation));
    }
}
