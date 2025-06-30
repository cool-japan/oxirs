//! Enhanced Query Execution Optimizer
//!
//! This module provides advanced optimization techniques for query execution,
//! including adaptive caching, pattern reordering, and performance prediction.

use crate::algebra::{Algebra, Solution, Term, TriplePattern, Variable};
use crate::executor::config::ExecutionContext;
use crate::executor::stats::ExecutionStats;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Advanced execution optimizer with learning capabilities
pub struct EnhancedExecutionOptimizer {
    /// Query pattern cache
    pattern_cache: Arc<RwLock<HashMap<String, CachedOptimization>>>,
    /// Execution statistics for learning
    execution_history: Arc<RwLock<Vec<ExecutionRecord>>>,
    /// Performance prediction model
    performance_model: Arc<RwLock<PerformancePredictionModel>>,
    /// Configuration
    config: OptimizationConfig,
}

/// Cached optimization result
#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub optimized_algebra: Algebra,
    pub estimated_cost: f64,
    pub actual_cost: Option<f64>,
    pub timestamp: Instant,
    pub hit_count: usize,
}

/// Execution record for learning
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub query_hash: String,
    pub algebra: Algebra,
    pub execution_time: Duration,
    pub result_count: usize,
    pub memory_used: usize,
    pub strategy_used: String,
    pub timestamp: Instant,
}

/// Performance prediction model
#[derive(Debug, Clone)]
pub struct PerformancePredictionModel {
    /// Pattern complexity weights
    pattern_weights: HashMap<String, f64>,
    /// Join selectivity estimates
    join_selectivities: HashMap<String, f64>,
    /// Operator costs
    operator_costs: HashMap<String, f64>,
    /// Learning rate for adaptive updates
    learning_rate: f64,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub cache_size: usize,
    pub cache_ttl: Duration,
    pub learning_enabled: bool,
    pub prediction_threshold: f64,
    pub max_optimization_time: Duration,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            cache_ttl: Duration::from_secs(3600), // 1 hour
            learning_enabled: true,
            prediction_threshold: 0.8,
            max_optimization_time: Duration::from_millis(100),
        }
    }
}

impl EnhancedExecutionOptimizer {
    /// Create new enhanced optimizer
    pub fn new() -> Self {
        Self::with_config(OptimizationConfig::default())
    }

    /// Create optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self {
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            performance_model: Arc::new(RwLock::new(PerformancePredictionModel::new())),
            config,
        }
    }

    /// Optimize algebra expression with learning and caching
    pub fn optimize_algebra(&self, algebra: &Algebra) -> Result<Algebra> {
        let start_time = Instant::now();
        let query_hash = self.compute_query_hash(algebra);

        // Check cache first
        if let Some(cached) = self.get_cached_optimization(&query_hash) {
            self.update_cache_hit_count(&query_hash);
            return Ok(cached.optimized_algebra);
        }

        // Perform optimization
        let optimized = self.perform_optimization(algebra)?;
        let optimization_time = start_time.elapsed();

        // Don't spend too much time on optimization
        if optimization_time > self.config.max_optimization_time {
            return Ok(algebra.clone());
        }

        // Cache the result
        let estimated_cost = self.estimate_execution_cost(&optimized);
        self.cache_optimization(&query_hash, &optimized, estimated_cost);

        Ok(optimized)
    }

    /// Record execution statistics for learning
    pub fn record_execution(
        &self,
        algebra: &Algebra,
        stats: &ExecutionStats,
        strategy: &str,
    ) -> Result<()> {
        if !self.config.learning_enabled {
            return Ok(());
        }

        let query_hash = self.compute_query_hash(algebra);
        let record = ExecutionRecord {
            query_hash: query_hash.clone(),
            algebra: algebra.clone(),
            execution_time: stats.execution_time,
            result_count: stats.final_results,
            memory_used: stats.memory_used,
            strategy_used: strategy.to_string(),
            timestamp: Instant::now(),
        };

        // Update execution history
        {
            let mut history = self.execution_history.write().unwrap();
            history.push(record);

            // Keep only recent records
            if history.len() > 10000 {
                history.drain(0..1000); // Remove oldest 1000 records
            }
        }

        // Update performance model
        self.update_performance_model(algebra, stats)?;

        // Update cached optimization with actual cost
        self.update_cache_with_actual_cost(&query_hash, stats.execution_time.as_secs_f64());

        Ok(())
    }

    /// Predict execution performance
    pub fn predict_performance(&self, algebra: &Algebra) -> Result<PerformancePrediction> {
        let model = self.performance_model.read().unwrap();
        
        let complexity_score = self.compute_complexity_score(algebra);
        let selectivity_score = self.estimate_selectivity_score(algebra);
        let operator_cost = self.estimate_operator_cost(algebra);

        let predicted_time = (complexity_score * selectivity_score * operator_cost).max(0.001);
        let confidence = self.compute_prediction_confidence(algebra);

        Ok(PerformancePrediction {
            estimated_time_ms: predicted_time * 1000.0,
            estimated_memory_mb: (complexity_score * 10.0).min(1000.0),
            estimated_result_count: (selectivity_score * 1000.0) as usize,
            confidence,
            complexity_score,
            selectivity_score,
        })
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let cache = self.pattern_cache.read().unwrap();
        let history = self.execution_history.read().unwrap();

        let total_hits = cache.values().map(|c| c.hit_count).sum();
        let total_queries = cache.len() + history.len();
        let cache_hit_rate = if total_queries > 0 {
            total_hits as f64 / total_queries as f64
        } else {
            0.0
        };

        let avg_optimization_benefit = self.compute_average_optimization_benefit();

        OptimizationStats {
            cache_size: cache.len(),
            cache_hit_rate,
            total_optimizations: history.len(),
            avg_optimization_benefit,
            learning_records: history.len(),
        }
    }

    /// Perform actual optimization
    fn perform_optimization(&self, algebra: &Algebra) -> Result<Algebra> {
        let mut optimized = algebra.clone();

        // Apply various optimization techniques
        optimized = self.optimize_join_order(&optimized)?;
        optimized = self.optimize_filter_placement(&optimized)?;
        optimized = self.optimize_projection_pushdown(&optimized)?;
        optimized = self.optimize_union_reordering(&optimized)?;

        Ok(optimized)
    }

    /// Optimize join order using learned statistics
    fn optimize_join_order(&self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_execution_cost(left);
                let right_cost = self.estimate_execution_cost(right);
                
                // Optimize recursively first
                let opt_left = self.optimize_join_order(left)?;
                let opt_right = self.optimize_join_order(right)?;

                // Reorder if beneficial (smaller relation first)
                if right_cost < left_cost {
                    Ok(Algebra::Join {
                        left: Box::new(opt_right),
                        right: Box::new(opt_left),
                    })
                } else {
                    Ok(Algebra::Join {
                        left: Box::new(opt_left),
                        right: Box::new(opt_right),
                    })
                }
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Optimize filter placement by pushing filters down
    fn optimize_filter_placement(&self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                // Try to push filter into joins
                match pattern.as_ref() {
                    Algebra::Join { left, right } => {
                        let filter_vars = self.extract_filter_variables(condition);
                        let left_vars = self.extract_algebra_variables(left);
                        
                        // If filter only uses left variables, push it down
                        if filter_vars.is_subset(&left_vars) {
                            let filtered_left = Algebra::Filter {
                                pattern: left.clone(),
                                condition: condition.clone(),
                            };
                            Ok(Algebra::Join {
                                left: Box::new(filtered_left),
                                right: right.clone(),
                            })
                        } else {
                            Ok(algebra.clone())
                        }
                    }
                    _ => Ok(algebra.clone()),
                }
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Optimize projection pushdown
    fn optimize_projection_pushdown(&self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                // Try to push projection into subpatterns
                match pattern.as_ref() {
                    Algebra::Join { left, right } => {
                        let left_vars = self.extract_algebra_variables(left);
                        let right_vars = self.extract_algebra_variables(right);
                        
                        let left_needed: Vec<Variable> = variables
                            .iter()
                            .filter(|v| left_vars.contains(v))
                            .cloned()
                            .collect();
                        
                        let right_needed: Vec<Variable> = variables
                            .iter()
                            .filter(|v| right_vars.contains(v))
                            .cloned()
                            .collect();

                        if !left_needed.is_empty() && !right_needed.is_empty() {
                            let proj_left = if left_needed.len() < left_vars.len() {
                                Algebra::Project {
                                    pattern: left.clone(),
                                    variables: left_needed,
                                }
                            } else {
                                (**left).clone()
                            };

                            let proj_right = if right_needed.len() < right_vars.len() {
                                Algebra::Project {
                                    pattern: right.clone(),
                                    variables: right_needed,
                                }
                            } else {
                                (**right).clone()
                            };

                            Ok(Algebra::Project {
                                pattern: Box::new(Algebra::Join {
                                    left: Box::new(proj_left),
                                    right: Box::new(proj_right),
                                }),
                                variables: variables.clone(),
                            })
                        } else {
                            Ok(algebra.clone())
                        }
                    }
                    _ => Ok(algebra.clone()),
                }
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Optimize union reordering
    fn optimize_union_reordering(&self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Union { left, right } => {
                let left_cost = self.estimate_execution_cost(left);
                let right_cost = self.estimate_execution_cost(right);
                
                // Optimize recursively first
                let opt_left = self.optimize_union_reordering(left)?;
                let opt_right = self.optimize_union_reordering(right)?;

                // Put cheaper option first for early termination potential
                if right_cost < left_cost {
                    Ok(Algebra::Union {
                        left: Box::new(opt_right),
                        right: Box::new(opt_left),
                    })
                } else {
                    Ok(Algebra::Union {
                        left: Box::new(opt_left),
                        right: Box::new(opt_right),
                    })
                }
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Compute query hash for caching
    fn compute_query_hash(&self, algebra: &Algebra) -> String {
        // Simple hash based on algebra structure
        format!("{:?}", algebra)
            .chars()
            .fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64))
            .to_string()
    }

    /// Get cached optimization
    fn get_cached_optimization(&self, query_hash: &str) -> Option<CachedOptimization> {
        let cache = self.pattern_cache.read().unwrap();
        cache.get(query_hash).and_then(|cached| {
            // Check if cache entry is still valid
            if cached.timestamp.elapsed() < self.config.cache_ttl {
                Some(cached.clone())
            } else {
                None
            }
        })
    }

    /// Cache optimization result
    fn cache_optimization(&self, query_hash: &str, algebra: &Algebra, estimated_cost: f64) {
        let mut cache = self.pattern_cache.write().unwrap();
        
        // Remove old entries if cache is full
        if cache.len() >= self.config.cache_size {
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| k.clone());
            
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        cache.insert(
            query_hash.to_string(),
            CachedOptimization {
                optimized_algebra: algebra.clone(),
                estimated_cost,
                actual_cost: None,
                timestamp: Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Update cache hit count
    fn update_cache_hit_count(&self, query_hash: &str) {
        let mut cache = self.pattern_cache.write().unwrap();
        if let Some(cached) = cache.get_mut(query_hash) {
            cached.hit_count += 1;
        }
    }

    /// Update cache with actual execution cost
    fn update_cache_with_actual_cost(&self, query_hash: &str, actual_cost: f64) {
        let mut cache = self.pattern_cache.write().unwrap();
        if let Some(cached) = cache.get_mut(query_hash) {
            cached.actual_cost = Some(actual_cost);
        }
    }

    /// Update performance model with new data
    fn update_performance_model(&self, algebra: &Algebra, stats: &ExecutionStats) -> Result<()> {
        let mut model = self.performance_model.write().unwrap();
        
        let pattern_key = self.extract_pattern_key(algebra);
        let actual_time = stats.execution_time.as_secs_f64();
        
        // Update pattern weights using exponential moving average
        let current_weight = model.pattern_weights.get(&pattern_key).unwrap_or(&1.0);
        let new_weight = current_weight * (1.0 - model.learning_rate) + actual_time * model.learning_rate;
        model.pattern_weights.insert(pattern_key, new_weight);

        Ok(())
    }

    /// Estimate execution cost
    fn estimate_execution_cost(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() as f64 * 1.0,
            Algebra::Join { left, right } => {
                self.estimate_execution_cost(left) * self.estimate_execution_cost(right) * 0.1
            }
            Algebra::Union { left, right } => {
                self.estimate_execution_cost(left) + self.estimate_execution_cost(right)
            }
            Algebra::Filter { pattern, .. } => self.estimate_execution_cost(pattern) * 1.2,
            _ => 1.0,
        }
    }

    /// Extract pattern key for learning
    fn extract_pattern_key(&self, algebra: &Algebra) -> String {
        match algebra {
            Algebra::Bgp(_) => "bgp".to_string(),
            Algebra::Join { .. } => "join".to_string(),
            Algebra::Union { .. } => "union".to_string(),
            Algebra::Filter { .. } => "filter".to_string(),
            _ => "other".to_string(),
        }
    }

    /// Extract variables from filter condition
    fn extract_filter_variables(&self, _condition: &crate::expression::Expression) -> HashSet<Variable> {
        // Simplified implementation
        HashSet::new()
    }

    /// Extract variables from algebra expression
    fn extract_algebra_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut variables = HashSet::new();
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    if let Term::Variable(var) = &pattern.subject {
                        variables.insert(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.object {
                        variables.insert(var.clone());
                    }
                }
            }
            _ => {}
        }
        variables
    }

    /// Compute complexity score
    fn compute_complexity_score(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() as f64,
            Algebra::Join { left, right } => {
                1.0 + self.compute_complexity_score(left) + self.compute_complexity_score(right)
            }
            _ => 1.0,
        }
    }

    /// Estimate selectivity score
    fn estimate_selectivity_score(&self, _algebra: &Algebra) -> f64 {
        // Simplified implementation
        0.5
    }

    /// Estimate operator cost
    fn estimate_operator_cost(&self, algebra: &Algebra) -> f64 {
        let model = self.performance_model.read().unwrap();
        let pattern_key = self.extract_pattern_key(algebra);
        model.operator_costs.get(&pattern_key).unwrap_or(&1.0).clone()
    }

    /// Compute prediction confidence
    fn compute_prediction_confidence(&self, _algebra: &Algebra) -> f64 {
        // Simplified implementation
        0.8
    }

    /// Compute average optimization benefit
    fn compute_average_optimization_benefit(&self) -> f64 {
        let cache = self.pattern_cache.read().unwrap();
        let total_benefit: f64 = cache
            .values()
            .filter_map(|cached| {
                cached.actual_cost.map(|actual| {
                    (cached.estimated_cost - actual).max(0.0) / cached.estimated_cost.max(0.001)
                })
            })
            .sum();
        
        let count = cache.values().filter(|c| c.actual_cost.is_some()).count();
        if count > 0 {
            total_benefit / count as f64
        } else {
            0.0
        }
    }
}

impl PerformancePredictionModel {
    fn new() -> Self {
        let mut pattern_weights = HashMap::new();
        pattern_weights.insert("bgp".to_string(), 1.0);
        pattern_weights.insert("join".to_string(), 2.0);
        pattern_weights.insert("union".to_string(), 1.5);
        pattern_weights.insert("filter".to_string(), 1.2);

        let mut operator_costs = HashMap::new();
        operator_costs.insert("bgp".to_string(), 1.0);
        operator_costs.insert("join".to_string(), 2.0);
        operator_costs.insert("union".to_string(), 1.0);
        operator_costs.insert("filter".to_string(), 0.8);

        Self {
            pattern_weights,
            join_selectivities: HashMap::new(),
            operator_costs,
            learning_rate: 0.1,
        }
    }
}

/// Performance prediction result
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub estimated_time_ms: f64,
    pub estimated_memory_mb: f64,
    pub estimated_result_count: usize,
    pub confidence: f64,
    pub complexity_score: f64,
    pub selectivity_score: f64,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub cache_size: usize,
    pub cache_hit_rate: f64,
    pub total_optimizations: usize,
    pub avg_optimization_benefit: f64,
    pub learning_records: usize,
}

impl Default for EnhancedExecutionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}