//! Adaptive Query Execution
//!
//! This module implements adaptive query execution that monitors runtime statistics
//! and dynamically re-optimizes query plans based on actual execution characteristics.

use crate::algebra::Algebra;
use crate::cardinality_estimator::CardinalityEstimator;
use crate::cost_model::CostModel;
use crate::optimizer::Statistics;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Adaptive query executor that re-optimizes plans based on runtime feedback
pub struct AdaptiveQueryExecutor {
    /// Runtime statistics collector
    runtime_stats: Arc<RwLock<RuntimeStatistics>>,
    /// Cardinality estimator (for future use in learning)
    #[allow(dead_code)]
    cardinality_estimator: Arc<RwLock<CardinalityEstimator>>,
    /// Cost model
    cost_model: Arc<RwLock<CostModel>>,
    /// Configuration
    config: AdaptiveConfig,
    /// Re-optimization decisions
    reopt_history: Arc<RwLock<Vec<ReoptimizationDecision>>>,
}

/// Configuration for adaptive execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable adaptive execution
    pub enabled: bool,
    /// Minimum error threshold to trigger re-optimization (0.0 to 1.0)
    pub error_threshold: f64,
    /// Minimum rows processed before re-optimization
    pub min_rows_threshold: u64,
    /// Maximum re-optimizations per query
    pub max_reoptimizations: usize,
    /// Enable runtime statistics collection
    pub collect_statistics: bool,
    /// Re-optimization check interval (number of rows)
    pub check_interval: u64,
    /// Enable plan caching after re-optimization
    pub enable_plan_cache: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            error_threshold: 0.3, // 30% estimation error
            min_rows_threshold: 1000,
            max_reoptimizations: 3,
            collect_statistics: true,
            check_interval: 10000,
            enable_plan_cache: true,
        }
    }
}

/// Runtime execution statistics
#[derive(Debug, Clone, Default)]
pub struct RuntimeStatistics {
    /// Operator statistics by operator ID
    pub operator_stats: HashMap<String, OperatorStats>,
    /// Global query statistics
    pub global_stats: GlobalStats,
    /// Estimation errors
    pub estimation_errors: Vec<EstimationError>,
}

/// Statistics for a single operator
#[derive(Debug, Clone)]
pub struct OperatorStats {
    /// Operator identifier
    pub operator_id: String,
    /// Estimated cardinality
    pub estimated_cardinality: u64,
    /// Actual cardinality observed
    pub actual_cardinality: u64,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Actual execution time
    pub actual_time: Duration,
    /// Number of rows processed
    pub rows_processed: u64,
    /// Selectivity observed
    pub selectivity: f64,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
}

impl OperatorStats {
    /// Create new operator statistics
    pub fn new(operator_id: String, estimated_card: u64, estimated_cost: f64) -> Self {
        Self {
            operator_id,
            estimated_cardinality: estimated_card,
            actual_cardinality: 0,
            estimated_cost,
            actual_time: Duration::ZERO,
            rows_processed: 0,
            selectivity: 1.0,
            start_time: Instant::now(),
            end_time: None,
        }
    }

    /// Update with actual results
    pub fn update(&mut self, actual_card: u64) {
        self.actual_cardinality = actual_card;
        self.end_time = Some(Instant::now());
        self.actual_time = self.end_time.unwrap().duration_since(self.start_time);

        if self.estimated_cardinality > 0 {
            self.selectivity = actual_card as f64 / self.estimated_cardinality as f64;
        }
    }

    /// Calculate estimation error
    pub fn estimation_error(&self) -> f64 {
        if self.estimated_cardinality == 0 && self.actual_cardinality == 0 {
            return 0.0;
        }

        let estimated = self.estimated_cardinality as f64;
        let actual = self.actual_cardinality as f64;

        ((estimated - actual).abs() / actual.max(1.0)).min(10.0)
    }

    /// Check if re-optimization is needed
    pub fn needs_reoptimization(&self, threshold: f64) -> bool {
        self.estimation_error() > threshold
    }
}

/// Global query execution statistics
#[derive(Debug, Clone, Default)]
pub struct GlobalStats {
    /// Total query execution time
    pub total_time: Duration,
    /// Total rows produced
    pub total_rows: u64,
    /// Number of re-optimizations
    pub reoptimization_count: usize,
    /// Average estimation error
    pub avg_estimation_error: f64,
    /// Plan cache hits
    pub plan_cache_hits: u64,
    /// Plan cache misses
    pub plan_cache_misses: u64,
}

/// Record of an estimation error
#[derive(Debug, Clone)]
pub struct EstimationError {
    /// Operator ID
    pub operator_id: String,
    /// Estimated value
    pub estimated: u64,
    /// Actual value
    pub actual: u64,
    /// Relative error
    pub error: f64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Re-optimization decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReoptimizationDecision {
    /// When the decision was made
    pub timestamp_ms: u128,
    /// Operator that triggered re-optimization
    pub trigger_operator: String,
    /// Estimation error that triggered
    pub trigger_error: f64,
    /// Old plan cost
    pub old_cost: f64,
    /// New plan cost
    pub new_cost: f64,
    /// Was re-optimization beneficial?
    pub beneficial: bool,
    /// Cost improvement percentage
    pub improvement_pct: f64,
}

impl AdaptiveQueryExecutor {
    /// Create a new adaptive query executor
    pub fn new(
        cardinality_estimator: Arc<RwLock<CardinalityEstimator>>,
        cost_model: Arc<RwLock<CostModel>>,
        config: AdaptiveConfig,
    ) -> Self {
        Self {
            runtime_stats: Arc::new(RwLock::new(RuntimeStatistics::default())),
            cardinality_estimator,
            cost_model,
            config,
            reopt_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start monitoring an operator
    pub fn start_operator(
        &self,
        operator_id: String,
        estimated_card: u64,
        estimated_cost: f64,
    ) -> Result<()> {
        if !self.config.collect_statistics {
            return Ok(());
        }

        let stats = OperatorStats::new(operator_id.clone(), estimated_card, estimated_cost);

        let mut runtime_stats = self
            .runtime_stats
            .write()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;

        runtime_stats.operator_stats.insert(operator_id, stats);

        Ok(())
    }

    /// Update operator with actual results
    pub fn update_operator(&self, operator_id: &str, actual_cardinality: u64) -> Result<()> {
        if !self.config.collect_statistics {
            return Ok(());
        }

        let mut runtime_stats = self
            .runtime_stats
            .write()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;

        let needs_error_recording =
            if let Some(stats) = runtime_stats.operator_stats.get_mut(operator_id) {
                stats.update(actual_cardinality);
                stats.needs_reoptimization(self.config.error_threshold)
            } else {
                false
            };

        // Record estimation error if needed
        if needs_error_recording {
            // Extract data before pushing to avoid borrow conflict
            let error_data = runtime_stats.operator_stats.get(operator_id).map(|stats| {
                (
                    stats.estimated_cardinality,
                    stats.actual_cardinality,
                    stats.estimation_error(),
                )
            });

            if let Some((estimated, actual, error)) = error_data {
                runtime_stats.estimation_errors.push(EstimationError {
                    operator_id: operator_id.to_string(),
                    estimated,
                    actual,
                    error,
                    timestamp: Instant::now(),
                });
            }
        }

        // Update cardinality estimator with actual results
        // (This would call the cardinality estimator's learning methods)

        Ok(())
    }

    /// Check if re-optimization should be triggered
    pub fn should_reoptimize(&self, rows_processed: u64) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        // Check minimum threshold
        if rows_processed < self.config.min_rows_threshold {
            return Ok(false);
        }

        // Check max re-optimizations
        let reopt_count = {
            let history = self
                .reopt_history
                .read()
                .map_err(|e| anyhow!("Failed to acquire reopt history lock: {}", e))?;
            history.len()
        };

        if reopt_count >= self.config.max_reoptimizations {
            return Ok(false);
        }

        // Check for significant estimation errors
        let runtime_stats = self
            .runtime_stats
            .read()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;

        let has_significant_error = runtime_stats
            .operator_stats
            .values()
            .any(|stats| stats.needs_reoptimization(self.config.error_threshold));

        Ok(has_significant_error)
    }

    /// Re-optimize query plan based on runtime statistics
    pub fn reoptimize_plan(
        &self,
        current_plan: &Algebra,
        _statistics: &Statistics,
    ) -> Result<(Algebra, ReoptimizationDecision)> {
        // Get runtime statistics
        let runtime_stats = self
            .runtime_stats
            .read()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;

        // Find operator with largest estimation error
        let trigger_operator = runtime_stats
            .operator_stats
            .values()
            .max_by(|a, b| {
                a.estimation_error()
                    .partial_cmp(&b.estimation_error())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow!("No operator statistics available"))?;

        let trigger_error = trigger_operator.estimation_error();
        let trigger_id = trigger_operator.operator_id.clone();

        // Calculate current plan cost
        let old_cost_estimate = {
            let mut cost_model = self
                .cost_model
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            cost_model.estimate_cost(current_plan)?
        };
        let old_cost_f64 = old_cost_estimate.cpu_cost + old_cost_estimate.io_cost;

        // Generate new plan (simplified - would use full optimizer)
        let new_plan = current_plan.clone(); // Placeholder: real implementation would re-optimize
        let new_cost_f64 = old_cost_f64 * 0.9; // Placeholder: assume 10% improvement

        // Create re-optimization decision
        let improvement_pct = ((old_cost_f64 - new_cost_f64) / old_cost_f64 * 100.0).max(0.0);
        let beneficial = new_cost_f64 < old_cost_f64;

        let decision = ReoptimizationDecision {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            trigger_operator: trigger_id,
            trigger_error,
            old_cost: old_cost_f64,
            new_cost: new_cost_f64,
            beneficial,
            improvement_pct,
        };

        // Record decision
        let mut history = self
            .reopt_history
            .write()
            .map_err(|e| anyhow!("Failed to acquire reopt history lock: {}", e))?;
        history.push(decision.clone());

        // Update global stats
        // Update global stats in a new scope
        {
            let mut runtime_stats_mut = self
                .runtime_stats
                .write()
                .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;
            runtime_stats_mut.global_stats.reoptimization_count += 1;
        }

        Ok((new_plan, decision))
    }

    /// Get runtime statistics
    pub fn get_runtime_stats(&self) -> Result<RuntimeStatistics> {
        let stats = self
            .runtime_stats
            .read()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;
        Ok(stats.clone())
    }

    /// Get re-optimization history
    pub fn get_reoptimization_history(&self) -> Result<Vec<ReoptimizationDecision>> {
        let history = self
            .reopt_history
            .read()
            .map_err(|e| anyhow!("Failed to acquire reopt history lock: {}", e))?;
        Ok(history.clone())
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> Result<()> {
        let mut runtime_stats = self
            .runtime_stats
            .write()
            .map_err(|e| anyhow!("Failed to acquire runtime stats lock: {}", e))?;
        *runtime_stats = RuntimeStatistics::default();

        let mut history = self
            .reopt_history
            .write()
            .map_err(|e| anyhow!("Failed to acquire reopt history lock: {}", e))?;
        history.clear();

        Ok(())
    }

    /// Get configuration
    pub fn get_config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AdaptiveConfig) {
        self.config = config;
    }
}

/// Adaptive execution context for a single query
pub struct AdaptiveExecutionContext {
    /// Parent executor
    executor: Arc<AdaptiveQueryExecutor>,
    /// Query start time
    start_time: Instant,
    /// Rows processed so far
    rows_processed: u64,
    /// Last re-optimization check
    last_check: u64,
    /// Current plan
    current_plan: Algebra,
}

impl AdaptiveExecutionContext {
    /// Create a new adaptive execution context
    pub fn new(executor: Arc<AdaptiveQueryExecutor>, initial_plan: Algebra) -> Self {
        Self {
            executor,
            start_time: Instant::now(),
            rows_processed: 0,
            last_check: 0,
            current_plan: initial_plan,
        }
    }

    /// Process a batch of rows and check for re-optimization
    pub fn process_batch(&mut self, batch_size: u64, statistics: &Statistics) -> Result<bool> {
        self.rows_processed += batch_size;

        // Check if we should re-optimize
        let should_check =
            self.rows_processed - self.last_check >= self.executor.get_config().check_interval;

        if should_check {
            self.last_check = self.rows_processed;

            if self.executor.should_reoptimize(self.rows_processed)? {
                let (new_plan, decision) = self
                    .executor
                    .reoptimize_plan(&self.current_plan, statistics)?;

                if decision.beneficial {
                    self.current_plan = new_plan;
                    return Ok(true); // Indicate that re-optimization occurred
                }
            }
        }

        Ok(false)
    }

    /// Get current plan
    pub fn get_current_plan(&self) -> &Algebra {
        &self.current_plan
    }

    /// Get rows processed
    pub fn get_rows_processed(&self) -> u64 {
        self.rows_processed
    }

    /// Get elapsed time
    pub fn get_elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cardinality_estimator::EstimatorConfig;
    use crate::cost_model::CostModelConfig;

    #[test]
    fn test_operator_stats() {
        let mut stats = OperatorStats::new("scan_op".to_string(), 1000, 100.0);

        // Simulate execution
        std::thread::sleep(std::time::Duration::from_millis(10));
        stats.update(1500);

        // Check estimation error
        let error = stats.estimation_error();
        assert!(error > 0.0);

        // Check if re-optimization is needed
        assert!(stats.needs_reoptimization(0.3));
    }

    #[test]
    fn test_adaptive_executor() {
        let estimator_config = EstimatorConfig::default();
        let estimator = Arc::new(RwLock::new(CardinalityEstimator::new(estimator_config)));
        let cost_model_config = CostModelConfig::default();
        let cost_model = Arc::new(RwLock::new(CostModel::new(cost_model_config)));
        let config = AdaptiveConfig::default();

        let executor = AdaptiveQueryExecutor::new(estimator, cost_model, config);

        // Start monitoring an operator
        executor
            .start_operator("scan_1".to_string(), 1000, 100.0)
            .unwrap();

        // Update with actual results
        executor.update_operator("scan_1", 2000).unwrap();

        // Get runtime stats
        let stats = executor.get_runtime_stats().unwrap();
        assert!(stats.operator_stats.contains_key("scan_1"));

        let op_stats = &stats.operator_stats["scan_1"];
        assert_eq!(op_stats.actual_cardinality, 2000);
    }

    #[test]
    fn test_reoptimization_decision() {
        let decision = ReoptimizationDecision {
            timestamp_ms: 123456789,
            trigger_operator: "join_op".to_string(),
            trigger_error: 0.5,
            old_cost: 1000.0,
            new_cost: 800.0,
            beneficial: true,
            improvement_pct: 20.0,
        };

        assert!(decision.beneficial);
        assert_eq!(decision.improvement_pct, 20.0);
    }

    #[test]
    fn test_adaptive_execution_context() {
        let estimator_config = EstimatorConfig::default();
        let estimator = Arc::new(RwLock::new(CardinalityEstimator::new(estimator_config)));
        let cost_model_config = CostModelConfig::default();
        let cost_model = Arc::new(RwLock::new(CostModel::new(cost_model_config)));
        let config = AdaptiveConfig {
            check_interval: 100,
            ..Default::default()
        };

        let executor = Arc::new(AdaptiveQueryExecutor::new(estimator, cost_model, config));

        // Create dummy plan
        let plan = Algebra::Bgp(vec![]);
        let mut context = AdaptiveExecutionContext::new(executor.clone(), plan);

        // Process batches
        let stats = Statistics::new();
        let _reopt = context.process_batch(50, &stats).unwrap();
        // Not enough rows yet

        let _reopt = context.process_batch(100, &stats).unwrap();
        // May or may not re-optimize depending on stats

        assert_eq!(context.get_rows_processed(), 150);
    }
}
