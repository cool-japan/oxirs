//! Adaptive Query Executor with Re-optimization
//!
//! Implements adaptive query execution that monitors runtime statistics and
//! dynamically re-optimizes query plans based on actual execution characteristics.
//! Supports time-based and deviation-based triggers with checkpointing for plan switching.

use crate::algebra::Algebra;
use crate::cardinality_estimator::CardinalityEstimator;
use crate::cost_model::{CostEstimate, CostModel};
use crate::optimizer::Statistics;
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Timer};
use scirs2_core::profiling::Profiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Adaptive query executor that re-optimizes plans based on runtime feedback
pub struct AdaptiveExecutor {
    /// Advanced optimizer for re-optimization
    optimizer: Arc<RwLock<AdaptiveOptimizer>>,
    /// Configuration
    config: AdaptiveConfig,
    /// Performance profiler
    profiler: Profiler,
    /// Metrics counters
    metrics: AdaptiveMetrics,
}

/// Configuration for adaptive execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable adaptive re-optimization
    pub enable_adaptive: bool,
    /// Re-optimization trigger as percentage of query execution (0.0-1.0)
    pub re_opt_trigger_percent: f64,
    /// Re-optimization trigger in seconds
    pub re_opt_trigger_seconds: u64,
    /// Minimum interval between re-optimizations in seconds
    pub min_reopt_interval_seconds: u64,
    /// Plan switch threshold (new plan must be N times better)
    pub plan_switch_threshold: f64,
    /// Deviation threshold (actual/estimated ratio to trigger re-opt)
    pub deviation_threshold: f64,
    /// Maximum number of re-optimizations per query
    pub max_reoptimizations: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enable_adaptive: true,
            re_opt_trigger_percent: 0.1, // 10% of execution
            re_opt_trigger_seconds: 5,
            min_reopt_interval_seconds: 5,
            plan_switch_threshold: 2.0, // Must be 2x better
            deviation_threshold: 5.0,    // 5x deviation triggers re-opt
            max_reoptimizations: 3,
        }
    }
}

/// Runtime execution statistics
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    /// Operator statistics by operator ID
    pub operator_stats: HashMap<OperatorId, OperatorStats>,
    /// Total execution time
    pub execution_time: Duration,
    /// Total rows processed
    pub rows_processed: u64,
    /// Query start time
    pub start_time: Instant,
}

impl Default for RuntimeStatistics {
    fn default() -> Self {
        Self {
            operator_stats: HashMap::new(),
            execution_time: Duration::ZERO,
            rows_processed: 0,
            start_time: Instant::now(),
        }
    }
}

impl RuntimeStatistics {
    /// Update from batch execution
    pub fn update_from_batch(&mut self, batch: &BatchResult) -> Result<()> {
        self.rows_processed += batch.rows_produced;
        self.execution_time = self.start_time.elapsed();

        for (op_id, op_result) in &batch.operator_results {
            let stats = self
                .operator_stats
                .entry(op_id.clone())
                .or_insert_with(|| OperatorStats::new(op_id.clone()));

            stats.actual_cardinality += op_result.rows_produced;
            stats.actual_time_ms += op_result.execution_time_ms;
            stats.update_deviation();
        }

        Ok(())
    }

    /// Get maximum deviation across all operators
    pub fn max_deviation(&self) -> f64 {
        self.operator_stats
            .values()
            .map(|s| s.deviation)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0)
    }
}

/// Statistics for a single operator
#[derive(Debug, Clone)]
pub struct OperatorStats {
    /// Operator identifier
    pub operator_id: OperatorId,
    /// Actual cardinality observed
    pub actual_cardinality: u64,
    /// Estimated cardinality (from initial plan)
    pub estimated_cardinality: u64,
    /// Actual execution time in milliseconds
    pub actual_time_ms: f64,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: f64,
    /// Deviation ratio (actual / estimated)
    pub deviation: f64,
}

impl OperatorStats {
    pub fn new(operator_id: OperatorId) -> Self {
        Self {
            operator_id,
            actual_cardinality: 0,
            estimated_cardinality: 1,
            actual_time_ms: 0.0,
            estimated_time_ms: 1.0,
            deviation: 1.0,
        }
    }

    pub fn update_deviation(&mut self) {
        if self.estimated_cardinality > 0 {
            self.deviation = self.actual_cardinality as f64 / self.estimated_cardinality as f64;
        }
    }

    pub fn set_estimates(&mut self, cardinality: u64, time_ms: f64) {
        self.estimated_cardinality = cardinality;
        self.estimated_time_ms = time_ms;
    }
}

/// Operator identifier
pub type OperatorId = String;

/// Batch execution result
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Rows produced in this batch
    pub rows_produced: u64,
    /// Per-operator results
    pub operator_results: HashMap<OperatorId, OperatorResult>,
    /// Is query execution complete?
    pub is_complete: bool,
}

/// Result from a single operator
#[derive(Debug, Clone)]
pub struct OperatorResult {
    /// Rows produced
    pub rows_produced: u64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Query plan representation
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Algebraic representation
    pub algebra: Algebra,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Estimated total rows
    pub estimated_total_rows: u64,
    /// Operator cardinality estimates
    pub operator_estimates: HashMap<OperatorId, u64>,
}

/// Adaptive optimizer for query re-optimization
pub struct AdaptiveOptimizer {
    /// Cardinality estimator
    cardinality_estimator: Arc<RwLock<CardinalityEstimator>>,
    /// Cost model
    cost_model: Arc<RwLock<CostModel>>,
}

impl AdaptiveOptimizer {
    pub fn new(
        cardinality_estimator: Arc<RwLock<CardinalityEstimator>>,
        cost_model: Arc<RwLock<CostModel>>,
    ) -> Self {
        Self {
            cardinality_estimator,
            cost_model,
        }
    }

    /// Update cardinality estimate for an operator
    pub fn update_cardinality_estimate(&mut self, _op_id: OperatorId, actual: u64) -> Result<()> {
        // Update the cardinality estimator with actual results
        // This would feed into learning models in a full implementation
        debug!("Updated cardinality estimate: actual={}", actual);
        Ok(())
    }

    /// Update cost estimate for an operator
    pub fn update_cost_estimate(&mut self, _op_id: OperatorId, actual_time_ms: f64) -> Result<()> {
        // Update the cost model with actual timing
        debug!("Updated cost estimate: actual_time_ms={}", actual_time_ms);
        Ok(())
    }

    /// Optimize a query plan
    pub fn optimize(&self, _algebra: &Algebra) -> Result<QueryPlan> {
        // In a full implementation, this would use the optimizer
        // For now, create a simplified plan
        Ok(QueryPlan {
            algebra: Algebra::Bgp(vec![]),
            estimated_cost: 100.0,
            estimated_total_rows: 1000,
            operator_estimates: HashMap::new(),
        })
    }
}

/// Metrics for adaptive execution
pub struct AdaptiveMetrics {
    /// Number of re-optimizations triggered
    pub reoptimizations: Counter,
    /// Number of successful plan switches
    pub plan_switches: Counter,
    /// Time spent in re-optimization
    pub reopt_time: Timer,
    /// Queries improved by adaptation
    pub queries_improved: Counter,
}

impl Default for AdaptiveMetrics {
    fn default() -> Self {
        Self {
            reoptimizations: Counter::new("adaptive.reoptimizations".to_string()),
            plan_switches: Counter::new("adaptive.plan_switches".to_string()),
            reopt_time: Timer::new("adaptive.reopt_time".to_string()),
            queries_improved: Counter::new("adaptive.queries_improved".to_string()),
        }
    }
}

impl AdaptiveExecutor {
    /// Create a new adaptive executor
    pub fn new(
        cardinality_estimator: Arc<RwLock<CardinalityEstimator>>,
        cost_model: Arc<RwLock<CostModel>>,
        config: AdaptiveConfig,
    ) -> Self {
        let optimizer = Arc::new(RwLock::new(AdaptiveOptimizer::new(
            cardinality_estimator,
            cost_model,
        )));

        Self {
            optimizer,
            config,
            profiler: Profiler::new(),
            metrics: AdaptiveMetrics::default(),
        }
    }

    /// Execute query with adaptive re-optimization
    pub async fn execute_adaptive(
        &mut self,
        query: &Algebra,
        initial_plan: QueryPlan,
    ) -> Result<QueryResults> {
        let mut current_plan = initial_plan;
        let mut stats = RuntimeStatistics {
            start_time: Instant::now(),
            ..Default::default()
        };
        let mut last_reopt = Instant::now();

        let start_time = Instant::now();

        // Execute with checkpointing
        let mut executor = CheckpointedExecutor::new(current_plan.clone())?;

        loop {
            // Execute batch
            let batch_result = executor.execute_batch(1000).await?;

            // Collect statistics
            stats.update_from_batch(&batch_result)?;

            // Check if should re-optimize
            let elapsed = start_time.elapsed();
            let should_reopt =
                self.should_reoptimize(&stats, elapsed, last_reopt.elapsed())?;

            if should_reopt {
                info!(
                    "Triggering adaptive re-optimization at {}s",
                    elapsed.as_secs_f64()
                );

                self.metrics.reoptimizations.inc();
                self.profiler.start();

                // Refine cost model with actual statistics
                let refined_plan = self.reoptimize_with_statistics(query, &stats)?;

                // Check if new plan is significantly better
                if self.is_plan_significantly_better(&current_plan, &refined_plan, &stats)? {
                    let improvement = self.estimate_improvement(&current_plan, &refined_plan, &stats)?;
                    info!(
                        "Switching to new plan (estimated {}x improvement)",
                        improvement
                    );

                    // Checkpoint current state
                    let checkpoint = executor.checkpoint()?;

                    // Switch to new plan
                    current_plan = refined_plan;
                    executor =
                        CheckpointedExecutor::new_from_checkpoint(current_plan.clone(), checkpoint)?;

                    self.metrics.plan_switches.inc();
                    last_reopt = Instant::now();
                } else {
                    info!("New plan not significantly better, continuing with current plan");
                }
            }

            // Check if done
            if batch_result.is_complete {
                break;
            }
        }

        Ok(executor.finalize()?)
    }

    /// Determine if should trigger re-optimization
    fn should_reoptimize(
        &self,
        stats: &RuntimeStatistics,
        elapsed: Duration,
        since_last_reopt: Duration,
    ) -> Result<bool> {
        if !self.config.enable_adaptive {
            return Ok(false);
        }

        // Don't re-optimize too frequently (hysteresis)
        if since_last_reopt.as_secs() < self.config.min_reopt_interval_seconds {
            return Ok(false);
        }

        // Trigger after time threshold
        if elapsed.as_secs() >= self.config.re_opt_trigger_seconds {
            debug!("Re-optimization triggered by time threshold");
            return Ok(true);
        }

        // Trigger if significant deviation detected
        let max_deviation = stats.max_deviation();

        if max_deviation > self.config.deviation_threshold {
            info!("Large deviation detected: {}x", max_deviation);
            return Ok(true);
        }

        Ok(false)
    }

    /// Re-optimize query with runtime statistics
    fn reoptimize_with_statistics(
        &self,
        query: &Algebra,
        stats: &RuntimeStatistics,
    ) -> Result<QueryPlan> {
        // Update cost model with actual cardinalities
        let mut optimizer = self
            .optimizer
            .write()
            .map_err(|e| anyhow!("Failed to acquire optimizer lock: {}", e))?;

        for (op_id, op_stats) in &stats.operator_stats {
            optimizer.update_cardinality_estimate(op_id.clone(), op_stats.actual_cardinality)?;
            optimizer.update_cost_estimate(op_id.clone(), op_stats.actual_time_ms)?;
        }

        // Re-optimize query
        let new_plan = optimizer.optimize(query)?;
        Ok(new_plan)
    }

    /// Check if new plan is significantly better
    fn is_plan_significantly_better(
        &self,
        current_plan: &QueryPlan,
        new_plan: &QueryPlan,
        stats: &RuntimeStatistics,
    ) -> Result<bool> {
        // Estimate remaining cost for both plans
        let current_remaining_cost = self.estimate_remaining_cost(current_plan, stats)?;
        let new_remaining_cost = self.estimate_remaining_cost(new_plan, stats)?;

        let improvement = current_remaining_cost / new_remaining_cost;
        Ok(improvement > self.config.plan_switch_threshold)
    }

    fn estimate_remaining_cost(&self, plan: &QueryPlan, stats: &RuntimeStatistics) -> Result<f64> {
        // Estimate cost for remaining rows
        let processed = stats.rows_processed;
        let total_estimated = plan.estimated_total_rows.max(1);
        let remaining_percent = if processed < total_estimated {
            (total_estimated - processed) as f64 / total_estimated as f64
        } else {
            0.1 // Still some work remaining
        };

        Ok(plan.estimated_cost * remaining_percent)
    }

    fn estimate_improvement(
        &self,
        current: &QueryPlan,
        new: &QueryPlan,
        stats: &RuntimeStatistics,
    ) -> Result<f64> {
        let current_cost = self.estimate_remaining_cost(current, stats)?;
        let new_cost = self.estimate_remaining_cost(new, stats)?.max(0.1);
        Ok(current_cost / new_cost)
    }

    /// Get configuration
    pub fn get_config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Get profiler for inspection
    pub fn get_profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Get metrics
    pub fn get_metrics(&self) -> &AdaptiveMetrics {
        &self.metrics
    }
}

/// Executor with checkpointing support
pub struct CheckpointedExecutor {
    plan: QueryPlan,
    state: ExecutorState,
    rows_produced: u64,
}

/// Executor state for checkpointing
#[derive(Debug, Clone)]
pub struct ExecutorState {
    /// Operator states by operator ID
    pub operator_states: HashMap<OperatorId, OperatorState>,
    /// Rows processed so far
    pub rows_processed: u64,
    /// Intermediate results
    pub intermediate_results: Vec<u8>, // Serialized results
}

impl Default for ExecutorState {
    fn default() -> Self {
        Self {
            operator_states: HashMap::new(),
            rows_processed: 0,
            intermediate_results: Vec::new(),
        }
    }
}

/// State for a single operator
#[derive(Debug, Clone)]
pub struct OperatorState {
    /// Operator ID
    pub operator_id: OperatorId,
    /// Serialized state (hash tables, sort buffers, etc.)
    pub data: Vec<u8>,
    /// Rows processed by this operator
    pub rows_processed: u64,
}

impl CheckpointedExecutor {
    /// Create new executor with a plan
    pub fn new(plan: QueryPlan) -> Result<Self> {
        Ok(Self {
            plan,
            state: ExecutorState::default(),
            rows_produced: 0,
        })
    }

    /// Create executor from checkpoint
    pub fn new_from_checkpoint(plan: QueryPlan, checkpoint: ExecutorState) -> Result<Self> {
        Ok(Self {
            plan,
            state: checkpoint,
            rows_produced: 0,
        })
    }

    /// Execute a batch of rows
    pub async fn execute_batch(&mut self, batch_size: u64) -> Result<BatchResult> {
        // Simulate batch execution
        // In a real implementation, this would execute the query plan

        let rows_produced = batch_size.min(100); // Simulate producing rows
        self.rows_produced += rows_produced;
        self.state.rows_processed += rows_produced;

        let mut operator_results = HashMap::new();
        operator_results.insert(
            "scan_op".to_string(),
            OperatorResult {
                rows_produced,
                execution_time_ms: 10.0,
            },
        );

        // Check if complete (simulate)
        let is_complete = self.rows_produced >= 1000;

        Ok(BatchResult {
            rows_produced,
            operator_results,
            is_complete,
        })
    }

    /// Checkpoint current execution state
    pub fn checkpoint(&self) -> Result<ExecutorState> {
        Ok(self.state.clone())
    }

    /// Finalize execution and return results
    pub fn finalize(self) -> Result<QueryResults> {
        Ok(QueryResults {
            rows: self.rows_produced,
            execution_time: Duration::from_millis(100),
        })
    }
}

/// Query execution results
#[derive(Debug, Clone)]
pub struct QueryResults {
    /// Number of rows returned
    pub rows: u64,
    /// Total execution time
    pub execution_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cardinality_estimator::EstimatorConfig;
    use crate::cost_model::CostModelConfig;

    #[tokio::test]
    async fn test_adaptive_executor_basic() -> Result<()> {
        let estimator = Arc::new(RwLock::new(CardinalityEstimator::new(
            EstimatorConfig::default(),
        )));
        let cost_model = Arc::new(RwLock::new(CostModel::new(CostModelConfig::default())));
        let config = AdaptiveConfig::default();

        let mut executor = AdaptiveExecutor::new(estimator, cost_model, config);

        let query = Algebra::Bgp(vec![]);
        let plan = QueryPlan {
            algebra: query.clone(),
            estimated_cost: 1000.0,
            estimated_total_rows: 10000,
            operator_estimates: HashMap::new(),
        };

        let results = executor.execute_adaptive(&query, plan).await?;
        assert!(results.rows > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_checkpointing() -> Result<()> {
        let plan = QueryPlan {
            algebra: Algebra::Bgp(vec![]),
            estimated_cost: 100.0,
            estimated_total_rows: 1000,
            operator_estimates: HashMap::new(),
        };

        let mut executor = CheckpointedExecutor::new(plan.clone())?;

        // Execute some batches
        let _batch1 = executor.execute_batch(100).await?;
        let _batch2 = executor.execute_batch(100).await?;

        // Checkpoint
        let checkpoint = executor.checkpoint()?;
        assert_eq!(checkpoint.rows_processed, 200);

        // Create new executor from checkpoint
        let mut executor2 = CheckpointedExecutor::new_from_checkpoint(plan, checkpoint)?;
        let _batch3 = executor2.execute_batch(100).await?;

        Ok(())
    }

    #[test]
    fn test_runtime_statistics() {
        let mut stats = RuntimeStatistics {
            start_time: Instant::now(),
            ..Default::default()
        };

        let batch = BatchResult {
            rows_produced: 100,
            operator_results: {
                let mut map = HashMap::new();
                map.insert(
                    "op1".to_string(),
                    OperatorResult {
                        rows_produced: 100,
                        execution_time_ms: 50.0,
                    },
                );
                map
            },
            is_complete: false,
        };

        stats.update_from_batch(&batch).ok();
        assert_eq!(stats.rows_processed, 100);
    }

    #[test]
    fn test_deviation_calculation() {
        let mut op_stats = OperatorStats::new("test_op".to_string());
        op_stats.set_estimates(100, 10.0);
        op_stats.actual_cardinality = 500;
        op_stats.update_deviation();

        assert!((op_stats.deviation - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_config_defaults() {
        let config = AdaptiveConfig::default();
        assert!(config.enable_adaptive);
        assert_eq!(config.re_opt_trigger_seconds, 5);
        assert_eq!(config.min_reopt_interval_seconds, 5);
        assert_eq!(config.plan_switch_threshold, 2.0);
        assert_eq!(config.deviation_threshold, 5.0);
    }
}
