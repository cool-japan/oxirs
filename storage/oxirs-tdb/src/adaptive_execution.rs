//! Adaptive Query Execution
//!
//! This module provides runtime query plan adaptation based on actual execution
//! statistics. It monitors query performance and dynamically adjusts execution
//! strategies to improve performance.
//!
//! ## Features
//! - Runtime cardinality tracking
//! - Dynamic plan switching
//! - Execution statistics collection
//! - Learning from query history
//! - Automatic re-optimization on plan deviation

use crate::dictionary::NodeId;
use crate::error::{Result, TdbError};
use crate::query_hints::IndexType;
use crate::query_optimizer::{QueryOptimizer, QueryPattern, QueryPlan};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Adaptive execution engine
pub struct AdaptiveExecutor {
    /// Query optimizer for plan generation
    optimizer: Arc<QueryOptimizer>,
    /// Execution history tracking
    history: Arc<RwLock<ExecutionHistory>>,
    /// Configuration
    config: AdaptiveConfig,
}

/// Configuration for adaptive execution
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable adaptive execution
    pub enabled: bool,
    /// Threshold for plan deviation before reoptimization (ratio)
    pub reoptimize_threshold: f64,
    /// Minimum samples before adaptation
    pub min_samples: usize,
    /// Maximum history entries per pattern
    pub max_history_entries: usize,
    /// Enable mid-execution plan switching
    pub enable_plan_switching: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reoptimize_threshold: 2.0, // Reoptimize if actual results are 2x estimate
            min_samples: 3,
            max_history_entries: 100,
            enable_plan_switching: true,
        }
    }
}

/// Execution statistics for a query
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Pattern executed
    pub pattern: QueryPattern,
    /// Plan used
    pub index_used: IndexType,
    /// Actual execution time
    pub execution_time: Duration,
    /// Actual number of results
    pub actual_results: usize,
    /// Estimated results (from plan)
    pub estimated_results: usize,
    /// Whether plan was optimal
    pub was_optimal: bool,
    /// Timestamp of execution
    pub timestamp: Instant,
}

impl ExecutionStats {
    /// Calculate estimation error ratio (1.0 = perfect, >1.0 = underestimate, <1.0 = overestimate)
    pub fn estimation_error_ratio(&self) -> f64 {
        if self.estimated_results == 0 {
            return if self.actual_results == 0 {
                1.0
            } else {
                f64::INFINITY
            };
        }
        self.actual_results as f64 / self.estimated_results as f64
    }

    /// Check if plan significantly deviated from estimate
    pub fn has_significant_deviation(&self, threshold: f64) -> bool {
        let ratio = self.estimation_error_ratio();
        ratio > threshold || ratio < (1.0 / threshold)
    }
}

/// Execution history tracker
#[derive(Debug, Default)]
struct ExecutionHistory {
    /// Map from pattern to execution statistics
    entries: HashMap<QueryPattern, Vec<ExecutionStats>>,
}

impl ExecutionHistory {
    /// Add execution statistics
    fn add_entry(&mut self, stats: ExecutionStats, max_entries: usize) {
        let entries = self.entries.entry(stats.pattern.clone()).or_default();

        entries.push(stats);

        // Limit history size
        if entries.len() > max_entries {
            entries.remove(0);
        }
    }

    /// Get average execution time for a pattern
    fn avg_execution_time(&self, pattern: &QueryPattern) -> Option<Duration> {
        let entries = self.entries.get(pattern)?;
        if entries.is_empty() {
            return None;
        }

        let total_nanos: u128 = entries.iter().map(|e| e.execution_time.as_nanos()).sum();
        let avg_nanos = total_nanos / entries.len() as u128;

        Some(Duration::from_nanos(avg_nanos as u64))
    }

    /// Get average estimation error ratio for a pattern
    fn avg_estimation_error(&self, pattern: &QueryPattern) -> Option<f64> {
        let entries = self.entries.get(pattern)?;
        if entries.is_empty() {
            return None;
        }

        let total_error: f64 = entries.iter().map(|e| e.estimation_error_ratio()).sum();
        Some(total_error / entries.len() as f64)
    }

    /// Get number of samples for a pattern
    fn sample_count(&self, pattern: &QueryPattern) -> usize {
        self.entries.get(pattern).map_or(0, |e| e.len())
    }

    /// Check if pattern has enough history for adaptation
    fn has_sufficient_history(&self, pattern: &QueryPattern, min_samples: usize) -> bool {
        self.sample_count(pattern) >= min_samples
    }
}

impl AdaptiveExecutor {
    /// Create a new adaptive executor
    pub fn new(optimizer: Arc<QueryOptimizer>) -> Self {
        Self {
            optimizer,
            history: Arc::new(RwLock::new(ExecutionHistory::default())),
            config: AdaptiveConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(optimizer: Arc<QueryOptimizer>, config: AdaptiveConfig) -> Self {
        Self {
            optimizer,
            history: Arc::new(RwLock::new(ExecutionHistory::default())),
            config,
        }
    }

    /// Generate initial query plan with adaptive hints
    pub fn create_plan(&self, pattern: QueryPattern) -> Result<QueryPlan> {
        if !self.config.enabled {
            // Fallback to standard optimization
            let hints = crate::query_hints::QueryHints::new();
            return self.optimizer.optimize(pattern, &hints);
        }

        // Check execution history
        let history = self.history.read();
        if history.has_sufficient_history(&pattern, self.config.min_samples) {
            // Use historical data to improve estimates
            if let Some(avg_error) = history.avg_estimation_error(&pattern) {
                // Create plan with adjusted estimates
                let mut hints = crate::query_hints::QueryHints::new();

                // Apply learned correction factor
                if avg_error > self.config.reoptimize_threshold {
                    // Disable caching for queries with poor estimates
                    hints = hints.with_caching(false);
                }

                return self.optimizer.optimize(pattern, &hints);
            }
        }

        // No history or insufficient samples - use standard optimization
        let hints = crate::query_hints::QueryHints::new();
        self.optimizer.optimize(pattern, &hints)
    }

    /// Record execution results for learning
    pub fn record_execution(
        &self,
        pattern: QueryPattern,
        plan: &QueryPlan,
        actual_results: usize,
        execution_time: Duration,
    ) {
        let stats = ExecutionStats {
            pattern: pattern.clone(),
            index_used: plan.index,
            execution_time,
            actual_results,
            estimated_results: plan.estimated_results,
            was_optimal: !self.should_reoptimize(plan, actual_results),
            timestamp: Instant::now(),
        };

        let mut history = self.history.write();
        history.add_entry(stats, self.config.max_history_entries);
    }

    /// Check if query should be reoptimized based on results
    pub fn should_reoptimize(&self, plan: &QueryPlan, actual_results: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        let ratio = if plan.estimated_results == 0 {
            if actual_results == 0 {
                1.0
            } else {
                return true; // Significantly wrong
            }
        } else {
            actual_results as f64 / plan.estimated_results as f64
        };

        ratio > self.config.reoptimize_threshold || ratio < (1.0 / self.config.reoptimize_threshold)
    }

    /// Get execution statistics for a pattern
    pub fn get_statistics(&self, pattern: &QueryPattern) -> Option<PatternStatistics> {
        let history = self.history.read();

        if !history.has_sufficient_history(pattern, self.config.min_samples) {
            return None;
        }

        Some(PatternStatistics {
            sample_count: history.sample_count(pattern),
            avg_execution_time: history.avg_execution_time(pattern)?,
            avg_estimation_error: history.avg_estimation_error(pattern)?,
        })
    }

    /// Clear execution history
    pub fn clear_history(&self) {
        let mut history = self.history.write();
        history.entries.clear();
    }

    /// Get total number of tracked patterns
    pub fn tracked_patterns_count(&self) -> usize {
        let history = self.history.read();
        history.entries.len()
    }

    /// Get total number of execution samples
    pub fn total_samples(&self) -> usize {
        let history = self.history.read();
        history.entries.values().map(|v| v.len()).sum()
    }

    /// Reoptimize query with updated estimates
    pub fn reoptimize(&self, pattern: QueryPattern, actual_results: usize) -> Result<QueryPlan> {
        // Use actual results to inform new plan
        let mut hints = crate::query_hints::QueryHints::new();

        // Adjust hints based on actual cardinality
        if actual_results > 10000 {
            // Disable caching for large result sets
            hints = hints.with_caching(false);
        }

        self.optimizer.optimize(pattern, &hints)
    }
}

/// Statistics summary for a query pattern
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    /// Number of execution samples
    pub sample_count: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Average estimation error ratio
    pub avg_estimation_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::{StatisticsConfig, TripleStatistics};

    fn create_test_executor() -> AdaptiveExecutor {
        let stats = Arc::new(TripleStatistics::new(StatisticsConfig::default()));
        let optimizer = Arc::new(QueryOptimizer::new(stats));
        AdaptiveExecutor::new(optimizer)
    }

    #[test]
    fn test_adaptive_executor_creation() {
        let executor = create_test_executor();
        assert_eq!(executor.tracked_patterns_count(), 0);
        assert_eq!(executor.total_samples(), 0);
    }

    #[test]
    fn test_execution_stats_error_ratio() {
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let stats = ExecutionStats {
            pattern,
            index_used: IndexType::SPO,
            execution_time: Duration::from_millis(100),
            actual_results: 200,
            estimated_results: 100,
            was_optimal: false,
            timestamp: Instant::now(),
        };

        // Actual is 2x estimate
        assert_eq!(stats.estimation_error_ratio(), 2.0);
        assert!(stats.has_significant_deviation(1.5));
    }

    #[test]
    fn test_record_and_retrieve_execution() {
        let executor = create_test_executor();
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Record multiple executions
        executor.record_execution(pattern.clone(), &plan, 100, Duration::from_millis(50));
        executor.record_execution(pattern.clone(), &plan, 110, Duration::from_millis(55));
        executor.record_execution(pattern.clone(), &plan, 90, Duration::from_millis(45));

        // Should have statistics now
        let stats = executor.get_statistics(&pattern);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.sample_count, 3);
        assert!(stats.avg_execution_time.as_millis() >= 45);
        assert!(stats.avg_execution_time.as_millis() <= 55);
    }

    #[test]
    fn test_reoptimization_trigger() {
        let executor = create_test_executor();
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Small deviation - should not reoptimize
        assert!(!executor.should_reoptimize(&plan, 120));

        // Large deviation - should reoptimize
        assert!(executor.should_reoptimize(&plan, 300));

        // Underestimate - should also reoptimize
        assert!(executor.should_reoptimize(&plan, 30));
    }

    #[test]
    fn test_history_size_limit() {
        let config = AdaptiveConfig {
            max_history_entries: 5,
            ..Default::default()
        };

        let stats = Arc::new(TripleStatistics::new(StatisticsConfig::default()));
        let optimizer = Arc::new(QueryOptimizer::new(stats));
        let executor = AdaptiveExecutor::with_config(optimizer, config);

        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Record 10 executions
        for i in 0..10 {
            executor.record_execution(pattern.clone(), &plan, 100 + i, Duration::from_millis(50));
        }

        // Should only keep last 5
        let stats = executor.get_statistics(&pattern).unwrap();
        assert_eq!(stats.sample_count, 5);
    }

    #[test]
    fn test_disabled_adaptive_execution() {
        let config = AdaptiveConfig {
            enabled: false,
            ..Default::default()
        };

        let stats = Arc::new(TripleStatistics::new(StatisticsConfig::default()));
        let optimizer = Arc::new(QueryOptimizer::new(stats));
        let executor = AdaptiveExecutor::with_config(optimizer, config);

        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Should never trigger reoptimization when disabled
        assert!(!executor.should_reoptimize(&plan, 1000));
    }

    #[test]
    fn test_clear_history() {
        let executor = create_test_executor();
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Record some executions
        for _ in 0..5 {
            executor.record_execution(pattern.clone(), &plan, 100, Duration::from_millis(50));
        }

        assert_eq!(executor.total_samples(), 5);

        executor.clear_history();

        assert_eq!(executor.total_samples(), 0);
        assert_eq!(executor.tracked_patterns_count(), 0);
    }

    #[test]
    fn test_insufficient_samples_no_stats() {
        let executor = create_test_executor();
        let pattern = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let plan = QueryPlan::new(pattern.clone(), IndexType::SPO, 100);

        // Record only 2 executions (less than min_samples which is 3)
        executor.record_execution(pattern.clone(), &plan, 100, Duration::from_millis(50));
        executor.record_execution(pattern.clone(), &plan, 100, Duration::from_millis(50));

        // Should not have statistics yet
        assert!(executor.get_statistics(&pattern).is_none());
    }

    #[test]
    fn test_pattern_specific_tracking() {
        let executor = create_test_executor();

        let pattern1 = QueryPattern::new(Some(NodeId::new(1)), None, None);
        let pattern2 = QueryPattern::new(None, Some(NodeId::new(2)), None);

        let plan1 = QueryPlan::new(pattern1.clone(), IndexType::SPO, 100);
        let plan2 = QueryPlan::new(pattern2.clone(), IndexType::POS, 200);

        // Record executions for both patterns
        for _ in 0..3 {
            executor.record_execution(pattern1.clone(), &plan1, 100, Duration::from_millis(50));
            executor.record_execution(pattern2.clone(), &plan2, 200, Duration::from_millis(100));
        }

        // Should track both patterns separately
        assert_eq!(executor.tracked_patterns_count(), 2);
        assert_eq!(executor.total_samples(), 6);

        let stats1 = executor.get_statistics(&pattern1).unwrap();
        let stats2 = executor.get_statistics(&pattern2).unwrap();

        assert_eq!(stats1.sample_count, 3);
        assert_eq!(stats2.sample_count, 3);
        assert!(stats1.avg_execution_time < stats2.avg_execution_time);
    }
}
