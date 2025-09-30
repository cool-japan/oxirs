//! Optimizer Statistics
//!
//! Collection and management of statistics for query optimization.

use std::collections::HashMap;
use std::time::Duration;

use super::execution_tracking::{ExecutionRecord, OptimizationDecision, OptimizationType};
use super::index_types::{IndexStatistics, IndexType};
use crate::algebra::Variable;

/// Query optimization statistics
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Index statistics by type
    pub index_stats: HashMap<IndexType, IndexStatistics>,
    /// Join selectivity statistics
    pub join_selectivities: HashMap<String, f64>,
    /// Filter selectivity statistics
    pub filter_selectivities: HashMap<String, f64>,
    /// Average execution times by pattern
    pub execution_times: HashMap<String, Duration>,
    /// Cardinality estimates
    pub cardinalities: HashMap<String, usize>,
    /// Pattern cardinality estimates
    pub pattern_cardinality: HashMap<String, usize>,
    /// Predicate frequency statistics
    pub predicate_frequency: HashMap<String, usize>,
    /// Subject cardinality statistics
    pub subject_cardinality: HashMap<String, usize>,
    /// Object cardinality statistics
    pub object_cardinality: HashMap<String, usize>,
    /// Variable selectivity statistics
    pub variable_selectivity: HashMap<Variable, f64>,
    /// Optimization success rates
    pub optimization_success_rates: HashMap<OptimizationType, f64>,
    /// Total queries processed
    pub total_queries: usize,
    /// Total optimization time
    pub total_optimization_time: Duration,
}

impl Statistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics with execution record
    pub fn update_with_execution(&mut self, record: &ExecutionRecord) {
        self.total_queries += 1;

        // Update optimization success rates
        for decision in &record.optimization_decisions {
            self.update_optimization_success_rate(decision);
        }

        // Update execution times
        let query_hash = record.query_hash.to_string();
        self.execution_times
            .insert(query_hash.clone(), record.execution_time);
        self.cardinalities.insert(query_hash, record.cardinality);
    }

    /// Update index statistics
    pub fn update_index_stats(&mut self, index_type: IndexType, stats: IndexStatistics) {
        self.index_stats.insert(index_type, stats);
    }

    /// Get join selectivity
    pub fn get_join_selectivity(&self, join_pattern: &str) -> Option<f64> {
        self.join_selectivities.get(join_pattern).copied()
    }

    /// Set join selectivity
    pub fn set_join_selectivity(&mut self, join_pattern: String, selectivity: f64) {
        self.join_selectivities.insert(join_pattern, selectivity);
    }

    /// Get filter selectivity
    pub fn get_filter_selectivity(&self, filter_pattern: &str) -> Option<f64> {
        self.filter_selectivities.get(filter_pattern).copied()
    }

    /// Set filter selectivity
    pub fn set_filter_selectivity(&mut self, filter_pattern: String, selectivity: f64) {
        self.filter_selectivities
            .insert(filter_pattern, selectivity);
    }

    /// Get estimated cardinality
    pub fn get_estimated_cardinality(&self, pattern: &str) -> Option<usize> {
        self.cardinalities.get(pattern).copied()
    }

    /// Get optimization success rate
    pub fn get_optimization_success_rate(&self, opt_type: &OptimizationType) -> f64 {
        self.optimization_success_rates
            .get(opt_type)
            .copied()
            .unwrap_or(0.5) // Default 50% success rate
    }

    /// Add optimization time
    pub fn add_optimization_time(&mut self, duration: Duration) {
        self.total_optimization_time += duration;
    }

    /// Get average optimization time per query
    pub fn average_optimization_time(&self) -> Duration {
        if self.total_queries > 0 {
            self.total_optimization_time / self.total_queries as u32
        } else {
            Duration::default()
        }
    }

    fn update_optimization_success_rate(&mut self, decision: &OptimizationDecision) {
        let current_rate = self
            .optimization_success_rates
            .get(&decision.optimization_type)
            .copied()
            .unwrap_or(0.5);

        // Simple exponential moving average
        let alpha = 0.1;
        let success_value = if decision.success { 1.0 } else { 0.0 };
        let new_rate = alpha * success_value + (1.0 - alpha) * current_rate;

        self.optimization_success_rates
            .insert(decision.optimization_type.clone(), new_rate);
    }
}

// Implement PartialEq for OptimizationType to use as HashMap key
impl PartialEq for OptimizationType {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for OptimizationType {}

impl std::hash::Hash for OptimizationType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}
