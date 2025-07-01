//! Join Optimizer Types
//!
//! This module contains core data types used by the join optimizer.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::planner::planning::{FilterExpression as PlanningFilterExpression, TriplePattern};
use crate::service_optimizer::{
    BushyNodeType, BushyTreeNode, ChainJoinPattern, ChainOptimizationType, CyclePattern,
    JoinAlgorithm, JoinEdge, JoinExecutionResult, JoinExecutionStrategy, JoinOperation,
    JoinOperationType, JoinPlan, ParallelizationOpportunity, QueryInfo, SpecialJoinPatterns,
    StarJoinPattern, StarOptimizationType, StrategyPerformance,
};

/// Join graph representation for optimization
#[derive(Debug, Clone)]
pub struct JoinGraph {
    pub nodes: Vec<JoinNode>,
    pub edges: Vec<JoinEdge>,
}

/// Join node representing a triple pattern or subquery
#[derive(Debug, Clone)]
pub struct JoinNode {
    pub id: String,
    pub pattern: TriplePattern,
    pub variables: HashSet<String>,
    pub selectivity: f64,
    pub estimated_cardinality: u64,
    pub execution_cost: f64,
}

/// Join optimization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum JoinOptimizationStrategy {
    StarJoin,
    ChainJoin,
    BushyTree,
    Dynamic,
}

/// Statistics for join operations
#[derive(Debug, Clone, Default)]
pub struct JoinStatistics {
    pub total_joins: u64,
    pub successful_joins: u64,
    pub failed_joins: u64,
    pub total_execution_time: Duration,
    pub average_join_time: Duration,
    pub strategy_performance: HashMap<JoinOptimizationStrategy, StrategyPerformance>,
    pub runtime_stats: RuntimeStatistics,
}

/// Runtime statistics for adaptive optimization
#[derive(Debug, Clone, Default)]
pub struct RuntimeStatistics {
    pub query_execution_times: Vec<Duration>,
    pub resource_utilization: HashMap<String, f64>,
    pub error_rates: HashMap<String, f64>,
    pub throughput_metrics: HashMap<String, f64>,
}

/// Cost model for join operations
#[derive(Debug, Clone)]
pub struct JoinCostModel {
    pub cpu_cost_factor: f64,
    pub network_cost_factor: f64,
    pub memory_cost_factor: f64,
    pub disk_io_cost_factor: f64,
}

impl Default for JoinCostModel {
    fn default() -> Self {
        Self {
            cpu_cost_factor: 1.0,
            network_cost_factor: 10.0,
            memory_cost_factor: 0.1,
            disk_io_cost_factor: 5.0,
        }
    }
}

/// Adaptive execution controller
#[derive(Debug, Clone)]
pub struct AdaptiveExecutionController {
    pub enabled: bool,
    pub learning_rate: f64,
    pub adaptation_threshold: f64,
    pub history_window: usize,
}

impl Default for AdaptiveExecutionController {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.1,
            adaptation_threshold: 0.15,
            history_window: 100,
        }
    }
}

/// Join execution context
#[derive(Debug, Clone)]
pub struct JoinExecutionContext {
    pub query_id: String,
    pub start_time: Instant,
    pub memory_budget: usize,
    pub parallelism_degree: usize,
    pub timeout: Duration,
}

/// Join pattern analysis result
#[derive(Debug, Clone)]
pub struct JoinPatternAnalysis {
    pub pattern_type: JoinPatternType,
    pub complexity_score: f64,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub estimated_benefit: f64,
}

/// Types of join patterns detected
#[derive(Debug, Clone, PartialEq)]
pub enum JoinPatternType {
    Star { center_variable: String },
    Chain { path_variables: Vec<String> },
    Cycle { cycle_variables: Vec<String> },
    Tree { root_variable: String },
    Complex,
}

/// Optimization opportunities identified
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OptimizationType,
    pub description: String,
    pub estimated_improvement: f64,
    pub implementation_complexity: ComplexityLevel,
}

/// Types of optimizations that can be applied
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    JoinReordering,
    IndexUtilization,
    Parallelization,
    Caching,
    Pruning,
    MaterializedView,
}

/// Complexity levels for optimization implementation
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl RuntimeStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_execution_time(&mut self, duration: Duration) {
        self.query_execution_times.push(duration);

        // Keep only recent measurements (sliding window)
        if self.query_execution_times.len() > 1000 {
            self.query_execution_times.remove(0);
        }
    }

    pub fn get_average_execution_time(&self) -> Duration {
        if self.query_execution_times.is_empty() {
            Duration::from_secs(0)
        } else {
            let total: Duration = self.query_execution_times.iter().sum();
            total / self.query_execution_times.len() as u32
        }
    }

    pub fn update_resource_utilization(&mut self, resource: String, utilization: f64) {
        self.resource_utilization.insert(resource, utilization);
    }

    pub fn update_error_rate(&mut self, component: String, error_rate: f64) {
        self.error_rates.insert(component, error_rate);
    }

    pub fn update_throughput_metric(&mut self, metric: String, value: f64) {
        self.throughput_metrics.insert(metric, value);
    }
}

impl JoinCostModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate the cost of a join operation
    pub fn estimate_join_cost(
        &self,
        left_cardinality: u64,
        right_cardinality: u64,
        join_algorithm: &JoinAlgorithm,
        network_latency: Duration,
    ) -> f64 {
        let base_cost = match join_algorithm {
            JoinAlgorithm::NestedLoop => {
                (left_cardinality as f64) * (right_cardinality as f64) * self.cpu_cost_factor
            }
            JoinAlgorithm::HashJoin => {
                (left_cardinality as f64 + right_cardinality as f64) * self.cpu_cost_factor
                    + (right_cardinality as f64) * self.memory_cost_factor
            }
            JoinAlgorithm::SortMergeJoin => {
                let sort_cost = (left_cardinality as f64).log2() * (left_cardinality as f64)
                    + (right_cardinality as f64).log2() * (right_cardinality as f64);
                sort_cost * self.cpu_cost_factor
                    + (left_cardinality + right_cardinality) as f64 * self.disk_io_cost_factor
            }
            JoinAlgorithm::IndexJoin => {
                (left_cardinality as f64) * (right_cardinality as f64).log2() * self.cpu_cost_factor
            }
        };

        // Add network cost
        let network_cost = network_latency.as_millis() as f64 * self.network_cost_factor;

        base_cost + network_cost
    }

    /// Estimate memory requirements for a join
    pub fn estimate_memory_requirements(
        &self,
        left_cardinality: u64,
        right_cardinality: u64,
        join_algorithm: &JoinAlgorithm,
    ) -> u64 {
        match join_algorithm {
            JoinAlgorithm::NestedLoop => 1024, // Minimal memory needed
            JoinAlgorithm::HashJoin => {
                // Hash table for smaller relation + some overhead
                let smaller_relation = left_cardinality.min(right_cardinality);
                smaller_relation * 64 // Assuming 64 bytes per tuple on average
            }
            JoinAlgorithm::SortMergeJoin => {
                // Need to sort both relations
                (left_cardinality + right_cardinality) * 32
            }
            JoinAlgorithm::IndexJoin => 2048, // Minimal memory for index lookups
        }
    }
}

impl AdaptiveExecutionController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Determine if adaptation should be triggered
    pub fn should_adapt(&self, current_performance: f64, historical_average: f64) -> bool {
        if !self.enabled {
            return false;
        }

        let performance_degradation =
            (historical_average - current_performance) / historical_average;
        performance_degradation > self.adaptation_threshold
    }

    /// Calculate adaptation factor based on performance metrics
    pub fn calculate_adaptation_factor(&self, performance_ratio: f64) -> f64 {
        if !self.enabled {
            return 1.0;
        }

        // Apply learning rate to smooth adaptation
        1.0 + (performance_ratio - 1.0) * self.learning_rate
    }
}
