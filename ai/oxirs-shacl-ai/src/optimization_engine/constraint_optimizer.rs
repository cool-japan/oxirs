//! Constraint optimizer for ordering and grouping strategies

use crate::{shape::PropertyConstraint, Result};

use super::types::OptimizationResult;

/// Constraint optimizer for ordering and grouping
#[derive(Debug)]
pub struct ConstraintOptimizer {
    ordering_strategies: Vec<ConstraintOrderingStrategy>,
    grouping_strategies: Vec<ConstraintGroupingStrategy>,
    optimization_history: Vec<OptimizationResult>,
}

/// Constraint ordering strategy
#[derive(Debug, Clone)]
pub struct ConstraintOrderingStrategy {
    pub strategy_name: String,
    pub strategy_type: OrderingStrategyType,
    pub effectiveness_score: f64,
    pub applicability_conditions: Vec<String>,
}

/// Types of constraint ordering strategies
#[derive(Debug, Clone)]
pub enum OrderingStrategyType {
    FailFast,        // Order by likelihood of failure
    CostBased,       // Order by execution cost
    DependencyBased, // Order by constraint dependencies
    DataDriven,      // Order based on data characteristics
    Hybrid,          // Combination of strategies
}

/// Constraint grouping strategy
#[derive(Debug, Clone)]
pub struct ConstraintGroupingStrategy {
    pub strategy_name: String,
    pub grouping_criteria: GroupingCriteria,
    pub parallel_execution: bool,
    pub cache_sharing: bool,
}

/// Criteria for grouping constraints
#[derive(Debug, Clone)]
pub enum GroupingCriteria {
    ByProperty,   // Group by property path
    ByComplexity, // Group by execution complexity
    ByDataAccess, // Group by data access patterns
    ByCache,      // Group by cache effectiveness
}

impl Default for ConstraintOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintOptimizer {
    pub fn new() -> Self {
        Self {
            ordering_strategies: Self::default_ordering_strategies(),
            grouping_strategies: Self::default_grouping_strategies(),
            optimization_history: Vec::new(),
        }
    }

    fn default_ordering_strategies() -> Vec<ConstraintOrderingStrategy> {
        vec![
            ConstraintOrderingStrategy {
                strategy_name: "FailFast".to_string(),
                strategy_type: OrderingStrategyType::FailFast,
                effectiveness_score: 0.8,
                applicability_conditions: vec!["has_high_failure_rate_constraints".to_string()],
            },
            ConstraintOrderingStrategy {
                strategy_name: "CostBased".to_string(),
                strategy_type: OrderingStrategyType::CostBased,
                effectiveness_score: 0.7,
                applicability_conditions: vec!["has_varied_execution_costs".to_string()],
            },
        ]
    }

    fn default_grouping_strategies() -> Vec<ConstraintGroupingStrategy> {
        vec![
            ConstraintGroupingStrategy {
                strategy_name: "ByProperty".to_string(),
                grouping_criteria: GroupingCriteria::ByProperty,
                parallel_execution: true,
                cache_sharing: false,
            },
            ConstraintGroupingStrategy {
                strategy_name: "ByComplexity".to_string(),
                grouping_criteria: GroupingCriteria::ByComplexity,
                parallel_execution: true,
                cache_sharing: true,
            },
        ]
    }

    pub fn optimize_constraint_order(
        &self,
        constraints: &[PropertyConstraint],
    ) -> Result<Vec<usize>> {
        // Simple ordering by complexity (fastest first)
        let mut indexed_constraints: Vec<(usize, f64)> = constraints
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let complexity = match c.constraint_type().as_str() {
                    "sh:pattern" => 3.0,
                    "sh:sparql" => 4.0,
                    "sh:class" => 2.5,
                    _ => 1.0,
                };
                (i, complexity)
            })
            .collect();

        // Sort by complexity (ascending for fail-fast)
        indexed_constraints
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(indexed_constraints.into_iter().map(|(i, _)| i).collect())
    }
}
