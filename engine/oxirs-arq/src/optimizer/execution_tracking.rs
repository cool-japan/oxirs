//! Execution Tracking and Learning
//!
//! Records and analyzes query execution for adaptive optimization.

use crate::algebra::Algebra;
use std::time::Duration;

/// Execution record for learning-based optimization
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub query_hash: u64,
    pub algebra: Algebra,
    pub execution_time: Duration,
    pub cardinality: usize,
    pub memory_usage: usize,
    pub optimization_decisions: Vec<OptimizationDecision>,
}

/// Optimization decision record
#[derive(Debug, Clone)]
pub struct OptimizationDecision {
    pub optimization_type: OptimizationType,
    pub before_cost: f64,
    pub after_cost: f64,
    pub success: bool,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    JoinReordering,
    FilterPushdown,
    ProjectionPushdown,
    ConstantFolding,
    IndexSelection,
    MaterializationPoint,
    ParallelizationStrategy,
}

/// Query characteristics for adaptive optimization
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    pub variable_count: usize,
    pub has_object_variables: bool,
    pub has_literal_constraints: bool,
    pub join_complexity: JoinComplexity,
    pub pattern_count: usize,
}

/// Join complexity classification
#[derive(Debug, Clone)]
pub enum JoinComplexity {
    Simple,   // 1-2 patterns, simple joins
    Moderate, // 3-5 patterns, some complexity
    Complex,  // 6+ patterns or complex join patterns
}

impl ExecutionRecord {
    /// Create a new execution record
    pub fn new(query_hash: u64, algebra: Algebra) -> Self {
        Self {
            query_hash,
            algebra,
            execution_time: Duration::default(),
            cardinality: 0,
            memory_usage: 0,
            optimization_decisions: Vec::new(),
        }
    }

    /// Add an optimization decision
    pub fn add_decision(&mut self, decision: OptimizationDecision) {
        self.optimization_decisions.push(decision);
    }

    /// Calculate total optimization benefit
    pub fn total_benefit(&self) -> f64 {
        self.optimization_decisions
            .iter()
            .map(|d| d.before_cost - d.after_cost)
            .sum()
    }
}

impl OptimizationDecision {
    /// Create a new optimization decision
    pub fn new(optimization_type: OptimizationType, before_cost: f64, after_cost: f64) -> Self {
        Self {
            optimization_type,
            before_cost,
            after_cost,
            success: after_cost < before_cost,
        }
    }

    /// Get the benefit of this optimization
    pub fn benefit(&self) -> f64 {
        if self.success {
            self.before_cost - self.after_cost
        } else {
            0.0
        }
    }
}
