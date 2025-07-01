//! Query Optimization Module
//!
//! This module provides query optimization capabilities including
//! rule-based and cost-based optimization passes.

pub mod config;
pub mod index_types;
pub mod execution_tracking;
pub mod statistics;

pub use config::*;
pub use index_types::*;
pub use execution_tracking::*;
pub use statistics::*;

use crate::algebra::{Algebra, Expression, TriplePattern, Variable};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Main query optimizer
pub struct Optimizer {
    config: OptimizerConfig,
    statistics: Statistics,
    execution_records: Vec<ExecutionRecord>,
}

impl Optimizer {
    /// Create a new optimizer with configuration
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            statistics: Statistics::new(),
            execution_records: Vec::new(),
        }
    }

    /// Optimize a query algebra
    pub fn optimize(&mut self, algebra: Algebra) -> Result<Algebra> {
        let mut optimized = algebra;
        let mut pass = 0;

        // Apply optimization passes
        while pass < self.config.max_passes {
            let before = optimized.clone();

            if self.config.filter_pushdown {
                optimized = self.apply_filter_pushdown(optimized)?;
            }

            if self.config.join_reordering {
                optimized = self.apply_join_reordering(optimized)?;
            }

            if self.config.projection_pushdown {
                optimized = self.apply_projection_pushdown(optimized)?;
            }

            if self.config.constant_folding {
                optimized = self.apply_constant_folding(optimized)?;
            }

            if self.config.dead_code_elimination {
                optimized = self.apply_dead_code_elimination(optimized)?;
            }

            // Check for convergence
            if self.algebra_equal(&before, &optimized) {
                break;
            }

            pass += 1;
        }

        Ok(optimized)
    }

    /// Add execution record for learning
    pub fn add_execution_record(&mut self, record: ExecutionRecord) {
        self.statistics.update_with_execution(&record);
        self.execution_records.push(record);
    }

    /// Get optimizer statistics
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Apply filter pushdown optimization
    fn apply_filter_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let optimized_pattern = self.push_filter_down(*pattern, &condition)?;
                Ok(optimized_pattern)
            }
            Algebra::Join { left, right } => {
                Ok(Algebra::Join {
                    left: Box::new(self.apply_filter_pushdown(*left)?),
                    right: Box::new(self.apply_filter_pushdown(*right)?),
                })
            }
            Algebra::Union { left, right } => {
                Ok(Algebra::Union {
                    left: Box::new(self.apply_filter_pushdown(*left)?),
                    right: Box::new(self.apply_filter_pushdown(*right)?),
                })
            }
            other => Ok(other),
        }
    }

    /// Push filter down into the algebra tree
    fn push_filter_down(&self, algebra: Algebra, condition: &Expression) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_vars = self.extract_variables(&left);
                let right_vars = self.extract_variables(&right);
                let filter_vars = self.extract_expression_variables(condition);

                if filter_vars.iter().all(|v| left_vars.contains(v)) {
                    // Filter only uses left variables - push to left
                    Ok(Algebra::Join {
                        left: Box::new(Algebra::Filter {
                            pattern: left,
                            condition: condition.clone(),
                        }),
                        right,
                    })
                } else if filter_vars.iter().all(|v| right_vars.contains(v)) {
                    // Filter only uses right variables - push to right  
                    Ok(Algebra::Join {
                        left,
                        right: Box::new(Algebra::Filter {
                            pattern: right,
                            condition: condition.clone(),
                        }),
                    })
                } else {
                    // Filter uses variables from both sides - keep at join level
                    Ok(Algebra::Filter {
                        pattern: Box::new(Algebra::Join { left, right }),
                        condition: condition.clone(),
                    })
                }
            }
            other => Ok(Algebra::Filter {
                pattern: Box::new(other),
                condition: condition.clone(),
            }),
        }
    }

    /// Apply join reordering optimization based on selectivity
    fn apply_join_reordering(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_cost(&left);
                let right_cost = self.estimate_cost(&right);
                
                // Always put lower cost operation first for left-deep join trees
                if left_cost > right_cost {
                    Ok(Algebra::Join {
                        left: Box::new(self.apply_join_reordering(*right)?),
                        right: Box::new(self.apply_join_reordering(*left)?),
                    })
                } else {
                    Ok(Algebra::Join {
                        left: Box::new(self.apply_join_reordering(*left)?),
                        right: Box::new(self.apply_join_reordering(*right)?),
                    })
                }
            }
            Algebra::Union { left, right } => {
                Ok(Algebra::Union {
                    left: Box::new(self.apply_join_reordering(*left)?),
                    right: Box::new(self.apply_join_reordering(*right)?),
                })
            }
            other => Ok(other),
        }
    }

    /// Apply projection pushdown optimization
    fn apply_projection_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let optimized_pattern = self.push_projection_down(*pattern, &variables)?;
                Ok(Algebra::Project {
                    pattern: Box::new(optimized_pattern),
                    variables,
                })
            }
            Algebra::Join { left, right } => {
                Ok(Algebra::Join {
                    left: Box::new(self.apply_projection_pushdown(*left)?),
                    right: Box::new(self.apply_projection_pushdown(*right)?),
                })
            }
            other => Ok(other),
        }
    }

    /// Push projection down into algebra tree
    fn push_projection_down(&self, algebra: Algebra, needed_vars: &[Variable]) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_vars = self.extract_variables(&left);
                let right_vars = self.extract_variables(&right);
                
                let left_needed: Vec<Variable> = needed_vars.iter()
                    .filter(|v| left_vars.contains(v))
                    .cloned()
                    .collect();
                    
                let right_needed: Vec<Variable> = needed_vars.iter()
                    .filter(|v| right_vars.contains(v))
                    .cloned()
                    .collect();

                let left_projected = if !left_needed.is_empty() && left_needed.len() < left_vars.len() {
                    Algebra::Project {
                        pattern: left,
                        variables: left_needed,
                    }
                } else {
                    *left
                };

                let right_projected = if !right_needed.is_empty() && right_needed.len() < right_vars.len() {
                    Algebra::Project {
                        pattern: right,
                        variables: right_needed,
                    }
                } else {
                    *right
                };

                Ok(Algebra::Join {
                    left: Box::new(left_projected),
                    right: Box::new(right_projected),
                })
            }
            other => Ok(other),
        }
    }

    /// Apply constant folding optimization
    fn apply_constant_folding(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let folded_condition = self.fold_expression_constants(condition)?;
                
                // Check if condition is constant true/false
                if let Some(constant_value) = self.evaluate_constant_expression(&folded_condition) {
                    if constant_value {
                        // Filter is always true - remove it
                        Ok(self.apply_constant_folding(*pattern)?)
                    } else {
                        // Filter is always false - return empty result
                        Ok(Algebra::Empty)
                    }
                } else {
                    Ok(Algebra::Filter {
                        pattern: Box::new(self.apply_constant_folding(*pattern)?),
                        condition: folded_condition,
                    })
                }
            }
            Algebra::Join { left, right } => {
                Ok(Algebra::Join {
                    left: Box::new(self.apply_constant_folding(*left)?),
                    right: Box::new(self.apply_constant_folding(*right)?),
                })
            }
            other => Ok(other),
        }
    }

    /// Apply dead code elimination
    fn apply_dead_code_elimination(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let used_vars = self.extract_variables(&pattern);
                let needed_vars: Vec<Variable> = variables.into_iter()
                    .filter(|v| used_vars.contains(v))
                    .collect();
                    
                if needed_vars.is_empty() {
                    Ok(Algebra::Empty)
                } else {
                    Ok(Algebra::Project {
                        pattern: Box::new(self.apply_dead_code_elimination(*pattern)?),
                        variables: needed_vars,
                    })
                }
            }
            Algebra::Join { left, right } => {
                let optimized_left = self.apply_dead_code_elimination(*left)?;
                let optimized_right = self.apply_dead_code_elimination(*right)?;
                
                match (&optimized_left, &optimized_right) {
                    (Algebra::Empty, _) | (_, Algebra::Empty) => Ok(Algebra::Empty),
                    _ => Ok(Algebra::Join {
                        left: Box::new(optimized_left),
                        right: Box::new(optimized_right),
                    }),
                }
            }
            other => Ok(other),
        }
    }

    /// Extract variables from an algebra expression
    fn extract_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    if let TriplePattern { subject, predicate, object } = pattern {
                        if let crate::term::Term::Variable(v) = subject {
                            vars.insert(v.clone());
                        }
                        if let crate::term::Term::Variable(v) = predicate {
                            vars.insert(v.clone());
                        }
                        if let crate::term::Term::Variable(v) = object {
                            vars.insert(v.clone());
                        }
                    }
                }
            }
            Algebra::Join { left, right } => {
                vars.extend(self.extract_variables(left));
                vars.extend(self.extract_variables(right));
            }
            Algebra::Union { left, right } => {
                vars.extend(self.extract_variables(left));
                vars.extend(self.extract_variables(right));
            }
            Algebra::Filter { pattern, .. } => {
                vars.extend(self.extract_variables(pattern));
            }
            Algebra::Project { pattern, variables } => {
                vars.extend(self.extract_variables(pattern));
                vars.extend(variables.iter().cloned());
            }
            _ => {} // Other algebra types
        }
        vars
    }

    /// Extract variables from an expression
    fn extract_expression_variables(&self, expr: &Expression) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        match expr {
            Expression::Variable(v) => {
                vars.insert(v.clone());
            }
            Expression::And(left, right) | Expression::Or(left, right) => {
                vars.extend(self.extract_expression_variables(left));
                vars.extend(self.extract_expression_variables(right));
            }
            Expression::Not(inner) => {
                vars.extend(self.extract_expression_variables(inner));
            }
            Expression::FunctionCall { arguments, .. } => {
                for arg in arguments {
                    vars.extend(self.extract_expression_variables(arg));
                }
            }
            _ => {} // Other expression types
        }
        vars
    }

    /// Estimate execution cost for algebra
    fn estimate_cost(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => {
                // BGP cost based on pattern count and estimated selectivity
                patterns.len() as f64 * 10.0
            }
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                left_cost * right_cost * 0.1 // Join selectivity factor
            }
            Algebra::Union { left, right } => {
                self.estimate_cost(left) + self.estimate_cost(right)
            }
            Algebra::Filter { pattern, .. } => {
                self.estimate_cost(pattern) * 0.5 // Filter selectivity
            }
            _ => 1.0,
        }
    }

    /// Fold constants in expressions
    fn fold_expression_constants(&self, expr: Expression) -> Result<Expression> {
        match expr {
            Expression::And(left, right) => {
                let folded_left = self.fold_expression_constants(*left)?;
                let folded_right = self.fold_expression_constants(*right)?;
                Ok(Expression::And(Box::new(folded_left), Box::new(folded_right)))
            }
            Expression::Or(left, right) => {
                let folded_left = self.fold_expression_constants(*left)?;
                let folded_right = self.fold_expression_constants(*right)?;
                Ok(Expression::Or(Box::new(folded_left), Box::new(folded_right)))
            }
            other => Ok(other),
        }
    }

    /// Evaluate constant expressions to boolean values
    fn evaluate_constant_expression(&self, expr: &Expression) -> Option<bool> {
        match expr {
            Expression::Literal(literal) => {
                // Simple boolean literal evaluation
                if literal.value == "true" {
                    Some(true)
                } else if literal.value == "false" {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if two algebra expressions are equal (simplified check)
    fn algebra_equal(&self, a: &Algebra, b: &Algebra) -> bool {
        // Simplified equality check - should be improved
        format!("{:?}", a) == format!("{:?}", b)
    }

    /// Hash an algebra expression for caching
    pub fn hash_algebra(&self, algebra: &Algebra) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", algebra).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }
}

/// Type alias for backwards compatibility
pub type QueryOptimizer = Optimizer;