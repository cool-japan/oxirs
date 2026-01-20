//! Query Optimization Module
//!
//! This module provides query optimization capabilities including
//! rule-based and cost-based optimization passes.

pub mod cardinality_integration;
pub mod config;
pub mod execution_tracking;
pub mod index_types;
pub mod production_tuning;
pub mod statistics;

pub use cardinality_integration::*;
pub use config::*;
pub use execution_tracking::*;
pub use index_types::*;
pub use production_tuning::*;
pub use statistics::*;

use crate::algebra::{Algebra, Expression, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel, CostModelConfig};
use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Query complexity metrics for adaptive optimization
#[derive(Debug, Clone, Default)]
struct QueryComplexity {
    /// Number of triple patterns in the query
    triple_patterns: usize,
    /// Number of joins
    joins: usize,
    /// Number of filters
    filters: usize,
    /// Whether query has ordering
    ordering: bool,
    /// Whether query has grouping
    grouping: bool,
}

/// Main query optimizer
pub struct Optimizer {
    config: OptimizerConfig,
    statistics: Statistics,
    execution_records: Vec<ExecutionRecord>,
    cost_model: CostModel,
}

impl Optimizer {
    /// Create a new optimizer with configuration
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            statistics: Statistics::new(),
            execution_records: Vec::new(),
            cost_model: CostModel::new(CostModelConfig::default()),
        }
    }

    /// Create a new optimizer with custom cost model configuration
    pub fn with_cost_model(config: OptimizerConfig, cost_config: CostModelConfig) -> Self {
        Self {
            config,
            statistics: Statistics::new(),
            execution_records: Vec::new(),
            cost_model: CostModel::new(cost_config),
        }
    }

    /// Optimize a query algebra
    pub fn optimize(&mut self, algebra: Algebra) -> Result<Algebra> {
        // Adaptive optimization: use fast path for simple queries
        let complexity = self.estimate_query_complexity(&algebra);

        // For simple queries (≤5 triple patterns), skip cost-based optimization
        // to avoid optimization overhead exceeding benefits
        let use_cost_based = if complexity.triple_patterns <= 5 {
            false // Fast path: simple heuristics only
        } else {
            self.config.cost_based // Complex queries benefit from cost model
        };

        let mut optimized = algebra;
        let mut pass = 0;

        // Limit passes for simple queries to minimize overhead
        let effective_max_passes = if complexity.triple_patterns <= 5 {
            2.min(self.config.max_passes) // Maximum 2 passes for simple queries
        } else {
            self.config.max_passes
        };

        // Apply optimization passes
        while pass < effective_max_passes {
            let before = optimized.clone();

            if self.config.filter_pushdown {
                optimized = self.apply_filter_pushdown(optimized)?;
            }

            if self.config.join_reordering {
                optimized = if use_cost_based {
                    self.apply_cost_based_join_reordering(optimized)?
                } else {
                    self.apply_join_reordering(optimized)?
                };
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

        // Update cost model with actual execution feedback
        // Convert execution time to cost units (milliseconds)
        let actual_cost = record.execution_time.as_millis() as f64;
        self.cost_model
            .update_with_feedback(&record.algebra, actual_cost, record.cardinality);

        self.execution_records.push(record);
    }

    /// Get cost estimate for an algebra expression
    pub fn estimate_cost(&mut self, algebra: &Algebra) -> Result<CostEstimate> {
        self.cost_model.estimate_cost(algebra)
    }

    /// Clear cost model cache
    pub fn clear_cost_cache(&mut self) {
        self.cost_model.clear_cache();
    }

    /// Get optimizer statistics
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Apply filter pushdown optimization
    fn apply_filter_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                // First, apply advanced filter optimizations
                let optimized_conditions = self.optimize_filter_conditions(&condition)?;

                // Apply each condition separately for better pushdown opportunities
                let mut result_pattern = *pattern;
                for cond in optimized_conditions {
                    result_pattern = self.push_filter_down(result_pattern, &cond)?;
                }
                Ok(result_pattern)
            }
            Algebra::Join { left, right } => Ok(Algebra::Join {
                left: Box::new(self.apply_filter_pushdown(*left)?),
                right: Box::new(self.apply_filter_pushdown(*right)?),
            }),
            Algebra::Union { left, right } => Ok(Algebra::Union {
                left: Box::new(self.apply_filter_pushdown(*left)?),
                right: Box::new(self.apply_filter_pushdown(*right)?),
            }),
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
                let left_cost = self.estimate_simple_cost(&left);
                let right_cost = self.estimate_simple_cost(&right);

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
            Algebra::Union { left, right } => Ok(Algebra::Union {
                left: Box::new(self.apply_join_reordering(*left)?),
                right: Box::new(self.apply_join_reordering(*right)?),
            }),
            other => Ok(other),
        }
    }

    /// Apply cost-based join reordering using detailed cost model
    fn apply_cost_based_join_reordering(&mut self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                // Get detailed cost estimates for both sides
                let left_estimate = self.cost_model.estimate_cost(&left)?;
                let right_estimate = self.cost_model.estimate_cost(&right)?;

                // Choose join order based on total cost and cardinality
                let reordered = if self.should_reorder_join(&left_estimate, &right_estimate) {
                    Algebra::Join {
                        left: Box::new(self.apply_cost_based_join_reordering(*right)?),
                        right: Box::new(self.apply_cost_based_join_reordering(*left)?),
                    }
                } else {
                    Algebra::Join {
                        left: Box::new(self.apply_cost_based_join_reordering(*left)?),
                        right: Box::new(self.apply_cost_based_join_reordering(*right)?),
                    }
                };

                Ok(reordered)
            }
            Algebra::Union { left, right } => Ok(Algebra::Union {
                left: Box::new(self.apply_cost_based_join_reordering(*left)?),
                right: Box::new(self.apply_cost_based_join_reordering(*right)?),
            }),
            Algebra::Filter { pattern, condition } => Ok(Algebra::Filter {
                pattern: Box::new(self.apply_cost_based_join_reordering(*pattern)?),
                condition,
            }),
            Algebra::Project { pattern, variables } => Ok(Algebra::Project {
                pattern: Box::new(self.apply_cost_based_join_reordering(*pattern)?),
                variables,
            }),
            other => Ok(other),
        }
    }

    /// Determine if a join should be reordered based on cost estimates
    fn should_reorder_join(
        &self,
        left_estimate: &CostEstimate,
        right_estimate: &CostEstimate,
    ) -> bool {
        // Use multiple criteria for join reordering decision

        // Primary criterion: smaller relation should be build side (left)
        if left_estimate.cardinality > right_estimate.cardinality * 2 {
            return true;
        }

        // Secondary criterion: total cost consideration
        if left_estimate.total_cost > right_estimate.total_cost * 1.5 {
            return true;
        }

        // Tertiary criterion: selectivity (more selective should go first)
        if left_estimate.selectivity > right_estimate.selectivity * 2.0 {
            return true;
        }

        false
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
            Algebra::Join { left, right } => Ok(Algebra::Join {
                left: Box::new(self.apply_projection_pushdown(*left)?),
                right: Box::new(self.apply_projection_pushdown(*right)?),
            }),
            other => Ok(other),
        }
    }

    /// Push projection down into algebra tree
    fn push_projection_down(&self, algebra: Algebra, needed_vars: &[Variable]) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_vars = self.extract_variables(&left);
                let right_vars = self.extract_variables(&right);

                let left_needed: Vec<Variable> = needed_vars
                    .iter()
                    .filter(|v| left_vars.contains(v))
                    .cloned()
                    .collect();

                let right_needed: Vec<Variable> = needed_vars
                    .iter()
                    .filter(|v| right_vars.contains(v))
                    .cloned()
                    .collect();

                let left_projected =
                    if !left_needed.is_empty() && left_needed.len() < left_vars.len() {
                        Algebra::Project {
                            pattern: left,
                            variables: left_needed,
                        }
                    } else {
                        *left
                    };

                let right_projected =
                    if !right_needed.is_empty() && right_needed.len() < right_vars.len() {
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
                        Ok(Algebra::Bgp(vec![]))
                    }
                } else {
                    Ok(Algebra::Filter {
                        pattern: Box::new(self.apply_constant_folding(*pattern)?),
                        condition: folded_condition,
                    })
                }
            }
            Algebra::Join { left, right } => Ok(Algebra::Join {
                left: Box::new(self.apply_constant_folding(*left)?),
                right: Box::new(self.apply_constant_folding(*right)?),
            }),
            other => Ok(other),
        }
    }

    /// Apply dead code elimination
    fn apply_dead_code_elimination(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let used_vars = self.extract_variables(&pattern);
                let needed_vars: Vec<Variable> = variables
                    .into_iter()
                    .filter(|v| used_vars.contains(v))
                    .collect();

                if needed_vars.is_empty() {
                    Ok(Algebra::Bgp(vec![]))
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
                    (Algebra::Bgp(left_patterns), Algebra::Bgp(right_patterns))
                        if left_patterns.is_empty() || right_patterns.is_empty() =>
                    {
                        Ok(Algebra::Bgp(vec![]))
                    }
                    (Algebra::Bgp(patterns), _) if patterns.is_empty() => Ok(Algebra::Bgp(vec![])),
                    (_, Algebra::Bgp(patterns)) if patterns.is_empty() => Ok(Algebra::Bgp(vec![])),
                    _ => Ok(Algebra::Join {
                        left: Box::new(optimized_left),
                        right: Box::new(optimized_right),
                    }),
                }
            }
            other => Ok(other),
        }
    }

    /// Optimize filter conditions using advanced techniques
    fn optimize_filter_conditions(&self, condition: &Expression) -> Result<Vec<Expression>> {
        // Step 1: Factor AND conditions into separate filters
        let factored_conditions = Self::factor_and_conditions(condition);

        // Step 2: Remove redundant conditions
        let deduplicated = self.remove_redundant_filters(&factored_conditions);

        // Step 3: Order by estimated selectivity (most selective first)
        let mut ordered = deduplicated;
        ordered.sort_by(|a, b| {
            let selectivity_a = Self::estimate_filter_selectivity(a);
            let selectivity_b = Self::estimate_filter_selectivity(b);
            selectivity_a
                .partial_cmp(&selectivity_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ordered)
    }

    /// Factor AND conditions into separate expressions for better pushdown
    fn factor_and_conditions(expr: &Expression) -> Vec<Expression> {
        match expr {
            Expression::Binary { op, left, right } => {
                if let crate::algebra::BinaryOperator::And = op {
                    let mut conditions = Vec::new();
                    conditions.extend(Self::factor_and_conditions(left));
                    conditions.extend(Self::factor_and_conditions(right));
                    conditions
                } else {
                    vec![expr.clone()]
                }
            }
            _ => vec![expr.clone()],
        }
    }

    /// Remove redundant filter conditions
    fn remove_redundant_filters(&self, conditions: &[Expression]) -> Vec<Expression> {
        let mut result = Vec::new();
        let mut seen_hashes = HashSet::new();

        for condition in conditions {
            let hash = self.hash_expression(condition);
            if !seen_hashes.contains(&hash) {
                // Check for logical redundancy
                if !self.is_logically_redundant(condition, &result) {
                    result.push(condition.clone());
                    seen_hashes.insert(hash);
                }
            }
        }

        result
    }

    /// Estimate selectivity of a filter condition (lower is more selective)
    fn estimate_filter_selectivity(expr: &Expression) -> f64 {
        match expr {
            Expression::Binary { op, left, right } => {
                match op {
                    crate::algebra::BinaryOperator::Equal => {
                        // Equality is highly selective
                        match (left.as_ref(), right.as_ref()) {
                            (Expression::Variable(_), Expression::Literal(_))
                            | (Expression::Literal(_), Expression::Variable(_)) => 0.1, // Very selective
                            _ => 0.3,
                        }
                    }
                    crate::algebra::BinaryOperator::Less
                    | crate::algebra::BinaryOperator::LessEqual
                    | crate::algebra::BinaryOperator::Greater
                    | crate::algebra::BinaryOperator::GreaterEqual => 0.3, // Range conditions
                    crate::algebra::BinaryOperator::NotEqual => 0.9, // Usually not very selective
                    crate::algebra::BinaryOperator::And => {
                        // Combined selectivity (product for AND)
                        let left_sel = Self::estimate_filter_selectivity(left);
                        let right_sel = Self::estimate_filter_selectivity(right);
                        left_sel * right_sel
                    }
                    crate::algebra::BinaryOperator::Or => {
                        // Combined selectivity for OR (higher selectivity)
                        let left_sel = Self::estimate_filter_selectivity(left);
                        let right_sel = Self::estimate_filter_selectivity(right);
                        left_sel + right_sel - (left_sel * right_sel)
                    }
                    _ => 0.5, // Default moderate selectivity
                }
            }
            Expression::Function { name, args: _ } => {
                match name.as_str() {
                    "bound" => 0.8, // BOUND function is often not very selective
                    "isURI" | "isIRI" | "isLiteral" | "isBlank" => 0.4, // Type checks
                    "regex" => 0.6, // Regular expressions - moderate selectivity
                    "contains" | "strstarts" | "strends" => 0.5, // String functions
                    _ => 0.5,       // Default for other functions
                }
            }
            Expression::Unary {
                op: crate::algebra::UnaryOperator::Not,
                operand,
            } => {
                // Negation typically increases selectivity
                1.0 - Self::estimate_filter_selectivity(operand)
            }
            Expression::Unary { op: _, operand: _ } => 0.5,
            _ => 0.5, // Default moderate selectivity
        }
    }

    /// Check if a condition is logically redundant given existing conditions
    fn is_logically_redundant(&self, condition: &Expression, existing: &[Expression]) -> bool {
        // Simple redundancy check - could be enhanced with more sophisticated logic
        for existing_condition in existing {
            if Self::expressions_equivalent(condition, existing_condition) {
                return true;
            }

            // Check for simple cases like x = 1 AND x = 1
            if let (
                Expression::Binary {
                    op: op1,
                    left: left1,
                    right: right1,
                },
                Expression::Binary {
                    op: op2,
                    left: left2,
                    right: right2,
                },
            ) = (condition, existing_condition)
            {
                if op1 == op2
                    && Self::expressions_equivalent(left1, left2)
                    && Self::expressions_equivalent(right1, right2)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Check if two expressions are equivalent
    fn expressions_equivalent(expr1: &Expression, expr2: &Expression) -> bool {
        match (expr1, expr2) {
            (Expression::Variable(v1), Expression::Variable(v2)) => v1 == v2,
            (Expression::Literal(l1), Expression::Literal(l2)) => l1 == l2,
            (
                Expression::Binary {
                    op: op1,
                    left: left1,
                    right: right1,
                },
                Expression::Binary {
                    op: op2,
                    left: left2,
                    right: right2,
                },
            ) => {
                op1 == op2
                    && Self::expressions_equivalent(left1, left2)
                    && Self::expressions_equivalent(right1, right2)
            }
            (
                Expression::Unary {
                    op: op1,
                    operand: operand1,
                },
                Expression::Unary {
                    op: op2,
                    operand: operand2,
                },
            ) => op1 == op2 && Self::expressions_equivalent(operand1, operand2),
            (
                Expression::Function {
                    name: name1,
                    args: args1,
                },
                Expression::Function {
                    name: name2,
                    args: args2,
                },
            ) => {
                name1 == name2
                    && args1.len() == args2.len()
                    && args1
                        .iter()
                        .zip(args2.iter())
                        .all(|(a1, a2)| Self::expressions_equivalent(a1, a2))
            }
            _ => false,
        }
    }

    /// Hash an expression for deduplication
    fn hash_expression(&self, expr: &Expression) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{expr:?}").hash(&mut hasher);
        hasher.finish()
    }

    /// Extract variables from an algebra expression
    #[allow(clippy::only_used_in_recursion)]
    fn extract_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    let TriplePattern {
                        subject,
                        predicate,
                        object,
                    } = pattern;
                    if let crate::algebra::Term::Variable(v) = subject {
                        vars.insert(v.clone());
                    }
                    if let crate::algebra::Term::Variable(v) = predicate {
                        vars.insert(v.clone());
                    }
                    if let crate::algebra::Term::Variable(v) = object {
                        vars.insert(v.clone());
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
    #[allow(clippy::only_used_in_recursion)]
    fn extract_expression_variables(&self, expr: &Expression) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        match expr {
            Expression::Variable(v) => {
                vars.insert(v.clone());
            }
            Expression::Binary { left, right, .. } => {
                vars.extend(self.extract_expression_variables(left));
                vars.extend(self.extract_expression_variables(right));
            }
            Expression::Unary { operand, .. } => {
                vars.extend(self.extract_expression_variables(operand));
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    vars.extend(self.extract_expression_variables(arg));
                }
            }
            _ => {} // Other expression types
        }
        vars
    }

    /// Estimate execution cost for algebra (simple heuristic version)
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_simple_cost(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => {
                // BGP cost based on pattern count and estimated selectivity
                patterns.len() as f64 * 10.0
            }
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_simple_cost(left);
                let right_cost = self.estimate_simple_cost(right);
                left_cost * right_cost * 0.1 // Join selectivity factor
            }
            Algebra::Union { left, right } => {
                self.estimate_simple_cost(left) + self.estimate_simple_cost(right)
            }
            Algebra::Filter { pattern, .. } => {
                self.estimate_simple_cost(pattern) * 0.5 // Filter selectivity
            }
            _ => 1.0,
        }
    }

    /// Fold constants in expressions
    #[allow(clippy::only_used_in_recursion)]
    fn fold_expression_constants(&self, expr: Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { op, left, right } => {
                let folded_left = self.fold_expression_constants(*left)?;
                let folded_right = self.fold_expression_constants(*right)?;
                Ok(Expression::Binary {
                    op,
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                })
            }
            Expression::Unary { op, operand } => {
                let folded_operand = self.fold_expression_constants(*operand)?;
                Ok(Expression::Unary {
                    op,
                    operand: Box::new(folded_operand),
                })
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
        format!("{a:?}") == format!("{b:?}")
    }

    /// Hash an algebra expression for caching
    pub fn hash_algebra(&self, algebra: &Algebra) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{algebra:?}").hash(&mut hasher);
        hasher.finish()
    }

    /// Estimate query complexity for adaptive optimization
    fn estimate_query_complexity(&self, algebra: &Algebra) -> QueryComplexity {
        let mut complexity = QueryComplexity::default();
        self.analyze_complexity(algebra, &mut complexity);
        complexity
    }

    /// Recursively analyze query complexity
    fn analyze_complexity(&self, algebra: &Algebra, complexity: &mut QueryComplexity) {
        match algebra {
            Algebra::Bgp(patterns) => {
                complexity.triple_patterns += patterns.len();
            }
            Algebra::Join { left, right } | Algebra::Union { left, right } => {
                complexity.joins += 1;
                self.analyze_complexity(left, complexity);
                self.analyze_complexity(right, complexity);
            }
            Algebra::Filter { pattern, .. } => {
                complexity.filters += 1;
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::Extend { pattern, .. } => {
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::Project { pattern, .. } => {
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::Distinct { pattern } | Algebra::Reduced { pattern } => {
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::OrderBy { pattern, .. } => {
                complexity.ordering = true;
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::Slice { pattern, .. } => {
                self.analyze_complexity(pattern, complexity);
            }
            Algebra::Group { pattern, .. } => {
                complexity.grouping = true;
                self.analyze_complexity(pattern, complexity);
            }
            _ => {}
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }
}

/// Type alias for backwards compatibility
pub type QueryOptimizer = Optimizer;
