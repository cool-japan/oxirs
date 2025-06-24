//! Query Optimization Module
//!
//! This module provides query optimization capabilities including
//! rule-based and cost-based optimization passes.

use crate::algebra::{Algebra, Expression, TriplePattern, Term, Variable, BinaryOperator, UnaryOperator};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};

/// Query optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable join reordering
    pub join_reordering: bool,
    /// Enable filter pushdown
    pub filter_pushdown: bool,
    /// Enable projection pushdown
    pub projection_pushdown: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable cost-based optimization
    pub cost_based: bool,
    /// Maximum optimization passes
    pub max_passes: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            join_reordering: true,
            filter_pushdown: true,
            projection_pushdown: true,
            constant_folding: true,
            dead_code_elimination: true,
            cost_based: true,
            max_passes: 10,
        }
    }
}

/// Statistics for cost-based optimization
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Variable selectivity estimates
    pub variable_selectivity: HashMap<Variable, f64>,
    /// Triple pattern cardinality estimates
    pub pattern_cardinality: HashMap<String, usize>,
    /// Join selectivity estimates
    pub join_selectivity: HashMap<String, f64>,
}

impl Statistics {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Estimate cardinality of a triple pattern
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        let pattern_key = format!("{}", pattern);
        self.pattern_cardinality.get(&pattern_key).copied().unwrap_or(1000)
    }
    
    /// Estimate selectivity of a variable
    pub fn estimate_variable_selectivity(&self, var: &Variable) -> f64 {
        self.variable_selectivity.get(var).copied().unwrap_or(0.1)
    }
    
    /// Estimate cost of an algebra expression
    pub fn estimate_cost(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => {
                if patterns.is_empty() {
                    1.0 // Empty BGP has very low cost
                } else {
                    patterns.iter()
                        .map(|p| self.estimate_pattern_cardinality(p) as f64)
                        .sum()
                }
            }
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                // Join has base cost plus multiplicative cost
                10.0 + left_cost * right_cost * 0.1 // Base join cost + selectivity
            }
            Algebra::Union { left, right } => {
                self.estimate_cost(left) + self.estimate_cost(right)
            }
            Algebra::Filter { pattern, .. } => {
                // Filter has higher cost than the pattern alone
                self.estimate_cost(pattern) + 50.0 // Add filter overhead
            }
            Algebra::Zero => 0.0, // Zero has zero cost
            Algebra::Table => 1.0, // Table has minimal cost
            _ => 100.0, // Default cost
        }
    }
}

/// Query optimizer
pub struct QueryOptimizer {
    config: OptimizerConfig,
    statistics: Statistics,
}

impl QueryOptimizer {
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            statistics: Statistics::new(),
        }
    }
    
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            config,
            statistics: Statistics::new(),
        }
    }
    
    pub fn with_statistics(mut self, stats: Statistics) -> Self {
        self.statistics = stats;
        self
    }
    
    /// Optimize an algebra expression
    pub fn optimize(&self, algebra: Algebra) -> Result<Algebra> {
        let mut current = algebra;
        let mut pass = 0;
        
        while pass < self.config.max_passes {
            let before_cost = self.statistics.estimate_cost(&current);
            let optimized = self.apply_optimization_passes(current.clone())?;
            let after_cost = self.statistics.estimate_cost(&optimized);
            
            // Stop if no improvement
            if after_cost >= before_cost {
                break;
            }
            
            current = optimized;
            pass += 1;
        }
        Ok(current)
    }
    
    fn apply_optimization_passes(&self, algebra: Algebra) -> Result<Algebra> {
        let mut result = algebra;
        
        // Apply rule-based optimizations
        if self.config.constant_folding {
            result = self.constant_folding(result)?;
        }
        
        if self.config.dead_code_elimination {
            result = self.dead_code_elimination(result)?;
        }
        
        if self.config.filter_pushdown {
            result = self.filter_pushdown(result)?;
        }
        
        if self.config.projection_pushdown {
            result = self.projection_pushdown(result)?;
        }
        
        if self.config.join_reordering {
            result = self.join_reordering(result)?;
        }
        
        Ok(result)
    }
    
    /// Constant folding optimization
    fn constant_folding(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let optimized_pattern = self.constant_folding(*pattern)?;
                let optimized_condition = self.fold_expression(condition)?;
                
                // Check if condition is always true/false after folding
                if self.is_always_true(&optimized_condition) {
                    return Ok(optimized_pattern);
                }
                
                if self.is_always_false(&optimized_condition) {
                    return Ok(Algebra::Zero);
                }
                
                Ok(Algebra::Filter {
                    pattern: Box::new(optimized_pattern),
                    condition: optimized_condition,
                })
            }
            Algebra::Join { left, right } => {
                let optimized_left = self.constant_folding(*left)?;
                let optimized_right = self.constant_folding(*right)?;
                
                // Join with Zero is Zero
                if matches!(optimized_left, Algebra::Zero) || matches!(optimized_right, Algebra::Zero) {
                    return Ok(Algebra::Zero);
                }
                
                // Join with Table is the other pattern
                if matches!(optimized_left, Algebra::Table) {
                    return Ok(optimized_right);
                }
                if matches!(optimized_right, Algebra::Table) {
                    return Ok(optimized_left);
                }
                
                Ok(Algebra::Join {
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                })
            }
            Algebra::Union { left, right } => {
                let optimized_left = self.constant_folding(*left)?;
                let optimized_right = self.constant_folding(*right)?;
                
                // Union with Zero is the other pattern
                if matches!(optimized_left, Algebra::Zero) {
                    return Ok(optimized_right);
                }
                if matches!(optimized_right, Algebra::Zero) {
                    return Ok(optimized_left);
                }
                
                Ok(Algebra::Union {
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                })
            }
            _ => self.apply_to_children(algebra, |child| self.constant_folding(child))
        }
    }
    
    /// Dead code elimination
    fn dead_code_elimination(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                let optimized_pattern = self.dead_code_elimination(*pattern)?;
                
                // Remove projection if no variables are projected
                if variables.is_empty() {
                    return Ok(Algebra::Table);
                }
                
                // Remove projection if all variables from pattern are projected
                let pattern_vars = optimized_pattern.variables();
                if variables.len() >= pattern_vars.len() && 
                   pattern_vars.iter().all(|v| variables.contains(v)) {
                    return Ok(optimized_pattern);
                }
                
                Ok(Algebra::Project {
                    pattern: Box::new(optimized_pattern),
                    variables,
                })
            }
            Algebra::Slice { pattern, offset, limit } => {
                let optimized_pattern = self.dead_code_elimination(*pattern)?;
                
                // If limit is 0, return zero
                if limit == Some(0) {
                    return Ok(Algebra::Zero);
                }
                
                Ok(Algebra::Slice {
                    pattern: Box::new(optimized_pattern),
                    offset,
                    limit,
                })
            }
            _ => self.apply_to_children(algebra, |child| self.dead_code_elimination(child)),
        }
    }
    
    /// Filter pushdown optimization
    fn filter_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                match *pattern {
                    Algebra::Join { left, right } => {
                        let condition_vars = self.get_expression_variables(&condition);
                        let left_vars: HashSet<_> = left.variables().into_iter().collect();
                        let right_vars: HashSet<_> = right.variables().into_iter().collect();
                        
                        // If filter only uses left variables, push to left
                        if condition_vars.is_subset(&left_vars) {
                            let filtered_left = Algebra::Filter {
                                pattern: left,
                                condition: condition.clone(),
                            };
                            return Ok(Algebra::Join {
                                left: Box::new(self.filter_pushdown(filtered_left)?),
                                right: Box::new(self.filter_pushdown(*right)?),
                            });
                        }
                        
                        // If filter only uses right variables, push to right
                        if condition_vars.is_subset(&right_vars) {
                            let filtered_right = Algebra::Filter {
                                pattern: right,
                                condition: condition.clone(),
                            };
                            return Ok(Algebra::Join {
                                left: Box::new(self.filter_pushdown(*left)?),
                                right: Box::new(self.filter_pushdown(filtered_right)?),
                            });
                        }
                        
                        // Can't push down, keep as is
                        Ok(Algebra::Filter {
                            pattern: Box::new(Algebra::Join {
                                left: Box::new(self.filter_pushdown(*left)?),
                                right: Box::new(self.filter_pushdown(*right)?),
                            }),
                            condition,
                        })
                    }
                    Algebra::Union { left, right } => {
                        // Push filter into both sides of union
                        let filtered_left = Algebra::Filter {
                            pattern: left,
                            condition: condition.clone(),
                        };
                        let filtered_right = Algebra::Filter {
                            pattern: right,
                            condition,
                        };
                        Ok(Algebra::Union {
                            left: Box::new(self.filter_pushdown(filtered_left)?),
                            right: Box::new(self.filter_pushdown(filtered_right)?),
                        })
                    }
                    _ => {
                        // Can't push down further
                        Ok(Algebra::Filter {
                            pattern: Box::new(self.filter_pushdown(*pattern)?),
                            condition,
                        })
                    }
                }
            }
            _ => self.apply_to_children(algebra, |child| self.filter_pushdown(child)),
        }
    }
    
    /// Projection pushdown optimization
    fn projection_pushdown(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Project { pattern, variables } => {
                match *pattern {
                    Algebra::Join { left, right } => {
                        let left_vars: HashSet<_> = left.variables().into_iter().collect();
                        let right_vars: HashSet<_> = right.variables().into_iter().collect();
                        let needed_vars: HashSet<_> = variables.into_iter().collect();
                        
                        let left_needed: Vec<_> = needed_vars.intersection(&left_vars).cloned().collect();
                        let right_needed: Vec<_> = needed_vars.intersection(&right_vars).cloned().collect();
                        
                        let projected_left = if left_needed.len() < left_vars.len() {
                            Algebra::Project {
                                pattern: left,
                                variables: left_needed,
                            }
                        } else {
                            *left
                        };
                        
                        let projected_right = if right_needed.len() < right_vars.len() {
                            Algebra::Project {
                                pattern: right,
                                variables: right_needed,
                            }
                        } else {
                            *right
                        };
                        
                        Ok(Algebra::Join {
                            left: Box::new(self.projection_pushdown(projected_left)?),
                            right: Box::new(self.projection_pushdown(projected_right)?),
                        })
                    }
                    _ => {
                        let needed_vars: HashSet<_> = variables.into_iter().collect();
                        let pattern_vars: HashSet<_> = pattern.variables().into_iter().collect();
                        
                        // Only keep variables that are actually used
                        let filtered_vars: Vec<_> = needed_vars.intersection(&pattern_vars).cloned().collect();
                        
                        if filtered_vars.len() < pattern_vars.len() {
                            Ok(Algebra::Project {
                                pattern: Box::new(self.projection_pushdown(*pattern)?),
                                variables: filtered_vars,
                            })
                        } else {
                            Ok(self.projection_pushdown(*pattern)?)
                        }
                    }
                }
            }
            _ => self.apply_to_children(algebra, |child| self.projection_pushdown(child)),
        }
    }
    
    /// Join reordering optimization
    fn join_reordering(&self, algebra: Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_cost = self.statistics.estimate_cost(&left);
                let right_cost = self.statistics.estimate_cost(&right);
                
                // Reorder if right is cheaper than left
                if right_cost < left_cost {
                    Ok(Algebra::Join {
                        left: Box::new(self.join_reordering(*right)?),
                        right: Box::new(self.join_reordering(*left)?),
                    })
                } else {
                    Ok(Algebra::Join {
                        left: Box::new(self.join_reordering(*left)?),
                        right: Box::new(self.join_reordering(*right)?),
                    })
                }
            }
            _ => self.apply_to_children(algebra, |child| self.join_reordering(child)),
        }
    }
    
    /// Apply optimization function to all children
    fn apply_to_children<F>(&self, algebra: Algebra, optimize_fn: F) -> Result<Algebra>
    where
        F: Fn(Algebra) -> Result<Algebra>,
    {
        match algebra {
            Algebra::Join { left, right } => Ok(Algebra::Join {
                left: Box::new(optimize_fn(*left)?),
                right: Box::new(optimize_fn(*right)?),
            }),
            Algebra::Union { left, right } => Ok(Algebra::Union {
                left: Box::new(optimize_fn(*left)?),
                right: Box::new(optimize_fn(*right)?),
            }),
            Algebra::Filter { pattern, condition } => Ok(Algebra::Filter {
                pattern: Box::new(optimize_fn(*pattern)?),
                condition,
            }),
            Algebra::Project { pattern, variables } => Ok(Algebra::Project {
                pattern: Box::new(optimize_fn(*pattern)?),
                variables,
            }),
            Algebra::Distinct { pattern } => Ok(Algebra::Distinct {
                pattern: Box::new(optimize_fn(*pattern)?),
            }),
            _ => Ok(algebra), // No children to optimize
        }
    }
    
    /// Fold constant expressions
    fn fold_expression(&self, expr: Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { op, left, right } => {
                let folded_left = self.fold_expression(*left)?;
                let folded_right = self.fold_expression(*right)?;
                
                // Try to evaluate constant expressions
                if let (Expression::Literal(l), Expression::Literal(r)) = (&folded_left, &folded_right) {
                    if let Ok(result) = self.evaluate_constant_binary(&op, l, r) {
                        return Ok(Expression::Literal(result));
                    }
                }
                
                Ok(Expression::Binary {
                    op,
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                })
            }
            Expression::Unary { op, expr } => {
                let folded_expr = self.fold_expression(*expr)?;
                
                if let Expression::Literal(lit) = &folded_expr {
                    if let Ok(result) = self.evaluate_constant_unary(&op, lit) {
                        return Ok(Expression::Literal(result));
                    }
                }
                
                Ok(Expression::Unary {
                    op,
                    expr: Box::new(folded_expr),
                })
            }
            _ => Ok(expr),
        }
    }
    
    /// Check if expression is always true
    fn is_always_true(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Literal(lit) => lit.value == "true",
            Expression::Binary { op: BinaryOperator::Equal, left, right } => {
                // Check if it's comparing two identical literals
                if let (Expression::Literal(l), Expression::Literal(r)) = (left.as_ref(), right.as_ref()) {
                    l.value == r.value
                } else {
                    false
                }
            }
            _ => false,
        }
    }
    
    /// Check if expression is always false
    fn is_always_false(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Literal(lit) => lit.value == "false",
            _ => false,
        }
    }
    
    /// Get variables used in expression
    fn get_expression_variables(&self, expr: &Expression) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        self.collect_expression_variables(expr, &mut vars);
        vars
    }
    
    fn collect_expression_variables(&self, expr: &Expression, vars: &mut HashSet<Variable>) {
        match expr {
            Expression::Variable(var) => {
                vars.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.collect_expression_variables(left, vars);
                self.collect_expression_variables(right, vars);
            }
            Expression::Unary { expr, .. } => {
                self.collect_expression_variables(expr, vars);
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.collect_expression_variables(arg, vars);
                }
            }
            Expression::Conditional { condition, then_expr, else_expr } => {
                self.collect_expression_variables(condition, vars);
                self.collect_expression_variables(then_expr, vars);
                self.collect_expression_variables(else_expr, vars);
            }
            Expression::Bound(var) => {
                vars.insert(var.clone());
            }
            Expression::Exists(algebra) | Expression::NotExists(algebra) => {
                for var in algebra.variables() {
                    vars.insert(var);
                }
            }
            _ => {}
        }
    }
    
    /// Evaluate constant binary expressions
    fn evaluate_constant_binary(&self, op: &BinaryOperator, left: &crate::algebra::Literal, right: &crate::algebra::Literal) -> Result<crate::algebra::Literal> {
        match op {
            BinaryOperator::Add => {
                if let (Ok(l), Ok(r)) = (left.value.parse::<f64>(), right.value.parse::<f64>()) {
                    Ok(crate::algebra::Literal {
                        value: (l + r).to_string(),
                        language: None,
                        datatype: None,
                    })
                } else {
                    Err(anyhow!("Cannot add non-numeric literals"))
                }
            }
            BinaryOperator::Equal => {
                let result = left.value == right.value;
                Ok(crate::algebra::Literal {
                    value: result.to_string(),
                    language: None,
                    datatype: None,
                })
            }
            _ => Err(anyhow!("Binary operation not supported for constant folding")),
        }
    }
    
    /// Evaluate constant unary expressions
    fn evaluate_constant_unary(&self, op: &UnaryOperator, expr: &crate::algebra::Literal) -> Result<crate::algebra::Literal> {
        match op {
            UnaryOperator::Not => {
                let val = expr.value != "false" && !expr.value.is_empty();
                Ok(crate::algebra::Literal {
                    value: (!val).to_string(),
                    language: None,
                    datatype: None,
                })
            }
            UnaryOperator::IsLiteral => {
                Ok(crate::algebra::Literal {
                    value: "true".to_string(),
                    language: None,
                    datatype: None,
                })
            }
            _ => Err(anyhow!("Unary operation not supported for constant folding")),
        }
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization rule trait
pub trait OptimizationRule {
    fn name(&self) -> &str;
    fn apply(&self, algebra: Algebra) -> Result<Algebra>;
    fn applicable(&self, algebra: &Algebra) -> bool;
}

/// Collection of optimization rules
pub struct RuleSet {
    rules: Vec<Box<dyn OptimizationRule>>,
}

impl RuleSet {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
        }
    }
    
    pub fn add_rule<R: OptimizationRule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }
    
    pub fn apply_rules(&self, algebra: Algebra) -> Result<Algebra> {
        let mut result = algebra;
        
        for rule in &self.rules {
            if rule.applicable(&result) {
                result = rule.apply(result)?;
            }
        }
        
        Ok(result)
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, TriplePattern, Term, Iri, Expression, BinaryOperator};
    
    #[test]
    fn test_constant_folding() {
        let optimizer = QueryOptimizer::new();
        
        // Create a filter with always-true condition
        let pattern = Algebra::Bgp(vec![
            TriplePattern::new(
                Term::Variable("s".to_string()),
                Term::Variable("p".to_string()),
                Term::Variable("o".to_string()),
            )
        ]);
        
        let always_true = Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Literal(crate::algebra::Literal {
                value: "1".to_string(),
                language: None,
                datatype: None,
            })),
            right: Box::new(Expression::Literal(crate::algebra::Literal {
                value: "1".to_string(),
                language: None,
                datatype: None,
            })),
        };
        
        let filter = Algebra::Filter {
            pattern: Box::new(pattern.clone()),
            condition: always_true,
        };
        
        let optimized = optimizer.optimize(filter).unwrap();
        
        // Should remove the filter since condition is always true
        match optimized {
            Algebra::Bgp(_) => {}, // Expected
            _ => panic!("Expected BGP after optimizing always-true filter"),
        }
    }
    
    #[test]
    fn test_join_with_zero() {
        let optimizer = QueryOptimizer::new();
        
        let left = Algebra::Bgp(vec![]);
        let right = Algebra::Zero;
        
        let join = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };
        
        let optimized = optimizer.optimize(join).unwrap();
        
        // Join with Zero should become Zero
        assert!(matches!(optimized, Algebra::Zero));
    }
}
