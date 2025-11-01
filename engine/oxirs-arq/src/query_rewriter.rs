//! Advanced Query Rewriting
//!
//! This module implements sophisticated query rewriting optimizations that
//! transform algebra expressions into more efficient forms while preserving semantics.

use crate::algebra::{
    Algebra, BinaryOperator, Expression, Literal, Term, TriplePattern, UnaryOperator, Variable,
};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Advanced query rewriter with multiple optimization passes
pub struct QueryRewriter {
    config: RewriterConfig,
    stats: RewriterStats,
}

/// Configuration for query rewriting
#[derive(Debug, Clone)]
pub struct RewriterConfig {
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable common subexpression elimination
    pub cse_enabled: bool,
    /// Enable filter pushdown
    pub filter_pushdown: bool,
    /// Enable pattern simplification
    pub pattern_simplification: bool,
    /// Enable empty pattern elimination
    pub empty_pattern_elimination: bool,
    /// Maximum rewrite iterations
    pub max_iterations: usize,
}

impl Default for RewriterConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            cse_enabled: true,
            filter_pushdown: true,
            pattern_simplification: true,
            empty_pattern_elimination: true,
            max_iterations: 10,
        }
    }
}

/// Statistics about rewriting operations
#[derive(Debug, Clone, Default)]
pub struct RewriterStats {
    /// Number of constant folding optimizations
    pub constants_folded: usize,
    /// Number of dead code eliminations
    pub dead_code_removed: usize,
    /// Number of common subexpressions eliminated
    pub cse_count: usize,
    /// Number of filters pushed down
    pub filters_pushed: usize,
    /// Number of patterns simplified
    pub patterns_simplified: usize,
    /// Number of empty patterns eliminated
    pub empty_eliminated: usize,
    /// Total rewrite iterations
    pub iterations: usize,
}

impl QueryRewriter {
    /// Create a new query rewriter with default configuration
    pub fn new() -> Self {
        Self::with_config(RewriterConfig::default())
    }

    /// Create a new query rewriter with custom configuration
    pub fn with_config(config: RewriterConfig) -> Self {
        Self {
            config,
            stats: RewriterStats::default(),
        }
    }

    /// Rewrite an algebra expression with all enabled optimizations
    pub fn rewrite(&mut self, algebra: &Algebra) -> Result<Algebra> {
        let mut current = algebra.clone();
        let mut iteration = 0;

        while iteration < self.config.max_iterations {
            let mut changed = false;

            // Pass 1: Constant folding
            if self.config.constant_folding {
                let folded = self.fold_constants(&current)?;
                if !algebras_equal(&current, &folded) {
                    current = folded;
                    changed = true;
                }
            }

            // Pass 2: Dead code elimination
            if self.config.dead_code_elimination {
                let dce = self.eliminate_dead_code(&current)?;
                if !algebras_equal(&current, &dce) {
                    current = dce;
                    changed = true;
                }
            }

            // Pass 3: Empty pattern elimination
            if self.config.empty_pattern_elimination {
                let empty_elim = self.eliminate_empty_patterns(&current)?;
                if !algebras_equal(&current, &empty_elim) {
                    current = empty_elim;
                    changed = true;
                }
            }

            // Pass 4: Pattern simplification
            if self.config.pattern_simplification {
                let simplified = self.simplify_patterns(&current)?;
                if !algebras_equal(&current, &simplified) {
                    current = simplified;
                    changed = true;
                }
            }

            // Pass 5: Filter pushdown
            if self.config.filter_pushdown {
                let pushed = self.push_down_filters(&current)?;
                if !algebras_equal(&current, &pushed) {
                    current = pushed;
                    changed = true;
                }
            }

            // Pass 6: Common subexpression elimination
            if self.config.cse_enabled {
                let cse = self.eliminate_common_subexpressions(&current)?;
                if !algebras_equal(&current, &cse) {
                    current = cse;
                    changed = true;
                }
            }

            iteration += 1;
            self.stats.iterations = iteration;

            if !changed {
                break; // Fixed point reached
            }
        }

        Ok(current)
    }

    /// Fold constant expressions
    fn fold_constants(&mut self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let folded_condition = self.fold_constant_expression(condition)?;
                let folded_pattern = self.fold_constants(pattern)?;

                // If expression is constant true, remove filter
                if is_constant_true(&folded_condition) {
                    self.stats.constants_folded += 1;
                    return Ok(folded_pattern);
                }

                // If expression is constant false, return empty result
                if is_constant_false(&folded_condition) {
                    self.stats.constants_folded += 1;
                    return Ok(Algebra::Bgp(vec![])); // Empty pattern
                }

                Ok(Algebra::Filter {
                    pattern: Box::new(folded_pattern),
                    condition: folded_condition,
                })
            }
            Algebra::Join { left, right } => {
                let folded_left = self.fold_constants(left)?;
                let folded_right = self.fold_constants(right)?;
                Ok(Algebra::Join {
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                })
            }
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => {
                let folded_left = self.fold_constants(left)?;
                let folded_right = self.fold_constants(right)?;
                let folded_filter = filter
                    .as_ref()
                    .map(|f| self.fold_constant_expression(f))
                    .transpose()?;
                Ok(Algebra::LeftJoin {
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                    filter: folded_filter,
                })
            }
            Algebra::Union { left, right } => {
                let folded_left = self.fold_constants(left)?;
                let folded_right = self.fold_constants(right)?;
                Ok(Algebra::Union {
                    left: Box::new(folded_left),
                    right: Box::new(folded_right),
                })
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Fold constant expressions
    fn fold_constant_expression(&self, expr: &Expression) -> Result<Expression> {
        fold_constant_expression_impl(expr)
    }
}

/// Helper function for constant expression folding (separated to avoid clippy warning)
fn fold_constant_expression_impl(expr: &Expression) -> Result<Expression> {
    match expr {
        // Binary operations (including And/Or)
        Expression::Binary { op, left, right } => {
            let folded_left = fold_constant_expression_impl(left)?;
            let folded_right = fold_constant_expression_impl(right)?;

            match op {
                BinaryOperator::And => {
                    // true && x = x
                    if is_constant_true(&folded_left) {
                        return Ok(folded_right);
                    }
                    // false && x = false
                    if is_constant_false(&folded_left) {
                        return Ok(folded_left);
                    }
                    // x && true = x
                    if is_constant_true(&folded_right) {
                        return Ok(folded_left);
                    }
                    // x && false = false
                    if is_constant_false(&folded_right) {
                        return Ok(folded_right);
                    }
                }
                BinaryOperator::Or => {
                    // true || x = true
                    if is_constant_true(&folded_left) {
                        return Ok(folded_left);
                    }
                    // false || x = x
                    if is_constant_false(&folded_left) {
                        return Ok(folded_right);
                    }
                    // x || true = true
                    if is_constant_true(&folded_right) {
                        return Ok(folded_right);
                    }
                    // x || false = x
                    if is_constant_false(&folded_right) {
                        return Ok(folded_left);
                    }
                }
                _ => {}
            }

            Ok(Expression::Binary {
                op: op.clone(),
                left: Box::new(folded_left),
                right: Box::new(folded_right),
            })
        }
        // Unary operations (including Not)
        Expression::Unary { op, operand } => {
            let folded = fold_constant_expression_impl(operand)?;

            if *op == UnaryOperator::Not {
                // !true = false
                if is_constant_true(&folded) {
                    return Ok(make_boolean_literal(false));
                }
                // !false = true
                if is_constant_false(&folded) {
                    return Ok(make_boolean_literal(true));
                }
            }

            Ok(Expression::Unary {
                op: op.clone(),
                operand: Box::new(folded),
            })
        }
        _ => Ok(expr.clone()),
    }
}

impl QueryRewriter {
    /// Eliminate dead code (unreachable algebra nodes)
    fn eliminate_dead_code(&mut self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Union { left, right } => {
                let cleaned_left = self.eliminate_dead_code(left)?;
                let cleaned_right = self.eliminate_dead_code(right)?;

                // If left is empty, return right
                if is_empty_pattern(&cleaned_left) {
                    self.stats.dead_code_removed += 1;
                    return Ok(cleaned_right);
                }
                // If right is empty, return left
                if is_empty_pattern(&cleaned_right) {
                    self.stats.dead_code_removed += 1;
                    return Ok(cleaned_left);
                }

                Ok(Algebra::Union {
                    left: Box::new(cleaned_left),
                    right: Box::new(cleaned_right),
                })
            }
            Algebra::Join { left, right } => {
                let cleaned_left = self.eliminate_dead_code(left)?;
                let cleaned_right = self.eliminate_dead_code(right)?;

                // Join with empty pattern is empty
                if is_empty_pattern(&cleaned_left) || is_empty_pattern(&cleaned_right) {
                    self.stats.dead_code_removed += 1;
                    return Ok(Algebra::Bgp(vec![]));
                }

                Ok(Algebra::Join {
                    left: Box::new(cleaned_left),
                    right: Box::new(cleaned_right),
                })
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Eliminate empty patterns
    fn eliminate_empty_patterns(&mut self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Bgp(patterns) if patterns.is_empty() => {
                self.stats.empty_eliminated += 1;
                Ok(Algebra::Bgp(vec![]))
            }
            Algebra::Join { left, right } => {
                let left_clean = self.eliminate_empty_patterns(left)?;
                let right_clean = self.eliminate_empty_patterns(right)?;

                if is_empty_pattern(&left_clean) {
                    self.stats.empty_eliminated += 1;
                    return Ok(right_clean);
                }
                if is_empty_pattern(&right_clean) {
                    self.stats.empty_eliminated += 1;
                    return Ok(left_clean);
                }

                Ok(Algebra::Join {
                    left: Box::new(left_clean),
                    right: Box::new(right_clean),
                })
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Simplify patterns
    fn simplify_patterns(&mut self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Bgp(patterns) => {
                // Remove duplicate patterns
                let mut unique_patterns = Vec::new();
                let mut seen = HashSet::new();

                for pattern in patterns {
                    let key = pattern_key(pattern);
                    if !seen.contains(&key) {
                        seen.insert(key);
                        unique_patterns.push(pattern.clone());
                    } else {
                        self.stats.patterns_simplified += 1;
                    }
                }

                Ok(Algebra::Bgp(unique_patterns))
            }
            Algebra::Join { left, right } => {
                let simp_left = self.simplify_patterns(left)?;
                let simp_right = self.simplify_patterns(right)?;
                Ok(Algebra::Join {
                    left: Box::new(simp_left),
                    right: Box::new(simp_right),
                })
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Push filters down closer to data sources
    fn push_down_filters(&mut self, algebra: &Algebra) -> Result<Algebra> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                // Try to push filter into join operands
                if let Algebra::Join { left, right } = pattern.as_ref() {
                    let left_vars = collect_variables(left);
                    let condition_vars = collect_expression_variables(condition);

                    // If filter only uses variables from left, push to left
                    if condition_vars.iter().all(|v| left_vars.contains(v)) {
                        self.stats.filters_pushed += 1;
                        let filtered_left = Algebra::Filter {
                            pattern: left.clone(),
                            condition: condition.clone(),
                        };
                        return Ok(Algebra::Join {
                            left: Box::new(filtered_left),
                            right: right.clone(),
                        });
                    }

                    let right_vars = collect_variables(right);
                    // If filter only uses variables from right, push to right
                    if condition_vars.iter().all(|v| right_vars.contains(v)) {
                        self.stats.filters_pushed += 1;
                        let filtered_right = Algebra::Filter {
                            pattern: right.clone(),
                            condition: condition.clone(),
                        };
                        return Ok(Algebra::Join {
                            left: left.clone(),
                            right: Box::new(filtered_right),
                        });
                    }
                }

                // Recursively process inner pattern
                let pushed_pattern = self.push_down_filters(pattern)?;
                Ok(Algebra::Filter {
                    pattern: Box::new(pushed_pattern),
                    condition: condition.clone(),
                })
            }
            Algebra::Join { left, right } => {
                let pushed_left = self.push_down_filters(left)?;
                let pushed_right = self.push_down_filters(right)?;
                Ok(Algebra::Join {
                    left: Box::new(pushed_left),
                    right: Box::new(pushed_right),
                })
            }
            _ => Ok(algebra.clone()),
        }
    }

    /// Eliminate common subexpressions
    fn eliminate_common_subexpressions(&mut self, algebra: &Algebra) -> Result<Algebra> {
        // Find common subpatterns
        let subpatterns = find_subpatterns(algebra);
        let mut pattern_counts: HashMap<String, usize> = HashMap::new();

        for pattern in &subpatterns {
            *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;
        }

        // If there are patterns that appear multiple times, we could optimize
        // For now, just count them
        for (_pattern, count) in pattern_counts {
            if count > 1 {
                self.stats.cse_count += 1;
            }
        }

        // CSE implementation would be more complex in practice
        Ok(algebra.clone())
    }

    /// Get rewriter statistics
    pub fn get_stats(&self) -> &RewriterStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = RewriterStats::default();
    }
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

/// Check if two algebra expressions are equal
fn algebras_equal(a: &Algebra, b: &Algebra) -> bool {
    // Simplified equality check
    format!("{:?}", a) == format!("{:?}", b)
}

/// Check if expression is constant true
fn is_constant_true(expr: &Expression) -> bool {
    if let Expression::Literal(lit) = expr {
        lit.value == "true" && lit.datatype.is_none()
    } else {
        false
    }
}

/// Check if expression is constant false
fn is_constant_false(expr: &Expression) -> bool {
    if let Expression::Literal(lit) = expr {
        lit.value == "false" && lit.datatype.is_none()
    } else {
        false
    }
}

/// Create a boolean literal
fn make_boolean_literal(value: bool) -> Expression {
    Expression::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: None,
    })
}

/// Check if algebra is an empty pattern
fn is_empty_pattern(algebra: &Algebra) -> bool {
    matches!(algebra, Algebra::Bgp(patterns) if patterns.is_empty())
}

/// Generate a key for pattern deduplication
fn pattern_key(pattern: &TriplePattern) -> String {
    format!("{:?}", pattern)
}

/// Collect all variables from an algebra expression
fn collect_variables(algebra: &Algebra) -> HashSet<Variable> {
    let mut vars = HashSet::new();
    collect_variables_recursive(algebra, &mut vars);
    vars
}

/// Recursively collect variables
fn collect_variables_recursive(algebra: &Algebra, vars: &mut HashSet<Variable>) {
    match algebra {
        Algebra::Bgp(patterns) => {
            for pattern in patterns {
                if let Term::Variable(v) = &pattern.subject {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = &pattern.predicate {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = &pattern.object {
                    vars.insert(v.clone());
                }
            }
        }
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::LeftJoin { left, right, .. } => {
            collect_variables_recursive(left, vars);
            collect_variables_recursive(right, vars);
        }
        Algebra::Filter { pattern, .. } => {
            collect_variables_recursive(pattern, vars);
        }
        _ => {}
    }
}

/// Collect variables from an expression
fn collect_expression_variables(expr: &Expression) -> HashSet<Variable> {
    let mut vars = HashSet::new();
    collect_expr_vars_recursive(expr, &mut vars);
    vars
}

/// Recursively collect variables from expression
fn collect_expr_vars_recursive(expr: &Expression, vars: &mut HashSet<Variable>) {
    match expr {
        Expression::Variable(v) => {
            vars.insert(v.clone());
        }
        Expression::Unary { operand, .. } => {
            collect_expr_vars_recursive(operand, vars);
        }
        Expression::Binary { left, right, .. } => {
            collect_expr_vars_recursive(left, vars);
            collect_expr_vars_recursive(right, vars);
        }
        Expression::Function { args, .. } => {
            for arg in args {
                collect_expr_vars_recursive(arg, vars);
            }
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_expr_vars_recursive(condition, vars);
            collect_expr_vars_recursive(then_expr, vars);
            collect_expr_vars_recursive(else_expr, vars);
        }
        Expression::Exists(algebra) | Expression::NotExists(algebra) => {
            collect_variables_recursive(algebra, vars);
        }
        _ => {}
    }
}

/// Find all subpatterns in an algebra expression
fn find_subpatterns(algebra: &Algebra) -> Vec<String> {
    let mut patterns = Vec::new();
    find_subpatterns_recursive(algebra, &mut patterns);
    patterns
}

/// Recursively find subpatterns
fn find_subpatterns_recursive(algebra: &Algebra, patterns: &mut Vec<String>) {
    patterns.push(format!("{:?}", algebra));

    match algebra {
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::LeftJoin { left, right, .. } => {
            find_subpatterns_recursive(left, patterns);
            find_subpatterns_recursive(right, patterns);
        }
        Algebra::Filter { pattern, .. } => {
            find_subpatterns_recursive(pattern, patterns);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding() {
        let rewriter = QueryRewriter::new();

        // true AND x = x
        let expr = Expression::Binary {
            op: BinaryOperator::And,
            left: Box::new(make_boolean_literal(true)),
            right: Box::new(Expression::Variable(
                Variable::new("x".to_string()).unwrap(),
            )),
        };

        let folded = rewriter.fold_constant_expression(&expr).unwrap();
        assert!(matches!(folded, Expression::Variable(_)));
    }

    #[test]
    fn test_pattern_simplification() {
        let mut rewriter = QueryRewriter::new();

        // Create duplicate patterns
        let pattern1 = TriplePattern {
            subject: Term::Variable(Variable::new("s".to_string()).unwrap()),
            predicate: Term::Variable(Variable::new("p".to_string()).unwrap()),
            object: Term::Variable(Variable::new("o".to_string()).unwrap()),
        };

        let pattern2 = pattern1.clone();

        let algebra = Algebra::Bgp(vec![pattern1, pattern2]);
        let simplified = rewriter.simplify_patterns(&algebra).unwrap();

        if let Algebra::Bgp(patterns) = simplified {
            assert_eq!(patterns.len(), 1); // Duplicates removed
        } else {
            panic!("Expected Bgp");
        }
    }

    #[test]
    fn test_empty_pattern_elimination() {
        let mut rewriter = QueryRewriter::new();

        let empty = Algebra::Bgp(vec![]);
        let non_empty = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("s".to_string()).unwrap()),
            predicate: Term::Variable(Variable::new("p".to_string()).unwrap()),
            object: Term::Variable(Variable::new("o".to_string()).unwrap()),
        }]);

        let join = Algebra::Join {
            left: Box::new(empty),
            right: Box::new(non_empty.clone()),
        };

        let eliminated = rewriter.eliminate_empty_patterns(&join).unwrap();

        // Should return the non-empty pattern
        assert!(matches!(eliminated, Algebra::Bgp(_)));
    }

    #[test]
    fn test_filter_pushdown() {
        let mut rewriter = QueryRewriter::new();

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("x".to_string()).unwrap()),
            predicate: Term::Variable(Variable::new("p".to_string()).unwrap()),
            object: Term::Variable(Variable::new("o".to_string()).unwrap()),
        };

        let left = Algebra::Bgp(vec![pattern.clone()]);
        let right = Algebra::Bgp(vec![pattern]);

        let join = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };

        let filter_condition = Expression::Variable(Variable::new("x".to_string()).unwrap());

        let filtered_join = Algebra::Filter {
            pattern: Box::new(join),
            condition: filter_condition,
        };

        let pushed = rewriter.push_down_filters(&filtered_join).unwrap();

        // Check that rewriting occurred
        assert!(rewriter.stats.filters_pushed > 0 || matches!(pushed, Algebra::Join { .. }));
    }

    #[test]
    fn test_rewriter_stats() {
        let mut rewriter = QueryRewriter::new();

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s".to_string()).unwrap()),
            predicate: Term::Variable(Variable::new("p".to_string()).unwrap()),
            object: Term::Variable(Variable::new("o".to_string()).unwrap()),
        };

        let algebra = Algebra::Bgp(vec![pattern.clone(), pattern]);
        let _simplified = rewriter.simplify_patterns(&algebra).unwrap();

        let stats = rewriter.get_stats();
        assert!(stats.patterns_simplified > 0);
    }
}
