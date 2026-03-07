//! Additional Optimization Passes for the SPARQL Query Optimizer
//!
//! This module defines the `OptimizationPass` trait and three concrete passes:
//!
//! | Pass | What it does |
//! |---|---|
//! | `ConstantFoldingPass` | Evaluates constant sub-expressions and removes trivially true/false FILTERs |
//! | `UnusedVariableEliminationPass` | Strips PROJECT variables that are never produced by the inner pattern |
//! | `RedundantJoinEliminationPass` | Removes join branches that always produce empty results |
//!
//! The passes are composed by `OptimizationPipeline`, which iterates until no
//! further changes occur or `max_iterations` is reached.

use crate::algebra::{Algebra, BinaryOperator, Expression, Term, TriplePattern, Variable};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// OptimizationPass trait
// ---------------------------------------------------------------------------

/// A single rewriting pass over a `QueryAlgebra`.
///
/// Implementations should apply the rewriting in-place through the `&mut`
/// reference and return `true` if at least one node was changed, `false`
/// otherwise.  Returning `false` signals convergence to the pipeline.
pub trait OptimizationPass: Send + Sync {
    /// A short, human-readable identifier for this pass (e.g. `"constant_folding"`).
    fn name(&self) -> &str;

    /// Apply the pass to `algebra`.  Returns `true` when the algebra was
    /// modified so the pipeline can decide whether to run another iteration.
    fn apply(&self, algebra: &mut Algebra) -> bool;
}

// ---------------------------------------------------------------------------
// Helper: collect all variables produced by an algebra expression
// ---------------------------------------------------------------------------

fn produced_variables(algebra: &Algebra) -> HashSet<Variable> {
    let mut vars = HashSet::new();
    collect_produced(algebra, &mut vars);
    vars
}

fn collect_produced(algebra: &Algebra, vars: &mut HashSet<Variable>) {
    match algebra {
        Algebra::Bgp(patterns) => {
            for p in patterns {
                add_pattern_vars(p, vars);
            }
        }
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::Minus { left, right } => {
            collect_produced(left, vars);
            collect_produced(right, vars);
        }
        Algebra::LeftJoin { left, right, .. } => {
            collect_produced(left, vars);
            collect_produced(right, vars);
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. } => {
            collect_produced(pattern, vars);
        }
        Algebra::Project { pattern, variables } => {
            collect_produced(pattern, vars);
            vars.extend(variables.iter().cloned());
        }
        Algebra::Extend {
            pattern, variable, ..
        } => {
            collect_produced(pattern, vars);
            vars.insert(variable.clone());
        }
        Algebra::Group {
            pattern,
            variables,
            aggregates,
        } => {
            collect_produced(pattern, vars);
            // GroupCondition carries an optional alias Variable.
            for gc in variables {
                if let Some(alias) = &gc.alias {
                    vars.insert(alias.clone());
                }
            }
            for (v, _) in aggregates {
                vars.insert(v.clone());
            }
        }
        Algebra::Graph { pattern, .. } | Algebra::Service { pattern, .. } => {
            collect_produced(pattern, vars);
        }
        // Table is a unit variant (empty result) — no variables.
        Algebra::Table | Algebra::Zero | Algebra::Empty => {}
        Algebra::Values { variables, .. } => {
            vars.extend(variables.iter().cloned());
        }
        Algebra::PropertyPath {
            subject, object, ..
        } => {
            if let Term::Variable(v) = subject {
                vars.insert(v.clone());
            }
            if let Term::Variable(v) = object {
                vars.insert(v.clone());
            }
        }
        _ => {}
    }
}

fn add_pattern_vars(pattern: &TriplePattern, vars: &mut HashSet<Variable>) {
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

/// Return `true` when `algebra` always produces an empty result set (i.e. is a
/// `Bgp([])` — an empty basic graph pattern).
fn is_empty_algebra(algebra: &Algebra) -> bool {
    matches!(algebra, Algebra::Bgp(patterns) if patterns.is_empty())
}

// ---------------------------------------------------------------------------
// Helper: evaluate a constant boolean expression
// ---------------------------------------------------------------------------

/// Evaluate `expr` when it is a *constant* (contains no variables).
/// Returns `Some(true)` / `Some(false)`, or `None` when the expression
/// is not constant or cannot be statically evaluated.
fn evaluate_constant_bool(expr: &Expression) -> Option<bool> {
    match expr {
        // Boolean literals: "true" / "false"
        Expression::Literal(lit) => match lit.value.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            // Integer comparisons: "1" / "0"
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        },
        Expression::Binary { op, left, right } => {
            let l = evaluate_constant_bool(left);
            let r = evaluate_constant_bool(right);
            match op {
                BinaryOperator::And => match (l, r) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                BinaryOperator::Or => match (l, r) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
                BinaryOperator::Equal => {
                    // Two identical literal values.
                    equal_literals(left, right)
                }
                BinaryOperator::NotEqual => equal_literals(left, right).map(|eq| !eq),
                _ => None,
            }
        }
        Expression::Unary {
            op: crate::algebra::UnaryOperator::Not,
            operand,
        } => evaluate_constant_bool(operand).map(|v| !v),
        _ => None,
    }
}

/// Returns `Some(true)` when both expressions are equal literals, `Some(false)`
/// when they are provably different literals, `None` otherwise.
fn equal_literals(left: &Expression, right: &Expression) -> Option<bool> {
    match (left, right) {
        (Expression::Literal(l), Expression::Literal(r)) => Some(l == r),
        _ => None,
    }
}

/// Check whether an expression references any variable.
fn expr_contains_variable(expr: &Expression) -> bool {
    match expr {
        Expression::Variable(_) => true,
        Expression::Bound(_) => true,
        Expression::Binary { left, right, .. } => {
            expr_contains_variable(left) || expr_contains_variable(right)
        }
        Expression::Unary { operand, .. } => expr_contains_variable(operand),
        Expression::Function { args, .. } => args.iter().any(expr_contains_variable),
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            expr_contains_variable(condition)
                || expr_contains_variable(then_expr)
                || expr_contains_variable(else_expr)
        }
        Expression::Exists(inner) | Expression::NotExists(inner) => {
            // Conservatively treat EXISTS as containing a variable.
            let _ = inner;
            true
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Pass 1: ConstantFoldingPass
// ---------------------------------------------------------------------------

/// Folds constant sub-expressions and eliminates statically-decided FILTERs.
///
/// Concretely:
/// - `FILTER(true)` → the inner pattern unchanged (filter removed).
/// - `FILTER(false)` → `Bgp([])` (empty result).
/// - `FILTER(1 = 1)` → filter removed.
/// - `FILTER(1 = 2)` → empty result.
/// - Recursive descent into nested algebra nodes.
pub struct ConstantFoldingPass;

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn apply(&self, algebra: &mut Algebra) -> bool {
        fold_constant_recursive(algebra)
    }
}

/// Recursively walk the algebra tree, folding constant filters.
/// Returns `true` when any node was rewritten.
fn fold_constant_recursive(algebra: &mut Algebra) -> bool {
    match algebra {
        Algebra::Filter { pattern, condition } => {
            // First recurse into the child pattern.
            let child_changed = fold_constant_recursive(pattern);

            // Try to evaluate the filter condition as a constant.
            if !expr_contains_variable(condition) {
                if let Some(constant_value) = evaluate_constant_bool(condition) {
                    if constant_value {
                        // Filter is always true — replace with inner pattern.
                        let inner = std::mem::replace(pattern.as_mut(), Algebra::Bgp(vec![]));
                        *algebra = inner;
                    } else {
                        // Filter is always false — produce empty result.
                        *algebra = Algebra::Bgp(vec![]);
                    }
                    return true;
                }
            }

            child_changed
        }
        Algebra::Join { left, right } => {
            let l = fold_constant_recursive(left);
            let r = fold_constant_recursive(right);
            l || r
        }
        Algebra::Union { left, right } => {
            let l = fold_constant_recursive(left);
            let r = fold_constant_recursive(right);
            l || r
        }
        Algebra::LeftJoin { left, right, .. } => {
            let l = fold_constant_recursive(left);
            let r = fold_constant_recursive(right);
            l || r
        }
        Algebra::Project { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Slice { pattern, .. } => fold_constant_recursive(pattern),
        Algebra::Extend { pattern, .. } => fold_constant_recursive(pattern),
        Algebra::Graph { pattern, .. } | Algebra::Service { pattern, .. } => {
            fold_constant_recursive(pattern)
        }
        Algebra::Group { pattern, .. } => fold_constant_recursive(pattern),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Pass 2: UnusedVariableEliminationPass
// ---------------------------------------------------------------------------

/// Removes variables from `Project` nodes when those variables are never
/// produced by the inner pattern.  This can happen when a query projects
/// a variable that only appears in an outer BIND / extension that was
/// removed by an earlier pass.
///
/// If eliminating variables leaves the projection list empty the entire
/// `Project` node is replaced by `Bgp([])`.
pub struct UnusedVariableEliminationPass;

impl OptimizationPass for UnusedVariableEliminationPass {
    fn name(&self) -> &str {
        "unused_var_elimination"
    }

    fn apply(&self, algebra: &mut Algebra) -> bool {
        eliminate_unused_vars(algebra)
    }
}

fn eliminate_unused_vars(algebra: &mut Algebra) -> bool {
    match algebra {
        Algebra::Project { pattern, variables } => {
            // Recurse first.
            let child_changed = eliminate_unused_vars(pattern);

            // Determine which variables are actually produced by the inner pattern.
            let produced = produced_variables(pattern);

            let before = variables.len();
            variables.retain(|v| produced.contains(v));
            let after = variables.len();

            let changed = before != after || child_changed;

            if variables.is_empty() {
                *algebra = Algebra::Bgp(vec![]);
                return true;
            }

            changed
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. } => eliminate_unused_vars(pattern),
        Algebra::Join { left, right } => {
            let l = eliminate_unused_vars(left);
            let r = eliminate_unused_vars(right);
            l || r
        }
        Algebra::Union { left, right } => {
            let l = eliminate_unused_vars(left);
            let r = eliminate_unused_vars(right);
            l || r
        }
        Algebra::LeftJoin { left, right, .. } => {
            let l = eliminate_unused_vars(left);
            let r = eliminate_unused_vars(right);
            l || r
        }
        Algebra::Graph { pattern, .. } | Algebra::Service { pattern, .. } => {
            eliminate_unused_vars(pattern)
        }
        Algebra::Group { pattern, .. } => eliminate_unused_vars(pattern),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Pass 3: RedundantJoinEliminationPass
// ---------------------------------------------------------------------------

/// Eliminates join branches whose inner algebra always produces empty results.
///
/// Rules:
/// - `Join { left: Empty, right: _ }` → `Bgp([])`
/// - `Join { left: _, right: Empty }` → `Bgp([])`
///
/// (`Bgp([])` represents an empty result set — it acts like a zero for joins.)
///
/// Recursion continues into nested joins so that chains are fully simplified.
pub struct RedundantJoinEliminationPass;

impl OptimizationPass for RedundantJoinEliminationPass {
    fn name(&self) -> &str {
        "redundant_join_elimination"
    }

    fn apply(&self, algebra: &mut Algebra) -> bool {
        eliminate_redundant_joins(algebra)
    }
}

fn eliminate_redundant_joins(algebra: &mut Algebra) -> bool {
    match algebra {
        Algebra::Join { left, right } => {
            // Recurse first.
            let l = eliminate_redundant_joins(left);
            let r = eliminate_redundant_joins(right);

            // If either child is empty the whole join is empty.
            if is_empty_algebra(left) || is_empty_algebra(right) {
                *algebra = Algebra::Bgp(vec![]);
                return true;
            }

            l || r
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. } => eliminate_redundant_joins(pattern),
        Algebra::Project { pattern, .. } => eliminate_redundant_joins(pattern),
        Algebra::Union { left, right } => {
            let l = eliminate_redundant_joins(left);
            let r = eliminate_redundant_joins(right);
            l || r
        }
        Algebra::LeftJoin { left, right, .. } => {
            let l = eliminate_redundant_joins(left);
            let r = eliminate_redundant_joins(right);
            l || r
        }
        Algebra::Graph { pattern, .. } | Algebra::Service { pattern, .. } => {
            eliminate_redundant_joins(pattern)
        }
        Algebra::Group { pattern, .. } => eliminate_redundant_joins(pattern),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------

/// Summary of a `OptimizationPipeline::run` invocation.
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    /// Total number of pass invocations across all iterations.
    pub passes_run: usize,
    /// Number of fixed-point iterations performed.
    pub total_iterations: usize,
    /// `true` when at least one pass changed the algebra.
    pub changed: bool,
    /// Names of the passes that were part of this pipeline, in order.
    pub pass_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// OptimizationPipeline
// ---------------------------------------------------------------------------

/// A sequential composition of `OptimizationPass` instances.
///
/// Passes are run in the order they were added.  The pipeline repeats until
/// no pass produces a change (fixed-point convergence) or `max_iterations`
/// is reached.
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    /// Maximum number of fixed-point iterations.
    max_iterations: usize,
}

impl OptimizationPipeline {
    /// Create an empty pipeline with a default iteration limit of 10.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Append a pass to the end of the pipeline.  Returns `self` for chaining.
    pub fn add_pass(mut self, pass: Box<dyn OptimizationPass>) -> Self {
        self.passes.push(pass);
        self
    }

    /// Set the maximum number of fixed-point iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    /// Run the pipeline to fixed-point convergence (or `max_iterations`).
    pub fn run(&self, algebra: &mut Algebra) -> PipelineResult {
        let pass_names: Vec<String> = self.passes.iter().map(|p| p.name().to_string()).collect();
        let mut result = PipelineResult {
            pass_names,
            ..Default::default()
        };

        for _iter in 0..self.max_iterations {
            let mut iteration_changed = false;

            for pass in &self.passes {
                let changed = pass.apply(algebra);
                result.passes_run += 1;
                if changed {
                    iteration_changed = true;
                    result.changed = true;
                }
            }

            result.total_iterations += 1;

            if !iteration_changed {
                // Fixed point reached — no further changes possible.
                break;
            }
        }

        result
    }

    /// Construct the default pipeline:
    /// `ConstantFolding` → `UnusedVarElimination` → `RedundantJoinElimination`.
    pub fn default_pipeline() -> Self {
        Self::new()
            .add_pass(Box::new(ConstantFoldingPass))
            .add_pass(Box::new(UnusedVariableEliminationPass))
            .add_pass(Box::new(RedundantJoinEliminationPass))
    }
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Literal as AlgLiteral, Term, TriplePattern, Variable};
    use oxirs_core::model::NamedNode;

    fn make_var(name: &str) -> Variable {
        Variable::new(name).expect("valid variable name")
    }

    fn make_iri(iri: &str) -> Term {
        Term::Iri(NamedNode::new(iri).expect("valid IRI"))
    }

    fn make_var_term(name: &str) -> Term {
        Term::Variable(make_var(name))
    }

    fn make_triple_pattern(s: Term, p: Term, o: Term) -> TriplePattern {
        TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    fn make_bgp_with_var(var: &str) -> Algebra {
        Algebra::Bgp(vec![make_triple_pattern(
            make_var_term(var),
            make_iri("http://example.org/p"),
            make_iri("http://example.org/o"),
        )])
    }

    fn make_empty_bgp() -> Algebra {
        Algebra::Bgp(vec![])
    }

    fn make_true_filter_condition() -> Expression {
        Expression::Literal(AlgLiteral {
            value: "true".to_string(),
            language: None,
            datatype: None,
        })
    }

    fn make_false_filter_condition() -> Expression {
        Expression::Literal(AlgLiteral {
            value: "false".to_string(),
            language: None,
            datatype: None,
        })
    }

    fn make_equal_literals(a: &str, b: &str) -> Expression {
        Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Literal(AlgLiteral {
                value: a.to_string(),
                language: None,
                datatype: None,
            })),
            right: Box::new(Expression::Literal(AlgLiteral {
                value: b.to_string(),
                language: None,
                datatype: None,
            })),
        }
    }

    // ------------------------------------------------------------------
    // ConstantFoldingPass tests
    // ------------------------------------------------------------------

    #[test]
    fn test_constant_folding_true_filter_removed() {
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner.clone()),
            condition: make_true_filter_condition(),
        };

        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        // The filter should have been replaced with the inner pattern.
        assert_eq!(algebra, inner);
    }

    #[test]
    fn test_constant_folding_false_filter_produces_empty() {
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner),
            condition: make_false_filter_condition(),
        };

        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_constant_folding_equal_literals_true() {
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner.clone()),
            condition: make_equal_literals("hello", "hello"),
        };

        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        assert_eq!(algebra, inner);
    }

    #[test]
    fn test_constant_folding_equal_literals_false() {
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner),
            condition: make_equal_literals("a", "b"),
        };

        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_constant_folding_variable_filter_not_folded() {
        let inner = make_bgp_with_var("x");
        let var_filter = Expression::Binary {
            op: BinaryOperator::Equal,
            left: Box::new(Expression::Variable(make_var("x"))),
            right: Box::new(Expression::Literal(AlgLiteral {
                value: "42".to_string(),
                language: None,
                datatype: None,
            })),
        };
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner),
            condition: var_filter,
        };

        // Must not be changed because the filter references a variable.
        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(!changed);
    }

    #[test]
    fn test_constant_folding_no_filter_not_changed() {
        let mut algebra = make_bgp_with_var("x");
        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(!changed);
    }

    #[test]
    fn test_constant_folding_nested_filter() {
        // FILTER(false) inside a JOIN — the join child should become empty.
        let with_false_filter = Algebra::Filter {
            pattern: Box::new(make_bgp_with_var("x")),
            condition: make_false_filter_condition(),
        };
        let mut algebra = Algebra::Join {
            left: Box::new(make_bgp_with_var("y")),
            right: Box::new(with_false_filter),
        };

        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        // After folding the right child should be empty.
        if let Algebra::Join { right, .. } = &algebra {
            assert!(is_empty_algebra(right));
        } else {
            panic!("expected Join");
        }
    }

    #[test]
    fn test_constant_folding_and_true_true() {
        let cond = Expression::Binary {
            op: BinaryOperator::And,
            left: Box::new(make_true_filter_condition()),
            right: Box::new(make_true_filter_condition()),
        };
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner.clone()),
            condition: cond,
        };
        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        assert_eq!(algebra, inner);
    }

    #[test]
    fn test_constant_folding_or_false_false() {
        let cond = Expression::Binary {
            op: BinaryOperator::Or,
            left: Box::new(make_false_filter_condition()),
            right: Box::new(make_false_filter_condition()),
        };
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Filter {
            pattern: Box::new(inner),
            condition: cond,
        };
        let changed = ConstantFoldingPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    // ------------------------------------------------------------------
    // UnusedVariableEliminationPass tests
    // ------------------------------------------------------------------

    #[test]
    fn test_unused_var_elimination_removes_absent_var() {
        // The inner BGP produces only ?x; the project list asks for both ?x and ?ghost.
        let inner = make_bgp_with_var("x");
        let x_var = make_var("x");
        let ghost_var = make_var("ghost");
        let mut algebra = Algebra::Project {
            pattern: Box::new(inner),
            variables: vec![x_var.clone(), ghost_var],
        };

        let changed = UnusedVariableEliminationPass.apply(&mut algebra);
        assert!(changed);
        if let Algebra::Project { variables, .. } = &algebra {
            assert_eq!(variables, &[x_var]);
        } else {
            panic!("expected Project");
        }
    }

    #[test]
    fn test_unused_var_elimination_empty_project_becomes_empty_bgp() {
        // Project with no matching variables → Bgp([]).
        let inner = make_bgp_with_var("x");
        let mut algebra = Algebra::Project {
            pattern: Box::new(inner),
            variables: vec![make_var("does_not_exist")],
        };

        let changed = UnusedVariableEliminationPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_unused_var_elimination_all_vars_present_no_change() {
        let inner = make_bgp_with_var("x");
        let x_var = make_var("x");
        let mut algebra = Algebra::Project {
            pattern: Box::new(inner),
            variables: vec![x_var],
        };
        let changed = UnusedVariableEliminationPass.apply(&mut algebra);
        // Nothing removed, no change.
        assert!(!changed);
    }

    #[test]
    fn test_unused_var_elimination_non_project_no_change() {
        let mut algebra = make_bgp_with_var("x");
        let changed = UnusedVariableEliminationPass.apply(&mut algebra);
        assert!(!changed);
    }

    #[test]
    fn test_unused_var_join_both_sides() {
        // Inner join: left produces ?a, right produces ?b.
        let left = make_bgp_with_var("a");
        let right = make_bgp_with_var("b");
        let join = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };

        let mut algebra = Algebra::Project {
            pattern: Box::new(join),
            variables: vec![make_var("a"), make_var("b"), make_var("c")],
        };

        let changed = UnusedVariableEliminationPass.apply(&mut algebra);
        assert!(changed);
        if let Algebra::Project { variables, .. } = &algebra {
            assert!(variables.contains(&make_var("a")));
            assert!(variables.contains(&make_var("b")));
            assert!(!variables.contains(&make_var("c")));
        } else {
            panic!("expected Project");
        }
    }

    // ------------------------------------------------------------------
    // RedundantJoinEliminationPass tests
    // ------------------------------------------------------------------

    #[test]
    fn test_redundant_join_left_empty() {
        let mut algebra = Algebra::Join {
            left: Box::new(make_empty_bgp()),
            right: Box::new(make_bgp_with_var("x")),
        };
        let changed = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_redundant_join_right_empty() {
        let mut algebra = Algebra::Join {
            left: Box::new(make_bgp_with_var("x")),
            right: Box::new(make_empty_bgp()),
        };
        let changed = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_redundant_join_both_empty() {
        let mut algebra = Algebra::Join {
            left: Box::new(make_empty_bgp()),
            right: Box::new(make_empty_bgp()),
        };
        let changed = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_redundant_join_no_empty_no_change() {
        let mut algebra = Algebra::Join {
            left: Box::new(make_bgp_with_var("x")),
            right: Box::new(make_bgp_with_var("y")),
        };
        let changed = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(!changed);
    }

    #[test]
    fn test_redundant_join_nested_chain() {
        // Join(Join(empty, x), y) → recursive elimination produces empty in one pass.
        // The implementation recurses bottom-up: the inner Join(empty, x) becomes
        // empty, then the outer Join(empty, y) is also eliminated, all in one call.
        let inner = Algebra::Join {
            left: Box::new(make_empty_bgp()),
            right: Box::new(make_bgp_with_var("x")),
        };
        let mut algebra = Algebra::Join {
            left: Box::new(inner),
            right: Box::new(make_bgp_with_var("y")),
        };

        // Single pass is enough because the implementation recurses bottom-up.
        let changed1 = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(changed1);
        assert!(is_empty_algebra(&algebra));

        // Second pass on an already-empty algebra → no change.
        let changed2 = RedundantJoinEliminationPass.apply(&mut algebra);
        assert!(!changed2);
    }

    // ------------------------------------------------------------------
    // OptimizationPipeline tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pipeline_default_passes_included() {
        let pipeline = OptimizationPipeline::default_pipeline();
        assert_eq!(pipeline.passes.len(), 3);
        assert_eq!(pipeline.passes[0].name(), "constant_folding");
        assert_eq!(pipeline.passes[1].name(), "unused_var_elimination");
        assert_eq!(pipeline.passes[2].name(), "redundant_join_elimination");
    }

    #[test]
    fn test_pipeline_run_reports_changed() {
        let pipeline = OptimizationPipeline::default_pipeline();

        // A FILTER(false) should be folded to empty.
        let mut algebra = Algebra::Filter {
            pattern: Box::new(make_bgp_with_var("x")),
            condition: make_false_filter_condition(),
        };

        let result = pipeline.run(&mut algebra);
        assert!(result.changed);
        assert!(result.total_iterations >= 1);
        assert!(result.passes_run >= 1);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_pipeline_run_no_change_converges_immediately() {
        let pipeline = OptimizationPipeline::default_pipeline();
        let mut algebra = make_bgp_with_var("x");
        let result = pipeline.run(&mut algebra);
        assert!(!result.changed);
        // Should converge after a single iteration (all passes report no change).
        assert_eq!(result.total_iterations, 1);
    }

    #[test]
    fn test_pipeline_run_pass_names_in_result() {
        let pipeline = OptimizationPipeline::default_pipeline();
        let mut algebra = make_bgp_with_var("x");
        let result = pipeline.run(&mut algebra);
        assert!(result.pass_names.contains(&"constant_folding".to_string()));
        assert!(result
            .pass_names
            .contains(&"unused_var_elimination".to_string()));
        assert!(result
            .pass_names
            .contains(&"redundant_join_elimination".to_string()));
    }

    #[test]
    fn test_pipeline_empty_pipeline_no_change() {
        let pipeline = OptimizationPipeline::new();
        let mut algebra = make_bgp_with_var("x");
        let result = pipeline.run(&mut algebra);
        assert!(!result.changed);
    }

    #[test]
    fn test_pipeline_combined_fold_and_join_elimination() {
        let pipeline = OptimizationPipeline::default_pipeline();

        // FILTER(false) inside join → after fold the join branch is empty → join eliminated.
        let filter_false = Algebra::Filter {
            pattern: Box::new(make_bgp_with_var("x")),
            condition: make_false_filter_condition(),
        };
        let mut algebra = Algebra::Join {
            left: Box::new(make_bgp_with_var("y")),
            right: Box::new(filter_false),
        };

        let result = pipeline.run(&mut algebra);
        assert!(result.changed);
        assert!(is_empty_algebra(&algebra));
    }

    #[test]
    fn test_pipeline_result_total_iterations_bounded() {
        // Even if changes keep being found we must not exceed max_iterations.
        let pipeline = OptimizationPipeline::new().with_max_iterations(2);
        let mut algebra = make_bgp_with_var("x");
        let result = pipeline.run(&mut algebra);
        assert!(result.total_iterations <= 2);
    }
}
