//! # SPARQL 1.1 Federation Algebra Optimizer
//!
//! Algebraic rewrite passes that operate directly on
//! [`oxirs_arq::algebra::Algebra`] trees and improve federated execution
//! plans.  Each pass is a pure function `Algebra -> Algebra` so they can be
//! composed, reordered, or run individually under tests.
//!
//! ## Available passes
//!
//! - [`filter_pushdown::push_filters_into_services`] — pushes
//!   `Filter(Service(...))` into `Service(Filter(...))` whenever the filter's
//!   free variables are bound entirely by the service pattern.
//! - [`service_merge::merge_adjacent_services`] — collapses
//!   `Join(Service(A, p1), Service(A, p2))` into
//!   `Service(A, Join(p1, p2))` when the two services target the same
//!   endpoint with compatible `SILENT` semantics.
//! - [`join_decomposer::reorder_joins_by_selectivity`] — re-orders left-deep
//!   join chains so that the most-selective leg is evaluated first, using a
//!   [`SourceSelectivityProvider`] for cardinality estimates.
//!
//! ## Composition
//!
//! [`OptimizerPipeline`] chains passes in a sensible default order:
//! filter pushdown, then service merge, then join reordering.  Embedders may
//! omit individual passes or insert custom ones via
//! [`OptimizerPipeline::with_passes`].
//!
//! All passes are deterministic, side-effect free, and preserve solution
//! semantics modulo the documented exceptions in each pass's module docs.

use std::sync::Arc;

use oxirs_arq::algebra::Algebra;
use oxirs_arq::optimizer::federated_plan::SourceSelectivityProvider;

pub mod filter_pushdown;
pub mod join_decomposer;
pub mod service_merge;

pub use filter_pushdown::push_filters_into_services;
pub use join_decomposer::reorder_joins_by_selectivity;
pub use service_merge::merge_adjacent_services;

/// Composable pipeline of federation algebra rewrites.
///
/// A pass is `Fn(Algebra) -> Algebra` and runs in registration order.  Use
/// [`OptimizerPipeline::default`] for the standard ordering or
/// [`OptimizerPipeline::with_passes`] for custom configurations.
pub struct OptimizerPipeline {
    selectivity: Option<Arc<dyn SourceSelectivityProvider>>,
    enable_filter_pushdown: bool,
    enable_service_merge: bool,
    enable_join_reorder: bool,
}

impl Default for OptimizerPipeline {
    fn default() -> Self {
        Self {
            selectivity: None,
            enable_filter_pushdown: true,
            enable_service_merge: true,
            enable_join_reorder: false, // requires a selectivity provider
        }
    }
}

impl OptimizerPipeline {
    /// Build a pipeline with all rewrites enabled by default (except join
    /// reordering, which only activates when a selectivity provider is wired
    /// in).
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a [`SourceSelectivityProvider`] and enable join reordering.
    pub fn with_selectivity(mut self, provider: Arc<dyn SourceSelectivityProvider>) -> Self {
        self.selectivity = Some(provider);
        self.enable_join_reorder = true;
        self
    }

    /// Enable or disable individual passes.
    pub fn with_passes(
        mut self,
        filter_pushdown: bool,
        service_merge: bool,
        join_reorder: bool,
    ) -> Self {
        self.enable_filter_pushdown = filter_pushdown;
        self.enable_service_merge = service_merge;
        self.enable_join_reorder = join_reorder;
        self
    }

    /// Run every enabled pass in order and return the rewritten algebra.
    ///
    /// The pipeline runs `filter_pushdown` → `service_merge` →
    /// `filter_pushdown` (again) → `join_reorder`.  The second pushdown lets
    /// filters fall into freshly-merged SERVICE nodes that previously sat on
    /// top of a Join.  Both pushdown invocations are idempotent, so this is
    /// a fixed-point in two iterations for the rule set we currently
    /// support.
    pub fn run(&self, mut algebra: Algebra) -> Algebra {
        if self.enable_filter_pushdown {
            algebra = push_filters_into_services(algebra);
        }
        if self.enable_service_merge {
            algebra = merge_adjacent_services(algebra);
        }
        // After merge, any new Filter(Service(...)) shapes can be flattened.
        if self.enable_filter_pushdown {
            algebra = push_filters_into_services(algebra);
        }
        if self.enable_join_reorder {
            if let Some(provider) = self.selectivity.as_ref() {
                algebra = reorder_joins_by_selectivity(algebra, provider.as_ref());
            }
        }
        algebra
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-variable helpers — used by multiple passes
// ─────────────────────────────────────────────────────────────────────────────

use oxirs_arq::algebra::{Expression, Term, TriplePattern, Variable};
use std::collections::HashSet;

/// Collect every variable that appears in `pattern` (as subject, predicate, or
/// object).  Includes variables introduced by `Project`, `Extend`, and `Group`
/// nodes.
pub fn bound_variables(algebra: &Algebra) -> HashSet<Variable> {
    let mut out = HashSet::new();
    collect_bound_variables(algebra, &mut out);
    out
}

fn collect_bound_variables(algebra: &Algebra, out: &mut HashSet<Variable>) {
    match algebra {
        Algebra::Bgp(patterns) => {
            for tp in patterns {
                collect_pattern_vars(tp, out);
            }
        }
        Algebra::PropertyPath {
            subject, object, ..
        } => {
            term_var(subject, out);
            term_var(object, out);
        }
        Algebra::Join { left, right }
        | Algebra::Union { left, right }
        | Algebra::Minus { left, right } => {
            collect_bound_variables(left, out);
            collect_bound_variables(right, out);
        }
        Algebra::LeftJoin { left, right, .. } => {
            collect_bound_variables(left, out);
            collect_bound_variables(right, out);
        }
        Algebra::Filter { pattern, .. } => collect_bound_variables(pattern, out),
        Algebra::Extend {
            pattern, variable, ..
        } => {
            collect_bound_variables(pattern, out);
            out.insert(variable.clone());
        }
        Algebra::Service { pattern, .. } => collect_bound_variables(pattern, out),
        Algebra::Graph { pattern, graph } => {
            collect_bound_variables(pattern, out);
            term_var(graph, out);
        }
        Algebra::Project { variables, .. } => {
            for v in variables {
                out.insert(v.clone());
            }
        }
        Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Having { pattern, .. } => collect_bound_variables(pattern, out),
        Algebra::Group {
            pattern,
            variables,
            aggregates,
        } => {
            collect_bound_variables(pattern, out);
            for gc in variables {
                if let Some(alias) = &gc.alias {
                    out.insert(alias.clone());
                }
            }
            for (v, _) in aggregates {
                out.insert(v.clone());
            }
        }
        Algebra::Values {
            variables,
            bindings: _,
        } => {
            for v in variables {
                out.insert(v.clone());
            }
        }
        Algebra::Table | Algebra::Zero | Algebra::Empty => {}
    }
}

fn collect_pattern_vars(tp: &TriplePattern, out: &mut HashSet<Variable>) {
    term_var(&tp.subject, out);
    term_var(&tp.predicate, out);
    term_var(&tp.object, out);
}

fn term_var(term: &Term, out: &mut HashSet<Variable>) {
    if let Term::Variable(v) = term {
        out.insert(v.clone());
    }
}

/// Free variables of an expression — the variables it *uses* but does not
/// itself bind.
pub fn expression_free_vars(expr: &Expression) -> HashSet<Variable> {
    let mut out = HashSet::new();
    collect_expression_free_vars(expr, &mut out);
    out
}

fn collect_expression_free_vars(expr: &Expression, out: &mut HashSet<Variable>) {
    match expr {
        Expression::Variable(v) => {
            out.insert(v.clone());
        }
        Expression::Bound(v) => {
            out.insert(v.clone());
        }
        Expression::Literal(_) | Expression::Iri(_) => {}
        Expression::Function { args, .. } => {
            for a in args {
                collect_expression_free_vars(a, out);
            }
        }
        Expression::Binary { left, right, .. } => {
            collect_expression_free_vars(left, out);
            collect_expression_free_vars(right, out);
        }
        Expression::Unary { operand, .. } => {
            collect_expression_free_vars(operand, out);
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_expression_free_vars(condition, out);
            collect_expression_free_vars(then_expr, out);
            collect_expression_free_vars(else_expr, out);
        }
        Expression::Exists(algebra) | Expression::NotExists(algebra) => {
            for v in bound_variables(algebra) {
                out.insert(v);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_arq::algebra::{Literal, Term, TriplePattern};

    fn var(s: &str) -> Variable {
        Variable::new(s).expect("valid var name in test")
    }

    fn iri(s: &str) -> Term {
        Term::Iri(oxirs_core::model::NamedNode::new_unchecked(s))
    }

    fn lit(s: &str) -> Term {
        Term::Literal(Literal {
            value: s.to_string(),
            language: None,
            datatype: None,
        })
    }

    fn tp(s: Term, p: Term, o: Term) -> TriplePattern {
        TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    fn bgp_one(s: &str, p: &str, o: &str) -> Algebra {
        Algebra::Bgp(vec![tp(
            Term::Variable(var(s)),
            iri(p),
            Term::Variable(var(o)),
        )])
    }

    #[test]
    fn pipeline_default_runs_pushdown_and_merge() {
        let pipe = OptimizerPipeline::default();
        // No selectivity set — join reorder skipped.
        assert!(pipe.enable_filter_pushdown);
        assert!(pipe.enable_service_merge);
        assert!(!pipe.enable_join_reorder);
    }

    #[test]
    fn pipeline_with_passes_overrides() {
        let pipe = OptimizerPipeline::default().with_passes(false, false, false);
        let algebra = bgp_one("s", "http://p", "o");
        // No-op when all passes disabled.
        let out = pipe.run(algebra.clone());
        assert_eq!(out, algebra);
    }

    #[test]
    fn bound_variables_bgp() {
        let algebra = bgp_one("s", "http://p", "o");
        let vars = bound_variables(&algebra);
        assert!(vars.contains(&var("s")));
        assert!(vars.contains(&var("o")));
    }

    #[test]
    fn bound_variables_join_union() {
        let left = bgp_one("a", "http://p", "b");
        let right = bgp_one("c", "http://q", "d");
        let join = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };
        let vars = bound_variables(&join);
        assert!(vars.contains(&var("a")));
        assert!(vars.contains(&var("b")));
        assert!(vars.contains(&var("c")));
        assert!(vars.contains(&var("d")));
    }

    #[test]
    fn bound_variables_extend_and_project() {
        let inner = bgp_one("s", "http://p", "o");
        let extended = Algebra::Extend {
            pattern: Box::new(inner),
            variable: var("x"),
            expr: Expression::Literal(Literal {
                value: "1".into(),
                language: None,
                datatype: None,
            }),
        };
        let vars = bound_variables(&extended);
        assert!(vars.contains(&var("x")));
        assert!(vars.contains(&var("s")));
    }

    #[test]
    fn bound_variables_values() {
        let algebra = Algebra::Values {
            variables: vec![var("a"), var("b")],
            bindings: vec![],
        };
        let vars = bound_variables(&algebra);
        assert!(vars.contains(&var("a")));
        assert!(vars.contains(&var("b")));
    }

    #[test]
    fn expression_free_vars_simple_eq() {
        let e = Expression::Binary {
            op: oxirs_arq::algebra::BinaryOperator::Equal,
            left: Box::new(Expression::Variable(var("x"))),
            right: Box::new(Expression::Literal(Literal {
                value: "1".into(),
                language: None,
                datatype: None,
            })),
        };
        let frees = expression_free_vars(&e);
        assert!(frees.contains(&var("x")));
        assert_eq!(frees.len(), 1);
    }

    #[test]
    fn expression_free_vars_bound_predicate() {
        let e = Expression::Bound(var("y"));
        let frees = expression_free_vars(&e);
        assert!(frees.contains(&var("y")));
    }

    #[test]
    fn expression_free_vars_constant_only() {
        let e = Expression::Literal(Literal {
            value: "42".into(),
            language: None,
            datatype: None,
        });
        assert!(expression_free_vars(&e).is_empty());
    }

    #[test]
    fn iri_term_helper_does_not_panic() {
        let _ = iri("http://example.org/p");
        let _ = lit("hello");
    }
}
