//! # QueryExecutor - execute_index_optimized_join_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::{Algebra, Binding, Solution, Term, Variable};
use anyhow::Result;
use std::collections::HashSet;

use super::queryexecutor_queries::substitute_algebra_binding;
use super::queryexecutor_type::QueryExecutor;

/// Upper bound on the statically-known row count of the outer side for the
/// bound-join path. Above this, N index lookups may cost more than one scan
/// plus a hash join, so we fall back. Mirrors the threshold precedent in
/// `values_support::ValuesJoinOptimizer::can_push_values`.
const BOUND_JOIN_ROW_THRESHOLD: usize = 100;

impl QueryExecutor {
    /// Execute index-optimized join
    pub(super) fn execute_index_optimized_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        // A small, dataset-independent side (VALUES / BIND chain) joined
        // against a triple-pattern side must not evaluate the pattern side
        // unconstrained — `?s ?p ?o` with a syntactically-variable subject is
        // a full store scan even when the sibling pins ?s to one IRI.
        if let Some(result) = self.try_bound_join(left, right, dataset)? {
            return Ok(result);
        }
        if let Some(result) = self.try_bound_join(right, left, dataset)? {
            return Ok(result);
        }
        let left_results = self.execute_serial(left, dataset)?;
        let right_results = self.execute_serial(right, dataset)?;
        if left_results.len() < right_results.len() {
            self.hash_join(left_results, right_results)
        } else {
            self.hash_join(right_results, left_results)
        }
    }

    /// Attempt a bound (index-nested-loop) join: evaluate `outer` first and
    /// substitute each of its rows into `inner` before evaluating it, so a
    /// pinned subject/object becomes an index lookup instead of a full scan.
    /// Returns `Ok(None)` when a gate fails and the caller must fall back.
    pub(super) fn try_bound_join(
        &self,
        outer: &Algebra,
        inner: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Option<Solution>> {
        // Gate 1 (correctness): only substitute into shapes
        // `substitute_algebra_binding` rewrites all the way down — a BGP,
        // optionally under Filter(s). Sub-selects (Project/Group/Slice) and
        // PropertyPath must fall back: substitution either breaks variable
        // scoping or silently does nothing there.
        if !is_bgp_or_filter_over_bgp(inner) {
            return Ok(None);
        }
        // Gate 2 (cost): the outer side must be statically bounded WITHOUT
        // touching the dataset (VALUES rows / BIND-over-Table chains). Sizing
        // it by evaluating both sides would defeat the point.
        let Some(outer_rows_hint) = static_row_count(outer) else {
            return Ok(None);
        };
        if outer_rows_hint > BOUND_JOIN_ROW_THRESHOLD {
            return Ok(None);
        }
        // Gate 3 (usefulness): outer must actually pin at least one variable
        // occurring in inner's triple-pattern term positions; otherwise the
        // substitution is a no-op and N re-evaluations buy nothing.
        let pattern_vars = collect_pattern_variables(inner);
        let outer_rows = self.execute_serial(outer, dataset)?;
        let binds_pattern_var = outer_rows
            .iter()
            .any(|row| row.keys().any(|v| pattern_vars.contains(v)));
        if !binds_pattern_var {
            return Ok(None);
        }
        // Gate 4 (path parity): substituting into a FILTER condition is only
        // result-equivalent to the fallback when every condition variable the
        // outer side binds is ALSO a BGP pattern variable (then the pattern
        // substitution pins it identically on both paths). A condition-only
        // variable would see the outer constant here but be unbound (error →
        // row dropped) on the fallback path, making the result depend on the
        // VALUES row count. Bail to the fallback for that shape — and for
        // conditions we cannot analyze (EXISTS/NOT EXISTS).
        let Some(condition_vars) = collect_filter_condition_variables(inner) else {
            return Ok(None);
        };
        let condition_only_var_bound = outer_rows.iter().any(|row| {
            row.keys()
                .any(|v| condition_vars.contains(v) && !pattern_vars.contains(v))
        });
        if condition_only_var_bound {
            return Ok(None);
        }
        let mut result = Solution::new();
        for outer_row in &outer_rows {
            let specialized = substitute_algebra_binding(inner, outer_row);
            let inner_rows = self.execute_serial(&specialized, dataset)?;
            for inner_row in inner_rows {
                // Conflict-checked merge (same contract as hash_join): a
                // variable substituted out of `inner` can still come back
                // bound differently, and must reject the row, not be
                // overwritten.
                if let Some(merged) = compatible_merge(outer_row, &inner_row) {
                    result.push(merged);
                }
            }
        }
        Ok(Some(result))
    }

    /// `FILTER(?v IN (<iri>, …))` over a BGP: enumerate the (constant, all-IRI)
    /// list and run one substituted, index-driven evaluation per IRI instead of
    /// scanning the pattern unconstrained and testing membership row by row.
    ///
    /// Restricted to IRI lists on purpose: SPARQL `IN` uses value equality
    /// (`"1"^^xsd:integer IN ("1.0"^^xsd:decimal)` is true), whereas
    /// substitution matches terms exactly — the two coincide only for IRIs.
    /// Returns `Ok(None)` when any gate fails so the caller falls back to the
    /// unchanged evaluate-then-filter path.
    pub(super) fn try_filter_in_pushdown(
        &self,
        pattern: &Algebra,
        condition: &crate::algebra::Expression,
        dataset: &dyn Dataset,
    ) -> Result<Option<Solution>> {
        use crate::algebra::{BinaryOperator, Expression};
        let Expression::Binary {
            op: BinaryOperator::In,
            left,
            right,
        } = condition
        else {
            return Ok(None);
        };
        let Expression::Variable(v) = left.as_ref() else {
            return Ok(None);
        };
        let Some(iris) = static_iri_list(right) else {
            return Ok(None);
        };
        if iris.len() > BOUND_JOIN_ROW_THRESHOLD {
            return Ok(None);
        }
        if !is_bgp_or_filter_over_bgp(pattern) {
            return Ok(None);
        }
        // ?v must be a pattern variable — substitution then actually
        // constrains the scan, and any inner FILTER referencing ?v sees the
        // same pinned value it would after the join (Gate 4 reasoning).
        if !collect_pattern_variables(pattern).contains(v) {
            return Ok(None);
        }
        // IN is a membership TEST: a duplicated list entry must not
        // duplicate result rows.
        let mut seen: HashSet<Term> = HashSet::new();
        let mut result = Solution::new();
        for iri in iris {
            if !seen.insert(iri.clone()) {
                continue;
            }
            let mut row = Binding::new();
            row.insert(v.clone(), iri);
            let specialized = substitute_algebra_binding(pattern, &row);
            for inner_row in self.execute_serial(&specialized, dataset)? {
                if let Some(merged) = compatible_merge(&row, &inner_row) {
                    result.push(merged);
                }
            }
        }
        Ok(Some(result))
    }
}

/// The elements of an IN list when every one of them is a constant IRI;
/// `None` for anything else (variables, literals, mixed lists).
fn static_iri_list(expr: &crate::algebra::Expression) -> Option<Vec<Term>> {
    use crate::algebra::Expression as E;
    let items: &[E] = match expr {
        E::Function { name, args }
            if name.eq_ignore_ascii_case("list") || name.eq_ignore_ascii_case("in") =>
        {
            args
        }
        single => std::slice::from_ref(single),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            E::Iri(iri) => out.push(Term::Iri(iri.clone())),
            _ => return None,
        }
    }
    Some(out)
}

/// Shapes safe to substitute a binding into: a BGP, possibly wrapped in
/// FILTER layers (the filter condition is substituted too, which is exactly
/// the EXISTS-proven behavior).
fn is_bgp_or_filter_over_bgp(algebra: &Algebra) -> bool {
    match algebra {
        Algebra::Bgp(_) => true,
        Algebra::Filter { pattern, .. } => is_bgp_or_filter_over_bgp(pattern),
        _ => false,
    }
}

/// Row count of a dataset-independent algebra shape, if statically known.
/// `Extend` (BIND) keeps its child's row count; anything touching the store
/// returns `None`.
fn static_row_count(algebra: &Algebra) -> Option<usize> {
    match algebra {
        Algebra::Values { bindings, .. } => Some(bindings.len()),
        Algebra::Table => Some(1),
        Algebra::Zero | Algebra::Empty => Some(0),
        Algebra::Extend { pattern, .. } => static_row_count(pattern),
        _ => None,
    }
}

/// Variables appearing in term positions of the triple patterns under
/// `algebra` (BGP / Filter-over-BGP shapes only — matches Gate 1).
fn collect_pattern_variables(algebra: &Algebra) -> HashSet<Variable> {
    let mut vars = HashSet::new();
    collect_pattern_variables_into(algebra, &mut vars);
    vars
}

fn collect_pattern_variables_into(algebra: &Algebra, vars: &mut HashSet<Variable>) {
    match algebra {
        Algebra::Bgp(triples) => {
            for t in triples {
                for term in [&t.subject, &t.predicate, &t.object] {
                    if let Term::Variable(v) = term {
                        vars.insert(v.clone());
                    }
                }
            }
        }
        Algebra::Filter { pattern, .. } => collect_pattern_variables_into(pattern, vars),
        _ => {}
    }
}

/// Free variables of every FILTER condition along the inner chain (Gate 4).
/// Returns `None` when a condition contains a sub-pattern we do not analyze
/// (EXISTS / NOT EXISTS) — the caller must then fall back.
fn collect_filter_condition_variables(algebra: &Algebra) -> Option<HashSet<Variable>> {
    let mut vars = HashSet::new();
    let mut node = algebra;
    while let Algebra::Filter { pattern, condition } = node {
        if !collect_expression_variables_into(condition, &mut vars) {
            return None;
        }
        node = pattern;
    }
    Some(vars)
}

/// Collect variables mentioned by `expr` into `vars`; returns `false` when the
/// expression contains an EXISTS/NOT EXISTS sub-pattern (not analyzed here).
fn collect_expression_variables_into(
    expr: &crate::algebra::Expression,
    vars: &mut HashSet<Variable>,
) -> bool {
    use crate::algebra::Expression as E;
    match expr {
        E::Variable(v) | E::Bound(v) => {
            vars.insert(v.clone());
            true
        }
        E::Literal(_) | E::Iri(_) => true,
        E::Function { args, .. } => args
            .iter()
            .all(|a| collect_expression_variables_into(a, vars)),
        E::Binary { left, right, .. } => {
            collect_expression_variables_into(left, vars)
                && collect_expression_variables_into(right, vars)
        }
        E::Unary { operand, .. } => collect_expression_variables_into(operand, vars),
        E::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_expression_variables_into(condition, vars)
                && collect_expression_variables_into(then_expr, vars)
                && collect_expression_variables_into(else_expr, vars)
        }
        E::Exists(_) | E::NotExists(_) => false,
    }
}

/// Merge two bindings, rejecting the pair when a shared variable is bound to
/// different terms (identical to `hash_join`'s per-row compatibility check).
fn compatible_merge(outer: &Binding, inner: &Binding) -> Option<Binding> {
    let mut merged = outer.clone();
    for (var, term) in inner {
        if let Some(existing) = merged.get(var) {
            if existing != term {
                return None;
            }
        } else {
            merged.insert(var.clone(), term.clone());
        }
    }
    Some(merged)
}
