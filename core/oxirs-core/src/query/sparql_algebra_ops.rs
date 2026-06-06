//! Algebraic operations over SPARQL graph patterns.
//!
//! Contains constructor helpers and operator application functions for
//! `Join`, `LeftJoin`, `Union`, `Minus`, `Filter`, `Extend`, `Group`, etc.
//! These are *pure* algebraic operations that produce new `GraphPattern` values;
//! they contain no execution or cost-estimation logic.

use super::sparql_algebra_types::{
    AggregateExpression, Expression, GraphPattern, NamedNodePattern, OrderExpression, TermPattern,
    TriplePattern,
};
use crate::model::Variable;

// ────────────────────────────────────────────────────────────────────────────
// Constructor helpers
// ────────────────────────────────────────────────────────────────────────────

/// Construct a BGP (Basic Graph Pattern) from a list of triple patterns.
pub fn bgp(patterns: Vec<TriplePattern>) -> GraphPattern {
    GraphPattern::Bgp { patterns }
}

/// Construct an inner join of two graph patterns.
pub fn join(left: GraphPattern, right: GraphPattern) -> GraphPattern {
    GraphPattern::Join {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Construct an optional (left-join) of two graph patterns.
pub fn left_join(
    left: GraphPattern,
    right: GraphPattern,
    expression: Option<Expression>,
) -> GraphPattern {
    GraphPattern::LeftJoin {
        left: Box::new(left),
        right: Box::new(right),
        expression,
    }
}

/// Construct a filter over a graph pattern.
pub fn filter(expr: Expression, inner: GraphPattern) -> GraphPattern {
    GraphPattern::Filter {
        expr,
        inner: Box::new(inner),
    }
}

/// Construct a union of two graph patterns.
pub fn union(left: GraphPattern, right: GraphPattern) -> GraphPattern {
    GraphPattern::Union {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Construct a named-graph restriction.
pub fn graph(name: NamedNodePattern, inner: GraphPattern) -> GraphPattern {
    GraphPattern::Graph {
        name,
        inner: Box::new(inner),
    }
}

/// Construct an extend (BIND) operation.
pub fn extend(inner: GraphPattern, variable: Variable, expression: Expression) -> GraphPattern {
    GraphPattern::Extend {
        inner: Box::new(inner),
        variable,
        expression,
    }
}

/// Construct a minus (set-difference) operation.
pub fn minus(left: GraphPattern, right: GraphPattern) -> GraphPattern {
    GraphPattern::Minus {
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Construct an ORDER BY node.
pub fn order_by(inner: GraphPattern, expression: Vec<OrderExpression>) -> GraphPattern {
    GraphPattern::OrderBy {
        inner: Box::new(inner),
        expression,
    }
}

/// Construct a PROJECT node.
pub fn project(inner: GraphPattern, variables: Vec<Variable>) -> GraphPattern {
    GraphPattern::Project {
        inner: Box::new(inner),
        variables,
    }
}

/// Construct a DISTINCT node.
pub fn distinct(inner: GraphPattern) -> GraphPattern {
    GraphPattern::Distinct {
        inner: Box::new(inner),
    }
}

/// Construct a REDUCED node.
pub fn reduced(inner: GraphPattern) -> GraphPattern {
    GraphPattern::Reduced {
        inner: Box::new(inner),
    }
}

/// Construct a SLICE (LIMIT/OFFSET) node.
pub fn slice(inner: GraphPattern, start: usize, length: Option<usize>) -> GraphPattern {
    GraphPattern::Slice {
        inner: Box::new(inner),
        start,
        length,
    }
}

/// Construct a GROUP / aggregate node.
pub fn group(
    inner: GraphPattern,
    variables: Vec<Variable>,
    aggregates: Vec<(Variable, AggregateExpression)>,
) -> GraphPattern {
    GraphPattern::Group {
        inner: Box::new(inner),
        variables,
        aggregates,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Pattern inspection utilities
// ────────────────────────────────────────────────────────────────────────────

/// Recursively collect all variables that appear in a `GraphPattern`.
pub fn collect_variables(pattern: &GraphPattern) -> Vec<Variable> {
    let mut vars = Vec::new();
    collect_variables_impl(pattern, &mut vars);
    vars.sort_by(|a, b| a.name().cmp(b.name()));
    vars.dedup();
    vars
}

fn collect_variables_impl(pattern: &GraphPattern, out: &mut Vec<Variable>) {
    match pattern {
        GraphPattern::Bgp { patterns } => {
            for tp in patterns {
                collect_term_pattern_vars(&tp.subject, out);
                collect_term_pattern_vars(&tp.predicate, out);
                collect_term_pattern_vars(&tp.object, out);
            }
        }
        GraphPattern::Path {
            subject, object, ..
        } => {
            collect_term_pattern_vars(subject, out);
            collect_term_pattern_vars(object, out);
        }
        GraphPattern::Join { left, right }
        | GraphPattern::Union { left, right }
        | GraphPattern::Minus { left, right } => {
            collect_variables_impl(left, out);
            collect_variables_impl(right, out);
        }
        GraphPattern::LeftJoin { left, right, .. } => {
            collect_variables_impl(left, out);
            collect_variables_impl(right, out);
        }
        GraphPattern::Filter { inner, .. }
        | GraphPattern::Distinct { inner }
        | GraphPattern::Reduced { inner }
        | GraphPattern::OrderBy { inner, .. }
        | GraphPattern::Slice { inner, .. } => {
            collect_variables_impl(inner, out);
        }
        GraphPattern::Graph { inner, .. } => {
            collect_variables_impl(inner, out);
        }
        GraphPattern::Extend {
            inner, variable, ..
        } => {
            collect_variables_impl(inner, out);
            out.push(variable.clone());
        }
        GraphPattern::Project {
            inner,
            variables: proj_vars,
        } => {
            collect_variables_impl(inner, out);
            out.extend(proj_vars.iter().cloned());
        }
        GraphPattern::Group {
            inner,
            variables: group_vars,
            ..
        } => {
            collect_variables_impl(inner, out);
            out.extend(group_vars.iter().cloned());
        }
        GraphPattern::Values {
            variables: val_vars,
            ..
        } => {
            out.extend(val_vars.iter().cloned());
        }
        GraphPattern::Service { inner, .. } => {
            collect_variables_impl(inner, out);
        }
    }
}

fn collect_term_pattern_vars(tp: &TermPattern, out: &mut Vec<Variable>) {
    if let TermPattern::Variable(v) = tp {
        out.push(v.clone());
    }
}

/// Returns `true` when the pattern contains a SERVICE clause.
pub fn contains_service(pattern: &GraphPattern) -> bool {
    match pattern {
        GraphPattern::Service { .. } => true,
        GraphPattern::Join { left, right }
        | GraphPattern::Union { left, right }
        | GraphPattern::Minus { left, right } => contains_service(left) || contains_service(right),
        GraphPattern::LeftJoin { left, right, .. } => {
            contains_service(left) || contains_service(right)
        }
        GraphPattern::Filter { inner, .. }
        | GraphPattern::Distinct { inner }
        | GraphPattern::Reduced { inner }
        | GraphPattern::Graph { inner, .. }
        | GraphPattern::Extend { inner, .. }
        | GraphPattern::OrderBy { inner, .. }
        | GraphPattern::Project { inner, .. }
        | GraphPattern::Slice { inner, .. }
        | GraphPattern::Group { inner, .. } => contains_service(inner),
        GraphPattern::Bgp { .. } | GraphPattern::Path { .. } | GraphPattern::Values { .. } => false,
    }
}
