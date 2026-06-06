//! SPARQL graph pattern algebra node.
//!
//! Contains [`GraphPattern`] and its `Display`/SSE implementations. The pattern
//! references expression, term-pattern, ground-term, ordering and aggregate
//! types defined in the sibling modules.

use super::sparql_algebra_types_expr::Expression;
use super::sparql_algebra_types_paths::PropertyPathExpression;
use super::sparql_algebra_types_terms::{
    AggregateExpression, GroundTerm, NamedNodePattern, OrderExpression, TermPattern, TriplePattern,
};
use crate::model::*;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// GraphPattern
// ────────────────────────────────────────────────────────────────────────────

/// A [SPARQL graph pattern](https://www.w3.org/TR/sparql11-query/#GraphPattern)
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GraphPattern {
    /// A [basic graph pattern](https://www.w3.org/TR/sparql11-query/#defn_BasicGraphPattern).
    Bgp { patterns: Vec<TriplePattern> },
    /// A [property path pattern](https://www.w3.org/TR/sparql11-query/#defn_evalPP_predicate).
    Path {
        subject: TermPattern,
        path: PropertyPathExpression,
        object: TermPattern,
    },
    /// [Join](https://www.w3.org/TR/sparql11-query/#defn_algJoin).
    Join { left: Box<Self>, right: Box<Self> },
    /// [LeftJoin](https://www.w3.org/TR/sparql11-query/#defn_algLeftJoin).
    LeftJoin {
        left: Box<Self>,
        right: Box<Self>,
        expression: Option<Expression>,
    },
    /// [Filter](https://www.w3.org/TR/sparql11-query/#defn_algFilter).
    Filter { expr: Expression, inner: Box<Self> },
    /// [Union](https://www.w3.org/TR/sparql11-query/#defn_algUnion).
    Union { left: Box<Self>, right: Box<Self> },
    /// Graph pattern (GRAPH clause)
    Graph {
        name: NamedNodePattern,
        inner: Box<Self>,
    },
    /// [Extend](https://www.w3.org/TR/sparql11-query/#defn_extend).
    Extend {
        inner: Box<Self>,
        variable: Variable,
        expression: Expression,
    },
    /// [Minus](https://www.w3.org/TR/sparql11-query/#defn_algMinus).
    Minus { left: Box<Self>, right: Box<Self> },
    /// A table used to provide inline values
    Values {
        variables: Vec<Variable>,
        bindings: Vec<Vec<Option<GroundTerm>>>,
    },
    /// [OrderBy](https://www.w3.org/TR/sparql11-query/#defn_algOrdered).
    OrderBy {
        inner: Box<Self>,
        expression: Vec<OrderExpression>,
    },
    /// [Project](https://www.w3.org/TR/sparql11-query/#defn_algProjection).
    Project {
        inner: Box<Self>,
        variables: Vec<Variable>,
    },
    /// [Distinct](https://www.w3.org/TR/sparql11-query/#defn_algDistinct).
    Distinct { inner: Box<Self> },
    /// [Reduced](https://www.w3.org/TR/sparql11-query/#defn_algReduced).
    Reduced { inner: Box<Self> },
    /// [Slice](https://www.w3.org/TR/sparql11-query/#defn_algSlice).
    Slice {
        inner: Box<Self>,
        start: usize,
        length: Option<usize>,
    },
    /// [Group](https://www.w3.org/TR/sparql11-query/#aggregateAlgebra).
    Group {
        inner: Box<Self>,
        variables: Vec<Variable>,
        aggregates: Vec<(Variable, AggregateExpression)>,
    },
    /// [Service](https://www.w3.org/TR/sparql11-federated-query/#defn_evalService).
    Service {
        name: NamedNodePattern,
        inner: Box<Self>,
        silent: bool,
    },
}

impl fmt::Display for GraphPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bgp { patterns } => {
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" . ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                Ok(())
            }
            Self::Path {
                subject,
                path,
                object,
            } => {
                write!(f, "{subject} {path} {object}")
            }
            Self::Join { left, right } => {
                write!(f, "{left} . {right}")
            }
            Self::LeftJoin {
                left,
                right,
                expression,
            } => {
                write!(f, "{left} OPTIONAL {{ {right}")?;
                if let Some(expr) = expression {
                    write!(f, " FILTER ({expr})")?;
                }
                f.write_str(" }")
            }
            Self::Filter { expr, inner } => {
                write!(f, "{inner} FILTER ({expr})")
            }
            Self::Union { left, right } => {
                write!(f, "{{ {left} }} UNION {{ {right} }}")
            }
            Self::Graph { name, inner } => {
                write!(f, "GRAPH {name} {{ {inner} }}")
            }
            Self::Extend {
                inner,
                variable,
                expression,
            } => {
                write!(f, "{inner} BIND ({expression} AS {variable})")
            }
            Self::Minus { left, right } => {
                write!(f, "{left} MINUS {{ {right} }}")
            }
            Self::Values {
                variables,
                bindings,
            } => {
                f.write_str("VALUES ")?;
                if variables.len() == 1 {
                    write!(f, "{}", variables[0])?;
                } else {
                    f.write_str("(")?;
                    for (i, var) in variables.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        write!(f, "{var}")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(" { ")?;
                for (i, binding) in bindings.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    if variables.len() == 1 {
                        if let Some(term) = &binding[0] {
                            write!(f, "{term}")?;
                        } else {
                            f.write_str("UNDEF")?;
                        }
                    } else {
                        f.write_str("(")?;
                        for (j, value) in binding.iter().enumerate() {
                            if j > 0 {
                                f.write_str(" ")?;
                            }
                            if let Some(term) = value {
                                write!(f, "{term}")?;
                            } else {
                                f.write_str("UNDEF")?;
                            }
                        }
                        f.write_str(")")?;
                    }
                }
                f.write_str(" }")
            }
            Self::OrderBy { inner, expression } => {
                write!(f, "{inner} ORDER BY")?;
                for expr in expression {
                    write!(f, " {expr}")?;
                }
                Ok(())
            }
            Self::Project { inner, variables } => {
                f.write_str("SELECT ")?;
                for (i, var) in variables.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{var}")?;
                }
                write!(f, " WHERE {{ {inner} }}")
            }
            Self::Distinct { inner } => {
                write!(f, "SELECT DISTINCT * WHERE {{ {inner} }}")
            }
            Self::Reduced { inner } => {
                write!(f, "SELECT REDUCED * WHERE {{ {inner} }}")
            }
            Self::Slice {
                inner,
                start,
                length,
            } => {
                write!(f, "{inner} OFFSET {start}")?;
                if let Some(length) = length {
                    write!(f, " LIMIT {length}")?;
                }
                Ok(())
            }
            Self::Group {
                inner,
                variables,
                aggregates,
            } => {
                write!(f, "{inner} GROUP BY")?;
                for var in variables {
                    write!(f, " {var}")?;
                }
                if !aggregates.is_empty() {
                    f.write_str(" HAVING")?;
                    for (var, agg) in aggregates {
                        write!(f, " ({agg} AS {var})")?;
                    }
                }
                Ok(())
            }
            Self::Service {
                name,
                inner,
                silent,
            } => {
                if *silent {
                    write!(f, "SERVICE SILENT {name} {{ {inner} }}")
                } else {
                    write!(f, "SERVICE {name} {{ {inner} }}")
                }
            }
        }
    }
}

impl GraphPattern {
    /// Formats using the SPARQL S-Expression syntax
    pub fn fmt_sse(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Bgp { patterns } => {
                f.write_str("(bgp")?;
                for pattern in patterns {
                    f.write_str(" ")?;
                    pattern.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Path {
                subject,
                path,
                object,
            } => {
                f.write_str("(path ")?;
                subject.fmt_sse(f)?;
                f.write_str(" ")?;
                path.fmt_sse(f)?;
                f.write_str(" ")?;
                object.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Join { left, right } => {
                f.write_str("(join ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::LeftJoin {
                left,
                right,
                expression,
            } => {
                f.write_str("(leftjoin ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                if let Some(expr) = expression {
                    f.write_str(" ")?;
                    expr.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Filter { expr, inner } => {
                f.write_str("(filter ")?;
                expr.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Union { left, right } => {
                f.write_str("(union ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Graph { name, inner } => {
                f.write_str("(graph ")?;
                name.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Extend {
                inner,
                variable,
                expression,
            } => {
                f.write_str("(extend ")?;
                inner.fmt_sse(f)?;
                f.write_str(" (")?;
                variable.fmt_sse(f)?;
                f.write_str(" ")?;
                expression.fmt_sse(f)?;
                f.write_str("))")
            }
            Self::Minus { left, right } => {
                f.write_str("(minus ")?;
                left.fmt_sse(f)?;
                f.write_str(" ")?;
                right.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Values {
                variables,
                bindings,
            } => {
                f.write_str("(table")?;
                if !variables.is_empty() {
                    f.write_str(" (vars")?;
                    for var in variables {
                        f.write_str(" ")?;
                        var.fmt_sse(f)?;
                    }
                    f.write_str(")")?;
                }
                for binding in bindings {
                    f.write_str(" (row")?;
                    for (i, value) in binding.iter().enumerate() {
                        f.write_str(" (")?;
                        variables[i].fmt_sse(f)?;
                        f.write_str(" ")?;
                        if let Some(term) = value {
                            term.fmt_sse(f)?;
                        } else {
                            f.write_str("UNDEF")?;
                        }
                        f.write_str(")")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(")")
            }
            Self::OrderBy { inner, expression } => {
                f.write_str("(order ")?;
                inner.fmt_sse(f)?;
                for expr in expression {
                    f.write_str(" ")?;
                    expr.fmt_sse(f)?;
                }
                f.write_str(")")
            }
            Self::Project { inner, variables } => {
                f.write_str("(project ")?;
                inner.fmt_sse(f)?;
                f.write_str(" (")?;
                for (i, var) in variables.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    var.fmt_sse(f)?;
                }
                f.write_str("))")
            }
            Self::Distinct { inner } => {
                f.write_str("(distinct ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Reduced { inner } => {
                f.write_str("(reduced ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
            Self::Slice {
                inner,
                start,
                length,
            } => {
                f.write_str("(slice ")?;
                inner.fmt_sse(f)?;
                write!(f, " {start}")?;
                if let Some(length) = length {
                    write!(f, " {length}")?;
                }
                f.write_str(")")
            }
            Self::Group {
                inner,
                variables,
                aggregates,
            } => {
                f.write_str("(group ")?;
                inner.fmt_sse(f)?;
                if !variables.is_empty() {
                    f.write_str(" (")?;
                    for (i, var) in variables.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        var.fmt_sse(f)?;
                    }
                    f.write_str(")")?;
                }
                if !aggregates.is_empty() {
                    f.write_str(" (")?;
                    for (i, (var, agg)) in aggregates.iter().enumerate() {
                        if i > 0 {
                            f.write_str(" ")?;
                        }
                        f.write_str("(")?;
                        var.fmt_sse(f)?;
                        f.write_str(" ")?;
                        agg.fmt_sse(f)?;
                        f.write_str(")")?;
                    }
                    f.write_str(")")?;
                }
                f.write_str(")")
            }
            Self::Service {
                name,
                inner,
                silent,
            } => {
                if *silent {
                    f.write_str("(service silent ")?;
                } else {
                    f.write_str("(service ")?;
                }
                name.fmt_sse(f)?;
                f.write_str(" ")?;
                inner.fmt_sse(f)?;
                f.write_str(")")
            }
        }
    }
}
