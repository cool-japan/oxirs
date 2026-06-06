//! Type definitions for SHACL node expressions (SHACL-AF `sh:values`).
//!
//! This module contains the node expression AST, RDF term type, property path,
//! the minimal graph trait and an in-memory implementation, the evaluation
//! configuration, statistics, error type, and the fluent `ExprBuilder` API.
//!
//! Split from `node_expressions.rs` to keep each source file below the
//! workspace 2000-line refactor threshold while preserving the public API
//! surface re-exported through `super::node_expressions`.
//!
//! ## References
//!
//! - <https://www.w3.org/TR/shacl-af/#node-expressions>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Node expression AST
// ---------------------------------------------------------------------------

/// A SHACL node expression that can be evaluated against a focus node and graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeExpression {
    /// The current focus node itself (`sh:this`).
    FocusNode,

    /// A constant term.
    Constant(NodeTerm),

    /// Follow a property path from the focus node.
    Path(PropertyPath),

    /// Filter nodes through a shape.
    FilterShape {
        /// The set of candidate nodes.
        nodes: Box<NodeExpression>,
        /// The shape IRI to filter against.
        shape: String,
    },

    /// Intersection of multiple node sets.
    Intersection(Vec<NodeExpression>),

    /// Union of multiple node sets.
    Union(Vec<NodeExpression>),

    /// Set difference: nodes in `base` but not in `excluded`.
    Minus {
        /// Base node set.
        base: Box<NodeExpression>,
        /// Nodes to exclude.
        excluded: Box<NodeExpression>,
    },

    /// Conditional: if condition evaluates to non-empty, use `then_expr`,
    /// otherwise use `else_expr`.
    IfThenElse {
        /// Condition expression.
        condition: Box<NodeExpression>,
        /// Expression when condition is true.
        then_expr: Box<NodeExpression>,
        /// Expression when condition is false.
        else_expr: Box<NodeExpression>,
    },

    /// Apply a function to arguments.
    FunctionCall {
        /// Function IRI.
        function: String,
        /// Argument expressions.
        args: Vec<NodeExpression>,
    },

    /// Count the number of nodes from the inner expression.
    Count(Box<NodeExpression>),

    /// Deduplicate results from the inner expression.
    Distinct(Box<NodeExpression>),

    /// Sort results by a property path.
    OrderBy {
        /// Expression to sort.
        expr: Box<NodeExpression>,
        /// Property path to sort by.
        sort_path: PropertyPath,
        /// Ascending order.
        ascending: bool,
    },

    /// Limit the number of results.
    Limit {
        /// Expression to limit.
        expr: Box<NodeExpression>,
        /// Maximum number of results.
        limit: usize,
    },

    /// Skip the first N results.
    Offset {
        /// Expression to offset.
        expr: Box<NodeExpression>,
        /// Number of results to skip.
        offset: usize,
    },

    /// Group-concat: join string values with a separator.
    GroupConcat {
        /// Expression producing the values.
        expr: Box<NodeExpression>,
        /// Separator string.
        separator: String,
    },
}

/// A simplified property path for node expression evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyPath {
    /// A single predicate IRI.
    Predicate(String),
    /// Inverse path (^predicate).
    Inverse(Box<PropertyPath>),
    /// Sequence path (p1 / p2).
    Sequence(Vec<PropertyPath>),
    /// Alternative path (p1 | p2).
    Alternative(Vec<PropertyPath>),
    /// Zero-or-more repetition (p*).
    ZeroOrMore(Box<PropertyPath>),
    /// One-or-more repetition (p+).
    OneOrMore(Box<PropertyPath>),
    /// Zero-or-one (p?).
    ZeroOrOne(Box<PropertyPath>),
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Predicate(iri) => write!(f, "<{iri}>"),
            Self::Inverse(p) => write!(f, "^{p}"),
            Self::Sequence(steps) => {
                let parts: Vec<_> = steps.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" / "))
            }
            Self::Alternative(alts) => {
                let parts: Vec<_> = alts.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" | "))
            }
            Self::ZeroOrMore(p) => write!(f, "{p}*"),
            Self::OneOrMore(p) => write!(f, "{p}+"),
            Self::ZeroOrOne(p) => write!(f, "{p}?"),
        }
    }
}

// ---------------------------------------------------------------------------
// RDF term
// ---------------------------------------------------------------------------

/// An RDF term in node expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum NodeTerm {
    /// An IRI reference.
    Iri(String),
    /// A typed or untyped literal.
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A blank node.
    BlankNode(String),
}

impl fmt::Display for NodeTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Iri(iri) => write!(f, "<{iri}>"),
            Self::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                }
                Ok(())
            }
            Self::BlankNode(id) => write!(f, "_:{id}"),
        }
    }
}

impl NodeTerm {
    /// Create an IRI term.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create a string literal.
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: None,
            lang: None,
        }
    }

    /// Create a typed literal.
    pub fn typed_literal(s: impl Into<String>, dt: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: Some(dt.into()),
            lang: None,
        }
    }

    /// Create a language-tagged literal.
    pub fn lang_literal(s: impl Into<String>, lang: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: None,
            lang: Some(lang.into()),
        }
    }

    /// Check if this is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }

    /// Check if this is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal { .. })
    }

    /// Get the string value of this term.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Iri(s) | Self::BlankNode(s) => s,
            Self::Literal { value, .. } => value,
        }
    }

    /// Try to parse a literal as an integer.
    pub fn as_integer(&self) -> Option<i64> {
        if let Self::Literal { value, .. } = self {
            value.parse().ok()
        } else {
            None
        }
    }

    /// Try to parse a literal as a float.
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Literal { value, .. } = self {
            value.parse().ok()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Simple triple store for node expression evaluation
// ---------------------------------------------------------------------------

/// A triple in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    /// Subject.
    pub subject: NodeTerm,
    /// Predicate.
    pub predicate: NodeTerm,
    /// Object.
    pub object: NodeTerm,
}

/// A minimal graph interface for evaluating node expressions.
pub trait NodeExprGraph {
    /// Get all objects for a given subject and predicate.
    fn objects(&self, subject: &NodeTerm, predicate: &str) -> Vec<NodeTerm>;

    /// Get all subjects for a given predicate and object.
    fn subjects(&self, predicate: &str, object: &NodeTerm) -> Vec<NodeTerm>;

    /// Check if a node conforms to a shape (by IRI).
    fn conforms_to_shape(&self, node: &NodeTerm, shape_iri: &str) -> bool;

    /// Get all triples in the graph.
    fn all_triples(&self) -> Vec<Triple>;
}

/// In-memory graph for testing and simple evaluation.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraph {
    triples: Vec<Triple>,
    /// Shape conformance answers (for testing).
    shape_conformance: HashMap<(String, String), bool>,
}

impl InMemoryGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(&mut self, subject: NodeTerm, predicate: NodeTerm, object: NodeTerm) {
        self.triples.push(Triple {
            subject,
            predicate,
            object,
        });
    }

    /// Add a shape conformance fact (for testing shape filters).
    pub fn set_conforms(&mut self, node: &str, shape: &str, conforms: bool) {
        self.shape_conformance
            .insert((node.to_string(), shape.to_string()), conforms);
    }

    /// Number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

impl NodeExprGraph for InMemoryGraph {
    fn objects(&self, subject: &NodeTerm, predicate: &str) -> Vec<NodeTerm> {
        self.triples
            .iter()
            .filter(|t| {
                &t.subject == subject && matches!(&t.predicate, NodeTerm::Iri(p) if p == predicate)
            })
            .map(|t| t.object.clone())
            .collect()
    }

    fn subjects(&self, predicate: &str, object: &NodeTerm) -> Vec<NodeTerm> {
        self.triples
            .iter()
            .filter(|t| {
                &t.object == object && matches!(&t.predicate, NodeTerm::Iri(p) if p == predicate)
            })
            .map(|t| t.subject.clone())
            .collect()
    }

    fn conforms_to_shape(&self, node: &NodeTerm, shape_iri: &str) -> bool {
        let node_str = node.as_str().to_string();
        self.shape_conformance
            .get(&(node_str, shape_iri.to_string()))
            .copied()
            .unwrap_or(false)
    }

    fn all_triples(&self) -> Vec<Triple> {
        self.triples.clone()
    }
}

// ---------------------------------------------------------------------------
// Configuration, statistics, errors
// ---------------------------------------------------------------------------

/// Configuration for node expression evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Maximum path traversal depth to prevent infinite loops.
    pub max_path_depth: usize,
    /// Maximum number of results before truncation.
    pub max_results: usize,
    /// Enable deduplication by default.
    pub deduplicate: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_path_depth: 100,
            max_results: 10000,
            deduplicate: false,
        }
    }
}

/// Statistics from evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalStats {
    /// Number of graph lookups performed.
    pub graph_lookups: u64,
    /// Number of nodes evaluated.
    pub nodes_evaluated: u64,
    /// Number of path traversals.
    pub path_traversals: u64,
    /// Number of shape checks.
    pub shape_checks: u64,
    /// Number of function calls.
    pub function_calls: u64,
}

/// Errors from node expression evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeExprError {
    /// Path depth exceeded.
    PathDepthExceeded { depth: usize, max: usize },
    /// Result limit exceeded.
    ResultLimitExceeded { count: usize, max: usize },
    /// Unknown function.
    UnknownFunction(String),
    /// Invalid argument count.
    InvalidArgCount {
        function: String,
        expected: usize,
        got: usize,
    },
    /// Type error in computation.
    TypeError(String),
    /// Empty sequence where at least one value was expected.
    EmptySequence(String),
}

impl fmt::Display for NodeExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathDepthExceeded { depth, max } => {
                write!(f, "Path depth {depth} exceeded maximum {max}")
            }
            Self::ResultLimitExceeded { count, max } => {
                write!(f, "Result count {count} exceeded limit {max}")
            }
            Self::UnknownFunction(name) => write!(f, "Unknown function: {name}"),
            Self::InvalidArgCount {
                function,
                expected,
                got,
            } => {
                write!(f, "Function {function} expects {expected} args, got {got}")
            }
            Self::TypeError(msg) => write!(f, "Type error: {msg}"),
            Self::EmptySequence(ctx) => write!(f, "Empty sequence in {ctx}"),
        }
    }
}

impl std::error::Error for NodeExprError {}

// ---------------------------------------------------------------------------
// Expression builder (fluent API)
// ---------------------------------------------------------------------------

/// Fluent builder for constructing node expressions.
pub struct ExprBuilder;

impl ExprBuilder {
    /// The focus node expression.
    pub fn focus() -> NodeExpression {
        NodeExpression::FocusNode
    }

    /// A constant term expression.
    pub fn constant(term: NodeTerm) -> NodeExpression {
        NodeExpression::Constant(term)
    }

    /// A path expression (single predicate).
    pub fn path(predicate: impl Into<String>) -> NodeExpression {
        NodeExpression::Path(PropertyPath::Predicate(predicate.into()))
    }

    /// A sequence path (p1 / p2 / ...).
    pub fn sequence_path(predicates: &[&str]) -> NodeExpression {
        NodeExpression::Path(PropertyPath::Sequence(
            predicates
                .iter()
                .map(|p| PropertyPath::Predicate(p.to_string()))
                .collect(),
        ))
    }

    /// An intersection of expressions.
    pub fn intersection(exprs: Vec<NodeExpression>) -> NodeExpression {
        NodeExpression::Intersection(exprs)
    }

    /// A union of expressions.
    pub fn union(exprs: Vec<NodeExpression>) -> NodeExpression {
        NodeExpression::Union(exprs)
    }

    /// A set-minus expression.
    pub fn minus(base: NodeExpression, excluded: NodeExpression) -> NodeExpression {
        NodeExpression::Minus {
            base: Box::new(base),
            excluded: Box::new(excluded),
        }
    }

    /// An if-then-else expression.
    pub fn if_then_else(
        condition: NodeExpression,
        then_expr: NodeExpression,
        else_expr: NodeExpression,
    ) -> NodeExpression {
        NodeExpression::IfThenElse {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// A count expression.
    pub fn count(inner: NodeExpression) -> NodeExpression {
        NodeExpression::Count(Box::new(inner))
    }

    /// A distinct expression.
    pub fn distinct(inner: NodeExpression) -> NodeExpression {
        NodeExpression::Distinct(Box::new(inner))
    }

    /// A limit expression.
    pub fn limit(inner: NodeExpression, limit: usize) -> NodeExpression {
        NodeExpression::Limit {
            expr: Box::new(inner),
            limit,
        }
    }

    /// An offset expression.
    pub fn offset(inner: NodeExpression, offset: usize) -> NodeExpression {
        NodeExpression::Offset {
            expr: Box::new(inner),
            offset,
        }
    }
}
