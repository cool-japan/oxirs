//! GQL (ISO/IEC 39075:2024) Abstract Syntax Tree types.
//!
//! This module defines the AST produced by parsing a GQL MATCH query.
//! The subset supported covers node patterns, directed edge patterns,
//! label filters, property filters, an optional WHERE predicate, and
//! a RETURN clause.

/// A complete parsed GQL query.
///
/// Corresponds to:
/// ```text
/// GqlQuery ::= MATCH GraphPattern WHERE? Predicate? RETURN ReturnClause
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GqlQuery {
    /// Alternating Node / Edge segments that make up the MATCH path.
    pub match_pattern: Vec<PathSegment>,
    /// Optional equality predicate following the WHERE keyword.
    pub where_pred: Option<GqlPredicate>,
    /// Variables to project in the RETURN clause.
    pub return_vars: Vec<String>,
}

/// One element in a graph path pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum PathSegment {
    /// A node pattern `(var:Label {prop: value})`.
    Node(NodePattern),
    /// A directed edge pattern `-[var:Label]->` or `<-[var:Label]-`.
    Edge(EdgePattern),
}

/// A node pattern `( VarOpt LabelFilter? PropFilter? )`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    /// Optional variable binding.  `_` is treated as anonymous (None).
    pub var: Option<String>,
    /// Optional `:Label` filter.
    pub label: Option<String>,
    /// Zero or more `{ key: value }` property equality constraints.
    pub props: Vec<(String, GqlLiteral)>,
}

/// A directed edge pattern.
///
/// ```text
/// EdgePattern ::= '-[' VarOpt LabelFilter? ']->'
///               | '<-[' VarOpt LabelFilter? ']-'
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EdgePattern {
    /// Optional variable binding for the edge itself.
    pub var: Option<String>,
    /// Optional `:Label` filter on the edge type / predicate.
    pub label: Option<String>,
    /// Direction of traversal relative to textual left-to-right order.
    pub direction: EdgeDirection,
}

/// Traversal direction of an edge pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// `-[…]->` — left-to-right (subject to object).
    Forward,
    /// `<-[…]-` — right-to-left (object to subject).
    Backward,
}

/// A scalar literal value used inside property filters or WHERE predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum GqlLiteral {
    /// A double-quoted string, e.g. `"Alice"`.
    Str(String),
    /// A 64-bit integer, e.g. `42`.
    Int(i64),
    /// A 64-bit float, e.g. `3.14`.
    Float(f64),
    /// A boolean keyword `true` or `false`.
    Bool(bool),
}

/// An equality predicate in a WHERE clause.
///
/// ```text
/// Predicate ::= Ident '.' Ident '=' Literal
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GqlPredicate {
    /// The variable on the left-hand side of the `.`.
    pub var: String,
    /// The property name on the right-hand side of the `.`.
    pub prop: String,
    /// The value being compared to.
    pub value: GqlLiteral,
}
