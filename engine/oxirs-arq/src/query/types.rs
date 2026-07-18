//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{
    Aggregate, Algebra, Expression, GroupCondition, Iri, OrderCondition, TriplePattern, Variable,
};
use crate::update::UpdateOperation;
use std::collections::HashMap;

/// SPARQL query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
}
/// A single resource named by a `DESCRIBE` query.
///
/// `DESCRIBE <iri> ?var …` may mix explicit IRIs (already expanded against the
/// query prologue) and variables that resolve to bound resources in the WHERE
/// clause. `DESCRIBE *` is represented out-of-band by [`Query::describe_all`]
/// and leaves [`Query::describe_targets`] empty.
#[derive(Debug, Clone, PartialEq)]
pub enum DescribeTarget {
    /// A concrete IRI target (prefixed names are expanded before storage).
    Iri(Iri),
    /// A variable target whose bindings supply the resources to describe.
    Variable(Variable),
}
/// A single item in a SELECT projection list.
///
/// A projection is an ordered list of these. Plain `?x` produces
/// [`ProjectionItem::Variable`]; a parenthesized `(Expression AS ?v)` produces
/// [`ProjectionItem::Expression`] for a non-aggregate expression or
/// [`ProjectionItem::Aggregate`] when the expression is a SPARQL aggregate
/// (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, `SAMPLE`, `GROUP_CONCAT`). The
/// aggregate variant carries the crate's [`Aggregate`] value directly so a
/// consumer can assemble an `Algebra::Group { aggregates, .. }` without
/// re-parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionItem {
    /// Plain variable projection: `SELECT ?x`.
    Variable(Variable),
    /// Non-aggregate expression projection: `(?a + 1 AS ?b)`.
    Expression {
        /// The projected expression.
        expr: Expression,
        /// The variable the expression result is bound to.
        alias: Variable,
    },
    /// Aggregate projection: `(COUNT(*) AS ?n)`, `(SUM(?a * ?b) AS ?t)`, ….
    Aggregate {
        /// The parsed aggregate (carries DISTINCT / separator / `*` state).
        aggregate: Aggregate,
        /// The variable the aggregate result is bound to.
        alias: Variable,
    },
}
/// Dataset clause for FROM and FROM NAMED
#[derive(Debug, Clone, Default)]
pub struct DatasetClause {
    pub default_graphs: Vec<Iri>,
    pub named_graphs: Vec<Iri>,
}
/// SPARQL UPDATE request representation
#[derive(Debug, Clone)]
pub struct UpdateRequest {
    pub operations: Vec<UpdateOperation>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
}
/// Token types for SPARQL parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Select,
    Construct,
    Ask,
    Describe,
    Where,
    Optional,
    Union,
    Minus,
    Filter,
    Bind,
    Service,
    Graph,
    From,
    Named,
    Prefix,
    Base,
    Distinct,
    Reduced,
    OrderBy,
    GroupBy,
    Having,
    Limit,
    Offset,
    Asc,
    Desc,
    As,
    Values,
    Exists,
    NotExists,
    Insert,
    Delete,
    Update,
    Create,
    Drop,
    Clear,
    Load,
    Copy,
    Move,
    Add,
    Data,
    With,
    Using,
    Silent,
    All,
    Default,
    To,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
    Plus,
    Minus_,
    Multiply,
    Divide,
    Pipe,
    Caret,
    Slash,
    Question,
    Star,
    Bang,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Dot,
    Semicolon,
    Comma,
    Colon,
    Iri(String),
    PrefixedName(String, String),
    /// A SPARQL 1.1 built-in call name (e.g. `LANG`, `isIRI`, `REGEX`),
    /// carrying its canonical lower-case spelling. Only a *bare* (colon-free)
    /// keyword is classified as a built-in; a leading-colon default-prefix name
    /// such as `:lang` stays a [`Token::PrefixedName`] so user functions are
    /// never mistaken for built-ins.
    BuiltIn(String),
    /// The bare `a` keyword: the `rdf:type` predicate shorthand. Only a bare
    /// (colon-free) lowercase `a` is classified here; `?a`, `:a`, `"a"` and
    /// `a:` (a prefix name) are never affected.
    A,
    Variable(String),
    StringLiteral(String),
    /// A string literal carrying a language tag (`"foo"@ja`) or an explicit
    /// datatype (`"1"^^xsd:integer`). `datatype` holds the raw form as written
    /// (an absolute IRI, or a `prefix:local` name resolved at parse time).
    RdfLiteral {
        value: String,
        language: Option<String>,
        datatype: Option<String>,
    },
    NumericLiteral(String),
    BooleanLiteral(bool),
    BlankNode(String),
    Eof,
    Newline,
}
/// SPARQL query representation
#[derive(Debug, Clone)]
pub struct Query {
    pub query_type: QueryType,
    pub select_variables: Vec<Variable>,
    pub where_clause: Algebra,
    pub order_by: Vec<OrderCondition>,
    pub group_by: Vec<GroupCondition>,
    pub having: Option<Expression>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub distinct: bool,
    pub reduced: bool,
    pub construct_template: Vec<TriplePattern>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
    pub dataset: DatasetClause,
    /// Ordered SELECT projection items (parallel to `select_variables`, but
    /// also carrying `(Expression AS ?v)` / aggregate projections). Empty when
    /// the query is `SELECT *` or is not a SELECT query.
    pub projection_items: Vec<ProjectionItem>,
    /// Explicit `DESCRIBE` targets (IRIs and variables). Empty for non-DESCRIBE
    /// queries and for `DESCRIBE *` (see `describe_all`).
    pub describe_targets: Vec<DescribeTarget>,
    /// `true` for `DESCRIBE *`, which describes every in-scope variable binding
    /// rather than an explicit target list.
    pub describe_all: bool,
}
