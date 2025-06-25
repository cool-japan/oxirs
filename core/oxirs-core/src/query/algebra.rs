//! SPARQL 1.1 Query Algebra representation
//! 
//! Based on the W3C SPARQL 1.1 Query specification:
//! https://www.w3.org/TR/sparql11-query/#sparqlQuery

use crate::model::*;
use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A property path expression for navigating RDF graphs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyPath {
    /// A simple predicate path
    Predicate(NamedNode),
    /// Inverse path: ^path
    Inverse(Box<PropertyPath>),
    /// Sequence path: path1 / path2
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative path: path1 | path2
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero or more: path*
    ZeroOrMore(Box<PropertyPath>),
    /// One or more: path+
    OneOrMore(Box<PropertyPath>),
    /// Zero or one: path?
    ZeroOrOne(Box<PropertyPath>),
    /// Negated property set: !(p1 | p2 | ...)
    NegatedPropertySet(Vec<NamedNode>),
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyPath::Predicate(p) => write!(f, "{}", p),
            PropertyPath::Inverse(p) => write!(f, "^{}", p),
            PropertyPath::Sequence(a, b) => write!(f, "({} / {})", a, b),
            PropertyPath::Alternative(a, b) => write!(f, "({} | {})", a, b),
            PropertyPath::ZeroOrMore(p) => write!(f, "({})*", p),
            PropertyPath::OneOrMore(p) => write!(f, "({})+", p),
            PropertyPath::ZeroOrOne(p) => write!(f, "({})?", p),
            PropertyPath::NegatedPropertySet(ps) => {
                write!(f, "!(")?;
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A SPARQL expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// A constant term
    Term(Term),
    /// A variable
    Variable(Variable),
    /// Logical AND
    And(Box<Expression>, Box<Expression>),
    /// Logical OR
    Or(Box<Expression>, Box<Expression>),
    /// Logical NOT
    Not(Box<Expression>),
    /// Equality: =
    Equal(Box<Expression>, Box<Expression>),
    /// Inequality: !=
    NotEqual(Box<Expression>, Box<Expression>),
    /// Less than: <
    Less(Box<Expression>, Box<Expression>),
    /// Less than or equal: <=
    LessOrEqual(Box<Expression>, Box<Expression>),
    /// Greater than: >
    Greater(Box<Expression>, Box<Expression>),
    /// Greater than or equal: >=
    GreaterOrEqual(Box<Expression>, Box<Expression>),
    /// Addition: +
    Add(Box<Expression>, Box<Expression>),
    /// Subtraction: -
    Subtract(Box<Expression>, Box<Expression>),
    /// Multiplication: *
    Multiply(Box<Expression>, Box<Expression>),
    /// Division: /
    Divide(Box<Expression>, Box<Expression>),
    /// Unary plus: +expr
    UnaryPlus(Box<Expression>),
    /// Unary minus: -expr
    UnaryMinus(Box<Expression>),
    /// IN expression
    In(Box<Expression>, Vec<Expression>),
    /// NOT IN expression
    NotIn(Box<Expression>, Vec<Expression>),
    /// EXISTS pattern
    Exists(Box<GraphPattern>),
    /// NOT EXISTS pattern
    NotExists(Box<GraphPattern>),
    /// Function call
    FunctionCall(Function, Vec<Expression>),
    /// Bound variable test
    Bound(Variable),
    /// IF expression
    If(Box<Expression>, Box<Expression>, Box<Expression>),
    /// COALESCE expression
    Coalesce(Vec<Expression>),
}

/// Built-in SPARQL functions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Function {
    // String functions
    Str,
    Lang,
    LangMatches,
    Datatype,
    Iri,
    Bnode,
    StrDt,
    StrLang,
    StrLen,
    SubStr,
    UCase,
    LCase,
    StrStarts,
    StrEnds,
    Contains,
    StrBefore,
    StrAfter,
    Encode,
    Concat,
    Replace,
    Regex,
    
    // Numeric functions
    Abs,
    Round,
    Ceil,
    Floor,
    Rand,
    
    // Date/Time functions
    Now,
    Year,
    Month,
    Day,
    Hours,
    Minutes,
    Seconds,
    Timezone,
    Tz,
    
    // Hash functions
    Md5,
    Sha1,
    Sha256,
    Sha384,
    Sha512,
    
    // Type checking
    IsIri,
    IsBlank,
    IsLiteral,
    IsNumeric,
    
    // Custom function
    Custom(NamedNode),
}

/// A triple pattern in a graph pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: TermPattern,
    pub predicate: TermPattern,
    pub object: TermPattern,
}

/// A term pattern (can be a concrete term or variable)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TermPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
}

impl From<Variable> for TermPattern {
    fn from(v: Variable) -> Self {
        TermPattern::Variable(v)
    }
}

impl From<NamedNode> for TermPattern {
    fn from(n: NamedNode) -> Self {
        TermPattern::NamedNode(n)
    }
}

impl From<BlankNode> for TermPattern {
    fn from(b: BlankNode) -> Self {
        TermPattern::BlankNode(b)
    }
}

impl From<Literal> for TermPattern {
    fn from(l: Literal) -> Self {
        TermPattern::Literal(l)
    }
}

/// A graph pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphPattern {
    /// Basic graph pattern (set of triple patterns)
    Bgp(Vec<TriplePattern>),
    /// Path pattern
    Path {
        subject: TermPattern,
        path: PropertyPath,
        object: TermPattern,
    },
    /// Join of two patterns
    Join(Box<GraphPattern>, Box<GraphPattern>),
    /// Left join (OPTIONAL)
    LeftJoin {
        left: Box<GraphPattern>,
        right: Box<GraphPattern>,
        condition: Option<Expression>,
    },
    /// Filter pattern
    Filter {
        expr: Expression,
        inner: Box<GraphPattern>,
    },
    /// Union of patterns
    Union(Box<GraphPattern>, Box<GraphPattern>),
    /// Graph pattern (GRAPH)
    Graph {
        graph_name: TermPattern,
        inner: Box<GraphPattern>,
    },
    /// Service pattern (federated query)
    Service {
        service: TermPattern,
        inner: Box<GraphPattern>,
        silent: bool,
    },
    /// Group pattern
    Group {
        inner: Box<GraphPattern>,
        variables: Vec<Variable>,
        aggregates: Vec<(Variable, AggregateExpression)>,
    },
    /// Extend pattern (BIND)
    Extend {
        inner: Box<GraphPattern>,
        variable: Variable,
        expression: Expression,
    },
    /// Minus pattern
    Minus(Box<GraphPattern>, Box<GraphPattern>),
    /// Values pattern
    Values {
        variables: Vec<Variable>,
        bindings: Vec<Vec<Option<Term>>>,
    },
    /// Order by pattern
    OrderBy {
        inner: Box<GraphPattern>,
        order_by: Vec<OrderExpression>,
    },
    /// Project pattern
    Project {
        inner: Box<GraphPattern>,
        variables: Vec<Variable>,
    },
    /// Distinct pattern
    Distinct(Box<GraphPattern>),
    /// Reduced pattern
    Reduced(Box<GraphPattern>),
    /// Slice pattern (LIMIT/OFFSET)
    Slice {
        inner: Box<GraphPattern>,
        offset: usize,
        limit: Option<usize>,
    },
}

/// Aggregate expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateExpression {
    Count {
        expr: Option<Box<Expression>>,
        distinct: bool,
    },
    Sum {
        expr: Box<Expression>,
        distinct: bool,
    },
    Avg {
        expr: Box<Expression>,
        distinct: bool,
    },
    Min {
        expr: Box<Expression>,
        distinct: bool,
    },
    Max {
        expr: Box<Expression>,
        distinct: bool,
    },
    GroupConcat {
        expr: Box<Expression>,
        distinct: bool,
        separator: Option<String>,
    },
    Sample {
        expr: Box<Expression>,
        distinct: bool,
    },
}

/// Order expression for ORDER BY
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderExpression {
    Asc(Expression),
    Desc(Expression),
}

/// SPARQL query forms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryForm {
    /// SELECT query
    Select {
        /// SELECT * or specific variables
        variables: SelectVariables,
        /// WHERE clause pattern
        where_clause: GraphPattern,
        /// Solution modifiers
        distinct: bool,
        reduced: bool,
        order_by: Vec<OrderExpression>,
        offset: usize,
        limit: Option<usize>,
    },
    /// CONSTRUCT query
    Construct {
        /// Template for constructing triples
        template: Vec<TriplePattern>,
        /// WHERE clause pattern
        where_clause: GraphPattern,
        /// Solution modifiers
        order_by: Vec<OrderExpression>,
        offset: usize,
        limit: Option<usize>,
    },
    /// DESCRIBE query
    Describe {
        /// Resources to describe
        resources: Vec<TermPattern>,
        /// Optional WHERE clause
        where_clause: Option<GraphPattern>,
        /// Solution modifiers
        order_by: Vec<OrderExpression>,
        offset: usize,
        limit: Option<usize>,
    },
    /// ASK query
    Ask {
        /// Pattern to check
        where_clause: GraphPattern,
    },
}

/// Variables selection in SELECT
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectVariables {
    /// SELECT *
    All,
    /// SELECT ?var1 ?var2 ...
    Specific(Vec<Variable>),
}

/// A complete SPARQL query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// Base IRI for relative IRI resolution
    pub base: Option<NamedNode>,
    /// Namespace prefixes
    pub prefixes: HashMap<String, NamedNode>,
    /// Query form
    pub form: QueryForm,
    /// Dataset specification
    pub dataset: Dataset,
}

/// Dataset specification for a query
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Dataset {
    /// Default graph IRIs (FROM)
    pub default: Vec<NamedNode>,
    /// Named graph IRIs (FROM NAMED)
    pub named: Vec<NamedNode>,
}

/// SPARQL Update operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UpdateOperation {
    /// INSERT DATA
    InsertData {
        data: Vec<Quad>,
    },
    /// DELETE DATA
    DeleteData {
        data: Vec<Quad>,
    },
    /// DELETE WHERE
    DeleteWhere {
        pattern: Vec<QuadPattern>,
    },
    /// INSERT/DELETE with WHERE
    Modify {
        delete: Option<Vec<QuadPattern>>,
        insert: Option<Vec<QuadPattern>>,
        where_clause: GraphPattern,
        using: Dataset,
    },
    /// LOAD
    Load {
        source: NamedNode,
        destination: Option<NamedNode>,
        silent: bool,
    },
    /// CLEAR
    Clear {
        graph: GraphTarget,
        silent: bool,
    },
    /// CREATE
    Create {
        graph: NamedNode,
        silent: bool,
    },
    /// DROP
    Drop {
        graph: GraphTarget,
        silent: bool,
    },
    /// COPY
    Copy {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
    /// MOVE
    Move {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
    /// ADD
    Add {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
}

/// Graph targets for update operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphTarget {
    Default,
    Named(NamedNode),
    All,
}

/// Quad pattern for updates
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuadPattern {
    pub subject: TermPattern,
    pub predicate: TermPattern,
    pub object: TermPattern,
    pub graph: Option<TermPattern>,
}

/// A SPARQL Update request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Update {
    /// Base IRI for relative IRI resolution
    pub base: Option<NamedNode>,
    /// Namespace prefixes
    pub prefixes: HashMap<String, NamedNode>,
    /// Update operations
    pub operations: Vec<UpdateOperation>,
}