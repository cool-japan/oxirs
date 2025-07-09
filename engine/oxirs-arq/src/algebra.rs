//! SPARQL Algebra Module
//!
//! This module provides the core algebraic representation of SPARQL queries,
//! including basic graph patterns, joins, unions, filters, and other operations.

use oxirs_core::model::{
    BlankNode as CoreBlankNode, Literal as CoreLiteral, NamedNode, Object, Predicate, Subject,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Variable identifier - reuse from core
pub use oxirs_core::model::Variable;

/// IRI (Internationalized Resource Identifier) - use NamedNode from core
pub type Iri = NamedNode;

/// Literal value - create a bridge type that can convert to/from core literal
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Literal {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Option<NamedNode>,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.value)?;
        if let Some(lang) = &self.language {
            write!(f, "@{lang}")?;
        } else if let Some(dt) = &self.datatype {
            write!(f, "^^{dt}")?;
        }
        Ok(())
    }
}

impl Literal {
    /// Create a language-tagged literal
    pub fn with_language(value: String, language: String) -> Self {
        Self {
            value,
            language: Some(language),
            datatype: None,
        }
    }
}

impl From<CoreLiteral> for Literal {
    fn from(core_literal: CoreLiteral) -> Self {
        let (value, datatype, language) = core_literal.destruct();
        Self {
            value,
            language,
            datatype,
        }
    }
}

impl From<Literal> for CoreLiteral {
    fn from(literal: Literal) -> Self {
        if let Some(lang) = literal.language {
            CoreLiteral::new_language_tagged_literal(&literal.value, lang)
                .unwrap_or_else(|_| CoreLiteral::new_simple_literal(literal.value))
        } else if let Some(datatype) = literal.datatype {
            CoreLiteral::new_typed_literal(literal.value, datatype)
        } else {
            CoreLiteral::new_simple_literal(literal.value)
        }
    }
}

/// RDF term (subject, predicate, or object) - bridge with core types
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Term {
    Variable(Variable),
    Iri(NamedNode),
    Literal(Literal),
    BlankNode(String),
    QuotedTriple(Box<TriplePattern>),
    PropertyPath(PropertyPath),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(v) => write!(f, "?{v}"),
            Term::Iri(iri) => write!(f, "{iri}"),
            Term::Literal(lit) => write!(f, "{lit}"),
            Term::BlankNode(id) => write!(f, "_:{id}"),
            Term::QuotedTriple(triple) => write!(
                f,
                "<<{} {} {}>>",
                triple.subject, triple.predicate, triple.object
            ),
            Term::PropertyPath(path) => write!(f, "{path}"),
        }
    }
}

impl From<Subject> for Term {
    fn from(subject: Subject) -> Self {
        match subject {
            Subject::NamedNode(n) => Term::Iri(n),
            Subject::BlankNode(b) => Term::BlankNode(b.id().to_string()),
            Subject::Variable(v) => Term::Variable(v),
            Subject::QuotedTriple(quoted_triple) => {
                // Implement proper quoted triple support for RDF-star
                Term::QuotedTriple(Box::new(TriplePattern {
                    subject: Term::from(quoted_triple.subject().clone()),
                    predicate: Term::from(quoted_triple.predicate().clone()),
                    object: Term::from(quoted_triple.object().clone()),
                }))
            }
        }
    }
}

impl From<Predicate> for Term {
    fn from(predicate: Predicate) -> Self {
        match predicate {
            Predicate::NamedNode(n) => Term::Iri(n),
            Predicate::Variable(v) => Term::Variable(v),
        }
    }
}

impl From<Object> for Term {
    fn from(object: Object) -> Self {
        match object {
            Object::NamedNode(n) => Term::Iri(n),
            Object::BlankNode(b) => Term::BlankNode(b.id().to_string()),
            Object::Literal(l) => Term::Literal(l.into()),
            Object::Variable(v) => Term::Variable(v),
            Object::QuotedTriple(quoted_triple) => {
                // Implement proper quoted triple support for RDF-star
                Term::QuotedTriple(Box::new(TriplePattern {
                    subject: Term::from(quoted_triple.subject().clone()),
                    predicate: Term::from(quoted_triple.predicate().clone()),
                    object: Term::from(quoted_triple.object().clone()),
                }))
            }
        }
    }
}

impl From<NamedNode> for Term {
    fn from(node: NamedNode) -> Self {
        Term::Iri(node)
    }
}

impl From<CoreBlankNode> for Term {
    fn from(node: CoreBlankNode) -> Self {
        Term::BlankNode(node.id().to_string())
    }
}

impl From<CoreLiteral> for Term {
    fn from(literal: CoreLiteral) -> Self {
        Term::Literal(literal.into())
    }
}

impl From<Variable> for Term {
    fn from(variable: Variable) -> Self {
        Term::Variable(variable)
    }
}

/// Triple pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

impl fmt::Display for TriplePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// SPARQL expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expression {
    /// Variable reference
    Variable(Variable),
    /// Literal value
    Literal(Literal),
    /// IRI reference
    Iri(Iri),
    /// Function call
    Function { name: String, args: Vec<Expression> },
    /// Binary operation
    Binary {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    /// Unary operation
    Unary {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
    /// Conditional expression (IF)
    Conditional {
        condition: Box<Expression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
    },
    /// Bound variable check
    Bound(Variable),
    /// Exists clause
    Exists(Box<Algebra>),
    /// Not exists clause
    NotExists(Box<Algebra>),
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    SameTerm,
    In,
    NotIn,
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Equal => write!(f, "="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::Less => write!(f, "<"),
            BinaryOperator::LessEqual => write!(f, "<="),
            BinaryOperator::Greater => write!(f, ">"),
            BinaryOperator::GreaterEqual => write!(f, ">="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
            BinaryOperator::SameTerm => write!(f, "sameTerm"),
            BinaryOperator::In => write!(f, "IN"),
            BinaryOperator::NotIn => write!(f, "NOT IN"),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Plus,
    Minus,
    IsIri,
    IsBlank,
    IsLiteral,
    IsNumeric,
}

/// Aggregate function
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregate {
    Count {
        distinct: bool,
        expr: Option<Expression>,
    },
    Sum {
        distinct: bool,
        expr: Expression,
    },
    Min {
        distinct: bool,
        expr: Expression,
    },
    Max {
        distinct: bool,
        expr: Expression,
    },
    Avg {
        distinct: bool,
        expr: Expression,
    },
    Sample {
        distinct: bool,
        expr: Expression,
    },
    GroupConcat {
        distinct: bool,
        expr: Expression,
        separator: Option<String>,
    },
}

/// Variable binding
pub type Binding = HashMap<Variable, Term>;

/// Solution sequence (set of bindings)
pub type Solution = Vec<Binding>;

/// Order condition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderCondition {
    pub expr: Expression,
    pub ascending: bool,
}

/// Group condition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupCondition {
    pub expr: Expression,
    pub alias: Option<Variable>,
}

/// Property path expressions for advanced SPARQL 1.1 graph navigation
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Direct property IRI
    Iri(Iri),
    /// Variable property
    Variable(Variable),
    /// Inverse property path (^property)
    Inverse(Box<PropertyPath>),
    /// Sequence path (path1/path2)
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative path (path1|path2)
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero or more (path*)
    ZeroOrMore(Box<PropertyPath>),
    /// One or more (path+)
    OneOrMore(Box<PropertyPath>),
    /// Zero or one (path?)
    ZeroOrOne(Box<PropertyPath>),
    /// Negated property set (!property)
    NegatedPropertySet(Vec<PropertyPath>),
}

/// Property path triple pattern
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyPathPattern {
    pub subject: Term,
    pub path: PropertyPath,
    pub object: Term,
}

/// SPARQL algebra expressions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algebra {
    /// Basic Graph Pattern
    Bgp(Vec<TriplePattern>),

    /// Property Path Pattern
    PropertyPath {
        subject: Term,
        path: PropertyPath,
        object: Term,
    },

    /// Join two patterns
    Join {
        left: Box<Algebra>,
        right: Box<Algebra>,
    },

    /// Left join (OPTIONAL)
    LeftJoin {
        left: Box<Algebra>,
        right: Box<Algebra>,
        filter: Option<Expression>,
    },

    /// Union of patterns
    Union {
        left: Box<Algebra>,
        right: Box<Algebra>,
    },

    /// Filter pattern
    Filter {
        pattern: Box<Algebra>,
        condition: Expression,
    },

    /// Extend pattern (BIND)
    Extend {
        pattern: Box<Algebra>,
        variable: Variable,
        expr: Expression,
    },

    /// Minus pattern
    Minus {
        left: Box<Algebra>,
        right: Box<Algebra>,
    },

    /// Service pattern (federation)
    Service {
        endpoint: Term,
        pattern: Box<Algebra>,
        silent: bool,
    },

    /// Graph pattern
    Graph { graph: Term, pattern: Box<Algebra> },

    /// Projection
    Project {
        pattern: Box<Algebra>,
        variables: Vec<Variable>,
    },

    /// Distinct
    Distinct { pattern: Box<Algebra> },

    /// Reduced
    Reduced { pattern: Box<Algebra> },

    /// Slice (LIMIT/OFFSET)
    Slice {
        pattern: Box<Algebra>,
        offset: Option<usize>,
        limit: Option<usize>,
    },

    /// Order by
    OrderBy {
        pattern: Box<Algebra>,
        conditions: Vec<OrderCondition>,
    },

    /// Group by
    Group {
        pattern: Box<Algebra>,
        variables: Vec<GroupCondition>,
        aggregates: Vec<(Variable, Aggregate)>,
    },

    /// Having
    Having {
        pattern: Box<Algebra>,
        condition: Expression,
    },

    /// Values clause
    Values {
        variables: Vec<Variable>,
        bindings: Vec<Binding>,
    },

    /// Table (empty result)
    Table,

    /// Zero matches
    Zero,

    /// Empty result set
    Empty,
}

/// Join algorithm hints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum JoinAlgorithm {
    #[default]
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    IndexNestedLoopJoin,
    BindJoin,
}

/// Filter placement hints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum FilterPlacement {
    Early,   // Push down as much as possible
    Late,    // Keep at current level
    #[default]
    Optimal, // Let optimizer decide
}

/// Service capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceCapabilities {
    pub supports_projection: bool,
    pub supports_filtering: bool,
    pub supports_ordering: bool,
    pub supports_aggregation: bool,
    pub max_query_size: Option<usize>,
}

/// Projection types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ProjectionType {
    #[default]
    Standard,
    Streaming,
    Cached,
}

/// Sort algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum SortAlgorithm {
    #[default]
    QuickSort,
    MergeSort,
    HeapSort,
    ExternalSort,
}

/// Grouping algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum GroupingAlgorithm {
    #[default]
    HashGrouping,
    SortGrouping,
    StreamingGrouping,
}

/// Materialization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum MaterializationStrategy {
    InMemory,
    Disk,
    #[default]
    Adaptive,
}

/// Parallelism types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ParallelismType {
    #[default]
    DataParallel,
    PipelineParallel,
    Hybrid,
}

/// Re-export IndexType from optimizer module
pub use crate::optimizer::index_types::IndexType;

/// Statistics for cost-based optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Statistics {
    /// Estimated number of triples/results
    pub cardinality: u64,
    /// Selectivity factor (0.0 to 1.0)
    pub selectivity: f64,
    /// Index availability
    pub available_indexes: Vec<IndexType>,
    /// Approximate cost (arbitrary units)
    pub cost: f64,
    /// Memory requirement estimate (bytes)
    pub memory_estimate: u64,
    /// IO operations estimate
    pub io_estimate: u64,
}

/// Optimization hints for algebra nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OptimizationHints {
    /// Preferred join algorithm
    pub join_algorithm: Option<JoinAlgorithm>,
    /// Filter placement strategy
    pub filter_placement: FilterPlacement,
    /// Materialization strategy
    pub materialization: MaterializationStrategy,
    /// Parallelism recommendations
    pub parallelism: Option<ParallelismType>,
    /// Index hints
    pub preferred_indexes: Vec<IndexType>,
    /// Cost estimates
    pub statistics: Option<Statistics>,
}

/// Enhanced algebra with optimization annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedAlgebra {
    /// The core algebra expression
    pub algebra: Algebra,
    /// Optimization hints
    pub hints: OptimizationHints,
    /// Execution context
    pub context: Option<String>,
}

impl PropertyPath {
    /// Create a direct property path
    pub fn iri(iri: Iri) -> Self {
        PropertyPath::Iri(iri)
    }

    /// Create an inverse property path
    pub fn inverse(path: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(path))
    }

    /// Create a sequence property path
    pub fn sequence(left: PropertyPath, right: PropertyPath) -> Self {
        PropertyPath::Sequence(Box::new(left), Box::new(right))
    }

    /// Create an alternative property path
    pub fn alternative(left: PropertyPath, right: PropertyPath) -> Self {
        PropertyPath::Alternative(Box::new(left), Box::new(right))
    }

    /// Create a zero-or-more property path
    pub fn zero_or_more(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(path))
    }

    /// Create a one-or-more property path
    pub fn one_or_more(path: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(path))
    }

    /// Create a zero-or-one property path
    pub fn zero_or_one(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(path))
    }

    /// Check if path is simple (direct property)
    pub fn is_simple(&self) -> bool {
        matches!(self, PropertyPath::Iri(_) | PropertyPath::Variable(_))
    }

    /// Get all variables mentioned in this property path
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        match self {
            PropertyPath::Variable(var) => vars.push(var.clone()),
            PropertyPath::Inverse(path) => path.collect_variables(vars),
            PropertyPath::Sequence(left, right) | PropertyPath::Alternative(left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            PropertyPath::ZeroOrMore(path)
            | PropertyPath::OneOrMore(path)
            | PropertyPath::ZeroOrOne(path) => path.collect_variables(vars),
            PropertyPath::NegatedPropertySet(paths) => {
                for path in paths {
                    path.collect_variables(vars);
                }
            }
            PropertyPath::Iri(_) => {}
        }
    }

    /// Estimate complexity of property path evaluation
    pub fn complexity(&self) -> usize {
        match self {
            PropertyPath::Iri(_) | PropertyPath::Variable(_) => 1,
            PropertyPath::Inverse(path) => path.complexity() + 10,
            PropertyPath::Sequence(left, right) => left.complexity() + right.complexity() + 20,
            PropertyPath::Alternative(left, right) => {
                std::cmp::max(left.complexity(), right.complexity()) + 15
            }
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => 1000, // High complexity
            PropertyPath::ZeroOrOne(path) => path.complexity() + 5,
            PropertyPath::NegatedPropertySet(paths) => {
                paths.iter().map(|p| p.complexity()).sum::<usize>() + 50
            }
        }
    }
}

impl PropertyPathPattern {
    /// Create a new property path pattern
    pub fn new(subject: Term, path: PropertyPath, object: Term) -> Self {
        PropertyPathPattern {
            subject,
            path,
            object,
        }
    }

    /// Get all variables mentioned in this pattern
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        self.subject.collect_variables(vars);
        self.path.collect_variables(vars);
        self.object.collect_variables(vars);
    }
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyPath::Iri(iri) => write!(f, "{iri}"),
            PropertyPath::Variable(var) => write!(f, "?{var}"),
            PropertyPath::Inverse(path) => write!(f, "^{path}"),
            PropertyPath::Sequence(left, right) => write!(f, "{left}/{right}"),
            PropertyPath::Alternative(left, right) => write!(f, "{left}|{right}"),
            PropertyPath::ZeroOrMore(path) => write!(f, "{path}*"),
            PropertyPath::OneOrMore(path) => write!(f, "{path} +"),
            PropertyPath::ZeroOrOne(path) => write!(f, "{path}?"),
            PropertyPath::NegatedPropertySet(paths) => {
                write!(f, "!(")?;
                for (i, path) in paths.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?
                    }
                    write!(f, "{path}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for PropertyPathPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.path, self.object)
    }
}

impl Algebra {
    /// Create a new BGP from triple patterns
    pub fn bgp(patterns: Vec<TriplePattern>) -> Self {
        Algebra::Bgp(patterns)
    }

    /// Create a property path algebra node
    pub fn property_path(subject: Term, path: PropertyPath, object: Term) -> Self {
        Algebra::PropertyPath {
            subject,
            path,
            object,
        }
    }

    /// Create a join of two patterns
    pub fn join(left: Algebra, right: Algebra) -> Self {
        Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a left join (optional)
    pub fn left_join(left: Algebra, right: Algebra, filter: Option<Expression>) -> Self {
        Algebra::LeftJoin {
            left: Box::new(left),
            right: Box::new(right),
            filter,
        }
    }

    /// Create a union of two patterns
    pub fn union(left: Algebra, right: Algebra) -> Self {
        Algebra::Union {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a filter pattern
    pub fn filter(pattern: Algebra, condition: Expression) -> Self {
        Algebra::Filter {
            pattern: Box::new(pattern),
            condition,
        }
    }

    /// Create an extend pattern (BIND)
    pub fn extend(pattern: Algebra, variable: Variable, expr: Expression) -> Self {
        Algebra::Extend {
            pattern: Box::new(pattern),
            variable,
            expr,
        }
    }

    /// Create a projection
    pub fn project(pattern: Algebra, variables: Vec<Variable>) -> Self {
        Algebra::Project {
            pattern: Box::new(pattern),
            variables,
        }
    }

    /// Create a slice (LIMIT/OFFSET)
    pub fn slice(pattern: Algebra, offset: Option<usize>, limit: Option<usize>) -> Self {
        Algebra::Slice {
            pattern: Box::new(pattern),
            offset,
            limit,
        }
    }

    /// Get all variables mentioned in this algebra expression
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        match self {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    pattern.collect_variables(vars);
                }
            }
            Algebra::PropertyPath {
                subject,
                path,
                object,
            } => {
                subject.collect_variables(vars);
                path.collect_variables(vars);
                object.collect_variables(vars);
            }
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
                if let Some(filter) = filter {
                    filter.collect_variables(vars);
                }
            }
            Algebra::Filter { pattern, condition } => {
                pattern.collect_variables(vars);
                condition.collect_variables(vars);
            }
            Algebra::Extend {
                pattern,
                variable,
                expr,
            } => {
                pattern.collect_variables(vars);
                vars.push(variable.clone());
                expr.collect_variables(vars);
            }
            Algebra::Service { pattern, .. } => {
                pattern.collect_variables(vars);
            }
            Algebra::Graph { pattern, .. } => {
                pattern.collect_variables(vars);
            }
            Algebra::Project { pattern, variables } => {
                pattern.collect_variables(vars);
                vars.extend(variables.clone());
            }
            Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::Slice { pattern, .. } => {
                pattern.collect_variables(vars);
            }
            Algebra::OrderBy {
                pattern,
                conditions,
            } => {
                pattern.collect_variables(vars);
                for condition in conditions {
                    condition.expr.collect_variables(vars);
                }
            }
            Algebra::Group {
                pattern,
                variables: group_vars,
                aggregates,
            } => {
                pattern.collect_variables(vars);
                for group_var in group_vars {
                    group_var.expr.collect_variables(vars);
                    if let Some(alias) = &group_var.alias {
                        vars.push(alias.clone());
                    }
                }
                for (var, aggregate) in aggregates {
                    vars.push(var.clone());
                    aggregate.collect_variables(vars);
                }
            }
            Algebra::Having { pattern, condition } => {
                pattern.collect_variables(vars);
                condition.collect_variables(vars);
            }
            Algebra::Values { variables, .. } => {
                vars.extend(variables.clone());
            }
            Algebra::Table | Algebra::Zero | Algebra::Empty => {}
        }
    }
}

impl TriplePattern {
    /// Create a new triple pattern
    pub fn new(subject: Term, predicate: Term, object: Term) -> Self {
        TriplePattern {
            subject,
            predicate,
            object,
        }
    }

    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        self.subject.collect_variables(vars);
        self.predicate.collect_variables(vars);
        self.object.collect_variables(vars);
    }

    /// Returns all variables in this triple pattern
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars
    }
}

impl Term {
    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        if let Term::Variable(var) = self {
            vars.push(var.clone());
        }
    }
}

impl Expression {
    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        match self {
            Expression::Variable(var) => vars.push(var.clone()),
            Expression::Function { args, .. } => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            Expression::Binary { left, right, .. } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Expression::Unary { operand, .. } => {
                operand.collect_variables(vars);
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_variables(vars);
                then_expr.collect_variables(vars);
                else_expr.collect_variables(vars);
            }
            Expression::Bound(var) => vars.push(var.clone()),
            Expression::Exists(algebra) | Expression::NotExists(algebra) => {
                algebra.collect_variables(vars);
            }
            Expression::Literal(_) | Expression::Iri(_) => {}
        }
    }
}

impl Aggregate {
    fn collect_variables(&self, vars: &mut Vec<Variable>) {
        match self {
            Aggregate::Count {
                expr: Some(expr), ..
            }
            | Aggregate::Sum { expr, .. }
            | Aggregate::Min { expr, .. }
            | Aggregate::Max { expr, .. }
            | Aggregate::Avg { expr, .. }
            | Aggregate::Sample { expr, .. }
            | Aggregate::GroupConcat { expr, .. } => {
                expr.collect_variables(vars);
            }
            Aggregate::Count { expr: None, .. } => {}
        }
    }
}

/// Convenience macros for building algebra expressions
#[macro_export]
macro_rules! triple {
    ($s:expr_2021, $p:expr_2021, $o:expr_2021) => {
        TriplePattern::new($s, $p, $o)
    };
}

#[macro_export]
macro_rules! var {
    ($name:expr_2021) => {
        Term::Variable($name.to_string())
    };
}

#[macro_export]
macro_rules! iri {
    ($iri:expr_2021) => {
        Term::Iri(NamedNode::new($iri).unwrap())
    };
}

#[macro_export]
macro_rules! literal {
    ($value:expr_2021) => {
        Term::Literal(Literal::new($value.to_string(), None, None))
    };
    ($value:expr_2021, lang: $lang:expr_2021) => {
        Term::Literal(Literal::new(
            $value.to_string(),
            Some($lang.to_string()),
            None,
        ))
    };
    ($value:expr_2021, datatype: $dt:expr_2021) => {
        Term::Literal(Literal::new(
            $value.to_string(),
            None,
            Some(NamedNode::new($dt).unwrap()),
        ))
    };
}



impl Default for ServiceCapabilities {
    fn default() -> Self {
        Self {
            supports_projection: true,
            supports_filtering: true,
            supports_ordering: false,
            supports_aggregation: false,
            max_query_size: None,
        }
    }
}






impl Literal {
    /// Create a new literal with value only
    pub fn new(value: String, language: Option<String>, datatype: Option<Iri>) -> Self {
        Literal {
            value,
            language,
            datatype,
        }
    }

    /// Create a simple string literal
    pub fn string(value: impl Into<String>) -> Self {
        Literal {
            value: value.into(),
            language: None,
            datatype: None,
        }
    }

    /// Create a language-tagged literal
    pub fn lang_string(value: impl Into<String>, language: impl Into<String>) -> Self {
        Literal {
            value: value.into(),
            language: Some(language.into()),
            datatype: None,
        }
    }

    /// Create a typed literal
    pub fn typed(value: impl Into<String>, datatype: Iri) -> Self {
        Literal {
            value: value.into(),
            language: None,
            datatype: Some(datatype),
        }
    }

    /// Create an integer literal
    pub fn integer(value: i64) -> Self {
        Literal::typed(
            value.to_string(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        )
    }

    /// Create a decimal literal
    pub fn decimal(value: f64) -> Self {
        Literal::typed(
            value.to_string(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
        )
    }

    /// Create a boolean literal
    pub fn boolean(value: bool) -> Self {
        Literal::typed(
            value.to_string(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
        )
    }

    /// Create a date literal
    pub fn date(value: impl Into<String>) -> Self {
        Literal::typed(
            value.into(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#date").unwrap(),
        )
    }

    /// Create a datetime literal
    pub fn datetime(value: impl Into<String>) -> Self {
        Literal::typed(
            value.into(),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#dateTime").unwrap(),
        )
    }

    /// Get the effective datatype (with default string type if none specified)
    pub fn effective_datatype(&self) -> Iri {
        if let Some(ref dt) = self.datatype {
            dt.clone()
        } else if self.language.is_some() {
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString").unwrap()
        } else {
            NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap()
        }
    }

    /// Check if this is a numeric literal
    pub fn is_numeric(&self) -> bool {
        if let Some(ref dt) = self.datatype {
            matches!(
                dt.as_str(),
                "http://www.w3.org/2001/XMLSchema#integer"
                    | "http://www.w3.org/2001/XMLSchema#decimal"
                    | "http://www.w3.org/2001/XMLSchema#float"
                    | "http://www.w3.org/2001/XMLSchema#double"
                    | "http://www.w3.org/2001/XMLSchema#long"
                    | "http://www.w3.org/2001/XMLSchema#int"
                    | "http://www.w3.org/2001/XMLSchema#short"
                    | "http://www.w3.org/2001/XMLSchema#byte"
                    | "http://www.w3.org/2001/XMLSchema#unsignedLong"
                    | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                    | "http://www.w3.org/2001/XMLSchema#unsignedShort"
                    | "http://www.w3.org/2001/XMLSchema#unsignedByte"
                    | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                    | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                    | "http://www.w3.org/2001/XMLSchema#negativeInteger"
                    | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
            )
        } else {
            false
        }
    }

    /// Check if this is a string literal
    pub fn is_string(&self) -> bool {
        self.datatype.is_none() && self.language.is_none()
    }

    /// Check if this is a language-tagged literal
    pub fn is_lang_string(&self) -> bool {
        self.language.is_some()
    }

    /// Check if this is a boolean literal
    pub fn is_boolean(&self) -> bool {
        if let Some(ref dt) = self.datatype {
            dt.as_str() == "http://www.w3.org/2001/XMLSchema#boolean"
        } else {
            false
        }
    }

    /// Check if this is a date/time literal
    pub fn is_datetime(&self) -> bool {
        if let Some(ref dt) = self.datatype {
            matches!(
                dt.as_str(),
                "http://www.w3.org/2001/XMLSchema#date"
                    | "http://www.w3.org/2001/XMLSchema#dateTime"
                    | "http://www.w3.org/2001/XMLSchema#time"
                    | "http://www.w3.org/2001/XMLSchema#gYear"
                    | "http://www.w3.org/2001/XMLSchema#gYearMonth"
                    | "http://www.w3.org/2001/XMLSchema#gMonth"
                    | "http://www.w3.org/2001/XMLSchema#gMonthDay"
                    | "http://www.w3.org/2001/XMLSchema#gDay"
                    | "http://www.w3.org/2001/XMLSchema#duration"
                    | "http://www.w3.org/2001/XMLSchema#dayTimeDuration"
                    | "http://www.w3.org/2001/XMLSchema#yearMonthDuration"
            )
        } else {
            false
        }
    }
}


impl Default for Statistics {
    fn default() -> Self {
        Self {
            cardinality: 0,
            selectivity: 1.0,
            available_indexes: Vec::new(),
            cost: 0.0,
            memory_estimate: 0,
            io_estimate: 0,
        }
    }
}

impl Statistics {
    /// Create statistics with estimated cardinality
    pub fn with_cardinality(cardinality: u64) -> Self {
        Self {
            cardinality,
            selectivity: 1.0,
            available_indexes: Vec::new(),
            cost: cardinality as f64,
            memory_estimate: cardinality * 64, // Rough estimate
            io_estimate: cardinality / 1000,   // Pages
        }
    }

    /// Update statistics with selectivity factor
    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity.clamp(0.0, 1.0);
        self.cardinality = (self.cardinality as f64 * self.selectivity) as u64;
        self.cost *= self.selectivity;
        self
    }

    /// Add available index
    pub fn with_index(mut self, index: IndexType) -> Self {
        self.available_indexes.push(index);
        // Reduce cost if good indexes are available
        self.cost *= 0.8;
        self
    }

    /// Combine statistics (for joins)
    pub fn combine(&self, other: &Statistics) -> Self {
        Self {
            cardinality: self.cardinality * other.cardinality,
            selectivity: self.selectivity * other.selectivity,
            available_indexes: self
                .available_indexes
                .iter()
                .chain(other.available_indexes.iter())
                .cloned()
                .collect(),
            cost: self.cost + other.cost,
            memory_estimate: self.memory_estimate + other.memory_estimate,
            io_estimate: self.io_estimate + other.io_estimate,
        }
    }
}

impl OptimizationHints {
    /// Create hints for BGP patterns
    pub fn for_bgp(patterns: &[TriplePattern]) -> Self {
        let mut hints = OptimizationHints::default();

        // Estimate based on pattern complexity
        let cardinality = match patterns.len() {
            0 => 0,
            1 => 1000,                         // Single pattern estimate
            n => 1000 / (n as u64 * n as u64), // Selectivity decreases with more patterns
        };

        hints.statistics = Some(Statistics::with_cardinality(cardinality));

        // Suggest indexes based on pattern structure
        for pattern in patterns {
            if let Term::Variable(_) = pattern.subject {
                hints.preferred_indexes.push(IndexType::PredicateIndex);
            }
            if let Term::Variable(_) = pattern.predicate {
                hints.preferred_indexes.push(IndexType::SubjectIndex);
            }
            if let Term::Variable(_) = pattern.object {
                hints
                    .preferred_indexes
                    .push(IndexType::SubjectPredicateIndex);
            }
        }

        hints
    }

    /// Create hints for join operations
    pub fn for_join(left_hints: &OptimizationHints, right_hints: &OptimizationHints) -> Self {
        let mut hints = OptimizationHints::default();

        // Combine statistics
        if let (Some(left_stats), Some(right_stats)) =
            (&left_hints.statistics, &right_hints.statistics)
        {
            hints.statistics = Some(left_stats.combine(right_stats));

            // Choose join algorithm based on cardinalities
            hints.join_algorithm = Some(match (left_stats.cardinality, right_stats.cardinality) {
                (l, r) if l < 1000 && r < 1000 => JoinAlgorithm::NestedLoopJoin,
                (l, r) if l > 100000 || r > 100000 => JoinAlgorithm::SortMergeJoin,
                _ => JoinAlgorithm::HashJoin,
            });
        }

        // Inherit index preferences
        hints.preferred_indexes = left_hints
            .preferred_indexes
            .iter()
            .chain(right_hints.preferred_indexes.iter())
            .cloned()
            .collect();

        hints
    }

    /// Create hints for filter operations
    pub fn for_filter(pattern_hints: &OptimizationHints, condition: &Expression) -> Self {
        let mut hints = pattern_hints.clone();

        // Apply filter selectivity
        if let Some(ref mut stats) = hints.statistics {
            let filter_selectivity = estimate_filter_selectivity(condition);
            *stats = stats.clone().with_selectivity(filter_selectivity);
        }

        // Suggest early filter placement for selective filters
        hints.filter_placement = if estimate_filter_selectivity(condition) < 0.1 {
            FilterPlacement::Early
        } else {
            FilterPlacement::Optimal
        };

        hints
    }
}

impl AnnotatedAlgebra {
    /// Create annotated algebra with default hints
    pub fn new(algebra: Algebra) -> Self {
        let hints = match &algebra {
            Algebra::Bgp(patterns) => OptimizationHints::for_bgp(patterns),
            Algebra::Join { left: _, right: _ } => {
                // For now, use default hints - in practice, we'd analyze the children
                OptimizationHints::default()
            }
            Algebra::Filter { .. } => OptimizationHints::default(),
            _ => OptimizationHints::default(),
        };

        Self {
            algebra,
            hints,
            context: None,
        }
    }

    /// Create annotated algebra with custom hints
    pub fn with_hints(algebra: Algebra, hints: OptimizationHints) -> Self {
        Self {
            algebra,
            hints,
            context: None,
        }
    }

    /// Add execution context
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }

    /// Get estimated cost
    pub fn estimated_cost(&self) -> f64 {
        self.hints
            .statistics
            .as_ref()
            .map(|s| s.cost)
            .unwrap_or(0.0)
    }

    /// Get estimated cardinality
    pub fn estimated_cardinality(&self) -> u64 {
        self.hints
            .statistics
            .as_ref()
            .map(|s| s.cardinality)
            .unwrap_or(0)
    }
}

/// Estimate selectivity of a filter condition (rough heuristic)
fn estimate_filter_selectivity(condition: &Expression) -> f64 {
    match condition {
        Expression::Binary { op, .. } => match op {
            BinaryOperator::Equal => 0.01,    // Very selective
            BinaryOperator::NotEqual => 0.99, // Not selective
            BinaryOperator::Less
            | BinaryOperator::LessEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterEqual => 0.33, // Range
            BinaryOperator::And => 0.25,      // Compound - more selective
            BinaryOperator::Or => 0.75,       // Compound - less selective
            _ => 0.5,                         // Default
        },
        Expression::Function { name, .. } => match name.as_str() {
            "regex" | "contains" => 0.2,              // Text search
            "bound" => 0.8,                           // Usually true
            "isIRI" | "isLiteral" | "isBlank" => 0.3, // Type checks
            _ => 0.5,                                 // Default
        },
        Expression::Unary { op: UnaryOperator::Not, .. } => 0.5, // Invert selectivity (simplified)
        Expression::Unary { .. } => 0.5,
        _ => 0.5, // Default selectivity
    }
}

/// Evaluation context for query execution
#[derive(Debug, Clone, Default)]
pub struct EvaluationContext {
    /// Variable bindings
    pub bindings: HashMap<Variable, Term>,
    /// Dataset being queried
    pub dataset: Option<String>,
    /// Query execution options
    pub options: HashMap<String, String>,
}
