//! SPARQL Algebra Module
//!
//! This module provides the core algebraic representation of SPARQL queries,
//! including basic graph patterns, joins, unions, filters, and other operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Variable identifier
pub type Variable = String;

/// IRI (Internationalized Resource Identifier)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Iri(pub String);

impl fmt::Display for Iri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

/// Literal value with optional language tag or datatype
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Literal {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Option<Iri>,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.value)?;
        if let Some(lang) = &self.language {
            write!(f, "@{}", lang)?;
        } else if let Some(dt) = &self.datatype {
            write!(f, "^^{}", dt)?;
        }
        Ok(())
    }
}

/// RDF term (subject, predicate, or object)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Term {
    Variable(Variable),
    Iri(Iri),
    Literal(Literal),
    BlankNode(String),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Variable(v) => write!(f, "?{}", v),
            Term::Iri(iri) => write!(f, "{}", iri),
            Term::Literal(lit) => write!(f, "{}", lit),
            Term::BlankNode(id) => write!(f, "_:{}", id),
        }
    }
}

/// Triple pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        expr: Box<Expression>,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderCondition {
    pub expr: Expression,
    pub ascending: bool,
}

/// Group condition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupCondition {
    pub expr: Expression,
    pub alias: Option<Variable>,
}

/// Property path expressions for advanced SPARQL 1.1 graph navigation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyPathPattern {
    pub subject: Term,
    pub path: PropertyPath,
    pub object: Term,
}

/// SPARQL algebra expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
}

/// Join algorithm hints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JoinAlgorithm {
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    IndexNestedLoopJoin,
    BindJoin,
}

/// Filter placement hints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterPlacement {
    Early,   // Push down as much as possible
    Late,    // Keep at current level
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProjectionType {
    Standard,
    Streaming,
    Cached,
}

/// Sort algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortAlgorithm {
    QuickSort,
    MergeSort,
    HeapSort,
    ExternalSort,
}

/// Grouping algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GroupingAlgorithm {
    HashGrouping,
    SortGrouping,
    StreamingGrouping,
}

/// Materialization strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterializationStrategy {
    InMemory,
    Disk,
    Adaptive,
}

/// Parallelism types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParallelismType {
    DataParallel,
    PipelineParallel,
    Hybrid,
}

/// Index types for optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexType {
    SubjectIndex,
    PredicateIndex,
    ObjectIndex,
    SubjectPredicateIndex,
    PredicateObjectIndex,
    SubjectObjectIndex,
    FullIndex,
    CustomIndex(String),
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
            PropertyPath::Iri(iri) => write!(f, "{}", iri),
            PropertyPath::Variable(var) => write!(f, "?{}", var),
            PropertyPath::Inverse(path) => write!(f, "^{}", path),
            PropertyPath::Sequence(left, right) => write!(f, "{}/{}", left, right),
            PropertyPath::Alternative(left, right) => write!(f, "{}|{}", left, right),
            PropertyPath::ZeroOrMore(path) => write!(f, "{}*", path),
            PropertyPath::OneOrMore(path) => write!(f, "{} +", path),
            PropertyPath::ZeroOrOne(path) => write!(f, "{}?", path),
            PropertyPath::NegatedPropertySet(paths) => {
                write!(f, "!(")?;
                for (i, path) in paths.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?
                    }
                    write!(f, "{}", path)?;
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
            Algebra::Table | Algebra::Zero => {}
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
            Expression::Unary { expr, .. } => {
                expr.collect_variables(vars);
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
    ($s:expr, $p:expr, $o:expr) => {
        TriplePattern::new($s, $p, $o)
    };
}

#[macro_export]
macro_rules! var {
    ($name:expr) => {
        Term::Variable($name.to_string())
    };
}

#[macro_export]
macro_rules! iri {
    ($iri:expr) => {
        Term::Iri(Iri($iri.to_string()))
    };
}

#[macro_export]
macro_rules! literal {
    ($value:expr) => {
        Term::Literal(Literal::new($value.to_string(), None, None))
    };
    ($value:expr, lang: $lang:expr) => {
        Term::Literal(Literal::new(
            $value.to_string(),
            Some($lang.to_string()),
            None,
        ))
    };
    ($value:expr, datatype: $dt:expr) => {
        Term::Literal(Literal::new(
            $value.to_string(),
            None,
            Some(Iri($dt.to_string())),
        ))
    };
}

impl Default for FilterPlacement {
    fn default() -> Self {
        FilterPlacement::Optimal
    }
}

impl Default for ProjectionType {
    fn default() -> Self {
        ProjectionType::Standard
    }
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

impl Default for JoinAlgorithm {
    fn default() -> Self {
        JoinAlgorithm::HashJoin
    }
}

impl Default for SortAlgorithm {
    fn default() -> Self {
        SortAlgorithm::QuickSort
    }
}

impl Default for GroupingAlgorithm {
    fn default() -> Self {
        GroupingAlgorithm::HashGrouping
    }
}

impl Default for MaterializationStrategy {
    fn default() -> Self {
        MaterializationStrategy::Adaptive
    }
}

impl Default for ParallelismType {
    fn default() -> Self {
        ParallelismType::DataParallel
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
            Iri("http://www.w3.org/2001/XMLSchema#integer".to_string()),
        )
    }

    /// Create a decimal literal
    pub fn decimal(value: f64) -> Self {
        Literal::typed(
            value.to_string(),
            Iri("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
        )
    }

    /// Create a boolean literal
    pub fn boolean(value: bool) -> Self {
        Literal::typed(
            value.to_string(),
            Iri("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
        )
    }

    /// Create a date literal
    pub fn date(value: impl Into<String>) -> Self {
        Literal::typed(
            value.into(),
            Iri("http://www.w3.org/2001/XMLSchema#date".to_string()),
        )
    }

    /// Create a datetime literal
    pub fn datetime(value: impl Into<String>) -> Self {
        Literal::typed(
            value.into(),
            Iri("http://www.w3.org/2001/XMLSchema#dateTime".to_string()),
        )
    }

    /// Get the effective datatype (with default string type if none specified)
    pub fn effective_datatype(&self) -> Iri {
        if let Some(ref dt) = self.datatype {
            dt.clone()
        } else if self.language.is_some() {
            Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string())
        } else {
            Iri("http://www.w3.org/2001/XMLSchema#string".to_string())
        }
    }

    /// Check if this is a numeric literal
    pub fn is_numeric(&self) -> bool {
        if let Some(ref dt) = self.datatype {
            matches!(
                dt.0.as_str(),
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
            dt.0 == "http://www.w3.org/2001/XMLSchema#boolean"
        } else {
            false
        }
    }

    /// Check if this is a date/time literal
    pub fn is_datetime(&self) -> bool {
        if let Some(ref dt) = self.datatype {
            matches!(
                dt.0.as_str(),
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
