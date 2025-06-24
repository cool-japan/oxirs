//! SPARQL Algebra Module
//!
//! This module provides the core algebraic representation of SPARQL queries,
//! including basic graph patterns, joins, unions, filters, and other operations.

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Variable identifier
pub type Variable = String;

/// IRI (Internationalized Resource Identifier)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Iri(pub String);

impl fmt::Display for Iri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.0)
    }
}

/// Literal value with optional language tag or datatype
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Function {
        name: String,
        args: Vec<Expression>,
    },
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

/// SPARQL algebra expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Algebra {
    /// Basic Graph Pattern
    Bgp(Vec<TriplePattern>),
    
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
    Graph {
        graph: Term,
        pattern: Box<Algebra>,
    },
    
    /// Projection
    Project {
        pattern: Box<Algebra>,
        variables: Vec<Variable>,
    },
    
    /// Distinct
    Distinct {
        pattern: Box<Algebra>,
    },
    
    /// Reduced
    Reduced {
        pattern: Box<Algebra>,
    },
    
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

impl Algebra {
    /// Create a new BGP from triple patterns
    pub fn bgp(patterns: Vec<TriplePattern>) -> Self {
        Algebra::Bgp(patterns)
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
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Algebra::LeftJoin { left, right, filter } => {
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
            Algebra::Extend { pattern, variable, expr } => {
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
            Algebra::OrderBy { pattern, conditions } => {
                pattern.collect_variables(vars);
                for condition in conditions {
                    condition.expr.collect_variables(vars);
                }
            }
            Algebra::Group { pattern, variables: group_vars, aggregates } => {
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
        TriplePattern { subject, predicate, object }
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
            Expression::Conditional { condition, then_expr, else_expr } => {
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
            Aggregate::Count { expr: Some(expr), .. }
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
        Term::Literal(Literal {
            value: $value.to_string(),
            language: None,
            datatype: None,
        })
    };
    ($value:expr, lang: $lang:expr) => {
        Term::Literal(Literal {
            value: $value.to_string(),
            language: Some($lang.to_string()),
            datatype: None,
        })
    };
    ($value:expr, datatype: $dt:expr) => {
        Term::Literal(Literal {
            value: $value.to_string(),
            language: None,
            datatype: Some(Iri($dt.to_string())),
        })
    };
}
