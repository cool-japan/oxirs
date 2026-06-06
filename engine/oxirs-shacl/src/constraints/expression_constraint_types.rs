//! Core types for SHACL expression constraints.
//!
//! Defines the value, path, and AST types used by the SHACL Advanced Features
//! expression mechanism: [`ShaclPath`] (a lightweight property path),
//! [`ShaclValue`] (a typed runtime value), and [`ShaclExpression`] (the
//! expression AST itself).
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ExpressionConstraintComponent>

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ShaclPath — lightweight inline definition (mirrors the main paths module
// for expression evaluation purposes without creating a circular dependency)
// ---------------------------------------------------------------------------

/// Simplified property path for use within SHACL expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclPath {
    /// A direct predicate IRI
    Predicate(String),
    /// Inverse path (`^p`)
    Inverse(Box<ShaclPath>),
    /// Sequence path (`p1 / p2 / ...`)
    Sequence(Vec<ShaclPath>),
    /// Alternative path (`p1 | p2 | ...`)
    Alternative(Vec<ShaclPath>),
    /// Zero-or-more path (`p*`)
    ZeroOrMore(Box<ShaclPath>),
    /// One-or-more path (`p+`)
    OneOrMore(Box<ShaclPath>),
}

impl fmt::Display for ShaclPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaclPath::Predicate(iri) => write!(f, "<{iri}>"),
            ShaclPath::Inverse(p) => write!(f, "^{p}"),
            ShaclPath::Sequence(steps) => {
                let parts: Vec<_> = steps.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" / "))
            }
            ShaclPath::Alternative(alts) => {
                let parts: Vec<_> = alts.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", parts.join(" | "))
            }
            ShaclPath::ZeroOrMore(p) => write!(f, "{p}*"),
            ShaclPath::OneOrMore(p) => write!(f, "{p}+"),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaclValue — runtime value type
// ---------------------------------------------------------------------------

/// A typed value produced by evaluating a SHACL expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclValue {
    /// An IRI resource
    Iri(String),
    /// An RDF literal with optional datatype and language tag
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A plain integer (xsd:integer)
    Integer(i64),
    /// A floating-point number (xsd:double)
    Float(f64),
    /// A boolean (xsd:boolean)
    Boolean(bool),
    /// The null / absent value
    Null,
}

impl fmt::Display for ShaclValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaclValue::Iri(iri) => write!(f, "<{iri}>"),
            ShaclValue::Literal {
                value,
                datatype,
                lang,
            } => match (datatype, lang) {
                (Some(dt), _) => write!(f, "\"{value}\"^^<{dt}>"),
                (_, Some(l)) => write!(f, "\"{value}\"@{l}"),
                _ => write!(f, "\"{value}\""),
            },
            ShaclValue::Integer(n) => write!(f, "{n}"),
            ShaclValue::Float(x) => write!(f, "{x}"),
            ShaclValue::Boolean(b) => write!(f, "{b}"),
            ShaclValue::Null => write!(f, "null"),
        }
    }
}

impl ShaclValue {
    /// Returns `true` for values that are logically "truthy".
    pub fn is_truthy(&self) -> bool {
        match self {
            ShaclValue::Boolean(b) => *b,
            ShaclValue::Null => false,
            ShaclValue::Integer(n) => *n != 0,
            ShaclValue::Float(x) => *x != 0.0 && !x.is_nan(),
            ShaclValue::Literal { value, .. } => !value.is_empty(),
            ShaclValue::Iri(_) => true,
        }
    }

    /// Attempt to interpret the value as an `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ShaclValue::Float(x) => Some(*x),
            ShaclValue::Integer(n) => Some(*n as f64),
            ShaclValue::Literal { value, .. } => value.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Attempt to interpret the value as an `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ShaclValue::Integer(n) => Some(*n),
            ShaclValue::Float(x) => Some(*x as i64),
            ShaclValue::Literal { value, .. } => value.parse::<i64>().ok(),
            _ => None,
        }
    }

    /// Represent the value as a plain string.
    pub fn as_string(&self) -> String {
        match self {
            ShaclValue::Iri(iri) => iri.clone(),
            ShaclValue::Literal { value, .. } => value.clone(),
            ShaclValue::Integer(n) => n.to_string(),
            ShaclValue::Float(x) => x.to_string(),
            ShaclValue::Boolean(b) => b.to_string(),
            ShaclValue::Null => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaclExpression — AST
// ---------------------------------------------------------------------------

/// An AST node representing a SHACL expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaclExpression {
    // ---- Primitives --------------------------------------------------------
    /// A constant value
    Literal(ShaclValue),
    /// A variable reference (`$this`, `$value`, or user-defined names)
    Variable(String),
    /// A graph path expression — evaluated by the context's path resolver
    Path(ShaclPath),

    // ---- Arithmetic --------------------------------------------------------
    Add(Box<ShaclExpression>, Box<ShaclExpression>),
    Sub(Box<ShaclExpression>, Box<ShaclExpression>),
    Mul(Box<ShaclExpression>, Box<ShaclExpression>),
    Div(Box<ShaclExpression>, Box<ShaclExpression>),

    // ---- Comparison --------------------------------------------------------
    Eq(Box<ShaclExpression>, Box<ShaclExpression>),
    Ne(Box<ShaclExpression>, Box<ShaclExpression>),
    Lt(Box<ShaclExpression>, Box<ShaclExpression>),
    Gt(Box<ShaclExpression>, Box<ShaclExpression>),
    Lte(Box<ShaclExpression>, Box<ShaclExpression>),
    Gte(Box<ShaclExpression>, Box<ShaclExpression>),

    // ---- Logical -----------------------------------------------------------
    And(Box<ShaclExpression>, Box<ShaclExpression>),
    Or(Box<ShaclExpression>, Box<ShaclExpression>),
    Not(Box<ShaclExpression>),

    // ---- String functions --------------------------------------------------
    /// Concatenate a list of string values
    Concat(Vec<ShaclExpression>),
    /// Length of a string
    StrLen(Box<ShaclExpression>),
    /// Test a string against a regex pattern
    Regex(Box<ShaclExpression>, String),
    /// Convert to uppercase
    UpperCase(Box<ShaclExpression>),
    /// Convert to lowercase
    LowerCase(Box<ShaclExpression>),

    // ---- Numeric functions -------------------------------------------------
    Abs(Box<ShaclExpression>),
    Floor(Box<ShaclExpression>),
    Ceil(Box<ShaclExpression>),
    Round(Box<ShaclExpression>),

    // ---- Aggregate / graph -------------------------------------------------
    /// Count the values reachable via a path from `$this`
    Count(ShaclPath),

    // ---- Type functions ----------------------------------------------------
    IsIri(Box<ShaclExpression>),
    IsLiteral(Box<ShaclExpression>),
    Datatype(Box<ShaclExpression>),
    Lang(Box<ShaclExpression>),
}
