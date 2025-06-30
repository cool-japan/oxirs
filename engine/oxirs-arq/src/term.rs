//! Comprehensive Term System for SPARQL Query Processing
//!
//! This module provides a complete implementation of RDF terms with full datatype support,
//! SPARQL-compliant comparison and ordering, variable binding, and expression evaluation.

use crate::algebra::{Iri, Literal, Term as AlgebraTerm, TriplePattern, Variable};
use crate::path::PropertyPath;
use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Utc};
use ordered_float::OrderedFloat;
use oxirs_core::model::NamedNode;
use rust_decimal::Decimal;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// XSD namespace for datatype URIs
pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// RDF namespace
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// Common XSD datatypes
pub mod xsd {
    use super::XSD_NS;

    pub const STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
    pub const BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
    pub const DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
    pub const INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
    pub const DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
    pub const FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
    pub const DATE: &str = "http://www.w3.org/2001/XMLSchema#date";
    pub const TIME: &str = "http://www.w3.org/2001/XMLSchema#time";
    pub const DATE_TIME: &str = "http://www.w3.org/2001/XMLSchema#dateTime";
    pub const DATE_TIME_STAMP: &str = "http://www.w3.org/2001/XMLSchema#dateTimeStamp";
    pub const DURATION: &str = "http://www.w3.org/2001/XMLSchema#duration";
    pub const BYTE: &str = "http://www.w3.org/2001/XMLSchema#byte";
    pub const SHORT: &str = "http://www.w3.org/2001/XMLSchema#short";
    pub const INT: &str = "http://www.w3.org/2001/XMLSchema#int";
    pub const LONG: &str = "http://www.w3.org/2001/XMLSchema#long";
    pub const UNSIGNED_BYTE: &str = "http://www.w3.org/2001/XMLSchema#unsignedByte";
    pub const UNSIGNED_SHORT: &str = "http://www.w3.org/2001/XMLSchema#unsignedShort";
    pub const UNSIGNED_INT: &str = "http://www.w3.org/2001/XMLSchema#unsignedInt";
    pub const UNSIGNED_LONG: &str = "http://www.w3.org/2001/XMLSchema#unsignedLong";
    pub const POSITIVE_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#positiveInteger";
    pub const NON_NEGATIVE_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#nonNegativeInteger";
    pub const NEGATIVE_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#negativeInteger";
    pub const NON_POSITIVE_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#nonPositiveInteger";
    pub const HEX_BINARY: &str = "http://www.w3.org/2001/XMLSchema#hexBinary";
    pub const BASE64_BINARY: &str = "http://www.w3.org/2001/XMLSchema#base64Binary";
}

/// Enhanced RDF Term with full datatype support
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// IRI reference
    Iri(String),
    /// Blank node
    BlankNode(String),
    /// Literal with proper datatype handling
    Literal(LiteralValue),
    /// Variable (for query patterns)
    Variable(String),
    /// Quoted triple (RDF-star support)
    QuotedTriple(Box<QuotedTripleValue>),
    /// Property path expression
    PropertyPath(PropertyPath),
}

/// Quoted triple representation for RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedTripleValue {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

/// Literal value with parsed datatype
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LiteralValue {
    /// Lexical form
    pub lexical_form: String,
    /// Datatype IRI
    pub datatype: String,
    /// Language tag (for language-tagged strings)
    pub language_tag: Option<String>,
    /// Parsed value (cached for efficiency)
    parsed_value: ParsedValue,
}

/// Parsed literal values for efficient operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ParsedValue {
    String(String),
    Boolean(bool),
    Integer(i64),
    Decimal(OrderedFloat<f64>),
    Float(OrderedFloat<f32>),
    Double(OrderedFloat<f64>),
    DateTime(i64), // Unix timestamp in nanoseconds
    Date(i32),     // Days since epoch
    Time(i64),     // Nanoseconds since midnight
    Duration(i64), // Duration in nanoseconds
    Binary(Vec<u8>),
    Other,
}

impl Term {
    /// Create an IRI term
    pub fn iri(iri: &str) -> Self {
        Term::Iri(iri.to_string())
    }

    /// Create a blank node term
    pub fn blank_node(id: &str) -> Self {
        Term::BlankNode(id.to_string())
    }

    /// Create a simple literal term
    pub fn literal(value: &str) -> Self {
        Term::Literal(LiteralValue::new_simple(value))
    }

    /// Create a typed literal term
    pub fn typed_literal(value: &str, datatype: &str) -> Result<Self> {
        Ok(Term::Literal(LiteralValue::new_typed(value, datatype)?))
    }

    /// Create a language-tagged literal
    pub fn lang_literal(value: &str, lang: &str) -> Self {
        Term::Literal(LiteralValue::new_lang(value, lang))
    }

    /// Create a variable term
    pub fn variable(name: &str) -> Self {
        Term::Variable(name.to_string())
    }

    /// Create a quoted triple term
    pub fn quoted_triple(subject: Term, predicate: Term, object: Term) -> Self {
        Term::QuotedTriple(Box::new(QuotedTripleValue {
            subject,
            predicate,
            object,
        }))
    }

    /// Create a property path term
    pub fn property_path(path: PropertyPath) -> Self {
        Term::PropertyPath(path)
    }

    /// Check if term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Check if term is ground (not a variable)
    pub fn is_ground(&self) -> bool {
        !self.is_variable()
    }

    /// Check if term is an IRI
    pub fn is_iri(&self) -> bool {
        matches!(self, Term::Iri(_))
    }

    /// Check if term is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }

    /// Check if term is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal(_))
    }

    /// Check if term is a quoted triple
    pub fn is_quoted_triple(&self) -> bool {
        matches!(self, Term::QuotedTriple(_))
    }

    /// Check if term is a property path
    pub fn is_property_path(&self) -> bool {
        matches!(self, Term::PropertyPath(_))
    }
}

impl LiteralValue {
    /// Create a simple literal (xsd:string)
    pub fn new_simple(value: &str) -> Self {
        Self {
            lexical_form: value.to_string(),
            datatype: xsd::STRING.to_string(),
            language_tag: None,
            parsed_value: ParsedValue::String(value.to_string()),
        }
    }

    /// Create a language-tagged literal
    pub fn new_lang(value: &str, lang: &str) -> Self {
        Self {
            lexical_form: value.to_string(),
            datatype: RDF_NS.to_string() + "langString",
            language_tag: Some(lang.to_string()),
            parsed_value: ParsedValue::String(value.to_string()),
        }
    }

    /// Create a typed literal
    pub fn new_typed(value: &str, datatype: &str) -> Result<Self> {
        let parsed_value = Self::parse_value(value, datatype)?;
        Ok(Self {
            lexical_form: value.to_string(),
            datatype: datatype.to_string(),
            language_tag: None,
            parsed_value,
        })
    }

    /// Parse value according to datatype
    fn parse_value(value: &str, datatype: &str) -> Result<ParsedValue> {
        Ok(match datatype {
            xsd::STRING => ParsedValue::String(value.to_string()),
            xsd::BOOLEAN => {
                let b = match value {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => bail!("Invalid boolean value: {}", value),
                };
                ParsedValue::Boolean(b)
            }
            xsd::INTEGER | xsd::LONG | xsd::INT | xsd::SHORT | xsd::BYTE => {
                let i = value
                    .parse::<i64>()
                    .map_err(|_| anyhow!("Invalid integer value: {}", value))?;
                ParsedValue::Integer(i)
            }
            xsd::DECIMAL => {
                let d = value
                    .parse::<f64>()
                    .map_err(|_| anyhow!("Invalid decimal value: {}", value))?;
                ParsedValue::Decimal(OrderedFloat(d))
            }
            xsd::FLOAT => {
                let f = value
                    .parse::<f32>()
                    .map_err(|_| anyhow!("Invalid float value: {}", value))?;
                ParsedValue::Float(OrderedFloat(f))
            }
            xsd::DOUBLE => {
                let d = value
                    .parse::<f64>()
                    .map_err(|_| anyhow!("Invalid double value: {}", value))?;
                ParsedValue::Double(OrderedFloat(d))
            }
            xsd::DATE_TIME | xsd::DATE_TIME_STAMP => {
                let dt = DateTime::parse_from_rfc3339(value)
                    .map_err(|_| anyhow!("Invalid dateTime value: {}", value))?;
                ParsedValue::DateTime(dt.timestamp_nanos())
            }
            xsd::DATE => {
                let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .map_err(|_| anyhow!("Invalid date value: {}", value))?;
                ParsedValue::Date(date.num_days_from_ce())
            }
            xsd::TIME => {
                let time = NaiveTime::parse_from_str(value, "%H:%M:%S%.f")
                    .map_err(|_| anyhow!("Invalid time value: {}", value))?;
                ParsedValue::Time(
                    time.num_seconds_from_midnight() as i64 * 1_000_000_000
                        + time.nanosecond() as i64,
                )
            }
            xsd::HEX_BINARY => {
                let bytes = hex::decode(value)
                    .map_err(|_| anyhow!("Invalid hexBinary value: {}", value))?;
                ParsedValue::Binary(bytes)
            }
            xsd::BASE64_BINARY => {
                let bytes = base64::decode(value)
                    .map_err(|_| anyhow!("Invalid base64Binary value: {}", value))?;
                ParsedValue::Binary(bytes)
            }
            _ => ParsedValue::Other,
        })
    }

    /// Get effective boolean value
    pub fn effective_boolean_value(&self) -> Result<bool> {
        match &self.parsed_value {
            ParsedValue::Boolean(b) => Ok(*b),
            ParsedValue::String(s) => Ok(!s.is_empty()),
            ParsedValue::Integer(i) => Ok(*i != 0),
            ParsedValue::Decimal(d) => Ok(d.0 != 0.0),
            ParsedValue::Float(f) => Ok(f.0 != 0.0),
            ParsedValue::Double(d) => Ok(d.0 != 0.0),
            _ => Ok(true),
        }
    }

    /// Convert to numeric value
    pub fn to_numeric(&self) -> Result<NumericValue> {
        match &self.parsed_value {
            ParsedValue::Integer(i) => Ok(NumericValue::Integer(*i)),
            ParsedValue::Decimal(d) => Ok(NumericValue::Decimal(d.0)),
            ParsedValue::Float(f) => Ok(NumericValue::Float(f.0 as f64)),
            ParsedValue::Double(d) => Ok(NumericValue::Double(d.0)),
            ParsedValue::Boolean(b) => Ok(NumericValue::Integer(if *b { 1 } else { 0 })),
            ParsedValue::String(s) => {
                // Try parsing as number
                if let Ok(i) = s.parse::<i64>() {
                    Ok(NumericValue::Integer(i))
                } else if let Ok(d) = s.parse::<f64>() {
                    Ok(NumericValue::Double(d))
                } else {
                    bail!("Cannot convert string '{}' to numeric", s)
                }
            }
            _ => bail!("Cannot convert {} to numeric", self.datatype),
        }
    }

    /// Check if literal is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            &self.parsed_value,
            ParsedValue::Integer(_)
                | ParsedValue::Decimal(_)
                | ParsedValue::Float(_)
                | ParsedValue::Double(_)
        )
    }
}

/// Numeric values for arithmetic operations
#[derive(Debug, Clone, PartialEq)]
pub enum NumericValue {
    Integer(i64),
    Decimal(f64),
    Float(f64),
    Double(f64),
}

impl NumericValue {
    /// Promote to common numeric type
    pub fn promote_with(&self, other: &NumericValue) -> (NumericValue, NumericValue) {
        use NumericValue::*;
        match (self, other) {
            (Integer(a), Integer(b)) => (Integer(*a), Integer(*b)),
            (Integer(a), Decimal(b)) => (Decimal(*a as f64), Decimal(*b)),
            (Integer(a), Float(b)) => (Float(*a as f64), Float(*b)),
            (Integer(a), Double(b)) => (Double(*a as f64), Double(*b)),
            (Decimal(a), Integer(b)) => (Decimal(*a), Decimal(*b as f64)),
            (Decimal(a), Decimal(b)) => (Decimal(*a), Decimal(*b)),
            (Decimal(a), Float(b)) => (Float(*a), Float(*b)),
            (Decimal(a), Double(b)) => (Double(*a), Double(*b)),
            (Float(a), Integer(b)) => (Float(*a), Float(*b as f64)),
            (Float(a), Decimal(b)) => (Float(*a), Float(*b)),
            (Float(a), Float(b)) => (Float(*a), Float(*b)),
            (Float(a), Double(b)) => (Double(*a as f64), Double(*b)),
            (Double(a), Integer(b)) => (Double(*a), Double(*b as f64)),
            (Double(a), Decimal(b)) => (Double(*a), Double(*b)),
            (Double(a), Float(b)) => (Double(*a), Double(*b as f64)),
            (Double(a), Double(b)) => (Double(*a), Double(*b)),
        }
    }

    /// Convert back to term
    pub fn to_term(&self) -> Term {
        match self {
            NumericValue::Integer(i) => Term::typed_literal(&i.to_string(), xsd::INTEGER).unwrap(),
            NumericValue::Decimal(d) => Term::typed_literal(&d.to_string(), xsd::DECIMAL).unwrap(),
            NumericValue::Float(f) => Term::typed_literal(&f.to_string(), xsd::FLOAT).unwrap(),
            NumericValue::Double(d) => Term::typed_literal(&d.to_string(), xsd::DOUBLE).unwrap(),
        }
    }
}

/// SPARQL value ordering according to spec
impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Self) -> Ordering {
        use Term::*;

        // Order: Variable < BlankNode < Iri < Literal < QuotedTriple < PropertyPath
        match (self, other) {
            (Variable(_), Variable(_)) => Ordering::Equal,
            (Variable(_), _) => Ordering::Less,
            (_, Variable(_)) => Ordering::Greater,

            (BlankNode(a), BlankNode(b)) => a.cmp(b),
            (BlankNode(_), _) => Ordering::Less,
            (_, BlankNode(_)) => Ordering::Greater,

            (Iri(a), Iri(b)) => a.cmp(b),
            (Iri(_), Literal(_) | QuotedTriple(_) | PropertyPath(_)) => Ordering::Less,
            (Literal(_) | QuotedTriple(_) | PropertyPath(_), Iri(_)) => Ordering::Greater,

            (Literal(a), Literal(b)) => a.cmp(b),
            (Literal(_), QuotedTriple(_) | PropertyPath(_)) => Ordering::Less,
            (QuotedTriple(_) | PropertyPath(_), Literal(_)) => Ordering::Greater,

            (QuotedTriple(a), QuotedTriple(b)) => a.cmp(b),
            (QuotedTriple(_), PropertyPath(_)) => Ordering::Less,
            (PropertyPath(_), QuotedTriple(_)) => Ordering::Greater,

            (PropertyPath(a), PropertyPath(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for LiteralValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LiteralValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // Language tags take precedence
        match (&self.language_tag, &other.language_tag) {
            (Some(a), Some(b)) => match a.cmp(b) {
                Ordering::Equal => self.lexical_form.cmp(&other.lexical_form),
                ord => ord,
            },
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => {
                // Compare by datatype, then by value
                match self.datatype.cmp(&other.datatype) {
                    Ordering::Equal => self.compare_same_datatype(other),
                    ord => ord,
                }
            }
        }
    }
}

impl LiteralValue {
    fn compare_same_datatype(&self, other: &Self) -> Ordering {
        match (&self.parsed_value, &other.parsed_value) {
            (ParsedValue::Boolean(a), ParsedValue::Boolean(b)) => a.cmp(b),
            (ParsedValue::Integer(a), ParsedValue::Integer(b)) => a.cmp(b),
            (ParsedValue::Decimal(a), ParsedValue::Decimal(b)) => a.cmp(b),
            (ParsedValue::Float(a), ParsedValue::Float(b)) => a.cmp(b),
            (ParsedValue::Double(a), ParsedValue::Double(b)) => a.cmp(b),
            (ParsedValue::DateTime(a), ParsedValue::DateTime(b)) => a.cmp(b),
            (ParsedValue::Date(a), ParsedValue::Date(b)) => a.cmp(b),
            (ParsedValue::Time(a), ParsedValue::Time(b)) => a.cmp(b),
            (ParsedValue::Duration(a), ParsedValue::Duration(b)) => a.cmp(b),
            (ParsedValue::Binary(a), ParsedValue::Binary(b)) => a.cmp(b),
            _ => self.lexical_form.cmp(&other.lexical_form),
        }
    }
}

/// Variable binding context
#[derive(Debug, Clone, Default)]
pub struct BindingContext {
    /// Current variable bindings
    bindings: HashMap<Variable, Term>,
    /// Nested scopes for subqueries
    scopes: Vec<HashMap<Variable, Term>>,
}

impl BindingContext {
    /// Create new binding context
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable to a term
    pub fn bind(&mut self, var: &str, term: Term) {
        if let Ok(variable) = Variable::new(var) {
            self.bindings.insert(variable, term);
        }
    }

    /// Get binding for a variable
    pub fn get(&self, var: &str) -> Option<&Term> {
        // Check current scope first
        if let Ok(variable) = Variable::new(var) {
            if let Some(term) = self.bindings.get(&variable) {
                return Some(term);
            }
        }

        // Check parent scopes
        for scope in self.scopes.iter().rev() {
            if let Ok(variable) = Variable::new(var) {
                if let Some(term) = scope.get(&variable) {
                    return Some(term);
                }
            }
        }

        None
    }

    /// Check if variable is bound
    pub fn is_bound(&self, var: &str) -> bool {
        self.get(var).is_some()
    }

    /// Push new scope
    pub fn push_scope(&mut self) {
        let current = std::mem::take(&mut self.bindings);
        self.scopes.push(current);
    }

    /// Pop scope
    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            self.bindings = scope;
        }
    }

    /// Get all bound variables
    pub fn variables(&self) -> Vec<&str> {
        let mut vars: Vec<_> = self.bindings.keys().map(|s| s.as_str()).collect();

        for scope in &self.scopes {
            for var in scope.keys() {
                if !vars.contains(&var.as_str()) {
                    vars.push(var.as_str());
                }
            }
        }

        vars
    }

    /// Apply bindings to a term
    pub fn apply(&self, term: &Term) -> Term {
        match term {
            Term::Variable(var) => self.get(var).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }
}

/// Term pattern matching
pub fn matches_pattern(pattern: &Term, term: &Term, bindings: &mut BindingContext) -> bool {
    match (pattern, term) {
        (Term::Variable(var), _) => {
            // Check if variable is already bound
            if let Some(bound) = bindings.get(var) {
                bound == term
            } else {
                // Bind variable
                bindings.bind(var, term.clone());
                true
            }
        }
        (Term::Iri(p), Term::Iri(t)) => p == t,
        (Term::BlankNode(p), Term::BlankNode(t)) => p == t,
        (Term::Literal(p), Term::Literal(t)) => p == t,
        (Term::QuotedTriple(p), Term::QuotedTriple(t)) => {
            matches_pattern(&p.subject, &t.subject, bindings)
                && matches_pattern(&p.predicate, &t.predicate, bindings)
                && matches_pattern(&p.object, &t.object, bindings)
        }
        (Term::PropertyPath(p), Term::PropertyPath(t)) => p == t,
        _ => false,
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Iri(iri) => write!(f, "<{}>", iri),
            Term::BlankNode(id) => write!(f, "_:{}", id),
            Term::Literal(lit) => write!(f, "{}", lit),
            Term::Variable(var) => write!(f, "?{}", var),
            Term::QuotedTriple(triple) => {
                write!(f, "<<{} {} {}>>", triple.subject, triple.predicate, triple.object)
            }
            Term::PropertyPath(path) => write!(f, "{}", path),
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.lexical_form)?;
        if let Some(lang) = &self.language_tag {
            write!(f, "@{}", lang)?;
        } else if self.datatype != xsd::STRING {
            write!(f, "^^<{}>", self.datatype)?;
        }
        Ok(())
    }
}

impl Term {
    /// Convert from algebra::Term to term::Term
    pub fn from_algebra_term(algebra_term: &AlgebraTerm) -> Self {
        match algebra_term {
            AlgebraTerm::Iri(iri) => Term::iri(iri.as_str()),
            AlgebraTerm::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    Term::lang_literal(&lit.value, lang)
                } else if let Some(datatype) = &lit.datatype {
                    Term::typed_literal(&lit.value, datatype.as_str())
                        .unwrap_or_else(|_| Term::literal(&lit.value))
                } else {
                    Term::literal(&lit.value)
                }
            }
            AlgebraTerm::BlankNode(bn) => Term::blank_node(bn),
            AlgebraTerm::Variable(var) => Term::variable(var.as_str()),
            AlgebraTerm::QuotedTriple(quoted_triple) => {
                Term::QuotedTriple(Box::new(QuotedTripleValue {
                    subject: Term::from_algebra_term(&quoted_triple.subject),
                    predicate: Term::from_algebra_term(&quoted_triple.predicate),
                    object: Term::from_algebra_term(&quoted_triple.object),
                }))
            }
            AlgebraTerm::PropertyPath(path) => Term::PropertyPath(path.clone()),
        }
    }

    /// Convert from term::Term to algebra::Term
    pub fn to_algebra_term(&self) -> AlgebraTerm {
        match self {
            Term::Iri(iri) => AlgebraTerm::Iri(NamedNode::new_unchecked(iri)),
            Term::BlankNode(bn) => AlgebraTerm::BlankNode(bn.clone()),
            Term::Literal(lit_val) => AlgebraTerm::Literal(Literal {
                value: lit_val.lexical_form.clone(),
                language: lit_val.language_tag.clone(),
                datatype: if lit_val.datatype != xsd::STRING {
                    Some(NamedNode::new_unchecked(&lit_val.datatype))
                } else {
                    None
                },
            }),
            Term::Variable(var) => AlgebraTerm::Variable(Variable::new(var).unwrap()),
            Term::QuotedTriple(triple) => {
                AlgebraTerm::QuotedTriple(Box::new(TriplePattern {
                    subject: triple.subject.to_algebra_term(),
                    predicate: triple.predicate.to_algebra_term(),
                    object: triple.object.to_algebra_term(),
                }))
            }
            Term::PropertyPath(path) => AlgebraTerm::PropertyPath(path.clone()),
        }
    }

    /// Check if this term represents a truthy value for SPARQL evaluation
    pub fn effective_boolean_value(&self) -> Result<bool> {
        match self {
            Term::Literal(lit_val) => match lit_val.datatype.as_str() {
                xsd::BOOLEAN => Ok(lit_val.lexical_form == "true" || lit_val.lexical_form == "1"),
                xsd::STRING => Ok(!lit_val.lexical_form.is_empty()),
                dt if dt.starts_with(XSD_NS)
                    && (dt.ends_with("integer")
                        || dt.ends_with("decimal")
                        || dt.ends_with("double")
                        || dt.ends_with("float")) =>
                {
                    let val = lit_val.lexical_form.parse::<f64>().unwrap_or(0.0);
                    Ok(val != 0.0 && !val.is_nan())
                }
                _ => Ok(!lit_val.lexical_form.is_empty()),
            },
            Term::Iri(_) | Term::BlankNode(_) | Term::QuotedTriple(_) | Term::PropertyPath(_) => Ok(true),
            Term::Variable(_) => bail!("Cannot evaluate variable as boolean"),
        }
    }

    /// Convert term to numeric value
    pub fn to_numeric(&self) -> Result<NumericValue> {
        match self {
            Term::Literal(lit_val) => lit_val.to_numeric(),
            Term::Iri(_) | Term::BlankNode(_) | Term::QuotedTriple(_) | Term::PropertyPath(_) => {
                bail!("Cannot convert IRI, blank node, quoted triple, or property path to numeric")
            }
            Term::Variable(_) => bail!("Cannot convert unbound variable to numeric"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let iri = Term::iri("http://example.org/foo");
        assert!(iri.is_iri());

        let blank = Term::blank_node("b1");
        assert!(blank.is_blank_node());

        let lit = Term::literal("hello");
        assert!(lit.is_literal());

        let var = Term::variable("x");
        assert!(var.is_variable());
    }

    #[test]
    fn test_typed_literals() {
        let int_lit = Term::typed_literal("42", xsd::INTEGER).unwrap();
        assert!(matches!(int_lit, Term::Literal(_)));

        let bool_lit = Term::typed_literal("true", xsd::BOOLEAN).unwrap();
        assert_eq!(bool_lit.effective_boolean_value().unwrap(), true);

        let date_lit = Term::typed_literal("2023-01-01", xsd::DATE).unwrap();
        assert!(matches!(date_lit, Term::Literal(_)));
    }

    #[test]
    fn test_numeric_conversion() {
        let int_term = Term::typed_literal("42", xsd::INTEGER).unwrap();
        let num = int_term.to_numeric().unwrap();
        assert_eq!(num, NumericValue::Integer(42));

        let float_term = Term::typed_literal("3.14", xsd::FLOAT).unwrap();
        let num = float_term.to_numeric().unwrap();
        assert!(matches!(num, NumericValue::Float(_)));
    }

    #[test]
    fn test_term_ordering() {
        let var = Term::variable("x");
        let blank = Term::blank_node("b1");
        let iri = Term::iri("http://example.org");
        let lit = Term::literal("test");

        assert!(var < blank);
        assert!(blank < iri);
        assert!(iri < lit);
    }

    #[test]
    fn test_binding_context() {
        let mut ctx = BindingContext::new();

        let term = Term::literal("value");
        ctx.bind("x", term.clone());

        assert!(ctx.is_bound("x"));
        assert_eq!(ctx.get("x"), Some(&term));

        ctx.push_scope();
        ctx.bind("y", Term::literal("other"));

        assert!(ctx.is_bound("x")); // Still visible
        assert!(ctx.is_bound("y"));

        ctx.pop_scope();
        assert!(ctx.is_bound("x"));
        assert!(!ctx.is_bound("y")); // No longer visible
    }

    #[test]
    fn test_pattern_matching() {
        let mut ctx = BindingContext::new();

        let pattern = Term::variable("x");
        let term = Term::literal("test");

        assert!(matches_pattern(&pattern, &term, &mut ctx));
        assert_eq!(ctx.get("x"), Some(&term));

        // Second match with same variable should check equality
        let term2 = Term::literal("other");
        assert!(!matches_pattern(&pattern, &term2, &mut ctx));
    }

    #[test]
    fn test_quoted_triple_term() {
        let subject = Term::iri("http://example.org/subject");
        let predicate = Term::iri("http://example.org/predicate");
        let object = Term::literal("object");

        let quoted_triple = Term::quoted_triple(subject.clone(), predicate.clone(), object.clone());
        assert!(quoted_triple.is_quoted_triple());

        // Test display format
        let display = format!("{}", quoted_triple);
        assert!(display.starts_with("<<"));
        assert!(display.ends_with(">>"));
    }

    #[test]
    fn test_property_path_term() {
        use crate::path::PropertyPath;

        let direct_path = PropertyPath::Direct(Term::iri("http://example.org/prop"));
        let path_term = Term::property_path(direct_path);
        assert!(path_term.is_property_path());
    }

    #[test]
    fn test_term_ordering_with_new_variants() {
        let var = Term::variable("x");
        let blank = Term::blank_node("b1");
        let iri = Term::iri("http://example.org");
        let lit = Term::literal("test");
        let quoted = Term::quoted_triple(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::iri("http://example.org/o"),
        );
        let path = Term::property_path(crate::path::PropertyPath::Direct(
            Term::iri("http://example.org/prop"),
        ));

        // Test ordering: Variable < BlankNode < Iri < Literal < QuotedTriple < PropertyPath
        assert!(var < blank);
        assert!(blank < iri);
        assert!(iri < lit);
        assert!(lit < quoted);
        assert!(quoted < path);
    }
}
