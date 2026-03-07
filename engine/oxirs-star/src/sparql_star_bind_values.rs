//! # SPARQL-Star BIND/VALUES Extensions
//!
//! Extends SPARQL BIND and VALUES clauses to handle quoted triples (RDF-star),
//! enabling patterns like:
//!
//! ```sparql
//! BIND(<< ?s ?p ?o >> AS ?qt)
//! VALUES ?qt { << ex:s ex:p "o" >> << ex:s2 ex:p2 "o2" >> }
//! ```
//!
//! ## Features
//!
//! - BIND with quoted triple construction
//! - VALUES clause with inline quoted triples
//! - Quoted triple decomposition (extracting s/p/o from a bound QT variable)
//! - Nested quoted triple support in BIND/VALUES
//! - Coalesce/If expressions over quoted triples
//!
//! ## References
//!
//! - RDF-star and SPARQL-star Community Group Report
//! - W3C RDF-star Working Group specifications

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Quoted triple representation
// ---------------------------------------------------------------------------

/// An RDF-star term that may be a quoted triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StarTerm {
    /// An IRI reference.
    Iri(String),
    /// A literal with optional datatype and language tag.
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A blank node.
    BlankNode(String),
    /// A quoted triple (RDF-star).
    QuotedTriple(Box<QuotedTriple>),
    /// A SPARQL variable (for pattern matching).
    Variable(String),
}

impl fmt::Display for StarTerm {
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
            Self::QuotedTriple(qt) => write!(f, "<< {qt} >>"),
            Self::Variable(name) => write!(f, "?{name}"),
        }
    }
}

impl StarTerm {
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

    /// Create a quoted triple term.
    pub fn quoted(subject: StarTerm, predicate: StarTerm, object: StarTerm) -> Self {
        Self::QuotedTriple(Box::new(QuotedTriple {
            subject,
            predicate,
            object,
        }))
    }

    /// Create a variable.
    pub fn var(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }

    /// Check if this is a quoted triple.
    pub fn is_quoted_triple(&self) -> bool {
        matches!(self, Self::QuotedTriple(_))
    }

    /// Get the inner quoted triple if this is one.
    pub fn as_quoted_triple(&self) -> Option<&QuotedTriple> {
        match self {
            Self::QuotedTriple(qt) => Some(qt),
            _ => None,
        }
    }

    /// Check if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Nesting depth (0 for non-quoted, 1+ for quoted with possible nesting).
    pub fn nesting_depth(&self) -> usize {
        match self {
            Self::QuotedTriple(qt) => {
                1 + qt
                    .subject
                    .nesting_depth()
                    .max(qt.predicate.nesting_depth())
                    .max(qt.object.nesting_depth())
            }
            _ => 0,
        }
    }
}

/// A quoted triple (RDF-star triple used as a term).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuotedTriple {
    /// Subject of the quoted triple.
    pub subject: StarTerm,
    /// Predicate of the quoted triple.
    pub predicate: StarTerm,
    /// Object of the quoted triple.
    pub object: StarTerm,
}

impl fmt::Display for QuotedTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)
    }
}

// ---------------------------------------------------------------------------
// BIND expression for quoted triples
// ---------------------------------------------------------------------------

/// A BIND expression that may involve quoted triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StarBindExpr {
    /// Bind a quoted triple to a variable.
    /// `BIND(<< ?s ?p ?o >> AS ?qt)`
    ConstructQuotedTriple {
        subject: StarTerm,
        predicate: StarTerm,
        object: StarTerm,
        target_var: String,
    },

    /// Decompose a quoted triple variable into s/p/o.
    /// `BIND(SUBJECT(?qt) AS ?s)`, etc.
    DecomposeSubject {
        source_var: String,
        target_var: String,
    },
    /// Decompose predicate.
    DecomposePredicate {
        source_var: String,
        target_var: String,
    },
    /// Decompose object.
    DecomposeObject {
        source_var: String,
        target_var: String,
    },

    /// COALESCE over quoted triple expressions.
    Coalesce {
        alternatives: Vec<StarTerm>,
        target_var: String,
    },

    /// IF/THEN/ELSE with quoted triple awareness.
    IfThenElse {
        condition: StarCondition,
        then_value: StarTerm,
        else_value: StarTerm,
        target_var: String,
    },

    /// Bind a constant term.
    Constant { value: StarTerm, target_var: String },
}

/// A condition for IF expressions in BIND.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StarCondition {
    /// Check if a variable is bound.
    IsBound(String),
    /// Check if a term is a quoted triple.
    IsTriple(String),
    /// Equality check.
    Equals(StarTerm, StarTerm),
}

// ---------------------------------------------------------------------------
// VALUES clause for quoted triples
// ---------------------------------------------------------------------------

/// A VALUES clause that may contain quoted triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarValuesClause {
    /// Variable names in the VALUES clause.
    pub variables: Vec<String>,
    /// Rows of bindings (each row maps variable name -> term).
    pub rows: Vec<HashMap<String, StarTerm>>,
}

impl StarValuesClause {
    /// Create a new empty VALUES clause with the given variables.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            rows: Vec::new(),
        }
    }

    /// Add a row of bindings.
    pub fn add_row(&mut self, bindings: HashMap<String, StarTerm>) -> &mut Self {
        self.rows.push(bindings);
        self
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Check if any row contains a quoted triple.
    pub fn has_quoted_triples(&self) -> bool {
        self.rows
            .iter()
            .any(|row| row.values().any(|t| t.is_quoted_triple()))
    }

    /// Maximum nesting depth across all values.
    pub fn max_nesting_depth(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|row| row.values())
            .map(|t| t.nesting_depth())
            .max()
            .unwrap_or(0)
    }

    /// Get all unique values for a variable.
    pub fn unique_values(&self, variable: &str) -> Vec<&StarTerm> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for row in &self.rows {
            if let Some(val) = row.get(variable) {
                let key = format!("{val}");
                if seen.insert(key) {
                    result.push(val);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// BIND evaluator
// ---------------------------------------------------------------------------

/// Evaluates BIND expressions involving quoted triples.
pub struct StarBindEvaluator;

/// A solution binding: variable name -> term.
pub type StarBindings = HashMap<String, StarTerm>;

/// Error from BIND evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StarBindError {
    /// Variable is not bound.
    UnboundVariable(String),
    /// Term is not a quoted triple (when decomposition was requested).
    NotAQuotedTriple(String),
    /// Type error.
    TypeError(String),
}

impl fmt::Display for StarBindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundVariable(v) => write!(f, "Variable ?{v} is not bound"),
            Self::NotAQuotedTriple(v) => write!(f, "Variable ?{v} is not a quoted triple"),
            Self::TypeError(msg) => write!(f, "Type error: {msg}"),
        }
    }
}

impl std::error::Error for StarBindError {}

impl StarBindEvaluator {
    /// Evaluate a BIND expression against the current bindings.
    pub fn evaluate(
        expr: &StarBindExpr,
        bindings: &StarBindings,
    ) -> Result<StarBindings, StarBindError> {
        let mut result = bindings.clone();

        match expr {
            StarBindExpr::ConstructQuotedTriple {
                subject,
                predicate,
                object,
                target_var,
            } => {
                let s = Self::resolve_term(subject, bindings)?;
                let p = Self::resolve_term(predicate, bindings)?;
                let o = Self::resolve_term(object, bindings)?;
                let qt = StarTerm::quoted(s, p, o);
                result.insert(target_var.clone(), qt);
            }

            StarBindExpr::DecomposeSubject {
                source_var,
                target_var,
            } => {
                let qt = Self::get_quoted_triple(source_var, bindings)?;
                result.insert(target_var.clone(), qt.subject.clone());
            }

            StarBindExpr::DecomposePredicate {
                source_var,
                target_var,
            } => {
                let qt = Self::get_quoted_triple(source_var, bindings)?;
                result.insert(target_var.clone(), qt.predicate.clone());
            }

            StarBindExpr::DecomposeObject {
                source_var,
                target_var,
            } => {
                let qt = Self::get_quoted_triple(source_var, bindings)?;
                result.insert(target_var.clone(), qt.object.clone());
            }

            StarBindExpr::Coalesce {
                alternatives,
                target_var,
            } => {
                for alt in alternatives {
                    if let Ok(resolved) = Self::resolve_term(alt, bindings) {
                        result.insert(target_var.clone(), resolved);
                        return Ok(result);
                    }
                }
                // All alternatives failed — variable remains unbound
            }

            StarBindExpr::IfThenElse {
                condition,
                then_value,
                else_value,
                target_var,
            } => {
                let cond_result = Self::evaluate_condition(condition, bindings);
                let value = if cond_result {
                    Self::resolve_term(then_value, bindings)?
                } else {
                    Self::resolve_term(else_value, bindings)?
                };
                result.insert(target_var.clone(), value);
            }

            StarBindExpr::Constant { value, target_var } => {
                result.insert(target_var.clone(), value.clone());
            }
        }

        Ok(result)
    }

    /// Resolve a StarTerm, substituting variables from bindings.
    fn resolve_term(term: &StarTerm, bindings: &StarBindings) -> Result<StarTerm, StarBindError> {
        match term {
            StarTerm::Variable(name) => bindings
                .get(name)
                .cloned()
                .ok_or_else(|| StarBindError::UnboundVariable(name.clone())),
            StarTerm::QuotedTriple(qt) => {
                let s = Self::resolve_term(&qt.subject, bindings)?;
                let p = Self::resolve_term(&qt.predicate, bindings)?;
                let o = Self::resolve_term(&qt.object, bindings)?;
                Ok(StarTerm::quoted(s, p, o))
            }
            other => Ok(other.clone()),
        }
    }

    /// Get a quoted triple from a variable binding.
    fn get_quoted_triple<'a>(
        var: &str,
        bindings: &'a StarBindings,
    ) -> Result<&'a QuotedTriple, StarBindError> {
        let term = bindings
            .get(var)
            .ok_or_else(|| StarBindError::UnboundVariable(var.to_string()))?;
        term.as_quoted_triple()
            .ok_or_else(|| StarBindError::NotAQuotedTriple(var.to_string()))
    }

    /// Evaluate a condition.
    fn evaluate_condition(condition: &StarCondition, bindings: &StarBindings) -> bool {
        match condition {
            StarCondition::IsBound(var) => bindings.contains_key(var),
            StarCondition::IsTriple(var) => bindings
                .get(var)
                .map(|t| t.is_quoted_triple())
                .unwrap_or(false),
            StarCondition::Equals(a, b) => {
                let resolved_a = Self::resolve_term(a, bindings).ok();
                let resolved_b = Self::resolve_term(b, bindings).ok();
                match (resolved_a, resolved_b) {
                    (Some(ra), Some(rb)) => ra == rb,
                    _ => false,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VALUES evaluator
// ---------------------------------------------------------------------------

/// Evaluates VALUES clauses with quoted triple support.
pub struct StarValuesEvaluator;

impl StarValuesEvaluator {
    /// Join the VALUES rows with existing bindings.
    ///
    /// For each row in the VALUES clause, check compatibility with the
    /// input bindings and produce merged results.
    pub fn evaluate(
        clause: &StarValuesClause,
        input_bindings: &[StarBindings],
    ) -> Vec<StarBindings> {
        let mut results = Vec::new();

        if input_bindings.is_empty() {
            // No input bindings — just return the VALUES rows
            return clause.rows.clone();
        }

        for input in input_bindings {
            for row in &clause.rows {
                if Self::is_compatible(input, row) {
                    let mut merged = input.clone();
                    for (var, val) in row {
                        merged.insert(var.clone(), val.clone());
                    }
                    results.push(merged);
                }
            }
        }

        results
    }

    /// Check if two binding sets are compatible (shared variables have equal values).
    fn is_compatible(a: &StarBindings, b: &StarBindings) -> bool {
        for (var, val_a) in a {
            if let Some(val_b) = b.get(var) {
                if val_a != val_b {
                    return false;
                }
            }
        }
        true
    }

    /// Filter VALUES rows that match a pattern.
    pub fn filter_rows(
        clause: &StarValuesClause,
        variable: &str,
        predicate: impl Fn(&StarTerm) -> bool,
    ) -> Vec<HashMap<String, StarTerm>> {
        clause
            .rows
            .iter()
            .filter(|row| row.get(variable).map(&predicate).unwrap_or(false))
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(s: &str) -> String {
        format!("http://example.org/{s}")
    }

    fn ex_iri(s: &str) -> StarTerm {
        StarTerm::iri(ex(s))
    }

    fn make_qt(s: &str, p: &str, o: &str) -> StarTerm {
        StarTerm::quoted(ex_iri(s), ex_iri(p), StarTerm::literal(o))
    }

    // ── StarTerm tests ────────────────────────────────────────────────────

    #[test]
    fn test_star_term_display_iri() {
        let t = ex_iri("foo");
        assert_eq!(format!("{t}"), "<http://example.org/foo>");
    }

    #[test]
    fn test_star_term_display_literal() {
        let t = StarTerm::literal("hello");
        assert_eq!(format!("{t}"), "\"hello\"");
    }

    #[test]
    fn test_star_term_display_typed_literal() {
        let t = StarTerm::typed_literal("42", "xsd:integer");
        assert_eq!(format!("{t}"), "\"42\"^^<xsd:integer>");
    }

    #[test]
    fn test_star_term_display_quoted() {
        let t = make_qt("alice", "knows", "Bob");
        let s = format!("{t}");
        assert!(s.contains("<<"));
        assert!(s.contains(">>"));
        assert!(s.contains("alice"));
    }

    #[test]
    fn test_star_term_display_variable() {
        let t = StarTerm::var("x");
        assert_eq!(format!("{t}"), "?x");
    }

    #[test]
    fn test_star_term_is_quoted_triple() {
        assert!(make_qt("s", "p", "o").is_quoted_triple());
        assert!(!ex_iri("x").is_quoted_triple());
    }

    #[test]
    fn test_star_term_is_variable() {
        assert!(StarTerm::var("x").is_variable());
        assert!(!ex_iri("x").is_variable());
    }

    #[test]
    fn test_star_term_nesting_depth() {
        assert_eq!(ex_iri("x").nesting_depth(), 0);
        assert_eq!(make_qt("s", "p", "o").nesting_depth(), 1);

        // Nested: << << s p o >> p2 o2 >>
        let nested = StarTerm::quoted(
            make_qt("s", "p", "o"),
            ex_iri("p2"),
            StarTerm::literal("o2"),
        );
        assert_eq!(nested.nesting_depth(), 2);
    }

    #[test]
    fn test_star_term_as_quoted_triple() {
        let qt = make_qt("s", "p", "o");
        assert!(qt.as_quoted_triple().is_some());
        assert!(ex_iri("x").as_quoted_triple().is_none());
    }

    // ── QuotedTriple display ──────────────────────────────────────────────

    #[test]
    fn test_quoted_triple_display() {
        let qt = QuotedTriple {
            subject: ex_iri("s"),
            predicate: ex_iri("p"),
            object: StarTerm::literal("o"),
        };
        let s = format!("{qt}");
        assert!(s.contains("s"));
        assert!(s.contains("p"));
        assert!(s.contains("o"));
    }

    // ── BIND: construct quoted triple ─────────────────────────────────────

    #[test]
    fn test_bind_construct_qt() {
        let mut bindings = StarBindings::new();
        bindings.insert("s".to_string(), ex_iri("alice"));
        bindings.insert("p".to_string(), ex_iri("knows"));
        bindings.insert("o".to_string(), ex_iri("bob"));

        let expr = StarBindExpr::ConstructQuotedTriple {
            subject: StarTerm::var("s"),
            predicate: StarTerm::var("p"),
            object: StarTerm::var("o"),
            target_var: "qt".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let qt = result.get("qt").expect("qt should be bound");
        assert!(qt.is_quoted_triple());
    }

    #[test]
    fn test_bind_construct_qt_with_constants() {
        let bindings = StarBindings::new();

        let expr = StarBindExpr::ConstructQuotedTriple {
            subject: ex_iri("alice"),
            predicate: ex_iri("age"),
            object: StarTerm::typed_literal("30", "xsd:integer"),
            target_var: "qt".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let qt = result.get("qt").expect("qt should be bound");
        assert!(qt.is_quoted_triple());
    }

    // ── BIND: decompose ───────────────────────────────────────────────────

    #[test]
    fn test_bind_decompose_subject() {
        let mut bindings = StarBindings::new();
        bindings.insert("qt".to_string(), make_qt("alice", "knows", "Bob"));

        let expr = StarBindExpr::DecomposeSubject {
            source_var: "qt".to_string(),
            target_var: "subj".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let subj = result.get("subj").expect("subj should be bound");
        assert_eq!(*subj, ex_iri("alice"));
    }

    #[test]
    fn test_bind_decompose_predicate() {
        let mut bindings = StarBindings::new();
        bindings.insert("qt".to_string(), make_qt("alice", "knows", "Bob"));

        let expr = StarBindExpr::DecomposePredicate {
            source_var: "qt".to_string(),
            target_var: "pred".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let pred = result.get("pred").expect("pred should be bound");
        assert_eq!(*pred, ex_iri("knows"));
    }

    #[test]
    fn test_bind_decompose_object() {
        let mut bindings = StarBindings::new();
        bindings.insert("qt".to_string(), make_qt("alice", "knows", "Bob"));

        let expr = StarBindExpr::DecomposeObject {
            source_var: "qt".to_string(),
            target_var: "obj".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let obj = result.get("obj").expect("obj should be bound");
        assert_eq!(*obj, StarTerm::literal("Bob"));
    }

    #[test]
    fn test_bind_decompose_not_a_qt() {
        let mut bindings = StarBindings::new();
        bindings.insert("x".to_string(), ex_iri("alice"));

        let expr = StarBindExpr::DecomposeSubject {
            source_var: "x".to_string(),
            target_var: "subj".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings);
        assert!(matches!(result, Err(StarBindError::NotAQuotedTriple(_))));
    }

    #[test]
    fn test_bind_decompose_unbound() {
        let bindings = StarBindings::new();

        let expr = StarBindExpr::DecomposeSubject {
            source_var: "missing".to_string(),
            target_var: "subj".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings);
        assert!(matches!(result, Err(StarBindError::UnboundVariable(_))));
    }

    // ── BIND: coalesce ────────────────────────────────────────────────────

    #[test]
    fn test_bind_coalesce_first_bound() {
        let mut bindings = StarBindings::new();
        bindings.insert("a".to_string(), ex_iri("first"));

        let expr = StarBindExpr::Coalesce {
            alternatives: vec![StarTerm::var("a"), StarTerm::var("b")],
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&ex_iri("first")));
    }

    #[test]
    fn test_bind_coalesce_fallback() {
        let mut bindings = StarBindings::new();
        bindings.insert("b".to_string(), ex_iri("second"));

        let expr = StarBindExpr::Coalesce {
            alternatives: vec![StarTerm::var("a"), StarTerm::var("b")],
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&ex_iri("second")));
    }

    #[test]
    fn test_bind_coalesce_constant_fallback() {
        let bindings = StarBindings::new();

        let expr = StarBindExpr::Coalesce {
            alternatives: vec![StarTerm::var("missing"), StarTerm::literal("default")],
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&StarTerm::literal("default")));
    }

    // ── BIND: if-then-else ────────────────────────────────────────────────

    #[test]
    fn test_bind_if_is_bound_true() {
        let mut bindings = StarBindings::new();
        bindings.insert("x".to_string(), ex_iri("val"));

        let expr = StarBindExpr::IfThenElse {
            condition: StarCondition::IsBound("x".to_string()),
            then_value: StarTerm::literal("yes"),
            else_value: StarTerm::literal("no"),
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&StarTerm::literal("yes")));
    }

    #[test]
    fn test_bind_if_is_bound_false() {
        let bindings = StarBindings::new();

        let expr = StarBindExpr::IfThenElse {
            condition: StarCondition::IsBound("x".to_string()),
            then_value: StarTerm::literal("yes"),
            else_value: StarTerm::literal("no"),
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&StarTerm::literal("no")));
    }

    #[test]
    fn test_bind_if_is_triple() {
        let mut bindings = StarBindings::new();
        bindings.insert("qt".to_string(), make_qt("s", "p", "o"));

        let expr = StarBindExpr::IfThenElse {
            condition: StarCondition::IsTriple("qt".to_string()),
            then_value: StarTerm::literal("is_triple"),
            else_value: StarTerm::literal("not_triple"),
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&StarTerm::literal("is_triple")));
    }

    #[test]
    fn test_bind_if_equals() {
        let mut bindings = StarBindings::new();
        bindings.insert("x".to_string(), ex_iri("alice"));

        let expr = StarBindExpr::IfThenElse {
            condition: StarCondition::Equals(StarTerm::var("x"), ex_iri("alice")),
            then_value: StarTerm::literal("match"),
            else_value: StarTerm::literal("no_match"),
            target_var: "result".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert_eq!(result.get("result"), Some(&StarTerm::literal("match")));
    }

    // ── BIND: constant ────────────────────────────────────────────────────

    #[test]
    fn test_bind_constant() {
        let bindings = StarBindings::new();
        let expr = StarBindExpr::Constant {
            value: make_qt("s", "p", "o"),
            target_var: "qt".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        assert!(result.get("qt").expect("bound").is_quoted_triple());
    }

    // ── VALUES clause ─────────────────────────────────────────────────────

    #[test]
    fn test_values_clause_new() {
        let clause = StarValuesClause::new(vec!["x".to_string()]);
        assert!(clause.is_empty());
        assert_eq!(clause.len(), 0);
    }

    #[test]
    fn test_values_clause_add_rows() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        let mut row = HashMap::new();
        row.insert("x".to_string(), ex_iri("alice"));
        clause.add_row(row);
        assert_eq!(clause.len(), 1);
    }

    #[test]
    fn test_values_has_quoted_triples() {
        let mut clause = StarValuesClause::new(vec!["qt".to_string()]);
        let mut row = HashMap::new();
        row.insert("qt".to_string(), make_qt("s", "p", "o"));
        clause.add_row(row);
        assert!(clause.has_quoted_triples());
    }

    #[test]
    fn test_values_no_quoted_triples() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        let mut row = HashMap::new();
        row.insert("x".to_string(), ex_iri("alice"));
        clause.add_row(row);
        assert!(!clause.has_quoted_triples());
    }

    #[test]
    fn test_values_max_nesting_depth() {
        let mut clause = StarValuesClause::new(vec!["qt".to_string()]);
        let nested = StarTerm::quoted(
            make_qt("s", "p", "o"),
            ex_iri("p2"),
            StarTerm::literal("o2"),
        );
        let mut row = HashMap::new();
        row.insert("qt".to_string(), nested);
        clause.add_row(row);
        assert_eq!(clause.max_nesting_depth(), 2);
    }

    #[test]
    fn test_values_unique_values() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        for _ in 0..3 {
            let mut row = HashMap::new();
            row.insert("x".to_string(), ex_iri("same"));
            clause.add_row(row);
        }
        let mut row = HashMap::new();
        row.insert("x".to_string(), ex_iri("different"));
        clause.add_row(row);

        let unique = clause.unique_values("x");
        assert_eq!(unique.len(), 2);
    }

    // ── VALUES evaluator ──────────────────────────────────────────────────

    #[test]
    fn test_values_evaluate_no_input() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        let mut row = HashMap::new();
        row.insert("x".to_string(), ex_iri("alice"));
        clause.add_row(row);

        let results = StarValuesEvaluator::evaluate(&clause, &[]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_values_evaluate_with_compatible_bindings() {
        let mut clause = StarValuesClause::new(vec!["name".to_string()]);
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), StarTerm::literal("Alice"));
        clause.add_row(row1);
        let mut row2 = HashMap::new();
        row2.insert("name".to_string(), StarTerm::literal("Bob"));
        clause.add_row(row2);

        let mut input = StarBindings::new();
        input.insert("age".to_string(), StarTerm::literal("30"));

        let results = StarValuesEvaluator::evaluate(&clause, &[input]);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains_key("name"));
        assert!(results[0].contains_key("age"));
    }

    #[test]
    fn test_values_evaluate_with_conflict() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        let mut row = HashMap::new();
        row.insert("x".to_string(), ex_iri("alice"));
        clause.add_row(row);

        let mut input = StarBindings::new();
        input.insert("x".to_string(), ex_iri("bob")); // conflicts!

        let results = StarValuesEvaluator::evaluate(&clause, &[input]);
        assert!(results.is_empty()); // No compatible rows
    }

    #[test]
    fn test_values_evaluate_with_qt_bindings() {
        let mut clause = StarValuesClause::new(vec!["qt".to_string()]);
        let mut row = HashMap::new();
        row.insert("qt".to_string(), make_qt("alice", "knows", "Bob"));
        clause.add_row(row);

        let results = StarValuesEvaluator::evaluate(&clause, &[]);
        assert_eq!(results.len(), 1);
        assert!(results[0].get("qt").expect("bound").is_quoted_triple());
    }

    #[test]
    fn test_values_filter_rows() {
        let mut clause = StarValuesClause::new(vec!["x".to_string()]);
        for name in &["alice", "bob", "charlie"] {
            let mut row = HashMap::new();
            row.insert("x".to_string(), ex_iri(name));
            clause.add_row(row);
        }

        let filtered =
            StarValuesEvaluator::filter_rows(&clause, "x", |t| format!("{t}").contains("alice"));
        assert_eq!(filtered.len(), 1);
    }

    // ── Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = StarBindError::UnboundVariable("x".to_string());
        assert!(format!("{e}").contains("x"));

        let e = StarBindError::NotAQuotedTriple("y".to_string());
        assert!(format!("{e}").contains("y"));

        let e = StarBindError::TypeError("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    // ── Nested quoted triple in BIND ──────────────────────────────────────

    #[test]
    fn test_bind_nested_qt() {
        let inner_qt = make_qt("alice", "knows", "Bob");
        let mut bindings = StarBindings::new();
        bindings.insert("inner".to_string(), inner_qt);

        let expr = StarBindExpr::ConstructQuotedTriple {
            subject: StarTerm::var("inner"),
            predicate: ex_iri("certainty"),
            object: StarTerm::literal("0.9"),
            target_var: "outer".to_string(),
        };

        let result = StarBindEvaluator::evaluate(&expr, &bindings).expect("should succeed");
        let outer = result.get("outer").expect("outer bound");
        assert_eq!(outer.nesting_depth(), 2);
    }

    // ── Multiple BIND expressions in sequence ─────────────────────────────

    #[test]
    fn test_bind_chain() {
        let mut bindings = StarBindings::new();
        bindings.insert("s".to_string(), ex_iri("alice"));
        bindings.insert("p".to_string(), ex_iri("knows"));
        bindings.insert("o".to_string(), ex_iri("bob"));

        // Step 1: Construct QT
        let expr1 = StarBindExpr::ConstructQuotedTriple {
            subject: StarTerm::var("s"),
            predicate: StarTerm::var("p"),
            object: StarTerm::var("o"),
            target_var: "qt".to_string(),
        };
        let bindings = StarBindEvaluator::evaluate(&expr1, &bindings).expect("step 1");

        // Step 2: Decompose subject
        let expr2 = StarBindExpr::DecomposeSubject {
            source_var: "qt".to_string(),
            target_var: "extracted_s".to_string(),
        };
        let bindings = StarBindEvaluator::evaluate(&expr2, &bindings).expect("step 2");

        assert_eq!(bindings.get("extracted_s"), Some(&ex_iri("alice")));
    }
}
