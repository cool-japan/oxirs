//! RDF-star TRIPLE Functions
//!
//! This module implements RDF-star (RDF 1.2) triple term functions
//! for constructing and deconstructing quoted triples.
//!
//! Based on the RDF-star specification and Apache Jena ARQ implementation.

use crate::algebra::{Term, TriplePattern};
use crate::extensions::{CustomFunction, ExecutionContext, Value, ValueType};
use anyhow::{bail, Result};

/// Convert Value to Term for triple construction
fn value_to_term(value: &Value) -> Result<Term> {
    match value {
        Value::Iri(iri) => {
            use oxirs_core::model::NamedNode;
            let node = NamedNode::new(iri)?;
            Ok(Term::Iri(node))
        }
        Value::Literal {
            value,
            language,
            datatype,
        } => {
            use oxirs_core::model::NamedNode;
            let dt = if let Some(dt_str) = datatype {
                Some(NamedNode::new(dt_str)?)
            } else {
                None
            };
            Ok(Term::Literal(crate::algebra::Literal {
                value: value.clone(),
                language: language.clone(),
                datatype: dt,
            }))
        }
        Value::BlankNode(id) => Ok(Term::BlankNode(id.clone())),
        _ => bail!("Cannot convert {} to RDF term", value.type_name()),
    }
}

/// Convert Term to Value for function results
#[allow(dead_code)]
fn term_to_value(term: &Term) -> Value {
    match term {
        Term::Iri(iri) => Value::Iri(iri.as_str().to_string()),
        Term::Literal(lit) => Value::Literal {
            value: lit.value.clone(),
            language: lit.language.clone(),
            datatype: lit.datatype.as_ref().map(|dt| dt.as_str().to_string()),
        },
        Term::BlankNode(id) => Value::BlankNode(id.clone()),
        Term::Variable(_v) => {
            // Variables should not appear in concrete triples
            // Return as a string representation
            Value::String(format!("{}", term))
        }
        Term::QuotedTriple(triple) => {
            // Return as a special QuotedTriple value
            // For now, represent as a string
            Value::String(format!(
                "<<{} {} {}>>",
                triple.subject, triple.predicate, triple.object
            ))
        }
        Term::PropertyPath(path) => Value::String(format!("{}", path)),
    }
}

// ============================================================================
// TRIPLE Function - Construct a quoted triple
// ============================================================================

/// TRIPLE(?s, ?p, ?o) - Construct a quoted triple from three terms
///
/// Creates an RDF-star quoted triple (embedded triple) from subject,
/// predicate, and object terms.
///
/// # Examples
/// ```sparql
/// TRIPLE(?s, ?p, ?o) → <<s p o>>
/// ```
#[derive(Debug, Clone)]
pub struct TripleFunction;

impl CustomFunction for TripleFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/sw/DataAccess/rq23#triple"
    }

    fn arity(&self) -> Option<usize> {
        Some(3)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![
            ValueType::Union(vec![
                ValueType::Iri,
                ValueType::BlankNode,
                ValueType::Custom("QuotedTriple".to_string()),
            ]),
            ValueType::Iri,
            ValueType::Union(vec![
                ValueType::Iri,
                ValueType::BlankNode,
                ValueType::Literal,
                ValueType::Custom("QuotedTriple".to_string()),
            ]),
        ]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Custom("QuotedTriple".to_string())
    }

    fn documentation(&self) -> &str {
        "Constructs a quoted triple (RDF-star) from subject, predicate, and object"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 3 {
            bail!("TRIPLE() requires exactly 3 arguments (subject, predicate, object)");
        }

        let subject = value_to_term(&args[0])?;
        let predicate = value_to_term(&args[1])?;
        let object = value_to_term(&args[2])?;

        // Validate subject, predicate, object constraints
        match &subject {
            Term::Literal(_) => bail!("TRIPLE subject cannot be a literal"),
            Term::PropertyPath(_) => bail!("TRIPLE subject cannot be a property path"),
            _ => {}
        }

        match &predicate {
            Term::Literal(_) => bail!("TRIPLE predicate cannot be a literal"),
            Term::BlankNode(_) => bail!("TRIPLE predicate cannot be a blank node"),
            Term::QuotedTriple(_) => bail!("TRIPLE predicate cannot be a quoted triple"),
            Term::PropertyPath(_) => bail!("TRIPLE predicate cannot be a property path"),
            _ => {}
        }

        // Create quoted triple
        let triple = TriplePattern {
            subject,
            predicate,
            object,
        };

        // Return as formatted string (in a full implementation, this would be a proper QuotedTriple value)
        Ok(Value::String(format!(
            "<<{} {} {}>>",
            triple.subject, triple.predicate, triple.object
        )))
    }
}

// ============================================================================
// SUBJECT Function - Extract subject from quoted triple
// ============================================================================

/// SUBJECT(?triple) - Extract subject from a quoted triple
///
/// Returns the subject component of an RDF-star quoted triple.
///
/// # Examples
/// ```sparql
/// SUBJECT(<<:s :p :o>>) → :s
/// ```
#[derive(Debug, Clone)]
pub struct SubjectFunction;

impl CustomFunction for SubjectFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/sw/DataAccess/rq23#subject"
    }

    fn arity(&self) -> Option<usize> {
        Some(1)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Custom("QuotedTriple".to_string())]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Union(vec![
            ValueType::Iri,
            ValueType::BlankNode,
            ValueType::Custom("QuotedTriple".to_string()),
        ])
    }

    fn documentation(&self) -> &str {
        "Extracts the subject from a quoted triple (RDF-star)"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("SUBJECT() requires exactly 1 argument");
        }

        // In a full implementation, we would parse the quoted triple from the value
        // For now, we'll check for a string representation
        match &args[0] {
            Value::String(s) if s.starts_with("<<") && s.ends_with(">>") => {
                // Simple parsing for demonstration
                // Real implementation would use proper triple term parsing
                let inner = &s[2..s.len() - 2];
                let parts: Vec<&str> = inner.split_whitespace().collect();
                if parts.len() >= 3 {
                    Ok(Value::Iri(parts[0].to_string()))
                } else {
                    bail!("Invalid quoted triple format");
                }
            }
            _ => bail!("SUBJECT() requires a quoted triple argument"),
        }
    }
}

// ============================================================================
// PREDICATE Function - Extract predicate from quoted triple
// ============================================================================

/// PREDICATE(?triple) - Extract predicate from a quoted triple
///
/// Returns the predicate component of an RDF-star quoted triple.
///
/// # Examples
/// ```sparql
/// PREDICATE(<<:s :p :o>>) → :p
/// ```
#[derive(Debug, Clone)]
pub struct PredicateFunction;

impl CustomFunction for PredicateFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/sw/DataAccess/rq23#predicate"
    }

    fn arity(&self) -> Option<usize> {
        Some(1)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Custom("QuotedTriple".to_string())]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Iri
    }

    fn documentation(&self) -> &str {
        "Extracts the predicate from a quoted triple (RDF-star)"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("PREDICATE() requires exactly 1 argument");
        }

        match &args[0] {
            Value::String(s) if s.starts_with("<<") && s.ends_with(">>") => {
                let inner = &s[2..s.len() - 2];
                let parts: Vec<&str> = inner.split_whitespace().collect();
                if parts.len() >= 3 {
                    Ok(Value::Iri(parts[1].to_string()))
                } else {
                    bail!("Invalid quoted triple format");
                }
            }
            _ => bail!("PREDICATE() requires a quoted triple argument"),
        }
    }
}

// ============================================================================
// OBJECT Function - Extract object from quoted triple
// ============================================================================

/// OBJECT(?triple) - Extract object from a quoted triple
///
/// Returns the object component of an RDF-star quoted triple.
///
/// # Examples
/// ```sparql
/// OBJECT(<<:s :p :o>>) → :o
/// ```
#[derive(Debug, Clone)]
pub struct ObjectFunction;

impl CustomFunction for ObjectFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/sw/DataAccess/rq23#object"
    }

    fn arity(&self) -> Option<usize> {
        Some(1)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Custom("QuotedTriple".to_string())]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Union(vec![
            ValueType::Iri,
            ValueType::BlankNode,
            ValueType::Literal,
            ValueType::Custom("QuotedTriple".to_string()),
        ])
    }

    fn documentation(&self) -> &str {
        "Extracts the object from a quoted triple (RDF-star)"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("OBJECT() requires exactly 1 argument");
        }

        match &args[0] {
            Value::String(s) if s.starts_with("<<") && s.ends_with(">>") => {
                let inner = &s[2..s.len() - 2];
                let parts: Vec<&str> = inner.split_whitespace().collect();
                if parts.len() >= 3 {
                    // Join remaining parts for the object (in case it has spaces)
                    Ok(Value::Iri(parts[2..].join(" ")))
                } else {
                    bail!("Invalid quoted triple format");
                }
            }
            _ => bail!("OBJECT() requires a quoted triple argument"),
        }
    }
}

// ============================================================================
// isTRIPLE Function - Check if value is a quoted triple
// ============================================================================

/// isTRIPLE(?x) - Check if a value is a quoted triple
///
/// Returns true if the argument is an RDF-star quoted triple,
/// false otherwise.
///
/// # Examples
/// ```sparql
/// isTRIPLE(<<:s :p :o>>) → true
/// isTRIPLE(:s) → false
/// ```
#[derive(Debug, Clone)]
pub struct IsTripleFunction;

impl CustomFunction for IsTripleFunction {
    fn name(&self) -> &str {
        "http://www.w3.org/2001/sw/DataAccess/rq23#isTRIPLE"
    }

    fn arity(&self) -> Option<usize> {
        Some(1)
    }

    fn parameter_types(&self) -> Vec<ValueType> {
        vec![ValueType::Union(vec![
            ValueType::Iri,
            ValueType::BlankNode,
            ValueType::Literal,
            ValueType::Custom("QuotedTriple".to_string()),
        ])]
    }

    fn return_type(&self) -> ValueType {
        ValueType::Boolean
    }

    fn documentation(&self) -> &str {
        "Returns true if the argument is a quoted triple (RDF-star), false otherwise"
    }

    fn clone_function(&self) -> Box<dyn CustomFunction> {
        Box::new(self.clone())
    }

    fn execute(&self, args: &[Value], _context: &ExecutionContext) -> Result<Value> {
        if args.len() != 1 {
            bail!("isTRIPLE() requires exactly 1 argument");
        }

        let is_triple = match &args[0] {
            Value::String(s) => s.starts_with("<<") && s.ends_with(">>"),
            _ => false,
        };

        Ok(Value::Boolean(is_triple))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_context() -> ExecutionContext {
        ExecutionContext {
            variables: HashMap::new(),
            namespaces: HashMap::new(),
            base_iri: None,
            dataset_context: None,
            query_time: chrono::Utc::now(),
            optimization_level: crate::extensions::OptimizationLevel::None,
            memory_limit: None,
            time_limit: None,
        }
    }

    #[test]
    fn test_triple_function_construction() {
        let func = TripleFunction;
        let ctx = create_test_context();

        let subject = Value::Iri("http://example.org/subject".to_string());
        let predicate = Value::Iri("http://example.org/predicate".to_string());
        let object = Value::Iri("http://example.org/object".to_string());

        let result = func.execute(&[subject, predicate, object], &ctx);
        assert!(result.is_ok());

        if let Ok(Value::String(s)) = result {
            assert!(s.starts_with("<<"));
            assert!(s.ends_with(">>"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_triple_function_invalid_subject() {
        let func = TripleFunction;
        let ctx = create_test_context();

        // Literal as subject should fail
        let subject = Value::Literal {
            value: "literal".to_string(),
            language: None,
            datatype: None,
        };
        let predicate = Value::Iri("http://example.org/predicate".to_string());
        let object = Value::Iri("http://example.org/object".to_string());

        let result = func.execute(&[subject, predicate, object], &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_subject_function() {
        let func = SubjectFunction;
        let ctx = create_test_context();

        let triple = Value::String(
            "<<http://example.org/s http://example.org/p http://example.org/o>>".to_string(),
        );

        let result = func.execute(&[triple], &ctx);
        assert!(result.is_ok());

        if let Ok(Value::Iri(iri)) = result {
            assert_eq!(iri, "http://example.org/s");
        } else {
            panic!("Expected IRI result");
        }
    }

    #[test]
    fn test_predicate_function() {
        let func = PredicateFunction;
        let ctx = create_test_context();

        let triple = Value::String(
            "<<http://example.org/s http://example.org/p http://example.org/o>>".to_string(),
        );

        let result = func.execute(&[triple], &ctx);
        assert!(result.is_ok());

        if let Ok(Value::Iri(iri)) = result {
            assert_eq!(iri, "http://example.org/p");
        } else {
            panic!("Expected IRI result");
        }
    }

    #[test]
    fn test_object_function() {
        let func = ObjectFunction;
        let ctx = create_test_context();

        let triple = Value::String(
            "<<http://example.org/s http://example.org/p http://example.org/o>>".to_string(),
        );

        let result = func.execute(&[triple], &ctx);
        assert!(result.is_ok());

        if let Ok(Value::Iri(iri)) = result {
            assert_eq!(iri, "http://example.org/o");
        } else {
            panic!("Expected IRI result");
        }
    }

    #[test]
    fn test_is_triple_function() {
        let func = IsTripleFunction;
        let ctx = create_test_context();

        // Test with quoted triple
        let triple = Value::String(
            "<<http://example.org/s http://example.org/p http://example.org/o>>".to_string(),
        );
        let result = func.execute(&[triple], &ctx).unwrap();
        assert_eq!(result, Value::Boolean(true));

        // Test with non-triple
        let non_triple = Value::Iri("http://example.org/resource".to_string());
        let result = func.execute(&[non_triple], &ctx).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_triple_function_arity() {
        let func = TripleFunction;
        assert_eq!(func.arity(), Some(3));
        assert_eq!(
            func.name(),
            "http://www.w3.org/2001/sw/DataAccess/rq23#triple"
        );
    }

    #[test]
    fn test_accessor_function_arities() {
        assert_eq!(SubjectFunction.arity(), Some(1));
        assert_eq!(PredicateFunction.arity(), Some(1));
        assert_eq!(ObjectFunction.arity(), Some(1));
        assert_eq!(IsTripleFunction.arity(), Some(1));
    }
}
