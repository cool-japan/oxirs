//! SHACL shape parsing and representation module
//!
//! This module is refactored from the original shapes.rs file to improve maintainability
//! and adhere to the 2000-line file limit policy.

pub mod factory;
pub mod parser;
pub mod types;
pub mod validator;

// Re-export main types and functions for backward compatibility
pub use factory::ShapeFactory;
pub use parser::ShapeParser;
pub use types::{ShapeCacheStats, ShapeParsingConfig, ShapeParsingContext, ShapeParsingStats};
pub use validator::{ShapeValidationReport, ShapeValidator, SingleShapeValidationReport};

// Re-export helper functions that were in the original shapes.rs
// These need to be migrated from the original file

use oxirs_core::model::{Literal, NamedNode, Object, Term};
use crate::Result;

/// Format a term for use in SPARQL queries
pub fn format_term_for_sparql(term: &Term) -> String {
    match term {
        Term::NamedNode(nn) => format!("<{}>", nn.as_str()),
        Term::BlankNode(bn) => format!("_:{}", bn.as_str()),
        Term::Literal(lit) => format_literal_for_sparql(lit),
        _ => format!("{}", term),
    }
}

/// Format a literal for use in SPARQL queries
pub fn format_literal_for_sparql(literal: &Literal) -> String {
    let value = literal.value();
    let datatype = literal.datatype();
    let xsd_string = NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap();
    
    if datatype == xsd_string.as_ref() {
        format!("\"{}\"", escape_sparql_string(value))
    } else if let Some(language) = literal.language() {
        format!("\"{}\"@{}", escape_sparql_string(value), language)
    } else {
        format!("\"{}\"^^<{}>", escape_sparql_string(value), datatype.as_str())
    }
}

/// Escape a string for use in SPARQL queries
pub fn escape_sparql_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Convert an RDF object term to a constraint value term
pub fn object_to_term(obj: &Object) -> Result<Term> {
    match obj {
        Object::NamedNode(node) => Ok(Term::NamedNode(node.clone())),
        Object::BlankNode(node) => Ok(Term::BlankNode(node.clone())),
        Object::Literal(literal) => Ok(Term::Literal(literal.clone())),
        Object::Variable(var) => Ok(Term::Variable(var.clone())),
        Object::QuotedTriple(_) => Err(crate::ShaclError::ShapeParsing("QuotedTriple not supported in constraint values".to_string())),
    }
}

// TODO: Additional helper functions from the original shapes.rs file need to be
// migrated here as needed. The original file contains many utility functions
// for RDF parsing, IRI handling, and constraint extraction.