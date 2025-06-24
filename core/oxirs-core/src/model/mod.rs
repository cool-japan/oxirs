//! Core RDF data model types and implementations
//!
//! This module provides the fundamental RDF concepts following the RDF 1.2 specification,
//! ported from Oxigraph's oxrdf with OxiRS-specific enhancements.

pub mod term;
pub mod triple;
pub mod quad;
pub mod graph;
pub mod dataset;
pub mod literal;
pub mod iri;

// Re-export core types
pub use term::*;
pub use triple::*;
pub use quad::*;
pub use graph::*;
pub use dataset::*;
pub use literal::*;
pub use iri::*;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// A trait for all RDF terms
pub trait RdfTerm {
    /// Returns the string representation of this term
    fn as_str(&self) -> &str;
    
    /// Returns true if this is a named node (IRI)
    fn is_named_node(&self) -> bool { false }
    
    /// Returns true if this is a blank node
    fn is_blank_node(&self) -> bool { false }
    
    /// Returns true if this is a literal
    fn is_literal(&self) -> bool { false }
    
    /// Returns true if this is a variable
    fn is_variable(&self) -> bool { false }
}

/// A trait for terms that can be used as subjects in RDF triples
pub trait SubjectTerm: RdfTerm {}

/// A trait for terms that can be used as predicates in RDF triples
pub trait PredicateTerm: RdfTerm {}

/// A trait for terms that can be used as objects in RDF triples
pub trait ObjectTerm: RdfTerm {}

/// A trait for terms that can be used in graph names
pub trait GraphNameTerm: RdfTerm {}