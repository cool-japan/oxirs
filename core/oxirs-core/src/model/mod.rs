//! Core RDF data model types and implementations
//!
//! **Stability**: ✅ **Stable** - These APIs are production-ready and will maintain backward compatibility.
//!
//! This module provides the fundamental RDF concepts following the [RDF 1.2 specification](https://www.w3.org/TR/rdf12-concepts/),
//! with performance optimizations and OxiRS-specific enhancements.
//!
//! ## Overview
//!
//! The RDF (Resource Description Framework) data model represents information as triples:
//! **Subject** - **Predicate** - **Object**. This module provides Rust types for all RDF terms
//! and structures, enabling type-safe manipulation of RDF data.
//!
//! ## Core Types
//!
//! ### Terms (Basic Building Blocks)
//!
//! - **[`NamedNode`]** - An IRI (Internationalized Resource Identifier), the primary way to identify resources
//! - **[`BlankNode`]** - An anonymous node without a global identifier
//! - **[`Literal`]** - A data value with optional language tag or datatype
//! - **[`Variable`]** - A SPARQL query variable (used in query patterns)
//!
//! ### Composite Types
//!
//! - **[`Triple`]** - A statement with subject, predicate, and object
//! - **[`Quad`]** - A triple with an optional named graph
//! - **[`GraphName`]** - Identifies the graph containing a quad (named or default)
//!
//! ### Union Types
//!
//! - **[`Subject`]** - Can be a NamedNode, BlankNode, or Variable
//! - **[`Predicate`]** - Can be a NamedNode or Variable
//! - **[`Object`]** - Can be a NamedNode, BlankNode, Literal, or Variable
//! - **[`Term`]** - Any RDF term (superset of all above)
//!
//! ## Examples
//!
//! ### Creating RDF Terms
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a Named Node (IRI)
//! let alice = NamedNode::new("http://example.org/alice")?;
//!
//! // Create a Blank Node
//! let blank = BlankNode::new("b1")?;
//!
//! // Create Literals
//! let plain_literal = Literal::new("Hello, World!");
//! let typed_literal = Literal::new_typed("42", NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?);
//! let lang_literal = Literal::new_lang("Bonjour", "fr")?;
//!
//! // Create a Variable (for SPARQL patterns)
//! let var = Variable::new("name")?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Creating Triples
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create terms
//! let alice = NamedNode::new("http://example.org/alice")?;
//! let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
//! let bob = NamedNode::new("http://example.org/bob")?;
//!
//! // Create a triple: "Alice knows Bob"
//! let triple = Triple::new(alice, knows, bob);
//!
//! // Access components
//! println!("Subject: {}", triple.subject());
//! println!("Predicate: {}", triple.predicate());
//! println!("Object: {}", triple.object());
//! # Ok(())
//! # }
//! ```
//!
//! ### Creating Quads (Named Graphs)
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create terms
//! let alice = NamedNode::new("http://example.org/alice")?;
//! let name = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
//! let alice_lit = Literal::new("Alice");
//!
//! // Create a named graph
//! let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1")?);
//!
//! // Create a quad in the named graph
//! let quad = Quad::new(alice, name, alice_lit, graph);
//!
//! println!("Quad in graph: {}", quad.graph_name());
//! # Ok(())
//! # }
//! ```
//!
//! ### Working with Literals
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Plain literal
//! let plain = Literal::new("Hello");
//! assert_eq!(plain.value(), "Hello");
//!
//! // Language-tagged literal (for internationalization)
//! let french = Literal::new_lang("Bonjour", "fr")?;
//! assert_eq!(french.value(), "Bonjour");
//! assert_eq!(french.language(), Some("fr"));
//!
//! // Typed literal (with XSD datatype)
//! let number = Literal::new_typed(
//!     "42",
//!     NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?
//! );
//! assert_eq!(number.value(), "42");
//! # Ok(())
//! # }
//! ```
//!
//! ### Pattern Matching with Variables
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a query pattern: "?person foaf:name ?name"
//! let person_var = Variable::new("person")?;
//! let name_var = Variable::new("name")?;
//! let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
//!
//! let pattern = Triple::new(person_var, name_pred, name_var);
//!
//! // This pattern can be used in SPARQL queries
//! println!("Pattern: {:?}", pattern);
//! # Ok(())
//! # }
//! ```
//!
//! ## Type Conversions
//!
//! All term types implement `Into<>` for their union types:
//!
//! ```rust
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let node = NamedNode::new("http://example.org/resource")?;
//!
//! // Automatic conversion to union types
//! let subject: Subject = node.clone().into();
//! let predicate: Predicate = node.clone().into();
//! let object: Object = node.clone().into();
//! let term: Term = node.into();
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Optimizations
//!
//! This module includes several performance optimizations:
//!
//! - **String interning** via [`optimized_terms`] for reduced memory usage
//! - **Zero-copy operations** where possible
//! - **Efficient hashing** for use in hash-based collections
//! - **Ordered comparisons** for BTree-based indexes
//!
//! ## RDF 1.2 Features
//!
//! - **RDF-star support** via [`star`] module for quoted triples
//! - **Generalized RDF datasets** with named graph support
//! - **Extended literal datatypes** including language-tagged strings
//!
//! ## Related Modules
//!
//! - [`crate::parser`] - Parse RDF from various formats
//! - [`crate::serializer`] - Serialize RDF to various formats
//! - [`crate::rdf_store`] - Store and query RDF data
//! - [`crate::query`] - SPARQL query execution

pub mod dataset;
pub mod graph;
pub mod iri;
pub mod literal;
pub mod node;
pub mod optimized_terms; // Oxigraph-inspired performance optimizations
pub mod pattern;
pub mod quad;
pub mod star;
pub mod term;
pub mod triple;

// Re-export core types
pub use dataset::*;
pub use graph::*;
pub use iri::*;
pub use literal::*;
pub use node::*;
pub use optimized_terms::*; // Oxigraph-inspired optimizations
pub use pattern::*;
pub use quad::*;
pub use star::*;
pub use term::*;
pub use triple::*;

// Explicit re-exports of union types for external modules
pub use quad::GraphName;
pub use term::{Object, Predicate, Subject, Term, Variable};

/// A trait for all RDF terms
pub trait RdfTerm {
    /// Returns the string representation of this term
    fn as_str(&self) -> &str;

    /// Returns true if this is a named node (IRI)
    fn is_named_node(&self) -> bool {
        false
    }

    /// Returns true if this is a blank node
    fn is_blank_node(&self) -> bool {
        false
    }

    /// Returns true if this is a literal
    fn is_literal(&self) -> bool {
        false
    }

    /// Returns true if this is a variable
    fn is_variable(&self) -> bool {
        false
    }

    /// Returns true if this is a quoted triple (RDF-star)
    fn is_quoted_triple(&self) -> bool {
        false
    }
}

/// A trait for terms that can be used as subjects in RDF triples
pub trait SubjectTerm: RdfTerm {}

/// A trait for terms that can be used as predicates in RDF triples
pub trait PredicateTerm: RdfTerm {}

/// A trait for terms that can be used as objects in RDF triples
pub trait ObjectTerm: RdfTerm {}

/// A trait for terms that can be used in graph names
pub trait GraphNameTerm: RdfTerm {}
