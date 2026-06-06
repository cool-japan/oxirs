//! W3C RDF Dataset Normalization Algorithm — URDNA2015 / RDNA 2015.
//!
//! This module provides a complete, spec-faithful implementation of the W3C RDF
//! Canonicalization algorithm (URDNA2015) for deterministic blank node naming in
//! RDF datasets.  Canonical RDF is a prerequisite for cryptographic signing of
//! RDF documents and is used in:
//!
//! - W3C Verifiable Credentials
//! - W3C Data Integrity Proofs
//! - JSON-LD Signatures
//! - RDF Merkle trees
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_core::canon::{canonicalize, QuadTerm, RdfQuad};
//!
//! // Build a small RDF dataset
//! let quads = vec![
//!     RdfQuad::new(
//!         QuadTerm::blank("b0"),
//!         QuadTerm::iri("http://schema.org/name"),
//!         QuadTerm::string_literal("Alice"),
//!     ),
//! ];
//!
//! let canonical = canonicalize(&quads);
//! // "_:c14n0 <http://schema.org/name> "Alice"^^<http://www.w3.org/2001/XMLSchema#string> ."
//! assert!(canonical.starts_with("_:c14n0"));
//! ```
//!
//! ## Reference
//!
//! - W3C RDF Canonicalization 1.0: <https://www.w3.org/TR/rdf-canon/>

// Sub-modules
pub mod algorithm;
pub mod hash;
pub mod nquads;
pub mod types;

// Re-exports — public API surface
pub use algorithm::{canonicalize, Canonicalizer, IdentifierIssuer};
pub use hash::{sha256_hex, sha256_hex_bytes};
pub use nquads::{escape_literal, quad_to_nquad, term_to_nquad};
pub use types::{QuadTerm, RdfQuad};
