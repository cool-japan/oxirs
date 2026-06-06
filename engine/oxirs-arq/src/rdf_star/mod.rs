//! SPARQL-star (RDF-star) Completeness Module
//!
//! Implements the complete SPARQL 1.2 / SPARQL-star specification for quoted triples,
//! annotation queries, pattern matching, and CONSTRUCT support.
//!
//! # Overview
//!
//! RDF-star (W3C spec) allows triples as subjects or objects — called *quoted triples*:
//!
//! ```text
//! <<  <http://s>  <http://p>  <http://o>  >>  <http://certainty>  "0.9"
//! ```
//!
//! SPARQL-star introduces corresponding query syntax:
//!
//! ```sparql
//! SELECT ?s ?p ?o ?c WHERE {
//!     << ?s ?p ?o >> <http://certainty> ?c .
//! }
//! ```
//!
//! # References
//! - <https://www.w3.org/2021/12/rdf-star.html>
//! - <https://w3c.github.io/sparql-star/>
//!
//! This module is a thin facade. The implementation is split across:
//! - [`rdf_star_terms`] — the term types [`QuotedTriple`], [`StarSubject`],
//!   [`StarPredicate`], [`StarObject`], [`Annotation`] and their conversions.
//! - [`rdf_star_operator`] — the [`StarPattern`] annotation pattern and the
//!   [`StarOperator`] query-plan operators.
//! - [`rdf_star_store`] — the in-memory [`RdfStarStore`] and [`AnnotatedTriple`].
//! - [`rdf_star_binding`] — [`StarBinding`], pattern binding, CONSTRUCT
//!   instantiation, and the [`sparql_star_builtins`] functions.

mod rdf_star_binding;
mod rdf_star_operator;
mod rdf_star_store;
mod rdf_star_terms;

#[cfg(test)]
mod tests;

pub use rdf_star_binding::{
    bind_pattern, instantiate_quoted_triple, sparql_star_builtins, StarBinding,
};
pub use rdf_star_operator::{StarOperator, StarPattern};
pub use rdf_star_store::{AnnotatedTriple, RdfStarStore};
pub use rdf_star_terms::{Annotation, QuotedTriple, StarObject, StarPredicate, StarSubject};
