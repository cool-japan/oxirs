//! OWL 2 QL profile compliance checker.
//!
//! Verifies that OWL axioms stay within the syntactic restrictions of the OWL 2 QL profile,
//! enabling efficient conjunctive query answering.
//!
//! This module is independent of [`crate::owl_ql`] (the Query Rewriter / PerfectRef algorithm).

pub mod checker;
pub mod profile;

pub use checker::{Owl2QlProfileChecker, ProfileAxiomReport, ProfileReport};
pub use profile::{ClassExpr, OntologyAxiom};

#[derive(Debug, Clone, thiserror::Error)]
pub enum Owl2QlProfileError {
    #[error("Invalid IRI: {0}")]
    InvalidIri(String),
    #[error("Profile violation: {0}")]
    Violation(String),
}

#[cfg(test)]
mod tests;
