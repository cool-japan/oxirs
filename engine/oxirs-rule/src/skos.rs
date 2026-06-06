//! # SKOS Entailment Rules
//!
//! Implements SKOS (Simple Knowledge Organization System) entailment rules
//! as defined in the W3C SKOS Reference specification.
//!
//! ## Reference
//! - <https://www.w3.org/TR/skos-reference/>
//! - <https://www.w3.org/TR/skos-primer/>
//!
//! ## Rules Implemented
//!
//! | ID | Rule | Description |
//! |----|------|-------------|
//! | S1 | broaderTransitive transitivity | broaderTransitive chain propagation |
//! | S2 | narrowerTransitive transitivity | narrowerTransitive chain propagation |
//! | S3 | broader/narrower symmetry | broader implies narrower and vice-versa |
//! | S4 | related symmetry | related is symmetric |
//! | S5 | topConceptOf/hasTopConcept symmetry | inverse relationship |
//! | S6 | exactMatch reflexive, symmetric, transitive | equivalence-like mapping |
//! | S7 | inScheme via topConceptOf | topConceptOf implies inScheme |
//! | S8 | broader implies broaderTransitive | broader/narrower lift to transitive closure |
//! | S9 | closeMatch symmetry | closeMatch is symmetric |
//! | S10 | broader/narrower from match relations | broadMatch implies broadMatch inverse |
//!
//! ## Module Organization
//!
//! This module is a thin facade that re-exports the SKOS implementation, which
//! is split across focused sibling modules to keep each file maintainable:
//!
//! - [`crate::skos_types`] — vocabulary constants, error types, the in-memory
//!   [`Graph`] triple store and the [`ConceptNode`] / [`ConceptTree`] types.
//! - [`crate::skos_reasoner`] — the [`SkosReasoner`] entailment rules and
//!   hierarchy-traversal utilities.
//! - [`crate::skos_mappings`] — the [`ConceptSchemeAnalyzer`] for concept-scheme
//!   membership and concept-tree construction.
//! - [`crate::skos_validation`] — the [`SkosValidator`] integrity checks and the
//!   [`ValidationReport`] / [`ValidationFinding`] result types.
//! - [`crate::skos_inference`] — convenience re-export of [`SkosReasoner`]
//!   (already surfaced here through [`crate::skos_reasoner`]).

pub use crate::skos_mappings::*;
pub use crate::skos_reasoner::*;
pub use crate::skos_types::*;
pub use crate::skos_validation::*;
