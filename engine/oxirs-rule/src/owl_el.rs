//! # OWL 2 EL Profile Reasoner
//!
//! Implements the OWL 2 EL (Existential Language) profile using the EL++ completion
//! algorithm (consequence-based reasoning). OWL 2 EL is designed for very large ontologies
//! like SNOMED CT (>300k concepts).
//!
//! ## Complexity
//! PTime in the size of the TBox (tractable for millions of axioms).
//!
//! ## Supported Constructs
//! - SubClassOf with atomic concepts or ObjectIntersectionOf on left
//! - ObjectSomeValuesFrom on either side
//! - EquivalentClasses
//! - ObjectPropertyChain (role compositions)
//! - SubObjectPropertyOf, TransitiveObjectProperty
//! - ConceptAssertion, RoleAssertion
//!
//! ## Reference
//! Baader, Brandt, Lutz: "Pushing the EL Envelope" (IJCAI 2005).
//! <https://www.w3.org/TR/owl2-profiles/#OWL_2_EL>

// Re-export all public items from the split sibling crates.
pub use crate::owl_el_axioms::*;
pub use crate::owl_el_reasoner::Owl2ElReasoner;
