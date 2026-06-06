//! # OWL 2 RL Profile Reasoner
//!
//! Implements the OWL 2 RL (Rule Language) profile using forward-chaining rules.
//! OWL 2 RL is designed for scalable reasoning using production rule systems.
//!
//! ## Complexity
//! Polynomial time in the size of the data (ABox).
//!
//! ## Key Features
//! - Full W3C OWL 2 RL rule set (Table 4-8 of the spec)
//! - Forward-chaining materialization to fixpoint
//! - Triple pattern matching with variable unification
//! - Conflict detection for owl:disjointWith
//!
//! ## Reference
//! <https://www.w3.org/TR/owl2-profiles/#OWL_2_RL>

// Re-export all public items from the sibling modules
pub use crate::owl_rl_reasoner::Owl2RlReasoner;
pub use crate::owl_rl_rules::{
    vocab, Bindings, InferenceReport, Owl2RlRule, PatternElem, RlError, Triple, TriplePattern,
    OWL_ALL_VALUES_FROM, OWL_ASYMMETRIC_PROPERTY, OWL_DISJOINT_WITH, OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY, OWL_FUNCTIONAL_PROPERTY, OWL_HAS_VALUE, OWL_INTERSECTION_OF,
    OWL_INVERSE_OF, OWL_INV_FUNCTIONAL_PROPERTY, OWL_IRREFLEXIVE_PROPERTY, OWL_NOTHING,
    OWL_ON_PROPERTY, OWL_SAME_AS, OWL_SOME_VALUES_FROM, OWL_SYMMETRIC_PROPERTY, OWL_THING,
    OWL_TRANSITIVE_PROPERTY, OWL_UNION_OF, RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF,
    RDFS_SUBPROPERTY_OF, RDF_FIRST, RDF_REST, RDF_TYPE,
};
