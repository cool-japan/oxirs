//! # Probabilistic Datalog (ProbLog)
//!
//! This module implements ProbLog, a probabilistic extension of Datalog that allows
//! reasoning with uncertainty by attaching probabilities to facts and rules.
//!
//! ## Features
//!
//! - **Probabilistic Facts** - Facts with associated probabilities
//! - **Probabilistic Rules** - Rules that propagate probabilities
//! - **Query Evaluation** - Compute probability of queries
//! - **Independence Assumptions** - Handle independent and dependent events
//! - **Explanation Trees** - Track provenance of probabilistic derivations
//!
//! ## Probability Semantics
//!
//! 1. **Probabilistic Facts**: `p::fact(a).` means fact(a) is true with probability p
//! 2. **Conjunction**: P(A ∧ B) = P(A) × P(B) (assuming independence)
//! 3. **Disjunction**: P(A ∨ B) = P(A) + P(B) - P(A) × P(B)
//! 4. **Negation**: P(¬A) = 1 - P(A)
//!
//! The implementation is split across focused sub-modules:
//! - [`crate::problog_types`] — all types, structs, enums
//! - [`crate::problog_inference`] — WMC/SDD/BDD inference algorithms
//! - [`crate::problog_solver`] — top-level solver and ProbLogEngine

// Re-export everything from the sub-modules for backward compatibility
pub use crate::problog_inference::{
    apply_substitution_to_atom, apply_substitution_to_body, apply_substitution_to_term,
    find_all_bindings, materialize as materialize_facts, occurs_in_term, unify_atoms, unify_terms,
};
pub use crate::problog_solver::ProbLogEngine;
pub use crate::problog_types::{
    DerivationTree, EvaluationStrategy, ProbLogStats, ProbabilisticFact, ProbabilisticRule,
};
