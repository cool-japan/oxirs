//! # SKOS Inference Rules
//!
//! SKOS entailment rules per W3C SKOS Reference §8 and hierarchy traversal
//! utilities (broader/narrower transitive closures and label search).
//!
//! Re-exports [`SkosReasoner`] from the internal `skos_reasoner` module.

pub use crate::skos_reasoner::SkosReasoner;
