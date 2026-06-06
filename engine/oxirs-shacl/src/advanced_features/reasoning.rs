//! SHACL Advanced Features - Reasoning Integration
//!
//! Integration with reasoning engines for RDFS, OWL, and custom entailment regimes.
//! Enables reasoning-aware SHACL validation with support for:
//! - RDFS entailment (subclass, subproperty, domain, range)
//! - OWL 2 profiles (RL, QL, EL)
//! - Custom entailment regimes
//! - Closed-world assumption (CWA) support
//! - Negation as failure
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::advanced_features::reasoning_types`] — entailment regimes,
//!   reasoning config, [`InferredTriple`], validation results / statistics,
//!   and the closed-world / negation-as-failure helpers.
//! - [`crate::advanced_features::reasoning_validator`] — the
//!   [`ReasoningValidator`], implementing RDFS and OWL 2 RL entailment.
//! - [`crate::advanced_features::reasoning_probabilistic`] — the
//!   [`ProbabilisticValidator`] for Bayesian uncertainty quantification.

pub use super::reasoning_probabilistic::{
    EvidenceData, ProbabilisticConfig, ProbabilisticStats, ProbabilisticValidationResult,
    ProbabilisticValidator,
};
pub use super::reasoning_types::{
    ClosedWorldValidator, CustomReasoning, EntailmentRegime, InferredTriple, NafGoal,
    NegationAsFailure, ReasoningConfig, ReasoningStats, ReasoningValidationResult,
};
pub use super::reasoning_validator::ReasoningValidator;
