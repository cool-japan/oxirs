//! # OWL 2 Profile Dispatcher
//!
//! Unified entry point that exposes every OWL 2 reasoning profile under a
//! single typed call.  Each profile delegates to a specialised backend:
//!
//! - [`OwlProfile::Rl`] → [`crate::owl_rl::Owl2RlReasoner`]
//! - [`OwlProfile::El`] → [`crate::owl_el::Owl2ElReasoner`]
//! - [`OwlProfile::Ql`] → [`crate::owl_ql::QueryRewriter`]
//! - [`OwlProfile::RlEl`] → combined [`rlel_combined::RlElReasoner`]
//! - [`OwlProfile::Dl`] → [`crate::owl_dl::Owl2DLReasoner`]
//!
//! All profile-aware code in `oxirs-rule` lives under `crate::owl2`.
//! Older single-profile modules (`owl_rl`, `owl_el`, `owl_ql`,
//! `owl_dl`, `owl_profiles`) are intentionally left unchanged so
//! existing callers keep working.
//!
//! ## Example
//!
//! ```
//! use oxirs_rule::owl2::{
//!     reason_in_profile, OwlProfile, Owl2OntologyBuilder, ProfileEntailment,
//! };
//!
//! let ontology = Owl2OntologyBuilder::new()
//!     .sub_class_of("Dog", "Mammal")
//!     .sub_class_of("Mammal", "Animal")
//!     .type_of("rex", "Dog")
//!     .build();
//!
//! let outcome = reason_in_profile(OwlProfile::Rl, &ontology)
//!     .expect("RL reasoning");
//! assert!(outcome.contains(&ProfileEntailment::Type {
//!     individual: "rex".into(),
//!     class: "Animal".into(),
//! }));
//! ```

pub mod dl_reasoner;
pub mod el_reasoner;
pub mod ontology;
pub mod profile;
pub mod ql_reasoner;
pub mod rl_reasoner;
pub mod rlel_combined;

pub use dl_reasoner::{Dl2Error, ProfileDlReasoner};
pub use el_reasoner::{El2Error, ProfileElReasoner};
pub use ontology::{
    Owl2Axiom, Owl2Ontology, Owl2OntologyBuilder, ProfileEntailment, ReasoningOutcome,
};
pub use profile::OwlProfile;
pub use ql_reasoner::{ProfileQlReasoner, Ql2Error};
pub use rl_reasoner::{ProfileRlReasoner, Rl2Error};
pub use rlel_combined::{RlElError, RlElReasoner, RlElReport};

use thiserror::Error;

/// Error returned by the dispatcher.
#[derive(Debug, Error)]
pub enum Owl2DispatchError {
    /// The RL backend failed.
    #[error(transparent)]
    Rl(#[from] Rl2Error),
    /// The EL backend failed.
    #[error(transparent)]
    El(#[from] El2Error),
    /// The QL backend failed.
    #[error(transparent)]
    Ql(#[from] Ql2Error),
    /// The combined RL+EL backend failed.
    #[error(transparent)]
    RlEl(#[from] RlElError),
    /// The DL fallback backend failed.
    #[error(transparent)]
    Dl(#[from] Dl2Error),
}

/// Run reasoning under the requested profile and return the resulting
/// [`ReasoningOutcome`].  This is the canonical entry point.
pub fn reason_in_profile(
    profile: OwlProfile,
    ontology: &Owl2Ontology,
) -> Result<ReasoningOutcome, Owl2DispatchError> {
    match profile {
        OwlProfile::Rl => {
            let mut r = ProfileRlReasoner::new();
            r.load(ontology);
            Ok(r.reason()?)
        }
        OwlProfile::El => {
            let mut r = ProfileElReasoner::new();
            r.load(ontology);
            Ok(r.reason()?)
        }
        OwlProfile::Ql => {
            let mut r = ProfileQlReasoner::new();
            r.load(ontology);
            Ok(r.reason(ontology)?)
        }
        OwlProfile::RlEl => {
            let mut r = RlElReasoner::new();
            r.load(ontology);
            Ok(r.reason()?)
        }
        OwlProfile::Dl => {
            let mut r = ProfileDlReasoner::new();
            r.load(ontology);
            Ok(r.reason()?)
        }
    }
}

/// Trait abstracting any profile-aware reasoner so that callers can
/// programmatically swap backends.  `R` is the concrete reasoner's
/// own state type.
pub trait ProfileReasoner {
    /// Profile-specific error type.
    type Err;

    /// Load an ontology into the reasoner.
    fn load_ontology(&mut self, ontology: &Owl2Ontology);

    /// Run reasoning and return the outcome.
    fn reason(&mut self, ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err>;
}

impl ProfileReasoner for ProfileRlReasoner {
    type Err = Rl2Error;
    fn load_ontology(&mut self, ontology: &Owl2Ontology) {
        ProfileRlReasoner::load(self, ontology);
    }
    fn reason(&mut self, _ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err> {
        ProfileRlReasoner::reason(self)
    }
}

impl ProfileReasoner for ProfileElReasoner {
    type Err = El2Error;
    fn load_ontology(&mut self, ontology: &Owl2Ontology) {
        ProfileElReasoner::load(self, ontology);
    }
    fn reason(&mut self, _ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err> {
        ProfileElReasoner::reason(self)
    }
}

impl ProfileReasoner for ProfileQlReasoner {
    type Err = Ql2Error;
    fn load_ontology(&mut self, ontology: &Owl2Ontology) {
        ProfileQlReasoner::load(self, ontology);
    }
    fn reason(&mut self, ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err> {
        ProfileQlReasoner::reason(self, ontology)
    }
}

impl ProfileReasoner for RlElReasoner {
    type Err = RlElError;
    fn load_ontology(&mut self, ontology: &Owl2Ontology) {
        RlElReasoner::load(self, ontology);
    }
    fn reason(&mut self, _ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err> {
        RlElReasoner::reason(self)
    }
}

impl ProfileReasoner for ProfileDlReasoner {
    type Err = Dl2Error;
    fn load_ontology(&mut self, ontology: &Owl2Ontology) {
        ProfileDlReasoner::load(self, ontology);
    }
    fn reason(&mut self, _ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Self::Err> {
        ProfileDlReasoner::reason(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pizza_subset() -> Owl2Ontology {
        // A tiny Pizza-style taxonomy: Margherita ⊑ Pizza ⊑ Food
        // and PizzaWithMozzarella ≡ Pizza ⊓ ∃hasTopping.MozzarellaTopping
        // (the existential restriction is encoded as
        // `SubClassOfSomeValuesFrom + SomeValuesFromSubClassOf`).
        Owl2OntologyBuilder::new()
            .sub_class_of("Margherita", "Pizza")
            .sub_class_of("Pizza", "Food")
            .sub_class_of("MozzarellaTopping", "PizzaTopping")
            .sub_class_some_values("Margherita", "hasTopping", "MozzarellaTopping")
            .some_values_sub_class("hasTopping", "MozzarellaTopping", "PizzaWithMozzarella")
            .type_of("pizza1", "Margherita")
            .build()
    }

    #[test]
    fn dispatches_rl_profile() {
        let ontology = pizza_subset();
        let outcome = reason_in_profile(OwlProfile::Rl, &ontology).expect("rl ok");
        let pizza1_food = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "Food".into(),
        });
        assert!(pizza1_food, "RL should derive pizza1 ⊑ Food");
    }

    #[test]
    fn dispatches_el_profile() {
        let ontology = pizza_subset();
        let outcome = reason_in_profile(OwlProfile::El, &ontology).expect("el ok");
        // EL handles ∃ on both sides — pizza1 should be PizzaWithMozzarella.
        let pizza1_pwm = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "PizzaWithMozzarella".into(),
        });
        assert!(
            pizza1_pwm,
            "EL should derive pizza1 ⊑ PizzaWithMozzarella; got: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn dispatches_ql_profile() {
        let ontology = pizza_subset();
        let outcome = reason_in_profile(OwlProfile::Ql, &ontology).expect("ql ok");
        let pizza1_food = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "Food".into(),
        });
        assert!(pizza1_food, "QL should answer pizza1:Food");
    }

    #[test]
    fn dispatches_rlel_profile() {
        let ontology = pizza_subset();
        let outcome = reason_in_profile(OwlProfile::RlEl, &ontology).expect("rlel ok");
        let pizza1_pwm = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "PizzaWithMozzarella".into(),
        });
        let pizza1_food = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "Food".into(),
        });
        assert!(pizza1_food, "RLEL should derive pizza1 ⊑ Food");
        assert!(
            pizza1_pwm,
            "RLEL should derive pizza1 ⊑ PizzaWithMozzarella via EL leg; got: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn dispatches_dl_profile() {
        let ontology = pizza_subset();
        let outcome = reason_in_profile(OwlProfile::Dl, &ontology).expect("dl ok");
        let pizza1_food = outcome.contains(&ProfileEntailment::Type {
            individual: "pizza1".into(),
            class: "Food".into(),
        });
        assert!(pizza1_food, "DL should derive pizza1 ⊑ Food");
    }
}
