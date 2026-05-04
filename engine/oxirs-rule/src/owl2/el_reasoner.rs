//! # OWL 2 EL Reasoner — Profile-aware façade
//!
//! Wraps the existing [`crate::owl_el::Owl2ElReasoner`] (a complete CR1–CR6
//! completion algorithm — see Baader, Brandt, Lutz "Pushing the EL
//! Envelope", IJCAI 2005) with a [`super::ProfileReasoner`]-shaped API.
//!
//! ## Why a façade?
//!
//! The OWL 2 round-2 plan asks for a unified dispatcher
//! ([`super::reason_in_profile`]).  Each profile must be reachable via a
//! single typed call, while preserving the rich existing reasoner APIs.
//! This module is the EL adapter; it does not duplicate the completion
//! algorithm.

use std::collections::HashSet;
use thiserror::Error;

use crate::owl_el::{ElAxiom, ElClassification, ElConcept, ElError, Owl2ElReasoner};

use super::ontology::{Owl2Ontology, ProfileEntailment, ReasoningOutcome};

/// Errors specific to the EL profile façade.
#[derive(Debug, Error)]
pub enum El2Error {
    /// The wrapped EL reasoner reported an error.
    #[error("EL reasoning failed: {0}")]
    Reasoning(#[from] ElError),
}

/// Profile-aware EL reasoner.
///
/// Holds an [`Owl2ElReasoner`] and reproduces the dispatcher contract.
#[derive(Default)]
pub struct ProfileElReasoner {
    inner: Owl2ElReasoner,
}

impl ProfileElReasoner {
    /// Create a new EL reasoner.
    pub fn new() -> Self {
        Self {
            inner: Owl2ElReasoner::new(),
        }
    }

    /// Borrow the underlying CR1–CR6 reasoner for advanced use.
    pub fn inner(&self) -> &Owl2ElReasoner {
        &self.inner
    }

    /// Mutable access to the underlying reasoner.
    pub fn inner_mut(&mut self) -> &mut Owl2ElReasoner {
        &mut self.inner
    }

    /// Load an ontology into the reasoner — extracts the EL fragment.
    pub fn load(&mut self, ontology: &Owl2Ontology) {
        for axiom in ontology.el_axioms() {
            self.inner.add_axiom(axiom);
        }
    }

    /// Run classification and return the full [`ElClassification`].
    pub fn classify(&self) -> Result<ElClassification, El2Error> {
        Ok(self.inner.classify()?)
    }

    /// Compute and return the entailment set as a sorted list of
    /// `(sub, sup)` subsumption pairs derived from the closure.
    ///
    /// Includes:
    /// - `SubClassOf` entailments from the classified TBox hierarchy
    /// - `Type` entailments from individual classifications
    /// - `Property` entailments from role successors derived by CR3/CR5/CR6
    ///   (e.g. property chains, transitive roles, sub-role propagation)
    pub fn entailments(&self) -> Result<Vec<ProfileEntailment>, El2Error> {
        let cls = self.inner.classify()?;
        let mut out: HashSet<ProfileEntailment> = HashSet::new();

        for (sub, supers) in &cls.subsumption_hierarchy {
            for sup in supers {
                if sub == sup {
                    continue;
                }
                out.insert(ProfileEntailment::SubClassOf {
                    sub: sub.clone(),
                    sup: sup.clone(),
                });
            }
        }
        for (individual, types) in &cls.individual_types {
            if individual.starts_with("__wit_") {
                continue;
            }
            for ty in types {
                if ty == "owl:Thing" || ty.starts_with("__wit_") {
                    continue;
                }
                out.insert(ProfileEntailment::Type {
                    individual: individual.clone(),
                    class: ty.clone(),
                });
            }
        }
        for ((subject, role), successors) in &cls.role_successors {
            if subject.starts_with("__wit_") {
                continue;
            }
            for object in successors {
                if object.starts_with("__wit_") {
                    continue;
                }
                out.insert(ProfileEntailment::Property {
                    subject: subject.clone(),
                    property: role.clone(),
                    object: object.clone(),
                });
            }
        }

        let mut sorted: Vec<ProfileEntailment> = out.into_iter().collect();
        sorted.sort();
        Ok(sorted)
    }

    /// Convenience: run classification and return a [`ReasoningOutcome`].
    pub fn reason(&self) -> Result<ReasoningOutcome, El2Error> {
        let entailments = self.entailments()?;
        Ok(ReasoningOutcome::new(entailments))
    }

    /// Add a single EL axiom directly.
    pub fn add_axiom(&mut self, axiom: ElAxiom) {
        self.inner.add_axiom(axiom);
    }

    /// Add a simple subclass axiom.
    pub fn add_subclass_of(&mut self, sub: &str, sup: &str) {
        self.inner.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::named(sub),
            sup: ElConcept::named(sup),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owl2::ontology::Owl2OntologyBuilder;

    #[test]
    fn classifies_simple_taxonomy_via_facade() {
        let mut r = ProfileElReasoner::new();
        r.add_subclass_of("Cat", "Mammal");
        r.add_subclass_of("Mammal", "Animal");

        let cls = r.classify().expect("classify ok");
        assert!(cls.is_subclass_of("Cat", "Animal"));
    }

    #[test]
    fn loads_from_ontology_and_returns_entailments() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .build();

        let mut r = ProfileElReasoner::new();
        r.load(&ontology);

        let entailments = r.entailments().expect("entailments ok");
        let has_transitive_subclass = entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::SubClassOf { sub, sup } if sub == "Dog" && sup == "Animal"
            )
        });
        assert!(
            has_transitive_subclass,
            "Expected Dog ⊑ Animal in entailments, got: {entailments:?}"
        );
    }

    #[test]
    fn intersection_axioms_work_through_facade() {
        let ontology = Owl2OntologyBuilder::new()
            .intersection_sub_class(
                vec!["Doctor".to_string(), "PediatricSpecialist".to_string()],
                "Pediatrician",
            )
            .type_of("alice", "Doctor")
            .type_of("alice", "PediatricSpecialist")
            .build();

        let mut r = ProfileElReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("reason ok");

        let alice_pediatrician = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "alice" && class == "Pediatrician"
            )
        });
        assert!(alice_pediatrician, "alice should be Pediatrician");
    }
}
