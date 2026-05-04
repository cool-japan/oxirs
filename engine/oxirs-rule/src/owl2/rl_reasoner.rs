//! # OWL 2 RL Reasoner — Profile-aware façade
//!
//! Wraps the existing [`crate::owl_rl::Owl2RlReasoner`] (full W3C OWL 2 RL
//! rule set, Tables 4–9) so it can be invoked through the unified
//! [`super::reason_in_profile`] dispatcher with the shared
//! [`Owl2Ontology`] input shape.

use std::collections::HashSet;
use thiserror::Error;

use crate::owl_rl::{InferenceReport, Owl2RlReasoner, RlError, RDFS_SUBCLASS_OF, RDF_TYPE};

use super::ontology::{Owl2Ontology, ProfileEntailment, ReasoningOutcome};

/// Errors specific to the RL profile façade.
#[derive(Debug, Error)]
pub enum Rl2Error {
    /// The wrapped RL reasoner reported an error.
    #[error("RL reasoning failed: {0}")]
    Reasoning(#[from] RlError),
}

/// Profile-aware RL reasoner.
pub struct ProfileRlReasoner {
    inner: Owl2RlReasoner,
}

impl Default for ProfileRlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileRlReasoner {
    /// Create a new RL reasoner.
    pub fn new() -> Self {
        Self {
            inner: Owl2RlReasoner::new(),
        }
    }

    /// Borrow the underlying reasoner for advanced use.
    pub fn inner(&self) -> &Owl2RlReasoner {
        &self.inner
    }

    /// Mutable access to the underlying reasoner.
    pub fn inner_mut(&mut self) -> &mut Owl2RlReasoner {
        &mut self.inner
    }

    /// Load axioms + data from an ontology — translates to RL-shaped
    /// triples and adds them to the reasoner.
    pub fn load(&mut self, ontology: &Owl2Ontology) {
        for (s, p, o) in ontology.rl_triples() {
            self.inner.add_axiom(&s, &p, &o);
        }
    }

    /// Run forward-chaining materialisation to fixpoint.
    pub fn materialize(&mut self) -> Result<InferenceReport, Rl2Error> {
        Ok(self.inner.materialize()?)
    }

    /// Materialise the closure and return the entailments as
    /// [`ProfileEntailment`]s.
    pub fn entailments(&mut self) -> Result<Vec<ProfileEntailment>, Rl2Error> {
        let _ = self.inner.materialize()?;
        let mut out: HashSet<ProfileEntailment> = HashSet::new();

        for (s, p, o) in self.inner.all_triples() {
            if p == RDF_TYPE {
                out.insert(ProfileEntailment::Type {
                    individual: s.clone(),
                    class: o.clone(),
                });
            } else if p == RDFS_SUBCLASS_OF {
                if s != o {
                    out.insert(ProfileEntailment::SubClassOf {
                        sub: s.clone(),
                        sup: o.clone(),
                    });
                }
            } else {
                out.insert(ProfileEntailment::Property {
                    subject: s.clone(),
                    property: p.clone(),
                    object: o.clone(),
                });
            }
        }

        let mut sorted: Vec<ProfileEntailment> = out.into_iter().collect();
        sorted.sort();
        Ok(sorted)
    }

    /// Run reasoning and return a [`ReasoningOutcome`].
    pub fn reason(&mut self) -> Result<ReasoningOutcome, Rl2Error> {
        let entailments = self.entailments()?;
        Ok(ReasoningOutcome::new(entailments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owl2::ontology::Owl2OntologyBuilder;

    #[test]
    fn rl_simple_subclass_inference() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .build();

        let mut r = ProfileRlReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("reasoning ok");

        let has_rex_animal = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "rex" && class == "Animal"
            )
        });
        assert!(
            has_rex_animal,
            "rex should be inferred as Animal via RL transitivity; got: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn rl_inverse_property_is_inferred() {
        let ontology = Owl2OntologyBuilder::new()
            .inverse_of("hasParent", "hasChild")
            .property("alice", "hasChild", "bob")
            .build();

        let mut r = ProfileRlReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("reasoning ok");

        let has_inv = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Property { subject, property, object }
                    if subject == "bob" && property == "hasParent" && object == "alice"
            )
        });
        assert!(
            has_inv,
            "Expected bob hasParent alice via inverseOf; got: {:?}",
            outcome.entailments
        );
    }
}
