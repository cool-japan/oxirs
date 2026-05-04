//! # OWL 2 DL Reasoner — Profile-aware façade
//!
//! Wraps the existing [`crate::owl_dl::Owl2DLReasoner`] (structural
//! subsumption + ABox materialisation) with the unified
//! [`super::ProfileReasoner`] interface.  Used as the fallback profile
//! when an ontology is too expressive for RL/EL/QL.
//!
//! Per the round-2 plan, `src/owl_dl/` is **unchanged** — this module
//! only constructs and drives an `Owl2DLReasoner`, never extends it.

use std::collections::HashSet;
use thiserror::Error;

use crate::owl_dl::{vocab, DLError, Owl2DLReasoner};

use super::ontology::{Owl2Axiom, Owl2Ontology, ProfileEntailment, ReasoningOutcome};

/// Errors specific to the DL profile façade.
#[derive(Debug, Error)]
pub enum Dl2Error {
    /// The wrapped DL reasoner reported an error.
    #[error("DL reasoning failed: {0}")]
    Reasoning(#[from] DLError),
}

/// Profile-aware DL reasoner.
pub struct ProfileDlReasoner {
    inner: Owl2DLReasoner,
}

impl Default for ProfileDlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileDlReasoner {
    /// Create a new DL reasoner.
    pub fn new() -> Self {
        Self {
            inner: Owl2DLReasoner::new(),
        }
    }

    /// Borrow the underlying reasoner for advanced use.
    pub fn inner(&self) -> &Owl2DLReasoner {
        &self.inner
    }

    /// Mutable access to the underlying reasoner.
    pub fn inner_mut(&mut self) -> &mut Owl2DLReasoner {
        &mut self.inner
    }

    /// Load an ontology into the reasoner.
    pub fn load(&mut self, ontology: &Owl2Ontology) {
        for axiom in &ontology.axioms {
            match axiom {
                Owl2Axiom::SubClassOf { sub, sup } => {
                    self.inner.add_subclass_of(sub, sup);
                }
                Owl2Axiom::EquivalentClasses(a, b) => {
                    self.inner.add_equivalent_classes(a, b);
                }
                Owl2Axiom::IntersectionSubClassOf { parts, sup } => {
                    // Encode `parts[0] ⊓ … ⊑ sup` via add_intersection_of:
                    // any individual in all `parts` is classified into `sup`.
                    self.inner.add_intersection_of(sup, parts.clone());
                }
                Owl2Axiom::SubClassOfSomeValuesFrom { sub, role, filler } => {
                    // Encode A ⊑ ∃R.B as a someValuesFromRestriction
                    // bound to the named class B.  DL backend uses this for
                    // the someValuesFrom forward rule.
                    self.inner
                        .add_some_values_from_restriction(sub, role, filler);
                }
                Owl2Axiom::SomeValuesFromSubClassOf { role, filler, sup } => {
                    // ∃R.B ⊑ C — direct existential restriction → class
                    self.inner
                        .add_some_values_from_restriction(sup, role, filler);
                }
                Owl2Axiom::SubObjectPropertyOf { sub, sup } => {
                    self.inner.add_sub_object_property_of(sub, sup);
                }
                Owl2Axiom::EquivalentObjectProperties(a, b) => {
                    self.inner.add_equivalent_properties(a, b);
                }
                Owl2Axiom::InverseObjectProperties(a, b) => {
                    self.inner.add_inverse_of(a, b);
                }
                Owl2Axiom::PropertyChain { chain, result_role } => {
                    self.inner.add_property_chain(result_role, chain.clone());
                }
                Owl2Axiom::TransitiveProperty(role) => {
                    self.inner.add_transitive_property(role);
                }
                Owl2Axiom::ReflexiveProperty(role) => {
                    self.inner.add_reflexive_property(role);
                }
                Owl2Axiom::ObjectPropertyDomain { property, class } => {
                    self.inner.add_domain(property, class);
                }
                Owl2Axiom::ObjectPropertyRange { property, class } => {
                    self.inner.add_range(property, class);
                }
            }
        }
        for (s, p, o) in &ontology.data_triples {
            if p == "rdf:type" {
                self.inner.assert_type(s, o);
            } else {
                self.inner.add_property_assertion(s, p, o);
            }
        }
    }

    /// Run materialisation and return entailments as [`ProfileEntailment`]s.
    ///
    /// Includes both asserted and inferred triples so that the
    /// conformance corpus (which expects "all known facts") sees a
    /// complete answer set.  The DL reasoner stores its full ABox in a
    /// `pub(crate) HashSet<Triple>` field — `owl2` lives in the same
    /// crate, so we read it directly here without modifying `owl_dl`.
    pub fn entailments(&mut self) -> Result<Vec<ProfileEntailment>, Dl2Error> {
        let _ = self.inner.materialize()?;
        let mut out: HashSet<ProfileEntailment> = HashSet::new();
        for triple in self.inner.abox.iter() {
            let (s, p, o) = triple;
            if p == vocab::RDF_TYPE && !is_owl_meta_class(o) {
                out.insert(ProfileEntailment::Type {
                    individual: s.clone(),
                    class: o.clone(),
                });
            } else if p == vocab::RDFS_SUBCLASS_OF {
                if s != o {
                    out.insert(ProfileEntailment::SubClassOf {
                        sub: s.clone(),
                        sup: o.clone(),
                    });
                }
            } else if !is_meta_predicate(p) {
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
}

/// Predicates that the DL reasoner uses for TBox bookkeeping and that
/// should not surface as `Property` entailments to callers.
fn is_meta_predicate(p: &str) -> bool {
    matches!(
        p,
        "http://www.w3.org/2000/01/rdf-schema#subClassOf"
            | "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"
            | "http://www.w3.org/2000/01/rdf-schema#domain"
            | "http://www.w3.org/2000/01/rdf-schema#range"
            | "http://www.w3.org/2002/07/owl#equivalentClass"
            | "http://www.w3.org/2002/07/owl#equivalentProperty"
            | "http://www.w3.org/2002/07/owl#disjointWith"
            | "http://www.w3.org/2002/07/owl#inverseOf"
    )
}

/// Object classes used by the DL reasoner as property characteristics
/// markers (e.g. `rdf:type owl:TransitiveProperty`).
fn is_owl_meta_class(o: &str) -> bool {
    matches!(
        o,
        "http://www.w3.org/2002/07/owl#TransitiveProperty"
            | "http://www.w3.org/2002/07/owl#SymmetricProperty"
            | "http://www.w3.org/2002/07/owl#AsymmetricProperty"
            | "http://www.w3.org/2002/07/owl#ReflexiveProperty"
            | "http://www.w3.org/2002/07/owl#IrreflexiveProperty"
            | "http://www.w3.org/2002/07/owl#FunctionalProperty"
            | "http://www.w3.org/2002/07/owl#InverseFunctionalProperty"
    )
}

impl ProfileDlReasoner {
    /// Run reasoning and return a [`ReasoningOutcome`].
    pub fn reason(&mut self) -> Result<ReasoningOutcome, Dl2Error> {
        let entailments = self.entailments()?;
        Ok(ReasoningOutcome::new(entailments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owl2::ontology::Owl2OntologyBuilder;

    #[test]
    fn dl_subclass_inference() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .build();

        let mut r = ProfileDlReasoner::new();
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
            "rex should be inferred as Animal via DL closure; got: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn dl_intersection_classification() {
        let ontology = Owl2OntologyBuilder::new()
            .intersection_sub_class(vec!["Doctor".into(), "Surgeon".into()], "DoctorSurgeon")
            .type_of("alice", "Doctor")
            .type_of("alice", "Surgeon")
            .build();

        let mut r = ProfileDlReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("reasoning ok");
        let alice_intersection = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "alice" && class == "DoctorSurgeon"
            )
        });
        assert!(
            alice_intersection,
            "alice should be DoctorSurgeon via intersection; got: {:?}",
            outcome.entailments
        );
    }
}
