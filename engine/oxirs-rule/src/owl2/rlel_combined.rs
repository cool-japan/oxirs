//! # OWL 2 RL ⊕ EL Combined Reasoner
//!
//! Hybrid reasoner that closes a TBox under both the OWL 2 RL rule set
//! and the OWL 2 EL completion algorithm, alternating until a fixpoint
//! is reached.
//!
//! ## Algorithm
//!
//! 1. **Seed**: project the input ontology to RL triples and to EL
//!    axioms (see [`Owl2Ontology::rl_triples`] and
//!    [`Owl2Ontology::el_axioms`]).
//! 2. **RL pass**: run [`crate::owl_rl::Owl2RlReasoner`] forward-chaining
//!    materialisation to fixpoint over the RL triples.
//! 3. **Extract**: harvest from the RL closure every triple of the
//!    forms `?x rdf:type ?C`, `?C rdfs:subClassOf ?D`, `?P rdf:type
//!    owl:TransitiveProperty`, and replay them as additional EL
//!    axioms.  Cardinality, hasValue and complement triples produced
//!    by RL are not in the EL fragment and are dropped.
//! 4. **EL pass**: run [`crate::owl_el::Owl2ElReasoner::classify`].
//! 5. **Extract**: convert every new subsumption pair `(B, C)` with
//!    `B ⊑ C` from the EL classification into `B rdfs:subClassOf C`
//!    triples and feed them back to the RL reasoner.  Also propagate
//!    individual type assertions discovered by EL (e.g. via
//!    intersection/existential rules) back to the RL ABox.
//! 6. **Loop**: repeat steps 2–5 until neither pass derives a new
//!    triple/axiom.  A monotone counter guarantees termination because
//!    the universe of named classes/individuals is finite.
//!
//! ## Why this is useful
//!
//! Hybrid TBoxes that mix property chains (EL-only) with cardinality
//! restrictions (RL-only) cannot be classified by either profile in
//! isolation.  The combined closure is sound because both rule
//! systems are monotone, and the alternation never invalidates an
//! earlier inference.

use std::collections::HashSet;
use std::time::Instant;
use thiserror::Error;

use crate::owl_el::{ElAxiom, ElConcept, Owl2ElReasoner};
use crate::owl_rl::{Owl2RlReasoner, RDFS_SUBCLASS_OF, RDF_TYPE};

use super::el_reasoner::El2Error;
use super::ontology::{Owl2Ontology, ProfileEntailment, ReasoningOutcome};
use super::rl_reasoner::Rl2Error;

/// Errors specific to the combined RL+EL reasoner.
#[derive(Debug, Error)]
pub enum RlElError {
    /// Underlying RL reasoner reported an error.
    #[error("RL pass failed: {0}")]
    Rl(#[from] Rl2Error),

    /// Underlying EL reasoner reported an error.
    #[error("EL pass failed: {0}")]
    El(#[from] El2Error),

    /// Hybrid loop exceeded its safety bound.
    #[error("RL+EL loop exceeded max alternations ({0})")]
    MaxAlternationsExceeded(usize),
}

/// Combined RL + EL reasoner.
pub struct RlElReasoner {
    rl: Owl2RlReasoner,
    el: Owl2ElReasoner,
    /// Maximum RL↔EL alternations (default 16, enough for any
    /// realistic ontology because each alternation is monotone and the
    /// universe of named classes is bounded).
    max_alternations: usize,
}

/// Per-run statistics for an RL+EL closure.
#[derive(Debug, Clone, Default)]
pub struct RlElReport {
    /// How many RL↔EL alternations actually ran.
    pub alternations: usize,
    /// Triples added to the RL closure across all passes.
    pub rl_triples_added: usize,
    /// EL axioms forwarded from RL output across all passes.
    pub el_axioms_added: usize,
    /// Wall-clock duration.
    pub duration: std::time::Duration,
}

impl Default for RlElReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl RlElReasoner {
    /// Create a new combined reasoner.
    pub fn new() -> Self {
        Self {
            rl: Owl2RlReasoner::new(),
            el: Owl2ElReasoner::new(),
            max_alternations: 16,
        }
    }

    /// Override the RL↔EL alternation safety bound.
    pub fn with_max_alternations(mut self, n: usize) -> Self {
        self.max_alternations = n;
        self
    }

    /// Load an ontology — feeds both backends with their respective
    /// projections.
    pub fn load(&mut self, ontology: &Owl2Ontology) {
        for (s, p, o) in ontology.rl_triples() {
            self.rl.add_axiom(&s, &p, &o);
        }
        for axiom in ontology.el_axioms() {
            self.el.add_axiom(axiom);
        }
    }

    /// Run the combined RL ⊕ EL fixpoint.
    pub fn close(&mut self) -> Result<RlElReport, RlElError> {
        let start = Instant::now();
        let mut alternations = 0usize;
        let mut rl_added_total = 0usize;
        let mut el_added_total = 0usize;

        // Track which EL facts we have already asserted — guarantees the
        // EL axiom set grows monotonically.
        let mut el_known_subclass: HashSet<(String, String)> = HashSet::new();
        let mut el_known_type: HashSet<(String, String)> = HashSet::new();

        // Track the size of the RL closure between rounds.  RL's
        // `materialize()` rebuilds the closure from scratch each call,
        // so its `new_triples_count` cannot be used as a delta — we
        // compute the delta at the dispatcher level.
        let mut prev_rl_closure_size = 0usize;

        loop {
            if alternations >= self.max_alternations {
                return Err(RlElError::MaxAlternationsExceeded(self.max_alternations));
            }
            alternations += 1;

            // ── RL pass ───────────────────────────────────────────────
            let _rl_report = self
                .rl
                .materialize()
                .map_err(|e| RlElError::Rl(Rl2Error::from(e)))?;
            let new_rl_closure_size = self.rl.all_triples().len();
            let rl_grew = new_rl_closure_size > prev_rl_closure_size;
            prev_rl_closure_size = new_rl_closure_size;

            // Harvest EL-shape facts from the RL closure.
            let mut el_round_added = 0usize;
            for (s, p, o) in self.rl.all_triples() {
                if p == RDFS_SUBCLASS_OF
                    && s != o
                    && el_known_subclass.insert((s.clone(), o.clone()))
                {
                    self.el.add_axiom(ElAxiom::SubConceptOf {
                        sub: ElConcept::named(s.clone()),
                        sup: ElConcept::named(o.clone()),
                    });
                    el_round_added += 1;
                } else if p == RDF_TYPE
                    && o != "http://www.w3.org/2002/07/owl#Class"
                    && el_known_type.insert((s.clone(), o.clone()))
                {
                    self.el.add_axiom(ElAxiom::ConceptAssertion {
                        individual: s.clone(),
                        concept: ElConcept::named(o.clone()),
                    });
                    el_round_added += 1;
                }
            }
            el_added_total += el_round_added;

            // ── EL pass ───────────────────────────────────────────────
            let cls = self
                .el
                .classify()
                .map_err(|e| RlElError::El(El2Error::from(e)))?;

            // Forward EL-derived subsumptions back to the RL reasoner.
            let mut rl_round_added = 0usize;
            let rl_existing: HashSet<(String, String, String)> = self.rl.all_triples();

            for (sub, supers) in &cls.subsumption_hierarchy {
                for sup in supers {
                    if sub == sup {
                        continue;
                    }
                    let triple_key = (sub.clone(), RDFS_SUBCLASS_OF.to_string(), sup.clone());
                    if !rl_existing.contains(&triple_key) {
                        self.rl.add_axiom(sub, RDFS_SUBCLASS_OF, sup);
                        rl_round_added += 1;
                    }
                }
            }
            // Forward EL-derived individual types as well.
            for (individual, types) in &cls.individual_types {
                if individual.starts_with("__wit_") {
                    continue;
                }
                for ty in types {
                    if ty.starts_with("__wit_") || ty == "owl:Thing" {
                        continue;
                    }
                    let triple_key = (individual.clone(), RDF_TYPE.to_string(), ty.clone());
                    if !rl_existing.contains(&triple_key) {
                        self.rl.add_axiom(individual, RDF_TYPE, ty);
                        rl_round_added += 1;
                    }
                }
            }
            rl_added_total += rl_round_added;

            // Termination: a fixpoint is reached once
            //   (a) RL added no new closure triples this round AND
            //   (b) the EL axiom set is unchanged AND
            //   (c) EL produced no new axioms to forward to RL.
            if !rl_grew && el_round_added == 0 && rl_round_added == 0 {
                break;
            }
        }

        Ok(RlElReport {
            alternations,
            rl_triples_added: rl_added_total,
            el_axioms_added: el_added_total,
            duration: start.elapsed(),
        })
    }

    /// Borrow the underlying RL reasoner (e.g. for inspecting the closure).
    pub fn rl(&self) -> &Owl2RlReasoner {
        &self.rl
    }

    /// Borrow the underlying EL reasoner.
    pub fn el(&self) -> &Owl2ElReasoner {
        &self.el
    }

    /// Compute the hybrid closure and return entailments as
    /// [`ProfileEntailment`]s.
    pub fn entailments(&mut self) -> Result<Vec<ProfileEntailment>, RlElError> {
        let _ = self.close()?;
        let mut out: HashSet<ProfileEntailment> = HashSet::new();

        for (s, p, o) in self.rl.all_triples() {
            if s.starts_with("__wit_") || o.starts_with("__wit_") {
                continue;
            }
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
    pub fn reason(&mut self) -> Result<ReasoningOutcome, RlElError> {
        let entailments = self.entailments()?;
        Ok(ReasoningOutcome::new(entailments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owl2::ontology::Owl2OntologyBuilder;

    #[test]
    fn rlel_handles_pure_rl_subset() {
        // RL-only ontology — combined closure equals plain RL closure.
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .build();

        let mut r = RlElReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("close ok");

        let has_rex_animal = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "rex" && class == "Animal"
            )
        });
        assert!(
            has_rex_animal,
            "rex should be Animal: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn rlel_handles_pure_el_subset() {
        // EL-only constructs: intersection on the LHS.
        let ontology = Owl2OntologyBuilder::new()
            .intersection_sub_class(vec!["Doctor".into(), "Surgeon".into()], "DoctorSurgeon")
            .type_of("alice", "Doctor")
            .type_of("alice", "Surgeon")
            .build();

        let mut r = RlElReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("close ok");

        let alice_intersection = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "alice" && class == "DoctorSurgeon"
            )
        });
        assert!(
            alice_intersection,
            "alice should be DoctorSurgeon via combined intersection rule; got: {:?}",
            outcome.entailments
        );
    }

    #[test]
    fn rlel_handles_hybrid_ontology() {
        // RL fragment: Mammal ⊑ Animal (RL transitive subclass)
        // EL fragment: Doctor ⊓ ChiefSurgeon ⊑ ChiefDoctor (EL intersection)
        // Combined: alice is Doctor + ChiefSurgeon AND alice's Animal-ness
        // (via type_of("alice", "Mammal")) propagates through both layers.
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Mammal", "Animal")
            .intersection_sub_class(vec!["Doctor".into(), "ChiefSurgeon".into()], "ChiefDoctor")
            .type_of("alice", "Mammal")
            .type_of("alice", "Doctor")
            .type_of("alice", "ChiefSurgeon")
            .build();

        let mut r = RlElReasoner::new();
        r.load(&ontology);
        let outcome = r.reason().expect("close ok");

        let alice_animal = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "alice" && class == "Animal"
            )
        });
        let alice_chief_doctor = outcome.entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "alice" && class == "ChiefDoctor"
            )
        });
        assert!(alice_animal, "alice should be Animal via RL");
        assert!(
            alice_chief_doctor,
            "alice should be ChiefDoctor via EL intersection; got: {:?}",
            outcome.entailments
        );
    }
}
