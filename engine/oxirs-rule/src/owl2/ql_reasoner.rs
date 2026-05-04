//! # OWL 2 QL Reasoner — Profile-aware façade
//!
//! Wraps the existing [`crate::owl_ql::QueryRewriter`] (a PerfectRef
//! implementation extended with `ObjectUnionOf`) with the
//! [`super::ProfileReasoner`] interface.
//!
//! Unlike the EL backend the QL backend rewrites **queries**, not the
//! ontology — so the dispatcher offers two entry points:
//!
//! 1. [`ProfileQlReasoner::ucq_for_query`] — produces a
//!    [`crate::owl_ql::RewrittenQuery`] (Union of Conjunctive Queries).
//! 2. [`ProfileQlReasoner::entailments`] — answers the canonical
//!    "what type assertions does the TBox derive?" question by feeding
//!    the inferred subsumption hierarchy back through PerfectRef.

use std::collections::HashSet;
use thiserror::Error;

use crate::owl_ql::{
    ConjunctiveQuery, Owl2QLTBox, QlAxiom, QlConcept, QlError, QlRole, QueryAtom, QueryRewriter,
    QueryTerm, RewrittenQuery,
};

use super::ontology::{Owl2Ontology, ProfileEntailment, ReasoningOutcome};

/// Errors specific to the QL profile façade.
#[derive(Debug, Error)]
pub enum Ql2Error {
    /// The wrapped QL TBox/Rewriter reported an error.
    #[error("QL reasoning failed: {0}")]
    Reasoning(#[from] QlError),
}

/// Profile-aware QL reasoner.
pub struct ProfileQlReasoner {
    tbox: Owl2QLTBox,
    classified: bool,
}

impl Default for ProfileQlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileQlReasoner {
    /// Create a new QL reasoner with an empty TBox.
    pub fn new() -> Self {
        Self {
            tbox: Owl2QLTBox::new(),
            classified: false,
        }
    }

    /// Add a single QL axiom.
    pub fn add_axiom(&mut self, axiom: QlAxiom) {
        self.classified = false;
        self.tbox.add_axiom(axiom);
    }

    /// Load the QL fragment of an [`Owl2Ontology`] into the reasoner.
    pub fn load(&mut self, ontology: &Owl2Ontology) {
        for axiom in ontology.ql_axioms() {
            self.add_axiom(axiom);
        }
    }

    /// Force classification (computes the closure used by all subsequent
    /// query rewrites).
    pub fn classify(&mut self) -> Result<(), Ql2Error> {
        if !self.classified {
            self.tbox.classify()?;
            self.classified = true;
        }
        Ok(())
    }

    /// Borrow the underlying TBox for advanced inspection.
    pub fn tbox(&self) -> &Owl2QLTBox {
        &self.tbox
    }

    /// Rewrite a conjunctive query as a UCQ using PerfectRef.
    pub fn ucq_for_query(&mut self, query: &ConjunctiveQuery) -> Result<RewrittenQuery, Ql2Error> {
        self.classify()?;
        let rewriter = QueryRewriter::new(self.tbox.clone());
        Ok(rewriter.rewrite_query(query)?)
    }

    /// Compute the certain answers / type entailments for the loaded ABox.
    ///
    /// QL has no general "materialise the closure" operation, so this
    /// answers each ground type and property assertion in the ABox by
    /// rewriting it as a Boolean conjunctive query and checking whether
    /// any rewriting step produces a structurally-grounded match against
    /// the data (`ontology.data_triples`).
    ///
    /// Returns:
    /// - `Type(individual, class)` entailments for every named class such
    ///   that the rewritten query has a satisfying assignment
    /// - `Property(subject, property, object)` entailments for every named
    ///   property pair where the rewritten property query is satisfied
    /// - `SubClassOf(sub, sup)` entailments from the classified TBox closure
    pub fn entailments(
        &mut self,
        ontology: &Owl2Ontology,
    ) -> Result<Vec<ProfileEntailment>, Ql2Error> {
        self.classify()?;
        let rewriter = QueryRewriter::new(self.tbox.clone());

        let mut out: HashSet<ProfileEntailment> = HashSet::new();

        // Collect all individuals + class candidates + properties.
        let individuals = ontology.individuals();
        let classes = ontology.classes();
        let properties = ontology.properties();

        // Type entailments
        for individual in &individuals {
            for class in &classes {
                let q = ConjunctiveQuery::with_atoms(vec![QueryAtom::TypeAtom {
                    individual: QueryTerm::constant(individual),
                    class: class.clone(),
                }]);

                let rewritten = rewriter.rewrite_query(&q)?;
                if rewritten_matches_data(&rewritten, ontology) {
                    out.insert(ProfileEntailment::Type {
                        individual: individual.clone(),
                        class: class.clone(),
                    });
                }
            }
        }

        // Property entailments — for each (subject, predicate, object)
        // pair of individuals × properties, ask whether the boolean CQ
        // matches under PerfectRef rewriting.
        for property in &properties {
            for subject in &individuals {
                for object in &individuals {
                    let q = ConjunctiveQuery::with_atoms(vec![QueryAtom::PropertyAtom {
                        subject: QueryTerm::constant(subject),
                        property: property.clone(),
                        object: QueryTerm::constant(object),
                    }]);
                    let rewritten = rewriter.rewrite_query(&q)?;
                    if rewritten_matches_data(&rewritten, ontology) {
                        out.insert(ProfileEntailment::Property {
                            subject: subject.clone(),
                            property: property.clone(),
                            object: object.clone(),
                        });
                    }
                }
            }
        }

        // Subclass entailments — for each pair (B, C) with B ≠ C,
        // emit B ⊑ C if QL TBox closure derives it.
        for sub in &classes {
            let supers = self.tbox.superclasses(sub);
            for sup in supers {
                if sub != &sup {
                    out.insert(ProfileEntailment::SubClassOf {
                        sub: sub.clone(),
                        sup,
                    });
                }
            }
        }

        let mut sorted: Vec<ProfileEntailment> = out.into_iter().collect();
        sorted.sort();
        Ok(sorted)
    }

    /// Run reasoning and return a [`ReasoningOutcome`].
    pub fn reason(&mut self, ontology: &Owl2Ontology) -> Result<ReasoningOutcome, Ql2Error> {
        let entailments = self.entailments(ontology)?;
        Ok(ReasoningOutcome::new(entailments))
    }

    /// Convenience: add a SubClassOf axiom by class names.
    pub fn add_subclass_of(&mut self, sub: &str, sup: &str) {
        self.add_axiom(QlAxiom::SubClassOf {
            sub: QlConcept::Named(sub.to_string()),
            sup: QlConcept::Named(sup.to_string()),
        });
    }

    /// Convenience: add a SubObjectPropertyOf axiom by property names.
    pub fn add_subproperty_of(&mut self, sub: &str, sup: &str) {
        self.add_axiom(QlAxiom::SubObjectPropertyOf {
            sub: QlRole::Named(sub.to_string()),
            sup: QlRole::Named(sup.to_string()),
        });
    }

    /// Convenience: declare InverseObjectProperties.
    pub fn add_inverse(&mut self, p1: &str, p2: &str) {
        self.add_axiom(QlAxiom::InverseObjectProperties(
            p1.to_string(),
            p2.to_string(),
        ));
    }
}

/// Match the rewritten UCQ against ontology ABox data: returns true iff
/// any conjunctive query in the UCQ has a satisfying substitution.
fn rewritten_matches_data(rewritten: &RewrittenQuery, ontology: &Owl2Ontology) -> bool {
    rewritten
        .queries
        .iter()
        .any(|cq| ground_cq_holds(cq, ontology))
}

/// Boolean evaluation of a conjunctive query against the data triples
/// of `ontology` using simple substitution-set semantics.
fn ground_cq_holds(cq: &ConjunctiveQuery, ontology: &Owl2Ontology) -> bool {
    use std::collections::HashMap;

    // Backtracking search over variable bindings.
    fn search(
        atoms: &[QueryAtom],
        bindings: &HashMap<String, String>,
        ontology: &Owl2Ontology,
    ) -> bool {
        let Some((first, rest)) = atoms.split_first() else {
            return true;
        };

        // Collect candidate substitutions for `first`.
        let candidate_facts: Vec<(String, String, String)> = match first {
            QueryAtom::TypeAtom { individual, class } => ontology
                .data_triples
                .iter()
                .filter(|(_, p, o)| p == "rdf:type" && o == class)
                .filter(|(s, _, _)| match individual {
                    QueryTerm::Variable(v) => match bindings.get(v) {
                        Some(bound) => bound == s,
                        None => true,
                    },
                    QueryTerm::Constant(c) => c == s,
                })
                .cloned()
                .collect(),
            QueryAtom::PropertyAtom {
                subject,
                property,
                object,
            } => ontology
                .data_triples
                .iter()
                .filter(|(_, p, _)| p == property)
                .filter(|(s, _, o)| {
                    let s_ok = match subject {
                        QueryTerm::Variable(v) => bindings.get(v).map_or(true, |b| b == s),
                        QueryTerm::Constant(c) => c == s,
                    };
                    let o_ok = match object {
                        QueryTerm::Variable(v) => bindings.get(v).map_or(true, |b| b == o),
                        QueryTerm::Constant(c) => c == o,
                    };
                    s_ok && o_ok
                })
                .cloned()
                .collect(),
        };

        for fact in candidate_facts {
            let mut new_bindings = bindings.clone();
            let bind_ok = match first {
                QueryAtom::TypeAtom { individual, .. } => match individual {
                    QueryTerm::Variable(v) => match new_bindings.get(v) {
                        Some(existing) => existing == &fact.0,
                        None => {
                            new_bindings.insert(v.clone(), fact.0.clone());
                            true
                        }
                    },
                    QueryTerm::Constant(c) => c == &fact.0,
                },
                QueryAtom::PropertyAtom {
                    subject, object, ..
                } => {
                    let mut ok = true;
                    if let QueryTerm::Variable(v) = subject {
                        match new_bindings.get(v) {
                            Some(existing) if existing != &fact.0 => ok = false,
                            None => {
                                new_bindings.insert(v.clone(), fact.0.clone());
                            }
                            _ => {}
                        }
                    }
                    if ok {
                        if let QueryTerm::Variable(v) = object {
                            match new_bindings.get(v) {
                                Some(existing) if existing != &fact.2 => ok = false,
                                None => {
                                    new_bindings.insert(v.clone(), fact.2.clone());
                                }
                                _ => {}
                            }
                        }
                    }
                    ok
                }
            };

            if bind_ok && search(rest, &new_bindings, ontology) {
                return true;
            }
        }
        false
    }

    search(&cq.atoms, &HashMap::new(), ontology)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owl2::ontology::Owl2OntologyBuilder;

    #[test]
    fn ql_subclass_entailment() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .build();

        let mut r = ProfileQlReasoner::new();
        r.load(&ontology);
        let entailments = r.entailments(&ontology).expect("entailments");

        let has_rex_animal = entailments.iter().any(|e| {
            matches!(
                e,
                ProfileEntailment::Type { individual, class }
                    if individual == "rex" && class == "Animal"
            )
        });
        assert!(
            has_rex_animal,
            "rex should be inferred as Animal via subclass; got: {entailments:?}"
        );
    }

    #[test]
    fn ql_inverse_property() {
        let ontology = Owl2OntologyBuilder::new()
            .inverse_of("hasParent", "hasChild")
            .property("alice", "hasChild", "bob")
            .build();

        let mut r = ProfileQlReasoner::new();
        r.load(&ontology);
        r.classify().expect("classify ok");

        let q = ConjunctiveQuery::with_atoms(vec![QueryAtom::PropertyAtom {
            subject: QueryTerm::constant("bob"),
            property: "hasParent".to_string(),
            object: QueryTerm::constant("alice"),
        }]);
        let ucq = r.ucq_for_query(&q).expect("rewrite ok");
        // The UCQ must contain an alternative that uses hasChild with swapped args.
        let contains_inverse_branch = ucq.queries.iter().any(|cq| {
            cq.atoms.iter().any(|a| {
                matches!(
                    a,
                    QueryAtom::PropertyAtom {
                        subject: QueryTerm::Constant(s),
                        property,
                        object: QueryTerm::Constant(o),
                    } if property == "hasChild" && s == "alice" && o == "bob"
                )
            })
        });
        assert!(
            contains_inverse_branch,
            "Expected UCQ branch using inverse property: {ucq:?}"
        );
    }
}
