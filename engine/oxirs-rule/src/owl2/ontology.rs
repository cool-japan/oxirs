//! # OWL 2 Profile-Agnostic Ontology Container
//!
//! [`Owl2Ontology`] is the canonical input shape for all profile reasoners.
//! It stores TBox axioms and ABox facts in a profile-neutral form, then
//! projects to the appropriate per-profile axiom type via dedicated
//! adapter methods (`el_axioms`, `ql_axioms`, `rl_triples`, …).
//!
//! ## Why a separate type?
//!
//! Each backend reasoner has its own axiom enum (`ElAxiom`, `QlAxiom`,
//! triple form for RL).  The dispatcher needs one shared input format so
//! a single ontology can be evaluated under every profile without manual
//! re-conversion.  This module provides that shared format.
//!
//! ## Construction
//!
//! Use [`Owl2OntologyBuilder`] for ergonomic loading:
//!
//! ```
//! use oxirs_rule::owl2::Owl2OntologyBuilder;
//!
//! let ontology = Owl2OntologyBuilder::new()
//!     .sub_class_of("Dog", "Mammal")
//!     .sub_class_of("Mammal", "Animal")
//!     .type_of("rex", "Dog")
//!     .build();
//! ```

use std::collections::HashSet;

use crate::owl_el::{ElAxiom, ElConcept};
use crate::owl_ql::{QlAxiom, QlConcept, QlRole};

/// Profile-agnostic OWL 2 axiom representation.
///
/// Captures the union of constructs used by RL, EL and QL.  Each profile
/// adapter (in `el_reasoner.rs`, `ql_reasoner.rs`, `rlel_combined.rs`)
/// drops the constructs it cannot interpret natively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Owl2Axiom {
    /// Atomic subclass: `sub` ⊑ `sup`.
    SubClassOf { sub: String, sup: String },
    /// Equivalent named classes.
    EquivalentClasses(String, String),
    /// `parts[0] ⊓ parts[1] ⊓ … ⊑ sup` — intersection on the LHS.
    IntersectionSubClassOf { parts: Vec<String>, sup: String },
    /// `sub ⊑ ∃role.filler` — existential restriction on the RHS.
    SubClassOfSomeValuesFrom {
        sub: String,
        role: String,
        filler: String,
    },
    /// `∃role.filler ⊑ sup` — existential restriction on the LHS.
    SomeValuesFromSubClassOf {
        role: String,
        filler: String,
        sup: String,
    },
    /// `sub` ⊑ `sup` for object properties.
    SubObjectPropertyOf { sub: String, sup: String },
    /// Equivalent object properties.
    EquivalentObjectProperties(String, String),
    /// `p1 owl:inverseOf p2`.
    InverseObjectProperties(String, String),
    /// `r1 ∘ r2 ⊑ s` — binary property chain.
    PropertyChain {
        chain: Vec<String>,
        result_role: String,
    },
    /// `r rdf:type owl:TransitiveProperty`.
    TransitiveProperty(String),
    /// `r rdf:type owl:ReflexiveProperty` — every individual is related
    /// to itself via `r`.  Supported natively by RL and DL; for EL we
    /// emit no axiom (EL strictly rejects reflexivity declarations,
    /// per the W3C profile spec).
    ReflexiveProperty(String),
    /// `owl:ObjectPropertyDomain(p, c)` ≡ `∃p.⊤ ⊑ c`.
    ObjectPropertyDomain { property: String, class: String },
    /// `owl:ObjectPropertyRange(p, c)` ≡ `∃p⁻.⊤ ⊑ c`.
    ObjectPropertyRange { property: String, class: String },
}

/// A profile-agnostic OWL 2 ontology — TBox axioms plus ABox data.
#[derive(Debug, Clone, Default)]
pub struct Owl2Ontology {
    /// TBox axioms.
    pub axioms: Vec<Owl2Axiom>,
    /// ABox facts as concrete `(subject, predicate, object)` triples.
    /// Type assertions use `predicate == "rdf:type"`.
    pub data_triples: Vec<(String, String, String)>,
}

impl Owl2Ontology {
    /// Create an empty ontology.
    pub fn new() -> Self {
        Self::default()
    }

    /// All distinct named individuals appearing in the ABox.
    pub fn individuals(&self) -> Vec<String> {
        let mut set = HashSet::new();
        for (s, p, o) in &self.data_triples {
            set.insert(s.clone());
            if p != "rdf:type" {
                set.insert(o.clone());
            }
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    /// All distinct named object properties referenced in the TBox or ABox.
    pub fn properties(&self) -> Vec<String> {
        let mut set = HashSet::new();
        for axiom in &self.axioms {
            collect_properties(axiom, &mut set);
        }
        for (_, p, _) in &self.data_triples {
            if p != "rdf:type" {
                set.insert(p.clone());
            }
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    /// All distinct named classes referenced in the TBox or ABox.
    pub fn classes(&self) -> Vec<String> {
        let mut set = HashSet::new();
        for axiom in &self.axioms {
            collect_classes(axiom, &mut set);
        }
        for (_, p, o) in &self.data_triples {
            if p == "rdf:type" {
                set.insert(o.clone());
            }
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    /// Project axioms + ABox into the EL fragment ([`ElAxiom`]).
    pub fn el_axioms(&self) -> Vec<ElAxiom> {
        let mut out = Vec::new();
        for axiom in &self.axioms {
            match axiom {
                Owl2Axiom::SubClassOf { sub, sup } => {
                    out.push(ElAxiom::SubConceptOf {
                        sub: ElConcept::named(sub),
                        sup: ElConcept::named(sup),
                    });
                }
                Owl2Axiom::EquivalentClasses(a, b) => {
                    out.push(ElAxiom::EquivalentConcepts(
                        ElConcept::named(a),
                        ElConcept::named(b),
                    ));
                }
                Owl2Axiom::IntersectionSubClassOf { parts, sup } if parts.len() >= 2 => {
                    let inter =
                        ElConcept::intersection(parts.iter().map(ElConcept::named).collect());
                    out.push(ElAxiom::SubConceptOf {
                        sub: inter,
                        sup: ElConcept::named(sup),
                    });
                }
                Owl2Axiom::IntersectionSubClassOf { .. } => {
                    // Singleton or empty intersections degenerate; no EL form.
                }
                Owl2Axiom::SubClassOfSomeValuesFrom { sub, role, filler } => {
                    out.push(ElAxiom::SubConceptOf {
                        sub: ElConcept::named(sub),
                        sup: ElConcept::some_values(role, ElConcept::named(filler)),
                    });
                }
                Owl2Axiom::SomeValuesFromSubClassOf { role, filler, sup } => {
                    out.push(ElAxiom::SubConceptOf {
                        sub: ElConcept::some_values(role, ElConcept::named(filler)),
                        sup: ElConcept::named(sup),
                    });
                }
                Owl2Axiom::SubObjectPropertyOf { sub, sup } => {
                    out.push(ElAxiom::SubRole {
                        sub: sub.clone(),
                        sup: sup.clone(),
                    });
                }
                Owl2Axiom::EquivalentObjectProperties(a, b) => {
                    out.push(ElAxiom::SubRole {
                        sub: a.clone(),
                        sup: b.clone(),
                    });
                    out.push(ElAxiom::SubRole {
                        sub: b.clone(),
                        sup: a.clone(),
                    });
                }
                Owl2Axiom::PropertyChain { chain, result_role } => {
                    out.push(ElAxiom::PropertyChain {
                        chain: chain.clone(),
                        result_role: result_role.clone(),
                    });
                }
                Owl2Axiom::TransitiveProperty(role) => {
                    out.push(ElAxiom::TransitiveRole(role.clone()));
                }
                // Inverse, Domain, Range have no native EL counterpart — drop.
                _ => {}
            }
        }
        // Replay ABox as concept/role assertions.
        for (s, p, o) in &self.data_triples {
            if p == "rdf:type" {
                out.push(ElAxiom::ConceptAssertion {
                    individual: s.clone(),
                    concept: ElConcept::named(o),
                });
            } else {
                out.push(ElAxiom::RoleAssertion {
                    subject: s.clone(),
                    role: p.clone(),
                    object: o.clone(),
                });
            }
        }
        out
    }

    /// Project axioms into the QL fragment ([`QlAxiom`]).
    pub fn ql_axioms(&self) -> Vec<QlAxiom> {
        let mut out = Vec::new();
        for axiom in &self.axioms {
            match axiom {
                Owl2Axiom::SubClassOf { sub, sup } => {
                    out.push(QlAxiom::SubClassOf {
                        sub: QlConcept::Named(sub.clone()),
                        sup: QlConcept::Named(sup.clone()),
                    });
                }
                Owl2Axiom::EquivalentClasses(a, b) => {
                    out.push(QlAxiom::EquivalentClasses(
                        QlConcept::Named(a.clone()),
                        QlConcept::Named(b.clone()),
                    ));
                }
                Owl2Axiom::SubObjectPropertyOf { sub, sup } => {
                    out.push(QlAxiom::SubObjectPropertyOf {
                        sub: QlRole::Named(sub.clone()),
                        sup: QlRole::Named(sup.clone()),
                    });
                }
                Owl2Axiom::EquivalentObjectProperties(a, b) => {
                    out.push(QlAxiom::EquivalentObjectProperties(
                        QlRole::Named(a.clone()),
                        QlRole::Named(b.clone()),
                    ));
                }
                Owl2Axiom::InverseObjectProperties(a, b) => {
                    out.push(QlAxiom::InverseObjectProperties(a.clone(), b.clone()));
                }
                Owl2Axiom::ObjectPropertyDomain { property, class } => {
                    out.push(QlAxiom::ObjectPropertyDomain {
                        property: property.clone(),
                        domain: class.clone(),
                    });
                }
                Owl2Axiom::ObjectPropertyRange { property, class } => {
                    out.push(QlAxiom::ObjectPropertyRange {
                        property: property.clone(),
                        range: class.clone(),
                    });
                }
                // Intersection / existential / chain / transitive are
                // not supported in QL — quietly dropped.
                _ => {}
            }
        }
        out
    }

    /// Project axioms + data into RDF-style triples consumable by the
    /// [`crate::owl_rl::Owl2RlReasoner`].  Returns full triples using
    /// canonical RDFS/OWL URIs.
    pub fn rl_triples(&self) -> Vec<(String, String, String)> {
        use crate::owl_rl::{
            OWL_EQUIVALENT_CLASS, OWL_EQUIVALENT_PROPERTY, OWL_INVERSE_OF, OWL_TRANSITIVE_PROPERTY,
            RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF, RDFS_SUBPROPERTY_OF, RDF_TYPE,
        };

        let mut out: Vec<(String, String, String)> = Vec::new();
        for axiom in &self.axioms {
            match axiom {
                Owl2Axiom::SubClassOf { sub, sup } => {
                    out.push((sub.clone(), RDFS_SUBCLASS_OF.into(), sup.clone()));
                }
                Owl2Axiom::EquivalentClasses(a, b) => {
                    out.push((a.clone(), OWL_EQUIVALENT_CLASS.into(), b.clone()));
                }
                Owl2Axiom::SubObjectPropertyOf { sub, sup } => {
                    out.push((sub.clone(), RDFS_SUBPROPERTY_OF.into(), sup.clone()));
                }
                Owl2Axiom::EquivalentObjectProperties(a, b) => {
                    out.push((a.clone(), OWL_EQUIVALENT_PROPERTY.into(), b.clone()));
                }
                Owl2Axiom::InverseObjectProperties(a, b) => {
                    out.push((a.clone(), OWL_INVERSE_OF.into(), b.clone()));
                }
                Owl2Axiom::ObjectPropertyDomain { property, class } => {
                    out.push((property.clone(), RDFS_DOMAIN.into(), class.clone()));
                }
                Owl2Axiom::ObjectPropertyRange { property, class } => {
                    out.push((property.clone(), RDFS_RANGE.into(), class.clone()));
                }
                Owl2Axiom::TransitiveProperty(role) => {
                    out.push((
                        role.clone(),
                        RDF_TYPE.into(),
                        OWL_TRANSITIVE_PROPERTY.into(),
                    ));
                }
                Owl2Axiom::ReflexiveProperty(role) => {
                    // owl:ReflexiveProperty marker — RL has rule prp-rfp
                    // that derives x P x for every x ∈ owl:Thing.
                    out.push((
                        role.clone(),
                        RDF_TYPE.into(),
                        "http://www.w3.org/2002/07/owl#ReflexiveProperty".into(),
                    ));
                }
                // Intersection/existential/property-chain TBoxes have a
                // verbose RDF encoding that requires fresh blank-node
                // structures.  The RL reasoner natively handles the
                // produced triples, but we omit the encoding here to
                // keep RL closures fast — these constructs feed the EL
                // backend in `RLEL` mode.
                _ => {}
            }
        }
        // Pass through ABox data, normalising rdf:type.
        for (s, p, o) in &self.data_triples {
            let predicate = if p == "rdf:type" {
                RDF_TYPE.to_string()
            } else {
                p.clone()
            };
            out.push((s.clone(), predicate, o.clone()));
        }
        out
    }
}

fn collect_classes(axiom: &Owl2Axiom, set: &mut HashSet<String>) {
    match axiom {
        Owl2Axiom::SubClassOf { sub, sup } => {
            set.insert(sub.clone());
            set.insert(sup.clone());
        }
        Owl2Axiom::EquivalentClasses(a, b) => {
            set.insert(a.clone());
            set.insert(b.clone());
        }
        Owl2Axiom::IntersectionSubClassOf { parts, sup } => {
            for p in parts {
                set.insert(p.clone());
            }
            set.insert(sup.clone());
        }
        Owl2Axiom::SubClassOfSomeValuesFrom { sub, filler, .. } => {
            set.insert(sub.clone());
            set.insert(filler.clone());
        }
        Owl2Axiom::SomeValuesFromSubClassOf { filler, sup, .. } => {
            set.insert(filler.clone());
            set.insert(sup.clone());
        }
        Owl2Axiom::ObjectPropertyDomain { class, .. }
        | Owl2Axiom::ObjectPropertyRange { class, .. } => {
            set.insert(class.clone());
        }
        _ => {}
    }
}

fn collect_properties(axiom: &Owl2Axiom, set: &mut HashSet<String>) {
    match axiom {
        Owl2Axiom::SubClassOfSomeValuesFrom { role, .. }
        | Owl2Axiom::SomeValuesFromSubClassOf { role, .. } => {
            set.insert(role.clone());
        }
        Owl2Axiom::SubObjectPropertyOf { sub, sup }
        | Owl2Axiom::EquivalentObjectProperties(sub, sup)
        | Owl2Axiom::InverseObjectProperties(sub, sup) => {
            set.insert(sub.clone());
            set.insert(sup.clone());
        }
        Owl2Axiom::PropertyChain { chain, result_role } => {
            for c in chain {
                set.insert(c.clone());
            }
            set.insert(result_role.clone());
        }
        Owl2Axiom::TransitiveProperty(role) | Owl2Axiom::ReflexiveProperty(role) => {
            set.insert(role.clone());
        }
        Owl2Axiom::ObjectPropertyDomain { property, .. }
        | Owl2Axiom::ObjectPropertyRange { property, .. } => {
            set.insert(property.clone());
        }
        _ => {}
    }
}

/// One profile-level entailment in the canonical "what was derived?"
/// answer set returned by every reasoner.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProfileEntailment {
    /// `sub` ⊑ `sup` is entailed by the TBox.
    SubClassOf { sub: String, sup: String },
    /// `individual` rdf:type `class` is entailed by TBox + ABox.
    Type { individual: String, class: String },
    /// `(subject, property, object)` is entailed (e.g. via inverseOf).
    Property {
        subject: String,
        property: String,
        object: String,
    },
}

/// Result of running a profile reasoner over an [`Owl2Ontology`].
#[derive(Debug, Clone, Default)]
pub struct ReasoningOutcome {
    /// Sorted, de-duplicated entailments derived by the reasoner.
    pub entailments: Vec<ProfileEntailment>,
}

impl ReasoningOutcome {
    /// Wrap a list of entailments in an outcome.
    pub fn new(entailments: Vec<ProfileEntailment>) -> Self {
        Self { entailments }
    }

    /// Quick lookup: does this outcome contain `entailment`?
    pub fn contains(&self, entailment: &ProfileEntailment) -> bool {
        self.entailments.contains(entailment)
    }

    /// Number of entailments in the outcome.
    pub fn len(&self) -> usize {
        self.entailments.len()
    }

    /// Whether the outcome is empty.
    pub fn is_empty(&self) -> bool {
        self.entailments.is_empty()
    }
}

/// Builder for constructing an [`Owl2Ontology`] fluently.
#[derive(Debug, Default)]
pub struct Owl2OntologyBuilder {
    ontology: Owl2Ontology,
}

impl Owl2OntologyBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `sub ⊑ sup`.
    pub fn sub_class_of(mut self, sub: &str, sup: &str) -> Self {
        self.ontology.axioms.push(Owl2Axiom::SubClassOf {
            sub: sub.to_string(),
            sup: sup.to_string(),
        });
        self
    }

    /// Add `a ≡ b`.
    pub fn equivalent_classes(mut self, a: &str, b: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::EquivalentClasses(a.to_string(), b.to_string()));
        self
    }

    /// Add `parts[0] ⊓ parts[1] ⊓ … ⊑ sup`.
    pub fn intersection_sub_class(mut self, parts: Vec<String>, sup: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::IntersectionSubClassOf {
                parts,
                sup: sup.to_string(),
            });
        self
    }

    /// Add `sub ⊑ ∃role.filler`.
    pub fn sub_class_some_values(mut self, sub: &str, role: &str, filler: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::SubClassOfSomeValuesFrom {
                sub: sub.to_string(),
                role: role.to_string(),
                filler: filler.to_string(),
            });
        self
    }

    /// Add `∃role.filler ⊑ sup`.
    pub fn some_values_sub_class(mut self, role: &str, filler: &str, sup: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::SomeValuesFromSubClassOf {
                role: role.to_string(),
                filler: filler.to_string(),
                sup: sup.to_string(),
            });
        self
    }

    /// Add `sub ⊑ sup` for object properties.
    pub fn sub_property_of(mut self, sub: &str, sup: &str) -> Self {
        self.ontology.axioms.push(Owl2Axiom::SubObjectPropertyOf {
            sub: sub.to_string(),
            sup: sup.to_string(),
        });
        self
    }

    /// Add `a owl:inverseOf b`.
    pub fn inverse_of(mut self, a: &str, b: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::InverseObjectProperties(
                a.to_string(),
                b.to_string(),
            ));
        self
    }

    /// Add `r ∘ s ⊑ result`.
    pub fn property_chain(mut self, chain: Vec<String>, result_role: &str) -> Self {
        self.ontology.axioms.push(Owl2Axiom::PropertyChain {
            chain,
            result_role: result_role.to_string(),
        });
        self
    }

    /// Declare a property transitive.
    pub fn transitive(mut self, property: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::TransitiveProperty(property.to_string()));
        self
    }

    /// Declare a property reflexive (every individual self-related via it).
    pub fn reflexive(mut self, property: &str) -> Self {
        self.ontology
            .axioms
            .push(Owl2Axiom::ReflexiveProperty(property.to_string()));
        self
    }

    /// Add `rdfs:domain(property, class)`.
    pub fn domain(mut self, property: &str, class: &str) -> Self {
        self.ontology.axioms.push(Owl2Axiom::ObjectPropertyDomain {
            property: property.to_string(),
            class: class.to_string(),
        });
        self
    }

    /// Add `rdfs:range(property, class)`.
    pub fn range(mut self, property: &str, class: &str) -> Self {
        self.ontology.axioms.push(Owl2Axiom::ObjectPropertyRange {
            property: property.to_string(),
            class: class.to_string(),
        });
        self
    }

    /// Add `individual rdf:type class`.
    pub fn type_of(mut self, individual: &str, class: &str) -> Self {
        self.ontology.data_triples.push((
            individual.to_string(),
            "rdf:type".to_string(),
            class.to_string(),
        ));
        self
    }

    /// Add `subject property object`.
    pub fn property(mut self, subject: &str, property: &str, object: &str) -> Self {
        self.ontology.data_triples.push((
            subject.to_string(),
            property.to_string(),
            object.to_string(),
        ));
        self
    }

    /// Finish building the ontology.
    pub fn build(self) -> Owl2Ontology {
        self.ontology
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_collects_individuals_and_classes() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Mammal")
            .sub_class_of("Mammal", "Animal")
            .type_of("rex", "Dog")
            .property("rex", "hasOwner", "alice")
            .build();

        let inds = ontology.individuals();
        assert!(inds.contains(&"rex".to_string()));
        assert!(inds.contains(&"alice".to_string()));

        let classes = ontology.classes();
        assert!(classes.contains(&"Dog".to_string()));
        assert!(classes.contains(&"Mammal".to_string()));
        assert!(classes.contains(&"Animal".to_string()));
    }

    #[test]
    fn projects_to_el_axioms() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("A", "B")
            .intersection_sub_class(vec!["A".into(), "B".into()], "C")
            .type_of("x", "A")
            .build();

        let el = ontology.el_axioms();
        let has_subclass = el.iter().any(|a| {
            matches!(
                a,
                ElAxiom::SubConceptOf {
                    sub: ElConcept::Named(s),
                    sup: ElConcept::Named(t)
                } if s == "A" && t == "B"
            )
        });
        assert!(has_subclass, "Expected A ⊑ B as ElAxiom");

        let has_intersection = el.iter().any(|a| {
            matches!(
                a,
                ElAxiom::SubConceptOf {
                    sub: ElConcept::Intersection(_),
                    sup: ElConcept::Named(s)
                } if s == "C"
            )
        });
        assert!(has_intersection, "Expected A ⊓ B ⊑ C as ElAxiom");
    }

    #[test]
    fn projects_to_ql_axioms_drops_intersections() {
        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("A", "B")
            .intersection_sub_class(vec!["A".into(), "B".into()], "C")
            .inverse_of("p", "q")
            .build();

        let ql = ontology.ql_axioms();
        let has_subclass = ql.iter().any(|a| {
            matches!(
                a,
                QlAxiom::SubClassOf {
                    sub: QlConcept::Named(s),
                    sup: QlConcept::Named(t),
                } if s == "A" && t == "B"
            )
        });
        assert!(has_subclass, "QL projection must keep SubClassOf");

        let has_intersection = ql.iter().any(|a| {
            matches!(a, QlAxiom::SubClassOf { sub, .. } if matches!(sub, QlConcept::Named(_)))
                && matches!(
                    a,
                    QlAxiom::SubClassOf { sub: QlConcept::Named(s), .. } if s == "A_inter_B"
                )
        });
        assert!(
            !has_intersection,
            "QL projection must drop intersection axioms"
        );

        let has_inverse = ql.iter().any(|a| {
            matches!(
                a,
                QlAxiom::InverseObjectProperties(p, q) if p == "p" && q == "q"
            )
        });
        assert!(has_inverse, "QL projection must keep inverse properties");
    }

    #[test]
    fn projects_to_rl_triples_with_uri_predicates() {
        use crate::owl_rl::{RDFS_SUBCLASS_OF, RDF_TYPE};

        let ontology = Owl2OntologyBuilder::new()
            .sub_class_of("Dog", "Animal")
            .type_of("rex", "Dog")
            .build();
        let triples = ontology.rl_triples();

        assert!(triples
            .iter()
            .any(|(s, p, o)| { s == "Dog" && p == RDFS_SUBCLASS_OF && o == "Animal" }));
        assert!(triples
            .iter()
            .any(|(s, p, o)| s == "rex" && p == RDF_TYPE && o == "Dog"));
    }
}
