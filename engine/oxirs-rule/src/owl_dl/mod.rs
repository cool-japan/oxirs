//! # OWL 2 DL ABox Reasoning — Structural Subsumption
//!
//! This module advances OWL 2 DL reasoning from ~60% to 80%+ coverage by
//! providing a dedicated ABox reasoning engine focused on structural subsumption.
//!
//! ## Implemented ABox Features
//!
//! - **Individual classification**: Classify individuals under inferred `rdf:type` assertions
//!   using TBox subclass hierarchies and property restrictions.
//! - **Property chain reasoning**: Handle `owl:propertyChainAxiom` — compose chains of
//!   property assertions to produce inferred property triples.
//! - **Nominal reasoning**: Handle `owl:oneOf` (enumeration classes) — an individual
//!   is a member of a OneOf class if it appears in the enumeration list.
//! - **HasValue reasoning**: `owl:hasValue` restrictions — any individual satisfying a
//!   property assertion to the specified value is classified as a member.
//! - **AllValuesFrom**: Propagate `owl:allValuesFrom` restrictions through known property
//!   assertions to classify filler individuals.
//! - **SomeValuesFrom**: Check `owl:someValuesFrom` satisfiability and classify individuals
//!   when witnesses exist.
//! - **Transitivity**: Forward-close `owl:TransitiveProperty` assertions.
//! - **Symmetry**: Infer reverse assertions for `owl:SymmetricProperty`.
//! - **Asymmetry**: Detect inconsistencies for `owl:AsymmetricProperty`.
//! - **ObjectComplementOf**: Detect C(x) ∧ ¬C(x) contradictions.
//! - **ObjectIntersectionOf**: x ∈ C1 ∩ C2 iff x ∈ C1 and x ∈ C2.
//! - **DisjointUnionOf**: Classes are mutually disjoint and their union equals parent.
//! - **Key axioms (owl:hasKey)**: Unique identifying property sets per class.
//! - **FunctionalProperty**: x P a, x P b → a sameAs b.
//! - **InverseFunctionalProperty**: a P x, b P x → a sameAs b.
//! - **Universal quantifier check**: AllValuesFrom universal instantiation.
//! - **Existential check**: SomeValuesFrom witness obligation tracking.
//!
//! ## Architecture
//!
//! The [`Owl2DLReasoner`] stores ABox assertions (ground triples) and TBox axioms
//! (class/property descriptions) separately.  A fixpoint `materialize()` loop applies
//! all DL-specific rules until no new facts are derivable.  The engine builds several
//! indexed views of the ABox for efficient rule application.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::owl_dl::{Owl2DLReasoner, DLError};
//! use oxirs_rule::owl_dl::vocab::*;
//!
//! let mut r = Owl2DLReasoner::new();
//! // Declare hasValue restriction: ParentOf5 ≡ ∃hasChild.{:child5}
//! r.add_has_value_restriction("ParentOf5", "hasChild", ":child5");
//! r.add_property_assertion(":alice", "hasChild", ":child5");
//!
//! let report = r.materialize().expect("materialization failed");
//! assert!(r.is_type_entailed(":alice", "ParentOf5"));
//! ```
//!
//! ## References
//! - <https://www.w3.org/TR/owl2-syntax/>
//! - <https://www.w3.org/TR/owl2-direct-semantics/>
//! - <https://www.w3.org/TR/owl2-profiles/#OWL_2_DL>

pub mod vocab;

mod abox_rules;
mod cardinality_rules;
mod complex_constructors;
mod property_hierarchy;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_100pct;

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use thiserror::Error;

pub use vocab::RDF_TYPE;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the OWL 2 DL ABox reasoner
#[derive(Debug, Error)]
pub enum DLError {
    /// The ABox is inconsistent (e.g. disjoint classes both asserted)
    #[error("ABox inconsistency: {0}")]
    Inconsistency(String),

    /// Safety limit exceeded during fixpoint iteration
    #[error("Maximum fixpoint iterations ({0}) exceeded")]
    MaxIterationsExceeded(usize),

    /// A TBox axiom was structurally invalid
    #[error("Invalid TBox axiom: {0}")]
    InvalidAxiom(String),
}

// ── Data structures ───────────────────────────────────────────────────────────

/// A ground RDF triple (subject, predicate, object) — all strings
pub type Triple = (String, String, String);

/// Property chain axiom: `P owl:propertyChainAxiom [P1, P2, …, Pn]`
///
/// Semantics: if `x P1 v1`, `v1 P2 v2`, …, `v_{n-1} Pn y` then `x P y`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyChain {
    /// The entailed property
    pub entailed_property: String,
    /// Ordered list of constituent properties
    pub chain: Vec<String>,
}

/// A `owl:hasValue` restriction node
///
/// Semantics: if `x prop value` then `x rdf:type restriction_class`.
/// Conversely, if `x rdf:type restriction_class` then `x prop value`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HasValueRestriction {
    /// Named restriction class IRI
    pub restriction_class: String,
    /// The property
    pub property: String,
    /// The filler value (individual IRI)
    pub value: String,
}

/// A `owl:allValuesFrom` restriction
///
/// Semantics: if `x rdf:type restriction` and `x prop y` then `y rdf:type filler_class`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllValuesFromRestriction {
    pub restriction_class: String,
    pub property: String,
    pub filler_class: String,
}

/// A `owl:someValuesFrom` restriction
///
/// Semantics: if `x prop y` and `y rdf:type filler_class` then `x rdf:type restriction`.
/// Also used for existential classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SomeValuesFromRestriction {
    pub restriction_class: String,
    pub property: String,
    pub filler_class: String,
}

/// A `owl:oneOf` nominal class
///
/// Semantics: each individual in `members` is of type `class_iri`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NominalClass {
    pub class_iri: String,
    pub members: Vec<String>,
}

/// An `owl:intersectionOf` class constructor
///
/// Semantics: x ∈ IntersectionClass iff x ∈ operand for all operands.
/// Also used in reverse: if x ∈ all operands, infer x ∈ IntersectionClass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntersectionOfClass {
    /// The named class that represents the intersection
    pub class_iri: String,
    /// All operand classes that must hold simultaneously
    pub operands: Vec<String>,
}

/// An `owl:complementOf` class constructor
///
/// Semantics: x ∈ ComplementClass iff x ∉ base_class.
/// Used for inconsistency detection: x ∈ C and x ∈ ¬C is a contradiction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplementOfClass {
    /// The complement class (¬base_class)
    pub class_iri: String,
    /// The class being complemented
    pub base_class: String,
}

/// An `owl:disjointUnionOf` axiom
///
/// Semantics: the parent class is the union of all operands, and the operands
/// are mutually disjoint with each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjointUnionOf {
    /// The parent class equal to the union
    pub class_iri: String,
    /// Mutually disjoint operand classes whose union = class_iri
    pub operands: Vec<String>,
}

/// An `owl:hasKey` axiom
///
/// Semantics: for instances of `class_iri`, the combination of properties in
/// `key_properties` uniquely identifies each individual.  If two individuals
/// of the class share identical values for all key properties, they are sameAs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HasKeyAxiom {
    pub class_iri: String,
    pub key_properties: Vec<String>,
}

/// A `owl:NegativeObjectPropertyAssertion` axiom
///
/// Semantics: the triple `(source_individual, assertion_property, target_individual)`
/// must NOT hold in the ABox.  If it does, the ABox is inconsistent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegativeObjectPropertyAssertion {
    pub source_individual: String,
    pub assertion_property: String,
    pub target_individual: String,
}

/// A `owl:NegativeDataPropertyAssertion` axiom
///
/// Like [`NegativeObjectPropertyAssertion`] but the target is a data value (literal).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegativeDataPropertyAssertion {
    pub source_individual: String,
    pub assertion_property: String,
    pub target_value: String,
}

/// A `hasSelf` (local-reflexivity self-restriction) class
///
/// Semantics: `x ∈ restriction_class  ↔  (x, property, x) ∈ ABox`.
///
/// - *Forward*: if `x property x` then `x rdf:type restriction_class`.
/// - *Backward*: if `x rdf:type restriction_class` then `x property x`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HasSelfRestriction {
    /// The named restriction class
    pub restriction_class: String,
    /// The property for which the individual must be self-related
    pub property: String,
}

/// Characterisation of a property's OWL characteristics
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PropertyCharacteristics {
    pub is_transitive: bool,
    pub is_symmetric: bool,
    pub is_asymmetric: bool,
    pub is_reflexive: bool,
    pub is_irreflexive: bool,
    pub is_functional: bool,
    pub is_inverse_functional: bool,
    /// `owl:inverseOf` partner (if declared)
    pub inverse_of: Option<String>,
}

/// Per-rule firing statistics
#[derive(Debug, Clone, Default)]
pub struct RuleFirings {
    pub individual_classification: usize,
    pub property_chain: usize,
    pub nominal_classification: usize,
    pub has_value_forward: usize,
    pub has_value_backward: usize,
    pub all_values_from: usize,
    pub some_values_from: usize,
    pub transitivity: usize,
    pub symmetry: usize,
    pub asymmetry_checks: usize,
    pub subclass_propagation: usize,
    pub equivalent_class: usize,
    pub domain_range: usize,
    pub same_as_propagation: usize,
    pub inverse_property: usize,
    pub complement_of: usize,
    pub intersection_of: usize,
    pub disjoint_union: usize,
    pub has_key: usize,
    pub functional_property: usize,
    pub inverse_functional_property: usize,
    // ── 90% milestone additions ───────────────────────────────────────────────
    pub sub_object_property: usize,
    pub sub_data_property: usize,
    pub equivalent_properties: usize,
    pub reflexive_self: usize,
    pub has_self: usize,
    pub negative_property_assertion: usize,
    // ── 100% milestone additions ─────────────────────────────────────────────
    pub max_cardinality: usize,
    pub min_cardinality: usize,
    pub exact_cardinality: usize,
    pub union_of: usize,
    pub data_some_values_from: usize,
    pub data_all_values_from: usize,
    pub all_different: usize,
    pub same_as_congruence: usize,
}

/// Summary of a single materialization run
#[derive(Debug, Clone)]
pub struct DLInferenceReport {
    /// Number of fixpoint iterations
    pub iterations: usize,
    /// Total new triples inferred
    pub new_triples: usize,
    /// Per-rule firing counts
    pub rule_firings: RuleFirings,
    /// Wall-clock duration
    pub duration: std::time::Duration,
    /// Detected inconsistencies
    pub inconsistencies: Vec<String>,
}

// ── Main reasoner ─────────────────────────────────────────────────────────────

/// OWL 2 DL ABox reasoner implementing structural subsumption
///
/// This reasoner maintains a set of ground triples (the ABox) and a set of
/// TBox axioms (subclass relationships, property restrictions, etc.).  It
/// applies forward-chaining DL rules until a fixpoint is reached.
pub struct Owl2DLReasoner {
    // ── ABox ground triples (asserted + inferred) ─────────────────────
    pub(crate) abox: HashSet<Triple>,
    /// Triples that were explicitly asserted (not inferred)
    pub(crate) asserted: HashSet<Triple>,

    // ── TBox axioms ───────────────────────────────────────────────────
    /// `(sub, super)` direct subclass pairs
    pub(crate) subclass_of: HashSet<(String, String)>,
    /// Equivalent class pairs `(A, B)` — directional; both (A,B) and (B,A) are stored
    pub(crate) equivalent_classes: HashSet<(String, String)>,
    /// Disjoint class pairs
    pub(crate) disjoint_classes: HashSet<(String, String)>,

    // ── Restriction axioms ────────────────────────────────────────────
    pub(crate) has_value_restrictions: Vec<HasValueRestriction>,
    pub(crate) all_values_restrictions: Vec<AllValuesFromRestriction>,
    pub(crate) some_values_restrictions: Vec<SomeValuesFromRestriction>,
    pub(crate) nominal_classes: Vec<NominalClass>,
    pub(crate) property_chains: Vec<PropertyChain>,

    // ── Complex class constructors (new in 80% milestone) ─────────────
    pub(crate) complement_of_classes: Vec<ComplementOfClass>,
    pub(crate) intersection_of_classes: Vec<IntersectionOfClass>,
    pub(crate) disjoint_unions: Vec<DisjointUnionOf>,
    pub(crate) has_key_axioms: Vec<HasKeyAxiom>,

    // ── Property hierarchy axioms (new in 90% milestone) ──────────────
    /// `(sub_property, super_property)` direct object-property subsumption pairs
    pub(crate) sub_object_property_of: HashSet<(String, String)>,
    /// `(sub_property, super_property)` direct data-property subsumption pairs
    pub(crate) sub_data_property_of: HashSet<(String, String)>,
    /// Equivalent property pairs `(P1, P2)` — both directions stored
    pub(crate) equivalent_properties: HashSet<(String, String)>,
    /// Disjoint property pairs `(P1, P2)` — both directions stored
    pub(crate) disjoint_properties: HashSet<(String, String)>,

    // ── Self restrictions (new in 90% milestone) ──────────────────────
    pub(crate) has_self_restrictions: Vec<HasSelfRestriction>,

    // ── Negative property assertions (new in 90% milestone) ───────────
    pub(crate) negative_object_assertions: Vec<NegativeObjectPropertyAssertion>,
    pub(crate) negative_data_assertions: Vec<NegativeDataPropertyAssertion>,

    // ── 100% milestone: cardinality, union, data restrictions, AllDifferent ─
    pub(crate) cardinality_restrictions: Vec<cardinality_rules::CardinalityRestriction>,
    pub(crate) union_of_classes: Vec<cardinality_rules::UnionOfClass>,
    pub(crate) data_some_values_restrictions: Vec<cardinality_rules::DataSomeValuesFromRestriction>,
    pub(crate) data_all_values_restrictions: Vec<cardinality_rules::DataAllValuesFromRestriction>,
    pub(crate) all_different_axioms: Vec<cardinality_rules::AllDifferentAxiom>,

    // ── Property characteristics ──────────────────────────────────────
    pub(crate) property_chars: HashMap<String, PropertyCharacteristics>,

    // ── Configuration ─────────────────────────────────────────────────
    pub(crate) max_iterations: usize,

    // ── Runtime state ────────────────────────────────────────────────
    pub(crate) inconsistencies: Vec<String>,
}

// ── Constructor & configuration ───────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Create a new empty OWL 2 DL reasoner with sensible defaults
    pub fn new() -> Self {
        Self {
            abox: HashSet::new(),
            asserted: HashSet::new(),
            subclass_of: HashSet::new(),
            equivalent_classes: HashSet::new(),
            disjoint_classes: HashSet::new(),
            has_value_restrictions: Vec::new(),
            all_values_restrictions: Vec::new(),
            some_values_restrictions: Vec::new(),
            nominal_classes: Vec::new(),
            property_chains: Vec::new(),
            complement_of_classes: Vec::new(),
            intersection_of_classes: Vec::new(),
            disjoint_unions: Vec::new(),
            has_key_axioms: Vec::new(),
            sub_object_property_of: HashSet::new(),
            sub_data_property_of: HashSet::new(),
            equivalent_properties: HashSet::new(),
            disjoint_properties: HashSet::new(),
            has_self_restrictions: Vec::new(),
            negative_object_assertions: Vec::new(),
            negative_data_assertions: Vec::new(),
            cardinality_restrictions: Vec::new(),
            union_of_classes: Vec::new(),
            data_some_values_restrictions: Vec::new(),
            data_all_values_restrictions: Vec::new(),
            all_different_axioms: Vec::new(),
            property_chars: HashMap::new(),
            max_iterations: 500,
            inconsistencies: Vec::new(),
        }
    }

    /// Override the maximum number of fixpoint iterations (safety limit)
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
}

impl Default for Owl2DLReasoner {
    fn default() -> Self {
        Self::new()
    }
}

// ── ABox assertion API ────────────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Assert a raw RDF triple into the ABox
    pub fn assert_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let t = mk_triple(subject, predicate, object);
        self.asserted.insert(t.clone());
        self.abox.insert(t);
    }

    /// Assert `individual rdf:type class`
    pub fn assert_type(&mut self, individual: &str, class: &str) {
        self.assert_triple(individual, vocab::RDF_TYPE, class);
    }

    /// Assert a property assertion `subject property object`
    pub fn add_property_assertion(&mut self, subject: &str, property: &str, object: &str) {
        self.assert_triple(subject, property, object);
    }

    /// Assert `ind1 owl:sameAs ind2` (and its symmetric counterpart)
    pub fn assert_same_as(&mut self, ind1: &str, ind2: &str) {
        self.assert_triple(ind1, vocab::OWL_SAME_AS, ind2);
        self.assert_triple(ind2, vocab::OWL_SAME_AS, ind1);
    }

    /// Assert `ind1 owl:differentFrom ind2`
    pub fn assert_different_from(&mut self, ind1: &str, ind2: &str) {
        self.assert_triple(ind1, vocab::OWL_DIFFERENT_FROM, ind2);
        self.assert_triple(ind2, vocab::OWL_DIFFERENT_FROM, ind1);
    }
}

// ── TBox axiom API ────────────────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Add `sub rdfs:subClassOf sup`
    pub fn add_subclass_of(&mut self, sub: &str, sup: &str) {
        self.subclass_of.insert((sub.to_string(), sup.to_string()));
        // Also store as ABox triple so rule patterns can match
        self.abox
            .insert(mk_triple(sub, vocab::RDFS_SUBCLASS_OF, sup));
        self.asserted
            .insert(mk_triple(sub, vocab::RDFS_SUBCLASS_OF, sup));
    }

    /// Add `c1 owl:equivalentClass c2` (symmetric)
    pub fn add_equivalent_classes(&mut self, c1: &str, c2: &str) {
        self.equivalent_classes
            .insert((c1.to_string(), c2.to_string()));
        self.equivalent_classes
            .insert((c2.to_string(), c1.to_string()));
        self.abox
            .insert(mk_triple(c1, vocab::OWL_EQUIVALENT_CLASS, c2));
        self.abox
            .insert(mk_triple(c2, vocab::OWL_EQUIVALENT_CLASS, c1));
    }

    /// Add `c1 owl:disjointWith c2` (symmetric)
    pub fn add_disjoint_classes(&mut self, c1: &str, c2: &str) {
        self.disjoint_classes
            .insert((c1.to_string(), c2.to_string()));
        self.disjoint_classes
            .insert((c2.to_string(), c1.to_string()));
        self.abox
            .insert(mk_triple(c1, vocab::OWL_DISJOINT_WITH, c2));
        self.abox
            .insert(mk_triple(c2, vocab::OWL_DISJOINT_WITH, c1));
    }

    /// Add rdfs:domain axiom: `property rdfs:domain class`
    pub fn add_domain(&mut self, property: &str, class: &str) {
        self.abox
            .insert(mk_triple(property, vocab::RDFS_DOMAIN, class));
        self.asserted
            .insert(mk_triple(property, vocab::RDFS_DOMAIN, class));
    }

    /// Add rdfs:range axiom: `property rdfs:range class`
    pub fn add_range(&mut self, property: &str, class: &str) {
        self.abox
            .insert(mk_triple(property, vocab::RDFS_RANGE, class));
        self.asserted
            .insert(mk_triple(property, vocab::RDFS_RANGE, class));
    }

    /// Add `owl:inverseOf` declaration
    pub fn add_inverse_of(&mut self, p1: &str, p2: &str) {
        let chars1 = self.property_chars.entry(p1.to_string()).or_default();
        chars1.inverse_of = Some(p2.to_string());

        let chars2 = self.property_chars.entry(p2.to_string()).or_default();
        chars2.inverse_of = Some(p1.to_string());

        self.abox.insert(mk_triple(p1, vocab::OWL_INVERSE_OF, p2));
        self.asserted
            .insert(mk_triple(p1, vocab::OWL_INVERSE_OF, p2));
    }
}

// ── Property characteristic API ───────────────────────────────────────────────

impl Owl2DLReasoner {
    fn declare_property_char(&mut self, property: &str, char_type: &str) {
        self.property_chars.entry(property.to_string()).or_default();
        self.abox
            .insert(mk_triple(property, vocab::RDF_TYPE, char_type));
        self.asserted
            .insert(mk_triple(property, vocab::RDF_TYPE, char_type));
    }

    /// Declare `property rdf:type owl:TransitiveProperty`
    pub fn add_transitive_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_transitive = true;
        self.declare_property_char(property, vocab::OWL_TRANSITIVE_PROPERTY);
    }

    /// Declare `property rdf:type owl:SymmetricProperty`
    pub fn add_symmetric_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_symmetric = true;
        self.declare_property_char(property, vocab::OWL_SYMMETRIC_PROPERTY);
    }

    /// Declare `property rdf:type owl:AsymmetricProperty`
    pub fn add_asymmetric_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_asymmetric = true;
        self.declare_property_char(property, vocab::OWL_ASYMMETRIC_PROPERTY);
    }

    /// Declare `property rdf:type owl:ReflexiveProperty`
    pub fn add_reflexive_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_reflexive = true;
        self.declare_property_char(property, vocab::OWL_REFLEXIVE_PROPERTY);
    }

    /// Declare `property rdf:type owl:IrreflexiveProperty`
    pub fn add_irreflexive_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_irreflexive = true;
        self.declare_property_char(property, vocab::OWL_IRREFLEXIVE_PROPERTY);
    }

    /// Declare `property rdf:type owl:FunctionalProperty`
    pub fn add_functional_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_functional = true;
        self.declare_property_char(property, vocab::OWL_FUNCTIONAL_PROPERTY);
    }

    /// Declare `property rdf:type owl:InverseFunctionalProperty`
    pub fn add_inverse_functional_property(&mut self, property: &str) {
        self.property_chars
            .entry(property.to_string())
            .or_default()
            .is_inverse_functional = true;
        self.declare_property_char(property, vocab::OWL_INVERSE_FUNCTIONAL_PROPERTY);
    }
}

// ── Restriction & chain API ───────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Register a `owl:hasValue` restriction
    ///
    /// Declares that the class `restriction_class` is equivalent to
    /// `owl:Restriction owl:onProperty property; owl:hasValue value`.
    pub fn add_has_value_restriction(
        &mut self,
        restriction_class: &str,
        property: &str,
        value: &str,
    ) {
        self.has_value_restrictions.push(HasValueRestriction {
            restriction_class: restriction_class.to_string(),
            property: property.to_string(),
            value: value.to_string(),
        });
    }

    /// Register an `owl:allValuesFrom` restriction
    pub fn add_all_values_from_restriction(
        &mut self,
        restriction_class: &str,
        property: &str,
        filler_class: &str,
    ) {
        self.all_values_restrictions.push(AllValuesFromRestriction {
            restriction_class: restriction_class.to_string(),
            property: property.to_string(),
            filler_class: filler_class.to_string(),
        });
    }

    /// Register a `owl:someValuesFrom` restriction
    pub fn add_some_values_from_restriction(
        &mut self,
        restriction_class: &str,
        property: &str,
        filler_class: &str,
    ) {
        self.some_values_restrictions
            .push(SomeValuesFromRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                filler_class: filler_class.to_string(),
            });
    }

    /// Register an `owl:oneOf` nominal class
    pub fn add_nominal_class(&mut self, class_iri: &str, members: Vec<String>) {
        self.nominal_classes.push(NominalClass {
            class_iri: class_iri.to_string(),
            members,
        });
    }

    /// Register an `owl:propertyChainAxiom` — `entailed_property` is derived
    /// when all `chain` properties are traversed in sequence.
    pub fn add_property_chain(&mut self, entailed_property: &str, chain: Vec<String>) {
        self.property_chains.push(PropertyChain {
            entailed_property: entailed_property.to_string(),
            chain,
        });
    }

    /// Register an `owl:complementOf` class:
    /// `class_iri owl:complementOf base_class`
    ///
    /// Used for inconsistency detection: if `x rdf:type class_iri` and
    /// `x rdf:type base_class` both hold, that is a contradiction.
    pub fn add_complement_of(&mut self, class_iri: &str, base_class: &str) {
        self.complement_of_classes.push(ComplementOfClass {
            class_iri: class_iri.to_string(),
            base_class: base_class.to_string(),
        });
    }

    /// Register an `owl:intersectionOf` class:
    /// `class_iri owl:intersectionOf (op1 op2 … opN)`
    ///
    /// Forward: if `x rdf:type op_i` for ALL operands → infer `x rdf:type class_iri`.
    /// Backward: if `x rdf:type class_iri` → infer `x rdf:type op_i` for each operand.
    pub fn add_intersection_of(&mut self, class_iri: &str, operands: Vec<String>) {
        self.intersection_of_classes.push(IntersectionOfClass {
            class_iri: class_iri.to_string(),
            operands,
        });
    }

    /// Register an `owl:disjointUnionOf` axiom:
    /// `class_iri owl:disjointUnionOf (op1 op2 … opN)`
    ///
    /// Entails:
    /// - Each operand is a subclass of `class_iri` (union membership).
    /// - All pairs of operands are mutually disjoint.
    pub fn add_disjoint_union(&mut self, class_iri: &str, operands: Vec<String>) {
        self.disjoint_unions.push(DisjointUnionOf {
            class_iri: class_iri.to_string(),
            operands,
        });
    }

    /// Register an `owl:hasKey` axiom:
    /// `class_iri owl:hasKey (prop1 prop2 … propN)`
    ///
    /// If two individuals of `class_iri` share the same values for all key
    /// properties, they are asserted to be `owl:sameAs`.
    pub fn add_has_key(&mut self, class_iri: &str, key_properties: Vec<String>) {
        self.has_key_axioms.push(HasKeyAxiom {
            class_iri: class_iri.to_string(),
            key_properties,
        });
    }
}

// ── Property hierarchy API (90% milestone) ────────────────────────────────────

impl Owl2DLReasoner {
    /// Add `p1 rdfs:subPropertyOf p2` (object property subsumption)
    ///
    /// Semantics: if `x P1 y` then `x P2 y`.
    pub fn add_sub_object_property_of(&mut self, sub: &str, sup: &str) {
        self.sub_object_property_of
            .insert((sub.to_string(), sup.to_string()));
        self.abox
            .insert(mk_triple(sub, vocab::RDFS_SUBPROPERTY_OF, sup));
        self.asserted
            .insert(mk_triple(sub, vocab::RDFS_SUBPROPERTY_OF, sup));
    }

    /// Add `p1 rdfs:subPropertyOf p2` (data property subsumption)
    ///
    /// Semantics: if `x P1 v` (data value) then `x P2 v`.
    pub fn add_sub_data_property_of(&mut self, sub: &str, sup: &str) {
        self.sub_data_property_of
            .insert((sub.to_string(), sup.to_string()));
        self.abox
            .insert(mk_triple(sub, vocab::RDFS_SUBPROPERTY_OF, sup));
        self.asserted
            .insert(mk_triple(sub, vocab::RDFS_SUBPROPERTY_OF, sup));
    }

    /// Add `p1 owl:equivalentProperty p2` (bidirectional property subsumption)
    ///
    /// Both (p1, p2) and (p2, p1) are stored so that rules fire in both directions.
    pub fn add_equivalent_properties(&mut self, p1: &str, p2: &str) {
        self.equivalent_properties
            .insert((p1.to_string(), p2.to_string()));
        self.equivalent_properties
            .insert((p2.to_string(), p1.to_string()));
        self.abox
            .insert(mk_triple(p1, vocab::OWL_EQUIVALENT_PROPERTY, p2));
        self.abox
            .insert(mk_triple(p2, vocab::OWL_EQUIVALENT_PROPERTY, p1));
    }

    /// Add `p1 owl:propertyDisjointWith p2` (symmetric)
    ///
    /// If `x P1 y` and `x P2 y` both hold, the ABox is inconsistent.
    pub fn add_disjoint_properties(&mut self, p1: &str, p2: &str) {
        self.disjoint_properties
            .insert((p1.to_string(), p2.to_string()));
        self.disjoint_properties
            .insert((p2.to_string(), p1.to_string()));
        self.abox
            .insert(mk_triple(p1, vocab::OWL_PROPERTY_DISJOINT_WITH, p2));
        self.abox
            .insert(mk_triple(p2, vocab::OWL_PROPERTY_DISJOINT_WITH, p1));
    }

    /// Register a `hasSelf` local-reflexivity restriction
    ///
    /// Declares that `restriction_class` is equivalent to the set of individuals
    /// that are related to themselves via `property`.
    pub fn add_has_self_restriction(&mut self, restriction_class: &str, property: &str) {
        self.has_self_restrictions.push(HasSelfRestriction {
            restriction_class: restriction_class.to_string(),
            property: property.to_string(),
        });
    }

    /// Assert a `owl:NegativeObjectPropertyAssertion`
    ///
    /// The triple `(source, property, target)` must NOT hold.
    /// If it does appear in the ABox after materialisation, an inconsistency
    /// is reported.
    pub fn assert_negative_object_property_assertion(
        &mut self,
        source: &str,
        property: &str,
        target: &str,
    ) {
        self.negative_object_assertions
            .push(NegativeObjectPropertyAssertion {
                source_individual: source.to_string(),
                assertion_property: property.to_string(),
                target_individual: target.to_string(),
            });
    }

    /// Assert a `owl:NegativeDataPropertyAssertion`
    ///
    /// The triple `(source, property, target_value)` must NOT hold.
    /// If it appears in the ABox after materialisation, an inconsistency is reported.
    pub fn assert_negative_data_property_assertion(
        &mut self,
        source: &str,
        property: &str,
        target_value: &str,
    ) {
        self.negative_data_assertions
            .push(NegativeDataPropertyAssertion {
                source_individual: source.to_string(),
                assertion_property: property.to_string(),
                target_value: target_value.to_string(),
            });
    }
}

// ── 100% milestone: new axiom builder methods ─────────────────────────────────
impl Owl2DLReasoner {
    /// Add a `owl:maxCardinality` restriction.
    pub fn add_max_cardinality(&mut self, restriction_class: &str, property: &str, n: usize) {
        self.cardinality_restrictions
            .push(cardinality_rules::CardinalityRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                cardinality: n,
                kind: cardinality_rules::CardinalityKind::Max,
                qualifying_class: None,
            });
    }

    /// Add a `owl:maxQualifiedCardinality` restriction with a qualifying class.
    pub fn add_max_qualified_cardinality(
        &mut self,
        restriction_class: &str,
        property: &str,
        n: usize,
        qualifying_class: &str,
    ) {
        self.cardinality_restrictions
            .push(cardinality_rules::CardinalityRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                cardinality: n,
                kind: cardinality_rules::CardinalityKind::Max,
                qualifying_class: Some(qualifying_class.to_string()),
            });
    }

    /// Add a `owl:minCardinality` restriction.
    pub fn add_min_cardinality(&mut self, restriction_class: &str, property: &str, n: usize) {
        self.cardinality_restrictions
            .push(cardinality_rules::CardinalityRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                cardinality: n,
                kind: cardinality_rules::CardinalityKind::Min,
                qualifying_class: None,
            });
    }

    /// Add a `owl:minQualifiedCardinality` restriction with a qualifying class.
    pub fn add_min_qualified_cardinality(
        &mut self,
        restriction_class: &str,
        property: &str,
        n: usize,
        qualifying_class: &str,
    ) {
        self.cardinality_restrictions
            .push(cardinality_rules::CardinalityRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                cardinality: n,
                kind: cardinality_rules::CardinalityKind::Min,
                qualifying_class: Some(qualifying_class.to_string()),
            });
    }

    /// Add a `owl:exactCardinality` restriction.
    pub fn add_exact_cardinality(&mut self, restriction_class: &str, property: &str, n: usize) {
        self.cardinality_restrictions
            .push(cardinality_rules::CardinalityRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                cardinality: n,
                kind: cardinality_rules::CardinalityKind::Exact,
                qualifying_class: None,
            });
    }

    /// Add an `owl:unionOf` class expression.
    pub fn add_union_of(&mut self, class_iri: &str, operands: Vec<String>) {
        if operands.is_empty() {
            return;
        }
        self.union_of_classes.push(cardinality_rules::UnionOfClass {
            class_iri: class_iri.to_string(),
            operands,
        });
    }

    /// Add a `DataSomeValuesFrom` restriction.
    pub fn add_data_some_values_from(
        &mut self,
        restriction_class: &str,
        property: &str,
        datatype: Option<&str>,
    ) {
        self.data_some_values_restrictions
            .push(cardinality_rules::DataSomeValuesFromRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                datatype: datatype.map(|s| s.to_string()),
            });
    }

    /// Add a `DataAllValuesFrom` restriction.
    pub fn add_data_all_values_from(
        &mut self,
        restriction_class: &str,
        property: &str,
        datatype: &str,
    ) {
        self.data_all_values_restrictions
            .push(cardinality_rules::DataAllValuesFromRestriction {
                restriction_class: restriction_class.to_string(),
                property: property.to_string(),
                datatype: datatype.to_string(),
            });
    }

    /// Declare an `owl:AllDifferent` axiom.
    pub fn add_all_different(&mut self, members: Vec<String>) {
        if members.len() < 2 {
            return;
        }
        self.all_different_axioms
            .push(cardinality_rules::AllDifferentAxiom { members });
    }
}

// ── Query API ─────────────────────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Check whether `individual rdf:type class` is entailed (asserted or inferred)
    pub fn is_type_entailed(&self, individual: &str, class: &str) -> bool {
        self.abox
            .contains(&mk_triple(individual, vocab::RDF_TYPE, class))
    }

    /// Check whether a triple `(s, p, o)` is entailed
    pub fn is_triple_entailed(&self, s: &str, p: &str, o: &str) -> bool {
        self.abox.contains(&mk_triple(s, p, o))
    }

    /// Return all `rdf:type` classes asserted or inferred for `individual`
    pub fn get_types(&self, individual: &str) -> Vec<String> {
        self.abox
            .iter()
            .filter(|(s, p, _)| s == individual && p == vocab::RDF_TYPE)
            .map(|(_, _, o)| o.clone())
            .collect()
    }

    /// Return all inferred triples (those not in the original asserted set)
    pub fn inferred_triples(&self) -> Vec<Triple> {
        self.abox
            .iter()
            .filter(|t| !self.asserted.contains(*t))
            .cloned()
            .collect()
    }

    /// Whether the ABox is consistent (no inconsistencies detected)
    pub fn is_consistent(&self) -> bool {
        self.inconsistencies.is_empty()
    }

    /// Return all detected inconsistencies
    pub fn inconsistencies(&self) -> &[String] {
        &self.inconsistencies
    }

    /// Return all distinct individuals mentioned in the ABox
    pub fn all_individuals(&self) -> HashSet<String> {
        let mut individuals = HashSet::new();
        for (s, p, _o) in &self.abox {
            if p == vocab::RDF_TYPE {
                individuals.insert(s.clone());
            }
        }
        individuals
    }
}

// ── Materialization (fixpoint) ────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Run ABox reasoning to fixpoint.
    ///
    /// Applies all OWL 2 DL structural subsumption rules repeatedly until
    /// no new triples can be derived.  Returns a [`DLInferenceReport`] with
    /// statistics, or a [`DLError`] on inconsistency / iteration overflow.
    pub fn materialize(&mut self) -> Result<DLInferenceReport, DLError> {
        let start = Instant::now();
        self.inconsistencies.clear();
        let mut firings = RuleFirings::default();
        let mut total_new = 0usize;
        let mut iterations = 0usize;

        // Derive transitive closure of subclass axioms before main loop
        // so that rule application picks up inherited subsumptions.
        self.close_subclass_hierarchy();

        // Derive transitive closure of sub-property axioms before main loop.
        self.close_sub_property_hierarchy();

        // Pre-materialise disjoint union subclass and disjointness axioms (TBox expansion)
        self.expand_disjoint_unions();

        loop {
            if iterations >= self.max_iterations {
                return Err(DLError::MaxIterationsExceeded(self.max_iterations));
            }
            iterations += 1;

            let snapshot: HashSet<Triple> = self.abox.clone();
            let mut new_triples: HashSet<Triple> = HashSet::new();

            // --- Rule 1: Individual classification via subclass hierarchy -------
            self.apply_subclass_propagation(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 2: Equivalent class propagation ---------------------------
            self.apply_equivalent_class_propagation(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 3: nominal classification (owl:oneOf) ---------------------
            self.apply_nominal_classification(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 4: HasValue — forward (type → property) -------------------
            self.apply_has_value_forward(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 5: HasValue — backward (property → type) ------------------
            self.apply_has_value_backward(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 6: AllValuesFrom propagation ------------------------------
            self.apply_all_values_from(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 7: SomeValuesFrom classification --------------------------
            self.apply_some_values_from(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 8: Property chain reasoning --------------------------------
            self.apply_property_chains(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 9: Transitivity -------------------------------------------
            self.apply_transitivity(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 10: Symmetry ----------------------------------------------
            self.apply_symmetry(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 11: InverseOf ---------------------------------------------
            self.apply_inverse_of(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 12: Domain / range type inference -------------------------
            self.apply_domain_range(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 13: owl:sameAs propagation --------------------------------
            self.apply_same_as_propagation(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 14 (NEW): ObjectIntersectionOf (forward + backward) -------
            self.apply_intersection_of(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 15 (NEW): FunctionalProperty sameAs merging ---------------
            self.apply_functional_property(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 16 (NEW): InverseFunctionalProperty sameAs merging --------
            self.apply_inverse_functional_property(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 17 (NEW): HasKey unique-key sameAs entailment -------------
            self.apply_has_key(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 18 (90%): SubObjectPropertyOf / SubDataPropertyOf ---------
            self.apply_sub_property_of(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 19 (90%): EquivalentObjectProperties ----------------------
            self.apply_equivalent_properties(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 20 (90%): ReflexiveObjectProperty (self-loops) ------------
            self.apply_reflexive_property(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 21 (90%): hasSelf restrictions ----------------------------
            self.apply_has_self(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 22 (100%): MaxCardinality violation → owl:Nothing ----------
            self.apply_max_cardinality(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 23 (100%): MinCardinality backward classification ----------
            self.apply_min_cardinality(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 24 (100%): ExactCardinality (max+min combined) -------------
            self.apply_exact_cardinality(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 25 (100%): ObjectUnionOf membership propagation -----------
            self.apply_union_of(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 26 (100%): DataSomeValuesFrom backward classification -----
            self.apply_data_some_values_from(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 27 (100%): DataAllValuesFrom filler classification --------
            self.apply_data_all_values_from(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 28 (100%): AllDifferent axiom materialization -------------
            self.apply_all_different(&snapshot, &mut new_triples, &mut firings);

            // --- Rule 29 (100%): Full sameAs congruence (property inheritance) --
            self.apply_same_as_full_congruence(&snapshot, &mut new_triples, &mut firings);

            // Filter triples already in the ABox
            let genuinely_new: HashSet<Triple> = new_triples
                .into_iter()
                .filter(|t| !snapshot.contains(t))
                .collect();

            if genuinely_new.is_empty() {
                break;
            }

            total_new += genuinely_new.len();
            self.abox.extend(genuinely_new);
        }

        // Asymmetry and irreflexivity checks (inconsistency detection)
        self.check_asymmetry_violations(&firings);
        self.check_irreflexivity_violations();
        self.check_disjoint_violations();
        self.check_nothing_violations();

        // NEW: ComplementOf inconsistency detection
        self.check_complement_of_violations(&mut firings);

        // NEW: DisjointUnionOf operand pair disjointness violations
        self.check_disjoint_union_violations();

        // 90% milestone: DisjointProperties inconsistency check
        self.check_disjoint_property_violations();

        // 90% milestone: NegativePropertyAssertion inconsistency check
        self.check_negative_property_assertion_violations(&mut firings);

        // 100% milestone: AllDifferent + sameAs contradiction check
        self.check_all_different_violations();

        // 100% milestone: MaxCardinality post-loop violation check
        self.check_max_cardinality_violations();

        Ok(DLInferenceReport {
            iterations,
            new_triples: total_new,
            rule_firings: firings,
            duration: start.elapsed(),
            inconsistencies: self.inconsistencies.clone(),
        })
    }

    // ── Private: subproperty closure ──────────────────────────────────────────

    /// Compute the transitive closure of `rdfs:subPropertyOf` for both object-
    /// and data-property hierarchies before the main fixpoint loop.
    pub(crate) fn close_sub_property_hierarchy(&mut self) {
        // --- Object property closure ---
        let mut obj_pairs: HashSet<(String, String)> = self.sub_object_property_of.clone();
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<(String, String)> = obj_pairs.iter().cloned().collect();
            for (a, b) in &snapshot {
                for (c, d) in &snapshot {
                    if b == c && !obj_pairs.contains(&(a.clone(), d.clone())) {
                        obj_pairs.insert((a.clone(), d.clone()));
                        self.abox
                            .insert(mk_triple(a, vocab::RDFS_SUBPROPERTY_OF, d));
                        changed = true;
                    }
                }
            }
        }
        self.sub_object_property_of = obj_pairs;

        // --- Data property closure ---
        let mut data_pairs: HashSet<(String, String)> = self.sub_data_property_of.clone();
        let mut data_changed = true;
        while data_changed {
            data_changed = false;
            let snapshot: Vec<(String, String)> = data_pairs.iter().cloned().collect();
            for (a, b) in &snapshot {
                for (c, d) in &snapshot {
                    if b == c && !data_pairs.contains(&(a.clone(), d.clone())) {
                        data_pairs.insert((a.clone(), d.clone()));
                        self.abox
                            .insert(mk_triple(a, vocab::RDFS_SUBPROPERTY_OF, d));
                        data_changed = true;
                    }
                }
            }
        }
        self.sub_data_property_of = data_pairs;
    }

    // ── Private: subclass closure ─────────────────────────────────────────────

    /// Compute the transitive closure of `rdfs:subClassOf` in the TBox
    pub(crate) fn close_subclass_hierarchy(&mut self) {
        // Copy to avoid borrow conflicts
        let mut pairs: HashSet<(String, String)> = self.subclass_of.clone();

        // Floyd-Warshall style transitive closure
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<(String, String)> = pairs.iter().cloned().collect();
            for (a, b) in &snapshot {
                for (c, d) in &snapshot {
                    if b == c && !pairs.contains(&(a.clone(), d.clone())) {
                        pairs.insert((a.clone(), d.clone()));
                        self.abox.insert(mk_triple(a, vocab::RDFS_SUBCLASS_OF, d));
                        changed = true;
                    }
                }
            }
        }
        self.subclass_of = pairs;
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Construct a [`Triple`] from string slices
#[inline]
pub(crate) fn mk_triple(s: &str, p: &str, o: &str) -> Triple {
    (s.to_string(), p.to_string(), o.to_string())
}

/// BFS reachability: returns all nodes reachable from `start` via `adj`
pub(crate) fn bfs_reachable<'a>(
    start: &'a str,
    adj: &'a HashMap<&'a str, Vec<&'a str>>,
) -> Vec<&'a str> {
    let mut visited: HashSet<&str> = HashSet::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if visited.contains(node) {
            continue;
        }
        visited.insert(node);
        if let Some(neighbors) = adj.get(node) {
            for &n in neighbors {
                if !visited.contains(n) {
                    queue.push_back(n);
                }
            }
        }
    }

    visited.into_iter().collect()
}

/// Evaluate a property chain by joining through each segment.
///
/// Returns `(start, end)` pairs derivable from the chain.
pub(crate) fn evaluate_chain<'a>(
    chain: &[String],
    pred_index: &HashMap<&'a str, Vec<(&'a str, &'a str)>>,
) -> Vec<(&'a str, &'a str)> {
    if chain.is_empty() {
        return Vec::new();
    }

    // Start with all (s, o) pairs for chain[0]
    let first_pred = chain[0].as_str();
    let mut current: Vec<(&str, &str)> = pred_index.get(first_pred).cloned().unwrap_or_default();

    // Join through each subsequent chain step
    for step_pred in chain.iter().skip(1) {
        let step_pairs = pred_index
            .get(step_pred.as_str())
            .cloned()
            .unwrap_or_default();

        // Build a lookup: from → [to] for this step
        let mut step_map: HashMap<&str, Vec<&str>> = HashMap::new();
        for (s, o) in &step_pairs {
            step_map.entry(s).or_default().push(o);
        }

        let mut next: Vec<(&str, &str)> = Vec::new();
        for (start, mid) in &current {
            if let Some(ends) = step_map.get(mid) {
                for &end in ends {
                    next.push((start, end));
                }
            }
        }
        current = next;
    }

    current
}
