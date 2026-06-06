//! # OWL 2 EL Axiom and Concept Types
//!
//! Core data types for the OWL 2 EL profile: concept expressions, axioms,
//! normalised axiom forms, and classification results.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors from OWL 2 EL reasoning
#[derive(Debug, Error)]
pub enum ElError {
    #[error("Ontology is inconsistent: {0}")]
    Inconsistency(String),

    #[error("Invalid axiom: {0}")]
    InvalidAxiom(String),

    #[error("Maximum work items ({0}) exceeded during classification")]
    MaxWorkExceeded(usize),
}

/// An EL concept expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElConcept {
    /// owl:Thing (top concept)
    Top,
    /// owl:Nothing (bottom concept)
    Bottom,
    /// Named atomic concept (class IRI)
    Named(String),
    /// ObjectIntersectionOf(C1, C2, ...)
    Intersection(Vec<ElConcept>),
    /// ObjectSomeValuesFrom(role, filler)
    SomeValues {
        role: String,
        filler: Box<ElConcept>,
    },
}

impl ElConcept {
    /// Create a named concept
    pub fn named(iri: impl Into<String>) -> Self {
        Self::Named(iri.into())
    }

    /// Create an intersection
    pub fn intersection(concepts: Vec<ElConcept>) -> Self {
        match concepts.len() {
            0 => Self::Top,
            1 => concepts.into_iter().next().unwrap_or(Self::Top),
            _ => Self::Intersection(concepts),
        }
    }

    /// Create a SomeValuesFrom restriction
    pub fn some_values(role: impl Into<String>, filler: ElConcept) -> Self {
        Self::SomeValues {
            role: role.into(),
            filler: Box::new(filler),
        }
    }

    /// Return atomic name if this is Named
    pub fn as_named(&self) -> Option<&str> {
        if let Self::Named(n) = self {
            Some(n)
        } else {
            None
        }
    }
}

/// An OWL 2 EL axiom
#[derive(Debug, Clone)]
pub enum ElAxiom {
    /// C ⊑ D (subclass)
    SubConceptOf { sub: ElConcept, sup: ElConcept },
    /// C ≡ D (equivalent class)
    EquivalentConcepts(ElConcept, ElConcept),
    /// Individual x : C (concept assertion)
    ConceptAssertion {
        individual: String,
        concept: ElConcept,
    },
    /// (x, y) : r (role assertion)
    RoleAssertion {
        subject: String,
        role: String,
        object: String,
    },
    /// r1 o r2 ⊑ s (property chain, binary or longer)
    PropertyChain {
        chain: Vec<String>,
        result_role: String,
    },
    /// r1 ⊑ r2 (sub-property)
    SubRole { sub: String, sup: String },
    /// r rdf:type owl:TransitiveProperty
    TransitiveRole(String),
}

/// Normalized axiom form for the EL completion algorithm.
/// All complex axioms are reduced to one of these normal forms.
#[derive(Debug, Clone)]
pub enum NormalAxiom {
    /// A ⊑ B  (two atomic concept names)
    AtomSubAtom(String, String),
    /// A ⊓ B ⊑ C  (binary intersection on left)
    InterAtomSubAtom(String, String, String),
    /// A ⊑ ∃r.B  (existential on right)
    AtomSubSome(String, String, String),
    /// ∃r.A ⊑ B  (existential on left)
    SomeSubAtom(String, String, String),
    /// Transitive role r
    TransRole(String),
    /// Role chain: r1 o r2 ⊑ s
    RoleChain(String, String, String),
    /// Sub-role: r ⊑ s
    SubRole(String, String),
}

/// Result of EL classification
#[derive(Debug, Clone)]
pub struct ElClassification {
    /// subsumption_hierarchy: concept → set of all named superclasses (including transitive)
    pub subsumption_hierarchy: HashMap<String, HashSet<String>>,
    /// Groups of equivalent classes (each group contains mutually equivalent names)
    pub equivalent_classes: Vec<Vec<String>>,
    /// Role successor sets: (individual_or_concept, role) → set of successors
    pub role_successors: HashMap<(String, String), HashSet<String>>,
    /// ABox: per-individual concept memberships
    pub individual_types: HashMap<String, HashSet<String>>,
    /// Number of saturation loop iterations performed
    pub iterations: usize,
    /// Total subsumption relationships computed
    pub subsumptions_computed: usize,
}

impl ElClassification {
    pub(crate) fn new() -> Self {
        Self {
            subsumption_hierarchy: HashMap::new(),
            equivalent_classes: Vec::new(),
            role_successors: HashMap::new(),
            individual_types: HashMap::new(),
            iterations: 0,
            subsumptions_computed: 0,
        }
    }

    /// Get all named superclasses of a concept (direct and indirect)
    pub fn get_superclasses(&self, class: &str) -> Vec<String> {
        self.subsumption_hierarchy
            .get(class)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all named subclasses of a concept
    pub fn get_subclasses(&self, class: &str) -> Vec<String> {
        self.subsumption_hierarchy
            .iter()
            .filter(|(_, supers)| supers.contains(class))
            .map(|(sub, _)| sub.clone())
            .collect()
    }

    /// Check if `sub` ⊑ `sup`
    pub fn is_subclass_of(&self, sub: &str, sup: &str) -> bool {
        sub == sup
            || self
                .subsumption_hierarchy
                .get(sub)
                .map(|s| s.contains(sup))
                .unwrap_or(false)
    }

    /// Get all concept names an individual belongs to
    pub fn get_individual_types(&self, individual: &str) -> Vec<String> {
        self.individual_types
            .get(individual)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }
}
