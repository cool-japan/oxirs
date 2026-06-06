//! # OWL 2 RL Rule Definitions and Vocabulary
//!
//! Contains type aliases, error types, OWL 2 RL rule identifiers, pattern types,
//! inference report, and RDF/OWL URI constants.

use std::collections::HashMap;
use thiserror::Error;

/// A triple in the form (subject, predicate, object) using string URIs/literals
pub type Triple = (String, String, String);

/// Variable binding map from variable names to concrete values
pub type Bindings = HashMap<String, String>;

/// Errors from OWL 2 RL reasoning
#[derive(Debug, Error)]
pub enum RlError {
    #[error("Ontology inconsistency detected: {0}")]
    Inconsistency(String),

    #[error("Maximum iterations ({0}) exceeded during materialization")]
    MaxIterationsExceeded(usize),

    #[error("Invalid axiom: {0}")]
    InvalidAxiom(String),
}

/// OWL 2 RL rule identifiers per W3C OWL 2 RL spec
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Owl2RlRule {
    // --- Table 4: Semantics of Schema Vocabulary ---
    /// Subclass transitivity: C1 ⊑ C2, C2 ⊑ C3 → C1 ⊑ C3
    ScmSco,
    /// SubProperty transitivity: P1 ⊑ P2, P2 ⊑ P3 → P1 ⊑ P3
    ScmSpo,
    /// EquivalentClass to subClassOf: C1 ≡ C2 → C1 ⊑ C2, C2 ⊑ C1
    ScmEqc1,
    ScmEqc2,
    /// EquivalentProperty to subPropertyOf
    ScmEqp1,
    ScmEqp2,
    /// Domain/range inheritance via subClassOf
    ScmDom1,
    ScmDom2,
    ScmRng1,
    ScmRng2,
    /// SubProperty preserves domain/range
    ScmHv,
    /// IntersectionOf
    ScmInt,
    /// UnionOf
    ScmUni,

    // --- Table 5: Property axioms (prp-*) ---
    /// SubPropertyOf: P1 ⊑ P2, x P1 y → x P2 y
    PrpSpo1,
    PrpSpo2,
    /// EquivalentProperty propagation
    PrpEqp1,
    PrpEqp2,
    /// Domain: P rdfs:domain C, x P y → x rdf:type C
    PrpDom,
    /// Range: P rdfs:range C, x P y → y rdf:type C
    PrpRng,
    /// Functional property: x P y1, x P y2 → y1 owl:sameAs y2
    PrpFp,
    /// InverseFunctional: x1 P y, x2 P y → x1 owl:sameAs x2
    PrpIfp,
    /// IrreflexiveProperty violation detection
    PrpIrp,
    /// SymmetricProperty: x P y → y P x
    PrpSymp,
    /// AsymmetricProperty violation detection
    PrpAsynp,
    /// TransitiveProperty: x P y, y P z → x P z
    PrpTrp,
    /// InverseOf: P1 owl:inverseOf P2, x P1 y → y P2 x
    PrpInv1,
    PrpInv2,
    /// Key axioms
    PrpKey,
    /// DisjointWith
    PrpPdw,
    /// NegativePropertyAssertion
    PrpNpa1,
    PrpNpa2,

    // --- Table 6: Class axioms (cls-*) ---
    /// Type from intersection membership
    ClsInt1,
    ClsInt2,
    /// Type from union membership
    ClsUni,
    /// ExistentialRestriction: owl:someValuesFrom
    ClsSvf1,
    ClsSvf2,
    /// UniversalRestriction: owl:allValuesFrom
    ClsAvf,
    /// hasValue restriction
    ClsHv1,
    ClsHv2,
    /// MaxCardinality = 0
    ClsMaxc1,
    ClsMaxc2,
    /// MaxCardinality = 1
    ClsMaxqc1,
    ClsMaxqc2,
    /// owl:Nothing is bottom
    ClsNothing1,
    ClsNothing2,

    // --- Table 7: Class expression axioms (cax-*) ---
    /// SubClassOf: x rdf:type C1, C1 ⊑ C2 → x rdf:type C2
    CaxSco,
    /// EquivalentClass propagation
    CaxEqc1,
    CaxEqc2,
    /// DisjointWith consistency check
    CaxDw,
    /// DisjointUnion consistency check
    CaxAdc,

    // --- Table 8: RDFS rules included in RL ---
    /// rdfs:subClassOf transitivity (same as ScmSco)
    RdfsSubClassTransitivity,
    /// rdfs:subPropertyOf propagation
    RdfsSubPropertyPropagation,
    /// rdfs:domain type inference
    RdfsDomainInference,
    /// rdfs:range type inference
    RdfsRangeInference,

    // --- Table 9: owl:sameAs rules (eq-*) ---
    /// Reflexivity of owl:sameAs
    EqRef,
    /// Symmetry of owl:sameAs
    EqSym,
    /// Transitivity of owl:sameAs
    EqTrans,
    /// Type propagation via sameAs
    EqRep1,
    EqRep2,
    EqRep3,
}

/// Report of an inference run
#[derive(Debug, Clone)]
pub struct InferenceReport {
    /// Number of fixpoint iterations performed
    pub iterations: usize,
    /// Total new triples inferred
    pub new_triples_count: usize,
    /// Per-rule firing counts
    pub rules_fired: HashMap<Owl2RlRule, usize>,
    /// Wall-clock duration
    pub duration: std::time::Duration,
    /// Inconsistencies detected (if any)
    pub inconsistencies: Vec<String>,
}

/// Pattern element - either a concrete value or a variable name
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternElem {
    Var(String),
    Const(String),
}

impl PatternElem {
    pub(crate) fn var(s: &str) -> Self {
        Self::Var(s.to_string())
    }
    pub(crate) fn konst(s: &str) -> Self {
        Self::Const(s.to_string())
    }
}

/// A triple pattern using PatternElem
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: PatternElem,
    pub predicate: PatternElem,
    pub object: PatternElem,
}

impl TriplePattern {
    pub fn new(s: PatternElem, p: PatternElem, o: PatternElem) -> Self {
        Self {
            subject: s,
            predicate: p,
            object: o,
        }
    }
}

/// A compiled RL rule: antecedent patterns + consequent pattern
#[derive(Debug, Clone)]
pub(crate) struct CompiledRule {
    pub(crate) id: Owl2RlRule,
    pub(crate) antecedents: Vec<TriplePattern>,
    pub(crate) consequent: TriplePattern,
}

// RDF/OWL URI constants used in patterns
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
pub const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
pub const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
pub const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
pub const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
pub const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
pub const OWL_SAME_AS: &str = "http://www.w3.org/2002/07/owl#sameAs";
pub const OWL_EQUIVALENT_CLASS: &str = "http://www.w3.org/2002/07/owl#equivalentClass";
pub const OWL_EQUIVALENT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#equivalentProperty";
pub const OWL_INVERSE_OF: &str = "http://www.w3.org/2002/07/owl#inverseOf";
pub const OWL_SYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#SymmetricProperty";
pub const OWL_TRANSITIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#TransitiveProperty";
pub const OWL_FUNCTIONAL_PROPERTY: &str = "http://www.w3.org/2002/07/owl#FunctionalProperty";
pub const OWL_INV_FUNCTIONAL_PROPERTY: &str =
    "http://www.w3.org/2002/07/owl#InverseFunctionalProperty";
pub const OWL_ASYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AsymmetricProperty";
pub const OWL_IRREFLEXIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#IrreflexiveProperty";
pub const OWL_DISJOINT_WITH: &str = "http://www.w3.org/2002/07/owl#disjointWith";
pub const OWL_NOTHING: &str = "http://www.w3.org/2002/07/owl#Nothing";
pub const OWL_SOME_VALUES_FROM: &str = "http://www.w3.org/2002/07/owl#someValuesFrom";
pub const OWL_ALL_VALUES_FROM: &str = "http://www.w3.org/2002/07/owl#allValuesFrom";
pub const OWL_ON_PROPERTY: &str = "http://www.w3.org/2002/07/owl#onProperty";
pub const OWL_HAS_VALUE: &str = "http://www.w3.org/2002/07/owl#hasValue";
pub const OWL_INTERSECTION_OF: &str = "http://www.w3.org/2002/07/owl#intersectionOf";
pub const OWL_UNION_OF: &str = "http://www.w3.org/2002/07/owl#unionOf";
pub const OWL_THING: &str = "http://www.w3.org/2002/07/owl#Thing";

/// Re-exported URI constants for external use
pub mod vocab {
    pub use super::OWL_ALL_VALUES_FROM;
    pub use super::OWL_DISJOINT_WITH;
    pub use super::OWL_EQUIVALENT_CLASS;
    pub use super::OWL_EQUIVALENT_PROPERTY;
    pub use super::OWL_FUNCTIONAL_PROPERTY;
    pub use super::OWL_HAS_VALUE;
    pub use super::OWL_INVERSE_OF;
    pub use super::OWL_INV_FUNCTIONAL_PROPERTY;
    pub use super::OWL_NOTHING;
    pub use super::OWL_ON_PROPERTY;
    pub use super::OWL_SAME_AS;
    pub use super::OWL_SOME_VALUES_FROM;
    pub use super::OWL_SYMMETRIC_PROPERTY;
    pub use super::OWL_THING;
    pub use super::OWL_TRANSITIVE_PROPERTY;
    pub use super::RDFS_DOMAIN;
    pub use super::RDFS_RANGE;
    pub use super::RDFS_SUBCLASS_OF;
    pub use super::RDFS_SUBPROPERTY_OF;
    pub use super::RDF_TYPE;
}
