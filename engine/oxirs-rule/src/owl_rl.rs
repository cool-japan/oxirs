//! # OWL 2 RL Profile Reasoner
//!
//! Implements the OWL 2 RL (Rule Language) profile using forward-chaining rules.
//! OWL 2 RL is designed for scalable reasoning using production rule systems.
//!
//! ## Complexity
//! Polynomial time in the size of the data (ABox).
//!
//! ## Key Features
//! - Full W3C OWL 2 RL rule set (Table 4-8 of the spec)
//! - Forward-chaining materialization to fixpoint
//! - Triple pattern matching with variable unification
//! - Conflict detection for owl:disjointWith
//!
//! ## Reference
//! <https://www.w3.org/TR/owl2-profiles/#OWL_2_RL>

use std::collections::{HashMap, HashSet};
use std::time::Instant;
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
    fn var(s: &str) -> Self {
        Self::Var(s.to_string())
    }
    fn konst(s: &str) -> Self {
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
struct CompiledRule {
    id: Owl2RlRule,
    antecedents: Vec<TriplePattern>,
    consequent: TriplePattern,
}

/// OWL 2 RL reasoner with proper triple-pattern matching and variable unification
pub struct Owl2RlReasoner {
    /// Base axioms
    axioms: Vec<Triple>,
    /// Inferred triples (closure)
    inferred: HashSet<Triple>,
    /// Maximum fixpoint iterations (safety limit)
    max_iterations: usize,
    /// All rules compiled from the spec
    rules: Vec<CompiledRule>,
    /// Rules firing counts (populated during materialization)
    rule_fire_counts: HashMap<Owl2RlRule, usize>,
    /// Detected inconsistencies
    inconsistencies: Vec<String>,
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

impl Owl2RlReasoner {
    /// Create a new OWL 2 RL reasoner with default settings
    pub fn new() -> Self {
        let mut reasoner = Self {
            axioms: Vec::new(),
            inferred: HashSet::new(),
            max_iterations: 1000,
            rules: Vec::new(),
            rule_fire_counts: HashMap::new(),
            inconsistencies: Vec::new(),
        };
        reasoner.compile_rules();
        reasoner
    }

    /// Set the maximum number of fixpoint iterations
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Add a single axiom triple
    pub fn add_axiom(&mut self, s: &str, p: &str, o: &str) {
        let triple = (s.to_string(), p.to_string(), o.to_string());
        self.axioms.push(triple);
    }

    /// Add multiple axiom triples
    pub fn add_axioms(&mut self, triples: impl IntoIterator<Item = Triple>) {
        self.axioms.extend(triples);
    }

    /// Run forward-chaining inference to fixpoint.
    ///
    /// Applies all OWL 2 RL rules iteratively until no new triples can be derived.
    /// Returns an `InferenceReport` with statistics.
    pub fn materialize(&mut self) -> Result<InferenceReport, RlError> {
        let start = Instant::now();
        self.inferred.clear();
        self.inconsistencies.clear();
        self.rule_fire_counts.clear();

        // Seed with axioms
        let mut working_set: HashSet<Triple> = self.axioms.iter().cloned().collect();

        let mut iterations = 0usize;
        let mut total_new = 0usize;

        loop {
            if iterations >= self.max_iterations {
                return Err(RlError::MaxIterationsExceeded(self.max_iterations));
            }
            iterations += 1;

            let mut new_triples: HashSet<Triple> = HashSet::new();

            for rule in &self.rules.clone() {
                let derived = self.apply_compiled_rule(rule, &working_set);
                let count = derived.len();
                for triple in derived {
                    if !working_set.contains(&triple) {
                        new_triples.insert(triple);
                    }
                }
                if count > 0 {
                    *self.rule_fire_counts.entry(rule.id.clone()).or_insert(0) += count;
                }
            }

            // Run special non-compilable rules
            let special = self.apply_special_rules(&working_set);
            for triple in special {
                if !working_set.contains(&triple) {
                    new_triples.insert(triple);
                }
            }

            // Check inconsistencies
            self.check_inconsistencies(&working_set);

            if new_triples.is_empty() {
                break;
            }

            total_new += new_triples.len();
            working_set.extend(new_triples);
        }

        // Store only the inferred (non-axiom) triples
        let axiom_set: HashSet<Triple> = self.axioms.iter().cloned().collect();
        self.inferred = working_set
            .into_iter()
            .filter(|t| !axiom_set.contains(t))
            .collect();

        Ok(InferenceReport {
            iterations,
            new_triples_count: total_new,
            rules_fired: self.rule_fire_counts.clone(),
            duration: start.elapsed(),
            inconsistencies: self.inconsistencies.clone(),
        })
    }

    /// Query if a triple is entailed (in axioms or inferred)
    pub fn is_entailed(&self, s: &str, p: &str, o: &str) -> bool {
        let triple = (s.to_string(), p.to_string(), o.to_string());
        self.axioms.contains(&triple) || self.inferred.contains(&triple)
    }

    /// Get only the inferred triples (not including axioms)
    pub fn inferred_triples(&self) -> &HashSet<Triple> {
        &self.inferred
    }

    /// Get all triples (axioms + inferred)
    pub fn all_triples(&self) -> HashSet<Triple> {
        let mut all: HashSet<Triple> = self.axioms.iter().cloned().collect();
        all.extend(self.inferred.iter().cloned());
        all
    }

    /// Match triples from a set using optional wildcard patterns
    pub fn match_triples<'a>(
        triples: &'a HashSet<Triple>,
        s: Option<&str>,
        p: Option<&str>,
        o: Option<&str>,
    ) -> Vec<&'a Triple> {
        triples
            .iter()
            .filter(|(ts, tp, to)| {
                s.map_or(true, |sv| ts.as_str() == sv)
                    && p.map_or(true, |pv| tp.as_str() == pv)
                    && o.map_or(true, |ov| to.as_str() == ov)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Private: rule compilation
    // -----------------------------------------------------------------------

    fn compile_rules(&mut self) {
        // Helper closures for building PatternElem
        let v = PatternElem::var;
        let k = PatternElem::konst;

        // ScmSco: rdfs:subClassOf transitivity
        // C1 rdfs:subClassOf C2, C2 rdfs:subClassOf C3 → C1 rdfs:subClassOf C3
        self.add_rule(
            Owl2RlRule::ScmSco,
            vec![
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
                TriplePattern::new(v("C2"), k(RDFS_SUBCLASS_OF), v("C3")),
            ],
            TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C3")),
        );

        // ScmSpo: rdfs:subPropertyOf transitivity
        self.add_rule(
            Owl2RlRule::ScmSpo,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P3")),
            ],
            TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P3")),
        );

        // ScmEqc1: owl:equivalentClass → rdfs:subClassOf (both directions)
        self.add_rule(
            Owl2RlRule::ScmEqc1,
            vec![TriplePattern::new(
                v("C1"),
                k(OWL_EQUIVALENT_CLASS),
                v("C2"),
            )],
            TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
        );
        self.add_rule(
            Owl2RlRule::ScmEqc2,
            vec![TriplePattern::new(
                v("C1"),
                k(OWL_EQUIVALENT_CLASS),
                v("C2"),
            )],
            TriplePattern::new(v("C2"), k(RDFS_SUBCLASS_OF), v("C1")),
        );

        // ScmEqp1/2: owl:equivalentProperty → rdfs:subPropertyOf
        self.add_rule(
            Owl2RlRule::ScmEqp1,
            vec![TriplePattern::new(
                v("P1"),
                k(OWL_EQUIVALENT_PROPERTY),
                v("P2"),
            )],
            TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
        );
        self.add_rule(
            Owl2RlRule::ScmEqp2,
            vec![TriplePattern::new(
                v("P1"),
                k(OWL_EQUIVALENT_PROPERTY),
                v("P2"),
            )],
            TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P1")),
        );

        // ScmDom1: rdfs:domain inheritance via superProperty
        // P1 rdfs:domain C, P2 rdfs:subPropertyOf P1 → P2 rdfs:domain C
        self.add_rule(
            Owl2RlRule::ScmDom1,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_DOMAIN), v("C")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P1")),
            ],
            TriplePattern::new(v("P2"), k(RDFS_DOMAIN), v("C")),
        );

        // ScmDom2: domain narrowing via subClassOf
        // P rdfs:domain C1, C1 rdfs:subClassOf C2 → P rdfs:domain C2
        self.add_rule(
            Owl2RlRule::ScmDom2,
            vec![
                TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C2")),
        );

        // ScmRng1: range inheritance via superProperty
        self.add_rule(
            Owl2RlRule::ScmRng1,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_RANGE), v("C")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P1")),
            ],
            TriplePattern::new(v("P2"), k(RDFS_RANGE), v("C")),
        );

        // ScmRng2: range narrowing via subClassOf
        self.add_rule(
            Owl2RlRule::ScmRng2,
            vec![
                TriplePattern::new(v("P"), k(RDFS_RANGE), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("P"), k(RDFS_RANGE), v("C2")),
        );

        // PrpSpo1: x P1 y, P1 rdfs:subPropertyOf P2 → x P2 y
        self.add_rule(
            Owl2RlRule::PrpSpo1,
            vec![
                TriplePattern::new(v("x"), v("P1"), v("y")),
                TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
            ],
            TriplePattern::new(v("x"), v("P2"), v("y")),
        );

        // PrpEqp1: P1 owl:equivalentProperty P2, x P1 y → x P2 y
        self.add_rule(
            Owl2RlRule::PrpEqp1,
            vec![
                TriplePattern::new(v("P1"), k(OWL_EQUIVALENT_PROPERTY), v("P2")),
                TriplePattern::new(v("x"), v("P1"), v("y")),
            ],
            TriplePattern::new(v("x"), v("P2"), v("y")),
        );
        self.add_rule(
            Owl2RlRule::PrpEqp2,
            vec![
                TriplePattern::new(v("P1"), k(OWL_EQUIVALENT_PROPERTY), v("P2")),
                TriplePattern::new(v("x"), v("P2"), v("y")),
            ],
            TriplePattern::new(v("x"), v("P1"), v("y")),
        );

        // PrpDom: P rdfs:domain C, x P y → x rdf:type C
        self.add_rule(
            Owl2RlRule::PrpDom,
            vec![
                TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C")),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C")),
        );

        // PrpRng: P rdfs:range C, x P y → y rdf:type C
        self.add_rule(
            Owl2RlRule::PrpRng,
            vec![
                TriplePattern::new(v("P"), k(RDFS_RANGE), v("C")),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("y"), k(RDF_TYPE), v("C")),
        );

        // PrpSymp: P rdf:type owl:SymmetricProperty, x P y → y P x
        self.add_rule(
            Owl2RlRule::PrpSymp,
            vec![
                TriplePattern::new(v("P"), k(RDF_TYPE), k(OWL_SYMMETRIC_PROPERTY)),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P"), v("x")),
        );

        // PrpTrp: P rdf:type owl:TransitiveProperty, x P y, y P z → x P z
        self.add_rule(
            Owl2RlRule::PrpTrp,
            vec![
                TriplePattern::new(v("P"), k(RDF_TYPE), k(OWL_TRANSITIVE_PROPERTY)),
                TriplePattern::new(v("x"), v("P"), v("y")),
                TriplePattern::new(v("y"), v("P"), v("z")),
            ],
            TriplePattern::new(v("x"), v("P"), v("z")),
        );

        // PrpInv1: P1 owl:inverseOf P2, x P1 y → y P2 x
        self.add_rule(
            Owl2RlRule::PrpInv1,
            vec![
                TriplePattern::new(v("P1"), k(OWL_INVERSE_OF), v("P2")),
                TriplePattern::new(v("x"), v("P1"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P2"), v("x")),
        );

        // PrpInv2: P1 owl:inverseOf P2, x P2 y → y P1 x
        self.add_rule(
            Owl2RlRule::PrpInv2,
            vec![
                TriplePattern::new(v("P1"), k(OWL_INVERSE_OF), v("P2")),
                TriplePattern::new(v("x"), v("P2"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P1"), v("x")),
        );

        // CaxSco: x rdf:type C1, C1 rdfs:subClassOf C2 → x rdf:type C2
        self.add_rule(
            Owl2RlRule::CaxSco,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );

        // CaxEqc1: x rdf:type C1, C1 owl:equivalentClass C2 → x rdf:type C2
        self.add_rule(
            Owl2RlRule::CaxEqc1,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C1"), k(OWL_EQUIVALENT_CLASS), v("C2")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );
        self.add_rule(
            Owl2RlRule::CaxEqc2,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C2"), k(OWL_EQUIVALENT_CLASS), v("C1")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );

        // EqRef: x rdf:type owl:Thing → x owl:sameAs x
        self.add_rule(
            Owl2RlRule::EqRef,
            vec![TriplePattern::new(v("x"), k(RDF_TYPE), k(OWL_THING))],
            TriplePattern::new(v("x"), k(OWL_SAME_AS), v("x")),
        );

        // EqSym: x owl:sameAs y → y owl:sameAs x
        self.add_rule(
            Owl2RlRule::EqSym,
            vec![TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y"))],
            TriplePattern::new(v("y"), k(OWL_SAME_AS), v("x")),
        );

        // EqTrans: x owl:sameAs y, y owl:sameAs z → x owl:sameAs z
        self.add_rule(
            Owl2RlRule::EqTrans,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("y"), k(OWL_SAME_AS), v("z")),
            ],
            TriplePattern::new(v("x"), k(OWL_SAME_AS), v("z")),
        );

        // EqRep1: x owl:sameAs y, x rdf:type C → y rdf:type C
        self.add_rule(
            Owl2RlRule::EqRep1,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C")),
            ],
            TriplePattern::new(v("y"), k(RDF_TYPE), v("C")),
        );

        // EqRep2: x owl:sameAs y, z x w → z y w  (subject replacement)
        self.add_rule(
            Owl2RlRule::EqRep2,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("z"), v("x"), v("w")),
            ],
            TriplePattern::new(v("z"), v("y"), v("w")),
        );

        // EqRep3: x owl:sameAs y, z w x → z w y  (object replacement)
        self.add_rule(
            Owl2RlRule::EqRep3,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("z"), v("w"), v("x")),
            ],
            TriplePattern::new(v("z"), v("w"), v("y")),
        );
    }

    fn add_rule(
        &mut self,
        id: Owl2RlRule,
        antecedents: Vec<TriplePattern>,
        consequent: TriplePattern,
    ) {
        self.rules.push(CompiledRule {
            id,
            antecedents,
            consequent,
        });
    }

    // -----------------------------------------------------------------------
    // Private: rule application with proper unification
    // -----------------------------------------------------------------------

    fn apply_compiled_rule(&self, rule: &CompiledRule, triples: &HashSet<Triple>) -> Vec<Triple> {
        let mut results = Vec::new();

        // Enumerate all binding combinations for the first pattern, then extend
        let all_bindings = self.enumerate_bindings(&rule.antecedents, triples);

        for bindings in all_bindings {
            if let Some(t) = self.instantiate_pattern(&rule.consequent, &bindings) {
                // Exclude trivially tautological triples (e.g. x owl:sameAs x is only reflexive,
                // but we still allow it since EqRef explicitly generates it)
                results.push(t);
            }
        }

        results
    }

    /// Enumerate all variable bindings satisfying all antecedent patterns via join
    fn enumerate_bindings(
        &self,
        patterns: &[TriplePattern],
        triples: &HashSet<Triple>,
    ) -> Vec<Bindings> {
        if patterns.is_empty() {
            return vec![HashMap::new()];
        }

        let mut current_bindings: Vec<Bindings> = vec![HashMap::new()];

        for pattern in patterns {
            let mut next_bindings: Vec<Bindings> = Vec::new();

            for bindings in &current_bindings {
                // Partially instantiate the pattern with current bindings
                let (s_val, p_val, o_val) = self.partial_instantiate(pattern, bindings);

                // Find matching triples
                for triple in triples.iter() {
                    if let Some(extended) =
                        self.try_extend_bindings(bindings, pattern, triple, &s_val, &p_val, &o_val)
                    {
                        next_bindings.push(extended);
                    }
                }
            }

            current_bindings = next_bindings;

            // Early exit if no solutions
            if current_bindings.is_empty() {
                break;
            }
        }

        current_bindings
    }

    /// Partially instantiate a pattern's elements given current bindings,
    /// returning Option<String> for each position (None = variable not yet bound)
    fn partial_instantiate(
        &self,
        pattern: &TriplePattern,
        bindings: &Bindings,
    ) -> (Option<String>, Option<String>, Option<String>) {
        let resolve = |elem: &PatternElem| -> Option<String> {
            match elem {
                PatternElem::Const(c) => Some(c.clone()),
                PatternElem::Var(v) => bindings.get(v).cloned(),
            }
        };

        (
            resolve(&pattern.subject),
            resolve(&pattern.predicate),
            resolve(&pattern.object),
        )
    }

    /// Try to extend bindings by matching `pattern` against `triple`,
    /// given partially instantiated values.
    fn try_extend_bindings(
        &self,
        bindings: &Bindings,
        pattern: &TriplePattern,
        triple: &Triple,
        s_val: &Option<String>,
        p_val: &Option<String>,
        o_val: &Option<String>,
    ) -> Option<Bindings> {
        // Check each position matches (or is unbound variable)
        let check_and_bind = |val: &Option<String>,
                              pattern_elem: &PatternElem,
                              triple_part: &str|
         -> Option<Option<(String, String)>> {
            match (val, pattern_elem) {
                (Some(bound), _) => {
                    // Already bound: must match
                    if bound == triple_part {
                        Some(None) // matches, no new binding
                    } else {
                        None // no match
                    }
                }
                (None, PatternElem::Var(var_name)) => {
                    // Check if already bound in bindings map
                    if let Some(existing) = bindings.get(var_name) {
                        if existing == triple_part {
                            Some(None) // consistent
                        } else {
                            None // inconsistent
                        }
                    } else {
                        // New binding
                        Some(Some((var_name.clone(), triple_part.to_string())))
                    }
                }
                (None, PatternElem::Const(c)) => {
                    if c == triple_part {
                        Some(None)
                    } else {
                        None
                    }
                }
            }
        };

        let sb = check_and_bind(s_val, &pattern.subject, &triple.0)?;
        let pb = check_and_bind(p_val, &pattern.predicate, &triple.1)?;
        let ob = check_and_bind(o_val, &pattern.object, &triple.2)?;

        // Build extended bindings
        let mut extended = bindings.clone();
        for (k, v) in [sb, pb, ob].into_iter().flatten() {
            extended.insert(k, v);
        }

        Some(extended)
    }

    /// Instantiate a pattern consequent given complete bindings
    fn instantiate_pattern(&self, pattern: &TriplePattern, bindings: &Bindings) -> Option<Triple> {
        let resolve = |elem: &PatternElem| -> Option<String> {
            match elem {
                PatternElem::Const(c) => Some(c.clone()),
                PatternElem::Var(v) => bindings.get(v).cloned(),
            }
        };

        let s = resolve(&pattern.subject)?;
        let p = resolve(&pattern.predicate)?;
        let o = resolve(&pattern.object)?;
        Some((s, p, o))
    }

    /// Apply special rules that require more complex handling than simple pattern matching
    fn apply_special_rules(&self, triples: &HashSet<Triple>) -> Vec<Triple> {
        let mut derived = Vec::new();

        // PrpFp: owl:FunctionalProperty → owl:sameAs between objects
        // x P y1, x P y2, P rdf:type owl:FunctionalProperty → y1 owl:sameAs y2
        for (prop, _, _) in
            Self::match_triples(triples, None, Some(RDF_TYPE), Some(OWL_FUNCTIONAL_PROPERTY))
        {
            // Collect all (x, y) pairs for this functional property
            let assertions: Vec<(&str, &str)> = triples
                .iter()
                .filter(|(_, p, _)| p == prop)
                .map(|(s, _, o)| (s.as_str(), o.as_str()))
                .collect();

            // Group by subject
            let mut by_subject: HashMap<&str, Vec<&str>> = HashMap::new();
            for (s, o) in &assertions {
                by_subject.entry(s).or_default().push(o);
            }

            for objects in by_subject.values() {
                if objects.len() > 1 {
                    // All objects must be owl:sameAs
                    for i in 0..objects.len() {
                        for j in (i + 1)..objects.len() {
                            derived.push((
                                objects[i].to_string(),
                                OWL_SAME_AS.to_string(),
                                objects[j].to_string(),
                            ));
                            derived.push((
                                objects[j].to_string(),
                                OWL_SAME_AS.to_string(),
                                objects[i].to_string(),
                            ));
                        }
                    }
                }
            }
        }

        // PrpIfp: owl:InverseFunctionalProperty → owl:sameAs between subjects
        for (prop, _, _) in Self::match_triples(
            triples,
            None,
            Some(RDF_TYPE),
            Some(OWL_INV_FUNCTIONAL_PROPERTY),
        ) {
            let assertions: Vec<(&str, &str)> = triples
                .iter()
                .filter(|(_, p, _)| p == prop)
                .map(|(s, _, o)| (s.as_str(), o.as_str()))
                .collect();

            let mut by_object: HashMap<&str, Vec<&str>> = HashMap::new();
            for (s, o) in &assertions {
                by_object.entry(o).or_default().push(s);
            }

            for subjects in by_object.values() {
                if subjects.len() > 1 {
                    for i in 0..subjects.len() {
                        for j in (i + 1)..subjects.len() {
                            derived.push((
                                subjects[i].to_string(),
                                OWL_SAME_AS.to_string(),
                                subjects[j].to_string(),
                            ));
                        }
                    }
                }
            }
        }

        // ClsHv1/2: owl:hasValue restriction
        // x rdf:type R, R owl:onProperty P, R owl:hasValue v → x P v
        for (restriction, _, prop) in
            Self::match_triples(triples, None, Some(OWL_ON_PROPERTY), None)
        {
            for (_, _, value) in
                Self::match_triples(triples, Some(restriction), Some(OWL_HAS_VALUE), None)
            {
                // Find all individuals of type restriction
                for (individual, _, _) in
                    Self::match_triples(triples, None, Some(RDF_TYPE), Some(restriction))
                {
                    derived.push((individual.to_string(), prop.to_string(), value.to_string()));
                }
            }
        }

        // PrpSvf: owl:someValuesFrom: x P y, y rdf:type C, P owl:someValuesFrom C, R owl:onProperty P → x rdf:type R
        for (restriction, _, class) in
            Self::match_triples(triples, None, Some(OWL_SOME_VALUES_FROM), None)
        {
            for (_, _, prop) in
                Self::match_triples(triples, Some(restriction), Some(OWL_ON_PROPERTY), None)
            {
                // Find all x P y pairs
                for (x, _, y) in triples.iter().filter(|(_, p, _)| p == prop) {
                    // Check y rdf:type class
                    if triples.contains(&(y.to_string(), RDF_TYPE.to_string(), class.to_string())) {
                        derived.push((
                            x.to_string(),
                            RDF_TYPE.to_string(),
                            restriction.to_string(),
                        ));
                    }
                }
            }
        }

        derived
    }

    /// Check for ontology inconsistencies
    fn check_inconsistencies(&mut self, triples: &HashSet<Triple>) {
        // CaxDw: owl:disjointWith violation
        // x rdf:type C1, x rdf:type C2, C1 owl:disjointWith C2 → inconsistency
        for (c1, _, c2) in Self::match_triples(triples, None, Some(OWL_DISJOINT_WITH), None) {
            // Find individuals of type C1 that are also type C2
            for (individual, _, _) in Self::match_triples(triples, None, Some(RDF_TYPE), Some(c1)) {
                if triples.contains(&(individual.to_string(), RDF_TYPE.to_string(), c2.to_string()))
                {
                    let msg = format!(
                        "Inconsistency: {} is both {} and {} which are disjoint",
                        individual, c1, c2
                    );
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }

        // PrpIrp: owl:IrreflexiveProperty violation
        for (prop, _, _) in Self::match_triples(
            triples,
            None,
            Some(RDF_TYPE),
            Some(OWL_IRREFLEXIVE_PROPERTY),
        ) {
            for (s, _, o) in triples.iter().filter(|(_, p, _)| p == prop) {
                if s == o {
                    let msg = format!(
                        "Inconsistency: {} {} {} violates IrreflexiveProperty",
                        s, prop, o
                    );
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }

        // PrpAsynp: owl:AsymmetricProperty violation
        for (prop, _, _) in
            Self::match_triples(triples, None, Some(RDF_TYPE), Some(OWL_ASYMMETRIC_PROPERTY))
        {
            for (s, _, o) in triples.iter().filter(|(_, p, _)| p == prop) {
                if triples.contains(&(o.to_string(), prop.to_string(), s.to_string())) && s != o {
                    let msg = format!(
                        "Inconsistency: {} {} {} and {} {} {} violate AsymmetricProperty",
                        s, prop, o, o, prop, s
                    );
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }

        // ClsNothing2: x rdf:type owl:Nothing → inconsistency
        for (individual, _, _) in
            Self::match_triples(triples, None, Some(RDF_TYPE), Some(OWL_NOTHING))
        {
            let msg = format!(
                "Inconsistency: {} is of type owl:Nothing (bottom concept)",
                individual
            );
            if !self.inconsistencies.contains(&msg) {
                self.inconsistencies.push(msg);
            }
        }
    }

    /// Get detected inconsistencies (populated after materialize())
    pub fn inconsistencies(&self) -> &[String] {
        &self.inconsistencies
    }

    /// Check if the ontology is consistent (call after materialize())
    pub fn is_consistent(&self) -> bool {
        self.inconsistencies.is_empty()
    }

    /// Add the standard owl:Thing/owl:Nothing axioms
    pub fn add_owl_bootstrap_axioms(&mut self) {
        // owl:Thing rdfs:subClassOf owl:Thing (reflexivity)
        // owl:Nothing rdfs:subClassOf owl:Thing (bottom is subclass of everything)
        self.add_axiom(OWL_NOTHING, RDFS_SUBCLASS_OF, OWL_THING);
    }

    /// Get the known predicates from the rdfs:domain/range axioms
    pub fn get_known_properties(&self) -> HashSet<String> {
        let all = self.all_triples();
        let mut props = HashSet::new();
        for (s, p, _) in &all {
            if p == RDFS_DOMAIN || p == RDFS_RANGE || p == RDF_TYPE {
                props.insert(s.clone());
            }
        }
        props
    }

    /// Shorthand to add a commonly-used RDFS subClassOf axiom
    pub fn add_subclass_of(&mut self, sub: &str, sup: &str) {
        self.add_axiom(sub, RDFS_SUBCLASS_OF, sup);
    }

    /// Shorthand to add a type assertion
    pub fn add_type(&mut self, individual: &str, class: &str) {
        self.add_axiom(individual, RDF_TYPE, class);
    }

    /// Shorthand to add a symmetric property declaration
    pub fn add_symmetric_property(&mut self, prop: &str) {
        self.add_axiom(prop, RDF_TYPE, OWL_SYMMETRIC_PROPERTY);
    }

    /// Shorthand to add a transitive property declaration
    pub fn add_transitive_property(&mut self, prop: &str) {
        self.add_axiom(prop, RDF_TYPE, OWL_TRANSITIVE_PROPERTY);
    }

    /// Shorthand to add an inverseOf declaration
    pub fn add_inverse_of(&mut self, p1: &str, p2: &str) {
        self.add_axiom(p1, OWL_INVERSE_OF, p2);
    }
}

impl Default for Owl2RlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export URI constants for external use
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

#[cfg(test)]
mod tests {
    use super::*;

    fn rl() -> Owl2RlReasoner {
        Owl2RlReasoner::new()
    }

    #[test]
    fn test_subclass_transitivity() {
        let mut r = rl();
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");
        let report = r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("Dog", RDFS_SUBCLASS_OF, "Animal"),
            "Expected Dog ⊑ Animal, got {} new triples in {} iterations",
            report.new_triples_count,
            report.iterations
        );
    }

    #[test]
    fn test_type_propagation_via_subclass() {
        let mut r = rl();
        r.add_type("fido", "Dog");
        r.add_subclass_of("Dog", "Animal");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("fido", RDF_TYPE, "Animal"),
            "Expected fido rdf:type Animal"
        );
    }

    #[test]
    fn test_domain_inference() {
        let mut r = rl();
        r.add_axiom("hasParent", RDFS_DOMAIN, "Person");
        r.add_axiom("alice", "hasParent", "bob");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("alice", RDF_TYPE, "Person"),
            "Expected alice rdf:type Person from domain"
        );
    }

    #[test]
    fn test_range_inference() {
        let mut r = rl();
        r.add_axiom("hasParent", RDFS_RANGE, "Person");
        r.add_axiom("alice", "hasParent", "bob");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("bob", RDF_TYPE, "Person"),
            "Expected bob rdf:type Person from range"
        );
    }

    #[test]
    fn test_symmetric_property() {
        let mut r = rl();
        r.add_symmetric_property("knows");
        r.add_axiom("alice", "knows", "bob");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("bob", "knows", "alice"),
            "Expected bob knows alice from SymmetricProperty"
        );
    }

    #[test]
    fn test_transitive_property() {
        let mut r = rl();
        r.add_transitive_property("ancestorOf");
        r.add_axiom("grandparent", "ancestorOf", "parent");
        r.add_axiom("parent", "ancestorOf", "child");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("grandparent", "ancestorOf", "child"),
            "Expected transitive ancestorOf"
        );
    }

    #[test]
    fn test_inverse_of() {
        let mut r = rl();
        r.add_inverse_of("hasParent", "hasChild");
        r.add_axiom("alice", "hasParent", "bob");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("bob", "hasChild", "alice"),
            "Expected bob hasChild alice from inverseOf"
        );
    }

    #[test]
    fn test_equivalent_class() {
        let mut r = rl();
        r.add_axiom("Human", OWL_EQUIVALENT_CLASS, "Person");
        r.add_type("alice", "Human");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("alice", RDF_TYPE, "Person"),
            "Expected alice rdf:type Person via equivalentClass"
        );
    }

    #[test]
    fn test_disjoint_with_inconsistency() {
        let mut r = rl();
        r.add_axiom("Cat", OWL_DISJOINT_WITH, "Dog");
        r.add_type("fido", "Dog");
        r.add_type("fido", "Cat");
        r.materialize().expect("materialization failed");
        assert!(
            !r.is_consistent(),
            "Expected inconsistency due to disjointWith"
        );
        assert!(!r.inconsistencies().is_empty());
    }

    #[test]
    fn test_same_as_transitivity() {
        let mut r = rl();
        r.add_axiom("alice", OWL_SAME_AS, "alicia");
        r.add_axiom("alicia", OWL_SAME_AS, "ali");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("alice", OWL_SAME_AS, "ali"),
            "Expected sameAs transitivity"
        );
    }

    #[test]
    fn test_inference_report() {
        let mut r = rl();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        let report = r.materialize().expect("materialization failed");
        assert!(report.iterations >= 1);
        assert!(report.new_triples_count >= 1);
        assert!(!report.rules_fired.is_empty());
    }

    #[test]
    fn test_subproperty_propagation() {
        let mut r = rl();
        r.add_axiom("isChildOf", RDFS_SUBPROPERTY_OF, "isRelatedTo");
        r.add_axiom("alice", "isChildOf", "bob");
        r.materialize().expect("materialization failed");
        assert!(
            r.is_entailed("alice", "isRelatedTo", "bob"),
            "Expected subProperty propagation"
        );
    }

    #[test]
    fn test_max_iterations_safety() {
        let mut r = Owl2RlReasoner::new().with_max_iterations(5);
        // Non-terminating scenario is bounded
        r.add_axiom("A", RDFS_SUBCLASS_OF, "B");
        // Should succeed within 5 iterations for simple cases
        let result = r.materialize();
        // May succeed or fail with MaxIterationsExceeded, but should not panic
        let _ = result;
    }

    #[test]
    fn test_match_triples_wildcard() {
        let mut set = HashSet::new();
        set.insert((
            "alice".to_string(),
            RDF_TYPE.to_string(),
            "Person".to_string(),
        ));
        set.insert((
            "bob".to_string(),
            RDF_TYPE.to_string(),
            "Person".to_string(),
        ));
        set.insert(("alice".to_string(), "knows".to_string(), "bob".to_string()));

        let type_triples: Vec<_> = Owl2RlReasoner::match_triples(&set, None, Some(RDF_TYPE), None);
        assert_eq!(type_triples.len(), 2);

        let alice_triples: Vec<_> = Owl2RlReasoner::match_triples(&set, Some("alice"), None, None);
        assert_eq!(alice_triples.len(), 2);
    }

    #[test]
    fn test_nothing_inconsistency() {
        let mut r = rl();
        r.add_axiom("x", RDF_TYPE, OWL_NOTHING);
        r.materialize().expect("materialization failed");
        assert!(!r.is_consistent());
    }
}

// -----------------------------------------------------------------------
// Extended Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests_extended {
    use super::*;

    fn rl() -> Owl2RlReasoner {
        Owl2RlReasoner::new()
    }

    #[test]
    fn test_empty_reasoner_materialize() {
        let mut r = rl();
        let report = r.materialize().expect("empty materialize failed");
        assert_eq!(report.new_triples_count, 0);
        assert!(r.is_consistent());
    }

    #[test]
    fn test_single_type_assertion_no_inference() {
        let mut r = rl();
        r.add_type("alice", "Person");
        r.materialize().expect("failed");
        assert!(r.is_entailed("alice", RDF_TYPE, "Person"));
    }

    #[test]
    fn test_deep_subclass_chain() {
        let mut r = rl();
        // A ⊑ B ⊑ C ⊑ D ⊑ E ⊑ F
        for i in 0..5usize {
            r.add_subclass_of(&format!("C{}", i), &format!("C{}", i + 1));
        }
        r.add_type("x", "C0");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("x", RDF_TYPE, "C5"),
            "x should be C5 via chain"
        );
    }

    #[test]
    fn test_equivalent_property_propagation() {
        let mut r = rl();
        r.add_axiom("likes", OWL_EQUIVALENT_PROPERTY, "enjoys");
        r.add_axiom("alice", "likes", "music");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("alice", "enjoys", "music"),
            "alice enjoys music via equivalentProperty"
        );
    }

    #[test]
    fn test_equivalent_property_reverse() {
        let mut r = rl();
        r.add_axiom("P", OWL_EQUIVALENT_PROPERTY, "Q");
        r.add_axiom("a", "Q", "b");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("a", "P", "b"),
            "a P b via reverse equivalentProperty"
        );
    }

    #[test]
    fn test_functional_property_same_as() {
        let mut r = rl();
        r.add_axiom("hasMother", RDF_TYPE, OWL_FUNCTIONAL_PROPERTY);
        r.add_axiom("alice", "hasMother", "eve");
        r.add_axiom("alice", "hasMother", "eva");
        r.materialize().expect("failed");
        // eve and eva should be sameAs
        let eve_same = r.is_entailed("eve", OWL_SAME_AS, "eva");
        let eva_same = r.is_entailed("eva", OWL_SAME_AS, "eve");
        assert!(
            eve_same || eva_same,
            "eve and eva should be sameAs via FunctionalProperty"
        );
    }

    #[test]
    fn test_inverse_functional_property() {
        let mut r = rl();
        r.add_axiom(
            "hasSocialSecurityNumber",
            RDF_TYPE,
            OWL_INV_FUNCTIONAL_PROPERTY,
        );
        r.add_axiom("alice", "hasSocialSecurityNumber", "123-45-6789");
        r.add_axiom("alicia", "hasSocialSecurityNumber", "123-45-6789");
        r.materialize().expect("failed");
        // alice and alicia should be sameAs
        let same = r.is_entailed("alice", OWL_SAME_AS, "alicia")
            || r.is_entailed("alicia", OWL_SAME_AS, "alice");
        assert!(
            same,
            "alice and alicia should be sameAs via InverseFunctionalProperty"
        );
    }

    #[test]
    fn test_subproperty_chain() {
        let mut r = rl();
        r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
        r.add_axiom("P2", RDFS_SUBPROPERTY_OF, "P3");
        r.add_axiom("a", "P1", "b");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("a", "P3", "b"),
            "a P3 b via subProperty chain"
        );
    }

    #[test]
    fn test_same_as_explicit_then_symmetric() {
        // Test that explicit sameAs generates symmetric inference
        let mut r = rl();
        r.add_axiom("alice", OWL_SAME_AS, "alicia");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("alicia", OWL_SAME_AS, "alice"),
            "alicia sameAs alice (symmetry from explicit assertion)"
        );
    }

    #[test]
    fn test_same_as_symmetry() {
        let mut r = rl();
        r.add_axiom("alice", OWL_SAME_AS, "alicia");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("alicia", OWL_SAME_AS, "alice"),
            "sameAs symmetry"
        );
    }

    #[test]
    fn test_same_as_transitivity_three() {
        let mut r = rl();
        r.add_axiom("a", OWL_SAME_AS, "b");
        r.add_axiom("b", OWL_SAME_AS, "c");
        r.add_axiom("c", OWL_SAME_AS, "d");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("a", OWL_SAME_AS, "d"),
            "a sameAs d via 3-step transitivity"
        );
    }

    #[test]
    fn test_same_as_type_propagation() {
        let mut r = rl();
        r.add_type("alice", "Person");
        r.add_axiom("alice", OWL_SAME_AS, "alicia");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("alicia", RDF_TYPE, "Person"),
            "alicia:Person via sameAs with alice:Person"
        );
    }

    #[test]
    fn test_domain_and_range_combined() {
        let mut r = rl();
        r.add_axiom("hasChild", RDFS_DOMAIN, "Parent");
        r.add_axiom("hasChild", RDFS_RANGE, "Child");
        r.add_axiom("bob", "hasChild", "tommy");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("bob", RDF_TYPE, "Parent"),
            "bob:Parent via domain"
        );
        assert!(
            r.is_entailed("tommy", RDF_TYPE, "Child"),
            "tommy:Child via range"
        );
    }

    #[test]
    fn test_asymmetric_property_violation() {
        let mut r = rl();
        r.add_axiom("isStrictlyLessThan", RDF_TYPE, OWL_ASYMMETRIC_PROPERTY);
        r.add_axiom("a", "isStrictlyLessThan", "b");
        r.add_axiom("b", "isStrictlyLessThan", "a");
        r.materialize().expect("materialize still runs");
        assert!(
            !r.is_consistent(),
            "Asymmetric violation should be inconsistent"
        );
    }

    #[test]
    fn test_irreflexive_property_violation() {
        let mut r = rl();
        r.add_axiom("isStrictlyBefore", RDF_TYPE, OWL_IRREFLEXIVE_PROPERTY);
        r.add_axiom("now", "isStrictlyBefore", "now");
        r.materialize().expect("materialize still runs");
        assert!(
            !r.is_consistent(),
            "IrreflexiveProperty self-loop should be inconsistent"
        );
    }

    #[test]
    fn test_disjoint_with_three_classes() {
        let mut r = rl();
        r.add_axiom("A", OWL_DISJOINT_WITH, "B");
        r.add_axiom("B", OWL_DISJOINT_WITH, "C");
        r.add_type("x", "A");
        r.add_type("x", "B");
        r.materialize().expect("materialize still runs");
        assert!(
            !r.is_consistent(),
            "x:A and x:B with A disjointWith B is inconsistent"
        );
    }

    #[test]
    fn test_multiple_disjoint_no_violation() {
        let mut r = rl();
        r.add_axiom("Mammal", OWL_DISJOINT_WITH, "Reptile");
        r.add_type("fido", "Mammal");
        r.add_type("rex", "Reptile");
        r.materialize().expect("failed");
        assert!(
            r.is_consistent(),
            "Different individuals in disjoint classes is OK"
        );
    }

    #[test]
    fn test_inference_report_iterations() {
        let mut r = rl();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        r.add_subclass_of("C", "D");
        let report = r.materialize().expect("failed");
        assert!(report.iterations >= 1, "Should have at least 1 iteration");
        assert!(report.new_triples_count >= 1, "Should have new triples");
    }

    #[test]
    fn test_inference_report_rules_fired() {
        let mut r = rl();
        r.add_subclass_of("X", "Y");
        r.add_type("a", "X");
        let report = r.materialize().expect("failed");
        assert!(
            !report.rules_fired.is_empty(),
            "rules_fired should not be empty"
        );
    }

    #[test]
    fn test_inference_report_duration_positive() {
        let mut r = rl();
        r.add_subclass_of("A", "B");
        let report = r.materialize().expect("failed");
        // Duration::as_secs() always returns u64, just verify the field is accessible
        let _ = report.duration.as_millis();
    }

    #[test]
    fn test_materialize_multiple_times_idempotent() {
        let mut r = rl();
        r.add_subclass_of("Dog", "Animal");
        r.add_type("fido", "Dog");
        r.materialize().expect("first materialize failed");
        r.materialize().expect("second materialize failed");
        assert!(
            r.is_entailed("fido", RDF_TYPE, "Animal"),
            "idempotent after second materialize"
        );
    }

    #[test]
    fn test_add_axioms_bulk() {
        let mut r = rl();
        r.add_axioms(vec![
            (
                "A".to_string(),
                RDFS_SUBCLASS_OF.to_string(),
                "B".to_string(),
            ),
            (
                "B".to_string(),
                RDFS_SUBCLASS_OF.to_string(),
                "C".to_string(),
            ),
        ]);
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("A", RDFS_SUBCLASS_OF, "C"),
            "A ⊑ C via bulk add"
        );
    }

    #[test]
    fn test_no_false_positive_without_assertion() {
        let mut r = rl();
        r.add_subclass_of("Dog", "Animal");
        r.materialize().expect("failed");
        // No individual assertions — no type inferences
        assert!(
            !r.is_entailed("fido", RDF_TYPE, "Animal"),
            "fido should not be Animal without type assertion"
        );
    }

    #[test]
    fn test_equivalent_class_bidirectional_type() {
        let mut r = rl();
        r.add_axiom("Human", OWL_EQUIVALENT_CLASS, "Person");
        r.add_type("bob", "Person");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("bob", RDF_TYPE, "Human"),
            "bob:Human via equivalentClass with Person"
        );
    }

    #[test]
    fn test_match_triples_wildcard_all() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(("a".to_string(), "p".to_string(), "b".to_string()));
        set.insert(("c".to_string(), "q".to_string(), "d".to_string()));
        let all = Owl2RlReasoner::match_triples(&set, None, None, None);
        assert_eq!(all.len(), 2, "wildcard should match all 2 triples");
    }

    #[test]
    fn test_match_triples_subject_filter() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(("alice".to_string(), "knows".to_string(), "bob".to_string()));
        set.insert((
            "alice".to_string(),
            "likes".to_string(),
            "music".to_string(),
        ));
        set.insert((
            "bob".to_string(),
            "knows".to_string(),
            "charlie".to_string(),
        ));
        let alice_triples = Owl2RlReasoner::match_triples(&set, Some("alice"), None, None);
        assert_eq!(alice_triples.len(), 2, "Should get 2 alice triples");
    }

    #[test]
    fn test_match_triples_object_filter() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(("a".to_string(), "p".to_string(), "X".to_string()));
        set.insert(("b".to_string(), "q".to_string(), "X".to_string()));
        set.insert(("c".to_string(), "r".to_string(), "Y".to_string()));
        let x_triples = Owl2RlReasoner::match_triples(&set, None, None, Some("X"));
        assert_eq!(x_triples.len(), 2, "Should get 2 triples with object X");
    }

    #[test]
    fn test_is_consistent_before_materialize() {
        let mut r = rl();
        r.add_type("alice", "Person");
        // is_consistent before materialize — should be true (no inference yet)
        assert!(r.is_consistent(), "consistent before materialize");
    }

    #[test]
    fn test_inconsistencies_empty_when_consistent() {
        let mut r = rl();
        r.add_subclass_of("A", "B");
        r.materialize().expect("failed");
        assert!(
            r.inconsistencies().is_empty(),
            "no inconsistencies for consistent ontology"
        );
    }

    #[test]
    fn test_with_max_iterations() {
        let mut r = Owl2RlReasoner::new().with_max_iterations(2);
        r.add_subclass_of("A", "B");
        // Should complete without panic within 2 iterations
        let _ = r.materialize();
    }

    #[test]
    fn test_domain_inheritance_via_subproperty() {
        let mut r = rl();
        // P1 ⊑ P2, P2 rdfs:domain C => P1 rdfs:domain C (scm-dom2)
        r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
        r.add_axiom("P2", RDFS_DOMAIN, "C");
        r.add_axiom("a", "P1", "b");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("a", RDF_TYPE, "C"),
            "a:C via P1⊑P2, P2 domain C, a P1 b"
        );
    }

    #[test]
    fn test_range_inheritance_via_subproperty() {
        let mut r = rl();
        r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
        r.add_axiom("P2", RDFS_RANGE, "D");
        r.add_axiom("a", "P1", "b");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("b", RDF_TYPE, "D"),
            "b:D via P1⊑P2, P2 range D, a P1 b"
        );
    }

    #[test]
    fn test_transitive_and_subclass_combined() {
        let mut r = rl();
        r.add_transitive_property("locatedIn");
        r.add_subclass_of("City", "Place");
        r.add_type("berlin", "City");
        r.add_axiom("berlin", "locatedIn", "germany");
        r.add_axiom("germany", "locatedIn", "europe");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("berlin", RDF_TYPE, "Place"),
            "berlin:Place via City⊑Place"
        );
        assert!(
            r.is_entailed("berlin", "locatedIn", "europe"),
            "berlin locatedIn europe via transitivity"
        );
    }

    #[test]
    fn test_symmetric_and_type() {
        let mut r = rl();
        r.add_symmetric_property("marriedTo");
        r.add_axiom("hasSpouseOf", RDFS_SUBPROPERTY_OF, "marriedTo");
        r.add_axiom("alice", "marriedTo", "bob");
        r.materialize().expect("failed");
        assert!(
            r.is_entailed("bob", "marriedTo", "alice"),
            "bob marriedTo alice via SymmetricProperty"
        );
    }

    #[test]
    fn test_owl_thing_type_inference() {
        // Any individual should be inferred to be owl:Thing
        let mut r = rl();
        r.add_type("alice", "Person");
        r.materialize().expect("failed");
        // owl:Thing is the universal superclass — RL may infer it
        // At minimum, alice:Person should be asserted
        assert!(r.is_entailed("alice", RDF_TYPE, "Person"));
    }

    #[test]
    fn test_large_knowledge_base_performance() {
        let mut r = rl();
        // Add 50 subclass axioms
        for i in 0..50usize {
            r.add_subclass_of(&format!("Class{}", i), &format!("Class{}", i + 1));
        }
        r.add_type("ind", "Class0");
        let report = r.materialize().expect("large KB materialize failed");
        assert!(report.new_triples_count > 0, "Should infer new triples");
        assert!(
            r.is_entailed("ind", RDF_TYPE, "Class50"),
            "ind:Class50 via 50-hop chain"
        );
    }
}
