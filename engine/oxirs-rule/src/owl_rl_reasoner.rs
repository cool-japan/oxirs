//! # OWL 2 RL Reasoning Engine
//!
//! Forward-chaining OWL 2 RL reasoner: rule compilation, triple-pattern matching,
//! variable unification, fixpoint materialization, and inconsistency detection.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::owl_rl_rules::{
    Bindings, CompiledRule, InferenceReport, Owl2RlRule, PatternElem, RlError, Triple,
    TriplePattern, OWL_ASYMMETRIC_PROPERTY, OWL_DISJOINT_WITH, OWL_EQUIVALENT_CLASS,
    OWL_EQUIVALENT_PROPERTY, OWL_FUNCTIONAL_PROPERTY, OWL_HAS_VALUE, OWL_INVERSE_OF,
    OWL_INV_FUNCTIONAL_PROPERTY, OWL_IRREFLEXIVE_PROPERTY, OWL_NOTHING, OWL_ON_PROPERTY,
    OWL_SAME_AS, OWL_SOME_VALUES_FROM, OWL_SYMMETRIC_PROPERTY, OWL_THING, OWL_TRANSITIVE_PROPERTY,
    RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF, RDFS_SUBPROPERTY_OF, RDF_TYPE,
};

/// OWL 2 RL reasoner with proper triple-pattern matching and variable unification
pub struct Owl2RlReasoner {
    /// Base axioms
    pub(crate) axioms: Vec<Triple>,
    /// Inferred triples (closure)
    pub(crate) inferred: HashSet<Triple>,
    /// Maximum fixpoint iterations (safety limit)
    pub(crate) max_iterations: usize,
    /// All rules compiled from the spec
    pub(crate) rules: Vec<CompiledRule>,
    /// Rules firing counts (populated during materialization)
    pub(crate) rule_fire_counts: HashMap<Owl2RlRule, usize>,
    /// Detected inconsistencies
    pub(crate) inconsistencies: Vec<String>,
}

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

    // -----------------------------------------------------------------------
    // Private: rule compilation
    // -----------------------------------------------------------------------

    fn compile_rules(&mut self) {
        // Helper closures for building PatternElem
        let v = PatternElem::var;
        let k = PatternElem::konst;

        // ScmSco: rdfs:subClassOf transitivity
        // C1 rdfs:subClassOf C2, C2 rdfs:subClassOf C3 → C1 rdfs:subClassOf C3
        self.add_compiled_rule(
            Owl2RlRule::ScmSco,
            vec![
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
                TriplePattern::new(v("C2"), k(RDFS_SUBCLASS_OF), v("C3")),
            ],
            TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C3")),
        );

        // ScmSpo: rdfs:subPropertyOf transitivity
        self.add_compiled_rule(
            Owl2RlRule::ScmSpo,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P3")),
            ],
            TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P3")),
        );

        // ScmEqc1: owl:equivalentClass → rdfs:subClassOf (both directions)
        self.add_compiled_rule(
            Owl2RlRule::ScmEqc1,
            vec![TriplePattern::new(
                v("C1"),
                k(OWL_EQUIVALENT_CLASS),
                v("C2"),
            )],
            TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
        );
        self.add_compiled_rule(
            Owl2RlRule::ScmEqc2,
            vec![TriplePattern::new(
                v("C1"),
                k(OWL_EQUIVALENT_CLASS),
                v("C2"),
            )],
            TriplePattern::new(v("C2"), k(RDFS_SUBCLASS_OF), v("C1")),
        );

        // ScmEqp1/2: owl:equivalentProperty → rdfs:subPropertyOf
        self.add_compiled_rule(
            Owl2RlRule::ScmEqp1,
            vec![TriplePattern::new(
                v("P1"),
                k(OWL_EQUIVALENT_PROPERTY),
                v("P2"),
            )],
            TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
        );
        self.add_compiled_rule(
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
        self.add_compiled_rule(
            Owl2RlRule::ScmDom1,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_DOMAIN), v("C")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P1")),
            ],
            TriplePattern::new(v("P2"), k(RDFS_DOMAIN), v("C")),
        );

        // ScmDom2: domain narrowing via subClassOf
        // P rdfs:domain C1, C1 rdfs:subClassOf C2 → P rdfs:domain C2
        self.add_compiled_rule(
            Owl2RlRule::ScmDom2,
            vec![
                TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C2")),
        );

        // ScmRng1: range inheritance via superProperty
        self.add_compiled_rule(
            Owl2RlRule::ScmRng1,
            vec![
                TriplePattern::new(v("P1"), k(RDFS_RANGE), v("C")),
                TriplePattern::new(v("P2"), k(RDFS_SUBPROPERTY_OF), v("P1")),
            ],
            TriplePattern::new(v("P2"), k(RDFS_RANGE), v("C")),
        );

        // ScmRng2: range narrowing via subClassOf
        self.add_compiled_rule(
            Owl2RlRule::ScmRng2,
            vec![
                TriplePattern::new(v("P"), k(RDFS_RANGE), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("P"), k(RDFS_RANGE), v("C2")),
        );

        // PrpSpo1: x P1 y, P1 rdfs:subPropertyOf P2 → x P2 y
        self.add_compiled_rule(
            Owl2RlRule::PrpSpo1,
            vec![
                TriplePattern::new(v("x"), v("P1"), v("y")),
                TriplePattern::new(v("P1"), k(RDFS_SUBPROPERTY_OF), v("P2")),
            ],
            TriplePattern::new(v("x"), v("P2"), v("y")),
        );

        // PrpEqp1: P1 owl:equivalentProperty P2, x P1 y → x P2 y
        self.add_compiled_rule(
            Owl2RlRule::PrpEqp1,
            vec![
                TriplePattern::new(v("P1"), k(OWL_EQUIVALENT_PROPERTY), v("P2")),
                TriplePattern::new(v("x"), v("P1"), v("y")),
            ],
            TriplePattern::new(v("x"), v("P2"), v("y")),
        );
        self.add_compiled_rule(
            Owl2RlRule::PrpEqp2,
            vec![
                TriplePattern::new(v("P1"), k(OWL_EQUIVALENT_PROPERTY), v("P2")),
                TriplePattern::new(v("x"), v("P2"), v("y")),
            ],
            TriplePattern::new(v("x"), v("P1"), v("y")),
        );

        // PrpDom: P rdfs:domain C, x P y → x rdf:type C
        self.add_compiled_rule(
            Owl2RlRule::PrpDom,
            vec![
                TriplePattern::new(v("P"), k(RDFS_DOMAIN), v("C")),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C")),
        );

        // PrpRng: P rdfs:range C, x P y → y rdf:type C
        self.add_compiled_rule(
            Owl2RlRule::PrpRng,
            vec![
                TriplePattern::new(v("P"), k(RDFS_RANGE), v("C")),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("y"), k(RDF_TYPE), v("C")),
        );

        // PrpSymp: P rdf:type owl:SymmetricProperty, x P y → y P x
        self.add_compiled_rule(
            Owl2RlRule::PrpSymp,
            vec![
                TriplePattern::new(v("P"), k(RDF_TYPE), k(OWL_SYMMETRIC_PROPERTY)),
                TriplePattern::new(v("x"), v("P"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P"), v("x")),
        );

        // PrpTrp: P rdf:type owl:TransitiveProperty, x P y, y P z → x P z
        self.add_compiled_rule(
            Owl2RlRule::PrpTrp,
            vec![
                TriplePattern::new(v("P"), k(RDF_TYPE), k(OWL_TRANSITIVE_PROPERTY)),
                TriplePattern::new(v("x"), v("P"), v("y")),
                TriplePattern::new(v("y"), v("P"), v("z")),
            ],
            TriplePattern::new(v("x"), v("P"), v("z")),
        );

        // PrpInv1: P1 owl:inverseOf P2, x P1 y → y P2 x
        self.add_compiled_rule(
            Owl2RlRule::PrpInv1,
            vec![
                TriplePattern::new(v("P1"), k(OWL_INVERSE_OF), v("P2")),
                TriplePattern::new(v("x"), v("P1"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P2"), v("x")),
        );

        // PrpInv2: P1 owl:inverseOf P2, x P2 y → y P1 x
        self.add_compiled_rule(
            Owl2RlRule::PrpInv2,
            vec![
                TriplePattern::new(v("P1"), k(OWL_INVERSE_OF), v("P2")),
                TriplePattern::new(v("x"), v("P2"), v("y")),
            ],
            TriplePattern::new(v("y"), v("P1"), v("x")),
        );

        // CaxSco: x rdf:type C1, C1 rdfs:subClassOf C2 → x rdf:type C2
        self.add_compiled_rule(
            Owl2RlRule::CaxSco,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C1"), k(RDFS_SUBCLASS_OF), v("C2")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );

        // CaxEqc1: x rdf:type C1, C1 owl:equivalentClass C2 → x rdf:type C2
        self.add_compiled_rule(
            Owl2RlRule::CaxEqc1,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C1"), k(OWL_EQUIVALENT_CLASS), v("C2")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );
        self.add_compiled_rule(
            Owl2RlRule::CaxEqc2,
            vec![
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C1")),
                TriplePattern::new(v("C2"), k(OWL_EQUIVALENT_CLASS), v("C1")),
            ],
            TriplePattern::new(v("x"), k(RDF_TYPE), v("C2")),
        );

        // EqRef: x rdf:type owl:Thing → x owl:sameAs x
        self.add_compiled_rule(
            Owl2RlRule::EqRef,
            vec![TriplePattern::new(v("x"), k(RDF_TYPE), k(OWL_THING))],
            TriplePattern::new(v("x"), k(OWL_SAME_AS), v("x")),
        );

        // EqSym: x owl:sameAs y → y owl:sameAs x
        self.add_compiled_rule(
            Owl2RlRule::EqSym,
            vec![TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y"))],
            TriplePattern::new(v("y"), k(OWL_SAME_AS), v("x")),
        );

        // EqTrans: x owl:sameAs y, y owl:sameAs z → x owl:sameAs z
        self.add_compiled_rule(
            Owl2RlRule::EqTrans,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("y"), k(OWL_SAME_AS), v("z")),
            ],
            TriplePattern::new(v("x"), k(OWL_SAME_AS), v("z")),
        );

        // EqRep1: x owl:sameAs y, x rdf:type C → y rdf:type C
        self.add_compiled_rule(
            Owl2RlRule::EqRep1,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("x"), k(RDF_TYPE), v("C")),
            ],
            TriplePattern::new(v("y"), k(RDF_TYPE), v("C")),
        );

        // EqRep2: x owl:sameAs y, z x w → z y w  (subject replacement)
        self.add_compiled_rule(
            Owl2RlRule::EqRep2,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("z"), v("x"), v("w")),
            ],
            TriplePattern::new(v("z"), v("y"), v("w")),
        );

        // EqRep3: x owl:sameAs y, z w x → z w y  (object replacement)
        self.add_compiled_rule(
            Owl2RlRule::EqRep3,
            vec![
                TriplePattern::new(v("x"), k(OWL_SAME_AS), v("y")),
                TriplePattern::new(v("z"), v("w"), v("x")),
            ],
            TriplePattern::new(v("z"), v("w"), v("y")),
        );
    }

    fn add_compiled_rule(
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
}

impl Default for Owl2RlReasoner {
    fn default() -> Self {
        Self::new()
    }
}
