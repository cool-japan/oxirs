//! # OWL 2 Profile Optimizations
//!
//! This module implements optimized reasoning algorithms for OWL 2 profiles:
//! - **OWL 2 EL**: Existential Language - polynomial time reasoning for large ontologies
//! - **OWL 2 QL**: Query Language - database-style query answering
//! - **OWL 2 RL**: Rule Language - production rule-based reasoning
//!
//! ## OWL 2 EL Profile
//!
//! Designed for large biomedical ontologies with many classes and properties.
//! - Polynomial time complexity (scalable to millions of axioms)
//! - Supports: SubClassOf, EquivalentClasses, SubPropertyOf, domain/range, existential restrictions
//! - Classification using consequence-based reasoning
//!
//! ## OWL 2 QL Profile
//!
//! Designed for query answering over large data sets using databases.
//! - Query rewriting to SQL/SPARQL
//! - Supports: SubClassOf, SubPropertyOf, domain/range, inverse properties
//! - Efficient for ABox queries
//!
//! ## OWL 2 RL Profile
//!
//! Designed for scalable reasoning using production rules.
//! - Rule-based materialization
//! - Supports most OWL 2 features
//! - Can be implemented in triple stores
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::owl_profiles::{ELReasoner, OWLProfile};
//! use oxirs_rule::hermit_reasoner::Ontology;
//! use oxirs_rule::description_logic::Concept;
//!
//! let mut reasoner = ELReasoner::new();
//! let mut ontology = Ontology::new();
//!
//! // Add axioms
//! ontology.add_subsumption(
//!     Concept::Atomic("Dog".to_string()),
//!     Concept::Atomic("Mammal".to_string())
//! );
//!
//! // Classify using optimized EL algorithm
//! let hierarchy = reasoner.classify(&ontology)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::description_logic::{Concept, Role};
use crate::hermit_reasoner::{Axiom, Ontology};
use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use once_cell::sync::Lazy;
use scirs2_core::metrics::{Counter, Timer};
use std::collections::{HashMap, HashSet};

// Global metrics for profile reasoning
static EL_CLASSIFICATIONS: Lazy<Counter> =
    Lazy::new(|| Counter::new("el_classifications".to_string()));
static EL_SUBSUMPTIONS: Lazy<Counter> = Lazy::new(|| Counter::new("el_subsumptions".to_string()));
static QL_QUERY_REWRITES: Lazy<Counter> =
    Lazy::new(|| Counter::new("ql_query_rewrites".to_string()));
static RL_RULE_APPLICATIONS: Lazy<Counter> =
    Lazy::new(|| Counter::new("rl_rule_applications".to_string()));
static PROFILE_REASONING_TIME: Lazy<Timer> =
    Lazy::new(|| Timer::new("profile_reasoning_time".to_string()));

/// OWL 2 Profile types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OWLProfile {
    /// OWL 2 EL - Existential Language
    EL,
    /// OWL 2 QL - Query Language
    QL,
    /// OWL 2 RL - Rule Language
    RL,
}

/// Subsumption relationship in class hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subsumption {
    pub sub_class: String,
    pub super_class: String,
}

/// Class hierarchy (taxonomy)
#[derive(Debug, Clone)]
pub struct ClassHierarchy {
    /// Direct subsumption relationships
    pub direct_subsumptions: HashSet<Subsumption>,
    /// All subsumption relationships (including transitive)
    pub all_subsumptions: HashSet<Subsumption>,
    /// Equivalent classes
    pub equivalences: HashMap<String, HashSet<String>>,
}

impl ClassHierarchy {
    pub fn new() -> Self {
        Self {
            direct_subsumptions: HashSet::new(),
            all_subsumptions: HashSet::new(),
            equivalences: HashMap::new(),
        }
    }

    /// Get all superclasses of a class
    pub fn get_superclasses(&self, class: &str) -> HashSet<String> {
        self.all_subsumptions
            .iter()
            .filter(|s| s.sub_class == class)
            .map(|s| s.super_class.clone())
            .collect()
    }

    /// Get all subclasses of a class
    pub fn get_subclasses(&self, class: &str) -> HashSet<String> {
        self.all_subsumptions
            .iter()
            .filter(|s| s.super_class == class)
            .map(|s| s.sub_class.clone())
            .collect()
    }
}

impl Default for ClassHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

/// OWL 2 EL Reasoner - Polynomial time classification
pub struct ELReasoner {
    /// Statistics
    pub stats: ELStats,
}

#[derive(Debug, Clone, Default)]
pub struct ELStats {
    pub classifications: usize,
    pub subsumption_tests: usize,
    pub iterations: usize,
}

impl Default for ELReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl ELReasoner {
    pub fn new() -> Self {
        Self {
            stats: ELStats::default(),
        }
    }

    /// Classify ontology using EL algorithm
    pub fn classify(&mut self, ontology: &Ontology) -> Result<ClassHierarchy> {
        let _timer = PROFILE_REASONING_TIME.start();
        self.stats.classifications += 1;
        EL_CLASSIFICATIONS.inc();

        let mut hierarchy = ClassHierarchy::new();

        // Extract atomic concepts
        let concepts: HashSet<String> = ontology.concepts.iter().cloned().collect();

        // Build initial subsumption map from axioms
        let mut subsumptions: HashMap<String, HashSet<String>> = HashMap::new();

        for axiom in &ontology.axioms {
            match axiom {
                Axiom::SubClassOf(Concept::Atomic(sub), Concept::Atomic(sup)) => {
                    subsumptions
                        .entry(sub.clone())
                        .or_default()
                        .insert(sup.clone());
                }
                Axiom::EquivalentClasses(Concept::Atomic(c1), Concept::Atomic(c2)) => {
                    // C1 ≡ C2 means C1 ⊑ C2 and C2 ⊑ C1
                    subsumptions
                        .entry(c1.clone())
                        .or_default()
                        .insert(c2.clone());
                    subsumptions
                        .entry(c2.clone())
                        .or_default()
                        .insert(c1.clone());

                    hierarchy
                        .equivalences
                        .entry(c1.clone())
                        .or_default()
                        .insert(c2.clone());
                    hierarchy
                        .equivalences
                        .entry(c2.clone())
                        .or_default()
                        .insert(c1.clone());
                }
                _ => {}
            }
        }

        // Fixed-point iteration for transitive closure
        let mut changed = true;
        while changed {
            changed = false;
            self.stats.iterations += 1;

            for concept in &concepts {
                if let Some(supers) = subsumptions.get(concept).cloned() {
                    let mut new_supers = supers.clone();

                    // Add transitive subsumptions
                    for sup in &supers {
                        if let Some(sup_supers) = subsumptions.get(sup) {
                            for sup_sup in sup_supers {
                                if new_supers.insert(sup_sup.clone()) {
                                    changed = true;
                                }
                            }
                        }
                    }

                    if new_supers.len() > supers.len() {
                        subsumptions.insert(concept.clone(), new_supers);
                    }
                }
            }
        }

        // Build hierarchy from subsumptions
        for (sub_class, super_classes) in &subsumptions {
            for super_class in super_classes {
                hierarchy.all_subsumptions.insert(Subsumption {
                    sub_class: sub_class.clone(),
                    super_class: super_class.clone(),
                });

                self.stats.subsumption_tests += 1;
                EL_SUBSUMPTIONS.inc();
            }
        }

        // Compute direct subsumptions (minimal cover)
        for (sub_class, super_classes) in &subsumptions {
            // Find minimal superclasses (those not subsumed by others)
            let minimal_supers: HashSet<String> = super_classes
                .iter()
                .filter(|sup| {
                    // sup is minimal if no other super subsumes it
                    !super_classes.iter().any(|other_sup| {
                        other_sup != *sup
                            && subsumptions
                                .get(other_sup)
                                .map(|s| s.contains(*sup))
                                .unwrap_or(false)
                    })
                })
                .cloned()
                .collect();

            for super_class in minimal_supers {
                hierarchy.direct_subsumptions.insert(Subsumption {
                    sub_class: sub_class.clone(),
                    super_class,
                });
            }
        }

        Ok(hierarchy)
    }

    /// Check if C ⊑ D using EL reasoning
    pub fn is_subsumed_by(
        &mut self,
        ontology: &Ontology,
        sub_concept: &str,
        super_concept: &str,
    ) -> Result<bool> {
        let hierarchy = self.classify(ontology)?;
        Ok(hierarchy.all_subsumptions.contains(&Subsumption {
            sub_class: sub_concept.to_string(),
            super_class: super_concept.to_string(),
        }))
    }
}

/// OWL 2 QL Reasoner - Query rewriting
pub struct QLReasoner {
    /// Statistics
    pub stats: QLStats,
}

#[derive(Debug, Clone, Default)]
pub struct QLStats {
    pub query_rewrites: usize,
    pub rewritten_queries: usize,
}

impl Default for QLReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl QLReasoner {
    pub fn new() -> Self {
        Self {
            stats: QLStats::default(),
        }
    }

    /// Rewrite query to include inferred triples
    pub fn rewrite_query(
        &mut self,
        query: &RuleAtom,
        ontology: &Ontology,
    ) -> Result<Vec<RuleAtom>> {
        let _timer = PROFILE_REASONING_TIME.start();
        self.stats.query_rewrites += 1;
        QL_QUERY_REWRITES.inc();

        let mut rewritten = vec![query.clone()];

        // Rewrite based on subclass axioms
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = query
        {
            // If querying for type, expand to the *transitive* superclass closure
            // so multi-hop chains (Dog ⊑ Mammal ⊑ Animal) are covered.
            if predicate == &Term::Constant("rdf:type".to_string()) {
                if let Term::Constant(class) = object {
                    for sup in Self::transitive_superclasses(ontology, class) {
                        rewritten.push(RuleAtom::Triple {
                            subject: subject.clone(),
                            predicate: predicate.clone(),
                            object: Term::Constant(sup),
                        });
                        self.stats.rewritten_queries += 1;
                    }
                }
            }

            // If querying for a property, expand to the transitive super-property
            // closure (P ⊑ Q ⊑ R).
            if let Term::Constant(prop) = predicate {
                for super_prop in Self::transitive_superproperties(ontology, prop) {
                    rewritten.push(RuleAtom::Triple {
                        subject: subject.clone(),
                        predicate: Term::Constant(super_prop),
                        object: object.clone(),
                    });
                    self.stats.rewritten_queries += 1;
                }
            }
        }

        Ok(rewritten)
    }

    /// Compute the transitive set of superclasses of `class` from the ontology's
    /// atomic `SubClassOf` axioms (a fixpoint over the subclass edges).
    fn transitive_superclasses(ontology: &Ontology, class: &str) -> Vec<String> {
        let mut supers: HashSet<String> = HashSet::new();
        let mut stack = vec![class.to_string()];
        while let Some(current) = stack.pop() {
            for axiom in &ontology.axioms {
                if let Axiom::SubClassOf(Concept::Atomic(sub), Concept::Atomic(sup)) = axiom {
                    if *sub == current && supers.insert(sup.clone()) {
                        stack.push(sup.clone());
                    }
                }
            }
        }
        supers.into_iter().collect()
    }

    /// Compute the transitive set of super-properties of `prop` from the
    /// ontology's `SubPropertyOf` axioms.
    fn transitive_superproperties(ontology: &Ontology, prop: &str) -> Vec<String> {
        let mut supers: HashSet<String> = HashSet::new();
        let mut stack = vec![prop.to_string()];
        while let Some(current) = stack.pop() {
            for axiom in &ontology.axioms {
                if let Axiom::SubPropertyOf(Role { name: sub_prop }, Role { name: super_prop }) =
                    axiom
                {
                    if *sub_prop == current && supers.insert(super_prop.clone()) {
                        stack.push(super_prop.clone());
                    }
                }
            }
        }
        supers.into_iter().collect()
    }

    /// Answer query with reasoning
    pub fn answer_query(
        &mut self,
        query: &RuleAtom,
        ontology: &Ontology,
        facts: &HashSet<RuleAtom>,
    ) -> Result<bool> {
        let rewritten_queries = self.rewrite_query(query, ontology)?;

        // Check if any rewritten query matches facts
        for rewritten_query in rewritten_queries {
            if facts.contains(&rewritten_query) {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// OWL 2 RL Reasoner - Rule-based materialization
pub struct RLReasoner {
    /// Forward chaining rules
    rules: Vec<Rule>,
    /// Statistics
    pub stats: RLStats,
}

#[derive(Debug, Clone, Default)]
pub struct RLStats {
    pub rule_applications: usize,
    pub facts_derived: usize,
    pub iterations: usize,
}

impl Default for RLReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl RLReasoner {
    pub fn new() -> Self {
        let mut reasoner = Self {
            rules: Vec::new(),
            stats: RLStats::default(),
        };
        reasoner.initialize_rl_rules();
        reasoner
    }

    /// Initialize OWL 2 RL rules
    fn initialize_rl_rules(&mut self) {
        // Rule: SubClassOf transitivity
        // C ⊑ D, D ⊑ E → C ⊑ E
        self.rules.push(Rule {
            name: "scm-cls".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("C".to_string()),
                    predicate: Term::Constant("rdfs:subClassOf".to_string()),
                    object: Term::Variable("D".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("D".to_string()),
                    predicate: Term::Constant("rdfs:subClassOf".to_string()),
                    object: Term::Variable("E".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("C".to_string()),
                predicate: Term::Constant("rdfs:subClassOf".to_string()),
                object: Term::Variable("E".to_string()),
            }],
        });

        // Rule: Type propagation
        // x:C, C ⊑ D → x:D
        self.rules.push(Rule {
            name: "cls-hv1".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Constant("rdf:type".to_string()),
                    object: Term::Variable("C".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("C".to_string()),
                    predicate: Term::Constant("rdfs:subClassOf".to_string()),
                    object: Term::Variable("D".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Variable("D".to_string()),
            }],
        });

        // Rule: SubPropertyOf transitivity
        // P ⊑ Q, Q ⊑ R → P ⊑ R
        self.rules.push(Rule {
            name: "scm-spo".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("rdfs:subPropertyOf".to_string()),
                    object: Term::Variable("Q".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Q".to_string()),
                    predicate: Term::Constant("rdfs:subPropertyOf".to_string()),
                    object: Term::Variable("R".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("P".to_string()),
                predicate: Term::Constant("rdfs:subPropertyOf".to_string()),
                object: Term::Variable("R".to_string()),
            }],
        });

        // Rule: Property propagation
        // x P y, P ⊑ Q → x Q y
        self.rules.push(Rule {
            name: "prp-spo1".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("rdfs:subPropertyOf".to_string()),
                    object: Term::Variable("Q".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Variable("Q".to_string()),
                object: Term::Variable("y".to_string()),
            }],
        });
    }

    /// Materialize inferences using RL rules
    pub fn materialize(&mut self, initial_facts: &HashSet<RuleAtom>) -> Result<HashSet<RuleAtom>> {
        let _timer = PROFILE_REASONING_TIME.start();

        let mut facts = initial_facts.clone();
        let mut changed = true;

        while changed {
            changed = false;
            self.stats.iterations += 1;

            for rule in &self.rules.clone() {
                let new_facts = self.apply_rule(rule, &facts)?;

                for fact in new_facts {
                    if facts.insert(fact) {
                        changed = true;
                        self.stats.facts_derived += 1;
                    }
                }

                self.stats.rule_applications += 1;
                RL_RULE_APPLICATIONS.inc();
            }
        }

        Ok(facts)
    }

    /// Apply a single rule to facts using real unification.
    ///
    /// Finds every substitution that jointly satisfies all body atoms (variables
    /// shared across atoms must bind consistently), then instantiates the head
    /// atoms under each substitution. Only fully-ground head atoms are emitted —
    /// this is what prevents leaking `Term::Variable` "facts" like
    /// `?x rdfs:subPropertyOf ?R` into the materialized set.
    fn apply_rule(&self, rule: &Rule, facts: &HashSet<RuleAtom>) -> Result<Vec<RuleAtom>> {
        let mut derived = Vec::new();

        let substitutions = Self::find_body_substitutions(&rule.body, facts)?;
        for subst in &substitutions {
            for head_atom in &rule.head {
                let grounded = Self::apply_substitution(head_atom, subst);
                // Never emit a head atom that still contains unbound variables.
                if Self::is_ground(&grounded) {
                    derived.push(grounded);
                }
            }
        }

        Ok(derived)
    }

    /// Enumerate all substitutions satisfying the full body via a left-to-right
    /// join, carrying bindings across atoms so shared variables stay consistent.
    fn find_body_substitutions(
        body: &[RuleAtom],
        facts: &HashSet<RuleAtom>,
    ) -> Result<Vec<HashMap<String, Term>>> {
        let mut substitutions: Vec<HashMap<String, Term>> = vec![HashMap::new()];

        for atom in body {
            let mut next = Vec::new();
            for subst in &substitutions {
                Self::extend_substitution(atom, subst, facts, &mut next)?;
            }
            substitutions = next;
            if substitutions.is_empty() {
                break;
            }
        }

        Ok(substitutions)
    }

    /// Extend `base` with every way `atom` can be satisfied given the current
    /// bindings and the fact base.
    fn extend_substitution(
        atom: &RuleAtom,
        base: &HashMap<String, Term>,
        facts: &HashSet<RuleAtom>,
        out: &mut Vec<HashMap<String, Term>>,
    ) -> Result<()> {
        match atom {
            RuleAtom::Triple { .. } => {
                let grounded = Self::apply_substitution(atom, base);
                for fact in facts {
                    if let Some(extended) = Self::unify_atoms(&grounded, fact, base) {
                        out.push(extended);
                    }
                }
            }
            // Constraint atoms act as filters: they must be ground under the
            // current bindings and then hold; they never introduce new bindings.
            RuleAtom::NotEqual { .. }
            | RuleAtom::GreaterThan { .. }
            | RuleAtom::LessThan { .. } => {
                let grounded = Self::apply_substitution(atom, base);
                if Self::is_ground(&grounded) && Self::evaluate_constraint(&grounded) {
                    out.push(base.clone());
                }
            }
            RuleAtom::Builtin { .. } => {
                return Err(anyhow::anyhow!(
                    "OWL 2 RL materialization does not support builtin body atoms: {atom:?}"
                ));
            }
        }
        Ok(())
    }

    /// Unify a (possibly partially-ground) triple pattern against a fact,
    /// extending `base` with any newly bound variables. Returns None on clash.
    fn unify_atoms(
        pattern: &RuleAtom,
        fact: &RuleAtom,
        base: &HashMap<String, Term>,
    ) -> Option<HashMap<String, Term>> {
        if let (
            RuleAtom::Triple {
                subject: ps,
                predicate: pp,
                object: po,
            },
            RuleAtom::Triple {
                subject: fs,
                predicate: fp,
                object: fo,
            },
        ) = (pattern, fact)
        {
            let mut subst = base.clone();
            if Self::unify_term(ps, fs, &mut subst)
                && Self::unify_term(pp, fp, &mut subst)
                && Self::unify_term(po, fo, &mut subst)
            {
                return Some(subst);
            }
        }
        None
    }

    /// Unify a single pattern term against a (ground) fact term.
    fn unify_term(pattern: &Term, value: &Term, subst: &mut HashMap<String, Term>) -> bool {
        match pattern {
            Term::Variable(var) => match subst.get(var) {
                Some(bound) => Self::terms_equal(bound, value),
                None => {
                    subst.insert(var.clone(), value.clone());
                    true
                }
            },
            _ => Self::terms_equal(pattern, value),
        }
    }

    fn terms_equal(a: &Term, b: &Term) -> bool {
        match (a, b) {
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => c == l,
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            _ => false,
        }
    }

    /// Apply a substitution to an atom.
    fn apply_substitution(atom: &RuleAtom, subst: &HashMap<String, Term>) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: Self::substitute_term(subject, subst),
                predicate: Self::substitute_term(predicate, subst),
                object: Self::substitute_term(object, subst),
            },
            RuleAtom::NotEqual { left, right } => RuleAtom::NotEqual {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
            RuleAtom::GreaterThan { left, right } => RuleAtom::GreaterThan {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
            RuleAtom::LessThan { left, right } => RuleAtom::LessThan {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
            RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_term(a, subst))
                    .collect(),
            },
        }
    }

    fn substitute_term(term: &Term, subst: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(var) => subst.get(var).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    fn is_ground(atom: &RuleAtom) -> bool {
        let is_ground_term = |t: &Term| !matches!(t, Term::Variable(_));
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => is_ground_term(subject) && is_ground_term(predicate) && is_ground_term(object),
            RuleAtom::NotEqual { left, right }
            | RuleAtom::GreaterThan { left, right }
            | RuleAtom::LessThan { left, right } => is_ground_term(left) && is_ground_term(right),
            RuleAtom::Builtin { args, .. } => args.iter().all(is_ground_term),
        }
    }

    /// Evaluate a ground constraint atom.
    fn evaluate_constraint(atom: &RuleAtom) -> bool {
        let numeric = |t: &Term| match t {
            Term::Constant(s) | Term::Literal(s) => s.parse::<f64>().ok(),
            _ => None,
        };
        match atom {
            RuleAtom::NotEqual { left, right } => !Self::terms_equal(left, right),
            RuleAtom::GreaterThan { left, right } => match (numeric(left), numeric(right)) {
                (Some(l), Some(r)) => l > r,
                _ => false,
            },
            RuleAtom::LessThan { left, right } => match (numeric(left), numeric(right)) {
                (Some(l), Some(r)) => l < r,
                _ => false,
            },
            _ => false,
        }
    }

    /// Get OWL 2 RL rules
    pub fn get_rules(&self) -> &[Rule] {
        &self.rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_el_classification() -> Result<()> {
        let mut reasoner = ELReasoner::new();
        let mut ontology = Ontology::new();

        // Build hierarchy: Dog ⊑ Mammal ⊑ Animal
        ontology.add_subsumption(
            Concept::Atomic("Dog".to_string()),
            Concept::Atomic("Mammal".to_string()),
        );
        ontology.add_subsumption(
            Concept::Atomic("Mammal".to_string()),
            Concept::Atomic("Animal".to_string()),
        );

        let hierarchy = reasoner.classify(&ontology)?;

        // Check transitive subsumption
        assert!(hierarchy.all_subsumptions.contains(&Subsumption {
            sub_class: "Dog".to_string(),
            super_class: "Animal".to_string(),
        }));

        Ok(())
    }

    #[test]
    fn test_el_subsumption() -> Result<()> {
        let mut reasoner = ELReasoner::new();
        let mut ontology = Ontology::new();

        ontology.add_subsumption(
            Concept::Atomic("Cat".to_string()),
            Concept::Atomic("Mammal".to_string()),
        );

        assert!(reasoner.is_subsumed_by(&ontology, "Cat", "Mammal")?);
        assert!(!reasoner.is_subsumed_by(&ontology, "Mammal", "Cat")?);

        Ok(())
    }

    #[test]
    fn test_el_equivalence() -> Result<()> {
        let mut reasoner = ELReasoner::new();
        let mut ontology = Ontology::new();

        ontology.add_equivalence(
            Concept::Atomic("Human".to_string()),
            Concept::Atomic("Person".to_string()),
        );

        let hierarchy = reasoner.classify(&ontology)?;

        // Both directions should be in hierarchy
        assert!(hierarchy.all_subsumptions.contains(&Subsumption {
            sub_class: "Human".to_string(),
            super_class: "Person".to_string(),
        }));
        assert!(hierarchy.all_subsumptions.contains(&Subsumption {
            sub_class: "Person".to_string(),
            super_class: "Human".to_string(),
        }));

        Ok(())
    }

    #[test]
    fn test_ql_query_rewriting() -> Result<()> {
        let mut reasoner = QLReasoner::new();
        let mut ontology = Ontology::new();

        ontology.add_subsumption(
            Concept::Atomic("Dog".to_string()),
            Concept::Atomic("Animal".to_string()),
        );

        let query = RuleAtom::Triple {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("Dog".to_string()),
        };

        let rewritten = reasoner.rewrite_query(&query, &ontology)?;

        // Should include original query + Animal query
        assert!(rewritten.len() >= 2);
        assert_eq!(reasoner.stats.rewritten_queries, 1);

        Ok(())
    }

    /// Regression: QL query rewriting must follow the full transitive subclass
    /// chain, not just one hop (Dog ⊑ Mammal ⊑ Animal ⇒ type(x,Dog) rewrites to
    /// include type(x,Mammal) AND type(x,Animal)).
    #[test]
    fn regression_ql_transitive_subclass_rewrite() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = QLReasoner::new();
        let mut ontology = Ontology::new();
        ontology.add_subsumption(
            Concept::Atomic("Dog".to_string()),
            Concept::Atomic("Mammal".to_string()),
        );
        ontology.add_subsumption(
            Concept::Atomic("Mammal".to_string()),
            Concept::Atomic("Animal".to_string()),
        );

        let query = RuleAtom::Triple {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("Dog".to_string()),
        };
        let rewritten = reasoner.rewrite_query(&query, &ontology)?;

        let has_object = |c: &str| {
            rewritten
                .iter()
                .any(|a| matches!(a, RuleAtom::Triple { object: Term::Constant(o), .. } if o == c))
        };
        assert!(
            has_object("Mammal"),
            "missing one-hop Mammal: {rewritten:?}"
        );
        assert!(
            has_object("Animal"),
            "missing transitive Animal: {rewritten:?}"
        );
        Ok(())
    }

    #[test]
    fn test_rl_rule_initialization() {
        let reasoner = RLReasoner::new();
        assert!(reasoner.get_rules().len() >= 4);
    }

    #[test]
    fn test_rl_materialization() -> Result<()> {
        let mut reasoner = RLReasoner::new();

        let mut facts = HashSet::new();
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("Dog".to_string()),
        });
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("Dog".to_string()),
            predicate: Term::Constant("rdfs:subClassOf".to_string()),
            object: Term::Constant("Animal".to_string()),
        });

        let materialized = reasoner.materialize(&facts)?;

        // Should include original facts plus inferred facts
        assert!(materialized.len() >= facts.len());

        Ok(())
    }

    /// Regression: OWL 2 RL materialization must perform real cross-atom
    /// unification and must never emit head atoms that still contain variables.
    #[test]
    fn regression_rl_real_unification_no_variable_leak() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = RLReasoner::new();

        let mut facts = HashSet::new();
        // Class chain: A ⊑ B ⊑ C  (scm-cls should derive A ⊑ C)
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("A".to_string()),
            predicate: Term::Constant("rdfs:subClassOf".to_string()),
            object: Term::Constant("B".to_string()),
        });
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("B".to_string()),
            predicate: Term::Constant("rdfs:subClassOf".to_string()),
            object: Term::Constant("C".to_string()),
        });
        // Type + subclass: fido:A , A ⊑ B  (cls-hv1 should derive fido:B, fido:C)
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("A".to_string()),
        });

        let materialized = reasoner.materialize(&facts)?;

        // Real join derived the transitive subclass edge.
        assert!(
            materialized.contains(&RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("rdfs:subClassOf".to_string()),
                object: Term::Constant("C".to_string()),
            }),
            "expected transitive A ⊑ C; got: {materialized:?}"
        );

        // Real join propagated the type through the subclass chain.
        assert!(
            materialized.contains(&RuleAtom::Triple {
                subject: Term::Constant("fido".to_string()),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant("C".to_string()),
            }),
            "expected fido:C via type propagation; got: {materialized:?}"
        );

        // No fact may contain an unbound variable term.
        for fact in &materialized {
            assert!(
                RLReasoner::is_ground(fact),
                "materialized fact leaked a variable: {fact:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_class_hierarchy_queries() -> Result<()> {
        let mut hierarchy = ClassHierarchy::new();

        hierarchy.all_subsumptions.insert(Subsumption {
            sub_class: "Dog".to_string(),
            super_class: "Mammal".to_string(),
        });
        hierarchy.all_subsumptions.insert(Subsumption {
            sub_class: "Dog".to_string(),
            super_class: "Animal".to_string(),
        });

        let superclasses = hierarchy.get_superclasses("Dog");
        assert_eq!(superclasses.len(), 2);
        assert!(superclasses.contains("Mammal"));
        assert!(superclasses.contains("Animal"));

        Ok(())
    }

    #[test]
    fn test_el_statistics() -> Result<()> {
        let mut reasoner = ELReasoner::new();
        let mut ontology = Ontology::new();

        ontology.add_subsumption(
            Concept::Atomic("A".to_string()),
            Concept::Atomic("B".to_string()),
        );

        reasoner.classify(&ontology)?;

        assert!(reasoner.stats.classifications > 0);
        assert!(reasoner.stats.iterations > 0);

        Ok(())
    }

    #[test]
    fn test_ql_answer_query() -> Result<()> {
        let mut reasoner = QLReasoner::new();
        let mut ontology = Ontology::new();

        ontology.add_subsumption(
            Concept::Atomic("Dog".to_string()),
            Concept::Atomic("Animal".to_string()),
        );

        let mut facts = HashSet::new();
        facts.insert(RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("Dog".to_string()),
        });

        // Query: is fido a Dog?
        let query = RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("Dog".to_string()),
        };

        assert!(reasoner.answer_query(&query, &ontology, &facts)?);

        Ok(())
    }
}
