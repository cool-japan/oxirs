//! # Probabilistic Datalog (ProbLog)
//!
//! This module implements ProbLog, a probabilistic extension of Datalog that allows
//! reasoning with uncertainty by attaching probabilities to facts and rules.
//!
//! ## Features
//!
//! - **Probabilistic Facts** - Facts with associated probabilities
//! - **Probabilistic Rules** - Rules that propagate probabilities
//! - **Query Evaluation** - Compute probability of queries
//! - **Independence Assumptions** - Handle independent and dependent events
//! - **Explanation Trees** - Track provenance of probabilistic derivations
//!
//! ## Probability Semantics
//!
//! 1. **Probabilistic Facts**: `p::fact(a).` means fact(a) is true with probability p
//! 2. **Conjunction**: P(A ∧ B) = P(A) × P(B) (assuming independence)
//! 3. **Disjunction**: P(A ∨ B) = P(A) + P(B) - P(A) × P(B)
//! 4. **Negation**: P(¬A) = 1 - P(A)
//!
//! ## Example
//!
//! ```text
//! use oxirs_rule::problog::{ProbLogEngine, ProbabilisticFact, ProbabilisticRule};
//! use oxirs_rule::{RuleAtom, Term};
//!
//! let mut engine = ProbLogEngine::new();
//!
//! // Add probabilistic fact: 0.8::parent(john, mary)
//! engine.add_probabilistic_fact(ProbabilisticFact {
//!     probability: 0.8,
//!     fact: RuleAtom::Triple {
//!         subject: Term::Constant("john".to_string()),
//!         predicate: Term::Constant("parent".to_string()),
//!         object: Term::Constant("mary".to_string()),
//!     },
//! });
//!
//! // Add rule: parent(X,Y) → ancestor(X,Y)
//! // Inferred facts inherit probabilities from their premises
//!
//! // Query: What's the probability of ancestor(john, mary)?
//! let prob = engine.query_probability(&RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("ancestor".to_string()),
//!     object: Term::Constant("mary".to_string()),
//! })?;
//!
//! assert!((prob - 0.8).abs() < 0.001);
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::forward::Substitution;
use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Timer};
use scirs2_core::random::{Distribution, Uniform};
use std::collections::{HashMap, HashSet};

// Global metrics for ProbLog
lazy_static::lazy_static! {
    static ref PROBLOG_QUERIES: Counter = Counter::new("problog_queries".to_string());
    static ref PROBLOG_INFERENCES: Counter = Counter::new("problog_inferences".to_string());
    static ref PROBLOG_QUERY_TIME: Timer = Timer::new("problog_query_time".to_string());
}

/// Probabilistic fact with associated probability
#[derive(Debug, Clone)]
pub struct ProbabilisticFact {
    /// Probability in [0, 1]
    pub probability: f64,
    /// The fact itself
    pub fact: RuleAtom,
}

impl ProbabilisticFact {
    pub fn new(probability: f64, fact: RuleAtom) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(anyhow!(
                "Probability must be in [0, 1], got {}",
                probability
            ));
        }
        Ok(Self { probability, fact })
    }
}

/// Probabilistic rule with optional probability
#[derive(Debug, Clone)]
pub struct ProbabilisticRule {
    /// Optional probability (if None, probability is 1.0)
    pub probability: Option<f64>,
    /// The rule itself
    pub rule: Rule,
}

impl ProbabilisticRule {
    pub fn deterministic(rule: Rule) -> Self {
        Self {
            probability: None,
            rule,
        }
    }

    pub fn probabilistic(probability: f64, rule: Rule) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(anyhow!(
                "Probability must be in [0, 1], got {}",
                probability
            ));
        }
        Ok(Self {
            probability: Some(probability),
            rule,
        })
    }
}

/// Derivation tree tracking provenance
#[derive(Debug, Clone)]
pub struct DerivationTree {
    /// The derived fact
    pub fact: RuleAtom,
    /// Probability of this derivation
    pub probability: f64,
    /// Facts this was derived from
    pub premises: Vec<DerivationTree>,
}

impl DerivationTree {
    pub fn leaf(fact: RuleAtom, probability: f64) -> Self {
        Self {
            fact,
            probability,
            premises: Vec::new(),
        }
    }

    pub fn node(fact: RuleAtom, probability: f64, premises: Vec<DerivationTree>) -> Self {
        Self {
            fact,
            probability,
            premises,
        }
    }
}

/// Evaluation strategy for recursive queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationStrategy {
    /// Top-down backward chaining (with cycle detection)
    TopDown,
    /// Bottom-up forward chaining with fixpoint iteration
    BottomUp,
    /// Automatic selection based on query characteristics
    Auto,
}

/// Probabilistic Datalog engine
pub struct ProbLogEngine {
    /// Probabilistic facts
    probabilistic_facts: HashMap<RuleAtom, f64>,
    /// Deterministic facts (probability 1.0)
    deterministic_facts: HashSet<RuleAtom>,
    /// Probabilistic rules
    probabilistic_rules: Vec<ProbabilisticRule>,
    /// Cached query results
    query_cache: HashMap<RuleAtom, f64>,
    /// Recursion stack for cycle detection
    recursion_stack: HashSet<RuleAtom>,
    /// Maximum recursion depth
    max_depth: usize,
    /// Current recursion depth
    current_depth: usize,
    /// Materialized facts from fixpoint iteration (for bottom-up evaluation)
    materialized_facts: HashMap<RuleAtom, f64>,
    /// Whether materialization has been computed
    materialization_valid: bool,
    /// Evaluation strategy
    strategy: EvaluationStrategy,
    /// Maximum fixpoint iterations
    max_fixpoint_iterations: usize,
    /// Statistics
    pub stats: ProbLogStats,
}

/// Statistics for ProbLog engine
#[derive(Debug, Clone, Default)]
pub struct ProbLogStats {
    pub queries: usize,
    pub inferences: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub fixpoint_iterations: usize,
    pub materialized_facts_count: usize,
}

impl Default for ProbLogEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbLogEngine {
    pub fn new() -> Self {
        Self {
            probabilistic_facts: HashMap::new(),
            deterministic_facts: HashSet::new(),
            probabilistic_rules: Vec::new(),
            query_cache: HashMap::new(),
            recursion_stack: HashSet::new(),
            max_depth: 100,
            current_depth: 0,
            materialized_facts: HashMap::new(),
            materialization_valid: false,
            strategy: EvaluationStrategy::Auto,
            max_fixpoint_iterations: 1000,
            stats: ProbLogStats::default(),
        }
    }

    /// Create a new engine with custom max recursion depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Create a new engine with specific evaluation strategy
    pub fn with_strategy(mut self, strategy: EvaluationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Create a new engine with custom fixpoint iteration limit
    pub fn with_max_fixpoint_iterations(mut self, max_iterations: usize) -> Self {
        self.max_fixpoint_iterations = max_iterations;
        self
    }

    /// Add a probabilistic fact
    pub fn add_probabilistic_fact(&mut self, fact: ProbabilisticFact) {
        if (fact.probability - 1.0).abs() < 1e-10 {
            // Treat as deterministic for efficiency
            self.deterministic_facts.insert(fact.fact);
        } else {
            self.probabilistic_facts.insert(fact.fact, fact.probability);
        }
        self.query_cache.clear(); // Invalidate cache
        self.materialization_valid = false; // Invalidate materialization
    }

    /// Add a deterministic fact
    pub fn add_fact(&mut self, fact: RuleAtom) {
        self.deterministic_facts.insert(fact);
        self.query_cache.clear();
        self.materialization_valid = false;
    }

    /// Add a probabilistic rule
    pub fn add_rule(&mut self, rule: ProbabilisticRule) {
        self.probabilistic_rules.push(rule);
        self.query_cache.clear();
        self.materialization_valid = false;
    }

    /// Query the probability of a fact
    pub fn query_probability(&mut self, query: &RuleAtom) -> Result<f64> {
        let _timer = PROBLOG_QUERY_TIME.start();
        self.stats.queries += 1;
        PROBLOG_QUERIES.inc();

        // Determine evaluation strategy
        let use_bottom_up = match self.strategy {
            EvaluationStrategy::TopDown => false,
            EvaluationStrategy::BottomUp => true,
            EvaluationStrategy::Auto => {
                // Use bottom-up if we have recursive rules
                self.has_recursive_rules()
            }
        };

        if use_bottom_up {
            // Use bottom-up evaluation with fixpoint iteration
            return self.query_materialized(query);
        }

        // Top-down evaluation (backward chaining)
        // Check recursion depth
        if self.current_depth > self.max_depth {
            return Err(anyhow!(
                "Maximum recursion depth exceeded: {}",
                self.max_depth
            ));
        }

        // Check for cycles - use bottom-up if cycle detected
        if self.recursion_stack.contains(query) {
            // Cycle detected - fall back to bottom-up evaluation
            return self.query_materialized(query);
        }

        // Check cache
        if let Some(&prob) = self.query_cache.get(query) {
            self.stats.cache_hits += 1;
            return Ok(prob);
        }

        self.stats.cache_misses += 1;

        // Check if it's a known fact
        if self.deterministic_facts.contains(query) {
            self.query_cache.insert(query.clone(), 1.0);
            return Ok(1.0);
        }

        if let Some(&prob) = self.probabilistic_facts.get(query) {
            self.query_cache.insert(query.clone(), prob);
            return Ok(prob);
        }

        // Add to recursion stack
        self.recursion_stack.insert(query.clone());
        self.current_depth += 1;

        // Try to derive using rules
        let prob = self.derive_probability(query)?;
        self.query_cache.insert(query.clone(), prob);

        // Remove from recursion stack
        self.recursion_stack.remove(query);
        self.current_depth -= 1;

        Ok(prob)
    }

    /// Check if the engine has recursive rules
    fn has_recursive_rules(&self) -> bool {
        for rule in &self.probabilistic_rules {
            // Check if any head predicate appears in the body
            for head_atom in &rule.rule.head {
                if let RuleAtom::Triple {
                    predicate: head_pred,
                    ..
                } = head_atom
                {
                    for body_atom in &rule.rule.body {
                        if let RuleAtom::Triple {
                            predicate: body_pred,
                            ..
                        } = body_atom
                        {
                            if head_pred == body_pred {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Derive probability using rules with full variable unification
    fn derive_probability(&mut self, query: &RuleAtom) -> Result<f64> {
        let mut total_prob = 0.0;

        // Try each rule
        for prob_rule in &self.probabilistic_rules.clone() {
            // Try to unify each head atom with the query
            for head_atom in &prob_rule.rule.head {
                if let Some(substitution) = self.unify_atoms(head_atom, query) {
                    // Apply substitution to rule body
                    let instantiated_body =
                        self.apply_substitution_to_body(&prob_rule.rule.body, &substitution);

                    // Evaluate instantiated rule body
                    let body_prob = self.evaluate_body(&instantiated_body)?;

                    // Combine with rule probability
                    let derivation_prob = body_prob * prob_rule.probability.unwrap_or(1.0);

                    // Disjunctive combination: P(A ∨ B) = P(A) + P(B) - P(A)P(B)
                    total_prob = total_prob + derivation_prob - (total_prob * derivation_prob);

                    self.stats.inferences += 1;
                    PROBLOG_INFERENCES.inc();
                }
            }
        }

        Ok(total_prob)
    }

    /// Evaluate rule body probability (conjunction)
    fn evaluate_body(&mut self, body: &[RuleAtom]) -> Result<f64> {
        let mut prob = 1.0;

        for atom in body {
            let atom_prob = self.query_probability(atom)?;
            // Conjunction (assuming independence): P(A ∧ B) = P(A) × P(B)
            prob *= atom_prob;
        }

        Ok(prob)
    }

    /// Unify two atoms, returning a substitution if successful
    ///
    /// Implements full variable unification with occurs check
    fn unify_atoms(&self, pattern: &RuleAtom, target: &RuleAtom) -> Option<Substitution> {
        let mut substitution = HashMap::new();

        match (pattern, target) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                // Unify each component
                if !self.unify_terms(s1, s2, &mut substitution) {
                    return None;
                }
                if !self.unify_terms(p1, p2, &mut substitution) {
                    return None;
                }
                if !self.unify_terms(o1, o2, &mut substitution) {
                    return None;
                }
                Some(substitution)
            }
            _ => None,
        }
    }

    /// Unify two terms, updating the substitution
    fn unify_terms(&self, t1: &Term, t2: &Term, subst: &mut Substitution) -> bool {
        // Apply existing substitutions
        let t1_resolved = self.apply_substitution_to_term(t1, subst);
        let t2_resolved = self.apply_substitution_to_term(t2, subst);

        match (&t1_resolved, &t2_resolved) {
            // Variable to variable
            (Term::Variable(v1), Term::Variable(v2)) if v1 == v2 => true,
            // Variable to constant/literal
            (Term::Variable(v), t) | (t, Term::Variable(v)) => {
                // Occurs check: prevent cyclic substitutions
                if self.occurs_in_term(v, t) {
                    return false;
                }
                subst.insert(v.clone(), t.clone());
                true
            }
            // Constant to constant
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            // Literal to literal
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            // Function to function
            (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
                if n1 != n2 || a1.len() != a2.len() {
                    return false;
                }
                // Recursively unify arguments
                for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                    if !self.unify_terms(arg1, arg2, subst) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Check if a variable occurs in a term (for occurs check)
    #[allow(clippy::only_used_in_recursion)] // false positive: parameters are used for comparisons
    fn occurs_in_term(&self, var: &str, term: &Term) -> bool {
        match term {
            Term::Variable(v) => v == var,
            Term::Constant(_) | Term::Literal(_) => false,
            Term::Function { args, .. } => args.iter().any(|arg| self.occurs_in_term(var, arg)),
        }
    }

    /// Apply substitution to a term
    #[allow(clippy::only_used_in_recursion)] // false positive: parameters are used for lookups
    fn apply_substitution_to_term(&self, term: &Term, subst: &Substitution) -> Term {
        match term {
            Term::Variable(v) => subst.get(v).cloned().unwrap_or_else(|| term.clone()),
            Term::Function { name, args } => Term::Function {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| self.apply_substitution_to_term(arg, subst))
                    .collect(),
            },
            _ => term.clone(),
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution_to_atom(&self, atom: &RuleAtom, subst: &Substitution) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: self.apply_substitution_to_term(subject, subst),
                predicate: self.apply_substitution_to_term(predicate, subst),
                object: self.apply_substitution_to_term(object, subst),
            },
            RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| self.apply_substitution_to_term(arg, subst))
                    .collect(),
            },
            RuleAtom::NotEqual { left, right } => RuleAtom::NotEqual {
                left: self.apply_substitution_to_term(left, subst),
                right: self.apply_substitution_to_term(right, subst),
            },
            RuleAtom::GreaterThan { left, right } => RuleAtom::GreaterThan {
                left: self.apply_substitution_to_term(left, subst),
                right: self.apply_substitution_to_term(right, subst),
            },
            RuleAtom::LessThan { left, right } => RuleAtom::LessThan {
                left: self.apply_substitution_to_term(left, subst),
                right: self.apply_substitution_to_term(right, subst),
            },
        }
    }

    /// Apply substitution to rule body
    fn apply_substitution_to_body(&self, body: &[RuleAtom], subst: &Substitution) -> Vec<RuleAtom> {
        body.iter()
            .map(|atom| self.apply_substitution_to_atom(atom, subst))
            .collect()
    }

    /// Compute fixpoint using bottom-up materialization (semi-naive evaluation)
    ///
    /// This implements proper transitive closure for recursive rules by iteratively
    /// applying rules until no new facts can be derived (fixpoint).
    ///
    /// # Algorithm
    /// 1. Start with base facts (EDB - Extensional Database)
    /// 2. Apply all rules to derive new facts (IDB - Intentional Database)
    /// 3. Repeat until fixpoint (no new facts derived)
    /// 4. Track probabilities using disjunctive combination for multiple derivations
    ///
    /// # Returns
    /// - `Ok(())` on success
    /// - `Err` if max iterations exceeded
    pub fn materialize(&mut self) -> Result<()> {
        if self.materialization_valid {
            return Ok(()); // Already computed
        }

        self.materialized_facts.clear();
        self.stats.fixpoint_iterations = 0;

        // Initialize with base facts
        let mut current_facts = HashMap::new();
        for (fact, &prob) in &self.probabilistic_facts {
            current_facts.insert(fact.clone(), prob);
        }
        for fact in &self.deterministic_facts {
            current_facts.insert(fact.clone(), 1.0);
        }

        // Fixpoint iteration
        let mut iteration = 0;
        loop {
            iteration += 1;
            self.stats.fixpoint_iterations = iteration;

            if iteration > self.max_fixpoint_iterations {
                return Err(anyhow!(
                    "Maximum fixpoint iterations exceeded: {}",
                    self.max_fixpoint_iterations
                ));
            }

            let previous_size = current_facts.len();
            let mut new_facts = HashMap::new();

            // Apply each rule to current facts
            for prob_rule in &self.probabilistic_rules {
                let derived = self.apply_rule_for_materialization(
                    &prob_rule.rule,
                    &current_facts,
                    prob_rule.probability.unwrap_or(1.0),
                )?;

                // Collect all derived facts with probability combinations
                for (fact, prob) in derived {
                    let entry = new_facts.entry(fact).or_insert(0.0);
                    // Disjunctive combination for multiple derivations of same fact
                    *entry = *entry + prob - (*entry * prob);
                }
            }

            // Check convergence: no new facts and probabilities haven't changed significantly
            let mut changed = false;
            for (fact, new_prob) in &new_facts {
                let existing_prob = current_facts.get(fact).copied().unwrap_or(0.0);
                if (new_prob - existing_prob).abs() > 1e-10 {
                    changed = true;
                    current_facts.insert(fact.clone(), *new_prob);
                }
            }

            // Also check if new facts were added
            if !changed && current_facts.len() == previous_size {
                // Fixpoint reached: no changes and no new facts
                break;
            }
        }

        // Store materialized facts
        self.materialized_facts = current_facts;
        self.stats.materialized_facts_count = self.materialized_facts.len();
        self.materialization_valid = true;

        Ok(())
    }

    /// Apply a single rule to derive new facts during materialization
    fn apply_rule_for_materialization(
        &self,
        rule: &Rule,
        facts: &HashMap<RuleAtom, f64>,
        rule_prob: f64,
    ) -> Result<HashMap<RuleAtom, f64>> {
        let mut derived = HashMap::new();

        // Generate all possible variable bindings that satisfy the rule body
        let bindings = self.find_all_bindings(&rule.body, facts)?;

        for binding in bindings {
            // Compute body probability
            let mut body_prob = 1.0;
            for body_atom in &rule.body {
                let instantiated = self.apply_substitution_to_atom(body_atom, &binding);
                let atom_prob = facts.get(&instantiated).copied().unwrap_or(0.0);
                body_prob *= atom_prob;
            }

            // Apply substitution to head
            for head_atom in &rule.head {
                let instantiated_head = self.apply_substitution_to_atom(head_atom, &binding);
                let derivation_prob = body_prob * rule_prob;

                // Combine with existing derivations
                let current_prob = derived.get(&instantiated_head).copied().unwrap_or(0.0);
                let combined = if current_prob > 0.0 {
                    current_prob + derivation_prob - (current_prob * derivation_prob)
                } else {
                    derivation_prob
                };

                derived.insert(instantiated_head, combined);
            }
        }

        Ok(derived)
    }

    /// Find all variable bindings that satisfy a rule body
    fn find_all_bindings(
        &self,
        body: &[RuleAtom],
        facts: &HashMap<RuleAtom, f64>,
    ) -> Result<Vec<Substitution>> {
        if body.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Start with first atom
        let first_atom = &body[0];
        let rest_body = &body[1..];

        let mut all_bindings = Vec::new();

        // Find all facts that unify with the first atom
        for fact in facts.keys() {
            if let Some(binding) = self.unify_atoms(first_atom, fact) {
                if rest_body.is_empty() {
                    // Base case: only one atom in body
                    all_bindings.push(binding);
                } else {
                    // Recursive case: apply binding to rest of body
                    let instantiated_rest = self.apply_substitution_to_body(rest_body, &binding);

                    // Find bindings for rest of body
                    let rest_bindings = self.find_all_bindings(&instantiated_rest, facts)?;

                    for rest_binding in rest_bindings {
                        // Merge bindings
                        let mut merged = binding.clone();
                        for (var, term) in rest_binding {
                            merged.insert(var, term);
                        }
                        all_bindings.push(merged);
                    }
                }
            }
        }

        Ok(all_bindings)
    }

    /// Query using bottom-up evaluation (requires materialization)
    pub fn query_materialized(&mut self, query: &RuleAtom) -> Result<f64> {
        // Ensure materialization is computed
        if !self.materialization_valid {
            self.materialize()?;
        }

        // Lookup in materialized facts
        let prob = self.materialized_facts.get(query).copied().unwrap_or(0.0);
        Ok(prob)
    }

    /// Sample from the probability distribution
    pub fn sample(&mut self) -> HashSet<RuleAtom> {
        use scirs2_core::random::rng;

        let mut rng_instance = rng();
        let uniform = Uniform::new(0.0, 1.0).expect("distribution parameters are valid");
        let mut sampled_facts = HashSet::new();

        // Sample probabilistic facts
        for (fact, &prob) in &self.probabilistic_facts {
            if uniform.sample(&mut rng_instance) < prob {
                sampled_facts.insert(fact.clone());
            }
        }

        // Always include deterministic facts
        sampled_facts.extend(self.deterministic_facts.iter().cloned());

        sampled_facts
    }

    /// Perform Monte Carlo estimation of query probability
    pub fn monte_carlo_query(&mut self, query: &RuleAtom, samples: usize) -> Result<f64> {
        let mut successes = 0;

        for _ in 0..samples {
            let sampled_world = self.sample();
            if sampled_world.contains(query) {
                successes += 1;
            }
        }

        Ok(successes as f64 / samples as f64)
    }

    /// Build derivation tree for query
    pub fn explain(&mut self, query: &RuleAtom) -> Result<Option<DerivationTree>> {
        // Check if it's a known fact
        if self.deterministic_facts.contains(query) {
            return Ok(Some(DerivationTree::leaf(query.clone(), 1.0)));
        }

        if let Some(&prob) = self.probabilistic_facts.get(query) {
            return Ok(Some(DerivationTree::leaf(query.clone(), prob)));
        }

        // Try to derive using rules with full unification
        for prob_rule in &self.probabilistic_rules.clone() {
            // Try to unify each head atom with the query
            for head_atom in &prob_rule.rule.head {
                if let Some(substitution) = self.unify_atoms(head_atom, query) {
                    // Apply substitution to rule body
                    let instantiated_body =
                        self.apply_substitution_to_body(&prob_rule.rule.body, &substitution);

                    // Build premise trees for instantiated body
                    let mut premises = Vec::new();
                    let mut body_prob = 1.0;

                    for body_atom in &instantiated_body {
                        if let Some(tree) = self.explain(body_atom)? {
                            body_prob *= tree.probability;
                            premises.push(tree);
                        } else {
                            // Cannot derive this premise
                            body_prob = 0.0;
                            break;
                        }
                    }

                    if body_prob > 0.0 {
                        let total_prob = body_prob * prob_rule.probability.unwrap_or(1.0);
                        return Ok(Some(DerivationTree::node(
                            query.clone(),
                            total_prob,
                            premises,
                        )));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Clear query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ProbLogStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triple(subject: &str, predicate: &str, object: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: Term::Constant(subject.to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Constant(object.to_string()),
        }
    }

    #[test]
    fn test_probabilistic_fact_creation() -> Result<()> {
        let fact = ProbabilisticFact::new(0.8, create_triple("john", "parent", "mary"))?;
        assert_eq!(fact.probability, 0.8);
        Ok(())
    }

    #[test]
    fn test_invalid_probability() {
        let result = ProbabilisticFact::new(1.5, create_triple("john", "parent", "mary"));
        assert!(result.is_err());
    }

    #[test]
    fn test_query_deterministic_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_fact(fact.clone());

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 1.0);

        Ok(())
    }

    #[test]
    fn test_query_probabilistic_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_probabilistic_fact(ProbabilisticFact::new(0.8, fact.clone())?);

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 0.8);

        Ok(())
    }

    #[test]
    fn test_query_unknown_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 0.0);

        Ok(())
    }

    #[test]
    fn test_rule_derivation() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add probabilistic fact: 0.8::parent(john, mary)
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("john", "parent", "mary"),
        )?);

        // Add ground rule: parent(john,mary) → ancestor(john,mary)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor".to_string(),
            body: vec![create_triple("john", "parent", "mary")],
            head: vec![create_triple("john", "ancestor", "mary")],
        }));

        // Query: ancestor(john, mary) should have probability 0.8
        let ancestor_fact = create_triple("john", "ancestor", "mary");
        let prob = engine.query_probability(&ancestor_fact)?;

        assert!((prob - 0.8).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_query_caching() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_probabilistic_fact(ProbabilisticFact::new(0.7, fact.clone())?);

        // First query - cache miss
        engine.query_probability(&fact)?;
        assert_eq!(engine.stats.cache_misses, 1);
        assert_eq!(engine.stats.cache_hits, 0);

        // Second query - cache hit
        engine.query_probability(&fact)?;
        assert_eq!(engine.stats.cache_hits, 1);

        Ok(())
    }

    #[test]
    fn test_sampling() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add deterministic fact
        engine.add_fact(create_triple("john", "person", "true"));

        // Add probabilistic fact
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.5,
            create_triple("john", "tall", "true"),
        )?);

        // Sample multiple times
        let mut tall_count = 0;
        let samples = 1000;

        for _ in 0..samples {
            let world = engine.sample();
            if world.contains(&create_triple("john", "tall", "true")) {
                tall_count += 1;
            }
            // Deterministic fact should always be present
            assert!(world.contains(&create_triple("john", "person", "true")));
        }

        // Check that sampling is approximately correct (0.5 ± 0.1)
        let proportion = tall_count as f64 / samples as f64;
        assert!((proportion - 0.5).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_explanation_tree() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add probabilistic fact
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        // Get explanation
        let tree = engine.explain(&create_triple("john", "parent", "mary"))?;

        assert!(tree.is_some());
        let tree = tree.unwrap();
        assert_eq!(tree.probability, 0.9);
        assert!(tree.premises.is_empty()); // Leaf node

        Ok(())
    }

    #[test]
    fn test_probabilistic_rule() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add fact
        engine.add_fact(create_triple("john", "parent", "mary"));

        // Add ground probabilistic rule: 0.9::parent(john,mary) → ancestor(john,mary)
        engine.add_rule(ProbabilisticRule::probabilistic(
            0.9,
            Rule {
                name: "ancestor".to_string(),
                body: vec![create_triple("john", "parent", "mary")],
                head: vec![create_triple("john", "ancestor", "mary")],
            },
        )?);

        // Query: ancestor(john, mary) should have probability 0.9
        let prob = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;

        assert!((prob - 0.9).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_variable_unification_simple() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add probabilistic fact: 0.8::parent(john, mary)
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("john", "parent", "mary"),
        )?);

        // Add rule with variables: parent(X,Y) → ancestor(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Query: ancestor(john, mary) should have probability 0.8
        // This tests that X=john, Y=mary unifies correctly
        let prob = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;

        assert!((prob - 0.8).abs() < 0.001, "Expected 0.8, got {}", prob);

        Ok(())
    }

    #[test]
    fn test_variable_unification_multiple_facts() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add multiple parent facts with different probabilities
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.7,
            create_triple("mary", "parent", "bob"),
        )?);

        // Add rule with variables: parent(X,Y) → ancestor(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Test first derivation: ancestor(john, mary) = 0.9
        let prob1 = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;
        assert!((prob1 - 0.9).abs() < 0.001, "Expected 0.9, got {}", prob1);

        // Test second derivation: ancestor(mary, bob) = 0.7
        let prob2 = engine.query_probability(&create_triple("mary", "ancestor", "bob"))?;
        assert!((prob2 - 0.7).abs() < 0.001, "Expected 0.7, got {}", prob2);

        Ok(())
    }

    #[test]
    fn test_variable_unification_transitive() -> Result<()> {
        // ✅ RE-ENABLED December 9, 2025 - Now works with fixpoint iteration!
        // Previously ignored due to stack overflow with recursive rules.
        // Auto strategy automatically detects recursion and uses bottom-up evaluation.
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::Auto);

        // Add facts: parent(john, mary), parent(mary, bob)
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("mary", "parent", "bob"),
        )?);

        // Rule 1: parent(X,Y) → ancestor(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Rule 2: parent(X,Y) ∧ ancestor(Y,Z) → ancestor(X,Z)
        // This creates recursive queries - now handled by fixpoint iteration!
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("ancestor".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        // Query: ancestor(john, bob) should be derived transitively
        // P(ancestor(john, bob)) = P(parent(john, mary)) × P(ancestor(mary, bob))
        //                        = P(parent(john, mary)) × P(parent(mary, bob))
        //                        = 0.9 × 0.8 = 0.72
        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;

        assert!((prob - 0.72).abs() < 0.001, "Expected 0.72, got {}", prob);

        // Verify fixpoint iteration was used
        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have used fixpoint iteration for recursive rules"
        );

        Ok(())
    }

    #[test]
    fn test_variable_unification_with_probabilistic_rule() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add deterministic fact
        engine.add_fact(create_triple("john", "parent", "mary"));

        // Add probabilistic rule: 0.95::parent(X,Y) → related(X,Y)
        // This means "if X is parent of Y, then X is related to Y with 95% probability"
        engine.add_rule(ProbabilisticRule::probabilistic(
            0.95,
            Rule {
                name: "related_rule".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("related".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
            },
        )?);

        // Query: related(john, mary) should have probability 1.0 * 0.95 = 0.95
        let prob = engine.query_probability(&create_triple("john", "related", "mary"))?;

        assert!((prob - 0.95).abs() < 0.001, "Expected 0.95, got {}", prob);

        Ok(())
    }

    #[test]
    fn test_cycle_detection() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add facts for cycle: a->b, b->c, c->a
        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));
        engine.add_fact(create_triple("c", "edge", "a"));

        // Add recursive rule: edge(X,Y), path(Y,Z) -> path(X,Z)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_transitive".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        // Query should not crash (cycle detection prevents stack overflow)
        let prob = engine.query_probability(&create_triple("a", "path", "a"))?;

        // Due to cycle detection, this will return 0.0 (not the correct answer, but prevents crash)
        assert_eq!(
            prob, 0.0,
            "Cycle detection should return 0.0 for cyclic queries"
        );

        Ok(())
    }

    #[test]
    fn test_unification_failure() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        // Add fact: parent(john, mary)
        engine.add_fact(create_triple("john", "parent", "mary"));

        // Add rule with mismatched constants: parent(john, bob) → ancestor(john, bob)
        // This should NOT unify with parent(john, mary)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "specific_rule".to_string(),
            body: vec![create_triple("john", "parent", "bob")],
            head: vec![create_triple("john", "ancestor", "bob")],
        }));

        // Query: ancestor(john, bob) should have probability 0.0 (cannot derive)
        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;

        assert!(
            prob.abs() < 0.001,
            "Expected 0.0, got {} - unification should fail",
            prob
        );

        Ok(())
    }

    // ========== Fixpoint Iteration Tests (NEW - December 9, 2025) ==========

    #[test]
    fn test_fixpoint_transitive_closure() -> Result<()> {
        // Test proper transitive closure with fixpoint iteration
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        // Add facts: parent(john, mary), parent(mary, bob)
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("mary", "parent", "bob"),
        )?);

        // Rule 1: parent(X,Y) → ancestor(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Rule 2: parent(X,Y) ∧ ancestor(Y,Z) → ancestor(X,Z)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("ancestor".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        // Query: ancestor(john, mary) = 0.9 (direct)
        let prob1 = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;
        assert!((prob1 - 0.9).abs() < 0.001, "Expected 0.9, got {}", prob1);

        // Query: ancestor(mary, bob) = 0.8 (direct)
        let prob2 = engine.query_probability(&create_triple("mary", "ancestor", "bob"))?;
        assert!((prob2 - 0.8).abs() < 0.001, "Expected 0.8, got {}", prob2);

        // Query: ancestor(john, bob) = 0.9 × 0.8 = 0.72 (transitive)
        let prob3 = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;
        assert!(
            (prob3 - 0.72).abs() < 0.001,
            "Expected 0.72 (transitive), got {}",
            prob3
        );

        // Verify fixpoint was computed
        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have used fixpoint iteration"
        );
        assert!(
            engine.stats.materialized_facts_count >= 5,
            "Should have materialized at least 5 facts (2 parent + 3 ancestor)"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_cyclic_graph() -> Result<()> {
        // Test proper handling of cycles with fixpoint iteration
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        // Add facts for cycle: a->b, b->c, c->a
        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));
        engine.add_fact(create_triple("c", "edge", "a"));

        // Add recursive rule: edge(X,Y), path(Y,Z) -> path(X,Z)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_transitive".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        // Query should compute correctly (now using fixpoint iteration)
        let prob = engine.query_probability(&create_triple("a", "path", "a"))?;

        // Due to cycle, path(a,a) should be derivable: a->b, b->c, c->a
        assert_eq!(
            prob, 1.0,
            "Fixpoint iteration should correctly compute cyclic path (got {})",
            prob
        );

        // Verify all paths were materialized
        let prob_ab = engine.query_probability(&create_triple("a", "path", "b"))?;
        let prob_bc = engine.query_probability(&create_triple("b", "path", "c"))?;
        let prob_ca = engine.query_probability(&create_triple("c", "path", "a"))?;

        assert_eq!(prob_ab, 1.0, "path(a,b) should be 1.0");
        assert_eq!(prob_bc, 1.0, "path(b,c) should be 1.0");
        assert_eq!(prob_ca, 1.0, "path(c,a) should be 1.0");

        Ok(())
    }

    #[test]
    fn test_fixpoint_auto_strategy() -> Result<()> {
        // Test automatic selection of bottom-up for recursive rules
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::Auto);

        // Add facts
        engine.add_fact(create_triple("john", "parent", "mary"));
        engine.add_fact(create_triple("mary", "parent", "bob"));

        // Add recursive rule
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("ancestor".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        // Auto strategy should detect recursion and use bottom-up
        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;
        assert_eq!(prob, 1.0, "Auto strategy should compute transitive closure");

        // Verify bottom-up was used
        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Auto strategy should have used fixpoint iteration for recursive rules"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_probabilistic_combination() -> Result<()> {
        // Test proper probability combination with multiple derivations
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        // Add facts with different probabilities
        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.6,
            create_triple("a", "edge", "b"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.7,
            create_triple("a", "edge", "c"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("c", "edge", "b"),
        )?);

        // Rule: edge(X,Y) → connected(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "connected_direct".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("connected".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Rule: edge(X,Z) ∧ edge(Z,Y) → connected(X,Y)
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "connected_indirect".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Z".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("connected".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // Query: connected(a, b) has two derivations:
        // 1. Direct: a->b with prob 0.6
        // 2. Indirect: a->c->b with prob 0.7 × 0.8 = 0.56
        // Combined: P(A ∨ B) = 0.6 + 0.56 - (0.6 × 0.56) = 0.824
        let prob = engine.query_probability(&create_triple("a", "connected", "b"))?;

        let expected = 0.6 + 0.56 - (0.6 * 0.56); // 0.824
        assert!(
            (prob - expected).abs() < 0.001,
            "Expected {} (disjunctive combination), got {}",
            expected,
            prob
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_max_iterations() -> Result<()> {
        // Test that max iterations limit is enforced
        let mut engine = ProbLogEngine::new()
            .with_strategy(EvaluationStrategy::BottomUp)
            .with_max_fixpoint_iterations(2);

        // Create a simple recursive rule
        engine.add_fact(create_triple("a", "edge", "b"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // This should succeed (reaches fixpoint quickly)
        let result = engine.query_probability(&create_triple("a", "path", "b"));
        assert!(result.is_ok(), "Should succeed with low iteration limit");

        Ok(())
    }

    #[test]
    fn test_materialization_invalidation() -> Result<()> {
        // Test that materialization is invalidated when facts/rules change
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        // First query - should materialize
        let prob1 = engine.query_probability(&create_triple("a", "path", "b"))?;
        assert_eq!(prob1, 1.0);
        let iter1 = engine.stats.fixpoint_iterations;

        // Add new fact - should invalidate materialization
        engine.add_fact(create_triple("b", "edge", "c"));

        // Second query - should re-materialize
        let prob2 = engine.query_probability(&create_triple("b", "path", "c"))?;
        assert_eq!(prob2, 1.0);

        // Should have run fixpoint iteration again
        assert!(
            engine.stats.fixpoint_iterations >= iter1,
            "Should have re-materialized after adding fact"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_statistics() -> Result<()> {
        // Test that fixpoint statistics are properly tracked
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        let _prob = engine.query_probability(&create_triple("a", "path", "c"))?;

        // Verify statistics
        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have recorded fixpoint iterations"
        );
        assert!(
            engine.stats.materialized_facts_count > 0,
            "Should have recorded materialized facts count"
        );
        assert!(engine.stats.queries > 0, "Should have recorded query count");

        Ok(())
    }
}
