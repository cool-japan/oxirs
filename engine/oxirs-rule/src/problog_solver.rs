//! Probabilistic Datalog (ProbLog) — top-level solver
//!
//! This module contains the top-level solver: stratified probabilistic evaluation,
//! marginal inference, Monte Carlo estimation, and explanation building.

use crate::RuleAtom;
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Timer};
use scirs2_core::random::{Distribution, Uniform};
use std::collections::{HashMap, HashSet};

use super::problog_inference::{apply_substitution_to_body, materialize, unify_atoms};
use super::problog_types::{
    DerivationTree, EvaluationStrategy, ProbLogStats, ProbabilisticFact, ProbabilisticRule,
};

// Global metrics for ProbLog
lazy_static::lazy_static! {
    static ref PROBLOG_QUERIES: Counter = Counter::new("problog_queries".to_string());
    static ref PROBLOG_INFERENCES: Counter = Counter::new("problog_inferences".to_string());
    static ref PROBLOG_QUERY_TIME: Timer = Timer::new("problog_query_time".to_string());
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
            self.deterministic_facts.insert(fact.fact);
        } else {
            self.probabilistic_facts.insert(fact.fact, fact.probability);
        }
        self.query_cache.clear();
        self.materialization_valid = false;
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

        let use_bottom_up = match self.strategy {
            EvaluationStrategy::TopDown => false,
            EvaluationStrategy::BottomUp => true,
            EvaluationStrategy::Auto => self.has_recursive_rules(),
        };

        if use_bottom_up {
            return self.query_materialized(query);
        }

        if self.current_depth > self.max_depth {
            return Err(anyhow!(
                "Maximum recursion depth exceeded: {}",
                self.max_depth
            ));
        }

        if self.recursion_stack.contains(query) {
            return self.query_materialized(query);
        }

        if let Some(&prob) = self.query_cache.get(query) {
            self.stats.cache_hits += 1;
            return Ok(prob);
        }

        self.stats.cache_misses += 1;

        if self.deterministic_facts.contains(query) {
            self.query_cache.insert(query.clone(), 1.0);
            return Ok(1.0);
        }

        if let Some(&prob) = self.probabilistic_facts.get(query) {
            self.query_cache.insert(query.clone(), prob);
            return Ok(prob);
        }

        self.recursion_stack.insert(query.clone());
        self.current_depth += 1;

        let prob = self.derive_probability(query)?;
        self.query_cache.insert(query.clone(), prob);

        self.recursion_stack.remove(query);
        self.current_depth -= 1;

        Ok(prob)
    }

    /// Check if the engine has recursive rules
    fn has_recursive_rules(&self) -> bool {
        for rule in &self.probabilistic_rules {
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

    /// Derive probability using rules with full variable unification (top-down)
    fn derive_probability(&mut self, query: &RuleAtom) -> Result<f64> {
        let mut total_prob = 0.0;

        for prob_rule in &self.probabilistic_rules.clone() {
            for head_atom in &prob_rule.rule.head {
                if let Some(substitution) = unify_atoms(head_atom, query) {
                    let instantiated_body =
                        apply_substitution_to_body(&prob_rule.rule.body, &substitution);

                    let body_prob = self.evaluate_body(&instantiated_body)?;
                    let derivation_prob = body_prob * prob_rule.probability.unwrap_or(1.0);

                    total_prob = total_prob + derivation_prob - (total_prob * derivation_prob);

                    self.stats.inferences += 1;
                    PROBLOG_INFERENCES.inc();
                }
            }
        }

        Ok(total_prob)
    }

    /// Evaluate rule body probability (conjunction, assuming independence)
    fn evaluate_body(&mut self, body: &[RuleAtom]) -> Result<f64> {
        let mut prob = 1.0;

        for atom in body {
            let atom_prob = self.query_probability(atom)?;
            prob *= atom_prob;
        }

        Ok(prob)
    }

    /// Compute fixpoint using bottom-up materialization
    pub fn materialize(&mut self) -> Result<()> {
        if self.materialization_valid {
            return Ok(());
        }

        self.materialized_facts.clear();

        let result = materialize(
            &self.probabilistic_facts,
            &self.deterministic_facts,
            &self.probabilistic_rules,
            self.max_fixpoint_iterations,
            &mut self.stats,
        )?;

        self.materialized_facts = result;
        self.materialization_valid = true;

        Ok(())
    }

    /// Query using bottom-up evaluation (requires materialization)
    pub fn query_materialized(&mut self, query: &RuleAtom) -> Result<f64> {
        if !self.materialization_valid {
            self.materialize()?;
        }

        let prob = self.materialized_facts.get(query).copied().unwrap_or(0.0);
        Ok(prob)
    }

    /// Sample from the probability distribution
    pub fn sample(&mut self) -> HashSet<RuleAtom> {
        use scirs2_core::random::rng;

        let mut rng_instance = rng();
        let uniform = Uniform::new(0.0, 1.0).expect("distribution parameters are valid");
        let mut sampled_facts = HashSet::new();

        for (fact, &prob) in &self.probabilistic_facts {
            if uniform.sample(&mut rng_instance) < prob {
                sampled_facts.insert(fact.clone());
            }
        }

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
        if self.deterministic_facts.contains(query) {
            return Ok(Some(DerivationTree::leaf(query.clone(), 1.0)));
        }

        if let Some(&prob) = self.probabilistic_facts.get(query) {
            return Ok(Some(DerivationTree::leaf(query.clone(), prob)));
        }

        for prob_rule in &self.probabilistic_rules.clone() {
            for head_atom in &prob_rule.rule.head {
                if let Some(substitution) = unify_atoms(head_atom, query) {
                    let instantiated_body =
                        apply_substitution_to_body(&prob_rule.rule.body, &substitution);

                    let mut premises = Vec::new();
                    let mut body_prob = 1.0;

                    for body_atom in &instantiated_body {
                        if let Some(tree) = self.explain(body_atom)? {
                            body_prob *= tree.probability;
                            premises.push(tree);
                        } else {
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
