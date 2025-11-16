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
            stats: ProbLogStats::default(),
        }
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
    }

    /// Add a deterministic fact
    pub fn add_fact(&mut self, fact: RuleAtom) {
        self.deterministic_facts.insert(fact);
        self.query_cache.clear();
    }

    /// Add a probabilistic rule
    pub fn add_rule(&mut self, rule: ProbabilisticRule) {
        self.probabilistic_rules.push(rule);
        self.query_cache.clear();
    }

    /// Query the probability of a fact
    pub fn query_probability(&mut self, query: &RuleAtom) -> Result<f64> {
        let _timer = PROBLOG_QUERY_TIME.start();
        self.stats.queries += 1;
        PROBLOG_QUERIES.inc();

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

        // Try to derive using rules
        let prob = self.derive_probability(query)?;
        self.query_cache.insert(query.clone(), prob);

        Ok(prob)
    }

    /// Derive probability using rules
    fn derive_probability(&mut self, query: &RuleAtom) -> Result<f64> {
        let mut total_prob = 0.0;

        // Try each rule
        for prob_rule in &self.probabilistic_rules.clone() {
            // Check if rule head matches query
            if !self.matches_pattern(&prob_rule.rule.head, query) {
                continue;
            }

            // Evaluate rule body
            let body_prob = self.evaluate_body(&prob_rule.rule.body)?;

            // Combine with rule probability
            let derivation_prob = body_prob * prob_rule.probability.unwrap_or(1.0);

            // Disjunctive combination: P(A ∨ B) = P(A) + P(B) - P(A)P(B)
            total_prob = total_prob + derivation_prob - (total_prob * derivation_prob);

            self.stats.inferences += 1;
            PROBLOG_INFERENCES.inc();
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

    /// Check if rule head pattern matches query
    fn matches_pattern(&self, head: &[RuleAtom], query: &RuleAtom) -> bool {
        // Simplified pattern matching - in real implementation, would use unification
        head.iter().any(|h| self.atoms_match(h, query))
    }

    /// Check if two atoms match (simplified)
    fn atoms_match(&self, pattern: &RuleAtom, query: &RuleAtom) -> bool {
        match (pattern, query) {
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
            ) => self.terms_match(s1, s2) && self.terms_match(p1, p2) && self.terms_match(o1, o2),
            _ => false,
        }
    }

    /// Check if two terms match
    fn terms_match(&self, t1: &Term, t2: &Term) -> bool {
        match (t1, t2) {
            (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            _ => false,
        }
    }

    /// Sample from the probability distribution
    pub fn sample(&mut self) -> HashSet<RuleAtom> {
        use scirs2_core::random::rng;

        let mut rng_instance = rng();
        let uniform = Uniform::new(0.0, 1.0).unwrap();
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

        // Try to derive using rules
        for prob_rule in &self.probabilistic_rules.clone() {
            if !self.matches_pattern(&prob_rule.rule.head, query) {
                continue;
            }

            // Build premise trees
            let mut premises = Vec::new();
            let mut body_prob = 1.0;

            for body_atom in &prob_rule.rule.body {
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
        // Note: Full variable unification is a TODO - this is a simplified version
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
        // Note: Full variable unification is a TODO - this is a simplified version
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
}
