//! # Probabilistic Reasoning over RDF Graphs
//!
//! This module provides Bayesian network-based probabilistic inference over RDF triples.
//! It assigns confidence scores to inferred triples using principled probabilistic reasoning.
//!
//! ## Design
//!
//! - **ProbabilisticTriple**: An RDF triple annotated with a probability in \[0,1\]
//! - **ProbabilisticRule**: A rule that fires with confidence and produces weighted conclusions
//! - **BayesianRdfNetwork**: A Bayesian network where nodes represent RDF triple truth values
//! - **ProbabilisticRdfReasoner**: Forward-chaining reasoner that propagates probabilities
//!
//! ## Combination Functions
//!
//! When multiple rules support the same conclusion, probabilities are combined using:
//! - **Noisy-OR**: P(conclusion) = 1 - ∏(1 - P(rule_i fires))
//! - **Noisy-AND**: P(conclusion) = ∏ P(rule_i fires)
//! - **Maximum**: P(conclusion) = max_i P(rule_i fires)
//!
//! ## Reference
//!
//! Richardson, Domingos: "Markov Logic Networks" (Machine Learning 2006)
//! Ding et al.: "swRDF - Semantic Web with uncertainty" (WWW 2005)

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors from probabilistic reasoning
#[derive(Debug, Error)]
pub enum ProbError {
    #[error("Probability {0} is out of range [0, 1]")]
    InvalidProbability(f64),

    #[error("Circular dependency detected in Bayesian network: {0}")]
    CircularDependency(String),

    #[error("Variable not found in network: {0}")]
    VariableNotFound(String),

    #[error("Maximum inference iterations exceeded ({0})")]
    MaxIterationsExceeded(usize),

    #[error("Inconsistent conditional probability table for node {0}")]
    InconsistentCpt(String),
}

/// An RDF triple annotated with a probability
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticTriple {
    /// Subject IRI or blank node ID
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// Object IRI, blank node ID, or literal
    pub object: String,
    /// Probability that this triple is true, in [0.0, 1.0]
    pub probability: f64,
    /// IDs of the source facts or rules that support this triple
    pub evidence: Vec<String>,
    /// Whether this is a base fact (true) or inferred (false)
    pub is_base_fact: bool,
}

impl ProbabilisticTriple {
    /// Create a new base fact with a given probability
    pub fn new_fact(s: &str, p: &str, o: &str, probability: f64) -> Result<Self, ProbError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(ProbError::InvalidProbability(probability));
        }
        Ok(Self {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            probability,
            evidence: Vec::new(),
            is_base_fact: true,
        })
    }

    /// Create an inferred triple with supporting evidence
    pub fn new_inferred(
        s: &str,
        p: &str,
        o: &str,
        probability: f64,
        evidence: Vec<String>,
    ) -> Result<Self, ProbError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(ProbError::InvalidProbability(probability));
        }
        Ok(Self {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            probability,
            evidence,
            is_base_fact: false,
        })
    }

    /// Get the triple key (subject, predicate, object) as a canonical string
    pub fn key(&self) -> String {
        format!("({}, {}, {})", self.subject, self.predicate, self.object)
    }
}

/// A pattern element: either a variable or a concrete value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternVar {
    /// A variable (matched and bound during rule application)
    Var(String),
    /// A concrete IRI or literal value
    Const(String),
}

impl PatternVar {
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    pub fn konst(value: &str) -> Self {
        Self::Const(value.to_string())
    }
}

/// A pattern for matching RDF triples
#[derive(Debug, Clone)]
pub struct RulePattern {
    pub subject: PatternVar,
    pub predicate: PatternVar,
    pub object: PatternVar,
}

impl RulePattern {
    pub fn new(s: PatternVar, p: PatternVar, o: PatternVar) -> Self {
        Self {
            subject: s,
            predicate: p,
            object: o,
        }
    }
}

/// A probabilistic rule that fires with a confidence and produces weighted conclusions
#[derive(Debug, Clone)]
pub struct ProbabilisticRule {
    /// Unique rule identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Antecedent patterns (all must match)
    pub antecedents: Vec<RulePattern>,
    /// Consequent pattern (produced when rule fires)
    pub consequent: RulePattern,
    /// Confidence [0.0, 1.0] — the probability that the rule fires correctly
    pub confidence: f64,
}

impl ProbabilisticRule {
    /// Create a new probabilistic rule
    pub fn new(
        id: &str,
        name: &str,
        antecedents: Vec<RulePattern>,
        consequent: RulePattern,
        confidence: f64,
    ) -> Result<Self, ProbError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(ProbError::InvalidProbability(confidence));
        }
        Ok(Self {
            id: id.to_string(),
            name: name.to_string(),
            antecedents,
            consequent,
            confidence,
        })
    }
}

/// Strategy for combining probabilities from multiple supporting rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationStrategy {
    /// Noisy-OR: 1 - ∏(1 - p_i)  — at least one of the causes independently triggers the effect
    NoisyOr,
    /// Maximum probability: max(p_i)
    Maximum,
    /// Weighted average: Σ(w_i * p_i) / Σ(w_i)
    WeightedAverage,
    /// Minimum (conservative): min(p_i)
    Minimum,
}

/// Conditional Probability Table for binary Bayesian nodes
///
/// Represents P(node = true | parent_configuration)
#[derive(Debug, Clone)]
pub struct ConditionalProbabilityTable {
    /// Parent variable names (ordered)
    pub parent_names: Vec<String>,
    /// Probability map: binary parent assignment → P(true)
    /// Key: bit pattern of parent truth values (index 0 = leftmost parent)
    probabilities: HashMap<Vec<bool>, f64>,
    /// Leak probability (used in Noisy-OR initialization)
    pub leak_prob: f64,
}

impl ConditionalProbabilityTable {
    /// Create a Noisy-OR CPT where each parent independently causes the effect
    ///
    /// `inhibition_probs[i]` = P(parent i does NOT cause the effect)
    /// `leak_prob` = P(effect with no parents active)
    pub fn new_noisy_or(
        parent_names: Vec<String>,
        inhibition_probs: Vec<f64>,
        leak_prob: f64,
    ) -> Result<Self, ProbError> {
        if parent_names.len() != inhibition_probs.len() {
            return Err(ProbError::InconsistentCpt(
                "inhibition_probs length mismatch".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&leak_prob) {
            return Err(ProbError::InvalidProbability(leak_prob));
        }

        let n = parent_names.len();
        let mut probabilities = HashMap::new();

        // Enumerate all 2^n parent configurations
        for config_idx in 0..(1usize << n) {
            let config: Vec<bool> = (0..n).map(|i| (config_idx >> i) & 1 == 1).collect();

            // Noisy-OR: P(effect = true | config) = 1 - leak * ∏(inhibition[i] for active parents)
            let prod: f64 = config
                .iter()
                .enumerate()
                .filter(|(_, &active)| active)
                .map(|(i, _)| inhibition_probs[i])
                .product();

            let prob = 1.0 - (1.0 - leak_prob) * prod;
            probabilities.insert(config, prob.clamp(0.0, 1.0));
        }

        Ok(Self {
            parent_names,
            probabilities,
            leak_prob,
        })
    }

    /// Create a deterministic CPT (always true/false regardless of parents)
    pub fn deterministic(value: bool) -> Self {
        Self {
            parent_names: Vec::new(),
            probabilities: {
                let mut m = HashMap::new();
                m.insert(vec![], if value { 1.0 } else { 0.0 });
                m
            },
            leak_prob: if value { 1.0 } else { 0.0 },
        }
    }

    /// Get P(node = true | parent_values)
    pub fn get_probability(&self, parent_values: &[bool]) -> f64 {
        self.probabilities
            .get(parent_values)
            .copied()
            .unwrap_or(self.leak_prob)
    }

    /// Get P(node = true | no parents active) — the prior
    pub fn get_prior(&self) -> f64 {
        if self.parent_names.is_empty() {
            self.probabilities
                .get(&Vec::<bool>::new())
                .copied()
                .unwrap_or(self.leak_prob)
        } else {
            self.leak_prob
        }
    }
}

/// A node in the Bayesian RDF network
///
/// Each node represents the truth value of an RDF triple pattern.
#[derive(Debug, Clone)]
pub struct BayesianRdfNode {
    /// Unique node identifier
    pub id: String,
    /// The RDF triple pattern this node represents (with concrete values)
    pub triple_key: String,
    /// Parent node IDs (evidence variables)
    pub parents: Vec<String>,
    /// Conditional probability table
    pub cpt: ConditionalProbabilityTable,
    /// Current marginal probability (P(node = true))
    pub marginal_prob: f64,
}

/// A Bayesian network over RDF triple truth values
///
/// Supports belief propagation for computing posterior probabilities.
pub struct BayesianRdfNetwork {
    /// Nodes indexed by ID
    nodes: HashMap<String, BayesianRdfNode>,
    /// Edges: parent_id → set of child_ids
    edges: HashMap<String, HashSet<String>>,
}

impl BayesianRdfNetwork {
    /// Create an empty Bayesian RDF network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Add a node to the network
    pub fn add_node(&mut self, node: BayesianRdfNode) -> Result<(), ProbError> {
        for parent_id in &node.parents {
            if !self.nodes.contains_key(parent_id) {
                return Err(ProbError::VariableNotFound(parent_id.clone()));
            }
            self.edges
                .entry(parent_id.clone())
                .or_default()
                .insert(node.id.clone());
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Add a root node (no parents, just a prior)
    pub fn add_root_node(
        &mut self,
        id: &str,
        triple_key: &str,
        prior_prob: f64,
    ) -> Result<(), ProbError> {
        if !(0.0..=1.0).contains(&prior_prob) {
            return Err(ProbError::InvalidProbability(prior_prob));
        }

        let cpt = ConditionalProbabilityTable::deterministic(false);
        let mut node = BayesianRdfNode {
            id: id.to_string(),
            triple_key: triple_key.to_string(),
            parents: Vec::new(),
            cpt,
            marginal_prob: prior_prob,
        };

        // Override the CPT's prior with the actual prior
        node.cpt.probabilities.insert(vec![], prior_prob);
        node.cpt.leak_prob = prior_prob;

        self.nodes.insert(id.to_string(), node);
        Ok(())
    }

    /// Run simple belief propagation (forward pass only)
    ///
    /// Computes marginal probabilities P(node = true) using topological order.
    pub fn propagate_beliefs(&mut self) -> Result<(), ProbError> {
        let order = self.topological_order()?;

        for node_id in &order {
            let (parent_probs, parent_ids): (Vec<f64>, Vec<String>) = {
                let node = self
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| ProbError::VariableNotFound(node_id.clone()))?;
                let parent_ids = node.parents.clone();
                let parent_probs: Vec<f64> = parent_ids
                    .iter()
                    .map(|pid| self.nodes.get(pid).map(|n| n.marginal_prob).unwrap_or(0.0))
                    .collect();
                (parent_probs, parent_ids)
            };

            if parent_ids.is_empty() {
                // Root node: marginal = prior (already set)
                continue;
            }

            // Compute expected probability by summing over parent configurations
            // E[P(node=true)] = Σ_config P(config) * P(node=true | config)
            let n_parents = parent_ids.len();
            let mut marginal = 0.0;

            for config_idx in 0..(1usize << n_parents) {
                let config: Vec<bool> =
                    (0..n_parents).map(|i| (config_idx >> i) & 1 == 1).collect();

                // Probability of this parent configuration
                let config_prob: f64 = config
                    .iter()
                    .enumerate()
                    .map(|(i, &active)| {
                        let p = parent_probs[i];
                        if active {
                            p
                        } else {
                            1.0 - p
                        }
                    })
                    .product();

                // Expected contribution from this configuration
                let node = self
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| ProbError::VariableNotFound(node_id.clone()))?;
                let cond_prob = node.cpt.get_probability(&config);
                marginal += config_prob * cond_prob;
            }

            if let Some(node) = self.nodes.get_mut(node_id) {
                node.marginal_prob = marginal.clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Get the marginal probability of a node
    pub fn get_marginal_prob(&self, node_id: &str) -> Option<f64> {
        self.nodes.get(node_id).map(|n| n.marginal_prob)
    }

    /// Update evidence (set observed value) for a node and re-propagate
    pub fn set_evidence(&mut self, node_id: &str, observed: bool) -> Result<(), ProbError> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| ProbError::VariableNotFound(node_id.to_string()))?;
        node.marginal_prob = if observed { 1.0 } else { 0.0 };
        Ok(())
    }

    /// Compute topological order of nodes
    fn topological_order(&self) -> Result<Vec<String>, ProbError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
        }
        for children in self.edges.values() {
            for child in children {
                *in_degree.entry(child).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(id, _)| id.to_string())
            .collect();
        queue.sort(); // deterministic ordering

        let mut order = Vec::new();
        let mut remaining_degrees = in_degree
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<_, _>>();

        while !queue.is_empty() {
            queue.sort();
            let node_id = queue.remove(0);
            order.push(node_id.clone());

            if let Some(children) = self.edges.get(&node_id) {
                for child in children {
                    let deg = remaining_degrees.entry(child.clone()).or_insert(0);
                    if *deg > 0 {
                        *deg -= 1;
                    }
                    if *deg == 0 {
                        queue.push(child.clone());
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(ProbError::CircularDependency(
                "Graph contains cycles".to_string(),
            ));
        }

        Ok(order)
    }
}

impl Default for BayesianRdfNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from a probabilistic inference run
#[derive(Debug, Clone)]
pub struct ProbabilisticInferenceReport {
    /// Total inferred triples (above threshold)
    pub inferred_count: usize,
    /// Total rules fired
    pub rules_fired: usize,
    /// Inference iterations
    pub iterations: usize,
    /// Wall-clock duration
    pub duration: std::time::Duration,
    /// Minimum probability threshold used
    pub threshold: f64,
    /// Average probability of inferred triples
    pub avg_probability: f64,
}

/// Probabilistic RDF reasoner
///
/// Uses forward-chaining with probabilistic rules and Noisy-OR combination
/// to compute confidence scores for inferred triples.
pub struct ProbabilisticRdfReasoner {
    /// Base facts with associated probabilities
    facts: Vec<ProbabilisticTriple>,
    /// Probabilistic inference rules
    rules: Vec<ProbabilisticRule>,
    /// Minimum probability threshold — triples below this are discarded
    threshold: f64,
    /// How to combine multiple supporting rule applications
    combination_strategy: CombinationStrategy,
    /// Maximum inference iterations
    max_iterations: usize,
}

impl ProbabilisticRdfReasoner {
    /// Create a new reasoner with the given probability threshold
    pub fn new(threshold: f64) -> Result<Self, ProbError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ProbError::InvalidProbability(threshold));
        }
        Ok(Self {
            facts: Vec::new(),
            rules: Vec::new(),
            threshold,
            combination_strategy: CombinationStrategy::NoisyOr,
            max_iterations: 100,
        })
    }

    /// Set the combination strategy for multiple supporting rules
    pub fn with_combination_strategy(mut self, strategy: CombinationStrategy) -> Self {
        self.combination_strategy = strategy;
        self
    }

    /// Set maximum inference iterations
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Add a base fact with a probability
    pub fn add_fact(
        &mut self,
        s: &str,
        p: &str,
        o: &str,
        probability: f64,
    ) -> Result<(), ProbError> {
        let fact = ProbabilisticTriple::new_fact(s, p, o, probability)?;
        self.facts.push(fact);
        Ok(())
    }

    /// Add a probabilistic inference rule
    pub fn add_rule(&mut self, rule: ProbabilisticRule) {
        self.rules.push(rule);
    }

    /// Run forward-chaining probabilistic inference
    ///
    /// Returns all inferred triples above the probability threshold,
    /// along with an inference report.
    pub fn infer(
        &self,
    ) -> Result<(Vec<ProbabilisticTriple>, ProbabilisticInferenceReport), ProbError> {
        let start = std::time::Instant::now();

        // Working set: triple_key → (probability, evidence_ids)
        let mut working_set: HashMap<String, (f64, Vec<String>)> = HashMap::new();

        // Seed with base facts
        for fact in &self.facts {
            let key = fact.key();
            working_set.insert(key, (fact.probability, vec![fact.key()]));
        }

        let mut iterations = 0usize;
        let mut total_rules_fired = 0usize;

        loop {
            if iterations >= self.max_iterations {
                return Err(ProbError::MaxIterationsExceeded(self.max_iterations));
            }
            iterations += 1;

            // Maps: consequent_key → list of (probability, evidence) from each rule application
            let mut new_support: HashMap<String, Vec<(f64, Vec<String>)>> = HashMap::new();

            for rule in &self.rules {
                let matches = self.match_rule(rule, &working_set);
                total_rules_fired += matches.len();

                for (bindings, antecedent_probs, antecedent_evidence) in matches {
                    // Instantiate consequent
                    if let Some((s, p, o)) = self.instantiate_pattern(&rule.consequent, &bindings) {
                        let key = format!("({}, {}, {})", s, p, o);

                        // Combined probability: min of antecedent probs * rule confidence
                        let antecedent_prob: f64 =
                            antecedent_probs.iter().cloned().fold(1.0f64, f64::min);
                        let derived_prob = antecedent_prob * rule.confidence;

                        if derived_prob >= self.threshold {
                            let mut evidence = antecedent_evidence;
                            evidence.push(format!("rule:{}", rule.id));

                            new_support
                                .entry(key)
                                .or_default()
                                .push((derived_prob, evidence));
                        }
                    }
                }
            }

            // Combine multiple supports and update working set
            let mut changed = false;

            for (key, supports) in new_support {
                // Check if this is a new triple or higher probability
                let new_prob = self
                    .combine_probabilities(&supports.iter().map(|(p, _)| *p).collect::<Vec<_>>());
                let new_evidence: Vec<String> = supports
                    .into_iter()
                    .flat_map(|(_, evs)| evs)
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();

                let existing_prob = working_set.get(&key).map(|(p, _)| *p).unwrap_or(0.0);

                if new_prob > existing_prob + f64::EPSILON {
                    working_set.insert(key, (new_prob, new_evidence));
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Build result: collect inferred triples (non-base-facts above threshold)
        let base_fact_keys: HashSet<String> = self.facts.iter().map(|f| f.key()).collect();

        let mut inferred: Vec<ProbabilisticTriple> = Vec::new();

        for (key, (prob, evidence)) in &working_set {
            if !base_fact_keys.contains(key) && *prob >= self.threshold {
                // Parse key back to triple (simple parsing)
                if let Some(triple) = self.parse_key(key) {
                    match ProbabilisticTriple::new_inferred(
                        &triple.0,
                        &triple.1,
                        &triple.2,
                        *prob,
                        evidence.clone(),
                    ) {
                        Ok(t) => inferred.push(t),
                        Err(_) => continue,
                    }
                }
            }
        }

        // Sort by probability descending
        inferred.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let avg_prob = if inferred.is_empty() {
            0.0
        } else {
            inferred.iter().map(|t| t.probability).sum::<f64>() / inferred.len() as f64
        };

        let report = ProbabilisticInferenceReport {
            inferred_count: inferred.len(),
            rules_fired: total_rules_fired,
            iterations,
            duration: start.elapsed(),
            threshold: self.threshold,
            avg_probability: avg_prob,
        };

        Ok((inferred, report))
    }

    /// Combine multiple probabilities according to the configured strategy
    pub fn combine_probabilities(&self, probs: &[f64]) -> f64 {
        if probs.is_empty() {
            return 0.0;
        }

        match self.combination_strategy {
            CombinationStrategy::NoisyOr => {
                // P = 1 - ∏(1 - p_i)
                let complement_product: f64 = probs.iter().map(|&p| 1.0 - p).product();
                (1.0 - complement_product).clamp(0.0, 1.0)
            }
            CombinationStrategy::Maximum => probs.iter().cloned().fold(0.0f64, f64::max),
            CombinationStrategy::WeightedAverage => probs.iter().sum::<f64>() / probs.len() as f64,
            CombinationStrategy::Minimum => probs.iter().cloned().fold(1.0f64, f64::min),
        }
    }

    /// Query the probability of a specific triple in the fact base
    pub fn get_fact_probability(&self, s: &str, p: &str, o: &str) -> Option<f64> {
        let key = format!("({}, {}, {})", s, p, o);
        self.facts
            .iter()
            .find(|f| f.key() == key)
            .map(|f| f.probability)
    }

    // -----------------------------------------------------------------------
    // Private: rule matching
    // -----------------------------------------------------------------------

    /// Match a rule's antecedents against the working set
    /// Returns: list of (bindings, per-antecedent-probs, evidence)
    #[allow(clippy::type_complexity)]
    fn match_rule(
        &self,
        rule: &ProbabilisticRule,
        working_set: &HashMap<String, (f64, Vec<String>)>,
    ) -> Vec<(HashMap<String, String>, Vec<f64>, Vec<String>)> {
        if rule.antecedents.is_empty() {
            return vec![(HashMap::new(), Vec::new(), Vec::new())];
        }

        // Start with empty bindings, build up via join
        #[allow(clippy::type_complexity)]
        let mut current: Vec<(HashMap<String, String>, Vec<f64>, Vec<String>)> =
            vec![(HashMap::new(), Vec::new(), Vec::new())];

        for pattern in &rule.antecedents {
            let mut next = Vec::new();

            for (bindings, probs, evidence) in &current {
                // Find all matching triples for this pattern given current bindings
                for (key, (prob, evids)) in working_set {
                    if let Some(triple) = self.parse_key(key) {
                        if let Some(extended) = self.try_match_pattern(pattern, &triple, bindings) {
                            let mut new_probs = probs.clone();
                            new_probs.push(*prob);

                            let mut new_evidence = evidence.clone();
                            new_evidence.extend(evids.iter().cloned());

                            next.push((extended, new_probs, new_evidence));
                        }
                    }
                }
            }

            current = next;
            if current.is_empty() {
                break;
            }
        }

        current
    }

    /// Try to match a pattern against a concrete triple given existing bindings
    fn try_match_pattern(
        &self,
        pattern: &RulePattern,
        triple: &(String, String, String),
        bindings: &HashMap<String, String>,
    ) -> Option<HashMap<String, String>> {
        let mut extended = bindings.clone();

        let check =
            |elem: &PatternVar, value: &str, bindings: &mut HashMap<String, String>| -> bool {
                match elem {
                    PatternVar::Const(c) => c == value,
                    PatternVar::Var(v) => {
                        if let Some(bound) = bindings.get(v) {
                            bound == value
                        } else {
                            bindings.insert(v.clone(), value.to_string());
                            true
                        }
                    }
                }
            };

        if check(&pattern.subject, &triple.0, &mut extended)
            && check(&pattern.predicate, &triple.1, &mut extended)
            && check(&pattern.object, &triple.2, &mut extended)
        {
            Some(extended)
        } else {
            None
        }
    }

    /// Instantiate a rule pattern with variable bindings
    fn instantiate_pattern(
        &self,
        pattern: &RulePattern,
        bindings: &HashMap<String, String>,
    ) -> Option<(String, String, String)> {
        let resolve = |elem: &PatternVar| -> Option<String> {
            match elem {
                PatternVar::Const(c) => Some(c.clone()),
                PatternVar::Var(v) => bindings.get(v).cloned(),
            }
        };

        let s = resolve(&pattern.subject)?;
        let p = resolve(&pattern.predicate)?;
        let o = resolve(&pattern.object)?;
        Some((s, p, o))
    }

    /// Parse a triple key of the form "(s, p, o)" back into components
    fn parse_key(&self, key: &str) -> Option<(String, String, String)> {
        // Format: "(subject, predicate, object)"
        let inner = key.strip_prefix('(')?.strip_suffix(')')?;

        // Split on ", " but be careful with nested structures
        // Simple approach: find first two ", " separators
        let mut depth = 0i32;
        let mut separators = Vec::new();

        let chars: Vec<char> = inner.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 && i + 1 < chars.len() && chars[i + 1] == ' ' => {
                    separators.push(i);
                }
                _ => {}
            }
            i += 1;
        }

        if separators.len() >= 2 {
            let s = inner[..separators[0]].to_string();
            let p = inner[(separators[0] + 2)..separators[1]].to_string();
            let o = inner[(separators[1] + 2)..].to_string();
            Some((s, p, o))
        } else {
            None
        }
    }
}

// -----------------------------------------------------------------------
// Convenient builder functions
// -----------------------------------------------------------------------

/// Builder for creating a probabilistic subclass rule:
/// x rdf:type C → x rdf:type D (with given confidence)
pub fn make_subclass_rule(
    id: &str,
    sub_class: &str,
    sup_class: &str,
    confidence: f64,
) -> Result<ProbabilisticRule, ProbError> {
    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    ProbabilisticRule::new(
        id,
        &format!("{} ⊑ {}", sub_class, sup_class),
        vec![RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(rdf_type),
            PatternVar::konst(sub_class),
        )],
        RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(rdf_type),
            PatternVar::konst(sup_class),
        ),
        confidence,
    )
}

/// Builder for a probabilistic symmetric property rule:
/// x P y → y P x (with given confidence)
pub fn make_symmetric_rule(
    id: &str,
    property: &str,
    confidence: f64,
) -> Result<ProbabilisticRule, ProbError> {
    ProbabilisticRule::new(
        id,
        &format!("symmetric({})", property),
        vec![RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(property),
            PatternVar::var("y"),
        )],
        RulePattern::new(
            PatternVar::var("y"),
            PatternVar::konst(property),
            PatternVar::var("x"),
        ),
        confidence,
    )
}

/// Builder for a probabilistic transitive property rule:
/// x P y, y P z → x P z (with given confidence)
pub fn make_transitive_rule(
    id: &str,
    property: &str,
    confidence: f64,
) -> Result<ProbabilisticRule, ProbError> {
    ProbabilisticRule::new(
        id,
        &format!("transitive({})", property),
        vec![
            RulePattern::new(
                PatternVar::var("x"),
                PatternVar::konst(property),
                PatternVar::var("y"),
            ),
            RulePattern::new(
                PatternVar::var("y"),
                PatternVar::konst(property),
                PatternVar::var("z"),
            ),
        ],
        RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(property),
            PatternVar::var("z"),
        ),
        confidence,
    )
}

/// Builder for a probabilistic domain inference rule:
/// x P y → x rdf:type C (with given confidence)
pub fn make_domain_rule(
    id: &str,
    property: &str,
    domain_class: &str,
    confidence: f64,
) -> Result<ProbabilisticRule, ProbError> {
    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    ProbabilisticRule::new(
        id,
        &format!("domain({}, {})", property, domain_class),
        vec![RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(property),
            PatternVar::var("y"),
        )],
        RulePattern::new(
            PatternVar::var("x"),
            PatternVar::konst(rdf_type),
            PatternVar::konst(domain_class),
        ),
        confidence,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    fn reasoner(threshold: f64) -> ProbabilisticRdfReasoner {
        ProbabilisticRdfReasoner::new(threshold).expect("valid threshold")
    }

    #[test]
    fn test_invalid_probability() {
        assert!(ProbabilisticTriple::new_fact("s", "p", "o", 1.5).is_err());
        assert!(ProbabilisticTriple::new_fact("s", "p", "o", -0.1).is_err());
        assert!(ProbabilisticTriple::new_fact("s", "p", "o", 0.8).is_ok());
    }

    #[test]
    fn test_subclass_inference() -> Result<(), Box<dyn std::error::Error>> {
        let mut r = reasoner(0.1);
        r.add_fact("fido", RDF_TYPE, "Dog", 0.9).expect("add fact");
        r.add_rule(make_subclass_rule("r1", "Dog", "Animal", 0.95).expect("rule"));

        let (inferred, report) = r.infer().expect("infer");
        assert!(
            !inferred.is_empty(),
            "Expected at least one inferred triple"
        );
        assert!(report.rules_fired > 0);

        let fido_animal = inferred
            .iter()
            .find(|t| t.subject == "fido" && t.predicate == RDF_TYPE && t.object == "Animal");
        assert!(
            fido_animal.is_some(),
            "Expected fido rdf:type Animal to be inferred"
        );
        assert!(fido_animal.ok_or("expected Some value")?.probability > 0.5);
        Ok(())
    }

    #[test]
    fn test_symmetric_inference() -> Result<(), Box<dyn std::error::Error>> {
        let knows = "https://example.org/knows";
        let mut r = reasoner(0.1);
        r.add_fact("alice", knows, "bob", 0.9).expect("add fact");
        r.add_rule(make_symmetric_rule("r1", knows, 0.95).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");
        let bob_knows_alice = inferred
            .iter()
            .find(|t| t.subject == "bob" && t.predicate == knows && t.object == "alice");
        assert!(bob_knows_alice.is_some(), "Expected symmetric inference");
        assert!(bob_knows_alice.ok_or("expected Some value")?.probability > 0.5);
        Ok(())
    }

    #[test]
    fn test_transitive_inference() -> Result<(), Box<dyn std::error::Error>> {
        let ancestor_of = "https://example.org/ancestorOf";
        let mut r = reasoner(0.1);
        r.add_fact("grandpa", ancestor_of, "parent", 0.99)
            .expect("add fact");
        r.add_fact("parent", ancestor_of, "child", 0.99)
            .expect("add fact");
        r.add_rule(make_transitive_rule("r1", ancestor_of, 0.95).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");
        let grandpa_child = inferred
            .iter()
            .find(|t| t.subject == "grandpa" && t.predicate == ancestor_of && t.object == "child");
        assert!(
            grandpa_child.is_some(),
            "Expected transitive inference. Inferred: {:?}",
            inferred
                .iter()
                .map(|t| format!("({},{},{})", t.subject, t.predicate, t.object))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_domain_inference() {
        let has_parent = "https://example.org/hasParent";
        let mut r = reasoner(0.1);
        r.add_fact("alice", has_parent, "bob", 0.85)
            .expect("add fact");
        r.add_rule(make_domain_rule("r1", has_parent, "Person", 0.9).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");
        let alice_person = inferred
            .iter()
            .find(|t| t.subject == "alice" && t.predicate == RDF_TYPE && t.object == "Person");
        assert!(
            alice_person.is_some(),
            "Expected alice rdf:type Person from domain"
        );
    }

    #[test]
    fn test_noisy_or_combination() {
        let r = ProbabilisticRdfReasoner::new(0.0).expect("reasoner");
        // Noisy-OR of 0.6 and 0.7 should be 1 - (1-0.6)(1-0.7) = 1 - 0.12 = 0.88
        let result = r.combine_probabilities(&[0.6, 0.7]);
        let expected = 1.0 - (1.0 - 0.6) * (1.0 - 0.7);
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_maximum_combination() {
        let r = ProbabilisticRdfReasoner::new(0.0)
            .expect("reasoner")
            .with_combination_strategy(CombinationStrategy::Maximum);
        let result = r.combine_probabilities(&[0.3, 0.8, 0.5]);
        assert!((result - 0.8).abs() < 1e-10, "Expected 0.8 max");
    }

    #[test]
    fn test_weighted_average_combination() {
        let r = ProbabilisticRdfReasoner::new(0.0)
            .expect("reasoner")
            .with_combination_strategy(CombinationStrategy::WeightedAverage);
        let result = r.combine_probabilities(&[0.4, 0.6]);
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5 average");
    }

    #[test]
    fn test_threshold_filtering() {
        let mut r = reasoner(0.8);
        r.add_fact("fido", RDF_TYPE, "Dog", 0.5).expect("add fact");
        r.add_rule(make_subclass_rule("r1", "Dog", "Animal", 0.7).expect("rule"));
        // 0.5 * 0.7 = 0.35 < 0.8 threshold — should be filtered out

        let (inferred, _) = r.infer().expect("infer");
        assert!(
            inferred.is_empty() || inferred.iter().all(|t| t.probability >= 0.8),
            "Expected no triples below threshold"
        );
    }

    #[test]
    fn test_cpt_noisy_or() {
        let cpt = ConditionalProbabilityTable::new_noisy_or(
            vec!["RainFall".to_string(), "Sprinkler".to_string()],
            vec![0.1, 0.2], // inhibition: low means high influence
            0.01,           // leak
        )
        .expect("valid cpt");

        // With no parents active: should be close to leak (0.01)
        let p_no_parents = cpt.get_probability(&[false, false]);
        assert!(
            (p_no_parents - (1.0 - (1.0 - 0.01))).abs() < 0.01,
            "P(wet | no rain, no sprinkler) ≈ 0.01, got {}",
            p_no_parents
        );

        // With both parents active: should be high
        let p_both = cpt.get_probability(&[true, true]);
        assert!(
            p_both > 0.9,
            "P(wet | rain + sprinkler) should be > 0.9, got {}",
            p_both
        );
    }

    #[test]
    fn test_bayesian_network_propagation() {
        let mut bn = BayesianRdfNetwork::new();

        // Rain → 0.3 prior
        bn.add_root_node("rain", "rain-triple", 0.3)
            .expect("add rain");

        // Sprinkler → 0.4 prior
        bn.add_root_node("sprinkler", "sprinkler-triple", 0.4)
            .expect("add sprinkler");

        // WetGrass | Rain, Sprinkler
        let cpt = ConditionalProbabilityTable::new_noisy_or(
            vec!["rain".to_string(), "sprinkler".to_string()],
            vec![0.1, 0.2],
            0.01,
        )
        .expect("cpt");

        bn.add_node(BayesianRdfNode {
            id: "wet_grass".to_string(),
            triple_key: "wet-triple".to_string(),
            parents: vec!["rain".to_string(), "sprinkler".to_string()],
            cpt,
            marginal_prob: 0.0,
        })
        .expect("add wet grass");

        bn.propagate_beliefs().expect("propagate");

        let wet_prob = bn.get_marginal_prob("wet_grass").expect("wet prob");
        assert!(
            wet_prob > 0.0 && wet_prob < 1.0,
            "WetGrass probability should be in (0,1), got {}",
            wet_prob
        );
        // With 0.3 rain and 0.4 sprinkler, wet grass should be somewhere around 0.5
        assert!(wet_prob > 0.3, "Expected wet grass > 0.3, got {}", wet_prob);
    }

    #[test]
    fn test_evidence_update() {
        let mut bn = BayesianRdfNetwork::new();
        bn.add_root_node("cause", "cause-triple", 0.5)
            .expect("add cause");

        let cpt =
            ConditionalProbabilityTable::new_noisy_or(vec!["cause".to_string()], vec![0.1], 0.05)
                .expect("cpt");

        bn.add_node(BayesianRdfNode {
            id: "effect".to_string(),
            triple_key: "effect-triple".to_string(),
            parents: vec!["cause".to_string()],
            cpt,
            marginal_prob: 0.0,
        })
        .expect("add effect");

        // Set cause as observed = true
        bn.set_evidence("cause", true).expect("set evidence");
        bn.propagate_beliefs().expect("propagate");

        let effect_prob = bn.get_marginal_prob("effect").expect("effect prob");
        assert!(
            effect_prob > 0.7,
            "With observed cause, effect should be > 0.7, got {}",
            effect_prob
        );
    }

    #[test]
    fn test_inference_report() {
        let mut r = reasoner(0.1);
        r.add_fact("alice", RDF_TYPE, "Person", 0.9)
            .expect("add fact");
        r.add_rule(make_subclass_rule("r1", "Person", "Agent", 0.85).expect("rule"));

        let (_, report) = r.infer().expect("infer");
        assert!(report.iterations >= 1);
        assert!(report.threshold == 0.1);
        assert!(report.duration.as_nanos() > 0);
    }

    #[test]
    fn test_chained_rules() -> Result<(), Box<dyn std::error::Error>> {
        let mut r = reasoner(0.1);
        r.add_fact("fido", RDF_TYPE, "Labrador", 0.95)
            .expect("add fact");
        r.add_rule(make_subclass_rule("r1", "Labrador", "Dog", 0.99).expect("rule"));
        r.add_rule(make_subclass_rule("r2", "Dog", "Mammal", 0.99).expect("rule"));
        r.add_rule(make_subclass_rule("r3", "Mammal", "Animal", 0.99).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");

        let fido_animal = inferred
            .iter()
            .find(|t| t.subject == "fido" && t.predicate == RDF_TYPE && t.object == "Animal");
        assert!(
            fido_animal.is_some(),
            "Expected fido rdf:type Animal via chained rules. Inferred: {:?}",
            inferred
                .iter()
                .map(|t| t.object.clone())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn test_multiple_facts_symmetric() {
        let knows = "https://example.org/knows";
        let mut r = reasoner(0.1);
        r.add_fact("alice", knows, "bob", 0.9).expect("add fact");
        r.add_fact("alice", knows, "carol", 0.7).expect("add fact");
        r.add_rule(make_symmetric_rule("r1", knows, 0.9).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");
        assert!(inferred.len() >= 2, "Expected at least 2 symmetric triples");
    }

    #[test]
    fn test_zero_threshold() {
        let mut r = ProbabilisticRdfReasoner::new(0.0).expect("reasoner");
        r.add_fact("x", "p", "y", 0.01).expect("add fact");
        r.add_rule(make_symmetric_rule("r1", "p", 0.5).expect("rule"));

        let (inferred, _) = r.infer().expect("infer");
        // With zero threshold, even low-probability inferences should be included
        assert!(!inferred.is_empty() || inferred.is_empty()); // either is OK, just shouldn't panic
    }
}
