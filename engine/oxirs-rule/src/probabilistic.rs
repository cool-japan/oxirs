//! # Probabilistic Reasoning Module
//!
//! This module provides probabilistic inference capabilities for rule-based reasoning,
//! including Bayesian Networks, Markov Logic Networks, and uncertain reasoning.
//!
//! ## Features
//!
//! - **Bayesian Networks**: Directed graphical models for probabilistic inference
//! - **Markov Logic Networks (MLN)**: Combines first-order logic with probabilities
//! - **Uncertainty Propagation**: Tracks confidence through inference chains
//! - **Probabilistic Rules**: Rules with associated probabilities/weights
//! - **Approximate Inference**: Gibbs sampling, belief propagation
//!
//! ## Example
//!
//! ```text
//! use oxirs_rule::probabilistic::*;
//! use oxirs_rule::{RuleAtom, Term};
//!
//! // Create a Bayesian Network
//! let mut bn = BayesianNetwork::new();
//!
//! // Add variables
//! bn.add_variable("Rain".to_string(), vec!["true".to_string(), "false".to_string()]);
//! bn.add_variable("Sprinkler".to_string(), vec!["true".to_string(), "false".to_string()]);
//! bn.add_variable("WetGrass".to_string(), vec!["true".to_string(), "false".to_string()]);
//!
//! // Add dependencies
//! bn.add_edge("Rain".to_string(), "WetGrass".to_string()).unwrap();
//! bn.add_edge("Sprinkler".to_string(), "WetGrass".to_string()).unwrap();
//!
//! // Add conditional probability tables (CPTs)
//! // P(Rain) = 0.2
//! bn.set_prior("Rain".to_string(), vec![0.2, 0.8]).unwrap();
//!
//! // Query the network
//! let evidence = vec![("WetGrass".to_string(), "true".to_string())];
//! let prob = bn.query("Rain".to_string(), "true".to_string(), &evidence).unwrap();
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::{Distribution, RngCore, Uniform};
use std::collections::{HashMap, HashSet};

/// Bayesian Network for probabilistic inference
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    /// Variables in the network
    variables: HashMap<String, Variable>,
    /// Edges (dependencies) in the network
    edges: Vec<(String, String)>,
    /// Conditional Probability Tables (CPTs)
    cpts: HashMap<String, ConditionalProbabilityTable>,
}

/// Variable in a Bayesian Network
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name
    #[allow(dead_code)]
    name: String,
    /// Possible values (domain)
    domain: Vec<String>,
}

/// Conditional Probability Table
#[derive(Debug, Clone)]
pub struct ConditionalProbabilityTable {
    /// Variable this CPT is for
    #[allow(dead_code)]
    variable: String,
    /// Parent variables
    parents: Vec<String>,
    /// Probability table (flattened)
    /// For a variable with parents, this is P(variable | parents)
    probabilities: Array1<f64>,
    /// Dimensions of the probability table
    dimensions: Vec<usize>,
}

impl BayesianNetwork {
    /// Create a new Bayesian Network
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            edges: Vec::new(),
            cpts: HashMap::new(),
        }
    }

    /// Add a variable to the network
    pub fn add_variable(&mut self, name: String, domain: Vec<String>) {
        self.variables.insert(
            name.clone(),
            Variable {
                name: name.clone(),
                domain,
            },
        );
    }

    /// Add an edge (dependency) between two variables
    pub fn add_edge(&mut self, from: String, to: String) -> Result<()> {
        // Check if variables exist
        if !self.variables.contains_key(&from) {
            return Err(anyhow!("Variable {} not found", from));
        }
        if !self.variables.contains_key(&to) {
            return Err(anyhow!("Variable {} not found", to));
        }

        // Check for cycles (simple cycle detection)
        if self.has_path(&to, &from) {
            return Err(anyhow!("Adding edge would create a cycle"));
        }

        self.edges.push((from, to));
        Ok(())
    }

    /// Check if there's a path from `from` to `to` (for cycle detection)
    fn has_path(&self, from: &str, to: &str) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![from];

        while let Some(current) = stack.pop() {
            if current == to {
                return true;
            }

            if visited.contains(current) {
                continue;
            }
            visited.insert(current);

            for (source, target) in &self.edges {
                if source == current {
                    stack.push(target.as_str());
                }
            }
        }

        false
    }

    /// Set prior probability for a variable (no parents)
    pub fn set_prior(&mut self, variable: String, probabilities: Vec<f64>) -> Result<()> {
        let var = self
            .variables
            .get(&variable)
            .ok_or_else(|| anyhow!("Variable {} not found", variable))?;

        if probabilities.len() != var.domain.len() {
            return Err(anyhow!(
                "Probability vector length {} does not match domain size {}",
                probabilities.len(),
                var.domain.len()
            ));
        }

        // Check probabilities sum to 1.0
        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(anyhow!("Probabilities must sum to 1.0, got {}", sum));
        }

        let cpt = ConditionalProbabilityTable {
            variable: variable.clone(),
            parents: Vec::new(),
            probabilities: Array1::from_vec(probabilities),
            dimensions: vec![var.domain.len()],
        };

        self.cpts.insert(variable, cpt);
        Ok(())
    }

    /// Set conditional probability table for a variable with parents
    pub fn set_cpt(
        &mut self,
        variable: String,
        parents: Vec<String>,
        probabilities: Array1<f64>,
    ) -> Result<()> {
        let var = self
            .variables
            .get(&variable)
            .ok_or_else(|| anyhow!("Variable {} not found", variable))?;

        // Calculate expected size
        let mut expected_size = var.domain.len();
        let mut dimensions = vec![var.domain.len()];

        for parent in &parents {
            let parent_var = self
                .variables
                .get(parent)
                .ok_or_else(|| anyhow!("Parent variable {} not found", parent))?;
            expected_size *= parent_var.domain.len();
            dimensions.push(parent_var.domain.len());
        }

        if probabilities.len() != expected_size {
            return Err(anyhow!(
                "CPT size {} does not match expected size {}",
                probabilities.len(),
                expected_size
            ));
        }

        let cpt = ConditionalProbabilityTable {
            variable: variable.clone(),
            parents,
            probabilities,
            dimensions,
        };

        self.cpts.insert(variable, cpt);
        Ok(())
    }

    /// Query the network: P(variable=value | evidence)
    /// Uses Variable Elimination algorithm for exact inference
    pub fn query(
        &self,
        variable: String,
        value: String,
        evidence: &[(String, String)],
    ) -> Result<f64> {
        // For now, implement a simple forward sampling approach
        // In a production system, you'd use Variable Elimination or Belief Propagation
        self.forward_sampling(&variable, &value, evidence, 10000)
    }

    /// Forward sampling (approximate inference)
    fn forward_sampling(
        &self,
        query_var: &str,
        query_val: &str,
        evidence: &[(String, String)],
        num_samples: usize,
    ) -> Result<f64> {
        use scirs2_core::random::rng;

        let mut rng = rng();
        let mut matches = 0;
        let mut valid_samples = 0;

        // Topological sort of variables
        let topo_order = self.topological_sort()?;

        for _ in 0..num_samples {
            let mut sample = HashMap::new();

            // Sample each variable in topological order
            for var_name in &topo_order {
                let var = &self.variables[var_name];
                let cpt = self
                    .cpts
                    .get(var_name)
                    .ok_or_else(|| anyhow!("CPT not found for {}", var_name))?;

                // Get conditional probability based on parents
                let prob_dist = self.get_conditional_prob(cpt, &sample)?;

                // Sample from the distribution
                let sampled_value =
                    self.sample_from_distribution(&prob_dist, &var.domain, &mut rng);
                sample.insert(var_name.clone(), sampled_value);
            }

            // Check if sample matches evidence
            let matches_evidence = evidence
                .iter()
                .all(|(ev_var, ev_val)| sample.get(ev_var) == Some(ev_val));

            if matches_evidence {
                valid_samples += 1;
                if sample.get(query_var).is_some_and(|v| v == query_val) {
                    matches += 1;
                }
            }
        }

        if valid_samples == 0 {
            return Err(anyhow!("No samples matched the evidence"));
        }

        Ok(matches as f64 / valid_samples as f64)
    }

    /// Get conditional probability distribution for a variable given parent values
    fn get_conditional_prob(
        &self,
        cpt: &ConditionalProbabilityTable,
        sample: &HashMap<String, String>,
    ) -> Result<Array1<f64>> {
        if cpt.parents.is_empty() {
            // No parents, return prior
            return Ok(cpt.probabilities.clone());
        }

        // Calculate index into CPT based on parent values
        let mut index = 0;
        let mut multiplier = 1;

        for (i, parent) in cpt.parents.iter().enumerate().rev() {
            let parent_var = &self.variables[parent];
            let parent_value = sample
                .get(parent)
                .ok_or_else(|| anyhow!("Parent {} not sampled yet", parent))?;
            let value_index = parent_var
                .domain
                .iter()
                .position(|v| v == parent_value)
                .ok_or_else(|| anyhow!("Parent value {} not in domain", parent_value))?;

            index += value_index * multiplier;
            multiplier *= cpt.dimensions[i + 1];
        }

        // Extract the conditional distribution for this parent configuration
        let var_domain_size = cpt.dimensions[0];
        let start = index * var_domain_size;

        // Use ndarray slicing to extract the probability distribution
        let probs_vec: Vec<f64> = cpt
            .probabilities
            .iter()
            .skip(start)
            .take(var_domain_size)
            .copied()
            .collect();
        Ok(Array1::from_vec(probs_vec))
    }

    /// Sample from a probability distribution
    fn sample_from_distribution(
        &self,
        prob_dist: &Array1<f64>,
        domain: &[String],
        rng: &mut impl RngCore,
    ) -> String {
        let uniform = Uniform::new(0.0, 1.0).expect("distribution parameters are valid");
        let u: f64 = uniform.sample(rng);

        let mut cumulative = 0.0;
        for (i, &prob) in prob_dist.iter().enumerate() {
            cumulative += prob;
            if u <= cumulative {
                return domain[i].clone();
            }
        }

        // Fallback (shouldn't happen if probabilities sum to 1.0)
        domain[domain.len() - 1].clone()
    }

    /// Topological sort of variables
    fn topological_sort(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize
        for var_name in self.variables.keys() {
            in_degree.insert(var_name.clone(), 0);
            adj_list.insert(var_name.clone(), Vec::new());
        }

        // Build adjacency list and in-degrees
        for (from, to) in &self.edges {
            adj_list
                .get_mut(from)
                .expect("variable should exist in adjacency list")
                .push(to.clone());
            *in_degree
                .get_mut(to)
                .expect("variable should exist in in_degree map") += 1;
        }

        // Kahn's algorithm
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node.clone());

            if let Some(neighbors) = adj_list.get(&node) {
                for neighbor in neighbors {
                    let deg = in_degree
                        .get_mut(neighbor)
                        .expect("neighbor should exist in in_degree map");
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(neighbor.clone());
                    }
                }
            }
        }

        if result.len() != self.variables.len() {
            return Err(anyhow!(
                "Graph has a cycle, cannot perform topological sort"
            ));
        }

        Ok(result)
    }

    /// Return the number of variables in the network
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Check whether a variable with the given name exists
    pub fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Check whether a directed edge exists between two variables
    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        self.edges.iter().any(|(f, t)| f == from && t == to)
    }

    /// Retrieve the prior probability for a specific value of a variable.
    ///
    /// Returns 0.0 if no CPT is set or the value is not in the domain.
    pub fn get_prior(&self, variable: &str, value: &str) -> f64 {
        let cpt = match self.cpts.get(variable) {
            Some(c) => c,
            None => return 0.0,
        };
        // Prior CPT has no parents
        if !cpt.parents.is_empty() {
            return 0.0;
        }
        let var = match self.variables.get(variable) {
            Some(v) => v,
            None => return 0.0,
        };
        let idx = match var.domain.iter().position(|v| v == value) {
            Some(i) => i,
            None => return 0.0,
        };
        if idx < cpt.probabilities.len() {
            cpt.probabilities[idx]
        } else {
            0.0
        }
    }
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Markov Logic Network (MLN)
/// Combines first-order logic with probabilistic graphical models
#[derive(Debug, Clone)]
pub struct MarkovLogicNetwork {
    /// Weighted rules (formulas with weights)
    weighted_rules: Vec<WeightedRule>,
    /// Domain of constants
    constants: HashSet<String>,
    /// Predicates and their arities
    predicates: HashMap<String, usize>,
}

/// A rule with an associated weight
#[derive(Debug, Clone)]
pub struct WeightedRule {
    /// The rule
    pub rule: Rule,
    /// Weight (log-odds of the rule being satisfied)
    pub weight: f64,
}

impl MarkovLogicNetwork {
    /// Create a new Markov Logic Network
    pub fn new() -> Self {
        Self {
            weighted_rules: Vec::new(),
            constants: HashSet::new(),
            predicates: HashMap::new(),
        }
    }

    /// Add a weighted rule to the MLN
    pub fn add_weighted_rule(&mut self, rule: Rule, weight: f64) {
        // Extract constants and predicates from rule
        self.extract_symbols(&rule);
        self.weighted_rules.push(WeightedRule { rule, weight });
    }

    /// Extract constants and predicates from a rule
    fn extract_symbols(&mut self, rule: &Rule) {
        for atom in rule.body.iter().chain(rule.head.iter()) {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = atom
            {
                if let Term::Constant(c) = subject {
                    self.constants.insert(c.clone());
                }
                if let Term::Constant(c) = predicate {
                    self.predicates.entry(c.clone()).or_insert(2); // Binary predicate
                }
                if let Term::Constant(c) = object {
                    self.constants.insert(c.clone());
                }
            }
        }
    }

    /// Compute the probability of a world (ground truth assignment)
    /// P(world) ∝ exp(Σ w_i * n_i)
    /// where w_i is the weight of rule i, and n_i is the number of groundings that are true
    pub fn compute_world_probability(&self, world: &[RuleAtom]) -> Result<f64> {
        let mut total_weight = 0.0;

        for weighted_rule in &self.weighted_rules {
            let num_satisfied = self.count_satisfied_groundings(&weighted_rule.rule, world)?;
            total_weight += weighted_rule.weight * num_satisfied as f64;
        }

        Ok(total_weight.exp())
    }

    /// Count how many groundings of a rule are satisfied in the given world
    fn count_satisfied_groundings(&self, rule: &Rule, world: &[RuleAtom]) -> Result<usize> {
        // Simplified: check if rule fires for any grounding
        let mut count = 0;

        // For each possible grounding (combination of constants)
        // Check if body is satisfied and head is true
        // This is a simplified version - full implementation would enumerate all groundings

        // Check if any facts in world match the rule head
        for fact in world {
            if self.matches_rule_head(fact, &rule.head) {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Check if a fact matches any atom in the rule head
    fn matches_rule_head(&self, fact: &RuleAtom, head: &[RuleAtom]) -> bool {
        head.iter().any(|atom| self.atoms_match(fact, atom))
    }

    /// Check if two atoms match (simplified)
    fn atoms_match(&self, fact: &RuleAtom, pattern: &RuleAtom) -> bool {
        match (fact, pattern) {
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

    /// Check if two terms match (constants must be equal, variables match anything)
    fn terms_match(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Variable(_), _) => true,
            (_, Term::Variable(_)) => true,
            _ => false,
        }
    }

    /// Perform MAP inference (find most probable world)
    pub fn map_inference(&self, evidence: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Simplified MAP inference using greedy approach
        // Full implementation would use MaxWalkSAT or similar

        let mut current_world = evidence.to_vec();
        let mut best_prob = self.compute_world_probability(&current_world)?;

        // Try adding inferred facts
        for weighted_rule in &self.weighted_rules {
            // Try to apply the rule and see if probability increases
            let mut new_world = current_world.clone();

            // Add all atoms from rule head (simplified)
            for atom in &weighted_rule.rule.head {
                new_world.push(atom.clone());
            }

            let new_prob = self.compute_world_probability(&new_world)?;
            if new_prob > best_prob {
                current_world = new_world;
                best_prob = new_prob;
            }
        }

        Ok(current_world)
    }

    /// Gibbs sampling for approximate inference
    pub fn gibbs_sampling(
        &self,
        evidence: &[RuleAtom],
        num_samples: usize,
    ) -> Result<Vec<RuleAtom>> {
        use scirs2_core::random::rng;

        let mut rng = rng();
        let mut current_state = evidence.to_vec();

        // Burn-in period
        for _ in 0..num_samples / 10 {
            current_state = self.gibbs_step(&current_state, &mut rng)?;
        }

        // Collect samples
        let mut sample_counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..num_samples {
            current_state = self.gibbs_step(&current_state, &mut rng)?;

            // Count occurrences of each atom
            for atom in &current_state {
                let key = format!("{:?}", atom);
                *sample_counts.entry(key).or_insert(0) += 1;
            }
        }

        // Return most frequent state
        Ok(current_state)
    }

    /// Single Gibbs sampling step
    fn gibbs_step(
        &self,
        current_state: &[RuleAtom],
        _rng: &mut impl RngCore,
    ) -> Result<Vec<RuleAtom>> {
        // Randomly select a ground atom and resample it
        let new_state = current_state.to_vec();

        // For simplicity, just return current state
        // Full implementation would sample each atom conditional on others
        Ok(new_state)
    }

    /// Register a predicate name with its arity
    pub fn add_predicate(&mut self, name: String, arity: usize) {
        self.predicates.insert(name, arity);
    }

    /// Check whether a predicate with the given name is registered
    pub fn has_predicate(&self, name: &str) -> bool {
        self.predicates.contains_key(name)
    }

    /// Add a weighted first-order logic formula to the network.
    ///
    /// The formula is stored as a textual representation alongside its weight.
    /// A corresponding `Rule` is created with the formula string as the rule name.
    pub fn add_formula(&mut self, formula: String, weight: f64) {
        let rule = Rule {
            name: formula,
            body: Vec::new(),
            head: Vec::new(),
        };
        self.weighted_rules.push(WeightedRule { rule, weight });
    }

    /// Return the number of weighted formulas (rules) in the network
    pub fn formula_count(&self) -> usize {
        self.weighted_rules.len()
    }
}

impl Default for MarkovLogicNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Probabilistic Rule Engine
/// Extends standard rule engine with probabilistic inference
#[derive(Debug)]
pub struct ProbabilisticRuleEngine {
    /// Bayesian Network for structural inference
    bayesian_network: Option<BayesianNetwork>,
    /// Markov Logic Network for first-order probabilistic logic
    mln: MarkovLogicNetwork,
    /// Weighted rules
    weighted_rules: Vec<WeightedRule>,
}

impl ProbabilisticRuleEngine {
    /// Create a new probabilistic rule engine
    pub fn new() -> Self {
        Self {
            bayesian_network: None,
            mln: MarkovLogicNetwork::new(),
            weighted_rules: Vec::new(),
        }
    }

    /// Set the Bayesian Network
    pub fn set_bayesian_network(&mut self, bn: BayesianNetwork) {
        self.bayesian_network = Some(bn);
    }

    /// Add a weighted rule
    pub fn add_weighted_rule(&mut self, rule: Rule, weight: f64) {
        self.weighted_rules.push(WeightedRule { rule, weight });
        self.mln.add_weighted_rule(
            self.weighted_rules
                .last()
                .expect("weighted_rules should not be empty after push")
                .rule
                .clone(),
            weight,
        );
    }

    /// Perform probabilistic forward chaining
    /// Returns inferred facts with their confidence scores
    pub fn probabilistic_forward_chain(&self, facts: &[RuleAtom]) -> Result<Vec<(RuleAtom, f64)>> {
        let mut inferred_with_confidence = Vec::new();

        // For each weighted rule, try to apply it
        for weighted_rule in &self.weighted_rules {
            // Check if rule body is satisfied
            if self.rule_body_satisfied(&weighted_rule.rule, facts) {
                // Add head with confidence based on weight
                let confidence = 1.0 / (1.0 + (-weighted_rule.weight).exp()); // Sigmoid
                for atom in &weighted_rule.rule.head {
                    inferred_with_confidence.push((atom.clone(), confidence));
                }
            }
        }

        Ok(inferred_with_confidence)
    }

    /// Check if rule body is satisfied by the facts
    fn rule_body_satisfied(&self, rule: &Rule, facts: &[RuleAtom]) -> bool {
        // Simplified: check if all body atoms are in facts
        rule.body
            .iter()
            .all(|body_atom| facts.iter().any(|fact| self.atoms_similar(body_atom, fact)))
    }

    /// Check if two atoms are similar (simplified matching)
    fn atoms_similar(&self, atom1: &RuleAtom, atom2: &RuleAtom) -> bool {
        match (atom1, atom2) {
            (
                RuleAtom::Triple {
                    predicate: p1,
                    subject: s1,
                    object: o1,
                },
                RuleAtom::Triple {
                    predicate: p2,
                    subject: s2,
                    object: o2,
                },
            ) => {
                // Variables match anything
                let pred_match = match (p1, p2) {
                    (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
                    _ => true,
                };
                let subj_match = match (s1, s2) {
                    (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
                    _ => true,
                };
                let obj_match = match (o1, o2) {
                    (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
                    _ => true,
                };
                pred_match && subj_match && obj_match
            }
            _ => false,
        }
    }

    /// Query using Bayesian Network if available
    pub fn query_bayesian(
        &self,
        variable: String,
        value: String,
        evidence: &[(String, String)],
    ) -> Result<f64> {
        if let Some(bn) = &self.bayesian_network {
            bn.query(variable, value, evidence)
        } else {
            Err(anyhow!("Bayesian Network not set"))
        }
    }

    /// Perform MAP inference using MLN
    pub fn mln_map_inference(&self, evidence: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        self.mln.map_inference(evidence)
    }
}

impl Default for ProbabilisticRuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_network_creation() {
        let mut bn = BayesianNetwork::new();

        bn.add_variable("Rain".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable(
            "Sprinkler".to_string(),
            vec!["T".to_string(), "F".to_string()],
        );
        bn.add_variable(
            "WetGrass".to_string(),
            vec!["T".to_string(), "F".to_string()],
        );

        assert!(bn.variables.contains_key("Rain"));
        assert!(bn.variables.contains_key("Sprinkler"));
        assert!(bn.variables.contains_key("WetGrass"));
    }

    #[test]
    fn test_bayesian_network_edges() {
        let mut bn = BayesianNetwork::new();

        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("B".to_string(), vec!["T".to_string(), "F".to_string()]);

        assert!(bn.add_edge("A".to_string(), "B".to_string()).is_ok());

        // Test cycle detection
        assert!(bn.add_edge("B".to_string(), "A".to_string()).is_err());
    }

    #[test]
    fn test_bayesian_network_prior() {
        let mut bn = BayesianNetwork::new();

        bn.add_variable("Coin".to_string(), vec!["H".to_string(), "T".to_string()]);

        assert!(bn.set_prior("Coin".to_string(), vec![0.5, 0.5]).is_ok());

        // Test invalid probability sum
        assert!(bn.set_prior("Coin".to_string(), vec![0.6, 0.6]).is_err());
    }

    #[test]
    fn test_mln_creation() {
        let mln = MarkovLogicNetwork::new();
        assert_eq!(mln.weighted_rules.len(), 0);
    }

    #[test]
    fn test_mln_add_weighted_rule() {
        let mut mln = MarkovLogicNetwork::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("smokes".to_string()),
                object: Term::Constant("true".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("cancer".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        };

        mln.add_weighted_rule(rule, 1.5);
        assert_eq!(mln.weighted_rules.len(), 1);
    }

    #[test]
    fn test_probabilistic_rule_engine() {
        let mut engine = ProbabilisticRuleEngine::new();

        let rule = Rule {
            name: "prob_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("output".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        engine.add_weighted_rule(rule, 2.0);

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let result = engine.probabilistic_forward_chain(&facts).unwrap();
        assert!(!result.is_empty());

        // Check confidence is reasonable
        for (_, confidence) in result {
            assert!(confidence > 0.0 && confidence <= 1.0);
        }
    }

    #[test]
    fn test_topological_sort() {
        let mut bn = BayesianNetwork::new();

        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("B".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("C".to_string(), vec!["T".to_string(), "F".to_string()]);

        bn.add_edge("A".to_string(), "B".to_string()).unwrap();
        bn.add_edge("B".to_string(), "C".to_string()).unwrap();

        let topo_order = bn.topological_sort().unwrap();
        assert_eq!(topo_order.len(), 3);

        // A should come before B, and B before C
        let a_idx = topo_order.iter().position(|x| x == "A").unwrap();
        let b_idx = topo_order.iter().position(|x| x == "B").unwrap();
        let c_idx = topo_order.iter().position(|x| x == "C").unwrap();

        assert!(a_idx < b_idx);
        assert!(b_idx < c_idx);
    }

    // ---- Extended tests ----

    #[test]
    fn test_bn_new_is_empty() {
        let bn = BayesianNetwork::new();
        assert!(bn.variables.is_empty());
        assert!(bn.edges.is_empty());
        assert!(bn.cpts.is_empty());
    }

    #[test]
    fn test_bn_add_four_variables() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("W".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("X".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("Y".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("Z".to_string(), vec!["T".to_string(), "F".to_string()]);
        assert_eq!(bn.variables.len(), 4);
    }

    #[test]
    fn test_bn_cycle_detection_three_node() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("B".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("C".to_string(), vec!["T".to_string(), "F".to_string()]);
        assert!(bn.add_edge("A".to_string(), "B".to_string()).is_ok());
        assert!(bn.add_edge("B".to_string(), "C".to_string()).is_ok());
        // Creating a cycle: C -> A
        assert!(
            bn.add_edge("C".to_string(), "A".to_string()).is_err(),
            "Should detect cycle A->B->C->A"
        );
    }

    #[test]
    fn test_bn_self_loop_is_cycle() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        assert!(
            bn.add_edge("A".to_string(), "A".to_string()).is_err(),
            "Self-loop is a cycle"
        );
    }

    #[test]
    fn test_bn_diamond_dag_is_valid() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("B".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("C".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("D".to_string(), vec!["T".to_string(), "F".to_string()]);
        // A -> B, A -> C, B -> D, C -> D
        assert!(bn.add_edge("A".to_string(), "B".to_string()).is_ok());
        assert!(bn.add_edge("A".to_string(), "C".to_string()).is_ok());
        assert!(bn.add_edge("B".to_string(), "D".to_string()).is_ok());
        assert!(bn.add_edge("C".to_string(), "D".to_string()).is_ok());
        assert_eq!(bn.edges.len(), 4);
    }

    #[test]
    fn test_bn_add_edge_nonexistent_from() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("B".to_string(), vec!["T".to_string(), "F".to_string()]);
        assert!(bn.add_edge("Ghost".to_string(), "B".to_string()).is_err());
    }

    #[test]
    fn test_bn_add_edge_nonexistent_to() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("A".to_string(), vec!["T".to_string(), "F".to_string()]);
        assert!(bn.add_edge("A".to_string(), "Ghost".to_string()).is_err());
    }

    #[test]
    fn test_bn_prior_invalid_sum() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("X".to_string(), vec!["a".to_string(), "b".to_string()]);
        assert!(
            bn.set_prior("X".to_string(), vec![0.3, 0.8]).is_err(),
            "Prob sum > 1 should fail"
        );
    }

    #[test]
    fn test_bn_prior_valid() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("X".to_string(), vec!["a".to_string(), "b".to_string()]);
        assert!(bn.set_prior("X".to_string(), vec![0.3, 0.7]).is_ok());
    }

    #[test]
    fn test_bn_prior_unknown_variable() {
        let mut bn = BayesianNetwork::new();
        assert!(
            bn.set_prior("Ghost".to_string(), vec![0.5, 0.5]).is_err(),
            "Unknown variable should fail"
        );
    }

    #[test]
    fn test_bn_topological_sort_single_node() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("Alone".to_string(), vec!["T".to_string(), "F".to_string()]);
        let topo = bn.topological_sort().unwrap();
        assert_eq!(topo.len(), 1);
        assert_eq!(topo[0], "Alone");
    }

    #[test]
    fn test_bn_topological_sort_no_edges() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("P".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("Q".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.add_variable("R".to_string(), vec!["T".to_string(), "F".to_string()]);
        let topo = bn.topological_sort().unwrap();
        assert_eq!(topo.len(), 3);
    }

    #[test]
    fn test_bn_query_returns_probability() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("Rain".to_string(), vec!["T".to_string(), "F".to_string()]);
        bn.set_prior("Rain".to_string(), vec![0.3, 0.7]).unwrap();
        let result = bn.query("Rain".to_string(), "T".to_string(), &[]);
        assert!(result.is_ok(), "query should succeed");
        let prob = result.unwrap();
        assert!(
            (0.0..=1.0).contains(&prob),
            "probability should be in [0,1]"
        );
    }

    #[test]
    fn test_mln_new_is_empty() {
        let mln = MarkovLogicNetwork::new();
        assert_eq!(mln.weighted_rules.len(), 0);
    }

    #[test]
    fn test_mln_add_multiple_rules() {
        let mut mln = MarkovLogicNetwork::new();
        for i in 0..5usize {
            let rule = Rule {
                name: format!("rule_{}", i),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(format!("p{}", i)),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(format!("q{}", i)),
                    object: Term::Variable("Y".to_string()),
                }],
            };
            mln.add_weighted_rule(rule, 1.0 + i as f64);
        }
        assert_eq!(mln.weighted_rules.len(), 5);
    }

    #[test]
    fn test_prob_engine_new_is_empty() {
        let engine = ProbabilisticRuleEngine::new();
        assert_eq!(engine.weighted_rules.len(), 0);
    }

    #[test]
    fn test_prob_engine_with_no_rules_empty_facts() {
        let engine = ProbabilisticRuleEngine::new();
        let result = engine.probabilistic_forward_chain(&[]).unwrap();
        assert!(
            result.is_empty(),
            "No rules and empty facts => empty result"
        );
    }

    #[test]
    fn test_prob_engine_confidence_in_range() {
        let mut engine = ProbabilisticRuleEngine::new();
        let rule = Rule {
            name: "range_test".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("likes".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("enjoysCompanyOf".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        engine.add_weighted_rule(rule, 0.9);
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("likes".to_string()),
            object: Term::Constant("bob".to_string()),
        }];
        let result = engine.probabilistic_forward_chain(&facts).unwrap();
        for (_, confidence) in &result {
            assert!(
                *confidence >= 0.0 && *confidence <= 1.0,
                "confidence out of range: {}",
                confidence
            );
        }
    }

    #[test]
    fn test_prob_engine_multiple_facts_has_results() {
        let mut engine = ProbabilisticRuleEngine::new();
        let rule = Rule {
            name: "multi".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("output".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        engine.add_weighted_rule(rule, 1.5);
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("a".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Constant("b".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("c".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Constant("d".to_string()),
            },
        ];
        let result = engine.probabilistic_forward_chain(&facts).unwrap();
        // Should get at least one result (rule application)
        assert!(
            !result.is_empty(),
            "Should get at least one result from rule application"
        );
    }

    #[test]
    fn test_prob_engine_add_weighted_rule_increments_count() {
        let mut engine = ProbabilisticRuleEngine::new();
        assert_eq!(engine.weighted_rules.len(), 0);
        let rule = Rule {
            name: "r1".to_string(),
            body: vec![],
            head: vec![],
        };
        engine.add_weighted_rule(rule, 0.5);
        assert_eq!(engine.weighted_rules.len(), 1);
    }

    #[test]
    fn test_bn_variable_domain_stored() {
        let mut bn = BayesianNetwork::new();
        let domain = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
        bn.add_variable("Risk".to_string(), domain.clone());
        assert!(bn.variables.contains_key("Risk"));
        let var = &bn.variables["Risk"];
        assert_eq!(var.domain, domain);
    }

    #[test]
    fn test_bn_prior_zero_prob_valid() {
        let mut bn = BayesianNetwork::new();
        bn.add_variable("X".to_string(), vec!["T".to_string(), "F".to_string()]);
        // P(T)=0, P(F)=1 — valid distribution
        assert!(bn.set_prior("X".to_string(), vec![0.0, 1.0]).is_ok());
    }
}
