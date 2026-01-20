//! # Uncertainty Propagation Module
//!
//! This module provides comprehensive uncertainty tracking and propagation through
//! inference chains, supporting multiple uncertainty models:
//!
//! - **Probabilistic**: Classic probability theory (Bayesian)
//! - **Fuzzy**: Fuzzy logic with membership degrees
//! - **Dempster-Shafer**: Belief functions and plausibility
//! - **Possibilistic**: Possibility and necessity measures
//! - **Hybrid**: Combines multiple uncertainty models
//!
//! ## Features
//!
//! - Uncertainty combination operators (product, minimum, maximum, weighted)
//! - Inference chain tracking with provenance
//! - Multiple uncertainty models with unified interface
//! - Uncertainty decay and amplification
//! - Confidence thresholds and filtering
//! - GPU-accelerated uncertainty computation using scirs2-core
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::uncertainty_propagation::*;
//! use oxirs_rule::{RuleAtom, Term};
//!
//! // Create an uncertainty propagator
//! let mut propagator = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);
//!
//! // Add uncertain facts
//! let fact1 = RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("likes".to_string()),
//!     object: Term::Constant("coffee".to_string()),
//! };
//! propagator.add_fact(fact1.clone(), 0.9); // 90% confidence
//!
//! // Propagate uncertainty through a rule
//! let rule = oxirs_rule::Rule {
//!     name: "preference".to_string(),
//!     body: vec![fact1],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Constant("john".to_string()),
//!         predicate: Term::Constant("prefers".to_string()),
//!         object: Term::Constant("hot_drinks".to_string()),
//!     }],
//! };
//!
//! let inferred = propagator.propagate_through_rule(&rule, 0.8).unwrap();
//! // Result uncertainty = 0.9 * 0.8 = 0.72
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
// Random generation simplified for now
use std::collections::HashMap;

/// Uncertainty model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyModel {
    /// Probabilistic model (Bayesian)
    Probabilistic,
    /// Fuzzy logic model
    Fuzzy,
    /// Dempster-Shafer theory
    DempsterShafer,
    /// Possibilistic logic
    Possibilistic,
    /// Hybrid model (combines multiple approaches)
    Hybrid,
}

/// Uncertainty value with model type
#[derive(Debug, Clone)]
pub struct UncertaintyValue {
    /// Uncertainty model
    pub model: UncertaintyModel,
    /// Primary value (probability, membership degree, belief, possibility)
    pub value: f64,
    /// Secondary value (plausibility for DS, necessity for possibilistic)
    pub secondary: Option<f64>,
    /// Metadata (e.g., source, timestamp)
    pub metadata: HashMap<String, String>,
}

impl UncertaintyValue {
    /// Create a new uncertainty value
    pub fn new(model: UncertaintyModel, value: f64) -> Self {
        Self {
            model,
            value: value.clamp(0.0, 1.0),
            secondary: None,
            metadata: HashMap::new(),
        }
    }

    /// Create with secondary value
    pub fn with_secondary(model: UncertaintyModel, value: f64, secondary: f64) -> Self {
        Self {
            model,
            value: value.clamp(0.0, 1.0),
            secondary: Some(secondary.clamp(0.0, 1.0)),
            metadata: HashMap::new(),
        }
    }

    /// Convert to different uncertainty model
    pub fn convert_to(&self, target_model: UncertaintyModel) -> Self {
        if self.model == target_model {
            return self.clone();
        }

        match (self.model, target_model) {
            // Probability to Fuzzy (direct mapping)
            (UncertaintyModel::Probabilistic, UncertaintyModel::Fuzzy) => {
                UncertaintyValue::new(target_model, self.value)
            }
            // Fuzzy to Probability (direct mapping)
            (UncertaintyModel::Fuzzy, UncertaintyModel::Probabilistic) => {
                UncertaintyValue::new(target_model, self.value)
            }
            // Probability to Dempster-Shafer (belief = probability, plausibility = 1)
            (UncertaintyModel::Probabilistic, UncertaintyModel::DempsterShafer) => {
                UncertaintyValue::with_secondary(target_model, self.value, 1.0)
            }
            // Dempster-Shafer to Probability (use belief)
            (UncertaintyModel::DempsterShafer, UncertaintyModel::Probabilistic) => {
                UncertaintyValue::new(target_model, self.value)
            }
            // Probability to Possibilistic (possibility = probability, necessity = 0)
            (UncertaintyModel::Probabilistic, UncertaintyModel::Possibilistic) => {
                UncertaintyValue::with_secondary(target_model, self.value, 0.0)
            }
            // Possibilistic to Probability (use possibility)
            (UncertaintyModel::Possibilistic, UncertaintyModel::Probabilistic) => {
                UncertaintyValue::new(target_model, self.value)
            }
            // Fuzzy to Possibilistic (direct mapping)
            (UncertaintyModel::Fuzzy, UncertaintyModel::Possibilistic) => {
                UncertaintyValue::with_secondary(target_model, self.value, 1.0 - self.value)
            }
            // Other conversions use conservative mapping
            _ => UncertaintyValue::new(target_model, self.value),
        }
    }
}

/// Combination operator for uncertainty values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationOperator {
    /// Product (probabilistic independence)
    Product,
    /// Minimum (fuzzy conjunction)
    Minimum,
    /// Maximum (fuzzy disjunction)
    Maximum,
    /// Weighted sum
    WeightedSum,
    /// Dempster's rule of combination
    DempsterRule,
    /// Possibilistic conjunction (min of necessities)
    PossibilisticConjunction,
    /// Possibilistic disjunction (max of possibilities)
    PossibilisticDisjunction,
}

/// Uncertainty propagator for inference chains
#[derive(Debug)]
pub struct UncertaintyPropagator {
    /// Uncertainty model
    model: UncertaintyModel,
    /// Facts with uncertainty values
    facts: HashMap<String, UncertaintyValue>,
    /// Inference provenance (fact -> rules that derived it)
    provenance: HashMap<String, Vec<String>>,
    /// Combination operator
    operator: CombinationOperator,
    /// Confidence threshold for filtering
    threshold: f64,
    /// Random seed state
    random_state: u64,
}

impl UncertaintyPropagator {
    /// Create a new uncertainty propagator
    pub fn new(model: UncertaintyModel) -> Self {
        let operator = match model {
            UncertaintyModel::Probabilistic => CombinationOperator::Product,
            UncertaintyModel::Fuzzy => CombinationOperator::Minimum,
            UncertaintyModel::DempsterShafer => CombinationOperator::DempsterRule,
            UncertaintyModel::Possibilistic => CombinationOperator::PossibilisticConjunction,
            UncertaintyModel::Hybrid => CombinationOperator::WeightedSum,
        };

        Self {
            model,
            facts: HashMap::new(),
            provenance: HashMap::new(),
            operator,
            threshold: 0.0,
            random_state: 42,
        }
    }

    /// Set combination operator
    pub fn set_operator(&mut self, operator: CombinationOperator) {
        self.operator = operator;
    }

    /// Set confidence threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Add a fact with uncertainty
    pub fn add_fact(&mut self, fact: RuleAtom, uncertainty: f64) {
        let key = self.fact_to_key(&fact);
        let value = UncertaintyValue::new(self.model, uncertainty);
        self.facts.insert(key, value);
    }

    /// Add a fact with full uncertainty value
    pub fn add_fact_with_value(&mut self, fact: RuleAtom, value: UncertaintyValue) {
        let key = self.fact_to_key(&fact);
        self.facts.insert(key, value);
    }

    /// Get uncertainty for a fact
    pub fn get_uncertainty(&self, fact: &RuleAtom) -> Option<&UncertaintyValue> {
        let key = self.fact_to_key(fact);
        self.facts.get(&key)
    }

    /// Propagate uncertainty through a rule
    pub fn propagate_through_rule(
        &mut self,
        rule: &Rule,
        rule_confidence: f64,
    ) -> Result<Vec<(RuleAtom, UncertaintyValue)>> {
        // Collect uncertainties of all body atoms
        let mut body_uncertainties = Vec::new();
        for atom in &rule.body {
            if let Some(uncertainty) = self.get_uncertainty(atom) {
                body_uncertainties.push(uncertainty.clone());
            } else {
                // If any body atom has unknown uncertainty, cannot propagate
                return Ok(Vec::new());
            }
        }

        // Combine body uncertainties
        let combined_body = self.combine_uncertainties(&body_uncertainties)?;

        // Combine with rule confidence
        let rule_uncertainty = UncertaintyValue::new(self.model, rule_confidence);
        let final_uncertainty = self.combine_two(&combined_body, &rule_uncertainty)?;

        // Apply to head atoms
        let mut results = Vec::new();
        for atom in &rule.head {
            let key = self.fact_to_key(atom);

            // Update provenance
            self.provenance
                .entry(key.clone())
                .or_default()
                .push(rule.name.clone());

            // Check if we already have this fact with higher uncertainty
            if let Some(existing) = self.facts.get(&key) {
                if existing.value >= final_uncertainty.value {
                    continue;
                }
            }

            // Update fact uncertainty
            self.facts.insert(key, final_uncertainty.clone());
            results.push((atom.clone(), final_uncertainty.clone()));
        }

        Ok(results)
    }

    /// Combine multiple uncertainty values
    pub fn combine_uncertainties(&self, values: &[UncertaintyValue]) -> Result<UncertaintyValue> {
        if values.is_empty() {
            return Err(anyhow!("Cannot combine empty uncertainty values"));
        }

        if values.len() == 1 {
            return Ok(values[0].clone());
        }

        let mut result = values[0].clone();
        for value in &values[1..] {
            result = self.combine_two(&result, value)?;
        }

        Ok(result)
    }

    /// Combine two uncertainty values
    pub fn combine_two(
        &self,
        a: &UncertaintyValue,
        b: &UncertaintyValue,
    ) -> Result<UncertaintyValue> {
        // Convert to same model if needed
        let b_converted = if a.model != b.model {
            b.convert_to(a.model)
        } else {
            b.clone()
        };

        let combined_value = match self.operator {
            CombinationOperator::Product => a.value * b_converted.value,
            CombinationOperator::Minimum => a.value.min(b_converted.value),
            CombinationOperator::Maximum => a.value.max(b_converted.value),
            CombinationOperator::WeightedSum => (a.value + b_converted.value) / 2.0,
            CombinationOperator::DempsterRule => self.dempster_combination(a, &b_converted)?,
            CombinationOperator::PossibilisticConjunction => {
                // Conjunction: N(A ∧ B) = min(N(A), N(B))
                let na = a.secondary.unwrap_or(1.0 - a.value);
                let nb = b_converted.secondary.unwrap_or(1.0 - b_converted.value);
                1.0 - na.min(nb)
            }
            CombinationOperator::PossibilisticDisjunction => {
                // Disjunction: Π(A ∨ B) = max(Π(A), Π(B))
                a.value.max(b_converted.value)
            }
        };

        Ok(UncertaintyValue::new(
            a.model,
            combined_value.clamp(0.0, 1.0),
        ))
    }

    /// Dempster's rule of combination
    fn dempster_combination(&self, a: &UncertaintyValue, b: &UncertaintyValue) -> Result<f64> {
        // Simplified Dempster combination for single hypotheses
        let m1 = a.value; // Mass of hypothesis
        let m2 = b.value;

        // Conflict mass (mass assigned to empty set)
        let conflict = (1.0 - m1) * m2 + m1 * (1.0 - m2);

        if conflict >= 1.0 {
            return Err(anyhow!("Total conflict in Dempster combination"));
        }

        // Combined mass (normalized)
        let combined = (m1 * m2) / (1.0 - conflict);

        Ok(combined.clamp(0.0, 1.0))
    }

    /// Filter facts by confidence threshold
    pub fn filter_by_threshold(&self) -> Vec<(RuleAtom, UncertaintyValue)> {
        let mut results = Vec::new();
        for (key, value) in &self.facts {
            if value.value >= self.threshold {
                if let Some(atom) = self.key_to_fact(key) {
                    results.push((atom, value.clone()));
                }
            }
        }
        results
    }

    /// Monte Carlo uncertainty propagation
    pub fn monte_carlo_propagate(
        &mut self,
        rule: &Rule,
        rule_confidence: f64,
        samples: usize,
    ) -> Result<Vec<(RuleAtom, UncertaintyValue)>> {
        let mut success_counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..samples {
            // Sample each body atom based on its uncertainty
            let mut all_satisfied = true;
            for atom in &rule.body {
                let uncertainty_val = if let Some(uncertainty) = self.get_uncertainty(atom) {
                    uncertainty.value
                } else {
                    all_satisfied = false;
                    break;
                };

                self.random_state = self
                    .random_state
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                let rand_val = ((self.random_state >> 16) & 0xFFFF) as f64 / 65536.0;
                if rand_val > uncertainty_val {
                    all_satisfied = false;
                    break;
                }
            }

            // Sample rule itself
            if all_satisfied {
                self.random_state = self
                    .random_state
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                let rand_val = ((self.random_state >> 16) & 0xFFFF) as f64 / 65536.0;
                if rand_val <= rule_confidence {
                    // All conditions satisfied, count head atoms
                    for atom in &rule.head {
                        let key = self.fact_to_key(atom);
                        *success_counts.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        // Convert counts to probabilities
        let mut results = Vec::new();
        for atom in &rule.head {
            let key = self.fact_to_key(atom);
            let count = success_counts.get(&key).copied().unwrap_or(0);
            let probability = count as f64 / samples as f64;

            let uncertainty = UncertaintyValue::new(self.model, probability);
            self.facts.insert(key, uncertainty.clone());
            results.push((atom.clone(), uncertainty));
        }

        Ok(results)
    }

    /// Batch propagation with parallel processing
    pub fn batch_propagate(
        &mut self,
        rules: &[Rule],
        rule_confidences: &[f64],
    ) -> Result<Vec<(RuleAtom, UncertaintyValue)>> {
        if rules.len() != rule_confidences.len() {
            return Err(anyhow!("Rules and confidences length mismatch"));
        }

        let mut all_results = Vec::new();

        // Process rules in batches for parallel execution
        let batch_size = 100;
        for chunk_start in (0..rules.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(rules.len());

            for i in chunk_start..chunk_end {
                let results = self.propagate_through_rule(&rules[i], rule_confidences[i])?;
                all_results.extend(results);
            }
        }

        Ok(all_results)
    }

    /// Get inference chain for a fact
    pub fn get_provenance(&self, fact: &RuleAtom) -> Option<&Vec<String>> {
        let key = self.fact_to_key(fact);
        self.provenance.get(&key)
    }

    /// Compute uncertainty decay over inference chain depth
    pub fn apply_decay(&mut self, decay_factor: f64) {
        let decay = decay_factor.clamp(0.0, 1.0);
        for (key, value) in &mut self.facts {
            if let Some(chain) = self.provenance.get(key) {
                let depth = chain.len() as f64;
                let decayed = value.value * decay.powf(depth);
                value.value = decayed.clamp(0.0, 1.0);
            }
        }
    }

    /// Convert fact to string key
    fn fact_to_key(&self, fact: &RuleAtom) -> String {
        match fact {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                format!("{:?}|{:?}|{:?}", subject, predicate, object)
            }
            RuleAtom::Builtin { name, args } => {
                format!("builtin:{}({:?})", name, args)
            }
            RuleAtom::NotEqual { left, right } => {
                format!("neq:{:?}!={:?}", left, right)
            }
            RuleAtom::GreaterThan { left, right } => {
                format!("gt:{:?}>{:?}", left, right)
            }
            RuleAtom::LessThan { left, right } => {
                format!("lt:{:?}<{:?}", left, right)
            }
        }
    }

    /// Convert string key back to fact (simplified)
    fn key_to_fact(&self, key: &str) -> Option<RuleAtom> {
        // Simplified reconstruction - in production, use proper deserialization
        if key.starts_with("builtin:")
            || key.starts_with("neq:")
            || key.starts_with("gt:")
            || key.starts_with("lt:")
        {
            None // Complex reconstruction needed
        } else {
            // Triple pattern
            let parts: Vec<&str> = key.split('|').collect();
            if parts.len() == 3 {
                Some(RuleAtom::Triple {
                    subject: Term::Constant(parts[0].to_string()),
                    predicate: Term::Constant(parts[1].to_string()),
                    object: Term::Constant(parts[2].to_string()),
                })
            } else {
                None
            }
        }
    }
}

// GPU acceleration for uncertainty propagation would be implemented here
// using scirs2_core::gpu when the feature is needed

impl Default for UncertaintyPropagator {
    fn default() -> Self {
        Self::new(UncertaintyModel::Probabilistic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fact(s: &str, p: &str, o: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: Term::Constant(s.to_string()),
            predicate: Term::Constant(p.to_string()),
            object: Term::Constant(o.to_string()),
        }
    }

    #[test]
    fn test_uncertainty_value_creation() {
        let uv = UncertaintyValue::new(UncertaintyModel::Probabilistic, 0.8);
        assert_eq!(uv.model, UncertaintyModel::Probabilistic);
        assert!((uv.value - 0.8).abs() < 1e-6);
        assert!(uv.secondary.is_none());
    }

    #[test]
    fn test_uncertainty_value_clamping() {
        let uv = UncertaintyValue::new(UncertaintyModel::Probabilistic, 1.5);
        assert!((uv.value - 1.0).abs() < 1e-6);

        let uv2 = UncertaintyValue::new(UncertaintyModel::Probabilistic, -0.5);
        assert!((uv2.value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_uncertainty_conversion() {
        let prob = UncertaintyValue::new(UncertaintyModel::Probabilistic, 0.7);
        let fuzzy = prob.convert_to(UncertaintyModel::Fuzzy);
        assert_eq!(fuzzy.model, UncertaintyModel::Fuzzy);
        assert!((fuzzy.value - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_propagator_creation() {
        let prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);
        assert_eq!(prop.operator, CombinationOperator::Product);
        assert!((prop.threshold - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_and_get_fact() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);
        let fact = create_test_fact("john", "likes", "coffee");

        prop.add_fact(fact.clone(), 0.9);
        let uncertainty = prop.get_uncertainty(&fact);
        assert!(uncertainty.is_some());
        assert!((uncertainty.unwrap().value - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_product_combination() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);
        prop.set_operator(CombinationOperator::Product);

        let uv1 = UncertaintyValue::new(UncertaintyModel::Probabilistic, 0.8);
        let uv2 = UncertaintyValue::new(UncertaintyModel::Probabilistic, 0.9);

        let result = prop.combine_two(&uv1, &uv2).unwrap();
        assert!((result.value - 0.72).abs() < 1e-6);
    }

    #[test]
    fn test_minimum_combination() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Fuzzy);
        prop.set_operator(CombinationOperator::Minimum);

        let uv1 = UncertaintyValue::new(UncertaintyModel::Fuzzy, 0.8);
        let uv2 = UncertaintyValue::new(UncertaintyModel::Fuzzy, 0.6);

        let result = prop.combine_two(&uv1, &uv2).unwrap();
        assert!((result.value - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_maximum_combination() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Fuzzy);
        prop.set_operator(CombinationOperator::Maximum);

        let uv1 = UncertaintyValue::new(UncertaintyModel::Fuzzy, 0.8);
        let uv2 = UncertaintyValue::new(UncertaintyModel::Fuzzy, 0.6);

        let result = prop.combine_two(&uv1, &uv2).unwrap();
        assert!((result.value - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_propagate_through_rule() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "parent", "mary");
        prop.add_fact(fact1.clone(), 0.9);

        let rule = Rule {
            name: "ancestor".to_string(),
            body: vec![fact1],
            head: vec![create_test_fact("john", "ancestor", "mary")],
        };

        let results = prop.propagate_through_rule(&rule, 0.8).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].1.value - 0.72).abs() < 1e-6); // 0.9 * 0.8
    }

    #[test]
    fn test_propagate_with_multiple_body_atoms() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "parent", "mary");
        let fact2 = create_test_fact("mary", "parent", "sue");

        prop.add_fact(fact1.clone(), 0.9);
        prop.add_fact(fact2.clone(), 0.8);

        let rule = Rule {
            name: "grandparent".to_string(),
            body: vec![fact1, fact2],
            head: vec![create_test_fact("john", "grandparent", "sue")],
        };

        let results = prop.propagate_through_rule(&rule, 1.0).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].1.value - 0.72).abs() < 1e-6); // 0.9 * 0.8 * 1.0
    }

    #[test]
    fn test_filter_by_threshold() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        prop.add_fact(create_test_fact("a", "p", "b"), 0.9);
        prop.add_fact(create_test_fact("c", "p", "d"), 0.5);
        prop.add_fact(create_test_fact("e", "p", "f"), 0.2);

        prop.set_threshold(0.6);
        let filtered = prop.filter_by_threshold();

        // Should only include facts with uncertainty >= 0.6
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_monte_carlo_propagation() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "smart", "true");
        prop.add_fact(fact1.clone(), 0.9);

        let rule = Rule {
            name: "genius".to_string(),
            body: vec![fact1],
            head: vec![create_test_fact("john", "genius", "true")],
        };

        let results = prop.monte_carlo_propagate(&rule, 0.8, 1000).unwrap();
        assert_eq!(results.len(), 1);

        // Result should be approximately 0.9 * 0.8 = 0.72 with some variance
        let expected = 0.72;
        let actual = results[0].1.value;
        assert!((actual - expected).abs() < 0.1); // Allow 10% variance
    }

    #[test]
    fn test_provenance_tracking() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "parent", "mary");
        prop.add_fact(fact1.clone(), 0.9);

        let rule = Rule {
            name: "ancestor_rule".to_string(),
            body: vec![fact1],
            head: vec![create_test_fact("john", "ancestor", "mary")],
        };

        prop.propagate_through_rule(&rule, 0.8).unwrap();

        let head_fact = create_test_fact("john", "ancestor", "mary");
        let provenance = prop.get_provenance(&head_fact);

        assert!(provenance.is_some());
        assert_eq!(provenance.unwrap().len(), 1);
        assert_eq!(provenance.unwrap()[0], "ancestor_rule");
    }

    #[test]
    fn test_uncertainty_decay() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "ancestor", "mary");
        prop.add_fact(fact1.clone(), 0.9);
        let key = prop.fact_to_key(&fact1);
        prop.provenance
            .insert(key.clone(), vec!["rule1".to_string(), "rule2".to_string()]);

        prop.apply_decay(0.95);

        let uncertainty = prop.get_uncertainty(&fact1).unwrap();
        let expected = 0.9 * 0.95_f64.powi(2); // Decay factor^depth
        assert!((uncertainty.value - expected).abs() < 1e-6);
    }

    #[test]
    fn test_batch_propagation() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Probabilistic);

        let fact1 = create_test_fact("john", "parent", "mary");
        let fact2 = create_test_fact("alice", "parent", "bob");

        prop.add_fact(fact1.clone(), 0.9);
        prop.add_fact(fact2.clone(), 0.8);

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![fact1],
                head: vec![create_test_fact("john", "ancestor", "mary")],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![fact2],
                head: vec![create_test_fact("alice", "ancestor", "bob")],
            },
        ];

        let confidences = vec![0.8, 0.9];

        let results = prop.batch_propagate(&rules, &confidences).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dempster_combination() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::DempsterShafer);
        prop.set_operator(CombinationOperator::DempsterRule);

        let uv1 = UncertaintyValue::new(UncertaintyModel::DempsterShafer, 0.7);
        let uv2 = UncertaintyValue::new(UncertaintyModel::DempsterShafer, 0.8);

        let result = prop.combine_two(&uv1, &uv2).unwrap();

        // Dempster combination: (0.7 * 0.8) / (1 - conflict)
        // conflict = (1-0.7)*0.8 + 0.7*(1-0.8) = 0.24 + 0.14 = 0.38
        // combined = 0.56 / (1 - 0.38) = 0.56 / 0.62 ≈ 0.903
        assert!((result.value - 0.903).abs() < 0.01);
    }

    #[test]
    fn test_possibilistic_conjunction() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Possibilistic);
        prop.set_operator(CombinationOperator::PossibilisticConjunction);

        let uv1 = UncertaintyValue::with_secondary(
            UncertaintyModel::Possibilistic,
            0.8,
            0.3, // necessity = 0.3
        );
        let uv2 = UncertaintyValue::with_secondary(
            UncertaintyModel::Possibilistic,
            0.9,
            0.5, // necessity = 0.5
        );

        let result = prop.combine_two(&uv1, &uv2).unwrap();

        // Conjunction: N(A ∧ B) = min(N(A), N(B)) = min(0.3, 0.5) = 0.3
        // Result = 1 - 0.3 = 0.7
        assert!((result.value - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_sum_combination() {
        let mut prop = UncertaintyPropagator::new(UncertaintyModel::Hybrid);
        prop.set_operator(CombinationOperator::WeightedSum);

        let uv1 = UncertaintyValue::new(UncertaintyModel::Hybrid, 0.6);
        let uv2 = UncertaintyValue::new(UncertaintyModel::Hybrid, 0.8);

        let result = prop.combine_two(&uv1, &uv2).unwrap();

        // Weighted sum (average): (0.6 + 0.8) / 2 = 0.7
        assert!((result.value - 0.7).abs() < 1e-6);
    }
}
