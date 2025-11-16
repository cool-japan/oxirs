//! # Possibilistic Logic Module
//!
//! This module provides possibilistic reasoning for handling uncertainty using
//! possibility theory, which is complementary to probability theory.
//!
//! ## Features
//!
//! - **Possibility Measures**: Π(A) - degree to which A is possible
//! - **Necessity Measures**: N(A) - degree to which A is certain
//! - **Possibility Distributions**: π-worlds with possibility degrees
//! - **Possibilistic Formulas**: (φ, α) pairs with certainty weights
//! - **Possibilistic Resolution**: Inference using resolution principle
//! - **Possibilistic Entailment**: Computing logical consequences
//! - **Fuzzy Integration**: Compatible with fuzzy logic reasoning
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::possibilistic::*;
//!
//! // Create a possibilistic knowledge base
//! let mut kb = PossibilisticKnowledgeBase::new();
//!
//! // Add possibilistic formulas: (formula, certainty)
//! // "If it's cloudy, it will rain" with certainty 0.8
//! kb.add_formula(
//!     "Cloudy -> Rain".to_string(),
//!     0.8
//! ).unwrap();
//!
//! // "It is cloudy" with certainty 0.9
//! kb.add_formula(
//!     "Cloudy".to_string(),
//!     0.9
//! ).unwrap();
//!
//! // Query: What's the necessity of Rain?
//! let necessity = kb.query_necessity("Rain").unwrap();
//! println!("Necessity(Rain) = {}", necessity);
//!
//! // Necessity(Rain) ≥ min(0.8, 0.9) = 0.8 by possibilistic modus ponens
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Theory Background
//!
//! Possibilistic logic combines classical logic with possibility theory:
//! - **Possibility Π(φ)**: How much φ is consistent with our knowledge
//! - **Necessity N(φ)**: How much φ is entailed by our knowledge
//! - **Duality**: N(φ) = 1 - Π(¬φ)
//! - **Resolution**: From (φ ∨ ψ, α) and (¬φ ∨ χ, β), derive (ψ ∨ χ, min(α,β))

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Possibilistic formula: (formula, certainty_degree)
#[derive(Debug, Clone, PartialEq)]
pub struct PossibilisticFormula {
    /// Propositional formula (simplified representation)
    pub formula: String,
    /// Certainty degree α ∈ \[0,1\]
    /// Higher values indicate more certain beliefs
    pub certainty: f64,
}

impl PossibilisticFormula {
    /// Create a new possibilistic formula
    pub fn new(formula: String, certainty: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&certainty) {
            return Err(anyhow!("Certainty must be in [0,1], got {}", certainty));
        }

        Ok(Self { formula, certainty })
    }

    /// Check if this formula is a fact (not an implication)
    pub fn is_fact(&self) -> bool {
        !self.formula.contains("->")
    }

    /// Check if this is an implication (rule)
    pub fn is_implication(&self) -> bool {
        self.formula.contains("->")
    }

    /// Parse implication into (antecedent, consequent)
    pub fn parse_implication(&self) -> Option<(String, String)> {
        if !self.is_implication() {
            return None;
        }

        let parts: Vec<&str> = self.formula.split("->").collect();
        if parts.len() != 2 {
            return None;
        }

        Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
}

/// Possibilistic knowledge base
#[derive(Debug, Clone)]
pub struct PossibilisticKnowledgeBase {
    /// Set of possibilistic formulas
    formulas: Vec<PossibilisticFormula>,
    /// Possibility distribution over atoms
    possibility_dist: HashMap<String, f64>,
}

impl PossibilisticKnowledgeBase {
    /// Create a new possibilistic knowledge base
    pub fn new() -> Self {
        Self {
            formulas: Vec::new(),
            possibility_dist: HashMap::new(),
        }
    }

    /// Add a possibilistic formula to the knowledge base
    pub fn add_formula(&mut self, formula: String, certainty: f64) -> Result<()> {
        let poss_formula = PossibilisticFormula::new(formula.clone(), certainty)?;

        // Update possibility distribution for atomic formulas
        if poss_formula.is_fact() {
            self.possibility_dist.insert(formula, certainty);
        }

        self.formulas.push(poss_formula);
        Ok(())
    }

    /// Get all formulas
    pub fn get_formulas(&self) -> &[PossibilisticFormula] {
        &self.formulas
    }

    /// Query the necessity degree of a proposition
    ///
    /// N(φ) = 1 - Π(¬φ) where Π is the possibility measure
    pub fn query_necessity(&self, proposition: &str) -> Result<f64> {
        // Compute using possibilistic resolution
        let necessity = self.compute_necessity(proposition)?;
        Ok(necessity)
    }

    /// Query the possibility degree of a proposition
    ///
    /// Π(φ) = max{π(ω) : ω ⊨ φ}
    pub fn query_possibility(&self, proposition: &str) -> Result<f64> {
        // Direct lookup for facts
        if let Some(&poss) = self.possibility_dist.get(proposition) {
            return Ok(poss);
        }

        // Compute using possibilistic inference
        let possibility = self.compute_possibility(proposition)?;
        Ok(possibility)
    }

    /// Compute necessity using possibilistic resolution
    fn compute_necessity(&self, proposition: &str) -> Result<f64> {
        let mut max_certainty: f64 = 0.0;

        // Direct facts
        for formula in &self.formulas {
            if formula.formula == proposition && formula.is_fact() {
                max_certainty = max_certainty.max(formula.certainty);
            }
        }

        // Possibilistic modus ponens: From (A, α) and (A -> B, β), derive (B, min(α,β))
        for formula in &self.formulas {
            if let Some((antecedent, consequent)) = formula.parse_implication() {
                if consequent == proposition {
                    // Check if antecedent is known
                    let antecedent_certainty = self.query_necessity(&antecedent)?;
                    if antecedent_certainty > 0.0 {
                        let derived_certainty = antecedent_certainty.min(formula.certainty);
                        max_certainty = max_certainty.max(derived_certainty);
                    }
                }
            }
        }

        Ok(max_certainty)
    }

    /// Compute possibility measure
    fn compute_possibility(&self, proposition: &str) -> Result<f64> {
        // For simple case, possibility is 1 - necessity of negation
        // In full implementation, this would use possibility distribution

        // Direct lookup
        if let Some(&poss) = self.possibility_dist.get(proposition) {
            return Ok(poss);
        }

        // For derived propositions, use inference
        let necessity = self.compute_necessity(proposition)?;

        // Upper bound: if necessity > 0, then possibility = 1
        // Lower bound: possibility ≥ necessity
        if necessity > 0.0 {
            Ok(1.0)
        } else {
            Ok(0.0) // No information about this proposition
        }
    }

    /// Perform possibilistic resolution between two formulas
    ///
    /// From (φ ∨ ψ, α) and (¬φ ∨ χ, β), derive (ψ ∨ χ, min(α,β))
    pub fn resolve(
        &self,
        formula1: &PossibilisticFormula,
        formula2: &PossibilisticFormula,
    ) -> Option<PossibilisticFormula> {
        // Simplified resolution for implications
        // In full implementation, this would use CNF and general resolution

        // Handle modus ponens: (A, α) and (A -> B, β) => (B, min(α,β))
        if !formula1.is_implication() {
            if let Some((ant, cons)) = formula2.parse_implication() {
                if formula1.formula == ant {
                    let certainty = formula1.certainty.min(formula2.certainty);
                    return PossibilisticFormula::new(cons, certainty).ok();
                }
            }
        }

        // Handle symmetric case
        if !formula2.is_implication() {
            if let Some((ant, cons)) = formula1.parse_implication() {
                if formula2.formula == ant {
                    let certainty = formula1.certainty.min(formula2.certainty);
                    return PossibilisticFormula::new(cons, certainty).ok();
                }
            }
        }

        None
    }

    /// Perform possibilistic forward chaining
    pub fn forward_chain(&mut self) -> Result<Vec<PossibilisticFormula>> {
        let mut derived = Vec::new();
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            let formulas_snapshot = self.formulas.clone();

            for i in 0..formulas_snapshot.len() {
                for j in 0..formulas_snapshot.len() {
                    if i == j {
                        continue;
                    }

                    if let Some(resolved) =
                        self.resolve(&formulas_snapshot[i], &formulas_snapshot[j])
                    {
                        // Check if this is a new formula
                        let is_new = !self.formulas.iter().any(|f| {
                            f.formula == resolved.formula
                                && (f.certainty - resolved.certainty).abs() < 1e-6
                        });

                        if is_new {
                            // Add to knowledge base
                            self.formulas.push(resolved.clone());
                            derived.push(resolved.clone());

                            // Update possibility distribution if it's a fact
                            if resolved.is_fact() {
                                self.possibility_dist
                                    .insert(resolved.formula.clone(), resolved.certainty);
                            }

                            changed = true;
                        }
                    }
                }
            }
        }

        Ok(derived)
    }

    /// Get all atomic propositions mentioned in the knowledge base
    pub fn get_atoms(&self) -> HashSet<String> {
        let mut atoms = HashSet::new();

        for formula in &self.formulas {
            if formula.is_fact() {
                atoms.insert(formula.formula.clone());
            } else if let Some((ant, cons)) = formula.parse_implication() {
                atoms.insert(ant);
                atoms.insert(cons);
            }
        }

        atoms
    }

    /// Get inconsistency degree (how contradictory the KB is)
    ///
    /// Inc(KB) = max{α : KB ⊢ (⊥, α)}
    pub fn inconsistency_degree(&self) -> f64 {
        // Simplified: check for explicit contradictions
        // In full implementation, would use refutation-based approach

        let mut max_inc: f64 = 0.0;

        for formula in &self.formulas {
            if formula.formula == "false" || formula.formula == "⊥" {
                max_inc = max_inc.max(formula.certainty);
            }
        }

        max_inc
    }

    /// Check if knowledge base is consistent (Inc = 0)
    pub fn is_consistent(&self) -> bool {
        self.inconsistency_degree() < 1e-6
    }
}

impl Default for PossibilisticKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Possibilistic reasoner for rule-based inference
#[derive(Debug, Clone)]
pub struct PossibilisticReasoner {
    /// Knowledge base
    kb: PossibilisticKnowledgeBase,
    /// Inference cache
    cache: HashMap<String, (f64, f64)>, // (necessity, possibility)
}

impl PossibilisticReasoner {
    /// Create a new possibilistic reasoner
    pub fn new() -> Self {
        Self {
            kb: PossibilisticKnowledgeBase::new(),
            cache: HashMap::new(),
        }
    }

    /// Add a rule with certainty
    pub fn add_rule(
        &mut self,
        antecedent: String,
        consequent: String,
        certainty: f64,
    ) -> Result<()> {
        let rule = format!("{} -> {}", antecedent, consequent);
        self.kb.add_formula(rule, certainty)?;
        self.cache.clear(); // Invalidate cache
        Ok(())
    }

    /// Add a fact with certainty
    pub fn add_fact(&mut self, fact: String, certainty: f64) -> Result<()> {
        self.kb.add_formula(fact, certainty)?;
        self.cache.clear(); // Invalidate cache
        Ok(())
    }

    /// Query with caching
    pub fn query(&mut self, proposition: &str) -> Result<(f64, f64)> {
        // Check cache
        if let Some(&cached) = self.cache.get(proposition) {
            return Ok(cached);
        }

        // Compute
        let necessity = self.kb.query_necessity(proposition)?;
        let possibility = self.kb.query_possibility(proposition)?;

        // Cache result
        self.cache
            .insert(proposition.to_string(), (necessity, possibility));

        Ok((necessity, possibility))
    }

    /// Perform inference (forward chaining)
    pub fn infer(&mut self) -> Result<Vec<PossibilisticFormula>> {
        let derived = self.kb.forward_chain()?;
        self.cache.clear(); // Invalidate cache after inference
        Ok(derived)
    }

    /// Get the knowledge base
    pub fn get_kb(&self) -> &PossibilisticKnowledgeBase {
        &self.kb
    }

    /// Check consistency
    pub fn is_consistent(&self) -> bool {
        self.kb.is_consistent()
    }

    /// Get most certain proposition
    pub fn get_most_certain(&self) -> Option<(String, f64)> {
        self.kb
            .get_formulas()
            .iter()
            .filter(|f| f.is_fact())
            .max_by(|a, b| a.certainty.partial_cmp(&b.certainty).unwrap())
            .map(|f| (f.formula.clone(), f.certainty))
    }
}

impl Default for PossibilisticReasoner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_possibilistic_formula() {
        let formula = PossibilisticFormula::new("Rain".to_string(), 0.8).unwrap();
        assert_eq!(formula.formula, "Rain");
        assert!((formula.certainty - 0.8).abs() < 1e-10);
        assert!(formula.is_fact());
        assert!(!formula.is_implication());
    }

    #[test]
    fn test_possibilistic_formula_invalid_certainty() {
        let result = PossibilisticFormula::new("Rain".to_string(), 1.5);
        assert!(result.is_err());

        let result = PossibilisticFormula::new("Rain".to_string(), -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_implication_parsing() {
        let formula = PossibilisticFormula::new("Cloudy -> Rain".to_string(), 0.8).unwrap();
        assert!(formula.is_implication());

        let (ant, cons) = formula.parse_implication().unwrap();
        assert_eq!(ant, "Cloudy");
        assert_eq!(cons, "Rain");
    }

    #[test]
    fn test_knowledge_base_basic() {
        let mut kb = PossibilisticKnowledgeBase::new();

        kb.add_formula("Sunny".to_string(), 0.9).unwrap();
        kb.add_formula("Hot".to_string(), 0.7).unwrap();

        assert_eq!(kb.get_formulas().len(), 2);
    }

    #[test]
    fn test_possibilistic_modus_ponens() {
        let mut kb = PossibilisticKnowledgeBase::new();

        // Add rule: Cloudy -> Rain with certainty 0.8
        kb.add_formula("Cloudy -> Rain".to_string(), 0.8).unwrap();

        // Add fact: Cloudy with certainty 0.9
        kb.add_formula("Cloudy".to_string(), 0.9).unwrap();

        // Query: Necessity of Rain
        let necessity = kb.query_necessity("Rain").unwrap();

        // Should be min(0.8, 0.9) = 0.8
        assert!((necessity - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_possibilistic_chain() {
        let mut kb = PossibilisticKnowledgeBase::new();

        // Rule chain:
        // A with certainty 1.0
        // A -> B with certainty 0.9
        // B -> C with certainty 0.8

        kb.add_formula("A".to_string(), 1.0).unwrap();
        kb.add_formula("A -> B".to_string(), 0.9).unwrap();
        kb.add_formula("B -> C".to_string(), 0.8).unwrap();

        // Perform forward chaining
        let _ = kb.forward_chain().unwrap();

        // Check necessity of C
        let necessity_c = kb.query_necessity("C").unwrap();

        // Should be min(1.0, 0.9, 0.8) = 0.8
        assert!((necessity_c - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_possibilistic_resolution() {
        let kb = PossibilisticKnowledgeBase::new();

        let f1 = PossibilisticFormula::new("Cloudy".to_string(), 0.9).unwrap();
        let f2 = PossibilisticFormula::new("Cloudy -> Rain".to_string(), 0.8).unwrap();

        let resolved = kb.resolve(&f1, &f2).unwrap();

        assert_eq!(resolved.formula, "Rain");
        assert!((resolved.certainty - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_possibilistic_forward_chaining() {
        let mut kb = PossibilisticKnowledgeBase::new();

        kb.add_formula("Cloudy".to_string(), 0.9).unwrap();
        kb.add_formula("Cloudy -> Rain".to_string(), 0.8).unwrap();
        kb.add_formula("Rain -> WetGround".to_string(), 0.7)
            .unwrap();

        let derived = kb.forward_chain().unwrap();

        // Should derive Rain and WetGround
        assert!(derived.iter().any(|f| f.formula == "Rain"));
        assert!(derived.iter().any(|f| f.formula == "WetGround"));

        // Check certainties
        let rain = derived.iter().find(|f| f.formula == "Rain").unwrap();
        assert!((rain.certainty - 0.8).abs() < 1e-10);

        let wet = derived.iter().find(|f| f.formula == "WetGround").unwrap();
        assert!((wet.certainty - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_possibilistic_reasoner() {
        let mut reasoner = PossibilisticReasoner::new();

        reasoner.add_fact("Cloudy".to_string(), 0.9).unwrap();
        reasoner
            .add_rule("Cloudy".to_string(), "Rain".to_string(), 0.8)
            .unwrap();

        // Perform inference
        reasoner.infer().unwrap();

        // Query
        let (necessity, possibility) = reasoner.query("Rain").unwrap();

        assert!((necessity - 0.8).abs() < 1e-10);
        assert!(possibility >= necessity);
    }

    #[test]
    fn test_inconsistency_detection() {
        let mut kb = PossibilisticKnowledgeBase::new();

        kb.add_formula("Sunny".to_string(), 0.9).unwrap();
        kb.add_formula("NotSunny".to_string(), 0.7).unwrap();

        // Not directly contradictory in our simplified system
        assert!(kb.is_consistent());

        // Add explicit contradiction
        kb.add_formula("false".to_string(), 0.5).unwrap();

        assert!(!kb.is_consistent());
        assert!((kb.inconsistency_degree() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_possibility_necessity_duality() {
        let mut kb = PossibilisticKnowledgeBase::new();

        kb.add_formula("Sunny".to_string(), 0.8).unwrap();

        let necessity = kb.query_necessity("Sunny").unwrap();
        let possibility = kb.query_possibility("Sunny").unwrap();

        // For a fact, necessity ≤ possibility
        assert!(necessity <= possibility + 1e-10);

        // For a fact with certainty α, necessity = α
        assert!((necessity - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reasoner_caching() {
        let mut reasoner = PossibilisticReasoner::new();

        reasoner.add_fact("A".to_string(), 0.9).unwrap();

        // First query (cache miss)
        let (n1, p1) = reasoner.query("A").unwrap();

        // Second query (cache hit)
        let (n2, p2) = reasoner.query("A").unwrap();

        assert_eq!(n1, n2);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_get_most_certain() {
        let mut reasoner = PossibilisticReasoner::new();

        reasoner.add_fact("A".to_string(), 0.7).unwrap();
        reasoner.add_fact("B".to_string(), 0.9).unwrap();
        reasoner.add_fact("C".to_string(), 0.6).unwrap();

        let (most_certain, certainty) = reasoner.get_most_certain().unwrap();

        assert_eq!(most_certain, "B");
        assert!((certainty - 0.9).abs() < 1e-10);
    }
}
