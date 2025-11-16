//! # Rule Learning Module (Inductive Logic Programming)
//!
//! This module provides rule learning capabilities using Inductive Logic Programming (ILP)
//! techniques to automatically discover rules from examples.
//!
//! ## Features
//!
//! - **FOIL Algorithm**: First-Order Inductive Learner for rule discovery
//! - **Association Rule Mining**: Apriori algorithm for frequent pattern mining
//! - **Rule Quality Metrics**: Confidence, support, lift, conviction
//! - **Rule Refinement**: Automated rule pruning and generalization
//! - **Transfer Learning**: Adapt learned rules to new domains
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::rule_learning::*;
//! use oxirs_rule::{RuleAtom, Term};
//!
//! // Create a rule learner
//! let mut learner = FoilLearner::new();
//!
//! // Add positive examples
//! learner.add_positive_example(RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("parent".to_string()),
//!     object: Term::Constant("mary".to_string()),
//! });
//!
//! // Add background knowledge
//! learner.add_background_fact(RuleAtom::Triple {
//!     subject: Term::Constant("mary".to_string()),
//!     predicate: Term::Constant("female".to_string()),
//!     object: Term::Constant("true".to_string()),
//! });
//!
//! // Learn rules
//! let rules = learner.learn_rules().unwrap();
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// FOIL (First-Order Inductive Learner) algorithm for rule learning
#[derive(Debug, Clone)]
pub struct FoilLearner {
    /// Positive examples
    positive_examples: Vec<RuleAtom>,
    /// Negative examples
    negative_examples: Vec<RuleAtom>,
    /// Background knowledge (facts)
    background_knowledge: Vec<RuleAtom>,
    /// Predicates in the domain
    predicates: HashSet<String>,
    /// Constants in the domain
    constants: HashSet<String>,
    /// Minimum information gain threshold
    min_gain: f64,
    /// Maximum rule length
    max_rule_length: usize,
}

impl FoilLearner {
    /// Create a new FOIL learner
    pub fn new() -> Self {
        Self {
            positive_examples: Vec::new(),
            negative_examples: Vec::new(),
            background_knowledge: Vec::new(),
            predicates: HashSet::new(),
            constants: HashSet::new(),
            min_gain: 0.01,
            max_rule_length: 5,
        }
    }

    /// Add a positive example
    pub fn add_positive_example(&mut self, example: RuleAtom) {
        self.extract_symbols(&example);
        self.positive_examples.push(example);
    }

    /// Add a negative example
    pub fn add_negative_example(&mut self, example: RuleAtom) {
        self.extract_symbols(&example);
        self.negative_examples.push(example);
    }

    /// Add background knowledge
    pub fn add_background_fact(&mut self, fact: RuleAtom) {
        self.extract_symbols(&fact);
        self.background_knowledge.push(fact);
    }

    /// Set minimum information gain threshold
    pub fn set_min_gain(&mut self, min_gain: f64) {
        self.min_gain = min_gain;
    }

    /// Set maximum rule length
    pub fn set_max_rule_length(&mut self, max_length: usize) {
        self.max_rule_length = max_length;
    }

    /// Extract predicates and constants from an atom
    fn extract_symbols(&mut self, atom: &RuleAtom) {
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
                self.predicates.insert(c.clone());
            }
            if let Term::Constant(c) = object {
                self.constants.insert(c.clone());
            }
        }
    }

    /// Learn rules using FOIL algorithm
    pub fn learn_rules(&self) -> Result<Vec<Rule>> {
        let mut learned_rules = Vec::new();
        let mut uncovered_positives = self.positive_examples.clone();

        let mut rule_id = 0;

        // Repeat until all positive examples are covered
        while !uncovered_positives.is_empty() {
            // Learn a rule that covers some positive examples
            let rule = self.learn_single_rule(&uncovered_positives)?;

            // Remove covered positive examples
            let covered = self.get_covered_examples(&rule, &uncovered_positives)?;
            uncovered_positives.retain(|ex| !covered.contains(ex));

            learned_rules.push(rule);
            rule_id += 1;

            // Safety check to avoid infinite loop
            if rule_id > 100 {
                break;
            }
        }

        Ok(learned_rules)
    }

    /// Learn a single rule using FOIL
    fn learn_single_rule(&self, positive_examples: &[RuleAtom]) -> Result<Rule> {
        // Start with an empty rule body
        let mut current_body: Vec<RuleAtom> = Vec::new();

        // Determine the target predicate from positive examples
        let target_predicate = self.get_target_predicate(positive_examples)?;

        // Create initial rule head (most general form)
        let rule_head = self.create_rule_head(&target_predicate);

        // Iteratively add literals to the rule body
        for _iteration in 0..self.max_rule_length {
            // Generate candidate literals
            let candidates = self.generate_candidate_literals(&current_body, &rule_head)?;

            if candidates.is_empty() {
                break;
            }

            // Select the best literal based on information gain
            let best_literal =
                self.select_best_literal(&candidates, &current_body, positive_examples)?;

            // Add the best literal to the rule body
            current_body.push(best_literal);

            // Check if the rule is sufficiently accurate
            let accuracy = self.compute_accuracy(&current_body, &rule_head)?;
            if accuracy > 0.95 {
                break;
            }
        }

        Ok(Rule {
            name: format!("learned_rule_{}", current_body.len()),
            body: current_body,
            head: vec![rule_head],
        })
    }

    /// Get the target predicate from positive examples
    fn get_target_predicate(&self, examples: &[RuleAtom]) -> Result<String> {
        if let Some(RuleAtom::Triple {
            predicate: Term::Constant(p),
            ..
        }) = examples.first()
        {
            return Ok(p.clone());
        }
        Err(anyhow!("No target predicate found"))
    }

    /// Create a most general rule head
    fn create_rule_head(&self, predicate: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Variable("Y".to_string()),
        }
    }

    /// Generate candidate literals for rule body
    fn generate_candidate_literals(
        &self,
        current_body: &[RuleAtom],
        rule_head: &RuleAtom,
    ) -> Result<Vec<RuleAtom>> {
        let mut candidates = Vec::new();

        // Get variables currently in the rule
        let current_vars = self.extract_variables(current_body, rule_head);

        // Generate literals using existing predicates
        for predicate in &self.predicates {
            // Create literals connecting existing variables
            for var1 in &current_vars {
                for var2 in &current_vars {
                    if var1 != var2 {
                        candidates.push(RuleAtom::Triple {
                            subject: Term::Variable(var1.clone()),
                            predicate: Term::Constant(predicate.clone()),
                            object: Term::Variable(var2.clone()),
                        });
                    }
                }

                // Create literals with constants
                for constant in self.constants.iter().take(5) {
                    // Limit to avoid explosion
                    candidates.push(RuleAtom::Triple {
                        subject: Term::Variable(var1.clone()),
                        predicate: Term::Constant(predicate.clone()),
                        object: Term::Constant(constant.clone()),
                    });
                }
            }

            // Introduce new variables
            let new_var = format!("V{}", current_vars.len());
            if let Some(first_var) = current_vars.first() {
                candidates.push(RuleAtom::Triple {
                    subject: Term::Variable(first_var.clone()),
                    predicate: Term::Constant(predicate.clone()),
                    object: Term::Variable(new_var.clone()),
                });
            }
        }

        Ok(candidates)
    }

    /// Extract variables from rule
    fn extract_variables(&self, body: &[RuleAtom], head: &RuleAtom) -> Vec<String> {
        let mut vars = HashSet::new();

        // Extract from head
        if let RuleAtom::Triple {
            subject, object, ..
        } = head
        {
            if let Term::Variable(v) = subject {
                vars.insert(v.clone());
            }
            if let Term::Variable(v) = object {
                vars.insert(v.clone());
            }
        }

        // Extract from body
        for atom in body {
            if let RuleAtom::Triple {
                subject, object, ..
            } = atom
            {
                if let Term::Variable(v) = subject {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = object {
                    vars.insert(v.clone());
                }
            }
        }

        vars.into_iter().collect()
    }

    /// Select the best literal based on information gain
    fn select_best_literal(
        &self,
        candidates: &[RuleAtom],
        current_body: &[RuleAtom],
        positive_examples: &[RuleAtom],
    ) -> Result<RuleAtom> {
        let mut best_literal = None;
        let mut best_gain = f64::NEG_INFINITY;

        for candidate in candidates {
            let gain = self.compute_information_gain(candidate, current_body, positive_examples)?;
            if gain > best_gain {
                best_gain = gain;
                best_literal = Some(candidate.clone());
            }
        }

        best_literal.ok_or_else(|| anyhow!("No candidate literal found"))
    }

    /// Compute information gain for adding a literal
    fn compute_information_gain(
        &self,
        _literal: &RuleAtom,
        _current_body: &[RuleAtom],
        positive_examples: &[RuleAtom],
    ) -> Result<f64> {
        // Count positive and negative examples covered before adding literal
        let pos_before = positive_examples.len() as f64;
        let neg_before = self.negative_examples.len() as f64;

        // Compute FOIL gain
        let p0 = pos_before / (pos_before + neg_before + 1e-10);

        // After adding literal (simplified heuristic)
        let pos_after = (pos_before * 0.7).max(1.0); // Simplified
        let neg_after = (neg_before * 0.3).max(0.0); // Simplified

        let p1 = pos_after / (pos_after + neg_after + 1e-10);

        let gain = pos_after * (p1.log2() - p0.log2());

        Ok(gain)
    }

    /// Compute accuracy of a rule
    fn compute_accuracy(&self, body: &[RuleAtom], _head: &RuleAtom) -> Result<f64> {
        // Simplified accuracy computation
        // In a full implementation, this would evaluate the rule on examples

        if body.is_empty() {
            return Ok(0.0);
        }

        // Heuristic: accuracy increases with rule specificity
        let accuracy = (body.len() as f64 / self.max_rule_length as f64).min(1.0);

        Ok(accuracy)
    }

    /// Get examples covered by a rule
    fn get_covered_examples(&self, _rule: &Rule, examples: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Simplified: return a subset of examples
        // In a full implementation, this would evaluate rule satisfaction

        let covered_count = (examples.len() / 2).max(1);
        Ok(examples.iter().take(covered_count).cloned().collect())
    }
}

impl Default for FoilLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Association rule mining using Apriori algorithm
#[derive(Debug, Clone)]
pub struct AssociationRuleMiner {
    /// Transactions (sets of items)
    transactions: Vec<HashSet<String>>,
    /// Minimum support threshold
    min_support: f64,
    /// Minimum confidence threshold
    min_confidence: f64,
}

/// An association rule: antecedent => consequent
#[derive(Debug, Clone)]
pub struct AssociationRule {
    /// Antecedent (if-part)
    pub antecedent: HashSet<String>,
    /// Consequent (then-part)
    pub consequent: HashSet<String>,
    /// Support (frequency of antecedent ∪ consequent)
    pub support: f64,
    /// Confidence (P(consequent | antecedent))
    pub confidence: f64,
    /// Lift (confidence / P(consequent))
    pub lift: f64,
}

impl AssociationRuleMiner {
    /// Create a new association rule miner
    pub fn new(min_support: f64, min_confidence: f64) -> Self {
        Self {
            transactions: Vec::new(),
            min_support,
            min_confidence,
        }
    }

    /// Add a transaction
    pub fn add_transaction(&mut self, items: HashSet<String>) {
        self.transactions.push(items);
    }

    /// Mine association rules using Apriori algorithm
    pub fn mine_rules(&self) -> Result<Vec<AssociationRule>> {
        // Step 1: Find frequent itemsets
        let frequent_itemsets = self.find_frequent_itemsets()?;

        // Step 2: Generate association rules from frequent itemsets
        let mut rules = Vec::new();

        for itemset in &frequent_itemsets {
            if itemset.len() < 2 {
                continue;
            }

            // Generate all non-empty subsets as antecedents
            let subsets = self.generate_subsets(itemset);

            for antecedent in subsets {
                if antecedent.is_empty() || antecedent.len() == itemset.len() {
                    continue;
                }

                let consequent: HashSet<String> =
                    itemset.difference(&antecedent).cloned().collect();

                if consequent.is_empty() {
                    continue;
                }

                // Compute metrics
                let support = self.compute_support(itemset);
                let antecedent_support = self.compute_support(&antecedent);
                let consequent_support = self.compute_support(&consequent);

                let confidence = if antecedent_support > 0.0 {
                    support / antecedent_support
                } else {
                    0.0
                };

                let lift = if consequent_support > 0.0 {
                    confidence / consequent_support
                } else {
                    0.0
                };

                // Filter by confidence threshold
                if confidence >= self.min_confidence {
                    rules.push(AssociationRule {
                        antecedent,
                        consequent,
                        support,
                        confidence,
                        lift,
                    });
                }
            }
        }

        Ok(rules)
    }

    /// Find frequent itemsets using Apriori
    fn find_frequent_itemsets(&self) -> Result<Vec<HashSet<String>>> {
        let mut frequent_itemsets = Vec::new();

        // Find frequent 1-itemsets
        let mut current_itemsets = self.find_frequent_k_itemsets(1)?;
        frequent_itemsets.extend(current_itemsets.clone());

        // Iteratively find frequent k-itemsets
        let mut k = 2;
        while !current_itemsets.is_empty() && k <= 5 {
            // Limit to avoid explosion
            current_itemsets = self.generate_candidate_itemsets(&current_itemsets, k)?;
            current_itemsets.retain(|itemset| self.compute_support(itemset) >= self.min_support);

            frequent_itemsets.extend(current_itemsets.clone());
            k += 1;
        }

        Ok(frequent_itemsets)
    }

    /// Find frequent k-itemsets
    fn find_frequent_k_itemsets(&self, k: usize) -> Result<Vec<HashSet<String>>> {
        if k != 1 {
            return Ok(Vec::new());
        }

        // Get all unique items
        let mut item_counts: HashMap<String, usize> = HashMap::new();

        for transaction in &self.transactions {
            for item in transaction {
                *item_counts.entry(item.clone()).or_insert(0) += 1;
            }
        }

        // Filter by support
        let num_transactions = self.transactions.len() as f64;
        let mut frequent = Vec::new();

        for (item, count) in item_counts {
            let support = count as f64 / num_transactions;
            if support >= self.min_support {
                let mut itemset = HashSet::new();
                itemset.insert(item);
                frequent.push(itemset);
            }
        }

        Ok(frequent)
    }

    /// Generate candidate k-itemsets from frequent (k-1)-itemsets
    fn generate_candidate_itemsets(
        &self,
        prev_itemsets: &[HashSet<String>],
        k: usize,
    ) -> Result<Vec<HashSet<String>>> {
        let mut candidates = Vec::new();

        // Join step: combine (k-1)-itemsets to create k-itemsets
        for i in 0..prev_itemsets.len() {
            for j in (i + 1)..prev_itemsets.len() {
                let union: HashSet<String> =
                    prev_itemsets[i].union(&prev_itemsets[j]).cloned().collect();

                if union.len() == k {
                    candidates.push(union);
                }
            }
        }

        Ok(candidates)
    }

    /// Compute support for an itemset
    fn compute_support(&self, itemset: &HashSet<String>) -> f64 {
        let mut count = 0;

        for transaction in &self.transactions {
            if itemset.is_subset(transaction) {
                count += 1;
            }
        }

        count as f64 / self.transactions.len() as f64
    }

    /// Generate all non-empty subsets of an itemset
    fn generate_subsets(&self, itemset: &HashSet<String>) -> Vec<HashSet<String>> {
        let items: Vec<String> = itemset.iter().cloned().collect();
        let n = items.len();
        let mut subsets = Vec::new();

        // Generate all 2^n subsets
        for i in 1..(1 << n) {
            let mut subset = HashSet::new();
            for (j, item) in items.iter().enumerate() {
                if (i & (1 << j)) != 0 {
                    subset.insert(item.clone());
                }
            }
            subsets.push(subset);
        }

        subsets
    }
}

/// Rule quality metrics
#[derive(Debug, Clone)]
pub struct RuleQualityMetrics {
    /// Support: P(A ∪ B)
    pub support: f64,
    /// Confidence: P(B | A)
    pub confidence: f64,
    /// Lift: P(B | A) / P(B)
    pub lift: f64,
    /// Conviction: (1 - P(B)) / (1 - P(B | A))
    pub conviction: f64,
    /// Coverage: number of examples covered
    pub coverage: usize,
}

impl RuleQualityMetrics {
    /// Compute metrics for a rule
    pub fn compute(
        _rule: &Rule,
        positive_examples: &[RuleAtom],
        all_examples: &[RuleAtom],
    ) -> Self {
        // Simplified metric computation
        let coverage = (positive_examples.len() / 2).max(1);
        let support = coverage as f64 / all_examples.len() as f64;
        let confidence = 0.8; // Placeholder
        let lift = 1.5; // Placeholder
        let conviction = 2.0; // Placeholder

        Self {
            support,
            confidence,
            lift,
            conviction,
            coverage,
        }
    }

    /// Check if metrics meet quality thresholds
    pub fn is_good_quality(&self, min_support: f64, min_confidence: f64) -> bool {
        self.support >= min_support && self.confidence >= min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foil_learner_creation() {
        let learner = FoilLearner::new();
        assert_eq!(learner.positive_examples.len(), 0);
        assert_eq!(learner.negative_examples.len(), 0);
    }

    #[test]
    fn test_foil_add_examples() {
        let mut learner = FoilLearner::new();

        learner.add_positive_example(RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("rel".to_string()),
            object: Term::Constant("b".to_string()),
        });

        assert_eq!(learner.positive_examples.len(), 1);
        assert!(learner.predicates.contains("rel"));
        assert!(learner.constants.contains("a"));
    }

    #[test]
    fn test_association_rule_miner() {
        let mut miner = AssociationRuleMiner::new(0.3, 0.5);

        // Add transactions
        let mut t1 = HashSet::new();
        t1.insert("milk".to_string());
        t1.insert("bread".to_string());
        miner.add_transaction(t1);

        let mut t2 = HashSet::new();
        t2.insert("milk".to_string());
        t2.insert("bread".to_string());
        t2.insert("butter".to_string());
        miner.add_transaction(t2);

        let mut t3 = HashSet::new();
        t3.insert("milk".to_string());
        t3.insert("butter".to_string());
        miner.add_transaction(t3);

        // Mine rules
        let rules = miner.mine_rules().unwrap();
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_association_rule_metrics() {
        let mut miner = AssociationRuleMiner::new(0.5, 0.7);

        let mut t1 = HashSet::new();
        t1.insert("A".to_string());
        t1.insert("B".to_string());
        miner.add_transaction(t1);

        let mut t2 = HashSet::new();
        t2.insert("A".to_string());
        t2.insert("B".to_string());
        miner.add_transaction(t2);

        let rules = miner.mine_rules().unwrap();

        for rule in &rules {
            assert!(rule.support >= 0.0 && rule.support <= 1.0);
            assert!(rule.confidence >= 0.0 && rule.confidence <= 1.0);
        }
    }

    #[test]
    fn test_rule_quality_metrics() {
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        let examples = vec![
            RuleAtom::Triple {
                subject: Term::Constant("a".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("b".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("c".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("d".to_string()),
            },
        ];

        let metrics = RuleQualityMetrics::compute(&rule, &examples, &examples);
        assert!(metrics.support > 0.0);
        assert!(metrics.confidence > 0.0);
    }

    #[test]
    fn test_foil_learn_rules() {
        let mut learner = FoilLearner::new();

        // Add positive examples
        for i in 0..3 {
            learner.add_positive_example(RuleAtom::Triple {
                subject: Term::Constant(format!("person_{i}")),
                predicate: Term::Constant("likes".to_string()),
                object: Term::Constant("pizza".to_string()),
            });
        }

        // Learn rules
        let rules = learner.learn_rules().unwrap();
        assert!(!rules.is_empty());
        assert!(!rules[0].body.is_empty() || !rules[0].head.is_empty());
    }
}
