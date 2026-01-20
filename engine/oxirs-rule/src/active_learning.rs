//! Active Learning for Rule Validation
//!
//! Implements active learning strategies to efficiently validate rules by selecting
//! the most informative examples for manual review, reducing validation effort.
//!
//! # Features
//!
//! - **Uncertainty Sampling**: Select rules with highest prediction uncertainty
//! - **Query-by-Committee**: Use ensemble disagreement to find ambiguous cases
//! - **Diversity Sampling**: Ensure diverse rule coverage
//! - **Expected Model Change**: Select examples that would change the model most
//! - **Validation Workflow**: Interactive validation with feedback loops
//! - **Confidence Tracking**: Monitor validation confidence over time
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::active_learning::{ActiveLearner, SamplingStrategy};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut learner = ActiveLearner::new();
//!
//! // Add candidate rules for validation
//! let rule = Rule {
//!     name: "candidate_rule".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("hasParent".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("Y".to_string()),
//!         predicate: Term::Constant("hasChild".to_string()),
//!         object: Term::Variable("X".to_string()),
//!     }],
//! };
//!
//! learner.add_candidate_rule(rule, 0.75);
//!
//! // Select most informative rule for validation
//! let next_rule = learner.select_next_for_validation(SamplingStrategy::UncertaintySampling);
//!
//! if let Some(selected) = next_rule {
//!     println!("Please validate: {}", selected.name);
//!     // User validates and provides feedback
//!     learner.record_validation(&selected.name, true, 0.95);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::Rule;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use tracing::info;

/// Sampling strategy for active learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Select rules with highest uncertainty
    UncertaintySampling,
    /// Use ensemble disagreement
    QueryByCommittee,
    /// Ensure diverse rule coverage
    DiversitySampling,
    /// Expected model change
    ExpectedModelChange,
    /// Random sampling (baseline)
    RandomSampling,
}

/// Candidate rule with confidence
#[derive(Debug, Clone)]
pub struct CandidateRule {
    /// Rule to validate
    pub rule: Rule,
    /// Predicted confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Validation status
    pub validated: bool,
    /// Validation result (if validated)
    pub validation_result: Option<bool>,
    /// Actual confidence after validation
    pub actual_confidence: Option<f64>,
    /// Selection count
    pub selection_count: usize,
}

/// Validation feedback
#[derive(Debug, Clone)]
pub struct ValidationFeedback {
    /// Rule name
    pub rule_name: String,
    /// Is rule correct?
    pub is_correct: bool,
    /// Confidence in validation (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Active learner for rule validation
pub struct ActiveLearner {
    /// Candidate rules to validate
    candidates: HashMap<String, CandidateRule>,
    /// Validation history
    validation_history: Vec<ValidationFeedback>,
    /// Random number generator
    rng: StdRng,
    /// Diversity weight (0.0 to 1.0)
    diversity_weight: f64,
    /// Minimum confidence threshold
    min_confidence_threshold: f64,
    /// Committee size for query-by-committee
    committee_size: usize,
}

impl Default for ActiveLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveLearner {
    /// Create a new active learner
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            candidates: HashMap::new(),
            validation_history: Vec::new(),
            rng: seeded_rng(seed),
            diversity_weight: 0.3,
            min_confidence_threshold: 0.5,
            committee_size: 5,
        }
    }

    /// Add a candidate rule for validation
    pub fn add_candidate_rule(&mut self, rule: Rule, confidence: f64) {
        let name = rule.name.clone();
        self.candidates.insert(
            name,
            CandidateRule {
                rule,
                confidence: confidence.clamp(0.0, 1.0),
                validated: false,
                validation_result: None,
                actual_confidence: None,
                selection_count: 0,
            },
        );
    }

    /// Select next rule for validation using specified strategy
    pub fn select_next_for_validation(&mut self, strategy: SamplingStrategy) -> Option<Rule> {
        let unvalidated: Vec<_> = self
            .candidates
            .iter()
            .filter(|(_, c)| !c.validated)
            .map(|(name, c)| (name.clone(), c.clone()))
            .collect();

        if unvalidated.is_empty() {
            return None;
        }

        let selected_name = match strategy {
            SamplingStrategy::UncertaintySampling => self.uncertainty_sampling(&unvalidated),
            SamplingStrategy::QueryByCommittee => self.query_by_committee(&unvalidated),
            SamplingStrategy::DiversitySampling => self.diversity_sampling(&unvalidated),
            SamplingStrategy::ExpectedModelChange => self.expected_model_change(&unvalidated),
            SamplingStrategy::RandomSampling => self.random_sampling(&unvalidated),
        }?;

        // Increment selection count
        if let Some(candidate) = self.candidates.get_mut(&selected_name) {
            candidate.selection_count += 1;
            info!(
                "Selected rule '{}' for validation (confidence: {:.3})",
                selected_name, candidate.confidence
            );
            return Some(candidate.rule.clone());
        }

        None
    }

    /// Record validation result
    pub fn record_validation(&mut self, rule_name: &str, is_correct: bool, confidence: f64) {
        if let Some(candidate) = self.candidates.get_mut(rule_name) {
            candidate.validated = true;
            candidate.validation_result = Some(is_correct);
            candidate.actual_confidence = Some(confidence.clamp(0.0, 1.0));

            self.validation_history.push(ValidationFeedback {
                rule_name: rule_name.to_string(),
                is_correct,
                confidence: confidence.clamp(0.0, 1.0),
                timestamp: std::time::SystemTime::now(),
            });

            info!(
                "Recorded validation for '{}': correct={}, confidence={:.3}",
                rule_name, is_correct, confidence
            );
        }
    }

    /// Get validation accuracy
    pub fn validation_accuracy(&self) -> f64 {
        let validated: Vec<_> = self
            .candidates
            .values()
            .filter(|c| c.validated && c.validation_result.is_some())
            .collect();

        if validated.is_empty() {
            return 0.0;
        }

        let correct_predictions = validated
            .iter()
            .filter(|c| {
                let predicted_correct = c.confidence >= self.min_confidence_threshold;
                let actually_correct = c.validation_result.unwrap_or(false);
                predicted_correct == actually_correct
            })
            .count();

        correct_predictions as f64 / validated.len() as f64
    }

    /// Get validation coverage (percentage validated)
    pub fn validation_coverage(&self) -> f64 {
        if self.candidates.is_empty() {
            return 0.0;
        }

        let validated_count = self.candidates.values().filter(|c| c.validated).count();
        validated_count as f64 / self.candidates.len() as f64
    }

    /// Get validation history
    pub fn get_validation_history(&self) -> &[ValidationFeedback] {
        &self.validation_history
    }

    /// Set diversity weight
    pub fn set_diversity_weight(&mut self, weight: f64) {
        self.diversity_weight = weight.clamp(0.0, 1.0);
    }

    /// Set minimum confidence threshold
    pub fn set_min_confidence_threshold(&mut self, threshold: f64) {
        self.min_confidence_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Uncertainty sampling: select rule with lowest confidence
    fn uncertainty_sampling(&self, candidates: &[(String, CandidateRule)]) -> Option<String> {
        candidates
            .iter()
            .min_by(|a, b| {
                let uncertainty_a = (a.1.confidence - 0.5).abs();
                let uncertainty_b = (b.1.confidence - 0.5).abs();
                uncertainty_a
                    .partial_cmp(&uncertainty_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
    }

    /// Query-by-committee: select rule with highest disagreement
    fn query_by_committee(&mut self, candidates: &[(String, CandidateRule)]) -> Option<String> {
        // Simulate committee by adding noise to confidence scores
        let mut scores: Vec<(String, f64)> = candidates
            .iter()
            .map(|(name, candidate)| {
                let mut committee_votes = Vec::new();
                for _ in 0..self.committee_size {
                    let noise: f64 = self.rng.random::<f64>() * 0.2 - 0.1; // +/- 10%
                    let vote = (candidate.confidence + noise).clamp(0.0, 1.0);
                    committee_votes.push(vote);
                }

                // Calculate variance as disagreement
                let mean: f64 = committee_votes.iter().sum::<f64>() / committee_votes.len() as f64;
                let variance: f64 = committee_votes
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / committee_votes.len() as f64;

                (name.clone(), variance)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.first().map(|(name, _)| name.clone())
    }

    /// Diversity sampling: select most diverse rule
    fn diversity_sampling(&self, candidates: &[(String, CandidateRule)]) -> Option<String> {
        // Simple diversity based on rule structure differences
        let validated_rules: Vec<_> = self
            .candidates
            .values()
            .filter(|c| c.validated)
            .map(|c| &c.rule)
            .collect();

        candidates
            .iter()
            .max_by(|a, b| {
                let diversity_a = self.calculate_diversity(&a.1.rule, &validated_rules);
                let diversity_b = self.calculate_diversity(&b.1.rule, &validated_rules);
                diversity_a
                    .partial_cmp(&diversity_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
    }

    /// Expected model change: select rule that would change model most
    fn expected_model_change(&self, candidates: &[(String, CandidateRule)]) -> Option<String> {
        // Prioritize rules with extreme confidence (very high or very low)
        candidates
            .iter()
            .max_by(|a, b| {
                let change_a = (a.1.confidence - 0.5).abs();
                let change_b = (b.1.confidence - 0.5).abs();
                change_a
                    .partial_cmp(&change_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
    }

    /// Random sampling: baseline strategy
    fn random_sampling(&mut self, candidates: &[(String, CandidateRule)]) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        let idx: usize = self.rng.gen_range(0..candidates.len());
        Some(candidates[idx].0.clone())
    }

    /// Calculate diversity score
    fn calculate_diversity(&self, rule: &Rule, validated_rules: &[&Rule]) -> f64 {
        if validated_rules.is_empty() {
            return 1.0;
        }

        let similarities: Vec<f64> = validated_rules
            .iter()
            .map(|vr| self.rule_similarity(rule, vr))
            .collect();

        let avg_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        1.0 - avg_similarity
    }

    /// Calculate rule similarity (simple structural comparison)
    fn rule_similarity(&self, rule1: &Rule, rule2: &Rule) -> f64 {
        let body_size_diff = (rule1.body.len() as i32 - rule2.body.len() as i32).abs() as f64;
        let head_size_diff = (rule1.head.len() as i32 - rule2.head.len() as i32).abs() as f64;

        let max_size =
            (rule1.body.len() + rule1.head.len()).max(rule2.body.len() + rule2.head.len()) as f64;

        if max_size == 0.0 {
            return 1.0;
        }

        1.0 - (body_size_diff + head_size_diff) / max_size
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> ValidationStatistics {
        let total = self.candidates.len();
        let validated = self.candidates.values().filter(|c| c.validated).count();
        let correct = self
            .candidates
            .values()
            .filter(|c| c.validated && c.validation_result == Some(true))
            .count();

        let avg_confidence = if validated > 0 {
            self.candidates
                .values()
                .filter(|c| c.validated)
                .filter_map(|c| c.actual_confidence)
                .sum::<f64>()
                / validated as f64
        } else {
            0.0
        };

        ValidationStatistics {
            total_candidates: total,
            validated_count: validated,
            correct_count: correct,
            incorrect_count: validated - correct,
            average_confidence: avg_confidence,
            validation_coverage: self.validation_coverage(),
            validation_accuracy: self.validation_accuracy(),
        }
    }

    /// Clear all candidates and history
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.validation_history.clear();
    }
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total number of candidate rules
    pub total_candidates: usize,
    /// Number of validated rules
    pub validated_count: usize,
    /// Number of correct rules
    pub correct_count: usize,
    /// Number of incorrect rules
    pub incorrect_count: usize,
    /// Average validation confidence
    pub average_confidence: f64,
    /// Validation coverage (0.0 to 1.0)
    pub validation_coverage: f64,
    /// Validation accuracy (0.0 to 1.0)
    pub validation_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuleAtom, Term};

    fn create_test_rule(name: &str) -> Rule {
        Rule {
            name: name.to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }
    }

    #[test]
    fn test_active_learner_creation() {
        let learner = ActiveLearner::new();
        assert_eq!(learner.validation_coverage(), 0.0);
        assert_eq!(learner.validation_accuracy(), 0.0);
    }

    #[test]
    fn test_add_candidate_rule() {
        let mut learner = ActiveLearner::new();
        let rule = create_test_rule("rule1");
        learner.add_candidate_rule(rule, 0.75);

        assert_eq!(learner.candidates.len(), 1);
        assert_eq!(learner.validation_coverage(), 0.0);
    }

    #[test]
    fn test_uncertainty_sampling() {
        let mut learner = ActiveLearner::new();

        // Add rules with varying confidence
        learner.add_candidate_rule(create_test_rule("high_conf"), 0.95);
        learner.add_candidate_rule(create_test_rule("low_conf"), 0.15);
        learner.add_candidate_rule(create_test_rule("uncertain"), 0.52);

        let selected = learner.select_next_for_validation(SamplingStrategy::UncertaintySampling);
        assert!(selected.is_some());

        // Should select the most uncertain (closest to 0.5)
        let rule_name = selected.unwrap().name;
        assert_eq!(rule_name, "uncertain");
    }

    #[test]
    fn test_record_validation() {
        let mut learner = ActiveLearner::new();
        learner.add_candidate_rule(create_test_rule("rule1"), 0.75);

        learner.record_validation("rule1", true, 0.9);

        assert_eq!(learner.validation_coverage(), 1.0);
        assert_eq!(learner.validation_history.len(), 1);
    }

    #[test]
    fn test_validation_accuracy() {
        let mut learner = ActiveLearner::new();
        learner.set_min_confidence_threshold(0.6);

        // Add rules and validate
        learner.add_candidate_rule(create_test_rule("correct_high"), 0.8);
        learner.record_validation("correct_high", true, 0.9);

        learner.add_candidate_rule(create_test_rule("correct_low"), 0.3);
        learner.record_validation("correct_low", false, 0.2);

        let accuracy = learner.validation_accuracy();
        assert_eq!(accuracy, 1.0); // Both predictions were correct
    }

    #[test]
    fn test_validation_coverage() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.7);
        learner.add_candidate_rule(create_test_rule("rule2"), 0.8);
        learner.add_candidate_rule(create_test_rule("rule3"), 0.6);

        assert_eq!(learner.validation_coverage(), 0.0);

        learner.record_validation("rule1", true, 0.9);
        assert!((learner.validation_coverage() - 0.333).abs() < 0.01);

        learner.record_validation("rule2", true, 0.85);
        assert!((learner.validation_coverage() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_query_by_committee() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.5);
        learner.add_candidate_rule(create_test_rule("rule2"), 0.9);

        let selected = learner.select_next_for_validation(SamplingStrategy::QueryByCommittee);
        assert!(selected.is_some());
    }

    #[test]
    fn test_diversity_sampling() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.7);
        learner.add_candidate_rule(create_test_rule("rule2"), 0.8);

        // Validate one rule first
        learner.record_validation("rule1", true, 0.9);

        let selected = learner.select_next_for_validation(SamplingStrategy::DiversitySampling);
        assert!(selected.is_some());
    }

    #[test]
    fn test_random_sampling() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.7);
        learner.add_candidate_rule(create_test_rule("rule2"), 0.8);

        let selected = learner.select_next_for_validation(SamplingStrategy::RandomSampling);
        assert!(selected.is_some());
    }

    #[test]
    fn test_validation_statistics() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.8);
        learner.add_candidate_rule(create_test_rule("rule2"), 0.6);
        learner.add_candidate_rule(create_test_rule("rule3"), 0.9);

        learner.record_validation("rule1", true, 0.95);
        learner.record_validation("rule2", false, 0.3);

        let stats = learner.get_statistics();
        assert_eq!(stats.total_candidates, 3);
        assert_eq!(stats.validated_count, 2);
        assert_eq!(stats.correct_count, 1);
        assert_eq!(stats.incorrect_count, 1);
        assert!((stats.validation_coverage - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut learner = ActiveLearner::new();

        learner.add_candidate_rule(create_test_rule("rule1"), 0.7);
        learner.record_validation("rule1", true, 0.9);

        learner.clear();

        assert_eq!(learner.candidates.len(), 0);
        assert_eq!(learner.validation_history.len(), 0);
    }
}
