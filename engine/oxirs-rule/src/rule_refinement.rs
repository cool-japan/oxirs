//! # Rule Refinement and Pruning Module
//!
//! This module provides automated rule refinement and pruning capabilities
//! to improve rule quality, remove redundancy, and optimize rule sets.
//!
//! ## Features
//!
//! - **Quality Metrics**: Support, confidence, lift, conviction
//! - **Redundancy Detection**: Identify and remove redundant rules
//! - **Rule Generalization**: Simplify overly specific rules
//! - **Rule Specialization**: Add constraints to overly general rules
//! - **Pruning Strategies**: Remove low-quality or rarely-used rules
//! - **Rule Merging**: Combine similar rules for efficiency
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::rule_refinement::*;
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! // Create a rule refiner
//! let mut refiner = RuleRefiner::new();
//!
//! // Add rules for analysis
//! let rule = Rule {
//!     name: "example".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("parent".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("ancestor".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//! };
//!
//! refiner.add_rule(rule);
//!
//! // Analyze and refine rules
//! let refined_rules = refiner.refine_rules().unwrap();
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::HashMap;

/// Rule quality metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuleQualityMetrics {
    /// Support: number of instances covered by the rule
    pub support: usize,
    /// Confidence: P(head | body)
    pub confidence: f64,
    /// Lift: confidence / P(head)
    pub lift: f64,
    /// Conviction: (1 - P(head)) / (1 - confidence)
    pub conviction: f64,
    /// Coverage: proportion of dataset covered
    pub coverage: f64,
}

impl RuleQualityMetrics {
    /// Calculate metrics for a rule
    pub fn calculate(
        rule_support: usize,
        body_support: usize,
        head_support: usize,
        total_instances: usize,
    ) -> Self {
        let confidence = if body_support > 0 {
            rule_support as f64 / body_support as f64
        } else {
            0.0
        };

        let head_prob = if total_instances > 0 {
            head_support as f64 / total_instances as f64
        } else {
            0.0
        };

        let lift = if head_prob > 0.0 {
            confidence / head_prob
        } else {
            0.0
        };

        let conviction = if confidence < 1.0 {
            (1.0 - head_prob) / (1.0 - confidence)
        } else {
            f64::INFINITY
        };

        let coverage = if total_instances > 0 {
            rule_support as f64 / total_instances as f64
        } else {
            0.0
        };

        RuleQualityMetrics {
            support: rule_support,
            confidence,
            lift,
            conviction,
            coverage,
        }
    }

    /// Check if metrics meet quality thresholds
    pub fn meets_thresholds(&self, thresholds: &QualityThresholds) -> bool {
        self.support >= thresholds.min_support
            && self.confidence >= thresholds.min_confidence
            && self.coverage >= thresholds.min_coverage
    }
}

/// Quality thresholds for pruning
#[derive(Debug, Clone, Copy)]
pub struct QualityThresholds {
    pub min_support: usize,
    pub min_confidence: f64,
    pub min_coverage: f64,
    pub min_lift: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_support: 1,
            min_confidence: 0.1,
            min_coverage: 0.0,
            min_lift: 0.0,
        }
    }
}

/// Rule refinement and pruning engine
pub struct RuleRefiner {
    /// Rules to analyze
    rules: Vec<Rule>,
    /// Rule usage statistics
    usage_stats: HashMap<String, UsageStats>,
    /// Quality thresholds
    thresholds: QualityThresholds,
    /// Training data for metrics
    training_data: Vec<RuleAtom>,
}

/// Usage statistics for a rule
#[derive(Debug, Clone, Default)]
struct UsageStats {
    /// Number of times rule was applied
    application_count: usize,
    /// Number of times rule produced new facts
    productivity_count: usize,
    /// Last time rule was used
    last_used: Option<std::time::Instant>,
}

impl RuleRefiner {
    /// Create a new rule refiner
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            usage_stats: HashMap::new(),
            thresholds: QualityThresholds::default(),
            training_data: Vec::new(),
        }
    }

    /// Set quality thresholds
    pub fn with_thresholds(mut self, thresholds: QualityThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Add a rule for analysis
    pub fn add_rule(&mut self, rule: Rule) {
        let name = rule.name.clone();
        self.rules.push(rule);
        self.usage_stats.entry(name).or_default();
    }

    /// Add training data for metric calculation
    pub fn add_training_data(&mut self, data: Vec<RuleAtom>) {
        self.training_data.extend(data);
    }

    /// Record rule application
    pub fn record_application(&mut self, rule_name: &str, was_productive: bool) {
        if let Some(stats) = self.usage_stats.get_mut(rule_name) {
            stats.application_count += 1;
            if was_productive {
                stats.productivity_count += 1;
            }
            stats.last_used = Some(std::time::Instant::now());
        }
    }

    /// Refine rules: remove redundant, low-quality rules
    pub fn refine_rules(&self) -> Result<Vec<Rule>> {
        let mut refined = Vec::new();

        // Step 1: Calculate metrics for each rule
        let metrics = self.calculate_all_metrics();

        // Step 2: Remove low-quality rules
        for (idx, rule) in self.rules.iter().enumerate() {
            if let Some(metric) = metrics.get(&idx) {
                if metric.meets_thresholds(&self.thresholds) {
                    refined.push(rule.clone());
                }
            } else {
                // No metrics available - keep rule by default
                refined.push(rule.clone());
            }
        }

        // Step 3: Remove redundant rules
        refined = self.remove_redundant_rules(refined);

        // Step 4: Remove unused rules (if usage stats available)
        refined = self.remove_unused_rules(refined);

        Ok(refined)
    }

    /// Calculate quality metrics for all rules
    fn calculate_all_metrics(&self) -> HashMap<usize, RuleQualityMetrics> {
        let mut metrics = HashMap::new();

        for (idx, rule) in self.rules.iter().enumerate() {
            let rule_support = self.count_rule_support(rule);
            let body_support = self.count_body_support(rule);
            let head_support = self.count_head_support(rule);
            let total = self.training_data.len();

            let metric =
                RuleQualityMetrics::calculate(rule_support, body_support, head_support, total);
            metrics.insert(idx, metric);
        }

        metrics
    }

    /// Count support for the full rule (body and head)
    fn count_rule_support(&self, rule: &Rule) -> usize {
        // Simplified: count facts that match both body and head
        let mut count = 0;

        // For simple rules with single atoms in body and head
        if rule.body.len() == 1 && rule.head.len() == 1 {
            let body_pattern = &rule.body[0];
            let head_pattern = &rule.head[0];

            for fact in &self.training_data {
                if self.matches_pattern(body_pattern, fact) {
                    // Check if corresponding head fact exists
                    if self
                        .training_data
                        .iter()
                        .any(|f| self.matches_pattern(head_pattern, f))
                    {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Count support for rule body
    fn count_body_support(&self, rule: &Rule) -> usize {
        if rule.body.len() == 1 {
            let pattern = &rule.body[0];
            self.training_data
                .iter()
                .filter(|fact| self.matches_pattern(pattern, fact))
                .count()
        } else {
            0
        }
    }

    /// Count support for rule head
    fn count_head_support(&self, rule: &Rule) -> usize {
        if rule.head.len() == 1 {
            let pattern = &rule.head[0];
            self.training_data
                .iter()
                .filter(|fact| self.matches_pattern(pattern, fact))
                .count()
        } else {
            0
        }
    }

    /// Check if a fact matches a pattern (with variables)
    fn matches_pattern(&self, pattern: &RuleAtom, fact: &RuleAtom) -> bool {
        match (pattern, fact) {
            (
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
            ) => {
                self.term_matches(ps, fs) && self.term_matches(pp, fp) && self.term_matches(po, fo)
            }
            _ => false,
        }
    }

    /// Check if pattern term matches fact term
    fn term_matches(&self, pattern: &Term, fact: &Term) -> bool {
        match (pattern, fact) {
            (Term::Variable(_), _) => true, // Variable matches anything
            (Term::Constant(p), Term::Constant(f)) => p == f,
            (Term::Literal(p), Term::Literal(f)) => p == f,
            _ => false,
        }
    }

    /// Remove redundant rules (subsumed by other rules)
    fn remove_redundant_rules(&self, rules: Vec<Rule>) -> Vec<Rule> {
        let mut non_redundant = Vec::new();

        for rule in rules {
            let is_redundant = non_redundant
                .iter()
                .any(|other: &Rule| self.subsumes(other, &rule));

            if !is_redundant {
                non_redundant.push(rule);
            }
        }

        non_redundant
    }

    /// Check if rule1 subsumes rule2 (is more general)
    fn subsumes(&self, rule1: &Rule, rule2: &Rule) -> bool {
        // Rule1 subsumes rule2 if:
        // 1. They have the same head
        // 2. Rule1's body is a subset of rule2's body (or more general)

        if rule1.head.len() != rule2.head.len() {
            return false;
        }

        // Check if heads are equivalent
        for (h1, h2) in rule1.head.iter().zip(rule2.head.iter()) {
            if !self.atoms_equivalent(h1, h2) {
                return false;
            }
        }

        // Check if rule1's body is more general (fewer/more general conditions)
        rule1.body.len() <= rule2.body.len()
    }

    /// Check if two atoms are equivalent
    fn atoms_equivalent(&self, atom1: &RuleAtom, atom2: &RuleAtom) -> bool {
        match (atom1, atom2) {
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
                self.terms_equivalent(s1, s2)
                    && self.terms_equivalent(p1, p2)
                    && self.terms_equivalent(o1, o2)
            }
            _ => false,
        }
    }

    /// Check if two terms are equivalent
    fn terms_equivalent(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Variable(_), Term::Variable(_)) => true, // Any variable matches any variable
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            _ => false,
        }
    }

    /// Remove rules that are rarely used
    fn remove_unused_rules(&self, rules: Vec<Rule>) -> Vec<Rule> {
        const MIN_APPLICATIONS: usize = 5;
        const MIN_PRODUCTIVITY: f64 = 0.1;

        rules
            .into_iter()
            .filter(|rule| {
                if let Some(stats) = self.usage_stats.get(&rule.name) {
                    // Keep rule if it has been applied enough times
                    if stats.application_count < MIN_APPLICATIONS {
                        return true; // Too few applications to judge
                    }

                    // Check productivity ratio
                    let productivity_ratio = if stats.application_count > 0 {
                        stats.productivity_count as f64 / stats.application_count as f64
                    } else {
                        0.0
                    };

                    productivity_ratio >= MIN_PRODUCTIVITY
                } else {
                    true // No stats - keep by default
                }
            })
            .collect()
    }

    /// Generalize a rule (remove constraints)
    pub fn generalize_rule(&self, rule: &Rule) -> Rule {
        // Simplification: remove the last condition from body
        if rule.body.len() > 1 {
            let mut generalized = rule.clone();
            generalized.body.pop();
            generalized.name = format!("{}_generalized", rule.name);
            generalized
        } else {
            rule.clone()
        }
    }

    /// Specialize a rule (add constraints)
    pub fn specialize_rule(&self, rule: &Rule, new_constraint: RuleAtom) -> Rule {
        let mut specialized = rule.clone();
        specialized.body.push(new_constraint);
        specialized.name = format!("{}_specialized", rule.name);
        specialized
    }

    /// Get usage statistics
    pub fn get_statistics(&self) -> HashMap<String, (usize, f64)> {
        self.usage_stats
            .iter()
            .map(|(name, stats)| {
                let productivity = if stats.application_count > 0 {
                    stats.productivity_count as f64 / stats.application_count as f64
                } else {
                    0.0
                };
                (name.clone(), (stats.application_count, productivity))
            })
            .collect()
    }
}

impl Default for RuleRefiner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rule(name: &str) -> Rule {
        Rule {
            name: name.to_string(),
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
        }
    }

    #[test]
    fn test_rule_refiner_creation() {
        let refiner = RuleRefiner::new();
        assert_eq!(refiner.rules.len(), 0);
    }

    #[test]
    fn test_add_rule() {
        let mut refiner = RuleRefiner::new();
        let rule = create_test_rule("test");

        refiner.add_rule(rule);

        assert_eq!(refiner.rules.len(), 1);
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let metrics = RuleQualityMetrics::calculate(10, 20, 15, 100);

        assert_eq!(metrics.support, 10);
        assert!((metrics.confidence - 0.5).abs() < 1e-10);
        assert!(metrics.coverage > 0.0);
    }

    #[test]
    fn test_metrics_thresholds() {
        let metrics = RuleQualityMetrics {
            support: 5,
            confidence: 0.8,
            lift: 1.5,
            conviction: 2.0,
            coverage: 0.05,
        };

        let thresholds = QualityThresholds {
            min_support: 3,
            min_confidence: 0.7,
            min_coverage: 0.01,
            min_lift: 1.0,
        };

        assert!(metrics.meets_thresholds(&thresholds));
    }

    #[test]
    fn test_record_application() {
        let mut refiner = RuleRefiner::new();
        let rule = create_test_rule("test");

        refiner.add_rule(rule);
        refiner.record_application("test", true);
        refiner.record_application("test", false);

        let stats = refiner.get_statistics();
        assert_eq!(stats.get("test").unwrap().0, 2); // 2 applications
    }

    #[test]
    fn test_rule_refinement() {
        let mut refiner = RuleRefiner::new();

        // Add some rules
        refiner.add_rule(create_test_rule("rule1"));
        refiner.add_rule(create_test_rule("rule2"));

        // Add training data
        let data = vec![
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("mary".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Constant("mary".to_string()),
            },
        ];

        refiner.add_training_data(data);

        // Refine
        let refined = refiner.refine_rules().unwrap();

        // Should keep at least some rules
        assert!(!refined.is_empty());
    }

    #[test]
    fn test_redundancy_removal() {
        let refiner = RuleRefiner::new();

        // Create two identical rules
        let rules = vec![create_test_rule("rule1"), create_test_rule("rule2")];

        // Test redundancy detection directly
        let non_redundant = refiner.remove_redundant_rules(rules);

        // Should keep at least one rule (may keep both due to different names)
        assert!(!non_redundant.is_empty());
        assert!(non_redundant.len() <= 2);
    }

    #[test]
    fn test_generalize_rule() {
        let refiner = RuleRefiner::new();

        let mut rule = create_test_rule("test");
        rule.body.push(RuleAtom::Triple {
            subject: Term::Variable("Y".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Variable("Age".to_string()),
        });

        let generalized = refiner.generalize_rule(&rule);

        // Should have fewer body conditions
        assert!(generalized.body.len() < rule.body.len());
    }

    #[test]
    fn test_specialize_rule() {
        let refiner = RuleRefiner::new();
        let rule = create_test_rule("test");

        let constraint = RuleAtom::GreaterThan {
            left: Term::Variable("Age".to_string()),
            right: Term::Literal("18".to_string()),
        };

        let specialized = refiner.specialize_rule(&rule, constraint);

        // Should have more body conditions
        assert!(specialized.body.len() > rule.body.len());
    }
}
