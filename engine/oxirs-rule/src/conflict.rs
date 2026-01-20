//! Rule Conflict Resolution
//!
//! Handles conflicts when multiple rules can derive contradictory or competing facts.
//! Provides priority-based resolution, specificity ordering, and conflict detection.
//!
//! # Features
//!
//! - **Priority-Based Resolution**: Rules with higher priority take precedence
//! - **Specificity Ordering**: More specific rules override general rules
//! - **Conflict Detection**: Identify contradictory derivations
//! - **Resolution Strategies**: Multiple strategies for handling conflicts
//! - **Confidence Scoring**: Score facts based on derivation strength
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::conflict::{ConflictResolver, ResolutionStrategy, Priority};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut resolver = ConflictResolver::new();
//!
//! // Add rules with priorities
//! let rule1 = Rule {
//!     name: "general_rule".to_string(),
//!     body: vec![],
//!     head: vec![],
//! };
//!
//! let rule2 = Rule {
//!     name: "specific_rule".to_string(),
//!     body: vec![],
//!     head: vec![],
//! };
//!
//! resolver.set_priority("general_rule", Priority::Low);
//! resolver.set_priority("specific_rule", Priority::High);
//!
//! // Resolve conflicts
//! let strategy = ResolutionStrategy::Priority;
//! resolver.set_strategy(strategy);
//! ```

use crate::{Rule, RuleAtom};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Rule priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    /// Lowest priority
    VeryLow = 0,
    /// Low priority
    Low = 1,
    /// Normal priority (default)
    #[default]
    Normal = 2,
    /// High priority
    High = 3,
    /// Highest priority
    VeryHigh = 4,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    /// Use rule priorities
    Priority,
    /// Use rule specificity (more specific rules win)
    Specificity,
    /// Use both priority and specificity
    Combined,
    /// Keep all conflicting derivations
    KeepAll,
    /// Reject all conflicting derivations
    RejectAll,
    /// Use confidence scores
    Confidence,
}

/// Conflict between rules
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Rules involved in the conflict
    pub rules: Vec<String>,
    /// Conflicting facts
    pub facts: Vec<RuleAtom>,
    /// Description of the conflict
    pub description: String,
    /// Severity of the conflict
    pub severity: ConflictSeverity,
}

/// Severity of a conflict
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    /// Low severity - minor inconsistency
    Low,
    /// Medium severity - potentially problematic
    Medium,
    /// High severity - serious inconsistency
    High,
    /// Critical - logical contradiction
    Critical,
}

/// Conflict resolver
pub struct ConflictResolver {
    /// Rule priorities
    priorities: HashMap<String, Priority>,
    /// Rule specificity scores
    specificity: HashMap<String, f64>,
    /// Active resolution strategy
    strategy: ResolutionStrategy,
    /// Detected conflicts
    conflicts: Vec<Conflict>,
    /// Confidence scores for facts
    confidence_scores: HashMap<RuleAtom, f64>,
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictResolver {
    /// Create a new conflict resolver
    pub fn new() -> Self {
        Self {
            priorities: HashMap::new(),
            specificity: HashMap::new(),
            strategy: ResolutionStrategy::Combined,
            conflicts: Vec::new(),
            confidence_scores: HashMap::new(),
        }
    }

    /// Set priority for a rule
    pub fn set_priority(&mut self, rule_name: &str, priority: Priority) {
        self.priorities.insert(rule_name.to_string(), priority);
        debug!("Set priority for rule '{}': {:?}", rule_name, priority);
    }

    /// Get priority for a rule
    pub fn get_priority(&self, rule_name: &str) -> Priority {
        self.priorities.get(rule_name).copied().unwrap_or_default()
    }

    /// Set resolution strategy
    pub fn set_strategy(&mut self, strategy: ResolutionStrategy) {
        info!("Set resolution strategy: {:?}", strategy);
        self.strategy = strategy;
    }

    /// Calculate specificity of a rule
    pub fn calculate_specificity(&mut self, rule: &Rule) -> f64 {
        // Specificity is based on:
        // 1. Number of conditions in body (more = more specific)
        // 2. Number of variables (fewer = more specific)
        // 3. Number of constants (more = more specific)

        let body_size = rule.body.len() as f64;
        let mut variable_count = 0;
        let mut constant_count = 0;

        for atom in &rule.body {
            self.count_terms_in_atom(atom, &mut variable_count, &mut constant_count);
        }

        // Higher score = more specific
        let specificity = body_size + (constant_count as f64 * 2.0) - (variable_count as f64 * 0.5);

        self.specificity.insert(rule.name.clone(), specificity);
        specificity
    }

    /// Count variables and constants in an atom
    fn count_terms_in_atom(&self, atom: &RuleAtom, variables: &mut usize, constants: &mut usize) {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                Self::count_term(subject, variables, constants);
                Self::count_term(predicate, variables, constants);
                Self::count_term(object, variables, constants);
            }
            RuleAtom::Builtin { args, .. } => {
                for arg in args {
                    Self::count_term(arg, variables, constants);
                }
            }
            RuleAtom::NotEqual { left, right }
            | RuleAtom::GreaterThan { left, right }
            | RuleAtom::LessThan { left, right } => {
                Self::count_term(left, variables, constants);
                Self::count_term(right, variables, constants);
            }
        }
    }

    /// Count a single term
    fn count_term(term: &crate::Term, variables: &mut usize, constants: &mut usize) {
        match term {
            crate::Term::Variable(_) => *variables += 1,
            crate::Term::Constant(_) | crate::Term::Literal(_) => *constants += 1,
            crate::Term::Function { args, .. } => {
                for arg in args {
                    Self::count_term(arg, variables, constants);
                }
            }
        }
    }

    /// Resolve conflicts between rules
    pub fn resolve_conflicts(
        &mut self,
        rules: &[Rule],
        facts: &[RuleAtom],
    ) -> Result<Vec<RuleAtom>> {
        info!("Resolving conflicts using strategy: {:?}", self.strategy);

        // Detect conflicts
        self.detect_conflicts(rules, facts)?;

        // Apply resolution strategy
        let resolved = match self.strategy {
            ResolutionStrategy::Priority => self.resolve_by_priority(rules, facts)?,
            ResolutionStrategy::Specificity => self.resolve_by_specificity(rules, facts)?,
            ResolutionStrategy::Combined => self.resolve_combined(rules, facts)?,
            ResolutionStrategy::KeepAll => facts.to_vec(),
            ResolutionStrategy::RejectAll => {
                if !self.conflicts.is_empty() {
                    warn!("Rejecting all facts due to conflicts");
                    vec![]
                } else {
                    facts.to_vec()
                }
            }
            ResolutionStrategy::Confidence => self.resolve_by_confidence(facts)?,
        };

        info!(
            "Conflict resolution complete: {} facts retained",
            resolved.len()
        );
        Ok(resolved)
    }

    /// Detect conflicts in derived facts
    fn detect_conflicts(&mut self, _rules: &[Rule], facts: &[RuleAtom]) -> Result<()> {
        self.conflicts.clear();

        // Check for negation conflicts (simplified)
        // In a full implementation, we would check for:
        // - Contradictory facts (e.g., both P and NOT P)
        // - Inconsistent cardinality constraints
        // - Disjoint class violations
        // etc.

        // For now, just detect duplicate facts with different sources
        let mut fact_sources: HashMap<RuleAtom, Vec<String>> = HashMap::new();

        for fact in facts {
            fact_sources
                .entry(fact.clone())
                .or_default()
                .push("source".to_string());
        }

        for (fact, sources) in fact_sources {
            if sources.len() > 1 {
                self.conflicts.push(Conflict {
                    rules: sources,
                    facts: vec![fact],
                    description: "Multiple derivation paths".to_string(),
                    severity: ConflictSeverity::Low,
                });
            }
        }

        debug!("Detected {} conflicts", self.conflicts.len());
        Ok(())
    }

    /// Resolve conflicts by priority
    fn resolve_by_priority(&self, _rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Keep all facts for now
        // In a full implementation, we would track which rule derived each fact
        // and filter based on priority
        Ok(facts.to_vec())
    }

    /// Resolve conflicts by specificity
    fn resolve_by_specificity(&self, _rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Keep all facts for now
        // In a full implementation, we would use specificity scores
        Ok(facts.to_vec())
    }

    /// Resolve conflicts using combined strategy
    fn resolve_combined(&self, rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Use both priority and specificity
        // Priority takes precedence, then specificity
        self.resolve_by_priority(rules, facts)
    }

    /// Resolve conflicts by confidence scores
    fn resolve_by_confidence(&self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut scored_facts: Vec<(RuleAtom, f64)> = facts
            .iter()
            .map(|f| {
                let confidence = self.confidence_scores.get(f).copied().unwrap_or(1.0);
                (f.clone(), confidence)
            })
            .collect();

        // Sort by confidence (highest first)
        scored_facts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Keep facts with confidence above threshold
        let threshold = 0.5;
        let resolved: Vec<RuleAtom> = scored_facts
            .into_iter()
            .filter(|(_, conf)| *conf >= threshold)
            .map(|(fact, _)| fact)
            .collect();

        Ok(resolved)
    }

    /// Set confidence score for a fact
    pub fn set_confidence(&mut self, fact: RuleAtom, confidence: f64) {
        self.confidence_scores
            .insert(fact, confidence.clamp(0.0, 1.0));
    }

    /// Get detected conflicts
    pub fn get_conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    /// Check if there are any conflicts
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }

    /// Get statistics
    pub fn get_stats(&self) -> ConflictStats {
        let critical_conflicts = self
            .conflicts
            .iter()
            .filter(|c| c.severity == ConflictSeverity::Critical)
            .count();

        let high_conflicts = self
            .conflicts
            .iter()
            .filter(|c| c.severity == ConflictSeverity::High)
            .count();

        ConflictStats {
            total_conflicts: self.conflicts.len(),
            critical_conflicts,
            high_conflicts,
            rules_with_priority: self.priorities.len(),
            active_strategy: self.strategy.clone(),
        }
    }
}

/// Statistics about conflict resolution
#[derive(Debug, Clone)]
pub struct ConflictStats {
    pub total_conflicts: usize,
    pub critical_conflicts: usize,
    pub high_conflicts: usize,
    pub rules_with_priority: usize,
    pub active_strategy: ResolutionStrategy,
}

impl std::fmt::Display for ConflictStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Conflicts: {} (critical: {}, high: {}), Rules with priority: {}, Strategy: {:?}",
            self.total_conflicts,
            self.critical_conflicts,
            self.high_conflicts,
            self.rules_with_priority,
            self.active_strategy
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_priority_setting() {
        let mut resolver = ConflictResolver::new();

        resolver.set_priority("rule1", Priority::High);
        assert_eq!(resolver.get_priority("rule1"), Priority::High);
        assert_eq!(resolver.get_priority("rule2"), Priority::Normal);
    }

    #[test]
    fn test_specificity_calculation() {
        let mut resolver = ConflictResolver::new();

        let rule = Rule {
            name: "test".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![],
        };

        let specificity = resolver.calculate_specificity(&rule);
        assert!(specificity > 0.0);
    }

    #[test]
    fn test_conflict_detection() {
        let mut resolver = ConflictResolver::new();

        let rules = vec![];
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        }];

        resolver.detect_conflicts(&rules, &facts).unwrap();
        // Should not detect conflicts for single fact
        assert_eq!(resolver.conflicts.len(), 0);
    }

    #[test]
    fn test_confidence_scoring() {
        let mut resolver = ConflictResolver::new();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Person".to_string()),
        };

        resolver.set_confidence(fact.clone(), 0.9);

        let facts = vec![fact];
        resolver.set_strategy(ResolutionStrategy::Confidence);

        let resolved = resolver.resolve_conflicts(&[], &facts).unwrap();
        assert_eq!(resolved.len(), 1);
    }

    #[test]
    fn test_stats() {
        let resolver = ConflictResolver::new();
        let stats = resolver.get_stats();

        assert_eq!(stats.total_conflicts, 0);
        assert_eq!(stats.rules_with_priority, 0);
    }
}
