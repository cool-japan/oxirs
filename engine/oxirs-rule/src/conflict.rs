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

    /// Detect conflicts in derived facts.
    ///
    /// Two kinds of conflict are reported, each tagged with the *actual* names
    /// of the rules whose heads derive the facts (via structural head matching):
    /// - **Value conflicts**: facts that share a subject+predicate but assert
    ///   different objects (e.g. `john age 30` vs `john age 31`).
    /// - **Multiple derivation paths**: a single fact derivable by more than one
    ///   distinct rule.
    fn detect_conflicts(&mut self, rules: &[Rule], facts: &[RuleAtom]) -> Result<()> {
        self.conflicts.clear();

        // Group triple facts by (subject, predicate) to find competing objects.
        let mut groups: HashMap<(String, String), Vec<(RuleAtom, String)>> = HashMap::new();
        for fact in facts {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = fact
            {
                let key = (Self::term_key(subject), Self::term_key(predicate));
                groups
                    .entry(key)
                    .or_default()
                    .push((fact.clone(), Self::term_key(object)));
            }
        }

        for members in groups.values() {
            let distinct_objects: std::collections::HashSet<&String> =
                members.iter().map(|(_, o)| o).collect();
            if distinct_objects.len() > 1 {
                let mut involved: Vec<String> = Vec::new();
                for (fact, _) in members {
                    for name in self.deriving_rule_names(rules, fact) {
                        if !involved.contains(&name) {
                            involved.push(name);
                        }
                    }
                }
                self.conflicts.push(Conflict {
                    rules: involved,
                    facts: members.iter().map(|(f, _)| f.clone()).collect(),
                    description: "Competing values for the same subject/predicate".to_string(),
                    severity: ConflictSeverity::High,
                });
            }
        }

        // Multiple derivation paths for a single fact.
        for fact in facts {
            let names = self.deriving_rule_names(rules, fact);
            if names.len() > 1 {
                self.conflicts.push(Conflict {
                    rules: names,
                    facts: vec![fact.clone()],
                    description: "Multiple derivation paths".to_string(),
                    severity: ConflictSeverity::Low,
                });
            }
        }

        debug!("Detected {} conflicts", self.conflicts.len());
        Ok(())
    }

    /// Names of the rules whose head could have derived `fact` (structural
    /// match, treating head variables as wildcards). Distinct, order-preserving.
    fn deriving_rule_names(&self, rules: &[Rule], fact: &RuleAtom) -> Vec<String> {
        let mut names = Vec::new();
        for rule in rules {
            if rule.head.iter().any(|h| Self::head_atom_matches(h, fact))
                && !names.contains(&rule.name)
            {
                names.push(rule.name.clone());
            }
        }
        names
    }

    /// Structural match of a rule head atom against a (ground) fact.
    fn head_atom_matches(head: &RuleAtom, fact: &RuleAtom) -> bool {
        match (head, fact) {
            (
                RuleAtom::Triple {
                    subject: hs,
                    predicate: hp,
                    object: ho,
                },
                RuleAtom::Triple {
                    subject: fs,
                    predicate: fp,
                    object: fo,
                },
            ) => {
                Self::term_matches(hs, fs)
                    && Self::term_matches(hp, fp)
                    && Self::term_matches(ho, fo)
            }
            (
                RuleAtom::NotEqual {
                    left: hl,
                    right: hr,
                },
                RuleAtom::NotEqual {
                    left: fl,
                    right: fr,
                },
            )
            | (
                RuleAtom::GreaterThan {
                    left: hl,
                    right: hr,
                },
                RuleAtom::GreaterThan {
                    left: fl,
                    right: fr,
                },
            )
            | (
                RuleAtom::LessThan {
                    left: hl,
                    right: hr,
                },
                RuleAtom::LessThan {
                    left: fl,
                    right: fr,
                },
            ) => Self::term_matches(hl, fl) && Self::term_matches(hr, fr),
            _ => false,
        }
    }

    fn term_matches(pattern: &crate::Term, value: &crate::Term) -> bool {
        use crate::Term;
        match (pattern, value) {
            (Term::Variable(_), _) => true,
            (Term::Constant(a), Term::Constant(b)) => a == b,
            (Term::Literal(a), Term::Literal(b)) => a == b,
            (Term::Constant(a), Term::Literal(b)) | (Term::Literal(a), Term::Constant(b)) => a == b,
            _ => false,
        }
    }

    fn term_key(term: &crate::Term) -> String {
        use crate::Term;
        match term {
            Term::Variable(v) => format!("?{v}"),
            Term::Constant(c) => c.clone(),
            Term::Literal(l) => format!("\"{l}\""),
            Term::Function { name, args } => {
                let inner: Vec<String> = args.iter().map(Self::term_key).collect();
                format!("{name}({})", inner.join(","))
            }
        }
    }

    /// Static specificity score (mirrors `calculate_specificity` without
    /// mutating the resolver's cache), for use on the read-only resolve path.
    fn rule_specificity_score(rule: &Rule) -> f64 {
        let body_size = rule.body.len() as f64;
        let mut variables = 0usize;
        let mut constants = 0usize;
        for atom in &rule.body {
            match atom {
                RuleAtom::Triple {
                    subject,
                    predicate,
                    object,
                } => {
                    Self::count_term(subject, &mut variables, &mut constants);
                    Self::count_term(predicate, &mut variables, &mut constants);
                    Self::count_term(object, &mut variables, &mut constants);
                }
                RuleAtom::Builtin { args, .. } => {
                    for arg in args {
                        Self::count_term(arg, &mut variables, &mut constants);
                    }
                }
                RuleAtom::NotEqual { left, right }
                | RuleAtom::GreaterThan { left, right }
                | RuleAtom::LessThan { left, right } => {
                    Self::count_term(left, &mut variables, &mut constants);
                    Self::count_term(right, &mut variables, &mut constants);
                }
            }
        }
        body_size + (constants as f64 * 2.0) - (variables as f64 * 0.5)
    }

    /// Score of a fact under a per-rule metric: the best (max) metric over the
    /// rules that could derive it. Facts with no deriving rule are treated as
    /// asserted ground truth (`+inf`) and always survive resolution.
    fn fact_score<F: Fn(&Rule) -> f64>(&self, rules: &[Rule], fact: &RuleAtom, metric: &F) -> f64 {
        let mut best = f64::NEG_INFINITY;
        let mut any = false;
        for rule in rules {
            if rule.head.iter().any(|h| Self::head_atom_matches(h, fact)) {
                any = true;
                best = best.max(metric(rule));
            }
        }
        if any {
            best
        } else {
            f64::INFINITY
        }
    }

    /// Generic conflict resolution: within each (subject, predicate) group keep
    /// only the facts whose deriving-rule score equals the group maximum; facts
    /// with a unique subject/predicate (or non-triple facts) are always kept.
    fn resolve_by_metric<F: Fn(&Rule) -> f64>(
        &self,
        rules: &[Rule],
        facts: &[RuleAtom],
        metric: F,
    ) -> Vec<RuleAtom> {
        // Precompute each fact's score once.
        let scores: Vec<f64> = facts
            .iter()
            .map(|f| self.fact_score(rules, f, &metric))
            .collect();

        // Group maxima keyed by (subject, predicate).
        let mut group_max: HashMap<(String, String), f64> = HashMap::new();
        for (idx, fact) in facts.iter().enumerate() {
            if let RuleAtom::Triple {
                subject, predicate, ..
            } = fact
            {
                let key = (Self::term_key(subject), Self::term_key(predicate));
                let entry = group_max.entry(key).or_insert(f64::NEG_INFINITY);
                *entry = entry.max(scores[idx]);
            }
        }

        let mut resolved = Vec::new();
        for (idx, fact) in facts.iter().enumerate() {
            let keep = match fact {
                RuleAtom::Triple {
                    subject, predicate, ..
                } => {
                    let key = (Self::term_key(subject), Self::term_key(predicate));
                    let max = group_max.get(&key).copied().unwrap_or(scores[idx]);
                    // Keep facts at the group maximum (handles +inf asserted facts
                    // and ties). Epsilon guards float comparison of finite scores.
                    scores[idx] == max || (scores[idx] - max).abs() < f64::EPSILON
                }
                // Non-triple facts have no subject/predicate grouping; keep them.
                _ => true,
            };
            if keep {
                resolved.push(fact.clone());
            }
        }
        resolved
    }

    /// Resolve conflicts by rule priority: among facts competing for the same
    /// subject/predicate, keep those derived by the highest-priority rule.
    fn resolve_by_priority(&self, rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        Ok(self.resolve_by_metric(rules, facts, |rule| {
            self.get_priority(&rule.name) as u8 as f64
        }))
    }

    /// Resolve conflicts by rule specificity: more specific rules win.
    fn resolve_by_specificity(&self, rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        Ok(self.resolve_by_metric(rules, facts, Self::rule_specificity_score))
    }

    /// Resolve conflicts using a combined metric: priority dominates, ties are
    /// broken by specificity.
    fn resolve_combined(&self, rules: &[Rule], facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        Ok(self.resolve_by_metric(rules, facts, |rule| {
            (self.get_priority(&rule.name) as u8 as f64) * 1_000_000.0
                + Self::rule_specificity_score(rule)
        }))
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
    fn test_conflict_detection() -> Result<(), Box<dyn std::error::Error>> {
        let mut resolver = ConflictResolver::new();

        let rules = vec![];
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        }];

        resolver.detect_conflicts(&rules, &facts)?;
        // Should not detect conflicts for single fact
        assert_eq!(resolver.conflicts.len(), 0);
        Ok(())
    }

    #[test]
    fn test_confidence_scoring() -> Result<(), Box<dyn std::error::Error>> {
        let mut resolver = ConflictResolver::new();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Person".to_string()),
        };

        resolver.set_confidence(fact.clone(), 0.9);

        let facts = vec![fact];
        resolver.set_strategy(ResolutionStrategy::Confidence);

        let resolved = resolver.resolve_conflicts(&[], &facts)?;
        assert_eq!(resolved.len(), 1);
        Ok(())
    }

    #[test]
    fn test_stats() {
        let resolver = ConflictResolver::new();
        let stats = resolver.get_stats();

        assert_eq!(stats.total_conflicts, 0);
        assert_eq!(stats.rules_with_priority, 0);
    }

    /// Regression: the Priority strategy must actually drop the lower-priority
    /// competing fact (previously it returned every fact unchanged, identical to
    /// KeepAll), and detect_conflicts must record real rule names.
    #[test]
    fn regression_priority_resolution_drops_competing_fact(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut resolver = ConflictResolver::new();

        let high_fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        };
        let low_fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("25".to_string()),
        };

        let rules = vec![
            Rule {
                name: "specific".to_string(),
                body: vec![],
                head: vec![high_fact.clone()],
            },
            Rule {
                name: "general".to_string(),
                body: vec![],
                head: vec![low_fact.clone()],
            },
        ];

        resolver.set_priority("specific", Priority::VeryHigh);
        resolver.set_priority("general", Priority::Low);
        resolver.set_strategy(ResolutionStrategy::Priority);

        let resolved =
            resolver.resolve_conflicts(&rules, &[high_fact.clone(), low_fact.clone()])?;

        assert!(
            resolved.contains(&high_fact),
            "high-priority fact must survive: {resolved:?}"
        );
        assert!(
            !resolved.contains(&low_fact),
            "low-priority competing fact must be dropped: {resolved:?}"
        );

        // A genuine value conflict was detected, tagged with real rule names.
        assert!(resolver.has_conflicts());
        let named: Vec<String> = resolver
            .get_conflicts()
            .iter()
            .flat_map(|c| c.rules.clone())
            .collect();
        assert!(named.contains(&"specific".to_string()));
        assert!(named.contains(&"general".to_string()));
        assert!(!named.contains(&"source".to_string()));
        Ok(())
    }
}
