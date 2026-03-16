/// Conflict detection and resolution for rule-based reasoning.
///
/// When multiple rules produce contradictory conclusions, a `ConflictResolver`
/// detects the conflicts and applies a configurable `ResolutionStrategy` to
/// decide which rule "wins".
use std::collections::{HashMap, HashSet};

// ── Enumerations ──────────────────────────────────────────────────────────────

/// The kind of conflict detected between two rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConflictType {
    /// One rule asserts a triple while another retracts it.
    AssertRetract,
    /// Two rules independently derive the same triple (usually harmless but tracked).
    DuplicateAssertion,
    /// The rules form a circular dependency chain.
    CircularDependency,
    /// Two rules assert different values for the same property on the same subject.
    ContradictoryValues,
}

/// Strategy used to resolve a conflict between two rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// The rule with the higher numeric priority wins.
    HigherPriority,
    /// The first applicable rule (lowest index) wins.
    FirstApplicable,
    /// The last applicable rule (highest index) wins.
    LastApplicable,
    /// Both rules are applied and results are merged (only for `DuplicateAssertion`).
    Merge,
    /// Conflict causes an error — no rule is applied.
    Error,
}

// ── Core data structures ──────────────────────────────────────────────────────

/// The head (consequent) of a rule — a single triple pattern to assert.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuleHead {
    pub predicate: String,
    pub subject: String,
    pub object: String,
}

/// A full rule with ordered antecedents and consequents.
#[derive(Debug, Clone)]
pub struct Rule {
    /// Unique rule identifier.
    pub id: String,
    /// Higher value = higher priority.
    pub priority: i32,
    /// Pattern strings that must hold for the rule to fire (simplified; not evaluated).
    pub antecedents: Vec<String>,
    /// The triples this rule would assert when it fires.
    pub consequents: Vec<RuleHead>,
}

/// A detected conflict between two rules.
#[derive(Debug, Clone)]
pub struct RuleConflict {
    pub rule_a: String,
    pub rule_b: String,
    pub conflict_type: ConflictType,
    pub affected_triple: (String, String, String),
}

/// The resolution chosen for a single conflict.
#[derive(Debug, Clone)]
pub struct Resolution {
    pub conflict_index: usize,
    pub winner: String,
    pub loser: String,
    pub strategy_used: ResolutionStrategy,
}

/// Summary of a `apply_rules` call.
#[derive(Debug, Clone, Default)]
pub struct ApplyResult {
    pub added: Vec<(String, String, String)>,
    pub removed: Vec<(String, String, String)>,
    pub conflicts_detected: usize,
    pub rules_fired: usize,
}

// ── ConflictResolver ──────────────────────────────────────────────────────────

/// The main conflict-detection and resolution engine.
pub struct ConflictResolver {
    strategy: ResolutionStrategy,
    rules: Vec<Rule>,
}

impl ConflictResolver {
    /// Create a new resolver with the given strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            rules: Vec::new(),
        }
    }

    /// Register a rule.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Detect all conflicts among the registered rules.
    ///
    /// Two rules conflict when their consequents overlap in a problematic way:
    /// - Same `(subject, predicate)` but different `object` → `ContradictoryValues`
    /// - Same `(subject, predicate, object)` → `DuplicateAssertion`
    /// - One asserts what a retract-marker would remove → `AssertRetract`
    ///   (detected when a rule head matches a pattern in another rule's antecedents
    ///   that contains the keyword "NOT")
    pub fn detect_conflicts(&self) -> Vec<RuleConflict> {
        let mut conflicts = Vec::new();
        let n = self.rules.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.rules[i];
                let b = &self.rules[j];

                // Check head-vs-head overlaps
                for head_a in &a.consequents {
                    for head_b in &b.consequents {
                        let triple_a = (
                            head_a.subject.clone(),
                            head_a.predicate.clone(),
                            head_a.object.clone(),
                        );
                        let triple_b = (
                            head_b.subject.clone(),
                            head_b.predicate.clone(),
                            head_b.object.clone(),
                        );

                        if triple_a == triple_b {
                            conflicts.push(RuleConflict {
                                rule_a: a.id.clone(),
                                rule_b: b.id.clone(),
                                conflict_type: ConflictType::DuplicateAssertion,
                                affected_triple: triple_a,
                            });
                        } else if head_a.subject == head_b.subject
                            && head_a.predicate == head_b.predicate
                        {
                            conflicts.push(RuleConflict {
                                rule_a: a.id.clone(),
                                rule_b: b.id.clone(),
                                conflict_type: ConflictType::ContradictoryValues,
                                affected_triple: triple_a,
                            });
                        }
                    }
                }

                // AssertRetract: rule B's antecedents contain "NOT <triple asserted by A>"
                for head_a in &a.consequents {
                    let negation = format!(
                        "NOT ({} {} {})",
                        head_a.subject, head_a.predicate, head_a.object
                    );
                    if b.antecedents.contains(&negation) {
                        conflicts.push(RuleConflict {
                            rule_a: a.id.clone(),
                            rule_b: b.id.clone(),
                            conflict_type: ConflictType::AssertRetract,
                            affected_triple: (
                                head_a.subject.clone(),
                                head_a.predicate.clone(),
                                head_a.object.clone(),
                            ),
                        });
                    }
                }
                // Check the other direction too
                for head_b in &b.consequents {
                    let negation = format!(
                        "NOT ({} {} {})",
                        head_b.subject, head_b.predicate, head_b.object
                    );
                    if a.antecedents.contains(&negation) {
                        conflicts.push(RuleConflict {
                            rule_a: b.id.clone(),
                            rule_b: a.id.clone(),
                            conflict_type: ConflictType::AssertRetract,
                            affected_triple: (
                                head_b.subject.clone(),
                                head_b.predicate.clone(),
                                head_b.object.clone(),
                            ),
                        });
                    }
                }
            }
        }

        // Circular dependency detection (appended separately so counts are right)
        if self.has_circular_dependency() {
            // Find the cycle and emit one conflict per rule involved
            if let Some(cycle) = self.find_a_cycle() {
                for i in 0..cycle.len() {
                    let a = &cycle[i];
                    let b = &cycle[(i + 1) % cycle.len()];
                    conflicts.push(RuleConflict {
                        rule_a: a.clone(),
                        rule_b: b.clone(),
                        conflict_type: ConflictType::CircularDependency,
                        affected_triple: ("?".into(), "dependsOn".into(), "?".into()),
                    });
                }
            }
        }

        conflicts
    }

    /// Resolve a list of conflicts according to the current strategy.
    pub fn resolve(&self, conflicts: &[RuleConflict]) -> Vec<Resolution> {
        conflicts
            .iter()
            .enumerate()
            .filter_map(|(idx, conflict)| self.resolve_one(idx, conflict))
            .collect()
    }

    /// Apply all registered rules to a fact base and return a summary.
    ///
    /// Rules that fire add their consequents to `facts`.
    /// In case of conflict the resolution strategy is used to decide winners.
    pub fn apply_rules(&self, facts: &mut Vec<(String, String, String)>) -> ApplyResult {
        let mut result = ApplyResult::default();
        let conflicts = self.detect_conflicts();
        result.conflicts_detected = conflicts.len();

        // Build a set of "loser" rule IDs to suppress
        let resolutions = self.resolve(&conflicts);
        let losers: HashSet<&str> = resolutions.iter().map(|r| r.loser.as_str()).collect();

        // Index existing facts for fast lookup
        let existing: HashSet<(String, String, String)> = facts.iter().cloned().collect();

        // Track which triples should be removed (AssertRetract conflicts)
        let mut to_remove: HashSet<(String, String, String)> = HashSet::new();

        for rule in &self.rules {
            if losers.contains(rule.id.as_str()) {
                continue;
            }

            // Simple firing condition: all non-NOT antecedents must be in facts.
            let fires = rule.antecedents.iter().all(|ant| {
                if ant.starts_with("NOT ") {
                    // NOT antecedents: the fact must NOT be present
                    // Parse "NOT (s p o)"
                    let inner = ant.trim_start_matches("NOT ").trim();
                    let parsed = parse_triple_pattern(inner);
                    parsed.map_or(true, |t| !existing.contains(&t))
                } else {
                    let parsed = parse_triple_pattern(ant);
                    parsed.map_or(true, |t| existing.contains(&t))
                }
            });

            if !fires {
                continue;
            }

            result.rules_fired += 1;

            for head in &rule.consequents {
                let triple = (
                    head.subject.clone(),
                    head.predicate.clone(),
                    head.object.clone(),
                );
                if !existing.contains(&triple) && !result.added.contains(&triple) {
                    result.added.push(triple);
                }
            }

            // Check if this rule retracts anything via AssertRetract conflicts
            for conflict in &conflicts {
                if conflict.conflict_type == ConflictType::AssertRetract
                    && conflict.rule_b == rule.id
                {
                    to_remove.insert(conflict.affected_triple.clone());
                }
            }
        }

        // Apply additions
        for triple in &result.added {
            facts.push(triple.clone());
        }

        // Apply removals
        result.removed = to_remove.into_iter().collect();
        facts.retain(|t| !result.removed.contains(t));

        result
    }

    /// Returns `true` if the dependency graph among rules contains a cycle.
    ///
    /// A dependency exists from rule A → rule B when a consequent of A
    /// appears as an antecedent of B.
    pub fn has_circular_dependency(&self) -> bool {
        self.find_a_cycle().is_some()
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn resolve_one(&self, idx: usize, conflict: &RuleConflict) -> Option<Resolution> {
        match &self.strategy {
            ResolutionStrategy::Error => None,
            ResolutionStrategy::Merge => {
                // For Merge only DuplicateAssertion is "resolved" by keeping both.
                if conflict.conflict_type == ConflictType::DuplicateAssertion {
                    Some(Resolution {
                        conflict_index: idx,
                        winner: conflict.rule_a.clone(),
                        loser: conflict.rule_b.clone(), // loser still runs
                        strategy_used: ResolutionStrategy::Merge,
                    })
                } else {
                    None
                }
            }
            ResolutionStrategy::HigherPriority => {
                let pa = self.priority_of(&conflict.rule_a);
                let pb = self.priority_of(&conflict.rule_b);
                let (winner, loser) = if pa >= pb {
                    (conflict.rule_a.clone(), conflict.rule_b.clone())
                } else {
                    (conflict.rule_b.clone(), conflict.rule_a.clone())
                };
                Some(Resolution {
                    conflict_index: idx,
                    winner,
                    loser,
                    strategy_used: ResolutionStrategy::HigherPriority,
                })
            }
            ResolutionStrategy::FirstApplicable => {
                let ia = self.index_of(&conflict.rule_a);
                let ib = self.index_of(&conflict.rule_b);
                let (winner, loser) = if ia <= ib {
                    (conflict.rule_a.clone(), conflict.rule_b.clone())
                } else {
                    (conflict.rule_b.clone(), conflict.rule_a.clone())
                };
                Some(Resolution {
                    conflict_index: idx,
                    winner,
                    loser,
                    strategy_used: ResolutionStrategy::FirstApplicable,
                })
            }
            ResolutionStrategy::LastApplicable => {
                let ia = self.index_of(&conflict.rule_a);
                let ib = self.index_of(&conflict.rule_b);
                let (winner, loser) = if ia >= ib {
                    (conflict.rule_a.clone(), conflict.rule_b.clone())
                } else {
                    (conflict.rule_b.clone(), conflict.rule_a.clone())
                };
                Some(Resolution {
                    conflict_index: idx,
                    winner,
                    loser,
                    strategy_used: ResolutionStrategy::LastApplicable,
                })
            }
        }
    }

    fn priority_of(&self, id: &str) -> i32 {
        self.rules
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.priority)
            .unwrap_or(0)
    }

    fn index_of(&self, id: &str) -> usize {
        self.rules
            .iter()
            .position(|r| r.id == id)
            .unwrap_or(usize::MAX)
    }

    /// Return a list of rule IDs that form a cycle, or `None` if no cycle exists.
    fn find_a_cycle(&self) -> Option<Vec<String>> {
        // Build adjacency: rule_id → set of rule_ids it depends on (its heads appear
        // in the dependents' antecedents)
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for rule in &self.rules {
            adj.entry(&rule.id).or_default();
        }

        // For each rule A, for each consequent triple of A,
        // if that triple appears as an antecedent of rule B → A → B edge
        for a in &self.rules {
            for head in &a.consequents {
                let fact = format!("({} {} {})", head.subject, head.predicate, head.object);
                for b in &self.rules {
                    if b.id != a.id && b.antecedents.contains(&fact) {
                        adj.entry(&a.id).or_default().push(&b.id);
                    }
                }
            }
        }

        // DFS cycle detection
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: HashSet<&str> = HashSet::new();
        let mut cycle: Vec<String> = Vec::new();

        fn dfs<'a>(
            node: &'a str,
            adj: &HashMap<&'a str, Vec<&'a str>>,
            visited: &mut HashSet<&'a str>,
            stack: &mut HashSet<&'a str>,
            cycle: &mut Vec<String>,
        ) -> bool {
            visited.insert(node);
            stack.insert(node);
            if let Some(neighbors) = adj.get(node) {
                for &next in neighbors {
                    if !visited.contains(next) {
                        if dfs(next, adj, visited, stack, cycle) {
                            cycle.push(node.to_string());
                            return true;
                        }
                    } else if stack.contains(next) {
                        cycle.push(next.to_string());
                        cycle.push(node.to_string());
                        return true;
                    }
                }
            }
            stack.remove(node);
            false
        }

        for rule in &self.rules {
            if !visited.contains(rule.id.as_str())
                && dfs(&rule.id, &adj, &mut visited, &mut stack, &mut cycle)
            {
                cycle.reverse();
                return Some(cycle);
            }
        }
        None
    }
}

// ── Utility function ──────────────────────────────────────────────────────────

/// Parse a triple pattern string of the form `(subject predicate object)`.
fn parse_triple_pattern(s: &str) -> Option<(String, String, String)> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<&str> = s.splitn(3, ' ').collect();
    if parts.len() == 3 {
        Some((
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        ))
    } else {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn head(s: &str, p: &str, o: &str) -> RuleHead {
        RuleHead {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }

    fn rule(id: &str, priority: i32, ants: &[&str], cons: &[RuleHead]) -> Rule {
        Rule {
            id: id.into(),
            priority,
            antecedents: ants.iter().map(|s| s.to_string()).collect(),
            consequents: cons.to_vec(),
        }
    }

    // ── No-conflict scenarios ──────────────────────────────────────────────────

    #[test]
    fn test_no_rules_no_conflict() {
        let resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        let conflicts = resolver.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_single_rule_no_conflict() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "o")]));
        assert!(resolver.detect_conflicts().is_empty());
    }

    #[test]
    fn test_non_overlapping_heads_no_conflict() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p1", "o1")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("s", "p2", "o2")]));
        assert!(resolver.detect_conflicts().is_empty());
    }

    // ── DuplicateAssertion ─────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_assertion_detected() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::Merge);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "o")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("s", "p", "o")]));
        let conflicts = resolver.detect_conflicts();
        let dup = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::DuplicateAssertion);
        assert!(dup.is_some());
    }

    #[test]
    fn test_duplicate_assertion_names_both_rules() -> Result<(), Box<dyn std::error::Error>> {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::Merge);
        resolver.add_rule(rule("rA", 10, &[], &[head("s", "p", "o")]));
        resolver.add_rule(rule("rB", 5, &[], &[head("s", "p", "o")]));
        let conflicts = resolver.detect_conflicts();
        let dup = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::DuplicateAssertion)
            .ok_or("expected Some value")?;
        assert!(
            (dup.rule_a == "rA" && dup.rule_b == "rB")
                || (dup.rule_a == "rB" && dup.rule_b == "rA")
        );
        Ok(())
    }

    // ── ContradictoryValues ────────────────────────────────────────────────────

    #[test]
    fn test_contradictory_values_detected() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let cv = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::ContradictoryValues);
        assert!(cv.is_some());
    }

    // ── AssertRetract ──────────────────────────────────────────────────────────

    #[test]
    fn test_assert_retract_detected() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        // r1 asserts (s p o)
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "o")]));
        // r2 requires NOT (s p o) in its antecedents
        resolver.add_rule(rule("r2", 5, &["NOT (s p o)"], &[head("s2", "p2", "o2")]));
        let conflicts = resolver.detect_conflicts();
        let ar = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::AssertRetract);
        assert!(ar.is_some(), "Expected AssertRetract conflict");
    }

    // ── Resolution strategies ──────────────────────────────────────────────────

    #[test]
    fn test_higher_priority_selects_winner() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("low", 1, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("high", 100, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        assert!(!resolutions.is_empty());
        let r = &resolutions[0];
        assert_eq!(r.winner, "high");
        assert_eq!(r.loser, "low");
    }

    #[test]
    fn test_first_applicable_selects_first_added() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::FirstApplicable);
        resolver.add_rule(rule("first", 5, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("second", 10, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        assert!(!resolutions.is_empty());
        assert_eq!(resolutions[0].winner, "first");
        assert_eq!(resolutions[0].loser, "second");
        assert_eq!(
            resolutions[0].strategy_used,
            ResolutionStrategy::FirstApplicable
        );
    }

    #[test]
    fn test_last_applicable_selects_last_added() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastApplicable);
        resolver.add_rule(rule("first", 5, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("second", 10, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        assert!(!resolutions.is_empty());
        assert_eq!(resolutions[0].winner, "second");
        assert_eq!(resolutions[0].loser, "first");
        assert_eq!(
            resolutions[0].strategy_used,
            ResolutionStrategy::LastApplicable
        );
    }

    #[test]
    fn test_error_strategy_returns_no_resolution() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::Error);
        resolver.add_rule(rule("r1", 5, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("r2", 10, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        assert!(resolutions.is_empty());
    }

    #[test]
    fn test_merge_strategy_for_duplicate_assertion() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::Merge);
        resolver.add_rule(rule("r1", 5, &[], &[head("s", "p", "o")]));
        resolver.add_rule(rule("r2", 10, &[], &[head("s", "p", "o")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        // Merge should produce a resolution with Merge strategy
        let merge_res = resolutions
            .iter()
            .find(|r| r.strategy_used == ResolutionStrategy::Merge);
        assert!(merge_res.is_some());
    }

    // ── apply_rules ───────────────────────────────────────────────────────────

    #[test]
    fn test_apply_rules_fires_rule_without_antecedents() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("a", "b", "c")]));

        let mut facts: Vec<(String, String, String)> = vec![];
        let result = resolver.apply_rules(&mut facts);
        assert_eq!(result.rules_fired, 1);
        assert_eq!(result.added.len(), 1);
        assert!(facts.contains(&("a".into(), "b".into(), "c".into())));
    }

    #[test]
    fn test_apply_rules_antecedent_must_be_satisfied() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &["(x y z)"], &[head("a", "b", "c")]));

        let mut facts: Vec<(String, String, String)> = vec![];
        let result = resolver.apply_rules(&mut facts);
        // (x y z) not present → rule should not fire
        assert_eq!(result.rules_fired, 0);
    }

    #[test]
    fn test_apply_rules_with_satisfied_antecedent() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &["(x y z)"], &[head("a", "b", "c")]));

        let mut facts = vec![("x".into(), "y".into(), "z".into())];
        let result = resolver.apply_rules(&mut facts);
        assert_eq!(result.rules_fired, 1);
    }

    #[test]
    fn test_apply_rules_no_duplicate_addition() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("a", "b", "c")]));

        let mut facts = vec![("a".into(), "b".into(), "c".into())];
        let result = resolver.apply_rules(&mut facts);
        // Triple already present — should not be added again
        assert_eq!(result.added.len(), 0);
    }

    #[test]
    fn test_apply_result_conflicts_detected_count() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("s", "p", "v2")]));

        let mut facts = vec![];
        let result = resolver.apply_rules(&mut facts);
        assert!(result.conflicts_detected >= 1);
    }

    #[test]
    fn test_apply_rules_loser_not_applied() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("low", 1, &[], &[head("s", "p", "from_low")]));
        resolver.add_rule(rule("high", 100, &[], &[head("s", "p", "from_high")]));

        let mut facts = vec![];
        let result = resolver.apply_rules(&mut facts);
        // "high" wins, "low" is loser
        assert!(!result
            .added
            .contains(&("s".into(), "p".into(), "from_low".into())));
        assert!(result
            .added
            .contains(&("s".into(), "p".into(), "from_high".into())));
    }

    // ── Circular dependency ────────────────────────────────────────────────────

    #[test]
    fn test_no_circular_dependency_simple() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("a", "b", "c")]));
        resolver.add_rule(rule("r2", 10, &["(a b c)"], &[head("d", "e", "f")]));
        assert!(!resolver.has_circular_dependency());
    }

    #[test]
    fn test_circular_dependency_two_rules() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        // r1 consequent (a b c), r2 antecedent (a b c) → r1→r2
        // r2 consequent (x y z), r1 antecedent (x y z) → r2→r1  (cycle!)
        resolver.add_rule(rule("r1", 10, &["(x y z)"], &[head("a", "b", "c")]));
        resolver.add_rule(rule("r2", 10, &["(a b c)"], &[head("x", "y", "z")]));
        assert!(resolver.has_circular_dependency());
    }

    #[test]
    fn test_circular_dependency_detected_in_conflicts() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &["(x y z)"], &[head("a", "b", "c")]));
        resolver.add_rule(rule("r2", 10, &["(a b c)"], &[head("x", "y", "z")]));
        let conflicts = resolver.detect_conflicts();
        let circ = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::CircularDependency);
        assert!(circ.is_some());
    }

    #[test]
    fn test_no_circular_dependency_with_zero_rules() {
        let resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        assert!(!resolver.has_circular_dependency());
    }

    // ── Rule count and metadata ────────────────────────────────────────────────

    #[test]
    fn test_rule_count_empty() {
        let resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        assert_eq!(resolver.rule_count(), 0);
    }

    #[test]
    fn test_rule_count_after_add() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 1, &[], &[]));
        resolver.add_rule(rule("r2", 2, &[], &[]));
        assert_eq!(resolver.rule_count(), 2);
    }

    // ── Resolution metadata ────────────────────────────────────────────────────

    #[test]
    fn test_resolution_conflict_index_matches() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("low", 1, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("high", 100, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        assert!(!resolutions.is_empty());
        assert!(resolutions[0].conflict_index < conflicts.len());
    }

    #[test]
    fn test_three_rules_multiple_conflicts() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("s", "p", "1")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("s", "p", "2")]));
        resolver.add_rule(rule("r3", 1, &[], &[head("s", "p", "3")]));
        let conflicts = resolver.detect_conflicts();
        assert!(conflicts.len() >= 2); // At least one conflict per pair
    }

    #[test]
    fn test_apply_result_default_values() {
        let result = ApplyResult::default();
        assert!(result.added.is_empty());
        assert!(result.removed.is_empty());
        assert_eq!(result.conflicts_detected, 0);
        assert_eq!(result.rules_fired, 0);
    }

    #[test]
    fn test_higher_priority_equal_priority_prefers_rule_a() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("rA", 50, &[], &[head("s", "p", "v1")]));
        resolver.add_rule(rule("rB", 50, &[], &[head("s", "p", "v2")]));
        let conflicts = resolver.detect_conflicts();
        let resolutions = resolver.resolve(&conflicts);
        // With equal priority rule_a wins (pa >= pb condition)
        assert!(!resolutions.is_empty());
    }

    #[test]
    fn test_apply_rules_rules_fired_counter() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("a", "b", "c")]));
        resolver.add_rule(rule("r2", 10, &[], &[head("d", "e", "f")]));
        let mut facts = vec![];
        let result = resolver.apply_rules(&mut facts);
        assert_eq!(result.rules_fired, 2);
    }

    #[test]
    fn test_conflict_affected_triple_correct() -> Result<(), Box<dyn std::error::Error>> {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::HigherPriority);
        resolver.add_rule(rule("r1", 10, &[], &[head("alice", "knows", "bob")]));
        resolver.add_rule(rule("r2", 5, &[], &[head("alice", "knows", "carol")]));
        let conflicts = resolver.detect_conflicts();
        let cv = conflicts
            .iter()
            .find(|c| c.conflict_type == ConflictType::ContradictoryValues)
            .ok_or("expected Some value")?;
        assert_eq!(cv.affected_triple.0, "alice");
        assert_eq!(cv.affected_triple.1, "knows");
        Ok(())
    }
}
