//! Rule priority/salience ordering and conflict resolution.
//!
//! Provides mechanisms to order and select rules from a conflict set
//! according to configurable selection strategies.

/// A rule with priority/salience metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrioritizedRule {
    /// Unique rule identifier.
    pub id: String,
    /// Salience value — higher value means higher priority.
    pub salience: i32,
    /// Rule group name.
    pub group: String,
    /// Optional mutex group — at most one rule per mutex group fires.
    pub mutex_group: Option<String>,
}

impl PrioritizedRule {
    /// Construct a new prioritised rule.
    pub fn new(
        id: impl Into<String>,
        salience: i32,
        group: impl Into<String>,
        mutex_group: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            salience,
            group: group.into(),
            mutex_group,
        }
    }
}

/// A conflict set holding eligible rules.
#[derive(Debug, Default, Clone)]
pub struct ConflictSet {
    /// Candidate rules.
    pub rules: Vec<PrioritizedRule>,
    /// Internal round-robin cursor.
    cursor: usize,
}

impl ConflictSet {
    /// Create an empty conflict set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the set.
    pub fn add(&mut self, rule: PrioritizedRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by id.  Returns `true` if a rule was removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != id);
        // Reset cursor if it would be out of bounds.
        if self.cursor >= self.rules.len() && !self.rules.is_empty() {
            self.cursor = 0;
        }
        self.rules.len() < before
    }

    /// Number of rules in the set.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// `true` if the conflict set contains no rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// Strategy used by [`RulePrioritizer`] to select a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select the rule with the highest salience.
    HighestSalience,
    /// Select the most recently added rule (last in list).
    MostRecent,
    /// Select the least recently added rule (first in list).
    LeastRecent,
    /// Select the first rule found (same as `LeastRecent`, semantically first match).
    FirstMatch,
    /// Rotate through rules in round-robin order.
    RoundRobin,
}

/// Selects rules from a conflict set and produces ordered firing sequences.
#[derive(Debug)]
pub struct RulePrioritizer {
    strategy: SelectionStrategy,
    /// Round-robin index, tracked internally across `select` calls.
    rr_index: usize,
}

impl RulePrioritizer {
    /// Create a new prioritizer with the given strategy.
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            strategy,
            rr_index: 0,
        }
    }

    /// Select a single rule from the conflict set according to the strategy.
    ///
    /// Returns `None` when the conflict set is empty.
    pub fn select<'a>(&mut self, conflict_set: &'a ConflictSet) -> Option<&'a PrioritizedRule> {
        if conflict_set.is_empty() {
            return None;
        }
        match self.strategy {
            SelectionStrategy::HighestSalience => conflict_set
                .rules
                .iter()
                .max_by_key(|r| r.salience),
            SelectionStrategy::MostRecent => conflict_set.rules.last(),
            SelectionStrategy::LeastRecent | SelectionStrategy::FirstMatch => {
                conflict_set.rules.first()
            }
            SelectionStrategy::RoundRobin => {
                let idx = self.rr_index % conflict_set.rules.len();
                self.rr_index = self.rr_index.wrapping_add(1);
                conflict_set.rules.get(idx)
            }
        }
    }

    /// Return a ordered slice of all rules in the conflict set according to the
    /// strategy.  For `HighestSalience` this is a descending sort; for
    /// `MostRecent` it is reversed insertion order; for `LeastRecent` /
    /// `FirstMatch` it is insertion order; for `RoundRobin` it is insertion
    /// order.
    pub fn order<'a>(&self, conflict_set: &'a ConflictSet) -> Vec<&'a PrioritizedRule> {
        let mut refs: Vec<&PrioritizedRule> = conflict_set.rules.iter().collect();
        match self.strategy {
            SelectionStrategy::HighestSalience => {
                refs.sort_by(|a, b| b.salience.cmp(&a.salience));
            }
            SelectionStrategy::MostRecent => {
                refs.reverse();
            }
            SelectionStrategy::LeastRecent
            | SelectionStrategy::FirstMatch
            | SelectionStrategy::RoundRobin => {
                // Already in insertion order.
            }
        }
        refs
    }

    /// Filter rules belonging to `group`.
    pub fn filter_by_group<'a>(rules: &'a [PrioritizedRule], group: &str) -> Vec<&'a PrioritizedRule> {
        rules.iter().filter(|r| r.group == group).collect()
    }

    /// From a slice of rules, keep at most one rule per mutex group.
    ///
    /// For each mutex group the rule with the highest salience (or first
    /// insertion when tied) is kept; rules without a mutex group are all kept.
    pub fn resolve_mutex<'a>(rules: &'a [PrioritizedRule]) -> Vec<&'a PrioritizedRule> {
        use std::collections::HashMap;
        let mut seen_mutex: HashMap<&str, &PrioritizedRule> = HashMap::new();
        let mut result: Vec<&PrioritizedRule> = Vec::new();

        for rule in rules {
            match &rule.mutex_group {
                None => result.push(rule),
                Some(mg) => {
                    let entry = seen_mutex.entry(mg.as_str()).or_insert(rule);
                    if rule.salience > entry.salience {
                        *entry = rule;
                    }
                }
            }
        }
        // Append the surviving mutex-group winners.
        for winner in seen_mutex.values() {
            result.push(winner);
        }
        result
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(id: &str, salience: i32, group: &str) -> PrioritizedRule {
        PrioritizedRule::new(id, salience, group, None)
    }

    fn make_mutex_rule(id: &str, salience: i32, group: &str, mg: &str) -> PrioritizedRule {
        PrioritizedRule::new(id, salience, group, Some(mg.to_string()))
    }

    // ── ConflictSet ──────────────────────────────────────────────────────────

    #[test]
    fn test_conflict_set_empty() {
        let cs = ConflictSet::new();
        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
    }

    #[test]
    fn test_conflict_set_add() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        assert_eq!(cs.len(), 1);
        assert!(!cs.is_empty());
    }

    #[test]
    fn test_conflict_set_add_multiple() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.add(make_rule("r2", 20, "g1"));
        cs.add(make_rule("r3", 5, "g2"));
        assert_eq!(cs.len(), 3);
    }

    #[test]
    fn test_conflict_set_remove_existing() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.add(make_rule("r2", 20, "g1"));
        let removed = cs.remove("r1");
        assert!(removed);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn test_conflict_set_remove_nonexistent() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        let removed = cs.remove("no_such");
        assert!(!removed);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn test_conflict_set_remove_all() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.remove("r1");
        assert!(cs.is_empty());
    }

    // ── SelectionStrategy::HighestSalience ───────────────────────────────────

    #[test]
    fn test_highest_salience_basic() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 5, "g1"));
        cs.add(make_rule("r2", 15, "g1"));
        cs.add(make_rule("r3", 10, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let selected = p.select(&cs).expect("should select");
        assert_eq!(selected.id, "r2");
    }

    #[test]
    fn test_highest_salience_single() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 99, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let selected = p.select(&cs).expect("should select");
        assert_eq!(selected.id, "r1");
    }

    #[test]
    fn test_highest_salience_empty() {
        let cs = ConflictSet::new();
        let mut p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        assert!(p.select(&cs).is_none());
    }

    #[test]
    fn test_highest_salience_order() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 5, "g1"));
        cs.add(make_rule("r2", 30, "g1"));
        cs.add(make_rule("r3", 20, "g1"));
        let p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let ordered = p.order(&cs);
        assert_eq!(ordered[0].id, "r2");
        assert_eq!(ordered[1].id, "r3");
        assert_eq!(ordered[2].id, "r1");
    }

    // ── SelectionStrategy::MostRecent ────────────────────────────────────────

    #[test]
    fn test_most_recent_basic() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.add(make_rule("r2", 10, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::MostRecent);
        let selected = p.select(&cs).expect("should select");
        assert_eq!(selected.id, "r2");
    }

    #[test]
    fn test_most_recent_order() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.add(make_rule("r2", 10, "g1"));
        cs.add(make_rule("r3", 10, "g1"));
        let p = RulePrioritizer::new(SelectionStrategy::MostRecent);
        let ordered = p.order(&cs);
        assert_eq!(ordered[0].id, "r3");
        assert_eq!(ordered[1].id, "r2");
        assert_eq!(ordered[2].id, "r1");
    }

    // ── SelectionStrategy::LeastRecent ───────────────────────────────────────

    #[test]
    fn test_least_recent_basic() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 10, "g1"));
        cs.add(make_rule("r2", 10, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::LeastRecent);
        let selected = p.select(&cs).expect("should select");
        assert_eq!(selected.id, "r1");
    }

    #[test]
    fn test_least_recent_order() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r3", 10, "g1"));
        cs.add(make_rule("r1", 10, "g1"));
        let p = RulePrioritizer::new(SelectionStrategy::LeastRecent);
        let ordered = p.order(&cs);
        // insertion order preserved
        assert_eq!(ordered[0].id, "r3");
        assert_eq!(ordered[1].id, "r1");
    }

    // ── SelectionStrategy::FirstMatch ────────────────────────────────────────

    #[test]
    fn test_first_match_basic() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 1, "g1"));
        cs.add(make_rule("r2", 100, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::FirstMatch);
        let selected = p.select(&cs).expect("should select");
        assert_eq!(selected.id, "r1");
    }

    // ── SelectionStrategy::RoundRobin ────────────────────────────────────────

    #[test]
    fn test_round_robin_cycles() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 0, "g1"));
        cs.add(make_rule("r2", 0, "g1"));
        cs.add(make_rule("r3", 0, "g1"));
        let mut p = RulePrioritizer::new(SelectionStrategy::RoundRobin);
        let s0 = p.select(&cs).expect("s0").id.clone();
        let s1 = p.select(&cs).expect("s1").id.clone();
        let s2 = p.select(&cs).expect("s2").id.clone();
        let s3 = p.select(&cs).expect("s3").id.clone();
        // 0 -> r1, 1 -> r2, 2 -> r3, 3 -> r1 again
        assert_eq!(s0, "r1");
        assert_eq!(s1, "r2");
        assert_eq!(s2, "r3");
        assert_eq!(s3, "r1");
    }

    // ── filter_by_group ──────────────────────────────────────────────────────

    #[test]
    fn test_filter_by_group_basic() {
        let rules = vec![
            make_rule("r1", 10, "A"),
            make_rule("r2", 20, "B"),
            make_rule("r3", 30, "A"),
        ];
        let filtered = RulePrioritizer::filter_by_group(&rules, "A");
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|r| r.group == "A"));
    }

    #[test]
    fn test_filter_by_group_empty() {
        let rules = vec![make_rule("r1", 10, "A")];
        let filtered = RulePrioritizer::filter_by_group(&rules, "B");
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_group_all_match() {
        let rules = vec![
            make_rule("r1", 10, "X"),
            make_rule("r2", 20, "X"),
        ];
        let filtered = RulePrioritizer::filter_by_group(&rules, "X");
        assert_eq!(filtered.len(), 2);
    }

    // ── resolve_mutex ────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_mutex_keeps_highest_salience() {
        let rules = vec![
            make_mutex_rule("r1", 10, "g1", "mx"),
            make_mutex_rule("r2", 20, "g1", "mx"),
        ];
        let resolved = RulePrioritizer::resolve_mutex(&rules);
        // Only one rule from mutex group "mx" should survive — the highest.
        let mx_rules: Vec<_> = resolved.iter().filter(|r| r.mutex_group.as_deref() == Some("mx")).collect();
        assert_eq!(mx_rules.len(), 1);
        assert_eq!(mx_rules[0].id, "r2");
    }

    #[test]
    fn test_resolve_mutex_no_mutex_rules_kept() {
        let rules = vec![
            make_rule("r1", 10, "g1"),
            make_rule("r2", 20, "g1"),
        ];
        let resolved = RulePrioritizer::resolve_mutex(&rules);
        // All non-mutex rules are kept.
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_resolve_mutex_mixed() {
        let rules = vec![
            make_rule("r1", 10, "g1"),
            make_mutex_rule("r2", 5, "g1", "mxA"),
            make_mutex_rule("r3", 15, "g1", "mxA"),
            make_mutex_rule("r4", 8, "g1", "mxB"),
        ];
        let resolved = RulePrioritizer::resolve_mutex(&rules);
        // r1 (no mutex), r3 (highest in mxA), r4 (only in mxB)
        assert!(resolved.iter().any(|r| r.id == "r1"));
        assert!(resolved.iter().any(|r| r.id == "r3"));
        assert!(resolved.iter().any(|r| r.id == "r4"));
        assert!(!resolved.iter().any(|r| r.id == "r2"));
    }

    #[test]
    fn test_resolve_mutex_different_groups() {
        let rules = vec![
            make_mutex_rule("r1", 10, "g1", "mxA"),
            make_mutex_rule("r2", 20, "g1", "mxB"),
        ];
        let resolved = RulePrioritizer::resolve_mutex(&rules);
        // One per mutex group → 2 survivors
        assert_eq!(resolved.len(), 2);
    }

    // ── PrioritizedRule constructors ─────────────────────────────────────────

    #[test]
    fn test_prioritized_rule_new() {
        let r = PrioritizedRule::new("x", 42, "g", Some("mg".to_string()));
        assert_eq!(r.id, "x");
        assert_eq!(r.salience, 42);
        assert_eq!(r.group, "g");
        assert_eq!(r.mutex_group, Some("mg".to_string()));
    }

    #[test]
    fn test_prioritized_rule_equality() {
        let r1 = make_rule("a", 1, "g");
        let r2 = make_rule("a", 1, "g");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_select_empty_returns_none() {
        let cs = ConflictSet::new();
        let mut p = RulePrioritizer::new(SelectionStrategy::MostRecent);
        assert!(p.select(&cs).is_none());
    }

    #[test]
    fn test_order_single_element() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("only", 5, "g"));
        let p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let ordered = p.order(&cs);
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].id, "only");
    }

    #[test]
    fn test_negative_salience() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("low", -100, "g"));
        cs.add(make_rule("high", 100, "g"));
        let mut p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let sel = p.select(&cs).expect("should select");
        assert_eq!(sel.id, "high");
    }

    #[test]
    fn test_remove_and_reselect() {
        let mut cs = ConflictSet::new();
        cs.add(make_rule("r1", 5, "g"));
        cs.add(make_rule("r2", 99, "g"));
        cs.remove("r2");
        let mut p = RulePrioritizer::new(SelectionStrategy::HighestSalience);
        let sel = p.select(&cs).expect("should select");
        assert_eq!(sel.id, "r1");
    }
}
