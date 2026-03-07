//! Rule forward-chaining execution engine.
//!
//! This module provides a simple but correct forward-chaining rule execution
//! engine that derives new facts from a dataset until a fixpoint is reached.
//!
//! # Architecture
//!
//! ```text
//! Facts ──┐
//!          ├──► RuleExecutor ──► ExecutionResult
//! Rules ──┘
//! ```
//!
//! Rules are represented as condition/head pairs where variable references
//! `?var` are substituted during execution.

use std::collections::{HashMap, HashSet};

// ─── Data Types ───────────────────────────────────────────────────────────────

/// An RDF-like fact (subject, predicate, object triple).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Fact {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// The head (consequent) of a rule — describes the fact to derive.
///
/// Fields may contain variable references such as `?x` that will be resolved
/// using the binding produced by matching the rule conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleHead {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RuleHead {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A single condition (antecedent) of a rule.
///
/// Each field is either a constant or a variable (`?var`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleCondition {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RuleCondition {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A production rule with a unique identifier, list of conditions, and a head.
#[derive(Debug, Clone)]
pub struct SimpleRule {
    pub id: String,
    pub conditions: Vec<RuleCondition>,
    pub head: RuleHead,
}

impl SimpleRule {
    pub fn new(id: impl Into<String>, conditions: Vec<RuleCondition>, head: RuleHead) -> Self {
        Self {
            id: id.into(),
            conditions,
            head,
        }
    }
}

/// Summary of a completed forward-chaining execution run.
#[derive(Debug, Clone, Default)]
pub struct ExecutionResult {
    /// All newly derived facts (not present in the original input).
    pub new_facts: Vec<Fact>,
    /// Number of fixpoint iterations performed.
    pub iterations: usize,
    /// Total number of times a rule produced a new fact.
    pub rules_fired: usize,
}

// ─── Rule Executor ───────────────────────────────────────────────────────────

/// Forward-chaining rule execution engine.
///
/// Rules are applied iteratively until no new facts are produced (fixpoint).
#[derive(Debug, Default)]
pub struct RuleExecutor {
    rules: Vec<SimpleRule>,
}

impl RuleExecutor {
    /// Create a new `RuleExecutor` with no rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to the executor.
    pub fn add_rule(&mut self, rule: SimpleRule) {
        self.rules.push(rule);
    }

    /// Return the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Execute forward chaining until fixpoint.
    ///
    /// Returns an `ExecutionResult` containing all newly derived facts and
    /// execution statistics.
    pub fn execute(&self, facts: &[Fact]) -> ExecutionResult {
        let mut all_facts: HashSet<Fact> = facts.iter().cloned().collect();
        let original_facts: HashSet<Fact> = all_facts.clone();
        let mut iterations = 0_usize;
        let mut rules_fired = 0_usize;

        loop {
            let current_facts: Vec<Fact> = all_facts.iter().cloned().collect();
            let new_in_step = self.execute_one_step_on(&current_facts);

            let mut added_any = false;
            for fact in new_in_step {
                if all_facts.insert(fact) {
                    added_any = true;
                    rules_fired += 1;
                }
            }

            iterations += 1;

            if !added_any {
                break;
            }
        }

        let new_facts: Vec<Fact> = all_facts
            .into_iter()
            .filter(|f| !original_facts.contains(f))
            .collect();

        ExecutionResult {
            new_facts,
            iterations,
            rules_fired,
        }
    }

    /// Perform one forward-chaining iteration.
    ///
    /// Returns a list of candidate new facts (may include already-known ones).
    pub fn execute_one_step(&self, facts: &[Fact]) -> Vec<Fact> {
        self.execute_one_step_on(facts)
    }

    fn execute_one_step_on(&self, facts: &[Fact]) -> Vec<Fact> {
        let mut derived = Vec::new();
        for rule in &self.rules {
            let bindings_list = self.match_conditions(facts, &rule.conditions);
            for bindings in bindings_list {
                let new_fact = self.apply_head(&rule.head, &bindings);
                derived.push(new_fact);
            }
        }
        derived
    }

    /// Find all variable assignments satisfying all rule conditions.
    ///
    /// Performs a nested-loop join across the conditions, accumulating
    /// compatible bindings.
    pub fn match_conditions(
        &self,
        facts: &[Fact],
        conditions: &[RuleCondition],
    ) -> Vec<HashMap<String, String>> {
        if conditions.is_empty() {
            // No conditions → one empty binding (rule always fires)
            return vec![HashMap::new()];
        }

        let mut current: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for condition in conditions {
            let mut next: Vec<HashMap<String, String>> = Vec::new();
            for bindings in &current {
                let matches = self.match_single_condition(facts, condition, bindings);
                next.extend(matches);
            }
            current = next;
        }

        current
    }

    /// Match a single condition against all facts given existing bindings.
    fn match_single_condition(
        &self,
        facts: &[Fact],
        condition: &RuleCondition,
        bindings: &HashMap<String, String>,
    ) -> Vec<HashMap<String, String>> {
        let mut results = Vec::new();
        for fact in facts {
            if let Some(new_bindings) = self.try_bind_condition(condition, fact, bindings) {
                results.push(new_bindings);
            }
        }
        results
    }

    /// Try to unify a condition with a fact given existing bindings.
    fn try_bind_condition(
        &self,
        condition: &RuleCondition,
        fact: &Fact,
        bindings: &HashMap<String, String>,
    ) -> Option<HashMap<String, String>> {
        let mut new_bindings = bindings.clone();
        new_bindings = Self::try_bind_term(&condition.subject, &fact.subject, new_bindings)?;
        new_bindings = Self::try_bind_term(&condition.predicate, &fact.predicate, new_bindings)?;
        new_bindings = Self::try_bind_term(&condition.object, &fact.object, new_bindings)?;
        Some(new_bindings)
    }

    /// Try to bind a pattern term to a concrete value.
    fn try_bind_term(
        term: &str,
        value: &str,
        mut bindings: HashMap<String, String>,
    ) -> Option<HashMap<String, String>> {
        if Self::is_variable(term) {
            let var_name = term.trim_start_matches('?').to_string();
            if let Some(existing) = bindings.get(&var_name) {
                if existing != value {
                    return None;
                }
            } else {
                bindings.insert(var_name, value.to_string());
            }
            Some(bindings)
        } else if term == value {
            Some(bindings)
        } else {
            None
        }
    }

    /// Instantiate a rule head by substituting variables with bound values.
    ///
    /// Returns a concrete `Fact`.
    pub fn apply_head(&self, head: &RuleHead, bindings: &HashMap<String, String>) -> Fact {
        Fact {
            subject: Self::resolve_term(&head.subject, bindings),
            predicate: Self::resolve_term(&head.predicate, bindings),
            object: Self::resolve_term(&head.object, bindings),
        }
    }

    /// Resolve a term using bindings (substitute variable, or return constant).
    fn resolve_term(term: &str, bindings: &HashMap<String, String>) -> String {
        if Self::is_variable(term) {
            let var_name = term.trim_start_matches('?');
            bindings
                .get(var_name)
                .cloned()
                .unwrap_or_else(|| term.to_string())
        } else {
            term.to_string()
        }
    }

    /// Returns `true` if the term is a variable (starts with `?`).
    fn is_variable(term: &str) -> bool {
        term.starts_with('?')
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_executor() -> RuleExecutor {
        RuleExecutor::new()
    }

    // Helper to build a simple transitive-like rule:
    // CONDITION: (?x type ?t), (?t subClassOf ?u)  =>  HEAD: (?x type ?u)
    fn type_hierarchy_rule() -> SimpleRule {
        SimpleRule::new(
            "type-hierarchy",
            vec![
                RuleCondition::new("?x", "type", "?t"),
                RuleCondition::new("?t", "subClassOf", "?u"),
            ],
            RuleHead::new("?x", "type", "?u"),
        )
    }

    // ── rule_count ────────────────────────────────────────────────────────────

    #[test]
    fn test_rule_count_empty() {
        let e = make_executor();
        assert_eq!(e.rule_count(), 0);
    }

    #[test]
    fn test_rule_count_after_add() {
        let mut e = make_executor();
        e.add_rule(type_hierarchy_rule());
        assert_eq!(e.rule_count(), 1);
    }

    #[test]
    fn test_rule_count_multiple() {
        let mut e = make_executor();
        for i in 0..5 {
            e.add_rule(SimpleRule::new(
                format!("r{i}"),
                vec![],
                RuleHead::new("a", "b", "c"),
            ));
        }
        assert_eq!(e.rule_count(), 5);
    }

    // ── match_conditions ──────────────────────────────────────────────────────

    #[test]
    fn test_match_single_condition_one_fact() {
        let e = make_executor();
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let conditions = vec![RuleCondition::new("?x", "knows", ":Bob")];
        let results = e.match_conditions(&facts, &conditions);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("x").map(|s| s.as_str()), Some(":Alice"));
    }

    #[test]
    fn test_match_conditions_no_facts() {
        let e = make_executor();
        let facts: Vec<Fact> = vec![];
        let conditions = vec![RuleCondition::new("?x", "knows", "?y")];
        let results = e.match_conditions(&facts, &conditions);
        assert!(results.is_empty());
    }

    #[test]
    fn test_match_conditions_empty_conditions() {
        let e = make_executor();
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let results = e.match_conditions(&facts, &[]);
        // No conditions → one empty binding
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn test_match_conditions_multiple_facts() {
        let e = make_executor();
        let facts = vec![
            Fact::new(":Alice", "knows", ":Bob"),
            Fact::new(":Carol", "knows", ":Dave"),
        ];
        let conditions = vec![RuleCondition::new("?x", "knows", "?y")];
        let results = e.match_conditions(&facts, &conditions);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_conditions_join_filters() {
        let e = make_executor();
        let facts = vec![
            Fact::new(":Alice", "knows", ":Bob"),
            Fact::new(":Bob", "knows", ":Carol"),
            Fact::new(":Carol", "knows", ":Dave"),
        ];
        // (?x knows ?y) AND (?y knows ?z)
        let conditions = vec![
            RuleCondition::new("?x", "knows", "?y"),
            RuleCondition::new("?y", "knows", "?z"),
        ];
        let results = e.match_conditions(&facts, &conditions);
        assert_eq!(results.len(), 2); // Alice→Bob→Carol and Bob→Carol→Dave
    }

    // ── apply_head ────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_head_all_constants() {
        let e = make_executor();
        let head = RuleHead::new(":Alice", "type", ":Person");
        let bindings = HashMap::new();
        let fact = e.apply_head(&head, &bindings);
        assert_eq!(fact.subject, ":Alice");
        assert_eq!(fact.predicate, "type");
        assert_eq!(fact.object, ":Person");
    }

    #[test]
    fn test_apply_head_variable_substitution() {
        let e = make_executor();
        let head = RuleHead::new("?x", "type", "?u");
        let bindings: HashMap<String, String> = [
            ("x".into(), ":Alice".into()),
            ("u".into(), ":Person".into()),
        ]
        .into();
        let fact = e.apply_head(&head, &bindings);
        assert_eq!(fact.subject, ":Alice");
        assert_eq!(fact.object, ":Person");
    }

    #[test]
    fn test_apply_head_unbound_variable_kept() {
        let e = make_executor();
        let head = RuleHead::new("?x", "type", "?unbound");
        let bindings: HashMap<String, String> = [("x".into(), ":Alice".into())].into();
        let fact = e.apply_head(&head, &bindings);
        assert_eq!(fact.subject, ":Alice");
        assert_eq!(fact.object, "?unbound"); // kept as-is
    }

    // ── execute (simple) ───────────────────────────────────────────────────────

    #[test]
    fn test_execute_no_rules_no_new_facts() {
        let e = make_executor();
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        assert!(result.new_facts.is_empty());
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_execute_simple_rule_fires() {
        let mut e = make_executor();
        // knows(?x, ?y) → knows_transitively(?x, ?y)
        e.add_rule(SimpleRule::new(
            "copy-knows",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "knows_transitively", "?y"),
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        assert_eq!(result.new_facts.len(), 1);
        assert_eq!(result.new_facts[0].predicate, "knows_transitively");
    }

    #[test]
    fn test_execute_fixpoint_one_iteration() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "const-fact",
            vec![],
            RuleHead::new(":A", "prop", ":B"),
        ));
        let facts: Vec<Fact> = vec![];
        let result = e.execute(&facts);
        // Rule fires once; next iteration finds it already present
        assert_eq!(result.new_facts.len(), 1);
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_execute_transitive_closure() {
        let mut e = make_executor();
        // knows(?x, ?y) ∧ knows(?y, ?z) → knows(?x, ?z)
        e.add_rule(SimpleRule::new(
            "transitivity",
            vec![
                RuleCondition::new("?x", "knows", "?y"),
                RuleCondition::new("?y", "knows", "?z"),
            ],
            RuleHead::new("?x", "knows", "?z"),
        ));
        let facts = vec![
            Fact::new(":A", "knows", ":B"),
            Fact::new(":B", "knows", ":C"),
        ];
        let result = e.execute(&facts);
        let derived = Fact::new(":A", "knows", ":C");
        assert!(result.new_facts.contains(&derived));
    }

    #[test]
    fn test_execute_chain_three_hops() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "transitivity",
            vec![
                RuleCondition::new("?x", "link", "?y"),
                RuleCondition::new("?y", "link", "?z"),
            ],
            RuleHead::new("?x", "link", "?z"),
        ));
        let facts = vec![
            Fact::new(":A", "link", ":B"),
            Fact::new(":B", "link", ":C"),
            Fact::new(":C", "link", ":D"),
        ];
        let result = e.execute(&facts);
        let fact_ac = Fact::new(":A", "link", ":C");
        let fact_bd = Fact::new(":B", "link", ":D");
        let fact_ad = Fact::new(":A", "link", ":D");
        assert!(result.new_facts.contains(&fact_ac));
        assert!(result.new_facts.contains(&fact_bd));
        assert!(result.new_facts.contains(&fact_ad));
    }

    #[test]
    fn test_execute_type_hierarchy_rule() {
        let mut e = make_executor();
        e.add_rule(type_hierarchy_rule());
        let facts = vec![
            Fact::new(":Alice", "type", ":Employee"),
            Fact::new(":Employee", "subClassOf", ":Person"),
        ];
        let result = e.execute(&facts);
        let expected = Fact::new(":Alice", "type", ":Person");
        assert!(result.new_facts.contains(&expected));
    }

    #[test]
    fn test_execute_empty_fact_set() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "transitivity",
            vec![
                RuleCondition::new("?x", "knows", "?y"),
                RuleCondition::new("?y", "knows", "?z"),
            ],
            RuleHead::new("?x", "knows", "?z"),
        ));
        let result = e.execute(&[]);
        assert!(result.new_facts.is_empty());
    }

    #[test]
    fn test_execute_no_new_facts_fixpoint_stats() {
        let mut e = make_executor();
        // Rule that could fire but won't produce anything new if base fact exists
        e.add_rule(SimpleRule::new(
            "reflexive",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "knows", "?y"), // same as condition
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        assert!(result.new_facts.is_empty());
        // rules_fired should be 0 since no genuinely new facts were added
        assert_eq!(result.rules_fired, 0);
    }

    #[test]
    fn test_execute_multiple_rules_chain() {
        let mut e = make_executor();
        // Rule 1: knows(?x,?y) → acquaintance(?x,?y)
        e.add_rule(SimpleRule::new(
            "r1",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "acquaintance", "?y"),
        ));
        // Rule 2: acquaintance(?x,?y) → acquaintance(?y,?x)
        e.add_rule(SimpleRule::new(
            "r2",
            vec![RuleCondition::new("?x", "acquaintance", "?y")],
            RuleHead::new("?y", "acquaintance", "?x"),
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        let fwd = Fact::new(":Alice", "acquaintance", ":Bob");
        let bwd = Fact::new(":Bob", "acquaintance", ":Alice");
        assert!(result.new_facts.contains(&fwd));
        assert!(result.new_facts.contains(&bwd));
    }

    #[test]
    fn test_execute_multiple_bindings_produce_multiple_facts() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "met", "?y"),
        ));
        let facts = vec![
            Fact::new(":Alice", "knows", ":Bob"),
            Fact::new(":Alice", "knows", ":Carol"),
            Fact::new(":Bob", "knows", ":Dave"),
        ];
        let result = e.execute(&facts);
        assert_eq!(result.new_facts.len(), 3);
    }

    // ── execute_one_step ──────────────────────────────────────────────────────

    #[test]
    fn test_execute_one_step_basic() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "met", "?y"),
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let derived = e.execute_one_step(&facts);
        assert!(derived.contains(&Fact::new(":Alice", "met", ":Bob")));
    }

    #[test]
    fn test_execute_one_step_empty_facts() {
        let mut e = make_executor();
        e.add_rule(type_hierarchy_rule());
        let derived = e.execute_one_step(&[]);
        assert!(derived.is_empty());
    }

    #[test]
    fn test_execute_rules_fired_counter() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "met", "?y"),
        ));
        let facts = vec![
            Fact::new(":A", "knows", ":B"),
            Fact::new(":A", "knows", ":C"),
        ];
        let result = e.execute(&facts);
        assert_eq!(result.rules_fired, 2);
    }

    #[test]
    fn test_execute_iterations_greater_than_one_for_chain() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "transitivity",
            vec![
                RuleCondition::new("?x", "p", "?y"),
                RuleCondition::new("?y", "p", "?z"),
            ],
            RuleHead::new("?x", "p", "?z"),
        ));
        let facts = vec![
            Fact::new(":A", "p", ":B"),
            Fact::new(":B", "p", ":C"),
            Fact::new(":C", "p", ":D"),
        ];
        let result = e.execute(&facts);
        assert!(
            result.iterations >= 2,
            "expected at least 2 iterations for 3-hop chain"
        );
    }

    // ── additional edge cases and scenarios ────────────────────────────────────

    #[test]
    fn test_add_multiple_rules_independent() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "r1",
            vec![RuleCondition::new("?x", "a", "?y")],
            RuleHead::new("?x", "b", "?y"),
        ));
        e.add_rule(SimpleRule::new(
            "r2",
            vec![RuleCondition::new("?x", "b", "?y")],
            RuleHead::new("?x", "c", "?y"),
        ));
        assert_eq!(e.rule_count(), 2);
    }

    #[test]
    fn test_execute_constant_head_rule() {
        // A rule with no conditions and a constant head produces exactly one fact.
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "const",
            vec![],
            RuleHead::new(":Universe", "exists", "true"),
        ));
        let result = e.execute(&[]);
        assert_eq!(result.new_facts.len(), 1);
        assert_eq!(result.new_facts[0].subject, ":Universe");
    }

    #[test]
    fn test_execute_self_loop_no_infinite_loop() {
        // Rule: (?x knows ?y) → (?x knows ?y) — should terminate because fixpoint is immediate.
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "loop",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "knows", "?y"),
        ));
        let facts = vec![Fact::new(":A", "knows", ":B")];
        let result = e.execute(&facts);
        assert!(
            result.new_facts.is_empty(),
            "reflexive rule produces no new facts"
        );
    }

    #[test]
    fn test_match_conditions_constant_triple() {
        let e = make_executor();
        let facts = vec![
            Fact::new(":X", "rdf:type", ":Class"),
            Fact::new(":Y", "rdf:type", ":OtherClass"),
        ];
        let conditions = vec![RuleCondition::new(":X", "rdf:type", ":Class")];
        let results = e.match_conditions(&facts, &conditions);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty(), "no variables to bind");
    }

    #[test]
    fn test_match_conditions_no_match_for_constant() {
        let e = make_executor();
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let conditions = vec![RuleCondition::new(":Carol", "knows", ":Bob")];
        let results = e.match_conditions(&facts, &conditions);
        assert!(results.is_empty());
    }

    #[test]
    fn test_apply_head_mixed_const_and_var() {
        let e = make_executor();
        let head = RuleHead::new("?x", "rdf:type", ":Person");
        let bindings: HashMap<String, String> = [("x".into(), ":Alice".into())].into();
        let fact = e.apply_head(&head, &bindings);
        assert_eq!(fact.subject, ":Alice");
        assert_eq!(fact.predicate, "rdf:type");
        assert_eq!(fact.object, ":Person");
    }

    #[test]
    fn test_execute_symmetric_rule() {
        let mut e = make_executor();
        // knows(?x, ?y) → knows(?y, ?x)
        e.add_rule(SimpleRule::new(
            "symmetric",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?y", "knows", "?x"),
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        assert!(result
            .new_facts
            .contains(&Fact::new(":Bob", "knows", ":Alice")));
    }

    #[test]
    fn test_execute_multi_hop_class_hierarchy() {
        let mut e = make_executor();
        // subClassOf(?a, ?b) ∧ subClassOf(?b, ?c) → subClassOf(?a, ?c)
        e.add_rule(SimpleRule::new(
            "transitive-subclass",
            vec![
                RuleCondition::new("?a", "subClassOf", "?b"),
                RuleCondition::new("?b", "subClassOf", "?c"),
            ],
            RuleHead::new("?a", "subClassOf", "?c"),
        ));
        let facts = vec![
            Fact::new(":Manager", "subClassOf", ":Employee"),
            Fact::new(":Employee", "subClassOf", ":Person"),
            Fact::new(":Person", "subClassOf", ":Agent"),
        ];
        let result = e.execute(&facts);
        assert!(result
            .new_facts
            .contains(&Fact::new(":Manager", "subClassOf", ":Person")));
        assert!(result
            .new_facts
            .contains(&Fact::new(":Employee", "subClassOf", ":Agent")));
        assert!(result
            .new_facts
            .contains(&Fact::new(":Manager", "subClassOf", ":Agent")));
    }

    #[test]
    fn test_execute_new_facts_excludes_input() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "met", "?y"),
        ));
        let input_fact = Fact::new(":Alice", "knows", ":Bob");
        let result = e.execute(std::slice::from_ref(&input_fact));
        // Original fact must NOT appear in new_facts
        assert!(!result.new_facts.contains(&input_fact));
    }

    #[test]
    fn test_execute_no_rules_preserves_input() {
        let e = make_executor();
        let facts = vec![Fact::new(":A", "b", ":C"), Fact::new(":D", "e", ":F")];
        let result = e.execute(&facts);
        assert!(result.new_facts.is_empty());
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_execute_one_step_returns_duplicates_ok() {
        // execute_one_step may return duplicate facts (deduplication is done in execute).
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy1",
            vec![RuleCondition::new("?x", "p", "?y")],
            RuleHead::new("?x", "q", "?y"),
        ));
        e.add_rule(SimpleRule::new(
            "copy2",
            vec![RuleCondition::new("?x", "p", "?y")],
            RuleHead::new("?x", "q", "?y"),
        ));
        let facts = vec![Fact::new(":A", "p", ":B")];
        let derived = e.execute_one_step(&facts);
        // Both rules produce the same fact → 2 entries (duplicates allowed in one-step output)
        assert_eq!(derived.len(), 2);
    }

    #[test]
    fn test_execute_deduplicates_across_rules() {
        // execute() deduplicates: two rules producing the same fact count as rules_fired=1.
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "copy1",
            vec![RuleCondition::new("?x", "p", "?y")],
            RuleHead::new("?x", "q", "?y"),
        ));
        e.add_rule(SimpleRule::new(
            "copy2",
            vec![RuleCondition::new("?x", "p", "?y")],
            RuleHead::new("?x", "q", "?y"),
        ));
        let facts = vec![Fact::new(":A", "p", ":B")];
        let result = e.execute(&facts);
        // Deduplicated: only 1 genuinely new fact
        assert_eq!(result.new_facts.len(), 1);
    }

    #[test]
    fn test_fact_equality() {
        let f1 = Fact::new(":A", "p", ":B");
        let f2 = Fact::new(":A", "p", ":B");
        let f3 = Fact::new(":A", "p", ":C");
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
    }

    #[test]
    fn test_rule_with_three_conditions() {
        let mut e = make_executor();
        // (?x p ?y) ∧ (?y p ?z) ∧ (?z p ?w) → (?x skip3 ?w)
        e.add_rule(SimpleRule::new(
            "skip3",
            vec![
                RuleCondition::new("?x", "p", "?y"),
                RuleCondition::new("?y", "p", "?z"),
                RuleCondition::new("?z", "p", "?w"),
            ],
            RuleHead::new("?x", "skip3", "?w"),
        ));
        let facts = vec![
            Fact::new(":A", "p", ":B"),
            Fact::new(":B", "p", ":C"),
            Fact::new(":C", "p", ":D"),
        ];
        let result = e.execute(&facts);
        assert!(result.new_facts.contains(&Fact::new(":A", "skip3", ":D")));
    }

    #[test]
    fn test_match_conditions_variable_reuse_across_conditions() {
        // Ensure that a variable bound in condition 1 constrains condition 2.
        let e = make_executor();
        let facts = vec![
            Fact::new(":X", "type", ":A"),
            Fact::new(":X", "type", ":B"),
            Fact::new(":Y", "type", ":A"),
        ];
        // Condition: (?s type ?t) ∧ (?s type :A) — second condition uses constant :A
        let conditions = vec![
            RuleCondition::new("?s", "type", "?t"),
            RuleCondition::new("?s", "type", ":A"),
        ];
        let results = e.match_conditions(&facts, &conditions);
        // :X has type :A and :B; :Y has type :A.
        // Solutions where s=:X and t in {:A,:B}, plus s=:Y and t=:A.
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_simple_rule_new_constructor() {
        let rule = SimpleRule::new(
            "test-rule",
            vec![RuleCondition::new("?x", "p", "?y")],
            RuleHead::new("?x", "q", "?y"),
        );
        assert_eq!(rule.id, "test-rule");
        assert_eq!(rule.conditions.len(), 1);
    }

    #[test]
    fn test_rule_condition_new_constructor() {
        let cond = RuleCondition::new("?subject", "predicate", "object");
        assert_eq!(cond.subject, "?subject");
        assert_eq!(cond.predicate, "predicate");
        assert_eq!(cond.object, "object");
    }

    #[test]
    fn test_rule_head_new_constructor() {
        let head = RuleHead::new("subject", "predicate", "?object");
        assert_eq!(head.subject, "subject");
        assert_eq!(head.predicate, "predicate");
        assert_eq!(head.object, "?object");
    }

    #[test]
    fn test_execution_result_default() {
        let result = ExecutionResult::default();
        assert!(result.new_facts.is_empty());
        assert_eq!(result.iterations, 0);
        assert_eq!(result.rules_fired, 0);
    }

    #[test]
    fn test_execute_two_independent_rules_same_input() {
        let mut e = make_executor();
        e.add_rule(SimpleRule::new(
            "rule-a",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?x", "met", "?y"),
        ));
        e.add_rule(SimpleRule::new(
            "rule-b",
            vec![RuleCondition::new("?x", "knows", "?y")],
            RuleHead::new("?y", "known_by", "?x"),
        ));
        let facts = vec![Fact::new(":Alice", "knows", ":Bob")];
        let result = e.execute(&facts);
        assert!(result
            .new_facts
            .contains(&Fact::new(":Alice", "met", ":Bob")));
        assert!(result
            .new_facts
            .contains(&Fact::new(":Bob", "known_by", ":Alice")));
        assert_eq!(result.new_facts.len(), 2);
    }
}
