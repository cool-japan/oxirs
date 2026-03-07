//! SPARQL EXISTS and NOT EXISTS evaluation.
//!
//! This module implements the EXISTS and NOT EXISTS graph pattern evaluation
//! as defined in SPARQL 1.1 specification (section 18.2.1).
//!
//! # Overview
//!
//! EXISTS checks whether a graph pattern matches at least one solution given
//! the current input bindings. NOT EXISTS is its negation.
//!
//! The evaluator supports:
//! - Triple patterns with variable bindings
//! - AND (join) patterns
//! - OPTIONAL (left outer join) patterns
//! - FILTER patterns
//! - UNION patterns

use std::collections::HashMap;

/// A single RDF triple fact in the dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TripleFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl TripleFact {
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

/// A SPARQL graph pattern.
#[derive(Debug, Clone)]
pub enum GraphPattern {
    /// A single triple pattern (may contain variables starting with `?`)
    Triple(TripleFact),
    /// Conjunction of multiple patterns
    And(Vec<GraphPattern>),
    /// Optional (left outer join) pattern
    Optional(Box<GraphPattern>),
    /// Filter with a condition expression (simplified: just variable=value)
    Filter {
        pattern: Box<GraphPattern>,
        condition: String,
    },
    /// Union of two patterns
    Union(Box<GraphPattern>, Box<GraphPattern>),
}

/// A solution mapping: variable name (without `?`) → bound value.
pub type SolutionMapping = HashMap<String, String>;

/// Evaluates SPARQL EXISTS and NOT EXISTS graph patterns.
#[derive(Debug, Default)]
pub struct ExistsEvaluator;

impl ExistsEvaluator {
    /// Create a new `ExistsEvaluator`.
    pub fn new() -> Self {
        Self
    }

    /// Returns `true` if the pattern has at least one solution given `input` bindings.
    pub fn evaluate_exists(
        &self,
        facts: &[TripleFact],
        pattern: &GraphPattern,
        input: &SolutionMapping,
    ) -> bool {
        let solutions = self.inner_match(facts, pattern, input.clone());
        !solutions.is_empty()
    }

    /// Returns `true` if the pattern has NO solutions given `input` bindings.
    pub fn evaluate_not_exists(
        &self,
        facts: &[TripleFact],
        pattern: &GraphPattern,
        input: &SolutionMapping,
    ) -> bool {
        !self.evaluate_exists(facts, pattern, input)
    }

    /// Recursively match a pattern against facts, extending `bindings`.
    ///
    /// Returns all possible solution mappings that satisfy the pattern.
    pub fn inner_match(
        &self,
        facts: &[TripleFact],
        pattern: &GraphPattern,
        bindings: SolutionMapping,
    ) -> Vec<SolutionMapping> {
        match pattern {
            GraphPattern::Triple(triple) => self.match_triple(facts, triple, bindings),

            GraphPattern::And(patterns) => {
                // Start with the initial bindings as a single solution, then
                // progressively join each pattern.
                let mut current_solutions = vec![bindings];
                for sub_pattern in patterns {
                    let mut next_solutions = Vec::new();
                    for sol in current_solutions {
                        let results = self.inner_match(facts, sub_pattern, sol);
                        next_solutions.extend(results);
                    }
                    current_solutions = next_solutions;
                }
                current_solutions
            }

            GraphPattern::Optional(sub_pattern) => {
                // Left outer join: if sub_pattern has solutions, return them;
                // otherwise return the original bindings unchanged.
                let results = self.inner_match(facts, sub_pattern, bindings.clone());
                if results.is_empty() {
                    vec![bindings]
                } else {
                    results
                }
            }

            GraphPattern::Filter {
                pattern: sub_pattern,
                condition,
            } => {
                let solutions = self.inner_match(facts, sub_pattern, bindings);
                solutions
                    .into_iter()
                    .filter(|sol| self.evaluate_filter_condition(sol, condition))
                    .collect()
            }

            GraphPattern::Union(left, right) => {
                let mut left_solutions = self.inner_match(facts, left, bindings.clone());
                let right_solutions = self.inner_match(facts, right, bindings);
                left_solutions.extend(right_solutions);
                left_solutions
            }
        }
    }

    /// Match a single triple pattern against all facts.
    fn match_triple(
        &self,
        facts: &[TripleFact],
        triple: &TripleFact,
        bindings: SolutionMapping,
    ) -> Vec<SolutionMapping> {
        let mut results = Vec::new();
        for fact in facts {
            if let Some(new_bindings) = self.try_bind_triple(triple, fact, bindings.clone()) {
                results.push(new_bindings);
            }
        }
        results
    }

    /// Try to bind a triple pattern to a concrete fact, extending `bindings`.
    ///
    /// Returns `None` if the triple pattern conflicts with the fact or bindings.
    fn try_bind_triple(
        &self,
        pattern: &TripleFact,
        fact: &TripleFact,
        mut bindings: SolutionMapping,
    ) -> Option<SolutionMapping> {
        // Attempt to match each component
        bindings = self.try_bind_term(&pattern.subject, &fact.subject, bindings)?;
        bindings = self.try_bind_term(&pattern.predicate, &fact.predicate, bindings)?;
        bindings = self.try_bind_term(&pattern.object, &fact.object, bindings)?;
        Some(bindings)
    }

    /// Try to bind a pattern term (variable or constant) to a concrete value.
    fn try_bind_term(
        &self,
        term: &str,
        value: &str,
        mut bindings: SolutionMapping,
    ) -> Option<SolutionMapping> {
        if Self::is_variable(term) {
            let var_name = term.trim_start_matches('?').to_string();
            if let Some(existing) = bindings.get(&var_name) {
                if existing != value {
                    return None; // Conflicting binding
                }
                // Already bound to same value — OK
            } else {
                bindings.insert(var_name, value.to_string());
            }
            Some(bindings)
        } else {
            // Constant: must match exactly
            if term == value {
                Some(bindings)
            } else {
                None
            }
        }
    }

    /// Evaluate a simplified filter condition string.
    ///
    /// Supports formats:
    /// - `?var=value` — variable equals literal
    /// - `?var!=value` — variable not equals literal
    fn evaluate_filter_condition(&self, bindings: &SolutionMapping, condition: &str) -> bool {
        if let Some(pos) = condition.find("!=") {
            let lhs = condition[..pos].trim();
            let rhs = condition[pos + 2..].trim();
            if Self::is_variable(lhs) {
                let var_name = lhs.trim_start_matches('?');
                if let Some(val) = bindings.get(var_name) {
                    return val != rhs;
                }
                // Unbound variable: condition is false (conservative)
                return false;
            }
        } else if let Some(pos) = condition.find('=') {
            let lhs = condition[..pos].trim();
            let rhs = condition[pos + 1..].trim();
            if Self::is_variable(lhs) {
                let var_name = lhs.trim_start_matches('?');
                if let Some(val) = bindings.get(var_name) {
                    return val == rhs;
                }
                return false;
            }
        }
        // Unrecognised condition: pass through
        true
    }

    /// Returns `true` if the term is a SPARQL variable (starts with `?`).
    pub fn is_variable(term: &str) -> bool {
        term.starts_with('?')
    }

    /// Returns `true` if two solution mappings have no conflicting bindings.
    ///
    /// Two mappings are *compatible* if for every variable bound in both, they
    /// agree on the same value.
    pub fn compatible(m1: &SolutionMapping, m2: &SolutionMapping) -> bool {
        for (var, val1) in m1 {
            if let Some(val2) = m2.get(var) {
                if val1 != val2 {
                    return false;
                }
            }
        }
        true
    }

    /// Merge two compatible solution mappings into one.
    ///
    /// Bindings from `m2` are added to `m1`; `m1` values win on conflicts.
    pub fn merge(mut m1: SolutionMapping, m2: SolutionMapping) -> SolutionMapping {
        for (k, v) in m2 {
            m1.entry(k).or_insert(v);
        }
        m1
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluator() -> ExistsEvaluator {
        ExistsEvaluator::new()
    }

    fn empty_input() -> SolutionMapping {
        SolutionMapping::new()
    }

    fn facts_abc() -> Vec<TripleFact> {
        vec![
            TripleFact::new(":Alice", "knows", ":Bob"),
            TripleFact::new(":Bob", "knows", ":Carol"),
            TripleFact::new(":Alice", "age", "30"),
            TripleFact::new(":Bob", "age", "25"),
        ]
    }

    // ── is_variable ──────────────────────────────────────────────────────────

    #[test]
    fn test_is_variable_true() {
        assert!(ExistsEvaluator::is_variable("?x"));
    }

    #[test]
    fn test_is_variable_false_constant() {
        assert!(!ExistsEvaluator::is_variable(":Alice"));
    }

    #[test]
    fn test_is_variable_empty_string() {
        assert!(!ExistsEvaluator::is_variable(""));
    }

    #[test]
    fn test_is_variable_question_mark_only() {
        assert!(ExistsEvaluator::is_variable("?"));
    }

    // ── compatible ───────────────────────────────────────────────────────────

    #[test]
    fn test_compatible_empty_maps() {
        let m1 = SolutionMapping::new();
        let m2 = SolutionMapping::new();
        assert!(ExistsEvaluator::compatible(&m1, &m2));
    }

    #[test]
    fn test_compatible_same_bindings() {
        let m1: SolutionMapping = [("x".into(), "1".into())].into();
        let m2: SolutionMapping = [("x".into(), "1".into())].into();
        assert!(ExistsEvaluator::compatible(&m1, &m2));
    }

    #[test]
    fn test_compatible_different_vars() {
        let m1: SolutionMapping = [("x".into(), "1".into())].into();
        let m2: SolutionMapping = [("y".into(), "2".into())].into();
        assert!(ExistsEvaluator::compatible(&m1, &m2));
    }

    #[test]
    fn test_compatible_conflict() {
        let m1: SolutionMapping = [("x".into(), "1".into())].into();
        let m2: SolutionMapping = [("x".into(), "2".into())].into();
        assert!(!ExistsEvaluator::compatible(&m1, &m2));
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_disjoint() {
        let m1: SolutionMapping = [("x".into(), "1".into())].into();
        let m2: SolutionMapping = [("y".into(), "2".into())].into();
        let merged = ExistsEvaluator::merge(m1, m2);
        assert_eq!(merged.get("x").map(|s| s.as_str()), Some("1"));
        assert_eq!(merged.get("y").map(|s| s.as_str()), Some("2"));
    }

    #[test]
    fn test_merge_m1_wins_on_overlap() {
        let m1: SolutionMapping = [("x".into(), "1".into())].into();
        let m2: SolutionMapping = [("x".into(), "99".into())].into();
        let merged = ExistsEvaluator::merge(m1, m2);
        assert_eq!(merged.get("x").map(|s| s.as_str()), Some("1"));
    }

    // ── EXISTS basic ──────────────────────────────────────────────────────────

    #[test]
    fn test_exists_simple_triple_found() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new(":Alice", "knows", ":Bob"));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_exists_simple_triple_not_found() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new(":Alice", "knows", ":Dave"));
        assert!(!e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_exists_variable_subject() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", ":Bob"));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_exists_all_variables() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "?p", "?o"));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_exists_empty_graph() {
        let e = evaluator();
        let facts: Vec<TripleFact> = vec![];
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "?p", "?o"));
        assert!(!e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_exists_with_pre_bound_input_matching() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", "?y"));
        let mut input = SolutionMapping::new();
        input.insert("x".into(), ":Alice".into());
        assert!(e.evaluate_exists(&facts, &pattern, &input));
    }

    #[test]
    fn test_exists_with_pre_bound_input_not_matching() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", "?y"));
        // Dave doesn't know anyone
        let mut input = SolutionMapping::new();
        input.insert("x".into(), ":Dave".into());
        assert!(!e.evaluate_exists(&facts, &pattern, &input));
    }

    #[test]
    fn test_exists_scoped_binding_not_leaked() {
        // EXISTS evaluation must not affect the outer solution mapping.
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", ":Bob"));
        let input = empty_input();
        e.evaluate_exists(&facts, &pattern, &input);
        // Original input unchanged
        assert!(input.is_empty());
    }

    // ── NOT EXISTS basic ──────────────────────────────────────────────────────

    #[test]
    fn test_not_exists_simple_triple_absent() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new(":Nobody", "knows", ":Bob"));
        assert!(e.evaluate_not_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_not_exists_simple_triple_present() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new(":Alice", "knows", ":Bob"));
        assert!(!e.evaluate_not_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_not_exists_all_variables_nonempty_graph() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "?p", "?o"));
        assert!(!e.evaluate_not_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_not_exists_empty_graph() {
        let e = evaluator();
        let facts: Vec<TripleFact> = vec![];
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "?p", "?o"));
        assert!(e.evaluate_not_exists(&facts, &pattern, &empty_input()));
    }

    // ── AND patterns ──────────────────────────────────────────────────────────

    #[test]
    fn test_and_both_match() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", ":Bob")),
            GraphPattern::Triple(TripleFact::new("?x", "age", "?a")),
        ]);
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_and_second_fails() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", ":Bob")),
            GraphPattern::Triple(TripleFact::new("?x", "flies", "?a")),
        ]);
        assert!(!e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_and_variable_join() {
        let e = evaluator();
        let facts = facts_abc();
        // Find ?x that knows ?y, and ?y also knows someone
        let pattern = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", "?y")),
            GraphPattern::Triple(TripleFact::new("?y", "knows", "?z")),
        ]);
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_and_empty_pattern_list() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::And(vec![]);
        // Empty AND → trivially satisfied (vacuous join)
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    // ── OPTIONAL patterns ─────────────────────────────────────────────────────

    #[test]
    fn test_optional_pattern_present() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Optional(Box::new(GraphPattern::Triple(TripleFact::new(
            ":Alice", "knows", ":Bob",
        ))));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_optional_pattern_absent_still_exists() {
        let e = evaluator();
        let facts = facts_abc();
        // Optional triple that doesn't exist: still returns the empty input binding
        let pattern = GraphPattern::Optional(Box::new(GraphPattern::Triple(TripleFact::new(
            ":Nobody", "knows", ":Bob",
        ))));
        // Should return 1 solution (the empty input)
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_empty());
    }

    #[test]
    fn test_optional_extends_bindings_when_found() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Optional(Box::new(GraphPattern::Triple(TripleFact::new(
            "?s", "age", "?a",
        ))));
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        // Each fact with predicate "age" should produce a binding
        assert!(solutions.iter().any(|sol| sol.contains_key("a")));
    }

    // ── UNION patterns ────────────────────────────────────────────────────────

    #[test]
    fn test_union_left_matches() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Union(
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Alice", "knows", ":Bob",
            ))),
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Nobody", "knows", ":Bob",
            ))),
        );
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_union_right_matches() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Union(
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Nobody", "knows", ":Bob",
            ))),
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Alice", "knows", ":Bob",
            ))),
        );
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_union_neither_matches() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Union(
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Nobody", "knows", ":Bob",
            ))),
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Noone", "knows", ":Bob",
            ))),
        );
        assert!(!e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_union_both_match_returns_all_solutions() {
        let e = evaluator();
        let facts = facts_abc();
        // Both branches match: should have 2 solutions
        let pattern = GraphPattern::Union(
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Alice", "knows", ":Bob",
            ))),
            Box::new(GraphPattern::Triple(TripleFact::new(
                ":Bob", "knows", ":Carol",
            ))),
        );
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 2);
    }

    // ── FILTER patterns ───────────────────────────────────────────────────────

    #[test]
    fn test_filter_passing() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Filter {
            pattern: Box::new(GraphPattern::Triple(TripleFact::new("?x", "age", "?a"))),
            condition: "?a=30".to_string(),
        };
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get("x").map(|s| s.as_str()), Some(":Alice"));
    }

    #[test]
    fn test_filter_not_equals() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Filter {
            pattern: Box::new(GraphPattern::Triple(TripleFact::new("?x", "age", "?a"))),
            condition: "?a!=30".to_string(),
        };
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get("x").map(|s| s.as_str()), Some(":Bob"));
    }

    #[test]
    fn test_filter_removes_all() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Filter {
            pattern: Box::new(GraphPattern::Triple(TripleFact::new("?x", "age", "?a"))),
            condition: "?a=999".to_string(),
        };
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert!(solutions.is_empty());
    }

    // ── inner_match detailed ──────────────────────────────────────────────────

    #[test]
    fn test_inner_match_returns_multiple_solutions() {
        let e = evaluator();
        let facts = facts_abc();
        // ?x knows ?y: should find 2 solutions
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", "?y"));
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_inner_match_pre_bound_narrows_results() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "knows", "?y"));
        let mut input = SolutionMapping::new();
        input.insert("x".into(), ":Alice".into());
        let solutions = e.inner_match(&facts, &pattern, input);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get("y").map(|s| s.as_str()), Some(":Bob"));
    }

    #[test]
    fn test_inner_match_constant_predicate() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "age", "?o"));
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_inner_match_no_matching_facts() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "flies", "?o"));
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert!(solutions.is_empty());
    }

    // ── scoped bindings (EXISTS must not leak) ─────────────────────────────────

    #[test]
    fn test_exists_bindings_scoped_does_not_modify_input() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?newvar", "knows", ":Bob"));
        let input = SolutionMapping::new();
        let result = e.evaluate_exists(&facts, &pattern, &input);
        assert!(result);
        // The original `input` is unchanged (Rust ownership ensures this)
        assert!(input.is_empty());
    }

    // ── complex nested patterns ───────────────────────────────────────────────

    #[test]
    fn test_and_within_union() {
        let e = evaluator();
        let facts = facts_abc();
        let knows_bob = GraphPattern::Triple(TripleFact::new("?x", "knows", ":Bob"));
        let bob_old = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", ":Carol")),
            GraphPattern::Triple(TripleFact::new("?x", "age", "?a")),
        ]);
        let pattern = GraphPattern::Union(Box::new(knows_bob), Box::new(bob_old));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_optional_within_and() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", "?y")),
            GraphPattern::Optional(Box::new(GraphPattern::Triple(TripleFact::new(
                "?x", "email", "?e",
            )))),
        ]);
        // Should return solutions (email is optional, won't block)
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert!(!solutions.is_empty());
    }

    #[test]
    fn test_filter_within_and() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::And(vec![
            GraphPattern::Triple(TripleFact::new("?x", "knows", "?y")),
            GraphPattern::Filter {
                pattern: Box::new(GraphPattern::Triple(TripleFact::new("?x", "age", "?a"))),
                condition: "?a=30".to_string(),
            },
        ]);
        let solutions = e.inner_match(&facts, &pattern, empty_input());
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get("x").map(|s| s.as_str()), Some(":Alice"));
    }

    #[test]
    fn test_not_exists_with_bound_variable() {
        let e = evaluator();
        let facts = facts_abc();
        let pattern = GraphPattern::Triple(TripleFact::new("?x", "flies", "?z"));
        let mut input = SolutionMapping::new();
        input.insert("x".into(), ":Alice".into());
        assert!(e.evaluate_not_exists(&facts, &pattern, &input));
    }

    #[test]
    fn test_triple_fact_new_constructor() {
        let t = TripleFact::new(":s", ":p", ":o");
        assert_eq!(t.subject, ":s");
        assert_eq!(t.predicate, ":p");
        assert_eq!(t.object, ":o");
    }

    #[test]
    fn test_merge_empty_maps() {
        let m1 = SolutionMapping::new();
        let m2 = SolutionMapping::new();
        let merged = ExistsEvaluator::merge(m1, m2);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_exists_single_fact_dataset_match() {
        let e = evaluator();
        let facts = vec![TripleFact::new("ex:Alice", "foaf:knows", "ex:Bob")];
        let pattern = GraphPattern::Triple(TripleFact::new("?s", "foaf:knows", "?o"));
        assert!(e.evaluate_exists(&facts, &pattern, &empty_input()));
    }

    #[test]
    fn test_not_exists_on_empty_dataset() {
        let e = evaluator();
        let pattern =
            GraphPattern::And(vec![GraphPattern::Triple(TripleFact::new("?x", "p", "?y"))]);
        assert!(e.evaluate_not_exists(&[], &pattern, &empty_input()));
    }
}
