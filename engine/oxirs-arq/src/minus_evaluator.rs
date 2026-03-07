//! SPARQL MINUS pattern evaluation (set-difference algebra).
//!
//! Implements the SPARQL 1.1 MINUS operator as specified in:
//! <https://www.w3.org/TR/sparql11-query/#neg-minus>
//!
//! The MINUS operator removes solution mappings from the left-hand side
//! that are *compatible* with at least one solution mapping in the right-hand side.
//! Two solution mappings are compatible when they agree on every variable
//! they share in common.  If the two mappings share no variables, they are
//! *not* compatible, so the left-hand row is **kept**.

use std::collections::HashMap;

/// A single solution mapping: variable name → RDF term string.
pub type Binding = HashMap<String, String>;

/// An ordered collection of solution mappings.
#[derive(Debug, Clone, Default)]
pub struct BindingSet {
    rows: Vec<Binding>,
}

impl BindingSet {
    /// Create an empty `BindingSet`.
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append a binding row.
    pub fn push(&mut self, b: Binding) {
        self.rows.push(b);
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterate over rows.
    pub fn iter(&self) -> impl Iterator<Item = &Binding> {
        self.rows.iter()
    }
}

/// Statistics produced alongside a MINUS evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinusStats {
    /// Number of rows in the left-hand operand.
    pub lhs_count: usize,
    /// Number of rows in the right-hand operand.
    pub rhs_count: usize,
    /// Number of rows that passed through (were *not* eliminated).
    pub result_count: usize,
    /// Number of rows that were eliminated.
    pub eliminated_count: usize,
}

/// Evaluates the SPARQL MINUS algebra operator.
///
/// ```
/// use oxirs_arq::minus_evaluator::{Binding, BindingSet, MinusEvaluator};
///
/// let mut lhs = BindingSet::new();
/// let mut row = Binding::new();
/// row.insert("x".to_string(), "<:a>".to_string());
/// lhs.push(row);
///
/// let rhs = BindingSet::new(); // empty RHS → nothing eliminated
/// let result = MinusEvaluator::new().evaluate(&lhs, &rhs);
/// assert_eq!(result.len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MinusEvaluator;

impl MinusEvaluator {
    /// Create a new evaluator (stateless).
    pub fn new() -> Self {
        Self
    }

    /// Return the variable names present in both `a` and `b`.
    pub fn shared_vars(a: &Binding, b: &Binding) -> Vec<String> {
        a.keys().filter(|k| b.contains_key(*k)).cloned().collect()
    }

    /// Return `true` when `a` and `b` share at least one variable **and**
    /// all shared variables have identical values.
    ///
    /// Per the SPARQL 1.1 spec (§18.5):
    /// - If the two mappings have *no* shared domain, they are **not** compatible
    ///   for the purpose of MINUS elimination, so `compatible` returns `false`.
    pub fn compatible(a: &Binding, b: &Binding) -> bool {
        let shared = Self::shared_vars(a, b);
        if shared.is_empty() {
            // No shared variables → row is NOT eliminated by this RHS row.
            return false;
        }
        // All shared variables must agree.
        shared.iter().all(|var| a.get(var) == b.get(var))
    }

    /// Evaluate MINUS: keep every row in `lhs` that is *not* compatible with
    /// any row in `rhs`.
    pub fn evaluate(&self, lhs: &BindingSet, rhs: &BindingSet) -> BindingSet {
        let mut result = BindingSet::new();
        for left_row in lhs.iter() {
            let eliminated = rhs
                .iter()
                .any(|right_row| Self::compatible(left_row, right_row));
            if !eliminated {
                result.push(left_row.clone());
            }
        }
        result
    }

    /// Like `evaluate`, but also returns statistics about the operation.
    pub fn evaluate_with_stats(
        &self,
        lhs: &BindingSet,
        rhs: &BindingSet,
    ) -> (BindingSet, MinusStats) {
        let lhs_count = lhs.len();
        let rhs_count = rhs.len();

        let result = self.evaluate(lhs, rhs);
        let result_count = result.len();
        let eliminated_count = lhs_count.saturating_sub(result_count);

        let stats = MinusStats {
            lhs_count,
            rhs_count,
            result_count,
            eliminated_count,
        };
        (result, stats)
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binding(pairs: &[(&str, &str)]) -> Binding {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn binding_set(rows: Vec<Binding>) -> BindingSet {
        let mut bs = BindingSet::new();
        for r in rows {
            bs.push(r);
        }
        bs
    }

    // ── BindingSet basic API ──────────────────────────────────────────────────

    #[test]
    fn test_binding_set_new_is_empty() {
        let bs = BindingSet::new();
        assert!(bs.is_empty());
        assert_eq!(bs.len(), 0);
    }

    #[test]
    fn test_binding_set_push_and_len() {
        let mut bs = BindingSet::new();
        bs.push(make_binding(&[("x", ":a")]));
        assert_eq!(bs.len(), 1);
        assert!(!bs.is_empty());
    }

    #[test]
    fn test_binding_set_iter() {
        let mut bs = BindingSet::new();
        bs.push(make_binding(&[("x", ":a")]));
        bs.push(make_binding(&[("x", ":b")]));
        let vals: Vec<&str> = bs.iter().map(|r| r["x"].as_str()).collect();
        assert_eq!(vals, vec![":a", ":b"]);
    }

    #[test]
    fn test_binding_set_default() {
        let bs = BindingSet::default();
        assert!(bs.is_empty());
    }

    // ── shared_vars ──────────────────────────────────────────────────────────

    #[test]
    fn test_shared_vars_none() {
        let a = make_binding(&[("x", ":a")]);
        let b = make_binding(&[("y", ":b")]);
        let mut sv = MinusEvaluator::shared_vars(&a, &b);
        sv.sort();
        assert!(sv.is_empty());
    }

    #[test]
    fn test_shared_vars_one() {
        let a = make_binding(&[("x", ":a"), ("y", ":c")]);
        let b = make_binding(&[("x", ":a"), ("z", ":d")]);
        let mut sv = MinusEvaluator::shared_vars(&a, &b);
        sv.sort();
        assert_eq!(sv, vec!["x"]);
    }

    #[test]
    fn test_shared_vars_all() {
        let a = make_binding(&[("x", ":a"), ("y", ":b")]);
        let b = make_binding(&[("x", ":a"), ("y", ":b")]);
        let mut sv = MinusEvaluator::shared_vars(&a, &b);
        sv.sort();
        assert_eq!(sv, vec!["x", "y"]);
    }

    #[test]
    fn test_shared_vars_empty_bindings() {
        let a = Binding::new();
        let b = make_binding(&[("x", ":a")]);
        assert!(MinusEvaluator::shared_vars(&a, &b).is_empty());
    }

    // ── compatible ───────────────────────────────────────────────────────────

    #[test]
    fn test_compatible_no_shared_vars_not_compatible() {
        let a = make_binding(&[("x", ":a")]);
        let b = make_binding(&[("y", ":b")]);
        assert!(!MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_shared_agree() {
        let a = make_binding(&[("x", ":a"), ("y", ":c")]);
        let b = make_binding(&[("x", ":a"), ("z", ":d")]);
        assert!(MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_shared_disagree() {
        let a = make_binding(&[("x", ":a")]);
        let b = make_binding(&[("x", ":b")]);
        assert!(!MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_multiple_shared_all_agree() {
        let a = make_binding(&[("x", ":a"), ("y", ":b")]);
        let b = make_binding(&[("x", ":a"), ("y", ":b")]);
        assert!(MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_multiple_shared_one_disagrees() {
        let a = make_binding(&[("x", ":a"), ("y", ":b")]);
        let b = make_binding(&[("x", ":a"), ("y", ":DIFFERENT")]);
        assert!(!MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_empty_both() {
        let a = Binding::new();
        let b = Binding::new();
        assert!(!MinusEvaluator::compatible(&a, &b));
    }

    // ── evaluate: basic cases ─────────────────────────────────────────────────

    #[test]
    fn test_evaluate_empty_lhs_empty_rhs() {
        let eval = MinusEvaluator::new();
        let lhs = BindingSet::new();
        let rhs = BindingSet::new();
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_nonempty_lhs_empty_rhs() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let rhs = BindingSet::new();
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_empty_lhs_nonempty_rhs() {
        let eval = MinusEvaluator::new();
        let lhs = BindingSet::new();
        let rhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_no_shared_vars_keeps_all() {
        // LHS and RHS have no variables in common → nothing eliminated.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let rhs = binding_set(vec![make_binding(&[("y", ":a")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_evaluate_all_eliminated() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let rhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_partial_overlap() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":c")]),
        ]);
        let rhs = binding_set(vec![make_binding(&[("x", ":b")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 2);
        let values: Vec<&str> = result.iter().map(|r| r["x"].as_str()).collect();
        assert!(!values.contains(&":b"));
    }

    #[test]
    fn test_evaluate_shared_var_disagrees_kept() {
        // Shared var x has different value → not compatible → kept.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":b")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_multiple_vars_partial_match() {
        // Shares x, but values differ → keep.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":1")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":2")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_extra_vars_in_lhs_do_not_affect() {
        // LHS has extra variable; shared var matches → eliminate.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a"), ("extra", ":e")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_extra_vars_in_rhs_do_not_affect() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a"), ("extra", ":e")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_large_rhs_no_match_keeps_lhs() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":target")])]);
        let rhs_rows: Vec<Binding> = (0..50)
            .map(|i| make_binding(&[("x", &format!(":item{}", i))]))
            .collect();
        let rhs = binding_set(rhs_rows);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_large_rhs_one_match_eliminates() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":item25")])]);
        let rhs_rows: Vec<Binding> = (0..50)
            .map(|i| make_binding(&[("x", &format!(":item{}", i))]))
            .collect();
        let rhs = binding_set(rhs_rows);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_order_preserved() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":c")]),
            make_binding(&[("x", ":d")]),
        ]);
        let rhs = binding_set(vec![make_binding(&[("x", ":b")])]);
        let result = eval.evaluate(&lhs, &rhs);
        let values: Vec<&str> = result.iter().map(|r| r["x"].as_str()).collect();
        assert_eq!(values, vec![":a", ":c", ":d"]);
    }

    #[test]
    fn test_evaluate_multiple_rhs_rows_eliminate_multiple() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":c")]),
        ]);
        let rhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":c")]),
        ]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
        assert_eq!(result.iter().next().map(|r| r["x"].as_str()), Some(":b"));
    }

    // ── evaluate_with_stats ──────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_operands() {
        let eval = MinusEvaluator::new();
        let (result, stats) = eval.evaluate_with_stats(&BindingSet::new(), &BindingSet::new());
        assert_eq!(stats.lhs_count, 0);
        assert_eq!(stats.rhs_count, 0);
        assert_eq!(stats.result_count, 0);
        assert_eq!(stats.eliminated_count, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_stats_nothing_eliminated() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let rhs = binding_set(vec![make_binding(&[("y", ":z")])]);
        let (result, stats) = eval.evaluate_with_stats(&lhs, &rhs);
        assert_eq!(stats.lhs_count, 2);
        assert_eq!(stats.rhs_count, 1);
        assert_eq!(stats.result_count, 2);
        assert_eq!(stats.eliminated_count, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_stats_all_eliminated() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let rhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
        ]);
        let (result, stats) = eval.evaluate_with_stats(&lhs, &rhs);
        assert_eq!(stats.lhs_count, 2);
        assert_eq!(stats.rhs_count, 2);
        assert_eq!(stats.result_count, 0);
        assert_eq!(stats.eliminated_count, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_stats_partial() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":c")]),
        ]);
        let rhs = binding_set(vec![make_binding(&[("x", ":b")])]);
        let (_result, stats) = eval.evaluate_with_stats(&lhs, &rhs);
        assert_eq!(stats.lhs_count, 3);
        assert_eq!(stats.rhs_count, 1);
        assert_eq!(stats.result_count, 2);
        assert_eq!(stats.eliminated_count, 1);
    }

    #[test]
    fn test_stats_count_consistency() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("x", ":a")]),
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":c")]),
            make_binding(&[("x", ":d")]),
        ]);
        let rhs = binding_set(vec![
            make_binding(&[("x", ":b")]),
            make_binding(&[("x", ":d")]),
        ]);
        let (result, stats) = eval.evaluate_with_stats(&lhs, &rhs);
        assert_eq!(stats.result_count + stats.eliminated_count, stats.lhs_count);
        assert_eq!(result.len(), stats.result_count);
    }

    // ── edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_lhs_with_empty_binding() {
        // An empty binding has no variables → not compatible with anything.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![Binding::new()]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_rhs_with_empty_binding() {
        // Empty RHS binding has no variables → not compatible with anything in LHS.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let rhs = binding_set(vec![Binding::new()]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_both_empty_bindings() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![Binding::new()]);
        let rhs = binding_set(vec![Binding::new()]);
        // Empty ∩ Empty = empty shared vars → not compatible → keep.
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_evaluate_three_vars_two_shared_all_agree() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":b"), ("z", ":c")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":b")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_three_vars_two_shared_one_disagrees() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":b"), ("z", ":c")])]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a"), ("y", ":DIFF")])]);
        let result = eval.evaluate(&lhs, &rhs);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_minus_evaluator_default() {
        let eval = MinusEvaluator;
        let lhs = BindingSet::new();
        let rhs = BindingSet::new();
        assert!(eval.evaluate(&lhs, &rhs).is_empty());
    }

    #[test]
    fn test_evaluate_duplicate_lhs_rows() {
        let eval = MinusEvaluator::new();
        let row = make_binding(&[("x", ":a")]);
        let lhs = binding_set(vec![row.clone(), row.clone()]);
        let rhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        // Both duplicate rows should be eliminated.
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_evaluate_rhs_eliminates_via_any_row() {
        // RHS has two rows; only the second matches.
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":match")])]);
        let rhs = binding_set(vec![
            make_binding(&[("x", ":no_match")]),
            make_binding(&[("x", ":match")]),
        ]);
        let result = eval.evaluate(&lhs, &rhs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_shared_vars_large_overlap() {
        let a: Binding = (0..20)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        let b: Binding = (10..30)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        let mut sv = MinusEvaluator::shared_vars(&a, &b);
        sv.sort();
        assert_eq!(sv.len(), 10);
        for i in 10..20usize {
            assert!(sv.contains(&format!("v{}", i)));
        }
    }

    #[test]
    fn test_compatible_large_shared_all_agree() {
        let a: Binding = (0..20)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        let b: Binding = (0..20)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        assert!(MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_large_shared_one_disagrees() {
        let mut a: Binding = (0..20)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        let b: Binding = (0..20)
            .map(|i| (format!("v{}", i), format!(":val{}", i)))
            .collect();
        a.insert("v5".to_string(), ":different".to_string());
        assert!(!MinusEvaluator::compatible(&a, &b));
    }

    #[test]
    fn test_stats_rhs_count_zero() {
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![make_binding(&[("x", ":a")])]);
        let (_, stats) = eval.evaluate_with_stats(&lhs, &BindingSet::new());
        assert_eq!(stats.rhs_count, 0);
        assert_eq!(stats.eliminated_count, 0);
    }

    #[test]
    fn test_binding_set_push_multiple() {
        let mut bs = BindingSet::new();
        for i in 0..10 {
            bs.push(make_binding(&[("x", &format!(":v{}", i))]));
        }
        assert_eq!(bs.len(), 10);
    }

    #[test]
    fn test_evaluate_complex_multi_var_scenario() {
        // Simulates a typical SPARQL MINUS scenario:
        //   LHS: SELECT ?person ?age → 4 people
        //   RHS: SELECT ?person → 2 people to exclude
        let eval = MinusEvaluator::new();
        let lhs = binding_set(vec![
            make_binding(&[("person", ":alice"), ("age", "30")]),
            make_binding(&[("person", ":bob"), ("age", "25")]),
            make_binding(&[("person", ":carol"), ("age", "35")]),
            make_binding(&[("person", ":dave"), ("age", "28")]),
        ]);
        let rhs = binding_set(vec![
            make_binding(&[("person", ":bob")]),
            make_binding(&[("person", ":dave")]),
        ]);
        let (result, stats) = eval.evaluate_with_stats(&lhs, &rhs);
        assert_eq!(stats.result_count, 2);
        assert_eq!(stats.eliminated_count, 2);
        let persons: Vec<&str> = result.iter().map(|r| r["person"].as_str()).collect();
        assert!(persons.contains(&":alice"));
        assert!(persons.contains(&":carol"));
    }
}
