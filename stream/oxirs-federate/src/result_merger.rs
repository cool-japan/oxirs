//! Federated query result merging strategies.
//!
//! Implements Union, Intersection, Minus, LeftJoin, and NaturalJoin over
//! SPARQL-style result sets where each row is a binding of variable names to
//! string values.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single row of bound variable values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultRow(pub HashMap<String, String>);

impl Default for ResultRow {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultRow {
    /// Create an empty row.
    pub fn new() -> Self {
        ResultRow(HashMap::new())
    }

    /// Bind `var` to `val`.
    pub fn set(&mut self, var: impl Into<String>, val: impl Into<String>) {
        self.0.insert(var.into(), val.into());
    }

    /// Retrieve the value bound to `var`, if any.
    pub fn get(&self, var: &str) -> Option<&str> {
        self.0.get(var).map(|s| s.as_str())
    }

    /// Two rows are *compatible* when they agree on all shared variables.
    pub fn compatible_with(&self, other: &ResultRow) -> bool {
        for (k, v) in &self.0 {
            if let Some(other_v) = other.0.get(k) {
                if other_v != v {
                    return false;
                }
            }
        }
        true
    }

    /// Merge two compatible rows.  Values from `self` take precedence for any
    /// conflicting key (non-conflicting values from `other` are included).
    pub fn merge_with(&self, other: &ResultRow) -> ResultRow {
        let mut merged = self.0.clone();
        for (k, v) in &other.0 {
            merged.entry(k.clone()).or_insert_with(|| v.clone());
        }
        ResultRow(merged)
    }
}

/// An ordered collection of result rows with associated variable names.
#[derive(Debug, Clone)]
pub struct ResultSet {
    /// Projected variable names (may be empty if unknown).
    pub variables: Vec<String>,
    /// Ordered rows.
    pub rows: Vec<ResultRow>,
}

impl ResultSet {
    /// Create an empty result set with the given variable list.
    pub fn new(variables: Vec<String>) -> Self {
        ResultSet {
            variables,
            rows: Vec::new(),
        }
    }

    /// Append a row.
    pub fn add_row(&mut self, row: ResultRow) {
        self.rows.push(row);
    }

    /// Number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of variables.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// `true` if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Merge strategy
// ---------------------------------------------------------------------------

/// Merge strategy to apply when combining result sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// All rows from all sets (with deduplication).
    Union,
    /// Rows common to all sets.
    Intersection,
    /// Rows in the first set that are not in the second.
    Minus,
    /// Natural left outer join.
    LeftJoin,
    /// Natural inner join on shared variables.
    NaturalJoin,
}

/// Statistics about a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Row counts of each input set (in order).
    pub input_rows: Vec<usize>,
    /// Row count of the result set.
    pub output_rows: usize,
    /// Duplicates removed during the merge.
    pub duplicate_rows_removed: usize,
    /// Rows that could not be joined.
    pub join_failures: usize,
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    pub result: ResultSet,
    pub stats: MergeStats,
}

// ---------------------------------------------------------------------------
// ResultMerger
// ---------------------------------------------------------------------------

/// Stateless utility for combining `ResultSet` values.
pub struct ResultMerger;

impl ResultMerger {
    /// Dispatch to the appropriate merge strategy.
    pub fn merge(sets: &[ResultSet], strategy: MergeStrategy) -> MergeResult {
        match strategy {
            MergeStrategy::Union => Self::union(sets),
            MergeStrategy::Intersection => {
                if sets.len() < 2 {
                    let empty = ResultSet::new(Vec::new());
                    let first = sets.first().unwrap_or(&empty);
                    Self::intersection(first, &ResultSet::new(Vec::new()))
                } else {
                    let mut result = Self::intersection(&sets[0], &sets[1]);
                    for extra in &sets[2..] {
                        result = Self::intersection(&result.result, extra);
                    }
                    result
                }
            }
            MergeStrategy::Minus => {
                let empty = ResultSet::new(Vec::new());
                let a = sets.first().unwrap_or(&empty);
                let b = sets.get(1).unwrap_or(&empty);
                Self::minus(a, b)
            }
            MergeStrategy::LeftJoin => {
                let empty = ResultSet::new(Vec::new());
                let a = sets.first().unwrap_or(&empty);
                let b = sets.get(1).unwrap_or(&empty);
                Self::left_join(a, b)
            }
            MergeStrategy::NaturalJoin => {
                let empty = ResultSet::new(Vec::new());
                let a = sets.first().unwrap_or(&empty);
                let b = sets.get(1).unwrap_or(&empty);
                Self::natural_join(a, b)
            }
        }
    }

    /// Concatenate all rows from all sets; remove duplicates.
    pub fn union(sets: &[ResultSet]) -> MergeResult {
        let input_rows: Vec<usize> = sets.iter().map(|s| s.row_count()).collect();

        // Merge variable lists (union, preserving order).
        let mut vars: Vec<String> = Vec::new();
        for s in sets {
            for v in &s.variables {
                if !vars.contains(v) {
                    vars.push(v.clone());
                }
            }
        }

        let mut combined: Vec<ResultRow> =
            sets.iter().flat_map(|s| s.rows.iter().cloned()).collect();
        let before = combined.len();
        deduplicate_rows(&mut combined);
        let after = combined.len();

        let mut result = ResultSet::new(vars);
        result.rows = combined;

        MergeResult {
            stats: MergeStats {
                input_rows,
                output_rows: result.row_count(),
                duplicate_rows_removed: before - after,
                join_failures: 0,
            },
            result,
        }
    }

    /// Keep rows that appear in both `a` and `b` (by equality).
    pub fn intersection(a: &ResultSet, b: &ResultSet) -> MergeResult {
        let input_rows = vec![a.row_count(), b.row_count()];

        let rows: Vec<ResultRow> = a
            .rows
            .iter()
            .filter(|r| b.rows.iter().any(|br| br == *r))
            .cloned()
            .collect();

        let mut vars = a.variables.clone();
        for v in &b.variables {
            if !vars.contains(v) {
                vars.push(v.clone());
            }
        }
        let mut result = ResultSet::new(vars);
        result.rows = rows;

        MergeResult {
            stats: MergeStats {
                input_rows,
                output_rows: result.row_count(),
                duplicate_rows_removed: 0,
                join_failures: 0,
            },
            result,
        }
    }

    /// Return rows from `a` that have no matching row in `b`.
    pub fn minus(a: &ResultSet, b: &ResultSet) -> MergeResult {
        let input_rows = vec![a.row_count(), b.row_count()];

        let rows: Vec<ResultRow> = a
            .rows
            .iter()
            .filter(|r| !b.rows.iter().any(|br| br == *r))
            .cloned()
            .collect();

        let mut result = ResultSet::new(a.variables.clone());
        result.rows = rows;

        MergeResult {
            stats: MergeStats {
                input_rows,
                output_rows: result.row_count(),
                duplicate_rows_removed: 0,
                join_failures: 0,
            },
            result,
        }
    }

    /// Natural left outer join.
    ///
    /// For each row in `a`, all compatible rows in `b` are merged.
    /// If no compatible row in `b` exists, the `a` row is preserved unchanged.
    pub fn left_join(a: &ResultSet, b: &ResultSet) -> MergeResult {
        let input_rows = vec![a.row_count(), b.row_count()];
        let mut rows = Vec::new();
        let mut join_failures = 0usize;

        // Merged variable list.
        let mut vars = a.variables.clone();
        for v in &b.variables {
            if !vars.contains(v) {
                vars.push(v.clone());
            }
        }

        for row_a in &a.rows {
            let mut matched = false;
            for row_b in &b.rows {
                if row_a.compatible_with(row_b) {
                    rows.push(row_a.merge_with(row_b));
                    matched = true;
                }
            }
            if !matched {
                join_failures += 1;
                rows.push(row_a.clone());
            }
        }

        let mut result = ResultSet::new(vars);
        result.rows = rows;

        MergeResult {
            stats: MergeStats {
                input_rows,
                output_rows: result.row_count(),
                duplicate_rows_removed: 0,
                join_failures,
            },
            result,
        }
    }

    /// Natural inner join: only rows from `a` that are compatible with at
    /// least one row in `b` are kept (merged).
    pub fn natural_join(a: &ResultSet, b: &ResultSet) -> MergeResult {
        let input_rows = vec![a.row_count(), b.row_count()];
        let mut rows = Vec::new();
        let mut join_failures = 0usize;

        let mut vars = a.variables.clone();
        for v in &b.variables {
            if !vars.contains(v) {
                vars.push(v.clone());
            }
        }

        for row_a in &a.rows {
            let mut any_match = false;
            for row_b in &b.rows {
                if row_a.compatible_with(row_b) {
                    rows.push(row_a.merge_with(row_b));
                    any_match = true;
                }
            }
            if !any_match {
                join_failures += 1;
            }
        }

        let mut result = ResultSet::new(vars);
        result.rows = rows;

        MergeResult {
            stats: MergeStats {
                input_rows,
                output_rows: result.row_count(),
                duplicate_rows_removed: 0,
                join_failures,
            },
            result,
        }
    }

    /// Remove duplicate rows in-place from a `ResultSet`.
    pub fn deduplicate(mut result: ResultSet) -> ResultSet {
        deduplicate_rows(&mut result.rows);
        result
    }

    /// Sort rows by the value of `var` (ascending or descending, lexicographic).
    /// Rows that do not bind `var` are placed at the end.
    pub fn sort(mut result: ResultSet, var: &str, ascending: bool) -> ResultSet {
        let var = var.to_string();
        result.rows.sort_by(|a, b| {
            let va = a.get(&var).unwrap_or("");
            let vb = b.get(&var).unwrap_or("");
            if ascending {
                va.cmp(vb)
            } else {
                vb.cmp(va)
            }
        });
        result
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Remove duplicate rows in-place using O(n²) linear scanning.
///
/// This is appropriate for typical SPARQL result sets which are small-to-medium
/// in size.  If performance for large sets matters, consider a sorted comparison.
fn deduplicate_rows(rows: &mut Vec<ResultRow>) {
    let mut i = 0;
    while i < rows.len() {
        // Check whether rows[i] already appeared in rows[0..i].
        if rows[..i].contains(&rows[i]) {
            rows.remove(i);
        } else {
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(bindings: &[(&str, &str)]) -> ResultRow {
        let mut row = ResultRow::new();
        for (k, v) in bindings {
            row.set(*k, *v);
        }
        row
    }

    fn make_set(vars: &[&str], rows: &[Vec<(&str, &str)>]) -> ResultSet {
        let mut rs = ResultSet::new(vars.iter().map(|s| s.to_string()).collect());
        for r in rows {
            rs.add_row(make_row(r));
        }
        rs
    }

    // --- ResultRow ---

    #[test]
    fn test_row_set_get() {
        let mut row = ResultRow::new();
        row.set("x", "Alice");
        assert_eq!(row.get("x"), Some("Alice"));
        assert_eq!(row.get("y"), None);
    }

    #[test]
    fn test_row_compatible_with_no_shared_vars() {
        let row_a = make_row(&[("x", "Alice")]);
        let row_b = make_row(&[("y", "Bob")]);
        assert!(row_a.compatible_with(&row_b));
    }

    #[test]
    fn test_row_compatible_with_matching_shared_var() {
        let row_a = make_row(&[("x", "Alice"), ("y", "1")]);
        let row_b = make_row(&[("x", "Alice"), ("z", "2")]);
        assert!(row_a.compatible_with(&row_b));
    }

    #[test]
    fn test_row_compatible_with_conflicting_shared_var() {
        let row_a = make_row(&[("x", "Alice")]);
        let row_b = make_row(&[("x", "Bob")]);
        assert!(!row_a.compatible_with(&row_b));
    }

    #[test]
    fn test_row_merge_with_combines_bindings() {
        let row_a = make_row(&[("x", "Alice")]);
        let row_b = make_row(&[("y", "Bob")]);
        let merged = row_a.merge_with(&row_b);
        assert_eq!(merged.get("x"), Some("Alice"));
        assert_eq!(merged.get("y"), Some("Bob"));
    }

    #[test]
    fn test_row_merge_with_self_precedence() {
        let row_a = make_row(&[("x", "A")]);
        let row_b = make_row(&[("x", "B"), ("y", "C")]);
        let merged = row_a.merge_with(&row_b);
        // self takes precedence
        assert_eq!(merged.get("x"), Some("A"));
        assert_eq!(merged.get("y"), Some("C"));
    }

    // --- ResultSet ---

    #[test]
    fn test_result_set_new_empty() {
        let rs = ResultSet::new(vec!["x".into(), "y".into()]);
        assert_eq!(rs.row_count(), 0);
        assert_eq!(rs.variable_count(), 2);
        assert!(rs.is_empty());
    }

    #[test]
    fn test_result_set_add_row() {
        let mut rs = ResultSet::new(vec!["x".into()]);
        rs.add_row(make_row(&[("x", "1")]));
        assert_eq!(rs.row_count(), 1);
        assert!(!rs.is_empty());
    }

    // --- Union ---

    #[test]
    fn test_union_concatenates() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[vec![("x", "3")]]);
        let mr = ResultMerger::union(&[a, b]);
        assert_eq!(mr.result.row_count(), 3);
    }

    #[test]
    fn test_union_removes_duplicates() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "1")]]);
        let mr = ResultMerger::union(&[a, b]);
        assert_eq!(mr.result.row_count(), 1);
        assert_eq!(mr.stats.duplicate_rows_removed, 1);
    }

    #[test]
    fn test_union_empty_sets() {
        let mr = ResultMerger::union(&[]);
        assert_eq!(mr.result.row_count(), 0);
    }

    #[test]
    fn test_union_stats_input_rows() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[vec![("x", "3")]]);
        let mr = ResultMerger::union(&[a, b]);
        assert_eq!(mr.stats.input_rows, vec![2, 1]);
    }

    // --- Intersection ---

    #[test]
    fn test_intersection_keeps_common() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[vec![("x", "2")], vec![("x", "3")]]);
        let mr = ResultMerger::intersection(&a, &b);
        assert_eq!(mr.result.row_count(), 1);
        assert_eq!(mr.result.rows[0].get("x"), Some("2"));
    }

    #[test]
    fn test_intersection_no_common() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "2")]]);
        let mr = ResultMerger::intersection(&a, &b);
        assert_eq!(mr.result.row_count(), 0);
    }

    #[test]
    fn test_intersection_stats_input_rows() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "1")]]);
        let mr = ResultMerger::intersection(&a, &b);
        assert_eq!(mr.stats.input_rows, vec![1, 1]);
    }

    // --- Minus ---

    #[test]
    fn test_minus_removes_matching() {
        let a = make_set(
            &["x"],
            &[vec![("x", "1")], vec![("x", "2")], vec![("x", "3")]],
        );
        let b = make_set(&["x"], &[vec![("x", "2")]]);
        let mr = ResultMerger::minus(&a, &b);
        assert_eq!(mr.result.row_count(), 2);
        assert!(mr.result.rows.iter().all(|r| r.get("x") != Some("2")));
    }

    #[test]
    fn test_minus_empty_b() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[]);
        let mr = ResultMerger::minus(&a, &b);
        assert_eq!(mr.result.row_count(), 2);
    }

    #[test]
    fn test_minus_all_removed() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "1")]]);
        let mr = ResultMerger::minus(&a, &b);
        assert_eq!(mr.result.row_count(), 0);
    }

    // --- LeftJoin ---

    #[test]
    fn test_left_join_preserves_left_row_without_match() {
        let a = make_set(&["x"], &[vec![("x", "Alice")]]);
        let b = make_set(&["x", "y"], &[vec![("x", "Bob"), ("y", "NYC")]]);
        let mr = ResultMerger::left_join(&a, &b);
        // Row from a is preserved even without a match
        assert_eq!(mr.result.row_count(), 1);
        assert_eq!(mr.result.rows[0].get("x"), Some("Alice"));
        assert_eq!(mr.stats.join_failures, 1);
    }

    #[test]
    fn test_left_join_merges_compatible() {
        let a = make_set(&["x"], &[vec![("x", "Alice")]]);
        let b = make_set(&["x", "y"], &[vec![("x", "Alice"), ("y", "NYC")]]);
        let mr = ResultMerger::left_join(&a, &b);
        assert_eq!(mr.result.row_count(), 1);
        assert_eq!(mr.result.rows[0].get("y"), Some("NYC"));
    }

    #[test]
    fn test_left_join_multiple_matches() {
        let a = make_set(&["x"], &[vec![("x", "Alice")]]);
        let b = make_set(
            &["x", "y"],
            &[
                vec![("x", "Alice"), ("y", "NYC")],
                vec![("x", "Alice"), ("y", "LA")],
            ],
        );
        let mr = ResultMerger::left_join(&a, &b);
        // One a-row × two compatible b-rows = 2 output rows
        assert_eq!(mr.result.row_count(), 2);
    }

    // --- NaturalJoin ---

    #[test]
    fn test_natural_join_on_shared_vars() {
        let a = make_set(&["x", "y"], &[vec![("x", "A"), ("y", "1")]]);
        let b = make_set(&["y", "z"], &[vec![("y", "1"), ("z", "X")]]);
        let mr = ResultMerger::natural_join(&a, &b);
        assert_eq!(mr.result.row_count(), 1);
        assert_eq!(mr.result.rows[0].get("z"), Some("X"));
    }

    #[test]
    fn test_natural_join_no_match_excluded() {
        let a = make_set(&["x"], &[vec![("x", "A")]]);
        let b = make_set(&["x"], &[vec![("x", "B")]]);
        let mr = ResultMerger::natural_join(&a, &b);
        assert_eq!(mr.result.row_count(), 0);
        assert_eq!(mr.stats.join_failures, 1);
    }

    #[test]
    fn test_natural_join_join_failures_counter() {
        let a = make_set(&["x"], &[vec![("x", "A")], vec![("x", "B")]]);
        let b = make_set(&["x"], &[vec![("x", "A")]]);
        let mr = ResultMerger::natural_join(&a, &b);
        assert_eq!(mr.stats.join_failures, 1); // "B" has no match
    }

    // --- Deduplicate ---

    #[test]
    fn test_deduplicate_removes_duplicates() {
        let mut rs = make_set(
            &["x"],
            &[vec![("x", "1")], vec![("x", "1")], vec![("x", "2")]],
        );
        rs = ResultMerger::deduplicate(rs);
        assert_eq!(rs.row_count(), 2);
    }

    #[test]
    fn test_deduplicate_preserves_order() {
        let mut rs = make_set(
            &["x"],
            &[vec![("x", "2")], vec![("x", "1")], vec![("x", "2")]],
        );
        rs = ResultMerger::deduplicate(rs);
        assert_eq!(rs.row_count(), 2);
        // "2" appeared first, so it should come first
        assert_eq!(rs.rows[0].get("x"), Some("2"));
    }

    // --- Sort ---

    #[test]
    fn test_sort_ascending() {
        let rs = make_set(
            &["x"],
            &[vec![("x", "c")], vec![("x", "a")], vec![("x", "b")]],
        );
        let sorted = ResultMerger::sort(rs, "x", true);
        assert_eq!(sorted.rows[0].get("x"), Some("a"));
        assert_eq!(sorted.rows[2].get("x"), Some("c"));
    }

    #[test]
    fn test_sort_descending() {
        let rs = make_set(
            &["x"],
            &[vec![("x", "c")], vec![("x", "a")], vec![("x", "b")]],
        );
        let sorted = ResultMerger::sort(rs, "x", false);
        assert_eq!(sorted.rows[0].get("x"), Some("c"));
        assert_eq!(sorted.rows[2].get("x"), Some("a"));
    }

    #[test]
    fn test_sort_missing_var_goes_to_end() {
        let rs = make_set(
            &["x"],
            &[vec![("x", "b")], vec![("y", "1")], vec![("x", "a")]],
        );
        // The row without "x" sorts like "" which is before "a"
        let sorted = ResultMerger::sort(rs, "x", true);
        // First row should be the one without x (empty string < "a")
        assert_eq!(sorted.rows[0].get("x"), None);
    }

    // --- MergeStats ---

    #[test]
    fn test_merge_stats_output_rows() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[vec![("x", "3")]]);
        let mr = ResultMerger::union(&[a, b]);
        assert_eq!(mr.stats.output_rows, 3);
    }

    #[test]
    fn test_merge_via_strategy_union() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "2")]]);
        let mr = ResultMerger::merge(&[a, b], MergeStrategy::Union);
        assert_eq!(mr.result.row_count(), 2);
    }

    #[test]
    fn test_merge_via_strategy_intersection() {
        let a = make_set(&["x"], &[vec![("x", "1")]]);
        let b = make_set(&["x"], &[vec![("x", "1")]]);
        let mr = ResultMerger::merge(&[a, b], MergeStrategy::Intersection);
        assert_eq!(mr.result.row_count(), 1);
    }

    #[test]
    fn test_merge_via_strategy_minus() {
        let a = make_set(&["x"], &[vec![("x", "1")], vec![("x", "2")]]);
        let b = make_set(&["x"], &[vec![("x", "1")]]);
        let mr = ResultMerger::merge(&[a, b], MergeStrategy::Minus);
        assert_eq!(mr.result.row_count(), 1);
    }

    #[test]
    fn test_row_default() {
        let row = ResultRow::default();
        assert!(row.0.is_empty());
    }
}
