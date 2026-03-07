/// SPARQL 1.1 VALUES clause (inline data) implementation.
///
/// The VALUES clause provides a mechanism for injecting a set of bindings
/// directly into a query without needing to retrieve them from a dataset.
use std::collections::HashMap;

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur when working with a VALUES clause.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValuesError {
    /// The number of values in a row did not match the number of declared variables.
    #[error("column count mismatch: expected {expected}, got {got}")]
    ColumnCountMismatch { expected: usize, got: usize },
}

// ── Core data structures ──────────────────────────────────────────────────────

/// A single row of VALUES data.  `None` represents the `UNDEF` keyword.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValuesRow(pub Vec<Option<String>>);

impl ValuesRow {
    /// Create a new row from a vec of optional values.
    pub fn new(values: Vec<Option<String>>) -> Self {
        Self(values)
    }

    /// Return the value at position `idx`, or `None` if out of range.
    pub fn get(&self, idx: usize) -> Option<&Option<String>> {
        self.0.get(idx)
    }

    /// Number of values (columns) in the row.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the row contains no values.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` when the value at position `idx` is `UNDEF` (i.e. `None`).
    pub fn is_undef(&self, idx: usize) -> bool {
        matches!(self.0.get(idx), Some(None))
    }
}

/// A single variable binding produced by expanding a VALUES row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingResult {
    /// Variable name (without the `?` sigil).
    pub var: String,
    /// Bound value, or `None` for UNDEF.
    pub value: Option<String>,
}

/// The result of expanding all rows in a VALUES clause into per-variable bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValuesExpansion {
    /// One entry per row; each inner `Vec` has one `BindingResult` per variable.
    pub rows: Vec<Vec<BindingResult>>,
}

// ── ValuesClause ──────────────────────────────────────────────────────────────

/// Represents a `VALUES` clause from a SPARQL 1.1 query.
///
/// ```sparql
/// VALUES (?x ?y) {
///   ("a" "b")
///   ("c" UNDEF)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValuesClause {
    /// The declared variable names (without the `?` sigil).
    pub variables: Vec<String>,
    /// The rows of data.
    pub rows: Vec<ValuesRow>,
}

impl ValuesClause {
    /// Create an empty VALUES clause with the given variable list.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            rows: Vec::new(),
        }
    }

    /// Append a row to the clause, validating the column count.
    pub fn add_row(&mut self, row: ValuesRow) -> Result<(), ValuesError> {
        let expected = self.variables.len();
        let got = row.len();
        if got != expected {
            return Err(ValuesError::ColumnCountMismatch { expected, got });
        }
        self.rows.push(row);
        Ok(())
    }

    /// Expand all rows into a flat `ValuesExpansion` (each row → set of `BindingResult`s).
    pub fn expand(&self) -> ValuesExpansion {
        let rows = self
            .rows
            .iter()
            .map(|row| {
                self.variables
                    .iter()
                    .enumerate()
                    .map(|(i, var)| BindingResult {
                        var: var.clone(),
                        value: row.0.get(i).and_then(|v| v.clone()),
                    })
                    .collect()
            })
            .collect();
        ValuesExpansion { rows }
    }

    /// Perform a VALUES join: for each existing binding map, try to extend it
    /// with each VALUES row.  Rows whose defined values conflict with the
    /// existing bindings are dropped.  UNDEF values never conflict.
    pub fn join_with(&self, bindings: &[HashMap<String, String>]) -> Vec<HashMap<String, String>> {
        if bindings.is_empty() {
            // With no incoming bindings the VALUES rows become the full solution.
            return self.expand_as_maps();
        }
        let mut out = Vec::new();
        for binding in bindings {
            for row in &self.rows {
                if let Some(merged) = self.try_merge(binding, row) {
                    out.push(merged);
                }
            }
        }
        out
    }

    /// Number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of declared variables.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Returns `true` when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Return a new `ValuesClause` with rows removed where every value is `UNDEF`.
    pub fn filter_undef(&self) -> Self {
        let rows = self
            .rows
            .iter()
            .filter(|row| row.0.iter().any(|v| v.is_some()))
            .cloned()
            .collect();
        Self {
            variables: self.variables.clone(),
            rows,
        }
    }

    /// Return a new `ValuesClause` keeping only the variables listed in `vars`.
    /// The column order follows the order of `vars`.
    pub fn project(&self, vars: &[&str]) -> Self {
        // Build index map: var name → original column index
        let indices: Vec<usize> = vars
            .iter()
            .filter_map(|v| self.variables.iter().position(|x| x == v))
            .collect();
        let new_variables: Vec<String> =
            indices.iter().map(|&i| self.variables[i].clone()).collect();
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let values = indices
                    .iter()
                    .map(|&i| row.0.get(i).and_then(|v| v.clone()))
                    .collect();
                ValuesRow(values)
            })
            .collect();
        Self {
            variables: new_variables,
            rows,
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Expand all rows as `HashMap<variable, value>` (UNDEF omitted).
    fn expand_as_maps(&self) -> Vec<HashMap<String, String>> {
        self.rows
            .iter()
            .map(|row| {
                self.variables
                    .iter()
                    .enumerate()
                    .filter_map(|(i, var)| {
                        row.0
                            .get(i)
                            .and_then(|v| v.as_ref())
                            .map(|v| (var.clone(), v.clone()))
                    })
                    .collect()
            })
            .collect()
    }

    /// Try to merge an incoming `binding` with a VALUES `row`.
    ///
    /// Returns `None` if there is a conflict (same variable, different value).
    /// UNDEF (`None`) in the row is treated as compatible with any binding.
    fn try_merge(
        &self,
        binding: &HashMap<String, String>,
        row: &ValuesRow,
    ) -> Option<HashMap<String, String>> {
        let mut merged = binding.clone();
        for (i, var) in self.variables.iter().enumerate() {
            if let Some(Some(val)) = row.0.get(i) {
                if let Some(existing) = merged.get(var) {
                    if existing != val {
                        return None; // conflict
                    }
                } else {
                    merged.insert(var.clone(), val.clone());
                }
            }
        }
        Some(merged)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(values: &[Option<&str>]) -> ValuesRow {
        ValuesRow::new(values.iter().map(|v| v.map(String::from)).collect())
    }

    // ── ValuesRow tests ────────────────────────────────────────────────────────

    #[test]
    fn test_row_new_and_len() {
        let r = make_row(&[Some("a"), None, Some("c")]);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_row_is_empty_false() {
        let r = make_row(&[Some("x")]);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_row_is_empty_true() {
        let r = ValuesRow::new(vec![]);
        assert!(r.is_empty());
    }

    #[test]
    fn test_row_get_some() {
        let r = make_row(&[Some("hello"), None]);
        assert_eq!(r.get(0), Some(&Some("hello".to_string())));
    }

    #[test]
    fn test_row_get_none_value() {
        let r = make_row(&[None]);
        assert_eq!(r.get(0), Some(&None));
    }

    #[test]
    fn test_row_get_out_of_range() {
        let r = make_row(&[Some("x")]);
        assert_eq!(r.get(5), None);
    }

    #[test]
    fn test_row_is_undef_true() {
        let r = make_row(&[Some("v"), None]);
        assert!(r.is_undef(1));
    }

    #[test]
    fn test_row_is_undef_false() {
        let r = make_row(&[Some("v"), None]);
        assert!(!r.is_undef(0));
    }

    #[test]
    fn test_row_is_undef_out_of_range() {
        let r = make_row(&[Some("v")]);
        assert!(!r.is_undef(99));
    }

    // ── ValuesClause construction tests ───────────────────────────────────────

    #[test]
    fn test_new_empty_clause() {
        let vc = ValuesClause::new(vec!["x".into()]);
        assert_eq!(vc.variable_count(), 1);
        assert!(vc.is_empty());
    }

    #[test]
    fn test_add_row_success() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        assert!(vc.add_row(make_row(&[Some("1"), Some("2")])).is_ok());
        assert_eq!(vc.row_count(), 1);
    }

    #[test]
    fn test_add_row_column_mismatch_too_few() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        let err = vc.add_row(make_row(&[Some("1")])).unwrap_err();
        assert!(matches!(
            err,
            ValuesError::ColumnCountMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn test_add_row_column_mismatch_too_many() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        let err = vc.add_row(make_row(&[Some("1"), Some("2")])).unwrap_err();
        assert!(matches!(
            err,
            ValuesError::ColumnCountMismatch {
                expected: 1,
                got: 2
            }
        ));
    }

    #[test]
    fn test_add_multiple_rows() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("a")])).unwrap();
        vc.add_row(make_row(&[Some("b")])).unwrap();
        vc.add_row(make_row(&[None])).unwrap();
        assert_eq!(vc.row_count(), 3);
    }

    #[test]
    fn test_variable_count() {
        let vc = ValuesClause::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(vc.variable_count(), 3);
    }

    #[test]
    fn test_row_count_zero() {
        let vc = ValuesClause::new(vec!["x".into()]);
        assert_eq!(vc.row_count(), 0);
    }

    // ── expand tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_expand_single_variable() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("hello")])).unwrap();
        let exp = vc.expand();
        assert_eq!(exp.rows.len(), 1);
        assert_eq!(exp.rows[0][0].var, "x");
        assert_eq!(exp.rows[0][0].value, Some("hello".to_string()));
    }

    #[test]
    fn test_expand_multiple_variables() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), Some("b")])).unwrap();
        let exp = vc.expand();
        assert_eq!(exp.rows[0][0].var, "x");
        assert_eq!(exp.rows[0][1].var, "y");
        assert_eq!(exp.rows[0][1].value, Some("b".to_string()));
    }

    #[test]
    fn test_expand_undef_value() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), None])).unwrap();
        let exp = vc.expand();
        assert_eq!(exp.rows[0][1].value, None);
    }

    #[test]
    fn test_expand_multiple_rows() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("1")])).unwrap();
        vc.add_row(make_row(&[Some("2")])).unwrap();
        let exp = vc.expand();
        assert_eq!(exp.rows.len(), 2);
        assert_eq!(exp.rows[1][0].value, Some("2".to_string()));
    }

    #[test]
    fn test_expand_empty_clause() {
        let vc = ValuesClause::new(vec!["x".into()]);
        let exp = vc.expand();
        assert!(exp.rows.is_empty());
    }

    // ── join_with tests ───────────────────────────────────────────────────────

    #[test]
    fn test_join_with_empty_bindings_returns_values_rows() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("1")])).unwrap();
        vc.add_row(make_row(&[Some("2")])).unwrap();
        let result = vc.join_with(&[]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_join_with_compatible_bindings() {
        let mut vc = ValuesClause::new(vec!["y".into()]);
        vc.add_row(make_row(&[Some("b")])).unwrap();

        let bindings = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "a".to_string());
            m
        }];

        let result = vc.join_with(&bindings);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x").map(String::as_str), Some("a"));
        assert_eq!(result[0].get("y").map(String::as_str), Some("b"));
    }

    #[test]
    fn test_join_with_conflict_drops_row() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("wrong")])).unwrap();

        let bindings = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "correct".to_string());
            m
        }];

        let result = vc.join_with(&bindings);
        assert!(result.is_empty());
    }

    #[test]
    fn test_join_with_undef_is_compatible() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[None])).unwrap(); // UNDEF

        let bindings = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "anything".to_string());
            m
        }];

        let result = vc.join_with(&bindings);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x").map(String::as_str), Some("anything"));
    }

    #[test]
    fn test_join_with_same_value_no_conflict() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("same")])).unwrap();

        let bindings = vec![{
            let mut m = HashMap::new();
            m.insert("x".to_string(), "same".to_string());
            m
        }];

        let result = vc.join_with(&bindings);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_join_cross_product_multiple_rows_multiple_bindings() {
        let mut vc = ValuesClause::new(vec!["y".into()]);
        vc.add_row(make_row(&[Some("1")])).unwrap();
        vc.add_row(make_row(&[Some("2")])).unwrap();

        let bindings: Vec<HashMap<String, String>> = vec![
            {
                let mut m = HashMap::new();
                m.insert("x".to_string(), "a".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("x".to_string(), "b".to_string());
                m
            },
        ];

        let result = vc.join_with(&bindings);
        assert_eq!(result.len(), 4); // 2 bindings × 2 rows
    }

    // ── filter_undef tests ────────────────────────────────────────────────────

    #[test]
    fn test_filter_undef_removes_all_undef_row() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), Some("b")])).unwrap();
        vc.add_row(make_row(&[None, None])).unwrap();
        let filtered = vc.filter_undef();
        assert_eq!(filtered.row_count(), 1);
    }

    #[test]
    fn test_filter_undef_keeps_partial_undef() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), None])).unwrap(); // partial UNDEF → kept
        let filtered = vc.filter_undef();
        assert_eq!(filtered.row_count(), 1);
    }

    #[test]
    fn test_filter_undef_empty_clause() {
        let vc = ValuesClause::new(vec!["x".into()]);
        let filtered = vc.filter_undef();
        assert!(filtered.is_empty());
    }

    // ── project tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_project_single_variable() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), Some("b")])).unwrap();
        let projected = vc.project(&["x"]);
        assert_eq!(projected.variable_count(), 1);
        assert_eq!(projected.variables[0], "x");
        assert_eq!(projected.rows[0].0[0], Some("a".to_string()));
    }

    #[test]
    fn test_project_all_variables() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), Some("b")])).unwrap();
        let projected = vc.project(&["x", "y"]);
        assert_eq!(projected.variable_count(), 2);
    }

    #[test]
    fn test_project_reorder_variables() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into(), "z".into()]);
        vc.add_row(make_row(&[Some("a"), Some("b"), Some("c")]))
            .unwrap();
        let projected = vc.project(&["z", "x"]);
        assert_eq!(projected.variables[0], "z");
        assert_eq!(projected.variables[1], "x");
        assert_eq!(projected.rows[0].0[0], Some("c".to_string()));
        assert_eq!(projected.rows[0].0[1], Some("a".to_string()));
    }

    #[test]
    fn test_project_unknown_variable_omitted() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[Some("a")])).unwrap();
        let projected = vc.project(&["unknown"]);
        assert_eq!(projected.variable_count(), 0);
    }

    #[test]
    fn test_project_preserves_undef() {
        let mut vc = ValuesClause::new(vec!["x".into(), "y".into()]);
        vc.add_row(make_row(&[Some("a"), None])).unwrap();
        let projected = vc.project(&["y"]);
        assert_eq!(projected.rows[0].0[0], None);
    }

    // ── edge-case / integration tests ─────────────────────────────────────────

    #[test]
    fn test_values_error_display() {
        let e = ValuesError::ColumnCountMismatch {
            expected: 3,
            got: 1,
        };
        let msg = e.to_string();
        assert!(msg.contains("3"));
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_single_variable_full_cycle() {
        let mut vc = ValuesClause::new(vec!["name".into()]);
        vc.add_row(make_row(&[Some("Alice")])).unwrap();
        vc.add_row(make_row(&[Some("Bob")])).unwrap();
        vc.add_row(make_row(&[None])).unwrap(); // UNDEF

        assert_eq!(vc.row_count(), 3);
        let exp = vc.expand();
        assert_eq!(exp.rows[2][0].value, None);

        let filtered = vc.filter_undef();
        assert_eq!(filtered.row_count(), 2);

        let projected = vc.project(&["name"]);
        assert_eq!(projected.variable_count(), 1);
    }

    #[test]
    fn test_join_with_no_variable_overlap_full_cross_product() {
        let mut vc = ValuesClause::new(vec!["b".into()]);
        vc.add_row(make_row(&[Some("x")])).unwrap();
        vc.add_row(make_row(&[Some("y")])).unwrap();

        let bindings = vec![{
            let mut m = HashMap::new();
            m.insert("a".to_string(), "1".to_string());
            m
        }];

        let result = vc.join_with(&bindings);
        // 1 binding × 2 rows = 2 merged rows
        assert_eq!(result.len(), 2);
        for r in &result {
            assert!(r.contains_key("a"));
            assert!(r.contains_key("b"));
        }
    }

    #[test]
    fn test_all_undef_row_filtered() {
        let mut vc = ValuesClause::new(vec!["x".into()]);
        vc.add_row(make_row(&[None])).unwrap();
        let filtered = vc.filter_undef();
        assert!(filtered.is_empty());
    }
}
