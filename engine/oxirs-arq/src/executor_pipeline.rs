//! Parallel Query Execution Pipeline
//!
//! This module provides parallel execution primitives for SPARQL sub-queries:
//!
//! - [`BindingMap`] — a single row of query result bindings (variable → value)
//! - [`PipelineStage`] — a map or filter transformation on binding rows
//! - [`ParallelPipelineStage`] — a composable sequence of pipeline stages
//! - [`UnionParallelExecutor`] — executes union branches in parallel using rayon
//!
//! # Design
//!
//! The pipeline model allows complex query execution plans to be expressed as
//! chains of transformations.  `ParallelPipelineStage::chain` runs multiple
//! independent pipelines and collects their outputs.  `UnionParallelExecutor`
//! merges results from parallel union branches and optionally handles LEFT JOIN
//! (OPTIONAL) semantics.

use rayon::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BindingMap
// ---------------------------------------------------------------------------

/// A single row of SPARQL query result bindings: variable name → value string.
///
/// This is a thin newtype over `HashMap<String, String>` that provides
/// convenient construction and merging helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingMap(pub HashMap<String, String>);

impl BindingMap {
    /// Create an empty binding map.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Create a binding map from an iterator of `(variable, value)` pairs.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self(
            pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        )
    }

    /// Bind a variable to a value.
    pub fn bind(&mut self, variable: impl Into<String>, value: impl Into<String>) {
        self.0.insert(variable.into(), value.into());
    }

    /// Get the value bound to `variable`, if any.
    pub fn get(&self, variable: &str) -> Option<&str> {
        self.0.get(variable).map(|s| s.as_str())
    }

    /// Number of bound variables.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no variables are bound.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Merge another binding map into this one.  Values from `other` override.
    pub fn merge(&mut self, other: &BindingMap) {
        for (k, v) in &other.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }

    /// Return a new binding map that is the left-outer-join result of `self`
    /// extended with compatible bindings from `other`.
    ///
    /// Two binding maps are *compatible* if they agree on all variables they
    /// share.  When compatible, the merged map contains all bindings from both.
    pub fn compatible_merge(&self, other: &BindingMap) -> Option<BindingMap> {
        // Check compatibility: shared variables must have the same value.
        for (k, v) in &other.0 {
            if let Some(existing) = self.0.get(k) {
                if existing != v {
                    return None;
                }
            }
        }
        let mut merged = self.clone();
        for (k, v) in &other.0 {
            merged.0.insert(k.clone(), v.clone());
        }
        Some(merged)
    }

    /// Canonical string representation for deduplication.
    pub fn canonical_key(&self) -> String {
        let mut pairs: Vec<(&str, &str)> = self
            .0
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        pairs.sort_unstable();
        pairs
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }
}

impl Default for BindingMap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BindingMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{}}}", self.canonical_key())
    }
}

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// A single transformation stage in a query execution pipeline.
pub enum PipelineStage {
    /// Map each binding row to an optional new row (None = remove from stream).
    Map(Box<dyn Fn(BindingMap) -> Option<BindingMap> + Send + Sync>),
    /// Filter binding rows: only rows where the predicate returns `true` pass.
    Filter(Box<dyn Fn(&BindingMap) -> bool + Send + Sync>),
}

impl PipelineStage {
    /// Apply this stage to a single binding map.  Returns `None` if the row
    /// is filtered out.
    pub fn apply(&self, row: BindingMap) -> Option<BindingMap> {
        match self {
            PipelineStage::Map(f) => f(row),
            PipelineStage::Filter(pred) => {
                if pred(&row) {
                    Some(row)
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelPipelineStage
// ---------------------------------------------------------------------------

/// A composable, sequential pipeline of [`PipelineStage`] transformations.
///
/// Stages are applied left-to-right.  Use `map_stage` and `filter_stage`
/// as constructors, then `add_map` / `add_filter` to extend the pipeline.
/// Use `process` to run the pipeline on a batch of inputs.
///
/// Multiple pipelines can be composed with `chain`.
pub struct ParallelPipelineStage {
    stages: Vec<PipelineStage>,
}

impl ParallelPipelineStage {
    /// Create a pipeline starting with a single map stage.
    pub fn map_stage(f: impl Fn(BindingMap) -> Option<BindingMap> + Send + Sync + 'static) -> Self {
        Self {
            stages: vec![PipelineStage::Map(Box::new(f))],
        }
    }

    /// Create a pipeline starting with a single filter stage.
    pub fn filter_stage(pred: impl Fn(&BindingMap) -> bool + Send + Sync + 'static) -> Self {
        Self {
            stages: vec![PipelineStage::Filter(Box::new(pred))],
        }
    }

    /// Append a map stage to this pipeline.
    pub fn add_map(
        mut self,
        f: impl Fn(BindingMap) -> Option<BindingMap> + Send + Sync + 'static,
    ) -> Self {
        self.stages.push(PipelineStage::Map(Box::new(f)));
        self
    }

    /// Append a filter stage to this pipeline.
    pub fn add_filter(
        mut self,
        pred: impl Fn(&BindingMap) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.stages.push(PipelineStage::Filter(Box::new(pred)));
        self
    }

    /// Apply all stages sequentially to every input row and collect the results.
    pub fn process(&self, inputs: Vec<BindingMap>) -> Vec<BindingMap> {
        inputs
            .into_iter()
            .filter_map(|mut row| {
                for stage in &self.stages {
                    row = stage.apply(row)?;
                }
                Some(row)
            })
            .collect()
    }

    /// Run multiple independent pipelines over the same `inputs` and collect
    /// all outputs (concatenation, not deduplication).
    pub fn chain(stages: Vec<Self>, inputs: Vec<BindingMap>) -> Vec<BindingMap> {
        stages
            .into_iter()
            .flat_map(|pipeline| pipeline.process(inputs.clone()))
            .collect()
    }

    /// Return the number of stages in this pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

// ---------------------------------------------------------------------------
// UnionParallelExecutor
// ---------------------------------------------------------------------------

/// Executes independent SPARQL union branches in parallel and merges results.
///
/// Uses rayon for data-parallelism across branches.  Deduplication uses the
/// canonical string key of each `BindingMap`.
pub struct UnionParallelExecutor;

impl UnionParallelExecutor {
    /// Execute multiple pre-computed branches (each a `Vec<BindingMap>`) in
    /// parallel and merge the results.
    ///
    /// Duplicates are removed using the canonical string representation of each
    /// `BindingMap` as the deduplication key.
    pub fn execute_branches(branches: Vec<Vec<BindingMap>>) -> Vec<BindingMap> {
        if branches.is_empty() {
            return vec![];
        }
        if branches.len() == 1 {
            return branches.into_iter().next().unwrap_or_default();
        }

        // Merge all branches in parallel.
        let merged: Vec<BindingMap> = branches
            .into_par_iter()
            .flat_map(|branch| branch.into_par_iter())
            .collect();

        Self::dedup(merged)
    }

    /// Compute the LEFT OUTER JOIN (OPTIONAL) of `main` rows with `optional` rows.
    ///
    /// For each row in `main`:
    /// - If there is at least one compatible row in `optional`, emit all compatible
    ///   merged rows.
    /// - Otherwise, emit the `main` row unchanged (preserving LEFT JOIN semantics).
    pub fn execute_optional(main: Vec<BindingMap>, optional: Vec<BindingMap>) -> Vec<BindingMap> {
        if optional.is_empty() {
            return main;
        }

        main.into_par_iter()
            .flat_map(|main_row| {
                let compatible: Vec<BindingMap> = optional
                    .iter()
                    .filter_map(|opt_row| main_row.compatible_merge(opt_row))
                    .collect();
                if compatible.is_empty() {
                    vec![main_row]
                } else {
                    compatible
                }
            })
            .collect()
    }

    /// Remove duplicate binding maps by canonical key.
    pub fn dedup(rows: Vec<BindingMap>) -> Vec<BindingMap> {
        let mut seen = std::collections::HashSet::new();
        rows.into_iter()
            .filter(|row| seen.insert(row.canonical_key()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(pairs: &[(&str, &str)]) -> BindingMap {
        BindingMap::from_pairs(pairs.iter().map(|&(k, v)| (k, v)))
    }

    // ------------------------------------------------------------------
    // BindingMap tests
    // ------------------------------------------------------------------

    #[test]
    fn test_binding_map_new_empty() {
        let m = BindingMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_binding_map_from_pairs() {
        let m = bm(&[("s", "http://ex.org/s"), ("p", "http://ex.org/p")]);
        assert_eq!(m.get("s"), Some("http://ex.org/s"));
        assert_eq!(m.get("p"), Some("http://ex.org/p"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_binding_map_bind() {
        let mut m = BindingMap::new();
        m.bind("x", "value");
        assert_eq!(m.get("x"), Some("value"));
    }

    #[test]
    fn test_binding_map_get_missing() {
        let m = BindingMap::new();
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn test_binding_map_merge_override() {
        let mut m1 = bm(&[("a", "1"), ("b", "2")]);
        let m2 = bm(&[("b", "override"), ("c", "3")]);
        m1.merge(&m2);
        assert_eq!(m1.get("b"), Some("override"));
        assert_eq!(m1.get("c"), Some("3"));
    }

    #[test]
    fn test_binding_map_compatible_merge_compatible() {
        let m1 = bm(&[("s", "http://ex.org/s"), ("p", "http://ex.org/p")]);
        let m2 = bm(&[("s", "http://ex.org/s"), ("o", "http://ex.org/o")]);
        let result = m1.compatible_merge(&m2);
        assert!(result.is_some());
        let merged = result.unwrap();
        assert_eq!(merged.get("o"), Some("http://ex.org/o"));
    }

    #[test]
    fn test_binding_map_compatible_merge_incompatible() {
        let m1 = bm(&[("s", "http://ex.org/s1")]);
        let m2 = bm(&[("s", "http://ex.org/s2")]);
        assert!(m1.compatible_merge(&m2).is_none());
    }

    #[test]
    fn test_binding_map_canonical_key_stable() {
        let m = bm(&[("z", "3"), ("a", "1"), ("m", "2")]);
        let key = m.canonical_key();
        // Should be sorted.
        assert!(key.starts_with("a=1"));
    }

    #[test]
    fn test_binding_map_display() {
        let m = bm(&[("x", "1")]);
        let s = format!("{m}");
        assert!(s.contains("x=1"));
    }

    #[test]
    fn test_binding_map_default() {
        let m = BindingMap::default();
        assert!(m.is_empty());
    }

    // ------------------------------------------------------------------
    // PipelineStage tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pipeline_stage_map_passes() {
        let stage = PipelineStage::Map(Box::new(|mut bm| {
            bm.bind("extra", "value");
            Some(bm)
        }));
        let row = bm(&[("x", "1")]);
        let result = stage.apply(row);
        assert!(result.is_some());
        assert_eq!(result.unwrap().get("extra"), Some("value"));
    }

    #[test]
    fn test_pipeline_stage_map_removes() {
        let stage = PipelineStage::Map(Box::new(|_| None));
        let row = bm(&[("x", "1")]);
        assert!(stage.apply(row).is_none());
    }

    #[test]
    fn test_pipeline_stage_filter_passes() {
        let stage = PipelineStage::Filter(Box::new(|_| true));
        let row = bm(&[("x", "1")]);
        assert!(stage.apply(row).is_some());
    }

    #[test]
    fn test_pipeline_stage_filter_removes() {
        let stage = PipelineStage::Filter(Box::new(|_| false));
        let row = bm(&[("x", "1")]);
        assert!(stage.apply(row).is_none());
    }

    // ------------------------------------------------------------------
    // ParallelPipelineStage tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pipeline_map_stage_constructor() {
        let pipeline = ParallelPipelineStage::map_stage(|mut b| {
            b.bind("new", "val");
            Some(b)
        });
        assert_eq!(pipeline.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_filter_stage_constructor() {
        let pipeline = ParallelPipelineStage::filter_stage(|_| true);
        assert_eq!(pipeline.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_add_map() {
        let pipeline = ParallelPipelineStage::filter_stage(|_| true).add_map(Some);
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_add_filter() {
        let pipeline = ParallelPipelineStage::map_stage(Some).add_filter(|_| true);
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_process_map_all() {
        let pipeline = ParallelPipelineStage::map_stage(|mut b| {
            b.bind("added", "yes");
            Some(b)
        });
        let inputs = vec![bm(&[("x", "1")]), bm(&[("x", "2")])];
        let result = pipeline.process(inputs);
        assert_eq!(result.len(), 2);
        for r in &result {
            assert_eq!(r.get("added"), Some("yes"));
        }
    }

    #[test]
    fn test_pipeline_process_filter_some() {
        let pipeline = ParallelPipelineStage::filter_stage(|b| b.get("x") == Some("1"));
        let inputs = vec![bm(&[("x", "1")]), bm(&[("x", "2")]), bm(&[("x", "1")])];
        let result = pipeline.process(inputs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_pipeline_process_filter_all_out() {
        let pipeline = ParallelPipelineStage::filter_stage(|_| false);
        let inputs = vec![bm(&[("x", "1")]), bm(&[("x", "2")])];
        let result = pipeline.process(inputs);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_process_empty_input() {
        let pipeline = ParallelPipelineStage::map_stage(Some);
        let result = pipeline.process(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_chain_two_pipelines() {
        let p1 = ParallelPipelineStage::filter_stage(|b| b.get("x") == Some("1"));
        let p2 = ParallelPipelineStage::filter_stage(|b| b.get("x") == Some("2"));
        let inputs = vec![bm(&[("x", "1")]), bm(&[("x", "2")]), bm(&[("x", "3")])];
        let result = ParallelPipelineStage::chain(vec![p1, p2], inputs);
        // p1 passes 1 row, p2 passes 1 row → 2 total.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_pipeline_chain_empty_stages() {
        let result = ParallelPipelineStage::chain(vec![], vec![bm(&[("x", "1")])]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_multi_stage_composition() {
        // Map: add a field, then Filter: keep only rows where new field equals "mapped".
        let pipeline = ParallelPipelineStage::map_stage(|mut b| {
            let x = b.get("x").unwrap_or("").to_string();
            b.bind("label", format!("item_{x}"));
            Some(b)
        })
        .add_filter(|b| b.get("label").is_some_and(|l| l.starts_with("item_")));

        let inputs = vec![bm(&[("x", "a")]), bm(&[("x", "b")])];
        let result = pipeline.process(inputs);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("label"), Some("item_a"));
    }

    // ------------------------------------------------------------------
    // UnionParallelExecutor tests
    // ------------------------------------------------------------------

    #[test]
    fn test_union_executor_empty() {
        let result = UnionParallelExecutor::execute_branches(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_union_executor_single_branch() {
        let branch = vec![bm(&[("x", "1")]), bm(&[("x", "2")])];
        let result = UnionParallelExecutor::execute_branches(vec![branch]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_union_executor_multiple_branches_dedup() {
        let b1 = vec![bm(&[("x", "1")]), bm(&[("x", "2")])];
        let b2 = vec![bm(&[("x", "2")]), bm(&[("x", "3")])]; // "x=2" is a duplicate.
        let result = UnionParallelExecutor::execute_branches(vec![b1, b2]);
        assert_eq!(result.len(), 3, "duplicate should be removed");
    }

    #[test]
    fn test_union_executor_multiple_branches_no_overlap() {
        let b1 = vec![bm(&[("x", "1")])];
        let b2 = vec![bm(&[("x", "2")])];
        let b3 = vec![bm(&[("x", "3")])];
        let result = UnionParallelExecutor::execute_branches(vec![b1, b2, b3]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_optional_executor_empty_optional() {
        let main = vec![bm(&[("s", "s1")]), bm(&[("s", "s2")])];
        let result = UnionParallelExecutor::execute_optional(main.clone(), vec![]);
        // When optional is empty, return main unchanged.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_optional_executor_compatible_rows() {
        // main row binds ?s; optional row adds ?o for the same ?s.
        let main = vec![bm(&[("s", "s1")])];
        let optional = vec![bm(&[("s", "s1"), ("o", "o1")])];
        let result = UnionParallelExecutor::execute_optional(main, optional);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("o"), Some("o1"));
    }

    #[test]
    fn test_optional_executor_no_compatible_rows() {
        // main and optional share ?s but with different values.
        let main = vec![bm(&[("s", "s1")])];
        let optional = vec![bm(&[("s", "s2"), ("o", "o1")])];
        let result = UnionParallelExecutor::execute_optional(main, optional);
        // No compatible rows → return main row unchanged.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("s"), Some("s1"));
        assert_eq!(result[0].get("o"), None);
    }

    #[test]
    fn test_optional_executor_multiple_compatible() {
        // main row binds ?s=s1; two optional rows are compatible (different ?o values).
        let main = vec![bm(&[("s", "s1")])];
        let optional = vec![
            bm(&[("s", "s1"), ("o", "o1")]),
            bm(&[("s", "s1"), ("o", "o2")]),
        ];
        let result = UnionParallelExecutor::execute_optional(main, optional);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dedup_removes_duplicates() {
        let rows = vec![bm(&[("x", "1")]), bm(&[("x", "1")]), bm(&[("x", "2")])];
        let deduped = UnionParallelExecutor::dedup(rows);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_dedup_empty() {
        let deduped = UnionParallelExecutor::dedup(vec![]);
        assert!(deduped.is_empty());
    }
}
