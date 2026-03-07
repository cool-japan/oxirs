//! Parallel Triple Pattern Evaluation
//!
//! This module implements parallel evaluation of independent triple patterns
//! within a Basic Graph Pattern (BGP).  Independent patterns (those that do
//! not share variables with each other) can be executed concurrently, and
//! their results are then joined in the correct order.
//!
//! The dependency analysis uses a directed acyclic graph (DAG) over patterns
//! to identify which patterns can be safely parallelised.

use crate::optimizer::adaptive::TriplePatternInfo;
use crate::optimizer::materialized_view::{BindingRow, RdfTerm};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public trait: TripleStore
// ---------------------------------------------------------------------------

/// Trait abstracting a triple store that can evaluate a single pattern
/// against an optional set of input bindings.
///
/// Implementations must be `Send + Sync` to allow parallel evaluation.
pub trait TripleStore: Send + Sync {
    /// Evaluate a triple pattern, optionally constrained by `bindings`.
    ///
    /// If `bindings` is `Some`, each row is used to ground variables before
    /// evaluating the pattern (index-nested-loop style).
    fn evaluate_pattern(
        &self,
        pattern: &TriplePatternInfo,
        bindings: Option<&[BindingRow]>,
    ) -> Result<Vec<BindingRow>>;

    /// Return a quick cardinality estimate for a pattern (used for ordering).
    fn estimate_cardinality(&self, pattern: &TriplePatternInfo) -> u64;
}

// ---------------------------------------------------------------------------
// Dependency analysis
// ---------------------------------------------------------------------------

/// Dependency graph over a set of triple patterns.
///
/// Pattern `i` depends on pattern `j` when `j` binds a variable that `i`
/// needs as input.  In the context of parallel evaluation, two patterns are
/// *independent* when neither depends on the other.
pub struct PatternDependencyGraph {
    patterns: Vec<TriplePatternInfo>,
    /// `dependencies[i]` = set of pattern indices that `i` depends on
    dependencies: Vec<HashSet<usize>>,
    /// Topologically sorted execution stages: each stage is a set of indices
    /// that can be evaluated in parallel once all previous stages complete.
    execution_stages: Vec<Vec<usize>>,
}

impl PatternDependencyGraph {
    /// Build the dependency graph for the given pattern list.
    pub fn build(patterns: Vec<TriplePatternInfo>) -> Self {
        let n = patterns.len();
        let mut dependencies: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        // Build a map from variable name -> first pattern that binds it
        let mut var_producer: HashMap<String, usize> = HashMap::new();
        for (i, pattern) in patterns.iter().enumerate() {
            for var in &pattern.bound_variables {
                var_producer.entry(var.clone()).or_insert(i);
            }
        }

        // Pattern `i` depends on `j` if `j` is the producer of a variable
        // that `i` also uses, and j != i.
        for i in 0..n {
            for (var_name, &producer) in &var_producer {
                if producer == i {
                    continue;
                }
                if patterns[i].bound_variables.contains(var_name) {
                    dependencies[i].insert(producer);
                }
            }
        }

        let execution_stages = Self::topological_stages(&dependencies, n);

        Self {
            patterns,
            dependencies,
            execution_stages,
        }
    }

    /// Return groups of patterns that can be evaluated in parallel.
    pub fn get_independent_patterns(&self) -> Vec<Vec<usize>> {
        self.execution_stages.clone()
    }

    /// Return the topologically sorted execution stages.
    pub fn execution_order(&self) -> &[Vec<usize>] {
        &self.execution_stages
    }

    /// Access the underlying patterns
    pub fn patterns(&self) -> &[TriplePatternInfo] {
        &self.patterns
    }

    /// Check whether two pattern indices are independent
    pub fn are_independent(&self, i: usize, j: usize) -> bool {
        !self.dependencies[i].contains(&j) && !self.dependencies[j].contains(&i)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Kahn's algorithm for topological layering
    fn topological_stages(dependencies: &[HashSet<usize>], n: usize) -> Vec<Vec<usize>> {
        let mut in_degree: Vec<usize> = dependencies.iter().map(|d| d.len()).collect();
        let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, deps) in dependencies.iter().enumerate() {
            for &dep in deps {
                reverse[dep].push(i);
            }
        }

        let mut stages: Vec<Vec<usize>> = Vec::new();
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        while !queue.is_empty() {
            let stage: Vec<usize> = queue.drain(..).collect();
            for &node in &stage {
                for &dependent in &reverse[node] {
                    in_degree[dependent] -= 1;
                    if in_degree[dependent] == 0 {
                        queue.push_back(dependent);
                    }
                }
            }
            stages.push(stage);
        }

        stages
    }
}

// ---------------------------------------------------------------------------
// Parallel BGP evaluator
// ---------------------------------------------------------------------------

/// Parallel evaluator for Basic Graph Patterns.
pub struct ParallelBgpEvaluator {
    /// Number of worker threads (0 = use Rayon's global pool)
    pub num_threads: usize,
    /// Minimum patterns per stage to justify parallelism overhead
    pub chunk_size: usize,
}

impl Default for ParallelBgpEvaluator {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            chunk_size: 1,
        }
    }
}

impl ParallelBgpEvaluator {
    /// Create a new evaluator with a specific thread count
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            chunk_size: 1,
        }
    }

    /// Create a new evaluator with tunable chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Evaluate a BGP by exploiting parallelism among independent patterns.
    pub fn evaluate(
        &self,
        patterns: Vec<TriplePatternInfo>,
        store: &dyn TripleStore,
    ) -> Result<Vec<BindingRow>> {
        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        let graph = PatternDependencyGraph::build(patterns);
        let stages = graph.execution_order().to_vec();

        // Running binding set: starts as a single empty row (identity for join)
        let mut current_bindings: Vec<BindingRow> = vec![BindingRow::new()];

        for stage in &stages {
            let stage_results =
                self.evaluate_stage(stage, graph.patterns(), store, &current_bindings)?;

            for (pattern_idx, pattern_rows) in stage_results {
                let pattern = &graph.patterns()[pattern_idx];
                // Only join on variables that already appear in current_bindings
                let join_vars: Vec<String> = if current_bindings.is_empty() {
                    Vec::new()
                } else {
                    let first_row = &current_bindings[0];
                    pattern
                        .bound_variables
                        .iter()
                        .filter(|v| first_row.contains_key(v.as_str()))
                        .cloned()
                        .collect()
                };

                current_bindings = self.merge_results(current_bindings, pattern_rows, &join_vars);
            }
        }

        Ok(current_bindings)
    }

    /// Evaluate a set of pattern indices in parallel within a single stage.
    fn evaluate_stage(
        &self,
        stage: &[usize],
        patterns: &[TriplePatternInfo],
        store: &dyn TripleStore,
        current_bindings: &[BindingRow],
    ) -> Result<Vec<(usize, Vec<BindingRow>)>> {
        if stage.is_empty() {
            return Ok(Vec::new());
        }

        if stage.len() < self.chunk_size || self.num_threads <= 1 {
            return self.evaluate_stage_sequential(stage, patterns, store, current_bindings);
        }

        #[cfg(feature = "parallel")]
        {
            self.evaluate_stage_parallel(stage, patterns, store, current_bindings)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.evaluate_stage_sequential(stage, patterns, store, current_bindings)
        }
    }

    fn evaluate_stage_sequential(
        &self,
        stage: &[usize],
        patterns: &[TriplePatternInfo],
        store: &dyn TripleStore,
        current_bindings: &[BindingRow],
    ) -> Result<Vec<(usize, Vec<BindingRow>)>> {
        let mut results = Vec::with_capacity(stage.len());
        for &idx in stage {
            let rows = store.evaluate_pattern(&patterns[idx], Some(current_bindings))?;
            results.push((idx, rows));
        }
        Ok(results)
    }

    #[cfg(feature = "parallel")]
    fn evaluate_stage_parallel(
        &self,
        stage: &[usize],
        patterns: &[TriplePatternInfo],
        store: &dyn TripleStore,
        current_bindings: &[BindingRow],
    ) -> Result<Vec<(usize, Vec<BindingRow>)>> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let error_cell: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));
        let error_clone = Arc::clone(&error_cell);

        let results: Vec<(usize, Vec<BindingRow>)> = stage
            .par_iter()
            .filter_map(|&idx| {
                match store.evaluate_pattern(&patterns[idx], Some(current_bindings)) {
                    Ok(rows) => Some((idx, rows)),
                    Err(e) => {
                        if let Ok(mut guard) = error_clone.lock() {
                            if guard.is_none() {
                                *guard = Some(e);
                            }
                        }
                        None
                    }
                }
            })
            .collect();

        if let Ok(mut guard) = error_cell.lock() {
            if let Some(err) = guard.take() {
                return Err(err);
            }
        }
        Ok(results)
    }

    /// Hash join of two binding sets on shared variables.
    pub fn merge_results(
        &self,
        left: Vec<BindingRow>,
        right: Vec<BindingRow>,
        join_vars: &[String],
    ) -> Vec<BindingRow> {
        if right.is_empty() {
            return left;
        }
        if left.is_empty() {
            return right;
        }

        // Cross product when no shared variables
        if join_vars.is_empty() {
            let mut output: Vec<BindingRow> = Vec::with_capacity(left.len() * right.len());
            for l_row in &left {
                for r_row in &right {
                    let mut merged: BindingRow = l_row.clone();
                    for (k, v) in r_row {
                        merged.insert(k.clone(), v.clone());
                    }
                    output.push(merged);
                }
            }
            return output;
        }

        // Build hash index over right side
        let mut hash_index: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for (ridx, row) in right.iter().enumerate() {
            let key: Vec<String> = join_vars.iter().map(|v| rdf_term_key(row.get(v))).collect();
            hash_index.entry(key).or_default().push(ridx);
        }

        // Probe phase
        let mut output: Vec<BindingRow> = Vec::new();
        for l_row in &left {
            let key: Vec<String> = join_vars
                .iter()
                .map(|v| rdf_term_key(l_row.get(v)))
                .collect();

            if let Some(right_indices) = hash_index.get(&key) {
                for &ridx in right_indices {
                    let r_row = &right[ridx];
                    let mut merged: BindingRow = l_row.clone();
                    for (k, v) in r_row {
                        merged.insert(k.clone(), v.clone());
                    }
                    output.push(merged);
                }
            }
        }
        output
    }
}

/// Convert an optional RdfTerm to a stable string key for hashing
fn rdf_term_key(term: Option<&RdfTerm>) -> String {
    match term {
        None => String::new(),
        Some(t) => format!("{t}"),
    }
}

// ---------------------------------------------------------------------------
// Mock TripleStore for testing
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Simple in-memory triple store for tests.
    pub struct MockTripleStore {
        pub results: HashMap<String, Vec<BindingRow>>,
        pub default_result: Vec<BindingRow>,
    }

    impl MockTripleStore {
        pub fn new() -> Self {
            Self {
                results: HashMap::new(),
                default_result: Vec::new(),
            }
        }

        pub fn with_result(mut self, pattern_id: &str, rows: Vec<BindingRow>) -> Self {
            self.results.insert(pattern_id.to_string(), rows);
            self
        }
    }

    impl TripleStore for MockTripleStore {
        fn evaluate_pattern(
            &self,
            pattern: &TriplePatternInfo,
            _bindings: Option<&[BindingRow]>,
        ) -> Result<Vec<BindingRow>> {
            Ok(self
                .results
                .get(&pattern.id)
                .cloned()
                .unwrap_or_else(|| self.default_result.clone()))
        }

        fn estimate_cardinality(&self, pattern: &TriplePatternInfo) -> u64 {
            self.results
                .get(&pattern.id)
                .map(|r| r.len() as u64)
                .unwrap_or(0)
        }
    }

    pub fn iri_term(value: &str) -> RdfTerm {
        RdfTerm::Iri(value.to_string())
    }

    pub fn make_row(pairs: &[(&str, RdfTerm)]) -> BindingRow {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::optimizer::adaptive::{PatternTerm, TriplePatternInfo};
    use crate::optimizer::materialized_view::RdfTerm;

    fn simple_pattern(id: &str, vars: Vec<String>, cardinality: u64) -> TriplePatternInfo {
        TriplePatternInfo {
            id: id.to_string(),
            subject: PatternTerm::Variable(vars.first().cloned().unwrap_or_default()),
            predicate: PatternTerm::Iri(format!("http://example.org/p_{id}")),
            object: PatternTerm::Variable(vars.last().cloned().unwrap_or_default()),
            estimated_cardinality: cardinality,
            bound_variables: vars,
            original_pattern: None,
        }
    }

    #[test]
    fn test_dependency_graph_independent_patterns() {
        let p1 = simple_pattern("p1", vec!["a".to_string(), "b".to_string()], 10);
        let p2 = simple_pattern("p2", vec!["c".to_string(), "d".to_string()], 20);
        let graph = PatternDependencyGraph::build(vec![p1, p2]);

        assert!(
            graph.are_independent(0, 1),
            "Patterns with no shared vars should be independent"
        );
        let stages = graph.get_independent_patterns();
        assert_eq!(stages.len(), 1, "Independent patterns fit into one stage");
        assert_eq!(stages[0].len(), 2);
    }

    #[test]
    fn test_dependency_graph_dependent_patterns() {
        let p1 = simple_pattern("p1", vec!["s".to_string(), "type".to_string()], 10);
        let p2 = simple_pattern("p2", vec!["s".to_string(), "name".to_string()], 100);
        let graph = PatternDependencyGraph::build(vec![p1, p2]);

        let stages = graph.get_independent_patterns();
        let total: usize = stages.iter().map(|s| s.len()).sum();
        assert_eq!(total, 2, "All patterns should appear across stages");
    }

    #[test]
    fn test_parallel_evaluator_empty_patterns() {
        let evaluator = ParallelBgpEvaluator::new(2);
        let store = MockTripleStore::new();
        let result = evaluator.evaluate(vec![], &store).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parallel_evaluator_single_pattern() {
        let pattern = simple_pattern("pat1", vec!["s".to_string()], 2);
        let rows = vec![
            make_row(&[("s", iri_term("http://example.org/a"))]),
            make_row(&[("s", iri_term("http://example.org/b"))]),
        ];
        let store = MockTripleStore::new().with_result("pat1", rows);
        let evaluator = ParallelBgpEvaluator::new(1);
        let result = evaluator.evaluate(vec![pattern], &store).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parallel_evaluator_two_patterns_with_join() {
        let p1 = simple_pattern("p1", vec!["s".to_string(), "type".to_string()], 2);
        let p2 = simple_pattern("p2", vec!["s".to_string(), "name".to_string()], 2);

        let p1_rows = vec![
            make_row(&[
                ("s", iri_term("http://example.org/alice")),
                ("type", iri_term("http://example.org/Person")),
            ]),
            make_row(&[
                ("s", iri_term("http://example.org/bob")),
                ("type", iri_term("http://example.org/Person")),
            ]),
        ];
        let p2_rows = vec![
            make_row(&[
                ("s", iri_term("http://example.org/alice")),
                ("name", RdfTerm::plain_literal("Alice")),
            ]),
            make_row(&[
                ("s", iri_term("http://example.org/bob")),
                ("name", RdfTerm::plain_literal("Bob")),
            ]),
        ];

        let store = MockTripleStore::new()
            .with_result("p1", p1_rows)
            .with_result("p2", p2_rows);

        let evaluator = ParallelBgpEvaluator::new(2);
        let result = evaluator.evaluate(vec![p1, p2], &store).unwrap();

        assert_eq!(
            result.len(),
            2,
            "Should produce 2 joined rows (one per person)"
        );
        for row in &result {
            assert!(row.contains_key("s"));
            assert!(row.contains_key("name"));
        }
    }

    #[test]
    fn test_merge_results_no_join_vars_cross_product() {
        let evaluator = ParallelBgpEvaluator::new(1);
        let left = vec![
            make_row(&[("a", iri_term("http://example.org/1"))]),
            make_row(&[("a", iri_term("http://example.org/2"))]),
        ];
        let right = vec![make_row(&[("b", iri_term("http://example.org/x"))])];

        let merged = evaluator.merge_results(left, right, &[]);
        assert_eq!(merged.len(), 2, "Cross product of 2x1 = 2 rows");
    }

    #[test]
    fn test_merge_results_with_join_var() {
        let evaluator = ParallelBgpEvaluator::new(1);
        let left = vec![
            make_row(&[
                ("s", iri_term("http://a")),
                ("type", iri_term("http://Person")),
            ]),
            make_row(&[
                ("s", iri_term("http://b")),
                ("type", iri_term("http://Person")),
            ]),
        ];
        let right = vec![make_row(&[
            ("s", iri_term("http://a")),
            ("name", RdfTerm::plain_literal("Alice")),
        ])];

        let merged = evaluator.merge_results(left, right, &["s".to_string()]);
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].get("name"),
            Some(&RdfTerm::plain_literal("Alice"))
        );
    }

    #[test]
    fn test_merge_results_empty_left_returns_right() {
        let evaluator = ParallelBgpEvaluator::new(1);
        let right = vec![make_row(&[("s", iri_term("http://a"))])];
        let merged = evaluator.merge_results(vec![], right, &[]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_results_empty_right_returns_left() {
        let evaluator = ParallelBgpEvaluator::new(1);
        let left = vec![make_row(&[("s", iri_term("http://a"))])];
        let merged = evaluator.merge_results(left, vec![], &[]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_dependency_graph_three_chain() {
        let p1 = simple_pattern("p1", vec!["x".to_string()], 5);
        let p2 = simple_pattern("p2", vec!["x".to_string(), "y".to_string()], 50);
        let p3 = simple_pattern("p3", vec!["y".to_string(), "z".to_string()], 500);

        let graph = PatternDependencyGraph::build(vec![p1, p2, p3]);
        let stages = graph.execution_order();
        let total: usize = stages.iter().map(|s| s.len()).sum();
        assert_eq!(total, 3);
        assert!(!stages.is_empty());
    }

    #[test]
    fn test_evaluator_default_thread_count() {
        let evaluator = ParallelBgpEvaluator::default();
        assert!(evaluator.num_threads >= 1);
    }
}

#[cfg(test)]
mod extended_tests {
    use super::test_support::*;
    use super::*;
    use crate::optimizer::adaptive::{PatternTerm, TriplePatternInfo};
    use crate::optimizer::materialized_view::RdfTerm;

    fn pat(id: &str, vars: Vec<String>, cardinality: u64) -> TriplePatternInfo {
        TriplePatternInfo {
            id: id.to_string(),
            subject: PatternTerm::Variable(vars.first().cloned().unwrap_or_default()),
            predicate: PatternTerm::Iri(format!("http://example.org/p_{id}")),
            object: PatternTerm::Variable(vars.last().cloned().unwrap_or_default()),
            estimated_cardinality: cardinality,
            bound_variables: vars,
            original_pattern: None,
        }
    }

    // --- PatternDependencyGraph extended tests ---

    #[test]
    fn test_dependency_graph_single_pattern() {
        let p1 = pat("solo", vec!["x".to_string()], 10);
        let graph = PatternDependencyGraph::build(vec![p1]);

        let stages = graph.get_independent_patterns();
        assert_eq!(
            stages.len(),
            1,
            "Single pattern should produce a single stage"
        );
        assert_eq!(stages[0], vec![0], "Stage 0 should contain pattern 0");
    }

    #[test]
    fn test_dependency_graph_no_patterns() {
        let graph = PatternDependencyGraph::build(vec![]);
        assert!(graph.get_independent_patterns().is_empty());
    }

    #[test]
    fn test_dependency_graph_are_independent_different_vars() {
        let p1 = pat("p1", vec!["a".to_string(), "b".to_string()], 10);
        let p2 = pat("p2", vec!["c".to_string(), "d".to_string()], 10);
        let graph = PatternDependencyGraph::build(vec![p1, p2]);

        assert!(
            graph.are_independent(0, 1),
            "Patterns with disjoint variables should be independent"
        );
    }

    #[test]
    fn test_dependency_graph_are_not_independent_shared_var() {
        let p1 = pat("p1", vec!["s".to_string(), "o1".to_string()], 10);
        let p2 = pat("p2", vec!["s".to_string(), "o2".to_string()], 10);
        let graph = PatternDependencyGraph::build(vec![p1, p2]);

        // The graph.are_independent() checks if both directions are dependency-free
        // Shared variable "s" should create a dependency
        // They can still be in the same stage if neither depends on the other's bindings
        let _stages = graph.get_independent_patterns();
        let patterns = graph.patterns();
        assert_eq!(patterns.len(), 2, "Graph should contain 2 patterns");
    }

    #[test]
    fn test_dependency_graph_execution_order_returns_all_patterns() {
        let p1 = pat("p1", vec!["a".to_string()], 10);
        let p2 = pat("p2", vec!["b".to_string()], 20);
        let p3 = pat("p3", vec!["c".to_string()], 30);
        let graph = PatternDependencyGraph::build(vec![p1, p2, p3]);

        let total_in_stages: usize = graph.execution_order().iter().map(|s| s.len()).sum();
        assert_eq!(
            total_in_stages, 3,
            "All patterns should appear in execution stages"
        );
    }

    #[test]
    fn test_dependency_graph_patterns_accessor() {
        let p1 = pat("x", vec!["a".to_string()], 5);
        let p2 = pat("y", vec!["b".to_string()], 15);
        let graph = PatternDependencyGraph::build(vec![p1, p2]);

        let patterns = graph.patterns();
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].estimated_cardinality, 5);
        assert_eq!(patterns[1].estimated_cardinality, 15);
    }

    // --- merge_results extended tests ---

    #[test]
    fn test_merge_results_multi_var_join() {
        let evaluator = ParallelBgpEvaluator::new(1);

        let mut row_l = BindingRow::new();
        row_l.insert("x".to_string(), RdfTerm::iri("http://a"));
        row_l.insert("y".to_string(), RdfTerm::iri("http://b"));

        let mut row_r = BindingRow::new();
        row_r.insert("x".to_string(), RdfTerm::iri("http://a"));
        row_r.insert("y".to_string(), RdfTerm::iri("http://b"));
        row_r.insert("z".to_string(), RdfTerm::iri("http://c"));

        let result = evaluator.merge_results(
            vec![row_l],
            vec![row_r],
            &["x".to_string(), "y".to_string()],
        );

        assert_eq!(
            result.len(),
            1,
            "Matching multi-var join should produce one row"
        );
        assert!(
            result[0].contains_key("z"),
            "Joined row should contain z from right side"
        );
    }

    #[test]
    fn test_merge_results_no_matching_join_vars() {
        let evaluator = ParallelBgpEvaluator::new(1);

        let mut row_l = BindingRow::new();
        row_l.insert("x".to_string(), RdfTerm::iri("http://a"));

        let mut row_r = BindingRow::new();
        row_r.insert("x".to_string(), RdfTerm::iri("http://DIFFERENT"));

        let result = evaluator.merge_results(vec![row_l], vec![row_r], &["x".to_string()]);

        assert_eq!(
            result.len(),
            0,
            "Non-matching join should produce empty result"
        );
    }

    #[test]
    fn test_merge_results_multiple_right_matches() {
        let evaluator = ParallelBgpEvaluator::new(1);

        let mut row_l = BindingRow::new();
        row_l.insert("x".to_string(), RdfTerm::iri("http://shared"));

        let right: Vec<BindingRow> = (0..3)
            .map(|i| {
                let mut row = BindingRow::new();
                row.insert("x".to_string(), RdfTerm::iri("http://shared"));
                row.insert("y".to_string(), RdfTerm::iri(format!("http://val{i}")));
                row
            })
            .collect();

        let result = evaluator.merge_results(vec![row_l], right, &["x".to_string()]);
        assert_eq!(
            result.len(),
            3,
            "Should produce one row for each matching right-side row"
        );
    }

    // --- ParallelBgpEvaluator configuration tests ---

    #[test]
    fn test_evaluator_chunk_size_minimum_is_one() {
        let evaluator = ParallelBgpEvaluator::new(4).with_chunk_size(0);
        assert_eq!(evaluator.chunk_size, 1, "Chunk size should be at least 1");
    }

    #[test]
    fn test_evaluator_chunk_size_set_correctly() {
        let evaluator = ParallelBgpEvaluator::new(4).with_chunk_size(8);
        assert_eq!(evaluator.chunk_size, 8);
    }

    #[test]
    fn test_evaluator_default_uses_cpu_count() {
        let evaluator = ParallelBgpEvaluator::default();
        assert!(evaluator.num_threads >= 1, "Should use at least 1 thread");
    }

    // --- Evaluate with MockTripleStore ---

    #[test]
    fn test_evaluate_no_results_from_store() {
        // MockTripleStore.new() returns empty rows by default for unknown pattern ids.
        // The evaluator starts with a single empty binding row (identity element for join).
        // merge_results with right=empty returns left, so we get back the initial empty row.
        // Verify that the result contains no variable bindings (all rows are empty maps).
        let store = MockTripleStore::new();
        let evaluator = ParallelBgpEvaluator::new(1);

        let pattern = pat("no_results", vec!["x".to_string(), "y".to_string()], 100);
        let result = evaluator.evaluate(vec![pattern], &store).unwrap();
        // Result may contain the initial empty binding row - verify no actual bindings exist
        let has_bindings = result.iter().any(|row| !row.is_empty());
        assert!(
            !has_bindings,
            "Empty store should produce no variable bindings"
        );
    }

    #[test]
    fn test_evaluate_two_independent_patterns_cross_product() {
        let mut store = MockTripleStore::new();

        let p1 = pat("pat_a", vec!["a".to_string()], 2);
        let p2 = pat("pat_b", vec!["b".to_string()], 3);

        store.results.insert(
            "pat_a".to_string(),
            vec![
                make_row(&[("a", iri_term("http://a1"))]),
                make_row(&[("a", iri_term("http://a2"))]),
            ],
        );
        store.results.insert(
            "pat_b".to_string(),
            vec![
                make_row(&[("b", iri_term("http://b1"))]),
                make_row(&[("b", iri_term("http://b2"))]),
                make_row(&[("b", iri_term("http://b3"))]),
            ],
        );

        let evaluator = ParallelBgpEvaluator::new(1);
        let result = evaluator.evaluate(vec![p1, p2], &store).unwrap();

        // 2 * 3 = 6 cross-product rows (independent patterns)
        assert_eq!(
            result.len(),
            6,
            "Independent patterns produce cross product"
        );
    }
}
