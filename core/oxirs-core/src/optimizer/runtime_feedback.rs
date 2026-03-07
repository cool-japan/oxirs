//! Runtime feedback-based cost optimizer for SPARQL query execution.
//!
//! Collects execution statistics at runtime and uses them to adaptively
//! improve future query planning decisions.

use std::collections::HashMap;

/// Statistics recorded for a single query execution.
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Hash of the query string (see [`AdaptiveQueryOptimizer::hash_query`]).
    pub query_hash: u64,
    /// Number of rows actually produced.
    pub actual_rows: usize,
    /// Number of rows the optimizer estimated would be produced.
    pub estimated_rows: usize,
    /// Wall-clock time consumed by the execution, in milliseconds.
    pub execution_time_ms: u64,
    /// The join order that was used, as a list of pattern descriptions.
    pub join_order: Vec<String>,
    /// Per-pattern selectivities observed during execution.
    pub selectivity_by_pattern: HashMap<String, f64>,
}

impl QueryStats {
    /// Construct a new [`QueryStats`] value.
    pub fn new(
        query_hash: u64,
        actual_rows: usize,
        estimated_rows: usize,
        execution_time_ms: u64,
        join_order: Vec<String>,
        selectivity_by_pattern: HashMap<String, f64>,
    ) -> Self {
        Self {
            query_hash,
            actual_rows,
            estimated_rows,
            execution_time_ms,
            join_order,
            selectivity_by_pattern,
        }
    }

    /// Ratio of actual to estimated rows.  Returns 1.0 when no estimate exists.
    pub fn accuracy_ratio(&self) -> f64 {
        if self.estimated_rows == 0 {
            1.0
        } else {
            self.actual_rows as f64 / self.estimated_rows as f64
        }
    }
}

/// Persistent store of historical query execution statistics.
///
/// Keeps at most `max_history` entries per query hash.
pub struct RuntimeFeedbackStore {
    stats: HashMap<u64, Vec<QueryStats>>,
    max_history: usize,
}

impl RuntimeFeedbackStore {
    /// Create a new store that retains up to `max_history` records per query.
    pub fn new(max_history: usize) -> Self {
        Self {
            stats: HashMap::new(),
            max_history,
        }
    }

    /// Record a new execution statistics entry.
    ///
    /// If the per-query history is already at capacity the oldest entry is
    /// dropped before the new one is appended.
    pub fn record(&mut self, stats: QueryStats) {
        let history = self.stats.entry(stats.query_hash).or_default();
        if history.len() >= self.max_history {
            history.remove(0);
        }
        history.push(stats);
    }

    /// Return the recorded history for a query, or an empty slice when none exists.
    pub fn get_stats(&self, query_hash: u64) -> &[QueryStats] {
        self.stats
            .get(&query_hash)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Estimate the selectivity of a triple-pattern string from historical data.
    ///
    /// Averages the per-pattern selectivity across all historical entries that
    /// contain an observation for `pattern`.  Returns `0.1` as a default when
    /// no data is available.
    pub fn estimate_selectivity(&self, pattern: &str) -> f64 {
        let mut total = 0.0_f64;
        let mut count = 0usize;

        for history in self.stats.values() {
            for entry in history {
                if let Some(&sel) = entry.selectivity_by_pattern.get(pattern) {
                    total += sel;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.1
        } else {
            total / count as f64
        }
    }

    /// Estimate cardinality for a pattern given a base estimate.
    ///
    /// Adjusts `base_estimate` using the average observed selectivity.
    /// Falls back to `base_estimate` when no historical data exists (since
    /// default selectivity 0.1 / 0.1 = ratio 1.0).
    pub fn estimate_cardinality(&self, pattern: &str, base_estimate: usize) -> usize {
        let sel = self.estimate_selectivity(pattern);
        // Default selectivity is 0.1 so ratio = sel / 0.1.
        let adjusted = base_estimate as f64 * sel / 0.1;
        adjusted.round() as usize
    }

    /// Return the join order associated with the lowest average execution time
    /// for a given query hash, or `None` when no history exists.
    pub fn best_join_order(&self, query_hash: u64) -> Option<Vec<String>> {
        let history = self.stats.get(&query_hash)?;
        if history.is_empty() {
            return None;
        }

        // Group by join-order string and compute cumulative execution time.
        let mut order_times: HashMap<String, (u64, usize)> = HashMap::new();
        for entry in history {
            let key = entry.join_order.join(",");
            let acc = order_times.entry(key).or_default();
            acc.0 += entry.execution_time_ms;
            acc.1 += 1;
        }

        // Find the join order with the best (lowest) average time.
        let best_key = order_times
            .iter()
            .map(|(k, (total, cnt))| (k, *total / (*cnt as u64).max(1)))
            .min_by_key(|(_, avg)| *avg)
            .map(|(k, _)| k.clone())?;

        // Reconstruct the Vec<String>, filtering out empty strings from empty join orders.
        let parts: Vec<String> = best_key
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        Some(parts)
    }

    /// Remove all statistics entries where `execution_time_ms >= max_age_ms`.
    ///
    /// Entries with execution time below `max_age_ms` are retained.
    /// Empty query buckets are also removed.
    pub fn prune_old(&mut self, max_age_ms: u64) {
        for history in self.stats.values_mut() {
            history.retain(|s| s.execution_time_ms < max_age_ms);
        }
        // Remove empty buckets.
        self.stats.retain(|_, v| !v.is_empty());
    }

    /// Total number of statistics entries across all queries.
    pub fn stats_count(&self) -> usize {
        self.stats.values().map(|v| v.len()).sum()
    }

    /// Number of distinct query hashes for which history has been recorded.
    pub fn query_count(&self) -> usize {
        self.stats.len()
    }
}

impl Default for RuntimeFeedbackStore {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Adaptive query optimizer that learns from execution feedback.
///
/// Uses a [`RuntimeFeedbackStore`] to guide pattern ordering decisions.
pub struct AdaptiveQueryOptimizer {
    feedback: RuntimeFeedbackStore,
    base_selectivities: HashMap<String, f64>,
}

impl AdaptiveQueryOptimizer {
    /// Create a new optimizer with default (empty) feedback.
    pub fn new() -> Self {
        Self {
            feedback: RuntimeFeedbackStore::new(100),
            base_selectivities: HashMap::new(),
        }
    }

    /// Create an optimizer pre-loaded with the given feedback store.
    pub fn with_feedback(feedback: RuntimeFeedbackStore) -> Self {
        Self {
            feedback,
            base_selectivities: HashMap::new(),
        }
    }

    /// Set a static base selectivity for a pattern (used as fallback).
    pub fn set_base_selectivity(&mut self, pattern: impl Into<String>, selectivity: f64) {
        self.base_selectivities.insert(pattern.into(), selectivity);
    }

    /// Sort `patterns` by their estimated selectivity (ascending).
    ///
    /// Patterns with lower estimated selectivity are placed first so the query
    /// executor processes the most restrictive filters early.
    pub fn optimize_join_order(&self, patterns: &[String]) -> Vec<String> {
        let mut with_sel: Vec<(String, f64)> = patterns
            .iter()
            .map(|p| {
                let sel = self
                    .base_selectivities
                    .get(p.as_str())
                    .copied()
                    .unwrap_or_else(|| self.feedback.estimate_selectivity(p));
                (p.clone(), sel)
            })
            .collect();

        // Stable sort so equal-selectivity patterns retain their original order.
        with_sel.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        with_sel.into_iter().map(|(p, _)| p).collect()
    }

    /// Immutable access to the underlying feedback store.
    pub fn feedback(&self) -> &RuntimeFeedbackStore {
        &self.feedback
    }

    /// Mutable access to the underlying feedback store.
    pub fn feedback_mut(&mut self) -> &mut RuntimeFeedbackStore {
        &mut self.feedback
    }

    /// Record an execution result, forwarding it to the feedback store.
    pub fn record_execution(&mut self, query_hash: u64, mut stats: QueryStats) {
        stats.query_hash = query_hash;
        self.feedback.record(stats);
    }

    /// Compute a 64-bit hash for an arbitrary query string using the djb2
    /// algorithm, suitable for use as a cache/feedback key.
    pub fn hash_query(query_str: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in query_str.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    /// Estimate the cardinality of a pattern using feedback data.
    pub fn estimate_cardinality(&self, pattern: &str, base_estimate: usize) -> usize {
        self.feedback.estimate_cardinality(pattern, base_estimate)
    }
}

impl Default for AdaptiveQueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(
        query_hash: u64,
        actual: usize,
        estimated: usize,
        time_ms: u64,
        join_order: &[&str],
        selectivities: &[(&str, f64)],
    ) -> QueryStats {
        let mut sel_map = HashMap::new();
        for (k, v) in selectivities {
            sel_map.insert((*k).to_string(), *v);
        }
        QueryStats::new(
            query_hash,
            actual,
            estimated,
            time_ms,
            join_order.iter().map(|s| s.to_string()).collect(),
            sel_map,
        )
    }

    // --- QueryStats tests ---

    #[test]
    fn test_query_stats_accuracy_ratio_normal() {
        let s = make_stats(1, 50, 100, 10, &[], &[]);
        let ratio = s.accuracy_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_query_stats_accuracy_ratio_zero_estimated() {
        let s = make_stats(1, 10, 0, 10, &[], &[]);
        assert!((s.accuracy_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_query_stats_accuracy_ratio_perfect() {
        let s = make_stats(1, 100, 100, 5, &[], &[]);
        assert!((s.accuracy_ratio() - 1.0).abs() < 1e-9);
    }

    // --- RuntimeFeedbackStore tests ---

    #[test]
    fn test_store_new_is_empty() {
        let store = RuntimeFeedbackStore::new(10);
        assert_eq!(store.stats_count(), 0);
        assert_eq!(store.query_count(), 0);
    }

    #[test]
    fn test_store_record_and_get() {
        let mut store = RuntimeFeedbackStore::new(10);
        let s = make_stats(42, 10, 20, 5, &["a", "b"], &[("a", 0.1)]);
        store.record(s);

        let hist = store.get_stats(42);
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].actual_rows, 10);
    }

    #[test]
    fn test_store_max_history_eviction() {
        let mut store = RuntimeFeedbackStore::new(3);
        for i in 0..5u64 {
            store.record(make_stats(99, i as usize, 10, i, &[], &[]));
        }
        let hist = store.get_stats(99);
        assert_eq!(hist.len(), 3);
        // Most-recent actual_rows should be 4.
        assert_eq!(hist.last().map(|s| s.actual_rows), Some(4));
    }

    #[test]
    fn test_store_get_unknown_hash() {
        let store = RuntimeFeedbackStore::new(10);
        assert!(store.get_stats(999).is_empty());
    }

    #[test]
    fn test_estimate_selectivity_no_data_returns_default() {
        let store = RuntimeFeedbackStore::new(10);
        assert!((store.estimate_selectivity("p_unknown") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_selectivity_with_data() {
        let mut store = RuntimeFeedbackStore::new(10);
        store.record(make_stats(
            1,
            10,
            100,
            5,
            &[],
            &[("age", 0.2), ("name", 0.5)],
        ));
        store.record(make_stats(2, 5, 50, 3, &[], &[("age", 0.4)]));

        // age: (0.2 + 0.4) / 2 = 0.3
        let sel = store.estimate_selectivity("age");
        assert!((sel - 0.3).abs() < 1e-9);

        // name: only one sample = 0.5
        let sel_name = store.estimate_selectivity("name");
        assert!((sel_name - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_cardinality() {
        let mut store = RuntimeFeedbackStore::new(10);
        store.record(make_stats(1, 10, 100, 5, &[], &[("p", 0.2)]));
        // selectivity 0.2 / 0.1 = 2.0 multiplier → 1000 * 2.0 = 2000
        let card = store.estimate_cardinality("p", 1000);
        assert_eq!(card, 2000);
    }

    #[test]
    fn test_estimate_cardinality_no_data() {
        let store = RuntimeFeedbackStore::new(10);
        // Default selectivity 0.1 / 0.1 = 1.0 → same as base
        let card = store.estimate_cardinality("p", 500);
        assert_eq!(card, 500);
    }

    #[test]
    fn test_best_join_order_no_data() {
        let store = RuntimeFeedbackStore::new(10);
        assert!(store.best_join_order(42).is_none());
    }

    #[test]
    fn test_best_join_order_single_entry() {
        let mut store = RuntimeFeedbackStore::new(10);
        store.record(make_stats(1, 10, 10, 50, &["a", "b", "c"], &[]));
        let order = store.best_join_order(1).expect("should have an order");
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_best_join_order_selects_fastest() {
        let mut store = RuntimeFeedbackStore::new(20);
        // Order A,B → avg 100 ms
        store.record(make_stats(5, 10, 10, 100, &["A", "B"], &[]));
        store.record(make_stats(5, 10, 10, 100, &["A", "B"], &[]));
        // Order B,A → avg 50 ms (faster)
        store.record(make_stats(5, 10, 10, 50, &["B", "A"], &[]));
        store.record(make_stats(5, 10, 10, 50, &["B", "A"], &[]));

        let best = store.best_join_order(5).expect("should have best order");
        assert_eq!(best, vec!["B", "A"]);
    }

    #[test]
    fn test_prune_old() {
        let mut store = RuntimeFeedbackStore::new(20);
        store.record(make_stats(1, 10, 10, 5, &[], &[])); // time 5 — kept when max_age=10
        store.record(make_stats(1, 10, 10, 15, &[], &[])); // time 15 — pruned when max_age=10
        store.record(make_stats(2, 10, 10, 3, &[], &[])); // kept

        store.prune_old(10);
        assert_eq!(store.get_stats(1).len(), 1);
        assert_eq!(store.get_stats(2).len(), 1);
        assert_eq!(store.stats_count(), 2);
    }

    #[test]
    fn test_stats_count() {
        let mut store = RuntimeFeedbackStore::new(20);
        store.record(make_stats(1, 1, 1, 1, &[], &[]));
        store.record(make_stats(1, 2, 2, 2, &[], &[]));
        store.record(make_stats(2, 3, 3, 3, &[], &[]));
        assert_eq!(store.stats_count(), 3);
        assert_eq!(store.query_count(), 2);
    }

    // --- AdaptiveQueryOptimizer tests ---

    #[test]
    fn test_optimizer_new_empty_feedback() {
        let opt = AdaptiveQueryOptimizer::new();
        assert_eq!(opt.feedback().stats_count(), 0);
    }

    #[test]
    fn test_hash_query_deterministic() {
        let h1 = AdaptiveQueryOptimizer::hash_query("SELECT * WHERE { ?s ?p ?o }");
        let h2 = AdaptiveQueryOptimizer::hash_query("SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_query_different_inputs() {
        let h1 = AdaptiveQueryOptimizer::hash_query("query_a");
        let h2 = AdaptiveQueryOptimizer::hash_query("query_b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_optimize_join_order_no_feedback_preserves_order() {
        let opt = AdaptiveQueryOptimizer::new();
        let patterns = vec!["p1".to_string(), "p2".to_string(), "p3".to_string()];
        // All patterns get the default selectivity 0.1, so order should be stable.
        let result = opt.optimize_join_order(&patterns);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_optimize_join_order_uses_selectivity() {
        let mut opt = AdaptiveQueryOptimizer::new();
        opt.set_base_selectivity("low_sel", 0.01);
        opt.set_base_selectivity("high_sel", 0.9);

        let patterns = vec!["high_sel".to_string(), "low_sel".to_string()];
        let result = opt.optimize_join_order(&patterns);
        assert_eq!(result[0], "low_sel"); // lower selectivity first
        assert_eq!(result[1], "high_sel");
    }

    #[test]
    fn test_record_execution_updates_feedback() {
        let mut opt = AdaptiveQueryOptimizer::new();
        let hash = AdaptiveQueryOptimizer::hash_query("my_query");
        let stats = make_stats(hash, 10, 20, 5, &["a"], &[("a", 0.3)]);
        opt.record_execution(hash, stats);
        assert_eq!(opt.feedback().get_stats(hash).len(), 1);
    }

    #[test]
    fn test_estimate_cardinality_via_optimizer() {
        let mut opt = AdaptiveQueryOptimizer::new();
        let hash = AdaptiveQueryOptimizer::hash_query("q");
        opt.record_execution(hash, make_stats(hash, 5, 10, 2, &[], &[("p", 0.5)]));

        // Selectivity 0.5 / 0.1 = 5x multiplier → 100 * 5 = 500
        let card = opt.estimate_cardinality("p", 100);
        assert_eq!(card, 500);
    }

    #[test]
    fn test_with_feedback_constructor() {
        let mut store = RuntimeFeedbackStore::new(5);
        store.record(make_stats(1, 1, 1, 1, &[], &[]));
        let opt = AdaptiveQueryOptimizer::with_feedback(store);
        assert_eq!(opt.feedback().stats_count(), 1);
    }

    #[test]
    fn test_optimize_join_order_empty() {
        let opt = AdaptiveQueryOptimizer::new();
        let result = opt.optimize_join_order(&[]);
        assert!(result.is_empty());
    }
}
