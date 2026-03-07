//! Runtime Statistics Collector for Query Feedback
//!
//! Collects real execution statistics per query and feeds them back to the
//! optimizer for adaptive plan improvement over time.  The collector maintains
//! a bounded circular history of `QueryExecutionStats` records, computes
//! exponential-moving-average selectivity per pattern, and exposes analysis
//! helpers such as `slowest_queries` and `pattern_hit_rate`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pattern-level stats
// ---------------------------------------------------------------------------

/// Execution statistics for a single triple-pattern within a query.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternStats {
    /// A human-readable identifier for the pattern (e.g. `"?s rdf:type ?t"`).
    pub pattern: String,
    /// The cardinality that the optimizer predicted before execution.
    pub cardinality_estimate: usize,
    /// The true cardinality observed during execution.
    pub actual_cardinality: usize,
    /// Wall-clock time spent evaluating this pattern (milliseconds).
    pub evaluation_time_ms: u64,
    /// Whether the result was served from a cache rather than evaluated.
    pub cache_hit: bool,
}

impl PatternStats {
    /// Construct a new `PatternStats` record.
    pub fn new(
        pattern: impl Into<String>,
        cardinality_estimate: usize,
        actual_cardinality: usize,
        evaluation_time_ms: u64,
        cache_hit: bool,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            cardinality_estimate,
            actual_cardinality,
            evaluation_time_ms,
            cache_hit,
        }
    }

    /// Ratio of actual to estimated cardinality; values < 1 mean the
    /// optimizer over-estimated, values > 1 mean it under-estimated.
    pub fn cardinality_ratio(&self) -> f64 {
        self.actual_cardinality as f64 / self.cardinality_estimate.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// Query-level stats
// ---------------------------------------------------------------------------

/// Aggregated execution statistics for a complete SPARQL query evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryExecutionStats {
    /// Unique identifier for the query execution (e.g. UUID or hash string).
    pub query_id: String,
    /// Total wall-clock time for the entire query (milliseconds).
    pub total_time_ms: u64,
    /// Per-pattern breakdown.
    pub pattern_stats: Vec<PatternStats>,
    /// Number of result rows produced.
    pub result_count: usize,
    /// Number of join operations performed.
    pub join_count: usize,
    /// Number of FILTER evaluations performed.
    pub filter_count: usize,
    /// Cache hits encountered during evaluation.
    pub cache_hits: usize,
    /// Cache misses encountered during evaluation.
    pub cache_misses: usize,
}

impl QueryExecutionStats {
    /// Construct a new `QueryExecutionStats` with the given fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_id: impl Into<String>,
        total_time_ms: u64,
        pattern_stats: Vec<PatternStats>,
        result_count: usize,
        join_count: usize,
        filter_count: usize,
        cache_hits: usize,
        cache_misses: usize,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            total_time_ms,
            pattern_stats,
            result_count,
            join_count,
            filter_count,
            cache_hits,
            cache_misses,
        }
    }

    /// Overall cache hit rate `[0.0, 1.0]` for this query execution.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Collector
// ---------------------------------------------------------------------------

/// Smoothing coefficient for exponential moving average selectivity updates.
const EMA_ALPHA: f64 = 0.3;

/// Collects runtime execution statistics and maintains adaptive selectivity
/// estimates per observed triple-pattern.
///
/// The history is bounded: once it reaches `max_history` entries the oldest
/// record is dropped (FIFO).
pub struct RuntimeStatsCollector {
    /// Circular history of query execution records (newest at the back).
    history: Vec<QueryExecutionStats>,
    /// Maximum number of history entries retained.
    max_history: usize,
    /// EMA-based selectivity per pattern key.
    /// `selectivity(p) = actual_cardinality / max(estimated_cardinality, 1)`.
    pattern_selectivity: HashMap<String, f64>,
}

impl RuntimeStatsCollector {
    /// Create a new collector that retains at most `max_history` records.
    ///
    /// `max_history` is clamped to at least 1.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(1),
            pattern_selectivity: HashMap::new(),
        }
    }

    /// Record a completed query execution.
    ///
    /// This also updates the per-pattern selectivity estimates for every
    /// pattern in the provided stats.
    pub fn record(&mut self, stats: QueryExecutionStats) {
        // Update per-pattern selectivity from the new observation.
        for ps in &stats.pattern_stats {
            self.update_selectivity(
                &ps.pattern.clone(),
                ps.actual_cardinality,
                ps.cardinality_estimate,
            );
        }

        // Maintain bounded history (FIFO eviction).
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(stats);
    }

    /// Update the EMA selectivity for `pattern` given a new observation.
    ///
    /// `selectivity = actual / max(estimated, 1)`.
    /// The update rule is: `new_ema = alpha * observation + (1 - alpha) * old_ema`.
    pub fn update_selectivity(&mut self, pattern: &str, actual: usize, estimated: usize) {
        let observation = actual as f64 / estimated.max(1) as f64;
        let entry = self
            .pattern_selectivity
            .entry(pattern.to_string())
            .or_insert(observation);
        // Apply EMA only if a prior estimate exists (otherwise just set it).
        *entry = EMA_ALPHA * observation + (1.0 - EMA_ALPHA) * *entry;
    }

    /// Return the current EMA selectivity estimate for `pattern`.
    ///
    /// Returns `1.0` (no selectivity information) when the pattern has never
    /// been observed.
    pub fn get_selectivity(&self, pattern: &str) -> f64 {
        self.pattern_selectivity
            .get(pattern)
            .copied()
            .unwrap_or(1.0)
    }

    /// Mean query execution time across all records in the history (milliseconds).
    ///
    /// Returns `0.0` when the history is empty.
    pub fn avg_query_time(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.history.iter().map(|q| q.total_time_ms).sum();
        sum as f64 / self.history.len() as f64
    }

    /// Return up to `n` references to the slowest queries, sorted by
    /// `total_time_ms` descending.
    pub fn slowest_queries(&self, n: usize) -> Vec<&QueryExecutionStats> {
        let mut refs: Vec<&QueryExecutionStats> = self.history.iter().collect();
        refs.sort_by(|a, b| b.total_time_ms.cmp(&a.total_time_ms));
        refs.into_iter().take(n).collect()
    }

    /// Fraction of evaluations of `pattern` that were cache hits.
    ///
    /// Returns `0.0` when the pattern has never been observed.
    pub fn pattern_hit_rate(&self, pattern: &str) -> f64 {
        let (hits, total) = self
            .history
            .iter()
            .flat_map(|q| &q.pattern_stats)
            .filter(|ps| ps.pattern == pattern)
            .fold((0usize, 0usize), |(h, t), ps| {
                (if ps.cache_hit { h + 1 } else { h }, t + 1)
            });
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Total number of query records currently stored in the history.
    pub fn total_queries(&self) -> usize {
        self.history.len()
    }

    /// Clear all history and reset all selectivity estimates.
    pub fn reset(&mut self) {
        self.history.clear();
        self.pattern_selectivity.clear();
    }

    /// Immutable access to the full history slice.
    pub fn history(&self) -> &[QueryExecutionStats] {
        &self.history
    }

    /// Number of distinct patterns for which selectivity data has been collected.
    pub fn tracked_pattern_count(&self) -> usize {
        self.pattern_selectivity.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern_stats(
        pattern: &str,
        estimated: usize,
        actual: usize,
        ms: u64,
        cache_hit: bool,
    ) -> PatternStats {
        PatternStats::new(pattern, estimated, actual, ms, cache_hit)
    }

    fn make_query_stats(id: &str, ms: u64, patterns: Vec<PatternStats>) -> QueryExecutionStats {
        let cache_hits = patterns.iter().filter(|p| p.cache_hit).count();
        let cache_misses = patterns.len() - cache_hits;
        QueryExecutionStats::new(id, ms, patterns, 10, 2, 1, cache_hits, cache_misses)
    }

    // ------------------------------------------------------------------
    // PatternStats helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_pattern_stats_construction() {
        let ps = make_pattern_stats("?s rdf:type ?t", 100, 50, 5, false);
        assert_eq!(ps.pattern, "?s rdf:type ?t");
        assert_eq!(ps.cardinality_estimate, 100);
        assert_eq!(ps.actual_cardinality, 50);
        assert_eq!(ps.evaluation_time_ms, 5);
        assert!(!ps.cache_hit);
    }

    #[test]
    fn test_pattern_stats_cardinality_ratio() {
        let ps = make_pattern_stats("?s ?p ?o", 200, 100, 10, false);
        let ratio = ps.cardinality_ratio();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_stats_cardinality_ratio_zero_estimate() {
        // Denominator is clamped to 1 so division by zero cannot occur.
        let ps = make_pattern_stats("?s ?p ?o", 0, 5, 1, false);
        let ratio = ps.cardinality_ratio();
        assert!((ratio - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_stats_cache_hit_flag() {
        let ps = make_pattern_stats("?x owl:sameAs ?y", 50, 50, 0, true);
        assert!(ps.cache_hit);
    }

    // ------------------------------------------------------------------
    // QueryExecutionStats helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_query_stats_cache_hit_rate() {
        let patterns = vec![
            make_pattern_stats("p1", 10, 10, 1, true),
            make_pattern_stats("p2", 10, 10, 1, false),
            make_pattern_stats("p3", 10, 10, 1, true),
            make_pattern_stats("p4", 10, 10, 1, false),
        ];
        let qs = make_query_stats("q1", 100, patterns);
        let rate = qs.cache_hit_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_query_stats_zero_cache_rate_when_no_patterns() {
        let qs = make_query_stats("q_empty", 10, vec![]);
        assert_eq!(qs.cache_hit_rate(), 0.0);
    }

    // ------------------------------------------------------------------
    // RuntimeStatsCollector – basic
    // ------------------------------------------------------------------

    #[test]
    fn test_collector_starts_empty() {
        let collector = RuntimeStatsCollector::new(50);
        assert_eq!(collector.total_queries(), 0);
        assert_eq!(collector.avg_query_time(), 0.0);
        assert_eq!(collector.tracked_pattern_count(), 0);
    }

    #[test]
    fn test_collector_records_single_query() {
        let mut collector = RuntimeStatsCollector::new(50);
        let qs = make_query_stats("q1", 120, vec![make_pattern_stats("p1", 50, 30, 10, false)]);
        collector.record(qs);
        assert_eq!(collector.total_queries(), 1);
        assert_eq!(collector.avg_query_time(), 120.0);
    }

    #[test]
    fn test_collector_avg_time_multiple_queries() {
        let mut collector = RuntimeStatsCollector::new(50);
        collector.record(make_query_stats("q1", 100, vec![]));
        collector.record(make_query_stats("q2", 200, vec![]));
        collector.record(make_query_stats("q3", 300, vec![]));
        let avg = collector.avg_query_time();
        assert!((avg - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_collector_history_bounded() {
        let max = 3;
        let mut collector = RuntimeStatsCollector::new(max);
        for i in 0..6u64 {
            collector.record(make_query_stats(&format!("q{i}"), i * 10, vec![]));
        }
        // History must not exceed the configured maximum.
        assert_eq!(collector.total_queries(), max);
        // Oldest queries should have been evicted; newest are retained.
        let ids: Vec<&str> = collector
            .history()
            .iter()
            .map(|q| q.query_id.as_str())
            .collect();
        assert_eq!(ids, vec!["q3", "q4", "q5"]);
    }

    #[test]
    fn test_collector_reset_clears_everything() {
        let mut collector = RuntimeStatsCollector::new(10);
        collector.record(make_query_stats("q1", 100, vec![]));
        collector.update_selectivity("p1", 10, 20);
        collector.reset();
        assert_eq!(collector.total_queries(), 0);
        assert_eq!(collector.tracked_pattern_count(), 0);
        assert_eq!(collector.avg_query_time(), 0.0);
    }

    // ------------------------------------------------------------------
    // Selectivity EMA
    // ------------------------------------------------------------------

    #[test]
    fn test_get_selectivity_unknown_pattern_returns_one() {
        let collector = RuntimeStatsCollector::new(10);
        assert_eq!(collector.get_selectivity("unknown"), 1.0);
    }

    #[test]
    fn test_update_selectivity_sets_initial_value() {
        let mut collector = RuntimeStatsCollector::new(10);
        // actual=10, estimated=100 → observation = 0.1
        collector.update_selectivity("p1", 10, 100);
        // After first update: EMA = 0.3 * 0.1 + 0.7 * 0.1 = 0.1
        let sel = collector.get_selectivity("p1");
        assert!((sel - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_selectivity_ema_converges() {
        let mut collector = RuntimeStatsCollector::new(100);
        // Repeatedly observe that selectivity = 0.5 (actual == estimated/2).
        for _ in 0..200 {
            collector.update_selectivity("p_conv", 50, 100);
        }
        let sel = collector.get_selectivity("p_conv");
        // After many identical observations the EMA should converge to 0.5.
        assert!((sel - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_selectivity_updated_on_record() {
        let mut collector = RuntimeStatsCollector::new(50);
        let ps = make_pattern_stats("?x :p ?y", 100, 10, 5, false);
        collector.record(make_query_stats("q1", 50, vec![ps]));
        // Pattern should now have a selectivity entry.
        assert_ne!(collector.get_selectivity("?x :p ?y"), 1.0);
    }

    // ------------------------------------------------------------------
    // slowest_queries
    // ------------------------------------------------------------------

    #[test]
    fn test_slowest_queries_returns_descending_order() {
        let mut collector = RuntimeStatsCollector::new(20);
        collector.record(make_query_stats("fast", 10, vec![]));
        collector.record(make_query_stats("slow", 500, vec![]));
        collector.record(make_query_stats("medium", 150, vec![]));

        let top2 = collector.slowest_queries(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].query_id, "slow");
        assert_eq!(top2[1].query_id, "medium");
    }

    #[test]
    fn test_slowest_queries_n_exceeds_history() {
        let mut collector = RuntimeStatsCollector::new(10);
        collector.record(make_query_stats("q1", 100, vec![]));
        // Requesting more than available should return all records.
        let result = collector.slowest_queries(50);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_slowest_queries_empty_history() {
        let collector = RuntimeStatsCollector::new(10);
        let result = collector.slowest_queries(5);
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // pattern_hit_rate
    // ------------------------------------------------------------------

    #[test]
    fn test_pattern_hit_rate_all_hits() {
        let mut collector = RuntimeStatsCollector::new(20);
        for _ in 0..4 {
            collector.record(make_query_stats(
                "q",
                10,
                vec![make_pattern_stats("hot_pattern", 10, 10, 0, true)],
            ));
        }
        let rate = collector.pattern_hit_rate("hot_pattern");
        assert!((rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_hit_rate_no_hits() {
        let mut collector = RuntimeStatsCollector::new(20);
        for _ in 0..3 {
            collector.record(make_query_stats(
                "q",
                10,
                vec![make_pattern_stats("cold_pattern", 10, 10, 5, false)],
            ));
        }
        assert_eq!(collector.pattern_hit_rate("cold_pattern"), 0.0);
    }

    #[test]
    fn test_pattern_hit_rate_mixed() {
        let mut collector = RuntimeStatsCollector::new(20);
        // 2 hits, 2 misses → 50%
        collector.record(make_query_stats(
            "q1",
            10,
            vec![make_pattern_stats("mp", 10, 10, 1, true)],
        ));
        collector.record(make_query_stats(
            "q2",
            10,
            vec![make_pattern_stats("mp", 10, 10, 1, false)],
        ));
        collector.record(make_query_stats(
            "q3",
            10,
            vec![make_pattern_stats("mp", 10, 10, 1, true)],
        ));
        collector.record(make_query_stats(
            "q4",
            10,
            vec![make_pattern_stats("mp", 10, 10, 1, false)],
        ));
        let rate = collector.pattern_hit_rate("mp");
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_hit_rate_unknown_pattern() {
        let collector = RuntimeStatsCollector::new(10);
        assert_eq!(collector.pattern_hit_rate("never_seen"), 0.0);
    }

    // ------------------------------------------------------------------
    // tracked_pattern_count
    // ------------------------------------------------------------------

    #[test]
    fn test_tracked_pattern_count_grows() {
        let mut collector = RuntimeStatsCollector::new(20);
        collector.update_selectivity("p1", 5, 10);
        collector.update_selectivity("p2", 5, 10);
        assert_eq!(collector.tracked_pattern_count(), 2);
    }

    #[test]
    fn test_tracked_pattern_count_no_duplicate() {
        let mut collector = RuntimeStatsCollector::new(20);
        collector.update_selectivity("p1", 5, 10);
        collector.update_selectivity("p1", 6, 10);
        // Same key – must not create a duplicate entry.
        assert_eq!(collector.tracked_pattern_count(), 1);
    }
}
