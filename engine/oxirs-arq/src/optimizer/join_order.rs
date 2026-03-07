//! Adaptive Join Ordering with Cost-Based Optimization
//!
//! This module implements cost-based join reordering for SPARQL query optimization.
//! It uses selectivity estimation and bound variable counting to order triple patterns
//! so that the most selective patterns execute first, reducing intermediate result sizes.
//!
//! # Key Types
//!
//! - [`JoinOrderOptimizer`] — stateless optimizer that reorders triple patterns
//! - [`CardinalityEstimate`] — min/max/expected cardinality range
//! - [`PatternCost`] — cost estimate combining cardinality and selectivity
//! - [`JoinGraphStats`] — graph statistics used for estimation
//! - [`AdaptiveQueryPlan`] — runtime-adaptive plan that can reorder itself

use crate::algebra::{Term, TriplePattern};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Cardinality / Cost types
// ---------------------------------------------------------------------------

/// Estimated cardinality range for a triple pattern or join result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardinalityEstimate {
    /// Minimum possible result count (optimistic).
    pub min: u64,
    /// Maximum possible result count (pessimistic).
    pub max: u64,
    /// Expected result count (used for join ordering decisions).
    pub expected: u64,
}

impl CardinalityEstimate {
    /// Create a new estimate with explicit min/max/expected.
    pub fn new(min: u64, max: u64, expected: u64) -> Self {
        Self { min, max, expected }
    }

    /// Single-point estimate (min == max == expected).
    pub fn exact(count: u64) -> Self {
        Self {
            min: count,
            max: count,
            expected: count,
        }
    }

    /// Unknown estimate — uses full triple count as upper bound.
    pub fn unknown(triple_count: u64) -> Self {
        Self {
            min: 0,
            max: triple_count,
            expected: triple_count / 2 + 1,
        }
    }
}

impl std::fmt::Display for CardinalityEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}..{}, ~{}]", self.min, self.max, self.expected)
    }
}

/// Cost estimate for a single triple pattern.
#[derive(Debug, Clone)]
pub struct PatternCost {
    /// Estimated cardinality of the result set.
    pub estimated_cardinality: u64,
    /// Selectivity factor in (0.0, 1.0] — lower is more selective.
    pub selectivity: f64,
    /// Number of bound (non-variable) terms in the pattern (0..=3).
    pub bound_count: u8,
}

impl PatternCost {
    /// Composite score used for greedy ordering: lower is better (execute first).
    ///
    /// Combines cardinality and selectivity, biased toward patterns with more bound terms.
    pub fn ordering_score(&self) -> f64 {
        let base = (self.estimated_cardinality as f64 + 1.0) * self.selectivity;
        // Patterns with more bound terms score lower (executed earlier).
        let bound_bias = 1.0 / (1.0 + self.bound_count as f64);
        base * bound_bias
    }
}

// ---------------------------------------------------------------------------
// Graph statistics
// ---------------------------------------------------------------------------

/// Graph statistics used by the join-order optimizer for cardinality estimation.
///
/// Intentionally named `JoinGraphStats` to avoid collision with
/// `analytics::GraphStats`.
#[derive(Debug, Clone, Default)]
pub struct JoinGraphStats {
    /// Total number of triples in the graph.
    pub triple_count: u64,
    /// Number of distinct subjects.
    pub distinct_subjects: u64,
    /// Number of distinct predicates.
    pub distinct_predicates: u64,
    /// Number of distinct objects.
    pub distinct_objects: u64,
    /// Per-predicate triple counts (predicate URI → count).
    pub per_predicate_counts: HashMap<String, u64>,
}

impl JoinGraphStats {
    /// Create a new stats object with the given triple count.
    pub fn new(triple_count: u64) -> Self {
        Self {
            triple_count,
            distinct_subjects: (triple_count / 3).max(1),
            distinct_predicates: (triple_count / 10).max(1),
            distinct_objects: (triple_count / 2).max(1),
            per_predicate_counts: HashMap::new(),
        }
    }

    /// Create fully specified stats.
    pub fn with_details(
        triple_count: u64,
        distinct_subjects: u64,
        distinct_predicates: u64,
        distinct_objects: u64,
        per_predicate_counts: HashMap<String, u64>,
    ) -> Self {
        Self {
            triple_count,
            distinct_subjects,
            distinct_predicates,
            distinct_objects,
            per_predicate_counts,
        }
    }

    /// Insert a predicate count entry.
    pub fn add_predicate_count(&mut self, predicate: impl Into<String>, count: u64) {
        self.per_predicate_counts.insert(predicate.into(), count);
    }

    /// Estimated selectivity of a given predicate (fraction of triples with this predicate).
    pub fn predicate_selectivity(&self, predicate_uri: &str) -> f64 {
        if self.triple_count == 0 {
            return 1.0;
        }
        let count = self
            .per_predicate_counts
            .get(predicate_uri)
            .copied()
            .unwrap_or(self.triple_count / self.distinct_predicates.max(1));
        (count as f64) / (self.triple_count as f64)
    }
}

// ---------------------------------------------------------------------------
// JoinOrderOptimizer
// ---------------------------------------------------------------------------

/// Cost-based join reordering optimizer for SPARQL triple patterns.
///
/// The optimizer is stateless — it uses `JoinGraphStats` passed at call time
/// to produce an ordered list of patterns that minimises intermediate result sizes.
///
/// # Algorithm
///
/// A greedy approach is used:
/// 1. Compute a `PatternCost` for each pattern based on bound-variable count
///    and cardinality estimate.
/// 2. Sort patterns by `PatternCost::ordering_score()` (ascending).
///
/// For small pattern lists (≤ 4), a dynamic-programming exhaustive search is
/// performed instead.
pub struct JoinOrderOptimizer {
    /// Maximum number of patterns below which DP exhaustive search is used.
    dp_threshold: usize,
}

impl JoinOrderOptimizer {
    /// Create a new optimizer with default settings.
    pub fn new() -> Self {
        Self { dp_threshold: 4 }
    }

    /// Set the DP threshold.  When `patterns.len() <= dp_threshold`, an exhaustive
    /// search is performed instead of greedy ordering.
    pub fn with_dp_threshold(mut self, threshold: usize) -> Self {
        self.dp_threshold = threshold;
        self
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Count the number of bound (non-variable) terms in a pattern.
    pub fn bound_variable_count(pattern: &TriplePattern) -> u8 {
        let subject_bound = !matches!(pattern.subject, Term::Variable(_));
        let predicate_bound = !matches!(pattern.predicate, Term::Variable(_));
        let object_bound = !matches!(pattern.object, Term::Variable(_));
        subject_bound as u8 + predicate_bound as u8 + object_bound as u8
    }

    /// Estimate the cardinality of a single triple pattern given graph statistics.
    pub fn estimate_cardinality(
        pattern: &TriplePattern,
        stats: &JoinGraphStats,
    ) -> CardinalityEstimate {
        if stats.triple_count == 0 {
            return CardinalityEstimate::exact(0);
        }

        let bound = Self::bound_variable_count(pattern);

        match bound {
            3 => {
                // Fully bound — at most 1 result.
                CardinalityEstimate::new(0, 1, 1)
            }
            2 => {
                // Two bound terms — highly selective.
                let expected = match (&pattern.subject, &pattern.predicate, &pattern.object) {
                    (Term::Variable(_), _, _) => {
                        // ?s <p> <o>
                        let pred_sel =
                            Self::predicate_selectivity_from_term(&pattern.predicate, stats);
                        ((stats.triple_count as f64 * pred_sel)
                            / stats.distinct_objects.max(1) as f64) as u64
                    }
                    (_, Term::Variable(_), _) => {
                        // <s> ?p <o>
                        (stats.triple_count / stats.distinct_subjects.max(1))
                            / stats.distinct_objects.max(1)
                    }
                    (_, _, Term::Variable(_)) => {
                        // <s> <p> ?o
                        let pred_sel =
                            Self::predicate_selectivity_from_term(&pattern.predicate, stats);
                        (stats.triple_count as f64 * pred_sel
                            / stats.distinct_subjects.max(1) as f64) as u64
                    }
                    _ => 1,
                };
                let expected = expected.max(1);
                CardinalityEstimate::new(0, expected * 2, expected)
            }
            1 => {
                // One bound term — moderately selective.
                let expected = match (&pattern.subject, &pattern.predicate, &pattern.object) {
                    (_, Term::Variable(_), Term::Variable(_)) => {
                        // <s> ?p ?o
                        stats.triple_count / stats.distinct_subjects.max(1)
                    }
                    (Term::Variable(_), _, Term::Variable(_)) => {
                        // ?s <p> ?o
                        let pred_sel =
                            Self::predicate_selectivity_from_term(&pattern.predicate, stats);
                        (stats.triple_count as f64 * pred_sel) as u64
                    }
                    (Term::Variable(_), Term::Variable(_), _) => {
                        // ?s ?p <o>
                        stats.triple_count / stats.distinct_objects.max(1)
                    }
                    _ => stats.triple_count / 3,
                };
                let expected = expected.max(1);
                CardinalityEstimate::new(0, stats.triple_count, expected)
            }
            _ => {
                // No bound terms — full scan.
                CardinalityEstimate::unknown(stats.triple_count)
            }
        }
    }

    /// Compute the full cost of a triple pattern.
    pub fn pattern_cost(pattern: &TriplePattern, stats: &JoinGraphStats) -> PatternCost {
        let bound_count = Self::bound_variable_count(pattern);
        let estimate = Self::estimate_cardinality(pattern, stats);
        let selectivity = if stats.triple_count > 0 {
            (estimate.expected as f64) / (stats.triple_count as f64)
        } else {
            1.0
        };
        PatternCost {
            estimated_cardinality: estimate.expected,
            selectivity: selectivity.clamp(1e-9, 1.0),
            bound_count,
        }
    }

    /// Reorder triple patterns for optimal join ordering.
    ///
    /// Uses exhaustive DP for small lists (≤ `dp_threshold`) and greedy ordering
    /// for larger lists.
    pub fn reorder_joins(
        &self,
        patterns: &[TriplePattern],
        stats: &JoinGraphStats,
    ) -> Vec<TriplePattern> {
        if patterns.len() <= 1 {
            return patterns.to_vec();
        }

        if patterns.len() <= self.dp_threshold {
            self.dp_reorder(patterns, stats)
        } else {
            self.greedy_reorder(patterns, stats)
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Greedy join reordering: sort by PatternCost::ordering_score() ascending.
    fn greedy_reorder(
        &self,
        patterns: &[TriplePattern],
        stats: &JoinGraphStats,
    ) -> Vec<TriplePattern> {
        let mut scored: Vec<(f64, TriplePattern)> = patterns
            .iter()
            .map(|p| {
                let cost = Self::pattern_cost(p, stats);
                (cost.ordering_score(), p.clone())
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, p)| p).collect()
    }

    /// DP exhaustive reordering for small pattern sets (n! permutations, n ≤ 4 → ≤ 24).
    fn dp_reorder(&self, patterns: &[TriplePattern], stats: &JoinGraphStats) -> Vec<TriplePattern> {
        let n = patterns.len();
        if n == 0 {
            return vec![];
        }

        let costs: Vec<PatternCost> = patterns
            .iter()
            .map(|p| Self::pattern_cost(p, stats))
            .collect();

        // DP over subsets: dp[mask] = (best_cost, last_pattern_index)
        let full_mask = (1usize << n) - 1;
        let mut dp = vec![(f64::INFINITY, usize::MAX); 1 << n];
        let mut parent = vec![(usize::MAX, usize::MAX); 1 << n];
        dp[0] = (0.0, usize::MAX);

        for mask in 0..=full_mask {
            if dp[mask].0 == f64::INFINITY && mask != 0 {
                continue;
            }
            // Position of the next pattern to place (0 = first, n-1 = last).
            let position = mask.count_ones() as usize;
            // Weight: patterns placed early carry more influence on total cost
            // (remaining patterns will join against this result).
            let weight = (n - position) as f64;
            for (i, cost) in costs.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    continue;
                }
                let new_mask = mask | (1 << i);
                let step_cost = cost.ordering_score() * weight;
                let new_cost = dp[mask].0 + step_cost;
                if new_cost < dp[new_mask].0 {
                    dp[new_mask] = (new_cost, i);
                    parent[new_mask] = (mask, i);
                }
            }
        }

        // Reconstruct path.
        let mut order = Vec::with_capacity(n);
        let mut mask = full_mask;
        while mask != 0 {
            let (prev_mask, idx) = parent[mask];
            if idx == usize::MAX {
                break;
            }
            order.push(idx);
            mask = prev_mask;
        }
        order.reverse();
        order.iter().map(|&i| patterns[i].clone()).collect()
    }

    /// Extract predicate selectivity from a Term (if it's a named node).
    fn predicate_selectivity_from_term(term: &Term, stats: &JoinGraphStats) -> f64 {
        if let Term::Iri(iri) = term {
            stats.predicate_selectivity(iri.as_str())
        } else {
            1.0 / stats.distinct_predicates.max(1) as f64
        }
    }
}

impl Default for JoinOrderOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AdaptiveQueryPlan
// ---------------------------------------------------------------------------

/// A query plan that can reorder its triple patterns at runtime based on
/// observed intermediate result sizes.
///
/// The plan starts with an optimized static ordering produced by
/// [`JoinOrderOptimizer`], then checks after each pattern execution whether
/// the actual cardinality diverges enough from the estimate to warrant
/// re-ordering the remaining patterns.
#[derive(Debug, Clone)]
pub struct AdaptiveQueryPlan {
    /// Current ordered list of triple patterns.
    pub patterns: Vec<TriplePattern>,
    /// Pre-computed cost estimates (same order as `patterns`).
    pub estimated_costs: Vec<PatternCost>,
    /// How much actual cardinality must differ from estimate (as a ratio)
    /// before we trigger a re-order.  Default: 5.0 (5×).
    pub reorder_threshold: u64,
    /// Index of the next pattern to execute.
    current_index: usize,
    /// Creation timestamp.
    created_at: Instant,
}

impl AdaptiveQueryPlan {
    /// Create a plan from a slice of patterns, immediately optimizing their order.
    pub fn new(patterns: Vec<TriplePattern>, stats: &JoinGraphStats) -> Self {
        let optimizer = JoinOrderOptimizer::new();
        let ordered = optimizer.reorder_joins(&patterns, stats);
        let estimated_costs = ordered
            .iter()
            .map(|p| JoinOrderOptimizer::pattern_cost(p, stats))
            .collect();
        Self {
            patterns: ordered,
            estimated_costs,
            reorder_threshold: 5,
            current_index: 0,
            created_at: Instant::now(),
        }
    }

    /// Set the reorder threshold ratio (actual / expected must exceed this).
    pub fn with_reorder_threshold(mut self, threshold: u64) -> Self {
        self.reorder_threshold = threshold;
        self
    }

    /// Check whether the observed actual cardinality for `pattern_idx` is far
    /// enough from the estimate to justify reordering remaining patterns.
    pub fn should_reorder(&self, actual_cardinality: u64, pattern_idx: usize) -> bool {
        if pattern_idx >= self.estimated_costs.len() {
            return false;
        }
        let estimated = self.estimated_costs[pattern_idx].estimated_cardinality;
        if estimated == 0 {
            return actual_cardinality > 0;
        }
        let ratio = if actual_cardinality > estimated {
            actual_cardinality / estimated
        } else {
            estimated / actual_cardinality.max(1)
        };
        ratio >= self.reorder_threshold
    }

    /// Reorder the remaining patterns (starting at `pattern_idx`) given updated
    /// stats derived from observed intermediate results.
    pub fn reorder_from(&mut self, pattern_idx: usize, stats: &JoinGraphStats) {
        if pattern_idx >= self.patterns.len() {
            return;
        }
        let remaining = self.patterns[pattern_idx..].to_vec();
        let optimizer = JoinOrderOptimizer::new();
        let reordered = optimizer.reorder_joins(&remaining, stats);
        let new_costs: Vec<PatternCost> = reordered
            .iter()
            .map(|p| JoinOrderOptimizer::pattern_cost(p, stats))
            .collect();
        // Splice back.
        self.patterns.splice(pattern_idx.., reordered);
        self.estimated_costs.splice(pattern_idx.., new_costs);
    }

    /// Advance the plan to the next pattern.  Returns the next pattern or `None`
    /// if the plan is exhausted.
    pub fn next_pattern(&mut self) -> Option<&TriplePattern> {
        if self.current_index < self.patterns.len() {
            let p = &self.patterns[self.current_index];
            self.current_index += 1;
            Some(p)
        } else {
            None
        }
    }

    /// Reset execution cursor to the beginning.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Number of patterns remaining (not yet executed).
    pub fn remaining(&self) -> usize {
        self.patterns.len().saturating_sub(self.current_index)
    }

    /// Elapsed time since plan creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Total number of patterns in the plan.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the plan has no patterns.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Term;
    use oxirs_core::model::NamedNode;

    // Helper to create a Term::Iri from a string.
    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new(s).expect("valid IRI"))
    }

    // Helper to create a Term::Variable.
    fn var(name: &str) -> Term {
        use oxirs_core::model::Variable;
        Term::Variable(Variable::new(name).expect("valid variable name"))
    }

    fn make_pattern(s: Term, p: Term, o: Term) -> TriplePattern {
        TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    fn default_stats() -> JoinGraphStats {
        let mut stats = JoinGraphStats::new(10_000);
        stats.add_predicate_count("http://ex.org/type", 500);
        stats.add_predicate_count("http://ex.org/name", 8_000);
        stats.add_predicate_count("http://ex.org/rare", 10);
        stats
    }

    // ------------------------------------------------------------------
    // CardinalityEstimate tests
    // ------------------------------------------------------------------

    #[test]
    fn test_cardinality_estimate_exact() {
        let est = CardinalityEstimate::exact(42);
        assert_eq!(est.min, 42);
        assert_eq!(est.max, 42);
        assert_eq!(est.expected, 42);
    }

    #[test]
    fn test_cardinality_estimate_unknown() {
        let est = CardinalityEstimate::unknown(1_000);
        assert_eq!(est.min, 0);
        assert!(est.expected > 0);
        assert!(est.max >= est.expected);
    }

    #[test]
    fn test_cardinality_estimate_display() {
        let est = CardinalityEstimate::new(1, 100, 50);
        let s = format!("{est}");
        assert!(s.contains("1..100"));
    }

    // ------------------------------------------------------------------
    // JoinGraphStats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_graph_stats_new() {
        let stats = JoinGraphStats::new(1_000);
        assert_eq!(stats.triple_count, 1_000);
        assert!(stats.distinct_subjects > 0);
    }

    #[test]
    fn test_predicate_selectivity_known() {
        let mut stats = JoinGraphStats::new(10_000);
        stats.add_predicate_count("http://ex.org/p", 1_000);
        let sel = stats.predicate_selectivity("http://ex.org/p");
        assert!((sel - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_predicate_selectivity_unknown() {
        let stats = JoinGraphStats::new(10_000);
        let sel = stats.predicate_selectivity("http://ex.org/unknown");
        assert!(sel > 0.0 && sel <= 1.0);
    }

    #[test]
    fn test_predicate_selectivity_zero_triples() {
        let stats = JoinGraphStats::new(0);
        let sel = stats.predicate_selectivity("http://ex.org/p");
        assert_eq!(sel, 1.0);
    }

    // ------------------------------------------------------------------
    // bound_variable_count tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bound_count_all_variables() {
        let p = make_pattern(var("s"), var("p"), var("o"));
        assert_eq!(JoinOrderOptimizer::bound_variable_count(&p), 0);
    }

    #[test]
    fn test_bound_count_one_bound() {
        let p = make_pattern(iri("http://ex.org/s"), var("p"), var("o"));
        assert_eq!(JoinOrderOptimizer::bound_variable_count(&p), 1);
    }

    #[test]
    fn test_bound_count_two_bound() {
        let p = make_pattern(iri("http://ex.org/s"), iri("http://ex.org/p"), var("o"));
        assert_eq!(JoinOrderOptimizer::bound_variable_count(&p), 2);
    }

    #[test]
    fn test_bound_count_all_bound() {
        let p = make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        assert_eq!(JoinOrderOptimizer::bound_variable_count(&p), 3);
    }

    // ------------------------------------------------------------------
    // estimate_cardinality tests
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_fully_bound() {
        let stats = default_stats();
        let p = make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        let est = JoinOrderOptimizer::estimate_cardinality(&p, &stats);
        assert_eq!(est.max, 1);
    }

    #[test]
    fn test_estimate_two_bound_spo() {
        let stats = default_stats();
        // <s> <p> ?o  — two bound
        let p = make_pattern(iri("http://ex.org/s"), iri("http://ex.org/type"), var("o"));
        let est = JoinOrderOptimizer::estimate_cardinality(&p, &stats);
        assert!(est.expected > 0);
    }

    #[test]
    fn test_estimate_full_scan() {
        let stats = default_stats();
        let p = make_pattern(var("s"), var("p"), var("o"));
        let est = JoinOrderOptimizer::estimate_cardinality(&p, &stats);
        assert_eq!(est.max, stats.triple_count);
    }

    #[test]
    fn test_estimate_zero_graph() {
        let stats = JoinGraphStats::new(0);
        let p = make_pattern(var("s"), var("p"), var("o"));
        let est = JoinOrderOptimizer::estimate_cardinality(&p, &stats);
        assert_eq!(est.expected, 0);
    }

    // ------------------------------------------------------------------
    // pattern_cost tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pattern_cost_ordering_score_fully_bound_lowest() {
        let stats = default_stats();
        let fully_bound = make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        let full_scan = make_pattern(var("s"), var("p"), var("o"));
        let cost_bound = JoinOrderOptimizer::pattern_cost(&fully_bound, &stats);
        let cost_scan = JoinOrderOptimizer::pattern_cost(&full_scan, &stats);
        assert!(
            cost_bound.ordering_score() < cost_scan.ordering_score(),
            "fully bound should score lower than full scan"
        );
    }

    #[test]
    fn test_pattern_cost_selectivity_range() {
        let stats = default_stats();
        let p = make_pattern(var("s"), iri("http://ex.org/type"), var("o"));
        let cost = JoinOrderOptimizer::pattern_cost(&p, &stats);
        assert!(cost.selectivity > 0.0);
        assert!(cost.selectivity <= 1.0);
    }

    // ------------------------------------------------------------------
    // reorder_joins tests
    // ------------------------------------------------------------------

    #[test]
    fn test_reorder_empty() {
        let optimizer = JoinOrderOptimizer::new();
        let stats = default_stats();
        let result = optimizer.reorder_joins(&[], &stats);
        assert!(result.is_empty());
    }

    #[test]
    fn test_reorder_single() {
        let optimizer = JoinOrderOptimizer::new();
        let stats = default_stats();
        let p = make_pattern(var("s"), var("p"), var("o"));
        let result = optimizer.reorder_joins(std::slice::from_ref(&p), &stats);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], p);
    }

    #[test]
    fn test_reorder_places_selective_first() {
        let optimizer = JoinOrderOptimizer::new();
        let stats = default_stats();

        // Fully bound = most selective.
        let fully_bound = make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/type"),
            iri("http://ex.org/o"),
        );
        // Full scan = least selective.
        let full_scan = make_pattern(var("s"), var("p"), var("o"));
        // One bound term.
        let one_bound = make_pattern(iri("http://ex.org/s"), var("p"), var("o"));

        let patterns = vec![full_scan.clone(), one_bound.clone(), fully_bound.clone()];
        let result = optimizer.reorder_joins(&patterns, &stats);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], fully_bound, "fully bound should come first");
    }

    #[test]
    fn test_reorder_greedy_large_list() {
        let optimizer = JoinOrderOptimizer::new().with_dp_threshold(2);
        let stats = default_stats();

        let patterns: Vec<TriplePattern> = (0..6)
            .map(|i| {
                if i % 2 == 0 {
                    make_pattern(
                        iri(&format!("http://ex.org/s{i}")),
                        iri("http://ex.org/type"),
                        var("o"),
                    )
                } else {
                    make_pattern(var("s"), var("p"), var("o"))
                }
            })
            .collect();

        let result = optimizer.reorder_joins(&patterns, &stats);
        assert_eq!(result.len(), 6);
        // First result should have a lower ordering score (more selective)
        let first_cost = JoinOrderOptimizer::pattern_cost(&result[0], &stats);
        let last_cost = JoinOrderOptimizer::pattern_cost(&result[5], &stats);
        assert!(first_cost.ordering_score() <= last_cost.ordering_score());
    }

    #[test]
    fn test_reorder_dp_vs_greedy_equivalence_small() {
        // For a small list, DP and greedy should both find a reasonable ordering.
        let optimizer_dp = JoinOrderOptimizer::new().with_dp_threshold(5);
        let optimizer_greedy = JoinOrderOptimizer::new().with_dp_threshold(0);
        let stats = default_stats();

        let patterns = vec![
            make_pattern(var("s"), var("p"), var("o")),
            make_pattern(iri("http://ex.org/s"), iri("http://ex.org/type"), var("o")),
            make_pattern(iri("http://ex.org/s"), var("p"), var("o")),
        ];

        let dp_result = optimizer_dp.reorder_joins(&patterns, &stats);
        let greedy_result = optimizer_greedy.reorder_joins(&patterns, &stats);

        // Both should have the same length.
        assert_eq!(dp_result.len(), greedy_result.len());
    }

    // ------------------------------------------------------------------
    // AdaptiveQueryPlan tests
    // ------------------------------------------------------------------

    #[test]
    fn test_adaptive_plan_creation() {
        let stats = default_stats();
        let patterns = vec![
            make_pattern(var("s"), var("p"), var("o")),
            make_pattern(iri("http://ex.org/s"), iri("http://ex.org/type"), var("o")),
        ];
        let plan = AdaptiveQueryPlan::new(patterns, &stats);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_adaptive_plan_should_reorder_large_divergence() {
        let stats = default_stats();
        let patterns = vec![make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/type"),
            var("o"),
        )];
        let plan = AdaptiveQueryPlan::new(patterns, &stats).with_reorder_threshold(5);

        // Estimated cardinality for that pattern should be small; actual is huge.
        let actual = plan.estimated_costs[0].estimated_cardinality * 100;
        assert!(plan.should_reorder(actual, 0));
    }

    #[test]
    fn test_adaptive_plan_should_not_reorder_small_divergence() {
        let stats = default_stats();
        let patterns = vec![make_pattern(
            iri("http://ex.org/s"),
            iri("http://ex.org/type"),
            var("o"),
        )];
        let plan = AdaptiveQueryPlan::new(patterns, &stats).with_reorder_threshold(10);

        let estimated = plan.estimated_costs[0].estimated_cardinality;
        // Actual is only 2× the estimate — below threshold.
        let actual = estimated * 2;
        assert!(!plan.should_reorder(actual, 0));
    }

    #[test]
    fn test_adaptive_plan_reorder_from() {
        let stats = default_stats();
        let patterns = vec![
            make_pattern(var("s"), var("p"), var("o")),
            make_pattern(var("s2"), var("p2"), var("o2")),
            make_pattern(iri("http://ex.org/s"), iri("http://ex.org/type"), var("o")),
        ];
        let mut plan = AdaptiveQueryPlan::new(patterns, &stats);
        // Reorder remaining from index 1.
        plan.reorder_from(1, &stats);
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn test_adaptive_plan_next_pattern() {
        let stats = default_stats();
        let patterns = vec![
            make_pattern(var("s"), var("p"), var("o")),
            make_pattern(iri("http://ex.org/s"), var("p"), var("o")),
        ];
        let mut plan = AdaptiveQueryPlan::new(patterns, &stats);
        assert!(plan.next_pattern().is_some());
        assert!(plan.next_pattern().is_some());
        assert!(plan.next_pattern().is_none());
    }

    #[test]
    fn test_adaptive_plan_reset() {
        let stats = default_stats();
        let patterns = vec![make_pattern(var("s"), var("p"), var("o"))];
        let mut plan = AdaptiveQueryPlan::new(patterns, &stats);
        plan.next_pattern();
        assert_eq!(plan.remaining(), 0);
        plan.reset();
        assert_eq!(plan.remaining(), 1);
    }

    #[test]
    fn test_adaptive_plan_remaining() {
        let stats = default_stats();
        let patterns = vec![
            make_pattern(var("s"), var("p"), var("o")),
            make_pattern(var("a"), var("b"), var("c")),
            make_pattern(var("x"), var("y"), var("z")),
        ];
        let mut plan = AdaptiveQueryPlan::new(patterns, &stats);
        assert_eq!(plan.remaining(), 3);
        plan.next_pattern();
        assert_eq!(plan.remaining(), 2);
    }

    #[test]
    fn test_adaptive_plan_elapsed() {
        let stats = default_stats();
        let patterns = vec![make_pattern(var("s"), var("p"), var("o"))];
        let plan = AdaptiveQueryPlan::new(patterns, &stats);
        // Elapsed should be very small (just created).
        assert!(plan.elapsed().as_secs() < 5);
    }

    #[test]
    fn test_adaptive_plan_out_of_bounds_reorder() {
        let stats = default_stats();
        let patterns = vec![make_pattern(var("s"), var("p"), var("o"))];
        let mut plan = AdaptiveQueryPlan::new(patterns, &stats);
        // Should not panic for out-of-bounds index.
        plan.reorder_from(100, &stats);
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn test_adaptive_plan_should_reorder_zero_estimated() {
        let stats = JoinGraphStats::new(0);
        let patterns = vec![make_pattern(var("s"), var("p"), var("o"))];
        let plan = AdaptiveQueryPlan::new(patterns, &stats);
        // With zero estimated cardinality, actual > 0 triggers reorder.
        assert!(plan.should_reorder(1, 0));
        // actual == 0 does not trigger.
        assert!(!plan.should_reorder(0, 0));
    }
}
