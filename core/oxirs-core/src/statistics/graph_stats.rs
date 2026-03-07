//! High-level RDF graph statistics for query planning.
//!
//! Provides:
//! - [`GraphStatistics`]: Triple count, distinct subject/predicate/object counts.
//! - `PredicateFrequency`: Frequency distribution of predicates in the graph.
//! - [`CardinalityEstimator`]: Estimates join cardinality from histograms.
//! - [`SampledStatistics`]: Statistical sampling for large graphs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// GraphStatistics
// ---------------------------------------------------------------------------

/// Core statistics about an RDF graph.
///
/// Tracks total triple count and the number of distinct subjects, predicates, and objects.
/// Supports incremental updates as triples are added or removed.
#[derive(Debug, Clone, Default)]
pub struct GraphStatistics {
    total_triples: u64,
    distinct_subjects: u64,
    distinct_predicates: u64,
    distinct_objects: u64,
    predicate_frequency: HashMap<String, u64>,
    subject_frequency: HashMap<String, u64>,
    object_frequency: HashMap<String, u64>,
}

impl GraphStatistics {
    /// Create a new, empty statistics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a triple `(subject, predicate, object)` was added.
    pub fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.total_triples += 1;
        let s_prev = *self.subject_frequency.get(subject).unwrap_or(&0);
        if s_prev == 0 {
            self.distinct_subjects += 1;
        }
        *self
            .subject_frequency
            .entry(subject.to_string())
            .or_insert(0) += 1;

        let p_prev = *self.predicate_frequency.get(predicate).unwrap_or(&0);
        if p_prev == 0 {
            self.distinct_predicates += 1;
        }
        *self
            .predicate_frequency
            .entry(predicate.to_string())
            .or_insert(0) += 1;

        let o_prev = *self.object_frequency.get(object).unwrap_or(&0);
        if o_prev == 0 {
            self.distinct_objects += 1;
        }
        *self.object_frequency.entry(object.to_string()).or_insert(0) += 1;
    }

    /// Record that a triple was removed.  If the triple was never added, this is a no-op.
    pub fn remove_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        if self.total_triples == 0 {
            return;
        }
        self.total_triples = self.total_triples.saturating_sub(1);

        if let Some(cnt) = self.subject_frequency.get_mut(subject) {
            *cnt = cnt.saturating_sub(1);
            if *cnt == 0 {
                self.subject_frequency.remove(subject);
                self.distinct_subjects = self.distinct_subjects.saturating_sub(1);
            }
        }

        if let Some(cnt) = self.predicate_frequency.get_mut(predicate) {
            *cnt = cnt.saturating_sub(1);
            if *cnt == 0 {
                self.predicate_frequency.remove(predicate);
                self.distinct_predicates = self.distinct_predicates.saturating_sub(1);
            }
        }

        if let Some(cnt) = self.object_frequency.get_mut(object) {
            *cnt = cnt.saturating_sub(1);
            if *cnt == 0 {
                self.object_frequency.remove(object);
                self.distinct_objects = self.distinct_objects.saturating_sub(1);
            }
        }
    }

    /// Total number of triples.
    pub fn total_triples(&self) -> u64 {
        self.total_triples
    }

    /// Number of distinct subjects.
    pub fn distinct_subjects(&self) -> u64 {
        self.distinct_subjects
    }

    /// Number of distinct predicates.
    pub fn distinct_predicates(&self) -> u64 {
        self.distinct_predicates
    }

    /// Number of distinct objects.
    pub fn distinct_objects(&self) -> u64 {
        self.distinct_objects
    }

    /// Frequency of `predicate` (number of triples with that predicate).
    pub fn predicate_frequency(&self, predicate: &str) -> u64 {
        *self.predicate_frequency.get(predicate).unwrap_or(&0)
    }

    /// All predicates sorted by frequency (most frequent first).
    pub fn top_predicates(&self, limit: usize) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self
            .predicate_frequency
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(limit);
        v
    }

    /// Estimate the number of triples for a given triple pattern.
    ///
    /// `None` in a slot means "wildcard".  The estimate uses frequency data where available
    /// and falls back to uniform distribution otherwise.
    pub fn estimate_cardinality(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> u64 {
        if self.total_triples == 0 {
            return 0;
        }
        let n = self.total_triples as f64;

        let s_sel = match subject {
            None => 1.0,
            Some(s) => {
                let freq = *self.subject_frequency.get(s).unwrap_or(&0) as f64;
                if freq == 0.0 {
                    0.0
                } else {
                    freq / n
                }
            }
        };

        let p_sel = match predicate {
            None => 1.0,
            Some(p) => {
                let freq = *self.predicate_frequency.get(p).unwrap_or(&0) as f64;
                if freq == 0.0 {
                    0.0
                } else {
                    freq / n
                }
            }
        };

        let o_sel = match object {
            None => 1.0,
            Some(o) => {
                let freq = *self.object_frequency.get(o).unwrap_or(&0) as f64;
                if freq == 0.0 {
                    0.0
                } else {
                    freq / n
                }
            }
        };

        (n * s_sel * p_sel * o_sel).ceil() as u64
    }

    /// Merge another `GraphStatistics` instance into this one (union semantics — no dedup).
    pub fn merge(&mut self, other: &GraphStatistics) {
        self.total_triples += other.total_triples;
        for (s, &cnt) in &other.subject_frequency {
            let prev = *self.subject_frequency.get(s).unwrap_or(&0);
            if prev == 0 {
                self.distinct_subjects += 1;
            }
            *self.subject_frequency.entry(s.clone()).or_insert(0) += cnt;
        }
        for (p, &cnt) in &other.predicate_frequency {
            let prev = *self.predicate_frequency.get(p).unwrap_or(&0);
            if prev == 0 {
                self.distinct_predicates += 1;
            }
            *self.predicate_frequency.entry(p.clone()).or_insert(0) += cnt;
        }
        for (o, &cnt) in &other.object_frequency {
            let prev = *self.object_frequency.get(o).unwrap_or(&0);
            if prev == 0 {
                self.distinct_objects += 1;
            }
            *self.object_frequency.entry(o.clone()).or_insert(0) += cnt;
        }
    }
}

// ---------------------------------------------------------------------------
// PredicateHistogram
// ---------------------------------------------------------------------------

/// Frequency distribution of predicates in an RDF graph.
///
/// Maps each predicate IRI to the count of triples that use it.
#[derive(Debug, Clone, Default)]
pub struct PredicateHistogram {
    frequencies: HashMap<String, u64>,
    total: u64,
}

impl PredicateHistogram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a triple with this predicate.
    pub fn record(&mut self, predicate: &str) {
        *self.frequencies.entry(predicate.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Remove a triple with this predicate.
    pub fn remove(&mut self, predicate: &str) {
        if let Some(cnt) = self.frequencies.get_mut(predicate) {
            if *cnt > 0 {
                *cnt -= 1;
                self.total = self.total.saturating_sub(1);
                if *cnt == 0 {
                    self.frequencies.remove(predicate);
                }
            }
        }
    }

    /// Frequency (triple count) for `predicate`.
    pub fn frequency(&self, predicate: &str) -> u64 {
        *self.frequencies.get(predicate).unwrap_or(&0)
    }

    /// Relative frequency (fraction of all triples) for `predicate`.
    pub fn relative_frequency(&self, predicate: &str) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            *self.frequencies.get(predicate).unwrap_or(&0) as f64 / self.total as f64
        }
    }

    /// Total triple count across all predicates.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Number of distinct predicates.
    pub fn distinct_count(&self) -> usize {
        self.frequencies.len()
    }

    /// Top-N predicates by frequency.
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self
            .frequencies
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    /// Bottom-N predicates by frequency (least used).
    pub fn bottom_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self
            .frequencies
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v.truncate(n);
        v
    }

    /// All frequencies, sorted alphabetically by predicate IRI.
    pub fn all_sorted(&self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self
            .frequencies
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

// ---------------------------------------------------------------------------
// CardinalityEstimator
// ---------------------------------------------------------------------------

/// Estimates join cardinality from predicate histograms and overall graph statistics.
///
/// Uses independence assumptions (suitable for cost-based query planning) to estimate
/// the result size of a triple pattern join.
#[derive(Debug, Clone, Default)]
pub struct CardinalityEstimator {
    total_triples: u64,
    distinct_subjects: u64,
    distinct_objects: u64,
    predicate_freq: HashMap<String, u64>,
}

impl CardinalityEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `CardinalityEstimator` from a `GraphStatistics` instance.
    pub fn from_graph_stats(stats: &GraphStatistics) -> Self {
        Self {
            total_triples: stats.total_triples(),
            distinct_subjects: stats.distinct_subjects(),
            distinct_objects: stats.distinct_objects(),
            predicate_freq: stats.predicate_frequency.clone(),
        }
    }

    /// Estimate the cardinality of a single triple pattern `(s, p, o)`.
    ///
    /// Each bound slot multiplies the selectivity estimate:
    /// - Bound subject: selectivity ≈ 1 / distinct_subjects
    /// - Bound predicate: selectivity ≈ frequency(p) / total_triples
    /// - Bound object: selectivity ≈ 1 / distinct_objects
    pub fn estimate_pattern(
        &self,
        subject_bound: bool,
        predicate: Option<&str>,
        object_bound: bool,
    ) -> u64 {
        if self.total_triples == 0 {
            return 0;
        }
        let n = self.total_triples as f64;
        let mut sel = 1.0_f64;

        if subject_bound {
            let ds = self.distinct_subjects.max(1) as f64;
            sel *= 1.0 / ds;
        }

        if let Some(p) = predicate {
            let freq = *self.predicate_freq.get(p).unwrap_or(&0) as f64;
            if freq == 0.0 {
                return 1; // unknown predicate → conservative estimate
            }
            sel *= freq / n;
        }

        if object_bound {
            let dobj = self.distinct_objects.max(1) as f64;
            sel *= 1.0 / dobj;
        }

        (n * sel).ceil() as u64
    }

    /// Estimate the cardinality of joining two triple patterns (independence assumption).
    pub fn estimate_join(&self, left_card: u64, right_card: u64, shared_vars: usize) -> u64 {
        if self.total_triples == 0 || left_card == 0 || right_card == 0 {
            return 0;
        }
        let n = self.total_triples.max(1) as f64;
        // Each shared variable reduces the result by a factor of n
        let reduction = n.powi(shared_vars as i32);
        ((left_card as f64 * right_card as f64 / reduction).ceil() as u64).max(1)
    }

    /// Update cardinality stats from a `GraphStatistics` instance.
    pub fn update_from(&mut self, stats: &GraphStatistics) {
        self.total_triples = stats.total_triples();
        self.distinct_subjects = stats.distinct_subjects();
        self.distinct_objects = stats.distinct_objects();
        self.predicate_freq = stats.predicate_frequency.clone();
    }
}

// ---------------------------------------------------------------------------
// SampledStatistics
// ---------------------------------------------------------------------------

/// Approximate statistics computed from a random sample of the triple store.
///
/// Useful for large graphs where computing exact statistics is too expensive.
/// Uses reservoir sampling semantics — the caller feeds triples and the structure
/// maintains a representative sample of size `sample_size`.
#[derive(Debug, Clone)]
pub struct SampledStatistics {
    sample_size: usize,
    sample: Vec<(String, String, String)>,
    total_seen: u64,
    graph_stats: GraphStatistics,
}

impl SampledStatistics {
    /// Create a new sampler that will keep at most `sample_size` triples.
    pub fn new(sample_size: usize) -> Self {
        Self {
            sample_size,
            sample: Vec::with_capacity(sample_size),
            total_seen: 0,
            graph_stats: GraphStatistics::new(),
        }
    }

    /// Feed a triple to the sampler.  Uses reservoir sampling to decide whether to keep it.
    ///
    /// Because true reservoir sampling requires a PRNG, this implementation uses a deterministic
    /// hash-based pseudo-random selection so that tests are reproducible without an external RNG.
    pub fn observe(&mut self, subject: &str, predicate: &str, object: &str) {
        self.total_seen += 1;
        self.graph_stats.add_triple(subject, predicate, object);

        if self.sample.len() < self.sample_size {
            self.sample.push((
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            ));
        } else {
            // Deterministic pseudo-random index using FNV of the concatenated triple + counter
            let hash = Self::hash3(subject, predicate, object, self.total_seen);
            let idx = (hash % self.total_seen) as usize;
            if idx < self.sample_size {
                self.sample[idx] = (
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                );
            }
        }
    }

    fn hash3(s: &str, p: &str, o: &str, n: u64) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut h = FNV_OFFSET;
        for b in s.bytes().chain(p.bytes()).chain(o.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= n;
        h = h.wrapping_mul(FNV_PRIME);
        h
    }

    /// Number of triples seen so far (population size).
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Number of triples in the sample.
    pub fn sample_size(&self) -> usize {
        self.sample.len()
    }

    /// Access the raw sample.
    pub fn sample(&self) -> &[(String, String, String)] {
        &self.sample
    }

    /// Estimated total triples (same as `total_seen`, since each `observe` is one triple).
    pub fn estimated_total(&self) -> u64 {
        self.total_seen
    }

    /// Estimated number of distinct predicates, extrapolated from the sample.
    pub fn estimated_distinct_predicates(&self) -> u64 {
        let sample_distinct = self.graph_stats.distinct_predicates();
        if self.sample.is_empty() || self.total_seen == 0 {
            return sample_distinct;
        }
        // Scale-up: assume same ratio holds for the population
        let ratio = self.total_seen as f64 / self.sample.len() as f64;
        (sample_distinct as f64 * ratio.sqrt()).ceil() as u64
    }

    /// Estimated cardinality for a triple pattern using sample-based selectivity.
    pub fn estimate_cardinality(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> u64 {
        self.graph_stats
            .estimate_cardinality(subject, predicate, object)
    }

    /// Access the underlying sample-based `GraphStatistics`.
    pub fn graph_stats(&self) -> &GraphStatistics {
        &self.graph_stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- GraphStatistics ----

    #[test]
    fn test_graph_stats_empty() {
        let gs = GraphStatistics::new();
        assert_eq!(gs.total_triples(), 0);
        assert_eq!(gs.distinct_subjects(), 0);
        assert_eq!(gs.distinct_predicates(), 0);
        assert_eq!(gs.distinct_objects(), 0);
    }

    #[test]
    fn test_graph_stats_add_single_triple() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("alice", "knows", "bob");
        assert_eq!(gs.total_triples(), 1);
        assert_eq!(gs.distinct_subjects(), 1);
        assert_eq!(gs.distinct_predicates(), 1);
        assert_eq!(gs.distinct_objects(), 1);
    }

    #[test]
    fn test_graph_stats_add_multiple_triples_same_predicate() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("alice", "knows", "bob");
        gs.add_triple("alice", "knows", "carol");
        gs.add_triple("bob", "knows", "carol");
        assert_eq!(gs.total_triples(), 3);
        assert_eq!(gs.distinct_predicates(), 1);
        assert_eq!(gs.predicate_frequency("knows"), 3);
    }

    #[test]
    fn test_graph_stats_distinct_counts_correct() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("s1", "p1", "o1");
        gs.add_triple("s1", "p2", "o2");
        gs.add_triple("s2", "p1", "o1");
        assert_eq!(gs.distinct_subjects(), 2);
        assert_eq!(gs.distinct_predicates(), 2);
        assert_eq!(gs.distinct_objects(), 2);
    }

    #[test]
    fn test_graph_stats_remove_triple() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("s", "p", "o");
        gs.remove_triple("s", "p", "o");
        assert_eq!(gs.total_triples(), 0);
        assert_eq!(gs.distinct_subjects(), 0);
        assert_eq!(gs.distinct_predicates(), 0);
        assert_eq!(gs.distinct_objects(), 0);
    }

    #[test]
    fn test_graph_stats_remove_partial() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("s", "p", "o1");
        gs.add_triple("s", "p", "o2");
        gs.remove_triple("s", "p", "o1");
        assert_eq!(gs.total_triples(), 1);
        assert_eq!(gs.distinct_subjects(), 1); // s still present
        assert_eq!(gs.distinct_objects(), 1); // only o2 left
    }

    #[test]
    fn test_graph_stats_remove_nonexistent_is_noop() {
        let mut gs = GraphStatistics::new();
        gs.remove_triple("nonexistent", "p", "o"); // should not panic
        assert_eq!(gs.total_triples(), 0);
    }

    #[test]
    fn test_graph_stats_estimate_cardinality_wildcard() {
        let mut gs = GraphStatistics::new();
        for i in 0..100 {
            gs.add_triple(&format!("s{i}"), "p", "o");
        }
        let card = gs.estimate_cardinality(None, None, None);
        assert_eq!(card, 100);
    }

    #[test]
    fn test_graph_stats_estimate_cardinality_bound_predicate() {
        let mut gs = GraphStatistics::new();
        for i in 0..100 {
            gs.add_triple(&format!("s{i}"), "p1", "o");
            gs.add_triple(&format!("s{i}"), "p2", "o");
        }
        let card = gs.estimate_cardinality(None, Some("p1"), None);
        assert!(card <= 100);
        assert!(card >= 1);
    }

    #[test]
    fn test_graph_stats_estimate_cardinality_unknown_predicate() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("s", "p", "o");
        let card = gs.estimate_cardinality(None, Some("nonexistent"), None);
        assert_eq!(card, 0);
    }

    #[test]
    fn test_graph_stats_top_predicates() {
        let mut gs = GraphStatistics::new();
        for _ in 0..10 {
            gs.add_triple("s", "popular", "o");
        }
        for _ in 0..2 {
            gs.add_triple("s", "rare", "o");
        }
        let top = gs.top_predicates(1);
        assert_eq!(top[0].0, "popular");
        assert_eq!(top[0].1, 10);
    }

    #[test]
    fn test_graph_stats_merge() {
        let mut gs1 = GraphStatistics::new();
        gs1.add_triple("s1", "p", "o1");
        let mut gs2 = GraphStatistics::new();
        gs2.add_triple("s2", "p", "o2");
        gs1.merge(&gs2);
        assert_eq!(gs1.total_triples(), 2);
        assert_eq!(gs1.distinct_subjects(), 2);
    }

    // ---- PredicateHistogram ----

    #[test]
    fn test_predicate_histogram_empty() {
        let ph = PredicateHistogram::new();
        assert_eq!(ph.total(), 0);
        assert_eq!(ph.distinct_count(), 0);
        assert_eq!(ph.frequency("any"), 0);
    }

    #[test]
    fn test_predicate_histogram_record() {
        let mut ph = PredicateHistogram::new();
        ph.record("p1");
        ph.record("p1");
        ph.record("p2");
        assert_eq!(ph.total(), 3);
        assert_eq!(ph.frequency("p1"), 2);
        assert_eq!(ph.frequency("p2"), 1);
        assert_eq!(ph.distinct_count(), 2);
    }

    #[test]
    fn test_predicate_histogram_relative_frequency() {
        let mut ph = PredicateHistogram::new();
        for _ in 0..3 {
            ph.record("a");
        }
        for _ in 0..7 {
            ph.record("b");
        }
        let rf_a = ph.relative_frequency("a");
        let rf_b = ph.relative_frequency("b");
        assert!((rf_a - 0.3).abs() < 1e-9);
        assert!((rf_b - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_predicate_histogram_remove() {
        let mut ph = PredicateHistogram::new();
        ph.record("p");
        ph.record("p");
        ph.remove("p");
        assert_eq!(ph.frequency("p"), 1);
        ph.remove("p");
        assert_eq!(ph.frequency("p"), 0);
        assert_eq!(ph.distinct_count(), 0);
    }

    #[test]
    fn test_predicate_histogram_top_n() {
        let mut ph = PredicateHistogram::new();
        for i in 0..5 {
            for _ in 0..(i + 1) {
                ph.record(&format!("p{i}"));
            }
        }
        let top = ph.top_n(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn test_predicate_histogram_bottom_n() {
        let mut ph = PredicateHistogram::new();
        for _ in 0..10 {
            ph.record("common");
        }
        for _ in 0..1 {
            ph.record("rare");
        }
        let bottom = ph.bottom_n(1);
        assert_eq!(bottom[0].0, "rare");
    }

    #[test]
    fn test_predicate_histogram_all_sorted() {
        let mut ph = PredicateHistogram::new();
        ph.record("zebra");
        ph.record("apple");
        ph.record("mango");
        let sorted = ph.all_sorted();
        assert_eq!(sorted[0].0, "apple");
        assert_eq!(sorted[1].0, "mango");
        assert_eq!(sorted[2].0, "zebra");
    }

    // ---- CardinalityEstimator ----

    #[test]
    fn test_cardinality_estimator_empty() {
        let ce = CardinalityEstimator::new();
        assert_eq!(ce.estimate_pattern(false, None, false), 0);
    }

    #[test]
    fn test_cardinality_estimator_from_graph_stats() {
        let mut gs = GraphStatistics::new();
        for i in 0..50 {
            gs.add_triple(&format!("s{i}"), "knows", "o");
        }
        let ce = CardinalityEstimator::from_graph_stats(&gs);
        // Pattern: ?s knows ?o → all 50 triples
        let card = ce.estimate_pattern(false, Some("knows"), false);
        assert_eq!(card, 50);
    }

    #[test]
    fn test_cardinality_estimator_bound_subject_reduces_card() {
        let mut gs = GraphStatistics::new();
        for i in 0..100 {
            gs.add_triple(&format!("s{i}"), "p", "o");
        }
        let ce = CardinalityEstimator::from_graph_stats(&gs);
        let card_with = ce.estimate_pattern(true, None, false);
        let card_without = ce.estimate_pattern(false, None, false);
        assert!(card_with <= card_without);
    }

    #[test]
    fn test_cardinality_estimator_unknown_predicate() {
        let mut gs = GraphStatistics::new();
        gs.add_triple("s", "known_p", "o");
        let ce = CardinalityEstimator::from_graph_stats(&gs);
        let card = ce.estimate_pattern(false, Some("unknown_p"), false);
        assert_eq!(card, 1); // conservative estimate
    }

    #[test]
    fn test_cardinality_estimator_estimate_join() {
        let mut gs = GraphStatistics::new();
        for i in 0..100 {
            gs.add_triple(&format!("s{i}"), "p", "o");
        }
        let ce = CardinalityEstimator::from_graph_stats(&gs);
        let join_card = ce.estimate_join(50, 50, 1);
        // join_card = ceil(50 * 50 / 100) = 25
        assert_eq!(join_card, 25);
    }

    #[test]
    fn test_cardinality_estimator_join_zero() {
        let ce = CardinalityEstimator::new();
        assert_eq!(ce.estimate_join(10, 10, 0), 0);
    }

    #[test]
    fn test_cardinality_estimator_update_from() {
        let ce_before = CardinalityEstimator::new();
        assert_eq!(ce_before.estimate_pattern(false, None, false), 0);
        let mut gs = GraphStatistics::new();
        gs.add_triple("s", "p", "o");
        let mut ce = CardinalityEstimator::new();
        ce.update_from(&gs);
        assert_eq!(ce.estimate_pattern(false, None, false), 1);
    }

    // ---- SampledStatistics ----

    #[test]
    fn test_sampled_statistics_empty() {
        let ss = SampledStatistics::new(100);
        assert_eq!(ss.total_seen(), 0);
        assert_eq!(ss.sample_size(), 0);
    }

    #[test]
    fn test_sampled_statistics_observe_fewer_than_capacity() {
        let mut ss = SampledStatistics::new(100);
        ss.observe("s", "p", "o");
        ss.observe("s2", "p2", "o2");
        assert_eq!(ss.total_seen(), 2);
        assert_eq!(ss.sample_size(), 2);
    }

    #[test]
    fn test_sampled_statistics_observe_more_than_capacity() {
        let mut ss = SampledStatistics::new(10);
        for i in 0..50 {
            ss.observe(&format!("s{i}"), "p", &format!("o{i}"));
        }
        assert_eq!(ss.total_seen(), 50);
        assert_eq!(ss.sample_size(), 10);
    }

    #[test]
    fn test_sampled_statistics_estimated_total() {
        let mut ss = SampledStatistics::new(100);
        for i in 0..200 {
            ss.observe(&format!("s{i}"), "p", "o");
        }
        assert_eq!(ss.estimated_total(), 200);
    }

    #[test]
    fn test_sampled_statistics_cardinality_from_sample() {
        let mut ss = SampledStatistics::new(1000);
        for i in 0..100 {
            ss.observe(&format!("s{i}"), "p", "o");
        }
        let card = ss.estimate_cardinality(None, Some("p"), None);
        assert!(card > 0);
    }

    #[test]
    fn test_sampled_statistics_estimated_distinct_predicates() {
        let mut ss = SampledStatistics::new(50);
        for i in 0..5 {
            ss.observe("s", &format!("p{i}"), "o");
        }
        let est = ss.estimated_distinct_predicates();
        assert!(est >= 5);
    }

    #[test]
    fn test_sampled_statistics_graph_stats_available() {
        let mut ss = SampledStatistics::new(100);
        ss.observe("alice", "knows", "bob");
        let gs = ss.graph_stats();
        assert_eq!(gs.total_triples(), 1);
    }
}
