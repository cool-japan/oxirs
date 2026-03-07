//! Triple store cardinality estimator for query optimization.
//!
//! Maintains frequency maps for subjects, predicates, and objects to provide
//! fast selectivity estimates without full table scans. Used by the query
//! optimizer to choose efficient join orderings and access patterns.

use std::collections::HashMap;

/// A term and its frequency count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermFrequency {
    /// The string representation of the RDF term.
    pub term: String,
    /// Number of times the term appears in the store.
    pub count: usize,
}

impl TermFrequency {
    fn new(term: impl Into<String>, count: usize) -> Self {
        TermFrequency {
            term: term.into(),
            count,
        }
    }
}

/// Raw frequency statistics for a triple store.
#[derive(Debug, Default, Clone)]
pub struct TripleStoreStats {
    /// Total number of triples recorded (including duplicates).
    pub total_triples: u64,
    /// Number of distinct subject terms.
    pub distinct_subjects: usize,
    /// Number of distinct predicate terms.
    pub distinct_predicates: usize,
    /// Number of distinct object terms.
    pub distinct_objects: usize,
    subject_freq: HashMap<String, u64>,
    predicate_freq: HashMap<String, u64>,
    object_freq: HashMap<String, u64>,
    /// count of distinct objects per predicate (co-occurrence)
    pred_obj_pairs: HashMap<String, usize>,
}

/// Cardinality estimator for a triple store.
///
/// Records triples as they are inserted (or bulk-loaded) and provides
/// selectivity estimates that can be used by a query planner.
#[derive(Debug, Default)]
pub struct IndexStatistics {
    stats: TripleStoreStats,
}

impl IndexStatistics {
    /// Create a new, empty statistics collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single triple.
    pub fn record_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.stats.total_triples += 1;

        // Subject frequency
        let s_entry = self
            .stats
            .subject_freq
            .entry(subject.to_string())
            .or_insert(0);
        *s_entry += 1;

        // Predicate frequency
        let p_entry = self
            .stats
            .predicate_freq
            .entry(predicate.to_string())
            .or_insert(0);
        *p_entry += 1;

        // Object frequency
        let o_entry = self
            .stats
            .object_freq
            .entry(object.to_string())
            .or_insert(0);
        *o_entry += 1;

        // Predicate→object co-occurrence (simplified: count distinct objects per pred)
        // We track it as a nested count using a suffixed key "pred\0obj"
        let pred_obj_key = format!("{}\x00{}", predicate, object);
        self.stats.pred_obj_pairs.entry(pred_obj_key).or_insert(0);
        // We just need to know how many distinct objects exist per predicate
        // Use a separate map: pred → HashSet<object>. Since we can't use
        // HashSet in a simple HashMap<String,usize>, we track it differently.
        // Re-compute distinct objects per predicate lazily from pred_obj_pairs.

        // Update derived distinct counts
        self.stats.distinct_subjects = self.stats.subject_freq.len();
        self.stats.distinct_predicates = self.stats.predicate_freq.len();
        self.stats.distinct_objects = self.stats.object_freq.len();

        // Update pred_obj_pairs count = number of distinct objects for this predicate
        let pred_key = predicate.to_string();
        let distinct_objs_for_pred: usize = self
            .stats
            .pred_obj_pairs
            .keys()
            .filter(|k| k.starts_with(&format!("{}\x00", pred_key)))
            .count();
        self.stats
            .pred_obj_pairs
            .insert(pred_key, distinct_objs_for_pred);
    }

    /// Record many triples at once.
    pub fn record_triples(&mut self, triples: &[(&str, &str, &str)]) {
        for (s, p, o) in triples {
            self.record_triple(s, p, o);
        }
    }

    /// Total number of triples recorded.
    pub fn total_triples(&self) -> u64 {
        self.stats.total_triples
    }

    /// Number of distinct subjects.
    pub fn distinct_subjects(&self) -> usize {
        self.stats.distinct_subjects
    }

    /// Number of distinct predicates.
    pub fn distinct_predicates(&self) -> usize {
        self.stats.distinct_predicates
    }

    /// Number of distinct objects.
    pub fn distinct_objects(&self) -> usize {
        self.stats.distinct_objects
    }

    /// Fraction of triples that have the given subject (0.0 if unknown or no triples).
    pub fn selectivity_subject(&self, subject: &str) -> f64 {
        if self.stats.total_triples == 0 {
            return 0.0;
        }
        match self.stats.subject_freq.get(subject) {
            Some(&count) => count as f64 / self.stats.total_triples as f64,
            None => 0.0,
        }
    }

    /// Fraction of triples that have the given predicate (0.0 if unknown or no triples).
    pub fn selectivity_predicate(&self, predicate: &str) -> f64 {
        if self.stats.total_triples == 0 {
            return 0.0;
        }
        match self.stats.predicate_freq.get(predicate) {
            Some(&count) => count as f64 / self.stats.total_triples as f64,
            None => 0.0,
        }
    }

    /// Fraction of triples that have the given object (0.0 if unknown or no triples).
    pub fn selectivity_object(&self, object: &str) -> f64 {
        if self.stats.total_triples == 0 {
            return 0.0;
        }
        match self.stats.object_freq.get(object) {
            Some(&count) => count as f64 / self.stats.total_triples as f64,
            None => 0.0,
        }
    }

    /// Estimated number of triples that match the given predicate.
    pub fn estimated_triple_count(&self, pred: &str) -> u64 {
        match self.stats.predicate_freq.get(pred) {
            Some(&count) => count,
            None => 0,
        }
    }

    /// Join size estimate: |σ_p1(T) ⋈ σ_p2(T)| ≈ total × sel(p1) × sel(p2).
    ///
    /// This is the standard histogram-based join size estimator.
    pub fn estimated_join_size(&self, pred_a: &str, pred_b: &str) -> u64 {
        let n = self.stats.total_triples as f64;
        let sel_a = self.selectivity_predicate(pred_a);
        let sel_b = self.selectivity_predicate(pred_b);
        (n * sel_a * sel_b).ceil() as u64
    }

    /// Top-`k` most frequent predicates (descending by count).
    pub fn most_frequent_predicates(&self, k: usize) -> Vec<TermFrequency> {
        Self::top_k(&self.stats.predicate_freq, k)
    }

    /// Top-`k` most frequent subjects (descending by count).
    pub fn most_frequent_subjects(&self, k: usize) -> Vec<TermFrequency> {
        Self::top_k(&self.stats.subject_freq, k)
    }

    /// Top-`k` most frequent objects (descending by count).
    pub fn most_frequent_objects(&self, k: usize) -> Vec<TermFrequency> {
        Self::top_k(&self.stats.object_freq, k)
    }

    /// Reset all recorded statistics.
    pub fn clear(&mut self) {
        self.stats = TripleStoreStats::default();
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn top_k(freq: &HashMap<String, u64>, k: usize) -> Vec<TermFrequency> {
        let mut pairs: Vec<(&String, &u64)> = freq.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        pairs
            .into_iter()
            .take(k)
            .map(|(term, &count)| TermFrequency::new(term.clone(), count as usize))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Empty stats ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_total_triples() {
        let stats = IndexStatistics::new();
        assert_eq!(stats.total_triples(), 0);
    }

    #[test]
    fn test_empty_distinct_counts() {
        let stats = IndexStatistics::new();
        assert_eq!(stats.distinct_subjects(), 0);
        assert_eq!(stats.distinct_predicates(), 0);
        assert_eq!(stats.distinct_objects(), 0);
    }

    #[test]
    fn test_empty_selectivity_zero() {
        let stats = IndexStatistics::new();
        assert_eq!(stats.selectivity_subject("anything"), 0.0);
        assert_eq!(stats.selectivity_predicate("anything"), 0.0);
        assert_eq!(stats.selectivity_object("anything"), 0.0);
    }

    #[test]
    fn test_empty_estimated_count_zero() {
        let stats = IndexStatistics::new();
        assert_eq!(stats.estimated_triple_count("pred"), 0);
    }

    #[test]
    fn test_empty_join_size_zero() {
        let stats = IndexStatistics::new();
        assert_eq!(stats.estimated_join_size("p1", "p2"), 0);
    }

    #[test]
    fn test_empty_most_frequent_empty() {
        let stats = IndexStatistics::new();
        assert!(stats.most_frequent_predicates(5).is_empty());
        assert!(stats.most_frequent_subjects(5).is_empty());
        assert!(stats.most_frequent_objects(5).is_empty());
    }

    // ── Single triple ─────────────────────────────────────────────────────────

    #[test]
    fn test_single_triple_total() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert_eq!(stats.total_triples(), 1);
    }

    #[test]
    fn test_single_triple_distinct_counts() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert_eq!(stats.distinct_subjects(), 1);
        assert_eq!(stats.distinct_predicates(), 1);
        assert_eq!(stats.distinct_objects(), 1);
    }

    #[test]
    fn test_single_triple_selectivity_one() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert!((stats.selectivity_subject("s") - 1.0).abs() < 1e-9);
        assert!((stats.selectivity_predicate("p") - 1.0).abs() < 1e-9);
        assert!((stats.selectivity_object("o") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_triple_unknown_selectivity_zero() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert_eq!(stats.selectivity_subject("UNKNOWN"), 0.0);
        assert_eq!(stats.selectivity_predicate("UNKNOWN"), 0.0);
        assert_eq!(stats.selectivity_object("UNKNOWN"), 0.0);
    }

    // ── Multiple triples ──────────────────────────────────────────────────────

    #[test]
    fn test_many_triples_total() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "type", "A"),
            ("s2", "type", "B"),
            ("s3", "type", "A"),
            ("s1", "name", "Alice"),
        ]);
        assert_eq!(stats.total_triples(), 4);
    }

    #[test]
    fn test_many_triples_distinct_subjects() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s1", "p", "o"), ("s1", "p2", "o"), ("s2", "p", "o")]);
        assert_eq!(stats.distinct_subjects(), 2);
    }

    #[test]
    fn test_many_triples_distinct_predicates() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "type", "A"),
            ("s2", "type", "B"),
            ("s3", "name", "foo"),
        ]);
        assert_eq!(stats.distinct_predicates(), 2);
    }

    #[test]
    fn test_many_triples_distinct_objects() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s", "p", "o1"),
            ("s", "p", "o1"), // duplicate object value
            ("s", "p", "o2"),
        ]);
        assert_eq!(stats.distinct_objects(), 2);
    }

    // ── Selectivity ───────────────────────────────────────────────────────────

    #[test]
    fn test_selectivity_predicate_half() {
        let mut stats = IndexStatistics::new();
        // 2 triples: 1 with "type", 1 with "name"
        stats.record_triples(&[("s1", "type", "A"), ("s2", "name", "B")]);
        let sel = stats.selectivity_predicate("type");
        assert!((sel - 0.5).abs() < 1e-9, "Expected 0.5, got {}", sel);
    }

    #[test]
    fn test_selectivity_subject_fraction() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "p", "o"),
            ("s1", "p", "o2"),
            ("s2", "p", "o3"),
            ("s3", "p", "o4"),
        ]);
        // s1 appears 2 out of 4 times
        let sel = stats.selectivity_subject("s1");
        assert!((sel - 0.5).abs() < 1e-9, "Expected 0.5, got {}", sel);
    }

    #[test]
    fn test_selectivity_object_fraction() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s1", "p", "obj"), ("s2", "p", "obj"), ("s3", "p", "other")]);
        let sel = stats.selectivity_object("obj");
        let expected = 2.0 / 3.0;
        assert!(
            (sel - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            sel
        );
    }

    // ── Estimated counts ──────────────────────────────────────────────────────

    #[test]
    fn test_estimated_triple_count_known_predicate() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "type", "A"),
            ("s2", "type", "B"),
            ("s3", "name", "foo"),
        ]);
        assert_eq!(stats.estimated_triple_count("type"), 2);
        assert_eq!(stats.estimated_triple_count("name"), 1);
    }

    #[test]
    fn test_estimated_triple_count_unknown_predicate() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert_eq!(stats.estimated_triple_count("MISSING"), 0);
    }

    // ── Join size estimate ────────────────────────────────────────────────────

    #[test]
    fn test_estimated_join_size_basic() {
        let mut stats = IndexStatistics::new();
        // 4 triples: 2 with "type", 2 with "name"
        stats.record_triples(&[
            ("s1", "type", "A"),
            ("s2", "type", "B"),
            ("s3", "name", "foo"),
            ("s4", "name", "bar"),
        ]);
        // sel(type) = 2/4 = 0.5, sel(name) = 2/4 = 0.5
        // estimate = 4 * 0.5 * 0.5 = 1.0 → ceil = 1
        let join = stats.estimated_join_size("type", "name");
        assert_eq!(join, 1);
    }

    #[test]
    fn test_estimated_join_size_unknown_pred_zero() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "type", "A");
        let join = stats.estimated_join_size("type", "MISSING");
        assert_eq!(join, 0);
    }

    #[test]
    fn test_estimated_join_size_same_predicate() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s1", "p", "o1"), ("s2", "p", "o2")]);
        // sel(p) = 1.0, join = total * 1 * 1 = 2
        let join = stats.estimated_join_size("p", "p");
        assert_eq!(join, 2);
    }

    // ── most_frequent ─────────────────────────────────────────────────────────

    #[test]
    fn test_most_frequent_predicates_ordering() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "type", "A"),
            ("s2", "type", "B"),
            ("s3", "type", "C"),
            ("s4", "name", "foo"),
            ("s5", "name", "bar"),
            ("s6", "age", "30"),
        ]);
        let top = stats.most_frequent_predicates(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].term, "type");
        assert_eq!(top[0].count, 3);
        assert_eq!(top[1].count, 2); // "name"
        assert_eq!(top[2].count, 1); // "age"
    }

    #[test]
    fn test_most_frequent_predicates_k_larger_than_actual() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s", "p1", "o"), ("s", "p2", "o")]);
        let top = stats.most_frequent_predicates(100);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_most_frequent_subjects_ordering() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("alice", "knows", "bob"),
            ("alice", "knows", "carol"),
            ("bob", "knows", "alice"),
        ]);
        let top = stats.most_frequent_subjects(2);
        assert_eq!(top[0].term, "alice");
        assert_eq!(top[0].count, 2);
        assert_eq!(top[1].term, "bob");
        assert_eq!(top[1].count, 1);
    }

    #[test]
    fn test_most_frequent_objects_ordering() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[
            ("s1", "p", "common"),
            ("s2", "p", "common"),
            ("s3", "p", "common"),
            ("s4", "p", "rare"),
        ]);
        let top = stats.most_frequent_objects(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].term, "common");
        assert_eq!(top[0].count, 3);
    }

    #[test]
    fn test_most_frequent_k_zero() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        assert!(stats.most_frequent_predicates(0).is_empty());
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_all() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s1", "p1", "o1"), ("s2", "p2", "o2")]);
        assert_eq!(stats.total_triples(), 2);

        stats.clear();

        assert_eq!(stats.total_triples(), 0);
        assert_eq!(stats.distinct_subjects(), 0);
        assert_eq!(stats.distinct_predicates(), 0);
        assert_eq!(stats.distinct_objects(), 0);
        assert_eq!(stats.selectivity_predicate("p1"), 0.0);
    }

    #[test]
    fn test_clear_then_re_record() {
        let mut stats = IndexStatistics::new();
        stats.record_triple("s", "p", "o");
        stats.clear();
        stats.record_triple("new_s", "new_p", "new_o");
        assert_eq!(stats.total_triples(), 1);
        assert_eq!(stats.distinct_predicates(), 1);
        assert!((stats.selectivity_predicate("new_p") - 1.0).abs() < 1e-9);
    }

    // ── Record many ───────────────────────────────────────────────────────────

    #[test]
    fn test_record_100_triples() {
        let mut stats = IndexStatistics::new();
        let triples: Vec<(String, String, String)> = (0..100)
            .map(|i| (format!("s{}", i), "p".to_string(), format!("o{}", i)))
            .collect();
        let refs: Vec<(&str, &str, &str)> = triples
            .iter()
            .map(|(s, p, o)| (s.as_str(), p.as_str(), o.as_str()))
            .collect();
        stats.record_triples(&refs);
        assert_eq!(stats.total_triples(), 100);
        assert_eq!(stats.distinct_subjects(), 100);
        assert_eq!(stats.distinct_predicates(), 1);
        assert_eq!(stats.distinct_objects(), 100);
    }

    #[test]
    fn test_selectivity_sums_to_one_for_single_predicate() {
        let mut stats = IndexStatistics::new();
        stats.record_triples(&[("s1", "p", "o"), ("s2", "p", "o")]);
        // Both triples share the same predicate → selectivity = 1.0
        assert!((stats.selectivity_predicate("p") - 1.0).abs() < 1e-9);
    }
}
