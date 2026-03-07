//! RDF-star graph statistics collector.
//!
//! Provides `StarStatisticsCollector` to accumulate `TriplePattern` values and
//! compute a rich `GraphStats` summary including quoted-triple counts, predicate
//! frequencies, subject out-degrees, and `<<…>>` nesting depth analysis.

use std::collections::{HashMap, HashSet};

// ── TriplePattern ─────────────────────────────────────────────────────────────

/// A single RDF(-star) triple, where subject or object may be a quoted-triple
/// string in the format `<<s p o>>` (potentially nested).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    /// The subject (may be a URI or a quoted-triple string).
    pub subject: String,
    /// The predicate (a URI string).
    pub predicate: String,
    /// The object (may be a URI, literal, or a quoted-triple string).
    pub object: String,
}

impl TriplePattern {
    /// Construct a new triple pattern.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        TriplePattern {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Return `true` if the subject is a quoted triple (starts with `<<`).
    pub fn subject_is_quoted(&self) -> bool {
        self.subject.trim_start().starts_with("<<")
    }

    /// Return `true` if the object is a quoted triple (starts with `<<`).
    pub fn object_is_quoted(&self) -> bool {
        self.object.trim_start().starts_with("<<")
    }

    /// Return `true` if this triple uses quoted-triple syntax anywhere.
    pub fn has_quoted_triple(&self) -> bool {
        self.subject_is_quoted() || self.object_is_quoted()
    }
}

// ── GraphStats ────────────────────────────────────────────────────────────────

/// Aggregated statistics for an RDF-star graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of triples in the graph.
    pub triple_count: usize,
    /// Number of triples that use quoted-triple syntax in subject or object.
    pub quoted_triple_count: usize,
    /// Number of distinct subject values.
    pub unique_subjects: usize,
    /// Number of distinct predicate values.
    pub unique_predicates: usize,
    /// Number of distinct object values.
    pub unique_objects: usize,
    /// Maximum `<<` nesting depth across all term strings.
    pub max_nesting_depth: usize,
    /// Count of triples per predicate URI.
    pub predicate_frequency: HashMap<String, usize>,
    /// Number of triples each subject appears in (out-degree).
    pub subject_out_degree: HashMap<String, usize>,
}

// ── Collector ─────────────────────────────────────────────────────────────────

/// Accumulates `TriplePattern` values and computes `GraphStats` on demand.
pub struct StarStatisticsCollector {
    triples: Vec<TriplePattern>,
}

impl StarStatisticsCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        StarStatisticsCollector {
            triples: Vec::new(),
        }
    }

    /// Add a single triple.
    pub fn add_triple(&mut self, t: TriplePattern) {
        self.triples.push(t);
    }

    /// Add multiple triples.
    pub fn add_triples(&mut self, ts: Vec<TriplePattern>) {
        self.triples.extend(ts);
    }

    /// Compute the `GraphStats` from the accumulated triples.
    pub fn compute(&self) -> GraphStats {
        let triple_count = self.triples.len();

        let mut quoted_triple_count = 0usize;
        let mut unique_subjects: HashSet<&str> = HashSet::new();
        let mut unique_predicates: HashSet<&str> = HashSet::new();
        let mut unique_objects: HashSet<&str> = HashSet::new();
        let mut max_nesting_depth = 0usize;
        let mut predicate_frequency: HashMap<String, usize> = HashMap::new();
        let mut subject_out_degree: HashMap<String, usize> = HashMap::new();

        for t in &self.triples {
            // Quoted triple detection
            if t.has_quoted_triple() {
                quoted_triple_count += 1;
            }

            // Unique term tracking
            unique_subjects.insert(t.subject.as_str());
            unique_predicates.insert(t.predicate.as_str());
            unique_objects.insert(t.object.as_str());

            // Predicate frequency
            *predicate_frequency.entry(t.predicate.clone()).or_insert(0) += 1;

            // Subject out-degree
            *subject_out_degree.entry(t.subject.clone()).or_insert(0) += 1;

            // Nesting depth for all three terms
            for term in &[&t.subject, &t.predicate, &t.object] {
                let depth = nesting_depth_of(term);
                if depth > max_nesting_depth {
                    max_nesting_depth = depth;
                }
            }
        }

        GraphStats {
            triple_count,
            quoted_triple_count,
            unique_subjects: unique_subjects.len(),
            unique_predicates: unique_predicates.len(),
            unique_objects: unique_objects.len(),
            max_nesting_depth,
            predicate_frequency,
            subject_out_degree,
        }
    }

    /// Return the top-`n` predicates by descending frequency.
    ///
    /// When two predicates have the same frequency they are ordered
    /// lexicographically by predicate URI for determinism.
    pub fn top_predicates(&self, n: usize) -> Vec<(String, usize)> {
        let stats = self.compute();
        let mut pairs: Vec<(String, usize)> = stats.predicate_frequency.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    /// Return subjects whose out-degree (number of triples they appear in as
    /// subject) is at least `threshold`.
    pub fn high_degree_subjects(&self, threshold: usize) -> Vec<String> {
        let stats = self.compute();
        let mut subjects: Vec<String> = stats
            .subject_out_degree
            .into_iter()
            .filter(|(_, deg)| *deg >= threshold)
            .map(|(s, _)| s)
            .collect();
        subjects.sort();
        subjects
    }

    /// Count the `<<` nesting levels in `triple_str`.
    ///
    /// Each `<<` in the string counts as one level (the maximum depth is the
    /// total count of `<<` tokens, reflecting worst-case nesting).
    pub fn nesting_depth(&self, triple_str: &str) -> usize {
        nesting_depth_of(triple_str)
    }
}

impl Default for StarStatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Count the `<<` nesting depth in a term string.
///
/// The depth is defined as the maximum concurrent open-angle-bracket pairs
/// observed while scanning left to right. Each `<<` increments the depth;
/// each `>>` decrements it.
fn nesting_depth_of(s: &str) -> usize {
    let mut depth: usize = 0;
    let mut max_depth: usize = 0;
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if i + 1 < len && bytes[i] == b'<' && bytes[i + 1] == b'<' {
            depth += 1;
            if depth > max_depth {
                max_depth = depth;
            }
            i += 2;
        } else if i + 1 < len && bytes[i] == b'>' && bytes[i + 1] == b'>' {
            depth = depth.saturating_sub(1);
            i += 2;
        } else {
            i += 1;
        }
    }
    max_depth
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tp(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern::new(s, p, o)
    }

    // ── Basic stats ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph_all_zeros() {
        let c = StarStatisticsCollector::new();
        let s = c.compute();
        assert_eq!(s.triple_count, 0);
        assert_eq!(s.quoted_triple_count, 0);
        assert_eq!(s.unique_subjects, 0);
        assert_eq!(s.unique_predicates, 0);
        assert_eq!(s.unique_objects, 0);
        assert_eq!(s.max_nesting_depth, 0);
        assert!(s.predicate_frequency.is_empty());
        assert!(s.subject_out_degree.is_empty());
    }

    #[test]
    fn test_single_triple_count() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":alice", ":knows", ":bob"));
        let s = c.compute();
        assert_eq!(s.triple_count, 1);
    }

    #[test]
    fn test_three_triples_count() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":a", ":p", ":b"),
            tp(":b", ":p", ":c"),
            tp(":c", ":q", ":d"),
        ]);
        let s = c.compute();
        assert_eq!(s.triple_count, 3);
    }

    #[test]
    fn test_unique_subjects() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":alice", ":knows", ":bob"),
            tp(":alice", ":age", "\"30\""),
            tp(":bob", ":knows", ":carol"),
        ]);
        let s = c.compute();
        assert_eq!(s.unique_subjects, 2); // alice, bob
    }

    #[test]
    fn test_unique_predicates() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":a", ":knows", ":b"),
            tp(":a", ":knows", ":c"),
            tp(":a", ":age", "\"25\""),
        ]);
        let s = c.compute();
        assert_eq!(s.unique_predicates, 2);
    }

    #[test]
    fn test_unique_objects() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":a", ":p", ":x"),
            tp(":b", ":p", ":x"),
            tp(":c", ":p", ":y"),
        ]);
        let s = c.compute();
        assert_eq!(s.unique_objects, 2);
    }

    // ── Quoted triple detection ───────────────────────────────────────────────

    #[test]
    fn test_quoted_triple_in_subject() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp("<< :alice :age 30 >>", ":certainty", "\"0.9\""));
        let s = c.compute();
        assert_eq!(s.quoted_triple_count, 1);
    }

    #[test]
    fn test_quoted_triple_in_object() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":source", ":states", "<< :alice :age 30 >>"));
        let s = c.compute();
        assert_eq!(s.quoted_triple_count, 1);
    }

    #[test]
    fn test_no_quoted_triple() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":alice", ":age", "\"30\""));
        let s = c.compute();
        assert_eq!(s.quoted_triple_count, 0);
    }

    #[test]
    fn test_mixed_quoted_and_plain() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![tp(":a", ":p", ":b"), tp("<< :a :p :b >>", ":q", ":c")]);
        let s = c.compute();
        assert_eq!(s.quoted_triple_count, 1);
    }

    // ── Nesting depth ─────────────────────────────────────────────────────────

    #[test]
    fn test_nesting_depth_zero_plain() {
        let c = StarStatisticsCollector::new();
        assert_eq!(c.nesting_depth(":alice"), 0);
    }

    #[test]
    fn test_nesting_depth_one() {
        let c = StarStatisticsCollector::new();
        assert_eq!(c.nesting_depth("<< :a :p :b >>"), 1);
    }

    #[test]
    fn test_nesting_depth_two() {
        let c = StarStatisticsCollector::new();
        assert_eq!(c.nesting_depth("<< << :a :p :b >> :q :c >>"), 2);
    }

    #[test]
    fn test_nesting_depth_three() {
        let c = StarStatisticsCollector::new();
        assert_eq!(c.nesting_depth("<< << << :a :p :b >> :q :c >> :r :d >>"), 3);
    }

    #[test]
    fn test_max_nesting_depth_in_stats() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp("<< :a :p :b >>", ":certainty", "\"1.0\""),
            tp("<< << :a :p :b >> :q :c >>", ":source", ":s"),
        ]);
        let s = c.compute();
        assert_eq!(s.max_nesting_depth, 2);
    }

    #[test]
    fn test_nesting_depth_empty_string() {
        let c = StarStatisticsCollector::new();
        assert_eq!(c.nesting_depth(""), 0);
    }

    // ── Predicate frequency ───────────────────────────────────────────────────

    #[test]
    fn test_predicate_frequency_single() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":a", ":knows", ":b"));
        let s = c.compute();
        assert_eq!(s.predicate_frequency.get(":knows"), Some(&1));
    }

    #[test]
    fn test_predicate_frequency_multiple() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":a", ":knows", ":b"),
            tp(":b", ":knows", ":c"),
            tp(":a", ":age", "\"25\""),
        ]);
        let s = c.compute();
        assert_eq!(s.predicate_frequency.get(":knows"), Some(&2));
        assert_eq!(s.predicate_frequency.get(":age"), Some(&1));
    }

    // ── Top predicates ────────────────────────────────────────────────────────

    #[test]
    fn test_top_predicates_returns_top_n() {
        let mut c = StarStatisticsCollector::new();
        for _ in 0..5 {
            c.add_triple(tp(":a", ":p1", ":b"));
        }
        for _ in 0..3 {
            c.add_triple(tp(":a", ":p2", ":b"));
        }
        c.add_triple(tp(":a", ":p3", ":b"));
        let top = c.top_predicates(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, ":p1");
        assert_eq!(top[0].1, 5);
        assert_eq!(top[1].0, ":p2");
    }

    #[test]
    fn test_top_predicates_empty() {
        let c = StarStatisticsCollector::new();
        assert!(c.top_predicates(5).is_empty());
    }

    #[test]
    fn test_top_predicates_n_exceeds_count() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":a", ":p", ":b"));
        let top = c.top_predicates(100);
        assert_eq!(top.len(), 1);
    }

    // ── Subject out-degree ────────────────────────────────────────────────────

    #[test]
    fn test_subject_out_degree_basic() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":alice", ":knows", ":bob"),
            tp(":alice", ":age", "\"30\""),
            tp(":bob", ":knows", ":carol"),
        ]);
        let s = c.compute();
        assert_eq!(s.subject_out_degree.get(":alice"), Some(&2));
        assert_eq!(s.subject_out_degree.get(":bob"), Some(&1));
    }

    // ── High-degree subjects ──────────────────────────────────────────────────

    #[test]
    fn test_high_degree_subjects_threshold_2() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp(":alice", ":knows", ":bob"),
            tp(":alice", ":age", "\"30\""),
            tp(":bob", ":knows", ":carol"),
        ]);
        let hs = c.high_degree_subjects(2);
        assert_eq!(hs, vec![":alice".to_string()]);
    }

    #[test]
    fn test_high_degree_subjects_threshold_1() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![tp(":a", ":p", ":b"), tp(":b", ":q", ":c")]);
        let hs = c.high_degree_subjects(1);
        assert_eq!(hs.len(), 2);
    }

    #[test]
    fn test_high_degree_subjects_none_qualify() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":a", ":p", ":b"));
        let hs = c.high_degree_subjects(5);
        assert!(hs.is_empty());
    }

    #[test]
    fn test_high_degree_subjects_sorted() {
        let mut c = StarStatisticsCollector::new();
        for _ in 0..3 {
            c.add_triple(tp(":zebra", ":p", ":x"));
            c.add_triple(tp(":alpha", ":p", ":y"));
        }
        let hs = c.high_degree_subjects(3);
        assert_eq!(hs, vec![":alpha".to_string(), ":zebra".to_string()]);
    }

    // ── add_triples ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_triples_batch() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![tp(":a", ":p", ":b"), tp(":b", ":q", ":c")]);
        let s = c.compute();
        assert_eq!(s.triple_count, 2);
    }

    // ── Triple pattern helpers ────────────────────────────────────────────────

    #[test]
    fn test_triple_pattern_subject_is_quoted() {
        let t = tp("<< :a :p :b >>", ":q", ":c");
        assert!(t.subject_is_quoted());
        assert!(!t.object_is_quoted());
    }

    #[test]
    fn test_triple_pattern_object_is_quoted() {
        let t = tp(":a", ":q", "<< :b :p :c >>");
        assert!(!t.subject_is_quoted());
        assert!(t.object_is_quoted());
    }

    #[test]
    fn test_triple_pattern_has_quoted() {
        let plain = tp(":a", ":p", ":b");
        assert!(!plain.has_quoted_triple());
        let quoted = tp("<< :a :p :b >>", ":q", ":c");
        assert!(quoted.has_quoted_triple());
    }

    #[test]
    fn test_default_constructor() {
        let c = StarStatisticsCollector::default();
        let s = c.compute();
        assert_eq!(s.triple_count, 0);
    }

    #[test]
    fn test_nesting_depth_only_angle_brackets_no_pairs() {
        let c = StarStatisticsCollector::new();
        // Single < and > are not <</>>, so depth is 0.
        assert_eq!(c.nesting_depth("<a>"), 0);
    }

    #[test]
    fn test_many_triples_same_subject() {
        let mut c = StarStatisticsCollector::new();
        for i in 0..10 {
            c.add_triple(tp(":subject", &format!(":pred{i}"), ":obj"));
        }
        let s = c.compute();
        assert_eq!(s.unique_subjects, 1);
        assert_eq!(s.unique_predicates, 10);
        assert_eq!(s.subject_out_degree.get(":subject"), Some(&10));
    }

    #[test]
    fn test_quoted_triple_count_multiple() {
        let mut c = StarStatisticsCollector::new();
        c.add_triples(vec![
            tp("<< :a :p :b >>", ":q", ":c"),
            tp(":x", ":y", "<< :b :p :c >>"),
            tp(":plain_s", ":plain_p", ":plain_o"),
        ]);
        let s = c.compute();
        assert_eq!(s.quoted_triple_count, 2);
    }

    #[test]
    fn test_top_predicates_deterministic_order() {
        let mut c = StarStatisticsCollector::new();
        c.add_triple(tp(":a", ":b_pred", ":x"));
        c.add_triple(tp(":a", ":a_pred", ":x"));
        // Both have frequency 1 — alphabetical tie-break
        let top = c.top_predicates(2);
        assert_eq!(top[0].0, ":a_pred");
        assert_eq!(top[1].0, ":b_pred");
    }
}
