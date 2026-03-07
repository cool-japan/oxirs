//! Range queries on indexed triple components.
//!
//! Provides a three-way `BTreeMap`-based index (SPO / POS / OSP) that supports
//! efficient range scans by subject prefix, exact predicate match, or
//! object lexicographic range.

use std::collections::BTreeMap;

/// A range predicate for triple-component queries.
#[derive(Debug, Clone, Default)]
pub struct TripleRange {
    /// If set, only triples whose subject starts with this prefix are returned.
    pub subject_prefix: Option<String>,
    /// If set, only triples with exactly this predicate are returned.
    pub predicate_exact: Option<String>,
    /// If set, only triples whose object is lexicographically >= this value.
    pub object_min: Option<String>,
    /// If set, only triples whose object is lexicographically <= this value.
    pub object_max: Option<String>,
}

impl TripleRange {
    /// Construct an empty range that matches all triples.
    pub fn new() -> Self {
        Self::default()
    }

    /// Match only triples where the subject starts with the given prefix.
    pub fn with_subject_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.subject_prefix = Some(prefix.into());
        self
    }

    /// Match only triples with exactly this predicate.
    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate_exact = Some(predicate.into());
        self
    }

    /// Match only triples whose object is >= this lower bound.
    pub fn with_object_min(mut self, min: impl Into<String>) -> Self {
        self.object_min = Some(min.into());
        self
    }

    /// Match only triples whose object is <= this upper bound.
    pub fn with_object_max(mut self, max: impl Into<String>) -> Self {
        self.object_max = Some(max.into());
        self
    }

    /// Evaluate whether a `(subject, predicate, object)` triple matches this range.
    fn matches(&self, s: &str, p: &str, o: &str) -> bool {
        if let Some(ref prefix) = self.subject_prefix {
            if !s.starts_with(prefix.as_str()) {
                return false;
            }
        }
        if let Some(ref exact) = self.predicate_exact {
            if p != exact.as_str() {
                return false;
            }
        }
        if let Some(ref min) = self.object_min {
            if o < min.as_str() {
                return false;
            }
        }
        if let Some(ref max) = self.object_max {
            if o > max.as_str() {
                return false;
            }
        }
        true
    }
}

/// A three-index structure for efficient triple range queries.
///
/// Maintains three `BTreeMap`s keyed by (S,P,O), (P,O,S), and (O,S,P)
/// respectively, enabling efficient scans regardless of the query access
/// pattern.
pub struct RangeIndex {
    /// Subject → Predicate → Object
    spo: BTreeMap<(String, String, String), ()>,
    /// Predicate → Object → Subject
    pos: BTreeMap<(String, String, String), ()>,
    /// Object → Subject → Predicate
    osp: BTreeMap<(String, String, String), ()>,
}

impl Default for RangeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            spo: BTreeMap::new(),
            pos: BTreeMap::new(),
            osp: BTreeMap::new(),
        }
    }

    /// Insert a triple into all three indexes.
    pub fn insert(&mut self, s: &str, p: &str, o: &str) {
        self.spo
            .insert((s.to_string(), p.to_string(), o.to_string()), ());
        self.pos
            .insert((p.to_string(), o.to_string(), s.to_string()), ());
        self.osp
            .insert((o.to_string(), s.to_string(), p.to_string()), ());
    }

    /// Remove a triple from all three indexes.  Returns `true` if the triple
    /// was present, `false` if it was not found.
    pub fn remove(&mut self, s: &str, p: &str, o: &str) -> bool {
        let key_spo = (s.to_string(), p.to_string(), o.to_string());
        if self.spo.remove(&key_spo).is_none() {
            return false;
        }
        self.pos
            .remove(&(p.to_string(), o.to_string(), s.to_string()));
        self.osp
            .remove(&(o.to_string(), s.to_string(), p.to_string()));
        true
    }

    /// Query triples matching the given range descriptor.
    ///
    /// Internally chooses the best index depending on which constraints are
    /// set, falling back to a full SPO scan with in-memory filtering when no
    /// single-index acceleration is possible.
    pub fn query(&self, range: &TripleRange) -> Vec<(String, String, String)> {
        let mut results = Vec::new();

        // Fast path: predicate-only or predicate + object range — use POS index
        if range.predicate_exact.is_some()
            && range.subject_prefix.is_none()
            && (range.object_min.is_some() || range.object_max.is_some())
        {
            let p = range.predicate_exact.as_deref().unwrap_or("");
            let o_lo: &str = range.object_min.as_deref().unwrap_or("");
            let o_hi: &str = range.object_max.as_deref().unwrap_or("\u{10FFFF}");

            let from = (p.to_string(), o_lo.to_string(), String::new());
            let to = (p.to_string(), o_hi.to_string(), "\u{10FFFF}".to_string());

            for ((kp, ko, ks), _) in self.pos.range(from..=to) {
                if kp != p {
                    break;
                }
                if ko.as_str() < o_lo || ko.as_str() > o_hi {
                    continue;
                }
                if range.matches(ks.as_str(), kp.as_str(), ko.as_str()) {
                    results.push((ks.clone(), kp.clone(), ko.clone()));
                }
            }
            return results;
        }

        // Predicate-only fast path — use POS index
        if let Some(ref p) = range.predicate_exact {
            if range.subject_prefix.is_none() && range.object_min.is_none() && range.object_max.is_none() {
                let from = (p.clone(), String::new(), String::new());
                let to = (p.clone(), "\u{10FFFF}".to_string(), "\u{10FFFF}".to_string());
                for ((kp, ko, ks), _) in self.pos.range(from..=to) {
                    if kp != p.as_str() {
                        break;
                    }
                    results.push((ks.clone(), kp.clone(), ko.clone()));
                }
                return results;
            }
        }

        // General path — iterate SPO and apply filter
        for ((s, p, o), _) in &self.spo {
            if range.matches(s.as_str(), p.as_str(), o.as_str()) {
                results.push((s.clone(), p.clone(), o.clone()));
            }
        }

        results
    }

    /// Return the total number of triples stored.
    pub fn count(&self) -> usize {
        self.spo.len()
    }

    /// Return the distinct subjects associated with the given predicate.
    pub fn subjects_for_predicate(&self, predicate: &str) -> Vec<String> {
        let from = (predicate.to_string(), String::new(), String::new());
        let to = (
            predicate.to_string(),
            "\u{10FFFF}".to_string(),
            "\u{10FFFF}".to_string(),
        );
        let mut subjects: Vec<String> = self
            .pos
            .range(from..=to)
            .filter(|((p, _, _), _)| p.as_str() == predicate)
            .map(|((_, _, s), _)| s.clone())
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    /// Return the distinct objects associated with the given predicate.
    pub fn objects_for_predicate(&self, predicate: &str) -> Vec<String> {
        let from = (predicate.to_string(), String::new(), String::new());
        let to = (
            predicate.to_string(),
            "\u{10FFFF}".to_string(),
            "\u{10FFFF}".to_string(),
        );
        let mut objects: Vec<String> = self
            .pos
            .range(from..=to)
            .filter(|((p, _, _), _)| p.as_str() == predicate)
            .map(|((_, o, _), _)| o.clone())
            .collect();
        objects.sort();
        objects.dedup();
        objects
    }

    /// Return the distinct predicates for the given subject.
    pub fn predicates_for_subject(&self, subject: &str) -> Vec<String> {
        let from = (subject.to_string(), String::new(), String::new());
        let to = (
            subject.to_string(),
            "\u{10FFFF}".to_string(),
            "\u{10FFFF}".to_string(),
        );
        let mut predicates: Vec<String> = self
            .spo
            .range(from..=to)
            .filter(|((s, _, _), _)| s.as_str() == subject)
            .map(|((_, p, _), _)| p.clone())
            .collect();
        predicates.sort();
        predicates.dedup();
        predicates
    }

    /// Return all triples where the subject equals `subject`.
    pub fn triples_for_subject(&self, subject: &str) -> Vec<(String, String, String)> {
        let from = (subject.to_string(), String::new(), String::new());
        let to = (
            subject.to_string(),
            "\u{10FFFF}".to_string(),
            "\u{10FFFF}".to_string(),
        );
        self.spo
            .range(from..=to)
            .filter(|((s, _, _), _)| s.as_str() == subject)
            .map(|((s, p, o), _)| (s.clone(), p.clone(), o.clone()))
            .collect()
    }

    /// Return all triples where the predicate equals `predicate`.
    pub fn triples_for_predicate(&self, predicate: &str) -> Vec<(String, String, String)> {
        let from = (predicate.to_string(), String::new(), String::new());
        let to = (
            predicate.to_string(),
            "\u{10FFFF}".to_string(),
            "\u{10FFFF}".to_string(),
        );
        self.pos
            .range(from..=to)
            .filter(|((p, _, _), _)| p.as_str() == predicate)
            .map(|((p, o, s), _)| (s.clone(), p.clone(), o.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_index() -> RangeIndex {
        let mut idx = RangeIndex::new();
        idx.insert("s:alice", "p:age", "30");
        idx.insert("s:alice", "p:name", "Alice");
        idx.insert("s:bob", "p:age", "25");
        idx.insert("s:bob", "p:name", "Bob");
        idx.insert("s:carol", "p:age", "35");
        idx.insert("s:carol", "p:name", "Carol");
        idx.insert("s:dave", "p:score", "100");
        idx
    }

    // --- insert / count ---
    #[test]
    fn test_insert_and_count() {
        let mut idx = RangeIndex::new();
        assert_eq!(idx.count(), 0);
        idx.insert("s", "p", "o");
        assert_eq!(idx.count(), 1);
        idx.insert("s", "p", "o2");
        assert_eq!(idx.count(), 2);
    }

    #[test]
    fn test_insert_duplicate_no_increase() {
        let mut idx = RangeIndex::new();
        idx.insert("s", "p", "o");
        idx.insert("s", "p", "o"); // duplicate
        assert_eq!(idx.count(), 1);
    }

    // --- remove ---
    #[test]
    fn test_remove_present() {
        let mut idx = RangeIndex::new();
        idx.insert("s", "p", "o");
        assert!(idx.remove("s", "p", "o"));
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_remove_absent() {
        let mut idx = RangeIndex::new();
        assert!(!idx.remove("s", "p", "o"));
    }

    #[test]
    fn test_remove_correct_triple() {
        let mut idx = RangeIndex::new();
        idx.insert("s", "p", "o1");
        idx.insert("s", "p", "o2");
        idx.remove("s", "p", "o1");
        assert_eq!(idx.count(), 1);
        let q = TripleRange::new();
        let results = idx.query(&q);
        assert_eq!(results[0].2, "o2");
    }

    // --- query: no constraints ---
    #[test]
    fn test_query_no_constraints_returns_all() {
        let idx = populated_index();
        let results = idx.query(&TripleRange::new());
        assert_eq!(results.len(), 7);
    }

    // --- query: subject prefix ---
    #[test]
    fn test_query_subject_prefix() {
        let idx = populated_index();
        let range = TripleRange::new().with_subject_prefix("s:alice");
        let results = idx.query(&range);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(s, _, _)| s.starts_with("s:alice")));
    }

    #[test]
    fn test_query_subject_prefix_no_match() {
        let idx = populated_index();
        let range = TripleRange::new().with_subject_prefix("s:zzzz");
        assert!(idx.query(&range).is_empty());
    }

    // --- query: predicate exact ---
    #[test]
    fn test_query_predicate_exact() {
        let idx = populated_index();
        let range = TripleRange::new().with_predicate("p:age");
        let results = idx.query(&range);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, p, _)| p == "p:age"));
    }

    #[test]
    fn test_query_predicate_exact_no_match() {
        let idx = populated_index();
        let range = TripleRange::new().with_predicate("p:unknown");
        assert!(idx.query(&range).is_empty());
    }

    // --- query: object range ---
    #[test]
    fn test_query_object_min() {
        let idx = populated_index();
        // ages are "25", "30", "35"
        let range = TripleRange::new()
            .with_predicate("p:age")
            .with_object_min("30");
        let results = idx.query(&range);
        assert_eq!(results.len(), 2); // "30" and "35"
    }

    #[test]
    fn test_query_object_max() {
        let idx = populated_index();
        let range = TripleRange::new()
            .with_predicate("p:age")
            .with_object_max("30");
        let results = idx.query(&range);
        assert_eq!(results.len(), 2); // "25" and "30"
    }

    #[test]
    fn test_query_object_range() {
        let idx = populated_index();
        let range = TripleRange::new()
            .with_predicate("p:age")
            .with_object_min("25")
            .with_object_max("30");
        let results = idx.query(&range);
        assert_eq!(results.len(), 2); // "25" and "30"
    }

    #[test]
    fn test_query_object_range_exclusive() {
        let idx = populated_index();
        let range = TripleRange::new()
            .with_predicate("p:age")
            .with_object_min("26")
            .with_object_max("34");
        let results = idx.query(&range);
        assert_eq!(results.len(), 1); // only "30"
    }

    // --- query: combined constraints ---
    #[test]
    fn test_query_subject_and_predicate() {
        let idx = populated_index();
        let range = TripleRange::new()
            .with_subject_prefix("s:alice")
            .with_predicate("p:age");
        let results = idx.query(&range);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], ("s:alice".to_string(), "p:age".to_string(), "30".to_string()));
    }

    // --- subjects_for_predicate ---
    #[test]
    fn test_subjects_for_predicate() {
        let idx = populated_index();
        let subs = idx.subjects_for_predicate("p:age");
        assert_eq!(subs.len(), 3);
        assert!(subs.contains(&"s:alice".to_string()));
        assert!(subs.contains(&"s:bob".to_string()));
        assert!(subs.contains(&"s:carol".to_string()));
    }

    #[test]
    fn test_subjects_for_predicate_empty() {
        let idx = populated_index();
        assert!(idx.subjects_for_predicate("p:unknown").is_empty());
    }

    // --- objects_for_predicate ---
    #[test]
    fn test_objects_for_predicate() {
        let idx = populated_index();
        let objs = idx.objects_for_predicate("p:name");
        assert_eq!(objs.len(), 3);
        assert!(objs.contains(&"Alice".to_string()));
        assert!(objs.contains(&"Bob".to_string()));
        assert!(objs.contains(&"Carol".to_string()));
    }

    #[test]
    fn test_objects_for_predicate_empty() {
        let idx = populated_index();
        assert!(idx.objects_for_predicate("p:missing").is_empty());
    }

    // --- predicates_for_subject ---
    #[test]
    fn test_predicates_for_subject() {
        let idx = populated_index();
        let preds = idx.predicates_for_subject("s:alice");
        assert_eq!(preds.len(), 2);
        assert!(preds.contains(&"p:age".to_string()));
        assert!(preds.contains(&"p:name".to_string()));
    }

    #[test]
    fn test_predicates_for_subject_empty() {
        let idx = populated_index();
        assert!(idx.predicates_for_subject("s:unknown").is_empty());
    }

    // --- triples_for_subject ---
    #[test]
    fn test_triples_for_subject() {
        let idx = populated_index();
        let trips = idx.triples_for_subject("s:bob");
        assert_eq!(trips.len(), 2);
        assert!(trips.iter().all(|(s, _, _)| s == "s:bob"));
    }

    #[test]
    fn test_triples_for_subject_empty() {
        let idx = populated_index();
        assert!(idx.triples_for_subject("s:nobody").is_empty());
    }

    // --- triples_for_predicate ---
    #[test]
    fn test_triples_for_predicate() {
        let idx = populated_index();
        let trips = idx.triples_for_predicate("p:score");
        assert_eq!(trips.len(), 1);
        assert_eq!(trips[0].0, "s:dave");
    }

    #[test]
    fn test_triples_for_predicate_multiple() {
        let idx = populated_index();
        let trips = idx.triples_for_predicate("p:age");
        assert_eq!(trips.len(), 3);
    }

    // --- empty index ---
    #[test]
    fn test_empty_index_query() {
        let idx = RangeIndex::new();
        assert!(idx.query(&TripleRange::new()).is_empty());
    }

    #[test]
    fn test_empty_index_count_zero() {
        assert_eq!(RangeIndex::new().count(), 0);
    }

    // --- TripleRange builder ---
    #[test]
    fn test_triple_range_default() {
        let r = TripleRange::new();
        assert!(r.subject_prefix.is_none());
        assert!(r.predicate_exact.is_none());
        assert!(r.object_min.is_none());
        assert!(r.object_max.is_none());
    }

    #[test]
    fn test_triple_range_builders() {
        let r = TripleRange::new()
            .with_subject_prefix("s:")
            .with_predicate("p:")
            .with_object_min("a")
            .with_object_max("z");
        assert_eq!(r.subject_prefix.as_deref(), Some("s:"));
        assert_eq!(r.predicate_exact.as_deref(), Some("p:"));
        assert_eq!(r.object_min.as_deref(), Some("a"));
        assert_eq!(r.object_max.as_deref(), Some("z"));
    }

    // --- remove consistency ---
    #[test]
    fn test_remove_updates_all_indexes() {
        let mut idx = RangeIndex::new();
        idx.insert("s", "p:age", "30");
        idx.remove("s", "p:age", "30");
        assert!(idx.subjects_for_predicate("p:age").is_empty());
        assert!(idx.objects_for_predicate("p:age").is_empty());
        assert!(idx.triples_for_subject("s").is_empty());
    }
}
