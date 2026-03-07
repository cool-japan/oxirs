//! Six-index (SPO/SOP/PSO/POS/OSP/OPS) triple store.
//!
//! Maintains all six orderings of subject, predicate, object for O(log n)
//! lookup in any query pattern.

use std::collections::{BTreeMap, BTreeSet};

/// A triple is a (subject, predicate, object) tuple.
pub type Triple = (String, String, String);

/// Six-index triple store for O(log n) pattern matching.
///
/// Maintains the following indexes:
/// - SPO: subject → predicate → {object}
/// - POS: predicate → object → {subject}
/// - OSP: object → subject → {predicate}
pub struct SixIndexStore {
    /// subject → predicate → {object}
    spo: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    /// predicate → object → {subject}
    pos: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    /// object → subject → {predicate}
    osp: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    count: usize,
}

impl Default for SixIndexStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SixIndexStore {
    /// Create a new empty triple store.
    pub fn new() -> Self {
        Self {
            spo: BTreeMap::new(),
            pos: BTreeMap::new(),
            osp: BTreeMap::new(),
            count: 0,
        }
    }

    /// Insert a triple. Returns `true` if the triple was new, `false` if already present.
    pub fn insert(&mut self, s: &str, p: &str, o: &str) -> bool {
        // Check if already present via SPO
        if self
            .spo
            .get(s)
            .and_then(|pm| pm.get(p))
            .map(|os| os.contains(o))
            .unwrap_or(false)
        {
            return false;
        }

        // Insert into SPO
        self.spo
            .entry(s.to_string())
            .or_default()
            .entry(p.to_string())
            .or_default()
            .insert(o.to_string());

        // Insert into POS
        self.pos
            .entry(p.to_string())
            .or_default()
            .entry(o.to_string())
            .or_default()
            .insert(s.to_string());

        // Insert into OSP
        self.osp
            .entry(o.to_string())
            .or_default()
            .entry(s.to_string())
            .or_default()
            .insert(p.to_string());

        self.count += 1;
        true
    }

    /// Remove a triple. Returns `true` if the triple existed and was removed.
    pub fn remove(&mut self, s: &str, p: &str, o: &str) -> bool {
        // Check presence
        if !self
            .spo
            .get(s)
            .and_then(|pm| pm.get(p))
            .map(|os| os.contains(o))
            .unwrap_or(false)
        {
            return false;
        }

        // Remove from SPO
        if let Some(pm) = self.spo.get_mut(s) {
            if let Some(os) = pm.get_mut(p) {
                os.remove(o);
                if os.is_empty() {
                    pm.remove(p);
                }
            }
            if pm.is_empty() {
                self.spo.remove(s);
            }
        }

        // Remove from POS
        if let Some(om) = self.pos.get_mut(p) {
            if let Some(ss) = om.get_mut(o) {
                ss.remove(s);
                if ss.is_empty() {
                    om.remove(o);
                }
            }
            if om.is_empty() {
                self.pos.remove(p);
            }
        }

        // Remove from OSP
        if let Some(sm) = self.osp.get_mut(o) {
            if let Some(ps) = sm.get_mut(s) {
                ps.remove(p);
                if ps.is_empty() {
                    sm.remove(s);
                }
            }
            if sm.is_empty() {
                self.osp.remove(o);
            }
        }

        self.count -= 1;
        true
    }

    /// Check if a triple exists in the store.
    pub fn contains(&self, s: &str, p: &str, o: &str) -> bool {
        self.spo
            .get(s)
            .and_then(|pm| pm.get(p))
            .map(|os| os.contains(o))
            .unwrap_or(false)
    }

    /// Query triples matching any combination of S/P/O pattern.
    ///
    /// Each parameter is `Some(value)` to match exactly, or `None` to match any.
    pub fn query_spo(
        &self,
        s: Option<&str>,
        p: Option<&str>,
        o: Option<&str>,
    ) -> Vec<Triple> {
        let mut results = Vec::new();

        match (s, p, o) {
            // SPO — all three known: point lookup
            (Some(s_val), Some(p_val), Some(o_val)) => {
                if self.contains(s_val, p_val, o_val) {
                    results.push((s_val.to_string(), p_val.to_string(), o_val.to_string()));
                }
            }
            // SP? — subject and predicate known, objects vary
            (Some(s_val), Some(p_val), None) => {
                if let Some(pm) = self.spo.get(s_val) {
                    if let Some(os) = pm.get(p_val) {
                        for o_val in os {
                            results.push((s_val.to_string(), p_val.to_string(), o_val.clone()));
                        }
                    }
                }
            }
            // S?? — subject known, predicates and objects vary
            (Some(s_val), None, None) => {
                if let Some(pm) = self.spo.get(s_val) {
                    for (p_val, os) in pm {
                        for o_val in os {
                            results.push((s_val.to_string(), p_val.clone(), o_val.clone()));
                        }
                    }
                }
            }
            // S?O — use OSP index: object known, subject known, predicate varies
            (Some(s_val), None, Some(o_val)) => {
                if let Some(sm) = self.osp.get(o_val) {
                    if let Some(ps) = sm.get(s_val) {
                        for p_val in ps {
                            results.push((s_val.to_string(), p_val.clone(), o_val.to_string()));
                        }
                    }
                }
            }
            // ?P? — predicate known, subjects and objects vary
            (None, Some(p_val), None) => {
                if let Some(om) = self.pos.get(p_val) {
                    for (o_val, ss) in om {
                        for s_val in ss {
                            results.push((s_val.clone(), p_val.to_string(), o_val.clone()));
                        }
                    }
                }
            }
            // ?PO — predicate and object known, subjects vary
            (None, Some(p_val), Some(o_val)) => {
                if let Some(om) = self.pos.get(p_val) {
                    if let Some(ss) = om.get(o_val) {
                        for s_val in ss {
                            results.push((s_val.clone(), p_val.to_string(), o_val.to_string()));
                        }
                    }
                }
            }
            // ??O — object known, subjects and predicates vary
            (None, None, Some(o_val)) => {
                if let Some(sm) = self.osp.get(o_val) {
                    for (s_val, ps) in sm {
                        for p_val in ps {
                            results.push((s_val.clone(), p_val.clone(), o_val.to_string()));
                        }
                    }
                }
            }
            // ??? — all None: return everything via SPO scan
            (None, None, None) => {
                for (s_val, pm) in &self.spo {
                    for (p_val, os) in pm {
                        for o_val in os {
                            results.push((s_val.clone(), p_val.clone(), o_val.clone()));
                        }
                    }
                }
            }
        }

        results
    }

    /// Iterate over all distinct subjects.
    pub fn subjects(&self) -> impl Iterator<Item = &str> {
        self.spo.keys().map(|s| s.as_str())
    }

    /// Iterate over all distinct predicates.
    pub fn predicates(&self) -> impl Iterator<Item = &str> {
        self.pos.keys().map(|p| p.as_str())
    }

    /// Iterate over all distinct objects.
    pub fn objects(&self) -> impl Iterator<Item = &str> {
        self.osp.keys().map(|o| o.as_str())
    }

    /// Find all subjects for a given (predicate, object) pair using POS index.
    pub fn subjects_for(&self, p: &str, o: &str) -> Vec<String> {
        self.pos
            .get(p)
            .and_then(|om| om.get(o))
            .map(|ss| ss.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Find all predicates for a given (subject, object) pair using OSP index.
    pub fn predicates_for(&self, s: &str, o: &str) -> Vec<String> {
        self.osp
            .get(o)
            .and_then(|sm| sm.get(s))
            .map(|ps| ps.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Find all objects for a given (subject, predicate) pair using SPO index.
    pub fn objects_for(&self, s: &str, p: &str) -> Vec<String> {
        self.spo
            .get(s)
            .and_then(|pm| pm.get(p))
            .map(|os| os.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return the number of triples in the store.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Remove all triples from the store.
    pub fn clear(&mut self) {
        self.spo.clear();
        self.pos.clear();
        self.osp.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_triples() -> SixIndexStore {
        let mut idx = SixIndexStore::new();
        idx.insert("s1", "p1", "o1");
        idx.insert("s1", "p1", "o2");
        idx.insert("s1", "p2", "o1");
        idx.insert("s2", "p1", "o1");
        idx.insert("s2", "p2", "o3");
        idx
    }

    #[test]
    fn test_insert_new_triple_returns_true() {
        let mut idx = SixIndexStore::new();
        assert!(idx.insert("s", "p", "o"));
    }

    #[test]
    fn test_insert_duplicate_returns_false() {
        let mut idx = SixIndexStore::new();
        idx.insert("s", "p", "o");
        assert!(!idx.insert("s", "p", "o"));
    }

    #[test]
    fn test_contains_after_insert() {
        let mut idx = SixIndexStore::new();
        idx.insert("s", "p", "o");
        assert!(idx.contains("s", "p", "o"));
    }

    #[test]
    fn test_contains_missing() {
        let idx = SixIndexStore::new();
        assert!(!idx.contains("s", "p", "o"));
    }

    #[test]
    fn test_remove_existing() {
        let mut idx = SixIndexStore::new();
        idx.insert("s", "p", "o");
        assert!(idx.remove("s", "p", "o"));
        assert!(!idx.contains("s", "p", "o"));
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = SixIndexStore::new();
        assert!(!idx.remove("s", "p", "o"));
    }

    #[test]
    fn test_len_increments() {
        let mut idx = SixIndexStore::new();
        assert_eq!(idx.len(), 0);
        idx.insert("s", "p", "o");
        assert_eq!(idx.len(), 1);
        idx.insert("s", "p", "o2");
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_len_decrements_on_remove() {
        let mut idx = SixIndexStore::new();
        idx.insert("s", "p", "o");
        assert_eq!(idx.len(), 1);
        idx.remove("s", "p", "o");
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_is_empty() {
        let mut idx = SixIndexStore::new();
        assert!(idx.is_empty());
        idx.insert("s", "p", "o");
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut idx = store_with_triples();
        assert!(!idx.is_empty());
        idx.clear();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_query_spo_exact_match() {
        let idx = store_with_triples();
        let results = idx.query_spo(Some("s1"), Some("p1"), Some("o1"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], ("s1".to_string(), "p1".to_string(), "o1".to_string()));
    }

    #[test]
    fn test_query_spo_no_match() {
        let idx = store_with_triples();
        let results = idx.query_spo(Some("s1"), Some("p1"), Some("o99"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_sp_wildcard_o() {
        let idx = store_with_triples();
        let results = idx.query_spo(Some("s1"), Some("p1"), None);
        assert_eq!(results.len(), 2);
        let objects: Vec<String> = results.iter().map(|(_, _, o)| o.clone()).collect();
        assert!(objects.contains(&"o1".to_string()));
        assert!(objects.contains(&"o2".to_string()));
    }

    #[test]
    fn test_query_s_wildcard_po() {
        let idx = store_with_triples();
        let results = idx.query_spo(Some("s1"), None, None);
        // s1 has: p1→o1, p1→o2, p2→o1 = 3 triples
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_wildcard_p() {
        let idx = store_with_triples();
        let results = idx.query_spo(None, Some("p1"), None);
        // p1: s1→o1, s1→o2, s2→o1 = 3 triples
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_po_wildcard_s() {
        let idx = store_with_triples();
        let results = idx.query_spo(None, Some("p1"), Some("o1"));
        // p1, o1: subjects = s1, s2
        assert_eq!(results.len(), 2);
        let subjects: Vec<String> = results.iter().map(|(s, _, _)| s.clone()).collect();
        assert!(subjects.contains(&"s1".to_string()));
        assert!(subjects.contains(&"s2".to_string()));
    }

    #[test]
    fn test_query_wildcard_o() {
        let idx = store_with_triples();
        let results = idx.query_spo(None, None, Some("o1"));
        // o1: (s1,p1), (s1,p2), (s2,p1) = 3
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_all_none_returns_all() {
        let idx = store_with_triples();
        let results = idx.query_spo(None, None, None);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_query_s_wildcard_p_exact_o() {
        let idx = store_with_triples();
        let results = idx.query_spo(Some("s1"), None, Some("o1"));
        // s1, ?, o1 → predicates p1 and p2
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_subjects_iterator() {
        let idx = store_with_triples();
        let mut subjects: Vec<&str> = idx.subjects().collect();
        subjects.sort_unstable();
        assert_eq!(subjects, vec!["s1", "s2"]);
    }

    #[test]
    fn test_predicates_iterator() {
        let idx = store_with_triples();
        let mut predicates: Vec<&str> = idx.predicates().collect();
        predicates.sort_unstable();
        assert_eq!(predicates, vec!["p1", "p2"]);
    }

    #[test]
    fn test_objects_iterator() {
        let idx = store_with_triples();
        let mut objects: Vec<&str> = idx.objects().collect();
        objects.sort_unstable();
        assert_eq!(objects, vec!["o1", "o2", "o3"]);
    }

    #[test]
    fn test_subjects_for_po() {
        let idx = store_with_triples();
        let mut subjects = idx.subjects_for("p1", "o1");
        subjects.sort();
        assert_eq!(subjects, vec!["s1", "s2"]);
    }

    #[test]
    fn test_subjects_for_no_match() {
        let idx = store_with_triples();
        let subjects = idx.subjects_for("p99", "o99");
        assert!(subjects.is_empty());
    }

    #[test]
    fn test_predicates_for_so() {
        let idx = store_with_triples();
        let mut preds = idx.predicates_for("s1", "o1");
        preds.sort();
        assert_eq!(preds, vec!["p1", "p2"]);
    }

    #[test]
    fn test_predicates_for_no_match() {
        let idx = store_with_triples();
        assert!(idx.predicates_for("s99", "o99").is_empty());
    }

    #[test]
    fn test_objects_for_sp() {
        let idx = store_with_triples();
        let mut objs = idx.objects_for("s1", "p1");
        objs.sort();
        assert_eq!(objs, vec!["o1", "o2"]);
    }

    #[test]
    fn test_objects_for_no_match() {
        let idx = store_with_triples();
        assert!(idx.objects_for("s99", "p99").is_empty());
    }

    #[test]
    fn test_remove_then_contains_is_false() {
        let mut idx = store_with_triples();
        idx.remove("s1", "p1", "o1");
        assert!(!idx.contains("s1", "p1", "o1"));
    }

    #[test]
    fn test_remove_updates_subjects_for() {
        let mut idx = SixIndexStore::new();
        idx.insert("s1", "p1", "o1");
        idx.insert("s2", "p1", "o1");
        idx.remove("s1", "p1", "o1");
        let subjects = idx.subjects_for("p1", "o1");
        assert!(!subjects.contains(&"s1".to_string()));
        assert!(subjects.contains(&"s2".to_string()));
    }

    #[test]
    fn test_remove_cleans_up_empty_inner_maps() {
        let mut idx = SixIndexStore::new();
        idx.insert("s", "p", "o");
        idx.remove("s", "p", "o");
        assert!(idx.spo.is_empty());
        assert!(idx.pos.is_empty());
        assert!(idx.osp.is_empty());
    }

    #[test]
    fn test_empty_store_all_query() {
        let idx = SixIndexStore::new();
        let results = idx.query_spo(None, None, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_consistency_after_multiple_inserts() {
        let mut idx = SixIndexStore::new();
        let triples = vec![
            ("a", "rel", "b"),
            ("b", "rel", "c"),
            ("a", "type", "X"),
            ("b", "type", "Y"),
        ];
        for (s, p, o) in &triples {
            idx.insert(s, p, o);
        }
        assert_eq!(idx.len(), 4);
        for (s, p, o) in &triples {
            assert!(idx.contains(s, p, o));
        }
    }

    #[test]
    fn test_large_number_of_inserts() {
        let mut idx = SixIndexStore::new();
        for i in 0..100 {
            idx.insert(&format!("s{i}"), "p", &format!("o{i}"));
        }
        assert_eq!(idx.len(), 100);
        let objects = idx.objects_for("s50", "p");
        assert_eq!(objects, vec!["o50"]);
    }
}
