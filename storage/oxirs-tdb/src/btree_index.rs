//! B-tree based index for triple pattern matching.
//!
//! Maintains three sorted indices (SPO, POS, OSP) to enable efficient
//! lookup for any combination of bound/unbound subject, predicate, object.

use std::collections::BTreeMap;

/// A triple key as (subject_id, predicate_id, object_id).
pub type TripleKey = (u32, u32, u32);

/// A B-tree triple index supporting efficient pattern matching.
///
/// Three index orderings are maintained:
/// - **spo**: (s, p, o) — primary ordering
/// - **pos**: (p, o, s) — efficient for `?s p o` patterns  
/// - **osp**: (o, s, p) — efficient for `s ?p o` style patterns
pub struct BTreeIndex {
    /// Primary index: (subject, predicate, object)
    spo: BTreeMap<TripleKey, ()>,
    /// (predicate, object, subject)
    pos: BTreeMap<(u32, u32, u32), ()>,
    /// (object, subject, predicate)
    osp: BTreeMap<(u32, u32, u32), ()>,
}

impl BTreeIndex {
    /// Create an empty B-tree index.
    pub fn new() -> Self {
        BTreeIndex {
            spo: BTreeMap::new(),
            pos: BTreeMap::new(),
            osp: BTreeMap::new(),
        }
    }

    /// Insert a triple. Returns `false` if the triple already exists.
    pub fn insert(&mut self, s: u32, p: u32, o: u32) -> bool {
        if self.spo.contains_key(&(s, p, o)) {
            return false;
        }
        self.spo.insert((s, p, o), ());
        self.pos.insert((p, o, s), ());
        self.osp.insert((o, s, p), ());
        true
    }

    /// Remove a triple. Returns `false` if the triple does not exist.
    pub fn remove(&mut self, s: u32, p: u32, o: u32) -> bool {
        if !self.spo.contains_key(&(s, p, o)) {
            return false;
        }
        self.spo.remove(&(s, p, o));
        self.pos.remove(&(p, o, s));
        self.osp.remove(&(o, s, p));
        true
    }

    /// Check whether a triple exists.
    pub fn contains(&self, s: u32, p: u32, o: u32) -> bool {
        self.spo.contains_key(&(s, p, o))
    }

    /// Return the number of stored triples.
    pub fn count(&self) -> usize {
        self.spo.len()
    }

    /// Match triples against a pattern where `None` means wildcard.
    ///
    /// Selects the most efficient index for the given binding pattern.
    pub fn match_pattern(&self, s: Option<u32>, p: Option<u32>, o: Option<u32>) -> Vec<TripleKey> {
        match (s, p, o) {
            // All bound — point lookup
            (Some(sv), Some(pv), Some(ov)) => {
                if self.spo.contains_key(&(sv, pv, ov)) {
                    vec![(sv, pv, ov)]
                } else {
                    vec![]
                }
            }
            // s, p bound — range scan in SPO
            (Some(sv), Some(pv), None) => self
                .spo
                .range((sv, pv, 0)..=(sv, pv, u32::MAX))
                .map(|(k, _)| *k)
                .collect(),
            // s bound only — range scan in SPO
            (Some(sv), None, None) => self
                .spo
                .range((sv, 0, 0)..=(sv, u32::MAX, u32::MAX))
                .map(|(k, _)| *k)
                .collect(),
            // p, o bound — use POS index
            (None, Some(pv), Some(ov)) => self
                .pos
                .range((pv, ov, 0)..=(pv, ov, u32::MAX))
                .map(|(k, _)| (k.2, k.0, k.1))
                .collect(),
            // p bound only — use POS index
            (None, Some(pv), None) => self
                .pos
                .range((pv, 0, 0)..=(pv, u32::MAX, u32::MAX))
                .map(|(k, _)| (k.2, k.0, k.1))
                .collect(),
            // o bound only — use OSP index
            (None, None, Some(ov)) => self
                .osp
                .range((ov, 0, 0)..=(ov, u32::MAX, u32::MAX))
                .map(|(k, _)| (k.1, k.2, k.0))
                .collect(),
            // s, o bound — use OSP index
            (Some(sv), None, Some(ov)) => self
                .osp
                .range((ov, sv, 0)..=(ov, sv, u32::MAX))
                .map(|(k, _)| (k.1, k.2, k.0))
                .collect(),
            // All wildcard — full scan via SPO
            (None, None, None) => self.spo.keys().copied().collect(),
        }
    }

    /// Return sorted unique subject IDs.
    pub fn subjects(&self) -> Vec<u32> {
        let mut seen: Vec<u32> = Vec::new();
        let mut last: Option<u32> = None;
        for (s, _, _) in self.spo.keys() {
            if last != Some(*s) {
                seen.push(*s);
                last = Some(*s);
            }
        }
        seen
    }

    /// Return sorted unique predicate IDs.
    pub fn predicates(&self) -> Vec<u32> {
        let mut seen: Vec<u32> = Vec::new();
        let mut last: Option<u32> = None;
        for (p, _, _) in self.pos.keys() {
            if last != Some(*p) {
                seen.push(*p);
                last = Some(*p);
            }
        }
        seen
    }

    /// Return sorted unique object IDs.
    pub fn objects(&self) -> Vec<u32> {
        let mut seen: Vec<u32> = Vec::new();
        let mut last: Option<u32> = None;
        for (o, _, _) in self.osp.keys() {
            if last != Some(*o) {
                seen.push(*o);
                last = Some(*o);
            }
        }
        seen
    }

    /// Remove all triples.
    pub fn clear(&mut self) {
        self.spo.clear();
        self.pos.clear();
        self.osp.clear();
    }

    /// Iterate triples in SPO order.
    pub fn iter_spo(&self) -> impl Iterator<Item = &TripleKey> {
        self.spo.keys()
    }
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> BTreeIndex {
        let mut idx = BTreeIndex::new();
        // Insert some test triples
        idx.insert(1, 10, 100);
        idx.insert(1, 10, 101);
        idx.insert(1, 11, 200);
        idx.insert(2, 10, 100);
        idx.insert(2, 12, 300);
        idx.insert(3, 13, 400);
        idx
    }

    #[test]
    fn test_new_empty() {
        let idx = BTreeIndex::new();
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_insert_returns_true() {
        let mut idx = BTreeIndex::new();
        assert!(idx.insert(1, 2, 3));
    }

    #[test]
    fn test_insert_duplicate_returns_false() {
        let mut idx = BTreeIndex::new();
        idx.insert(1, 2, 3);
        assert!(!idx.insert(1, 2, 3));
    }

    #[test]
    fn test_count_after_insert() {
        let idx = make_index();
        assert_eq!(idx.count(), 6);
    }

    #[test]
    fn test_contains_existing() {
        let idx = make_index();
        assert!(idx.contains(1, 10, 100));
    }

    #[test]
    fn test_contains_missing() {
        let idx = make_index();
        assert!(!idx.contains(99, 99, 99));
    }

    #[test]
    fn test_remove_existing() {
        let mut idx = make_index();
        assert!(idx.remove(1, 10, 100));
        assert!(!idx.contains(1, 10, 100));
        assert_eq!(idx.count(), 5);
    }

    #[test]
    fn test_remove_missing() {
        let mut idx = make_index();
        assert!(!idx.remove(99, 99, 99));
    }

    #[test]
    fn test_clear() {
        let mut idx = make_index();
        idx.clear();
        assert_eq!(idx.count(), 0);
    }

    // Pattern matching — all 8 combinations
    #[test]
    fn test_pattern_all_bound_found() {
        let idx = make_index();
        let res = idx.match_pattern(Some(1), Some(10), Some(100));
        assert_eq!(res, vec![(1, 10, 100)]);
    }

    #[test]
    fn test_pattern_all_bound_not_found() {
        let idx = make_index();
        let res = idx.match_pattern(Some(1), Some(10), Some(999));
        assert!(res.is_empty());
    }

    #[test]
    fn test_pattern_s_p_bound() {
        let idx = make_index();
        let mut res = idx.match_pattern(Some(1), Some(10), None);
        res.sort();
        assert_eq!(res, vec![(1, 10, 100), (1, 10, 101)]);
    }

    #[test]
    fn test_pattern_s_bound_only() {
        let idx = make_index();
        let mut res = idx.match_pattern(Some(1), None, None);
        res.sort();
        assert_eq!(res, vec![(1, 10, 100), (1, 10, 101), (1, 11, 200)]);
    }

    #[test]
    fn test_pattern_p_o_bound() {
        let idx = make_index();
        let mut res = idx.match_pattern(None, Some(10), Some(100));
        res.sort();
        assert_eq!(res, vec![(1, 10, 100), (2, 10, 100)]);
    }

    #[test]
    fn test_pattern_p_bound_only() {
        let idx = make_index();
        let mut res = idx.match_pattern(None, Some(10), None);
        res.sort();
        assert_eq!(res, vec![(1, 10, 100), (1, 10, 101), (2, 10, 100)]);
    }

    #[test]
    fn test_pattern_o_bound_only() {
        let idx = make_index();
        let mut res = idx.match_pattern(None, None, Some(100));
        res.sort();
        assert_eq!(res, vec![(1, 10, 100), (2, 10, 100)]);
    }

    #[test]
    fn test_pattern_s_o_bound() {
        let idx = make_index();
        let mut res = idx.match_pattern(Some(1), None, Some(100));
        res.sort();
        assert_eq!(res, vec![(1, 10, 100)]);
    }

    #[test]
    fn test_pattern_all_wildcard() {
        let idx = make_index();
        let res = idx.match_pattern(None, None, None);
        assert_eq!(res.len(), 6);
    }

    #[test]
    fn test_pattern_empty_result() {
        let idx = make_index();
        let res = idx.match_pattern(Some(999), None, None);
        assert!(res.is_empty());
    }

    // Unique terms
    #[test]
    fn test_subjects() {
        let idx = make_index();
        let subjs = idx.subjects();
        assert_eq!(subjs, vec![1, 2, 3]);
    }

    #[test]
    fn test_predicates() {
        let idx = make_index();
        let preds = idx.predicates();
        assert_eq!(preds, vec![10, 11, 12, 13]);
    }

    #[test]
    fn test_objects() {
        let idx = make_index();
        let objs = idx.objects();
        assert_eq!(objs, vec![100, 101, 200, 300, 400]);
    }

    #[test]
    fn test_subjects_empty() {
        let idx = BTreeIndex::new();
        assert!(idx.subjects().is_empty());
    }

    #[test]
    fn test_predicates_empty() {
        let idx = BTreeIndex::new();
        assert!(idx.predicates().is_empty());
    }

    #[test]
    fn test_objects_empty() {
        let idx = BTreeIndex::new();
        assert!(idx.objects().is_empty());
    }

    #[test]
    fn test_iter_spo_order() {
        let idx = make_index();
        let triples: Vec<TripleKey> = idx.iter_spo().copied().collect();
        // Should be in SPO lexicographic order
        for w in triples.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_insert_then_remove_all_indices_consistent() {
        let mut idx = BTreeIndex::new();
        idx.insert(5, 50, 500);
        idx.remove(5, 50, 500);
        // POS and OSP should also be empty
        assert_eq!(idx.count(), 0);
        assert!(idx.match_pattern(None, Some(50), None).is_empty());
        assert!(idx.match_pattern(None, None, Some(500)).is_empty());
    }

    #[test]
    fn test_default_trait() {
        let idx = BTreeIndex::default();
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_large_insert() {
        let mut idx = BTreeIndex::new();
        for s in 0..10u32 {
            for p in 0..10u32 {
                for o in 0..10u32 {
                    idx.insert(s, p, o);
                }
            }
        }
        assert_eq!(idx.count(), 1000);
        assert_eq!(idx.subjects().len(), 10);
        assert_eq!(idx.predicates().len(), 10);
        assert_eq!(idx.objects().len(), 10);
    }

    #[test]
    fn test_pattern_p_bound_returns_all_with_pred() {
        let mut idx = BTreeIndex::new();
        idx.insert(1, 5, 10);
        idx.insert(2, 5, 20);
        idx.insert(3, 6, 30);
        let res = idx.match_pattern(None, Some(5), None);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_remove_middle_triple() {
        let mut idx = make_index();
        idx.remove(2, 10, 100);
        let res = idx.match_pattern(Some(2), None, None);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], (2, 12, 300));
    }
}
