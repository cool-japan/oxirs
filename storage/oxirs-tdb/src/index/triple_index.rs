//! Triple index implementation using B+Tree

use super::triple::{prefix_bounds, EmptyValue, OspKey, PosKey, SpoKey, Triple};
use crate::btree::iterator::BTreeIterator;
use crate::btree::BTree;
use crate::dictionary::NodeId;
use crate::error::Result;
use crate::storage::{BufferPool, PageId};
use std::sync::Arc;

/// Triple index for one specific ordering
pub struct TripleIndex<K>
where
    K: Ord + Clone + serde::Serialize + for<'de> serde::Deserialize<'de> + std::fmt::Debug,
{
    btree: BTree<K, EmptyValue>,
}

impl<K> TripleIndex<K>
where
    K: Ord + Clone + serde::Serialize + for<'de> serde::Deserialize<'de> + std::fmt::Debug,
{
    /// Create a new triple index
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        TripleIndex {
            btree: BTree::new(buffer_pool),
        }
    }

    /// Reconstruct a triple index from a persisted root page.
    ///
    /// `root` is `None` for an empty (never-written) index and `Some(page)` to
    /// rebuild the on-disk B+Tree via [`BTree::from_root`].
    pub fn from_root(buffer_pool: Arc<BufferPool>, root: Option<PageId>) -> Self {
        let btree = match root {
            Some(page) => BTree::from_root(buffer_pool, page),
            None => BTree::new(buffer_pool),
        };
        TripleIndex { btree }
    }

    /// Current root page of the underlying B+Tree (for superblock persistence).
    pub fn root_page(&self) -> Option<PageId> {
        self.btree.root_page()
    }

    /// Insert a triple into the index.
    ///
    /// Returns `true` if the key was newly inserted and `false` if it already
    /// existed (the underlying B+Tree overwrites duplicate keys), so callers
    /// can maintain an accurate triple count.
    pub fn insert(&mut self, key: K) -> Result<bool> {
        Ok(self.btree.insert(key, EmptyValue)?.is_none())
    }

    /// Check if a triple exists in the index
    pub fn contains(&self, key: &K) -> Result<bool> {
        Ok(self.btree.search(key)?.is_some())
    }

    /// Delete a triple from the index
    pub fn delete(&mut self, key: &K) -> Result<bool> {
        Ok(self.btree.delete(key)?.is_some())
    }

    /// Range scan over the index
    pub fn range_scan(&self, start_key: Option<K>, end_key: Option<K>) -> Result<Vec<K>> {
        let mut results = Vec::new();
        let iter = self.btree.range_scan(start_key, end_key)?;

        for item in iter {
            let (key, _value) = item?;
            results.push(key);
        }

        Ok(results)
    }
}

/// SPO index (Subject-Predicate-Object ordering)
pub type SpoIndex = TripleIndex<SpoKey>;

/// POS index (Predicate-Object-Subject ordering)
pub type PosIndex = TripleIndex<PosKey>;

/// OSP index (Object-Subject-Predicate ordering)
pub type OspIndex = TripleIndex<OspKey>;

/// Manages all three triple indexes
pub struct TripleIndexes {
    spo: SpoIndex,
    pos: PosIndex,
    osp: OspIndex,
}

impl TripleIndexes {
    /// Create new triple indexes
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        TripleIndexes {
            spo: TripleIndex::new(buffer_pool.clone()),
            pos: TripleIndex::new(buffer_pool.clone()),
            osp: TripleIndex::new(buffer_pool),
        }
    }

    /// Reconstruct all three indexes from persisted root pages (on reopen).
    pub fn from_roots(
        buffer_pool: Arc<BufferPool>,
        spo_root: Option<PageId>,
        pos_root: Option<PageId>,
        osp_root: Option<PageId>,
    ) -> Self {
        TripleIndexes {
            spo: TripleIndex::from_root(buffer_pool.clone(), spo_root),
            pos: TripleIndex::from_root(buffer_pool.clone(), pos_root),
            osp: TripleIndex::from_root(buffer_pool, osp_root),
        }
    }

    /// Current root page of the SPO index (for superblock persistence).
    pub fn spo_root(&self) -> Option<PageId> {
        self.spo.root_page()
    }

    /// Current root page of the POS index (for superblock persistence).
    pub fn pos_root(&self) -> Option<PageId> {
        self.pos.root_page()
    }

    /// Current root page of the OSP index (for superblock persistence).
    pub fn osp_root(&self) -> Option<PageId> {
        self.osp.root_page()
    }

    /// Insert a triple into all three indexes.
    ///
    /// Returns `true` if the triple was newly added and `false` if it was
    /// already present, so the store can keep an accurate triple count.
    pub fn insert(&mut self, triple: Triple) -> Result<bool> {
        let is_new = self.spo.insert(triple.into())?;
        self.pos.insert(triple.into())?;
        self.osp.insert(triple.into())?;
        Ok(is_new)
    }

    /// Check if a triple exists (uses SPO index)
    pub fn contains(&self, triple: &Triple) -> Result<bool> {
        self.spo.contains(&(*triple).into())
    }

    /// Delete a triple from all three indexes
    pub fn delete(&mut self, triple: &Triple) -> Result<bool> {
        let spo_key: SpoKey = (*triple).into();
        let pos_key: PosKey = (*triple).into();
        let osp_key: OspKey = (*triple).into();

        let exists = self.spo.delete(&spo_key)?;
        if exists {
            self.pos.delete(&pos_key)?;
            self.osp.delete(&osp_key)?;
        }

        Ok(exists)
    }

    /// Query triples with pattern matching using optimal index selection
    ///
    /// Pattern is (subject, predicate, object) where None = wildcard
    /// Returns matching triples as Triple structs
    pub fn query_pattern(
        &self,
        s: Option<NodeId>,
        p: Option<NodeId>,
        o: Option<NodeId>,
    ) -> Result<Vec<Triple>> {
        match (s, p, o) {
            // All specified - exact lookup
            (Some(s), Some(p), Some(o)) => {
                let triple = Triple::new(s, p, o);
                if self.contains(&triple)? {
                    Ok(vec![triple])
                } else {
                    Ok(Vec::new())
                }
            }

            // S and P specified - use SPO index
            (Some(s), Some(p), None) => {
                let start_key = SpoKey(s, p, NodeId::NULL);
                let end_key = SpoKey(s, p.next(), NodeId::NULL);
                let keys = self.spo.range_scan(Some(start_key), Some(end_key))?;
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.0, k.1, k.2))
                    .collect())
            }

            // S specified - use SPO index
            (Some(s), None, None) => {
                let start_key = SpoKey(s, NodeId::NULL, NodeId::NULL);
                let end_key = SpoKey(s.next(), NodeId::NULL, NodeId::NULL);
                let keys = self.spo.range_scan(Some(start_key), Some(end_key))?;
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.0, k.1, k.2))
                    .collect())
            }

            // P and O specified - use POS index
            (None, Some(p), Some(o)) => {
                let start_key = PosKey(p, o, NodeId::NULL);
                let end_key = PosKey(p, o.next(), NodeId::NULL);
                let keys = self.pos.range_scan(Some(start_key), Some(end_key))?;
                // PosKey is (p, o, s) so we need to reconstruct Triple as (s, p, o)
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.2, k.0, k.1))
                    .collect())
            }

            // P specified - use POS index
            (None, Some(p), None) => {
                let start_key = PosKey(p, NodeId::NULL, NodeId::NULL);
                let end_key = PosKey(p.next(), NodeId::NULL, NodeId::NULL);
                let keys = self.pos.range_scan(Some(start_key), Some(end_key))?;
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.2, k.0, k.1))
                    .collect())
            }

            // O and S specified - use OSP index
            (Some(s), None, Some(o)) => {
                let start_key = OspKey(o, s, NodeId::NULL);
                let end_key = OspKey(o, s.next(), NodeId::NULL);
                let keys = self.osp.range_scan(Some(start_key), Some(end_key))?;
                // OspKey is (o, s, p) so we need to reconstruct Triple as (s, p, o)
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.1, k.2, k.0))
                    .collect())
            }

            // O specified - use OSP index
            (None, None, Some(o)) => {
                let start_key = OspKey(o, NodeId::NULL, NodeId::NULL);
                let end_key = OspKey(o.next(), NodeId::NULL, NodeId::NULL);
                let keys = self.osp.range_scan(Some(start_key), Some(end_key))?;
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.1, k.2, k.0))
                    .collect())
            }

            // All wildcards - full scan (use SPO index)
            (None, None, None) => {
                let keys = self.spo.range_scan(None, None)?;
                Ok(keys
                    .into_iter()
                    .map(|k| Triple::new(k.0, k.1, k.2))
                    .collect())
            }
        }
    }

    /// Open a lazy, streaming scan over the triples matching `(s, p, o)`.
    ///
    /// Unlike [`TripleIndexes::query_pattern`], which materializes the whole
    /// result into a `Vec`, this selects the optimal index, bounds the scan to
    /// the pattern's leading prefix, and returns a [`TripleScan`] that yields
    /// one [`Triple`] at a time (decoding at most one B+Tree leaf page into
    /// memory at a time). This is the node-level primitive behind the store's
    /// streaming query iterator (F5).
    pub fn scan(
        &self,
        s: Option<NodeId>,
        p: Option<NodeId>,
        o: Option<NodeId>,
    ) -> Result<TripleScan> {
        // Index selection mirrors `query_pattern`: choose the index whose
        // leading key components cover the bound pattern components, so the
        // range scan is as tight as possible.
        let inner = if s.is_some() && p.is_none() && o.is_some() {
            // (s, _, o): OSP gives the (o, s) prefix.
            let (start, end) = prefix_bounds([o, s, p]);
            TripleScanInner::Osp(self.osp.btree.range_scan(
                start.map(|k| OspKey(k[0], k[1], k[2])),
                end.map(|k| OspKey(k[0], k[1], k[2])),
            )?)
        } else if s.is_some() {
            // (s, p, o) / (s, p, _) / (s, _, _): SPO.
            let (start, end) = prefix_bounds([s, p, o]);
            TripleScanInner::Spo(self.spo.btree.range_scan(
                start.map(|k| SpoKey(k[0], k[1], k[2])),
                end.map(|k| SpoKey(k[0], k[1], k[2])),
            )?)
        } else if p.is_some() {
            // (_, p, o) / (_, p, _): POS.
            let (start, end) = prefix_bounds([p, o, s]);
            TripleScanInner::Pos(self.pos.btree.range_scan(
                start.map(|k| PosKey(k[0], k[1], k[2])),
                end.map(|k| PosKey(k[0], k[1], k[2])),
            )?)
        } else if o.is_some() {
            // (_, _, o): OSP.
            let (start, end) = prefix_bounds([o, s, p]);
            TripleScanInner::Osp(self.osp.btree.range_scan(
                start.map(|k| OspKey(k[0], k[1], k[2])),
                end.map(|k| OspKey(k[0], k[1], k[2])),
            )?)
        } else {
            // (_, _, _): full scan via SPO.
            TripleScanInner::Spo(self.spo.btree.range_scan(None, None)?)
        };

        Ok(TripleScan { inner, s, p, o })
    }

    /// Get SPO index for queries
    pub fn spo(&self) -> &SpoIndex {
        &self.spo
    }

    /// Get POS index for queries
    pub fn pos(&self) -> &PosIndex {
        &self.pos
    }

    /// Get OSP index for queries
    pub fn osp(&self) -> &OspIndex {
        &self.osp
    }

    /// Get mutable SPO index
    pub fn spo_mut(&mut self) -> &mut SpoIndex {
        &mut self.spo
    }

    /// Get mutable POS index
    pub fn pos_mut(&mut self) -> &mut PosIndex {
        &mut self.pos
    }

    /// Get mutable OSP index
    pub fn osp_mut(&mut self) -> &mut OspIndex {
        &mut self.osp
    }
}

/// The selected index iterator backing a [`TripleScan`].
///
/// Each variant reconstructs a [`Triple`] from that index's key ordering.
enum TripleScanInner {
    /// SPO-ordered key stream: key is `(s, p, o)`.
    Spo(BTreeIterator<SpoKey, EmptyValue>),
    /// POS-ordered key stream: key is `(p, o, s)`.
    Pos(BTreeIterator<PosKey, EmptyValue>),
    /// OSP-ordered key stream: key is `(o, s, p)`.
    Osp(BTreeIterator<OspKey, EmptyValue>),
}

/// A lazy, streaming iterator over triples matching a pattern.
///
/// Created by [`TripleIndexes::scan`]. It streams keys out of the selected
/// B+Tree one leaf page at a time (never materializing the full result set),
/// reconstructs each [`Triple`] from the index ordering, and applies a residual
/// filter for any pattern component not covered by the bounded prefix.
pub struct TripleScan {
    inner: TripleScanInner,
    s: Option<NodeId>,
    p: Option<NodeId>,
    o: Option<NodeId>,
}

impl TripleScan {
    /// Residual filter for pattern components not covered by the scan prefix.
    fn matches(&self, triple: &Triple) -> bool {
        self.s.map_or(true, |s| triple.subject == s)
            && self.p.map_or(true, |p| triple.predicate == p)
            && self.o.map_or(true, |o| triple.object == o)
    }
}

impl Iterator for TripleScan {
    type Item = Result<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = match &mut self.inner {
                TripleScanInner::Spo(it) => it
                    .next()
                    .map(|r| r.map(|(k, _)| Triple::new(k.0, k.1, k.2))),
                TripleScanInner::Pos(it) => {
                    // PosKey is (p, o, s) -> Triple (s, p, o).
                    it.next()
                        .map(|r| r.map(|(k, _)| Triple::new(k.2, k.0, k.1)))
                }
                TripleScanInner::Osp(it) => {
                    // OspKey is (o, s, p) -> Triple (s, p, o).
                    it.next()
                        .map(|r| r.map(|(k, _)| Triple::new(k.1, k.2, k.0)))
                }
            };

            match next {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(triple)) => {
                    if self.matches(&triple) {
                        return Some(Ok(triple));
                    }
                    // Otherwise keep scanning (residual filter rejected it).
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::FileManager;
    use tempfile::TempDir;

    fn create_test_indexes() -> (TempDir, TripleIndexes) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let indexes = TripleIndexes::new(buffer_pool);
        (temp_dir, indexes)
    }

    #[test]
    fn test_triple_index_insert_contains() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        assert!(!indexes.contains(&triple)?);

        indexes.insert(triple)?;

        assert!(indexes.contains(&triple)?);

        Ok(())
    }

    #[test]
    fn test_triple_index_delete() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triple = Triple::new(NodeId::new(10), NodeId::new(20), NodeId::new(30));

        indexes.insert(triple)?;
        assert!(indexes.contains(&triple)?);

        let deleted = indexes.delete(&triple)?;
        assert!(deleted);
        assert!(!indexes.contains(&triple)?);

        // Deleting again returns false
        let deleted_again = indexes.delete(&triple)?;
        assert!(!deleted_again);

        Ok(())
    }

    #[test]
    fn test_triple_index_multiple_triples() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triple2 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(4));
        let triple3 = Triple::new(NodeId::new(2), NodeId::new(3), NodeId::new(4));

        indexes.insert(triple1)?;
        indexes.insert(triple2)?;
        indexes.insert(triple3)?;

        assert!(indexes.contains(&triple1)?);
        assert!(indexes.contains(&triple2)?);
        assert!(indexes.contains(&triple3)?);

        // Delete one triple
        indexes.delete(&triple2)?;

        assert!(indexes.contains(&triple1)?);
        assert!(!indexes.contains(&triple2)?);
        assert!(indexes.contains(&triple3)?);

        Ok(())
    }

    #[test]
    fn test_all_three_indexes_updated() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triple = Triple::new(NodeId::new(100), NodeId::new(200), NodeId::new(300));

        indexes.insert(triple)?;

        // Check all three indexes contain the triple
        assert!(indexes.spo().contains(&triple.into())?);
        assert!(indexes.pos().contains(&triple.into())?);
        assert!(indexes.osp().contains(&triple.into())?);

        indexes.delete(&triple)?;

        // Check all three indexes no longer contain the triple
        assert!(!indexes.spo().contains(&triple.into())?);
        assert!(!indexes.pos().contains(&triple.into())?);
        assert!(!indexes.osp().contains(&triple.into())?);

        Ok(())
    }

    /// The streaming `scan` must yield exactly the same set as the materialized
    /// `query_pattern` for every bound-column combination.
    #[test]
    fn test_scan_matches_query_pattern() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triples = [
            Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3)),
            Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(4)),
            Triple::new(NodeId::new(1), NodeId::new(5), NodeId::new(3)),
            Triple::new(NodeId::new(2), NodeId::new(2), NodeId::new(3)),
            Triple::new(NodeId::new(2), NodeId::new(5), NodeId::new(9)),
        ];
        for t in &triples {
            indexes.insert(*t)?;
        }

        let s1 = Some(NodeId::new(1));
        let p2 = Some(NodeId::new(2));
        let o3 = Some(NodeId::new(3));
        let patterns = [
            (None, None, None),
            (s1, None, None),
            (None, p2, None),
            (None, None, o3),
            (s1, p2, None),
            (None, p2, o3),
            (s1, None, o3),
            (s1, p2, o3),
        ];

        for (s, p, o) in patterns {
            let mut expected = indexes.query_pattern(s, p, o)?;
            expected.sort_by_key(|t| (t.subject, t.predicate, t.object));

            let mut streamed: Vec<Triple> = indexes.scan(s, p, o)?.collect::<Result<_>>()?;
            streamed.sort_by_key(|t| (t.subject, t.predicate, t.object));

            assert_eq!(streamed, expected, "mismatch for pattern {s:?} {p:?} {o:?}");
        }

        Ok(())
    }
}
