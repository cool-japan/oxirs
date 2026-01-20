//! Triple index implementation using B+Tree

use super::triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
use crate::btree::BTree;
use crate::dictionary::NodeId;
use crate::error::Result;
use crate::storage::BufferPool;
use oxicode::Decode;
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

    /// Insert a triple into the index
    pub fn insert(&mut self, key: K) -> Result<()> {
        self.btree.insert(key, EmptyValue)?;
        Ok(())
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

    /// Insert a triple into all three indexes
    pub fn insert(&mut self, triple: Triple) -> Result<()> {
        self.spo.insert(triple.into())?;
        self.pos.insert(triple.into())?;
        self.osp.insert(triple.into())?;
        Ok(())
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
}
