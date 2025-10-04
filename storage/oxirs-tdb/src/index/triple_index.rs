//! Triple index implementation using B+Tree

use super::triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
use crate::btree::BTree;
use crate::dictionary::NodeId;
use crate::error::Result;
use crate::storage::BufferPool;
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

        let triple = Triple::new(
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
        );

        assert!(!indexes.contains(&triple)?);

        indexes.insert(triple)?;

        assert!(indexes.contains(&triple)?);

        Ok(())
    }

    #[test]
    fn test_triple_index_delete() -> Result<()> {
        let (_temp_dir, mut indexes) = create_test_indexes();

        let triple = Triple::new(
            NodeId::new(10),
            NodeId::new(20),
            NodeId::new(30),
        );

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

        let triple = Triple::new(
            NodeId::new(100),
            NodeId::new(200),
            NodeId::new(300),
        );

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
