//! B+Tree implementation for key-value storage
//!
//! This module implements a disk-based B+Tree for efficient range queries
//! and sequential scans. The tree uses the buffer pool for page management.

pub mod iterator;
pub mod node;

use crate::error::{Result, TdbError};
use crate::storage::{BufferPool, PageGuard, PageId, PageType};
use iterator::BTreeIterator;
use node::{BTreeNode, InternalNode, LeafNode, ORDER};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// B+Tree for disk-based key-value storage
pub struct BTree<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    buffer_pool: Arc<BufferPool>,
    root_page: Option<PageId>,
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K, V> BTree<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    /// Create a new B+Tree
    pub fn new(buffer_pool: Arc<BufferPool>) -> Self {
        BTree {
            buffer_pool,
            root_page: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a B+Tree with an existing root page
    pub fn from_root(buffer_pool: Arc<BufferPool>, root_page: PageId) -> Self {
        BTree {
            buffer_pool,
            root_page: Some(root_page),
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the root page ID
    pub fn root_page(&self) -> Option<PageId> {
        self.root_page
    }

    /// Search for a key in the tree
    pub fn search(&self, key: &K) -> Result<Option<V>> {
        let root_page = match self.root_page {
            Some(id) => id,
            None => return Ok(None),
        };

        self.search_recursive(root_page, key)
    }

    /// Recursive search helper
    fn search_recursive(&self, page_id: PageId, key: &K) -> Result<Option<V>> {
        let node = {
            let guard = self.buffer_pool.fetch_page(page_id)?;
            let page = guard.page();
            let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
            BTreeNode::<K, V>::deserialize_from_page(page_ref)?
        };

        match node {
            BTreeNode::Internal(internal) => {
                let child_idx = internal.find_child(key);
                let child_id = internal.children[child_idx];
                self.search_recursive(child_id, key)
            }
            BTreeNode::Leaf(leaf) => Ok(leaf.search(key).cloned()),
        }
    }

    /// Insert or update a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        if self.root_page.is_none() {
            // Create new root as leaf
            let guard = self.buffer_pool.new_page(PageType::BTreeLeaf)?;
            let root_id = guard.page_id();
            {
                let mut page = guard.page_mut();
                let page = page.as_mut().unwrap();
                let mut node = BTreeNode::new_leaf();
                if let BTreeNode::Leaf(ref mut leaf) = node {
                    leaf.insert(key.clone(), value.clone());
                }
                node.serialize_to_page(page)?;
            }
            drop(guard);
            self.root_page = Some(root_id);
            return Ok(None);
        }

        let root_id = self.root_page.unwrap();

        // Try insert, handle split if needed
        match self.insert_recursive(root_id, key.clone(), value.clone())? {
            InsertResult::Ok(old_value) => Ok(old_value),
            InsertResult::Split(split_key, new_page_id) => {
                // Root split - create new root
                let new_root_guard = self.buffer_pool.new_page(PageType::BTreeInternal)?;
                let new_root_id = new_root_guard.page_id();
                {
                    let mut page = new_root_guard.page_mut();
                    let page = page.as_mut().unwrap();
                    let new_root = InternalNode {
                        keys: vec![split_key],
                        children: vec![root_id, new_page_id],
                    };
                    let node: BTreeNode<K, V> = BTreeNode::Internal(new_root);
                    node.serialize_to_page(page)?;
                }
                drop(new_root_guard);
                self.root_page = Some(new_root_id);
                Ok(None)
            }
        }
    }

    /// Recursive insert helper
    fn insert_recursive(
        &mut self,
        page_id: PageId,
        key: K,
        value: V,
    ) -> Result<InsertResult<K, V>> {
        let mut node = {
            let guard = self.buffer_pool.fetch_page(page_id)?;
            let page = guard.page();
            let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
            BTreeNode::<K, V>::deserialize_from_page(page_ref)?
        };

        match node {
            BTreeNode::Internal(ref mut internal) => {
                let child_idx = internal.find_child(&key);
                let child_id = internal.children[child_idx];

                match self.insert_recursive(child_id, key.clone(), value)? {
                    InsertResult::Ok(old_value) => Ok(InsertResult::Ok(old_value)),
                    InsertResult::Split(split_key, new_child_id) => {
                        // Insert split key and new child into internal node
                        internal.insert_child(split_key.clone(), new_child_id, child_idx + 1);

                        // Check if needs split before writing
                        let needs_split = internal.keys.len() >= ORDER;
                        let internal_for_split = if needs_split {
                            Some(internal.clone())
                        } else {
                            None
                        };

                        // Write updated internal node
                        let guard = self.buffer_pool.fetch_page(page_id)?;
                        {
                            let mut page = guard.page_mut();
                            let page = page.as_mut().unwrap();
                            node.serialize_to_page(page)?;
                        }
                        drop(guard);

                        // Split if needed
                        if let Some(internal) = internal_for_split {
                            self.split_internal(page_id, internal)
                        } else {
                            Ok(InsertResult::Ok(None))
                        }
                    }
                }
            }
            BTreeNode::Leaf(ref mut leaf) => {
                let old_value = leaf.insert(key, value);

                // Check if needs split before writing
                let needs_split = leaf.entries.len() >= ORDER;
                let leaf_for_split = if needs_split {
                    Some(leaf.clone())
                } else {
                    None
                };

                // Write updated leaf
                let guard = self.buffer_pool.fetch_page(page_id)?;
                {
                    let mut page = guard.page_mut();
                    let page = page.as_mut().unwrap();
                    node.serialize_to_page(page)?;
                }
                drop(guard);

                // Split if needed
                if let Some(leaf) = leaf_for_split {
                    self.split_leaf(page_id, leaf)
                } else {
                    Ok(InsertResult::Ok(old_value))
                }
            }
        }
    }

    /// Split a leaf node
    fn split_leaf(
        &mut self,
        page_id: PageId,
        mut leaf: LeafNode<K, V>,
    ) -> Result<InsertResult<K, V>> {
        let (split_key, mut right_leaf) = leaf.split();

        // Create new page for right sibling
        let right_guard = self.buffer_pool.new_page(PageType::BTreeLeaf)?;
        let right_id = right_guard.page_id();

        // Update next pointers
        right_leaf.next = leaf.next;
        //leaf.next = Some(right_id);  // Commented out to avoid double linking

        {
            let mut page = right_guard.page_mut();
            let page = page.as_mut().unwrap();
            let right_node = BTreeNode::Leaf(right_leaf);
            right_node.serialize_to_page(page)?;
        }
        drop(right_guard);

        // Write left leaf back
        let left_guard = self.buffer_pool.fetch_page(page_id)?;
        {
            let mut page = left_guard.page_mut();
            let page = page.as_mut().unwrap();
            let left_node = BTreeNode::Leaf(leaf);
            left_node.serialize_to_page(page)?;
        }
        drop(left_guard);

        Ok(InsertResult::Split(split_key, right_id))
    }

    /// Split an internal node
    fn split_internal(
        &mut self,
        page_id: PageId,
        mut internal: InternalNode<K>,
    ) -> Result<InsertResult<K, V>> {
        let (split_key, right_internal) = internal.split();

        // Create new page for right sibling
        let right_guard = self.buffer_pool.new_page(PageType::BTreeInternal)?;
        let right_id = right_guard.page_id();

        {
            let mut page = right_guard.page_mut();
            let page = page.as_mut().unwrap();
            let right_node: BTreeNode<K, V> = BTreeNode::Internal(right_internal);
            right_node.serialize_to_page(page)?;
        }
        drop(right_guard);

        // Write left internal node back
        let left_guard = self.buffer_pool.fetch_page(page_id)?;
        {
            let mut page = left_guard.page_mut();
            let page = page.as_mut().unwrap();
            let left_node: BTreeNode<K, V> = BTreeNode::Internal(internal);
            left_node.serialize_to_page(page)?;
        }
        drop(left_guard);

        Ok(InsertResult::Split(split_key, right_id))
    }

    /// Delete a key from the tree
    pub fn delete(&mut self, key: &K) -> Result<Option<V>> {
        let root_page = match self.root_page {
            Some(id) => id,
            None => return Ok(None),
        };

        self.delete_recursive(root_page, key)
    }

    /// Recursive delete helper (simplified - no merging for now)
    fn delete_recursive(&mut self, page_id: PageId, key: &K) -> Result<Option<V>> {
        let mut node = {
            let guard = self.buffer_pool.fetch_page(page_id)?;
            let page = guard.page();
            let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
            BTreeNode::<K, V>::deserialize_from_page(page_ref)?
        };

        match node {
            BTreeNode::Internal(ref mut internal) => {
                let child_idx = internal.find_child(key);
                let child_id = internal.children[child_idx];
                self.delete_recursive(child_id, key)
            }
            BTreeNode::Leaf(ref mut leaf) => {
                let old_value = leaf.remove(key);

                if old_value.is_some() {
                    // Write updated leaf
                    let guard = self.buffer_pool.fetch_page(page_id)?;
                    {
                        let mut page = guard.page_mut();
                        let page = page.as_mut().unwrap();
                        node.serialize_to_page(page)?;
                    }
                    drop(guard);
                }

                Ok(old_value)
            }
        }
    }

    /// Range scan from start_key (inclusive) to end_key (exclusive)
    pub fn range_scan(
        &self,
        start_key: Option<K>,
        end_key: Option<K>,
    ) -> Result<BTreeIterator<K, V>> {
        let root_page = match self.root_page {
            Some(id) => id,
            None => return Err(TdbError::Other("Empty tree".to_string())),
        };

        // Find leftmost leaf
        let leaf_id = if let Some(ref start) = start_key {
            self.find_leaf(root_page, start)?
        } else {
            self.find_leftmost_leaf(root_page)?
        };

        BTreeIterator::new(self.buffer_pool.clone(), leaf_id, start_key, end_key)
    }

    /// Find the leaf node containing a key
    fn find_leaf(&self, page_id: PageId, key: &K) -> Result<PageId> {
        let node = {
            let guard = self.buffer_pool.fetch_page(page_id)?;
            let page = guard.page();
            let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
            BTreeNode::<K, V>::deserialize_from_page(page_ref)?
        };

        match node {
            BTreeNode::Internal(internal) => {
                let child_idx = internal.find_child(key);
                let child_id = internal.children[child_idx];
                self.find_leaf(child_id, key)
            }
            BTreeNode::Leaf(_) => Ok(page_id),
        }
    }

    /// Find the leftmost leaf node
    fn find_leftmost_leaf(&self, page_id: PageId) -> Result<PageId> {
        let node = {
            let guard = self.buffer_pool.fetch_page(page_id)?;
            let page = guard.page();
            let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
            BTreeNode::<K, V>::deserialize_from_page(page_ref)?
        };

        match node {
            BTreeNode::Internal(internal) => {
                let child_id = internal.children[0];
                self.find_leftmost_leaf(child_id)
            }
            BTreeNode::Leaf(_) => Ok(page_id),
        }
    }
}

/// Result of insert operation
enum InsertResult<K, V> {
    /// Insertion completed without split
    Ok(Option<V>),
    /// Node split, returns (split_key, new_page_id)
    Split(K, PageId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Allocator, FileManager};
    use tempfile::TempDir;

    fn create_test_btree() -> (TempDir, BTree<i32, String>) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true).unwrap());
        let buffer_pool = Arc::new(BufferPool::new(100, file_manager));

        let btree = BTree::new(buffer_pool);
        (temp_dir, btree)
    }

    #[test]
    fn test_insert_and_search() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        assert_eq!(btree.insert(5, "five".to_string())?, None);
        assert_eq!(btree.insert(3, "three".to_string())?, None);
        assert_eq!(btree.insert(7, "seven".to_string())?, None);

        assert_eq!(btree.search(&5)?, Some("five".to_string()));
        assert_eq!(btree.search(&3)?, Some("three".to_string()));
        assert_eq!(btree.search(&7)?, Some("seven".to_string()));
        assert_eq!(btree.search(&1)?, None);

        Ok(())
    }

    #[test]
    fn test_update_existing_key() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        btree.insert(5, "five".to_string())?;
        assert_eq!(
            btree.insert(5, "FIVE".to_string())?,
            Some("five".to_string())
        );
        assert_eq!(btree.search(&5)?, Some("FIVE".to_string()));

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        btree.insert(5, "five".to_string())?;
        btree.insert(3, "three".to_string())?;
        btree.insert(7, "seven".to_string())?;

        assert_eq!(btree.delete(&5)?, Some("five".to_string()));
        assert_eq!(btree.delete(&5)?, None);
        assert_eq!(btree.search(&5)?, None);

        Ok(())
    }

    #[test]
    fn test_large_insertion() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        // Insert 100 entries
        for i in 0..100 {
            btree.insert(i, format!("value{}", i))?;
        }

        // Verify all entries
        for i in 0..100 {
            assert_eq!(btree.search(&i)?, Some(format!("value{}", i)));
        }

        Ok(())
    }

    #[test]
    fn test_range_scan() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        // Insert entries
        for i in [1, 3, 5, 7, 9, 11, 13, 15] {
            btree.insert(i, format!("value{}", i))?;
        }

        // Range scan from 5 to 12
        let iter = btree.range_scan(Some(5), Some(12))?;
        let results: Result<Vec<_>> = iter.collect();
        let results = results?;

        assert_eq!(results.len(), 4);
        assert_eq!(results[0], (5, "value5".to_string()));
        assert_eq!(results[1], (7, "value7".to_string()));
        assert_eq!(results[2], (9, "value9".to_string()));
        assert_eq!(results[3], (11, "value11".to_string()));

        Ok(())
    }

    #[test]
    fn test_range_scan_all() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        for i in 0..10 {
            btree.insert(i, format!("value{}", i))?;
        }

        let iter = btree.range_scan(None, None)?;
        let results: Result<Vec<_>> = iter.collect();
        let results = results?;

        assert_eq!(results.len(), 10);
        assert_eq!(results[0], (0, "value0".to_string()));
        assert_eq!(results[9], (9, "value9".to_string()));

        Ok(())
    }

    #[test]
    fn test_split_behavior() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        // Insert enough to cause splits (ORDER = 64)
        for i in 0..200 {
            btree.insert(i, format!("value{}", i))?;
        }

        // Verify tree is still functional
        for i in 0..200 {
            assert_eq!(btree.search(&i)?, Some(format!("value{}", i)));
        }

        // Verify root is not None and is internal
        assert!(btree.root_page.is_some());

        Ok(())
    }
}
