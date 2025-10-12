//! B+Tree range scan iterator

use super::node::{BTreeNode, LeafNode};
use crate::error::Result;
use crate::storage::{BufferPool, PageId};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// Range scan iterator for B+Tree
pub struct BTreeIterator<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    buffer_pool: Arc<BufferPool>,
    current_leaf: Option<PageId>,
    current_entries: Vec<(K, V)>,
    current_index: usize,
    end_key: Option<K>,
}

impl<K, V> BTreeIterator<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    /// Create a new iterator starting from a leaf node
    pub fn new(
        buffer_pool: Arc<BufferPool>,
        leaf_id: PageId,
        start_key: Option<K>,
        end_key: Option<K>,
    ) -> Result<Self> {
        let mut iterator = BTreeIterator {
            buffer_pool,
            current_leaf: Some(leaf_id),
            current_entries: Vec::new(),
            current_index: 0,
            end_key,
        };

        // Load first leaf
        iterator.load_leaf(leaf_id, start_key.as_ref())?;

        Ok(iterator)
    }

    /// Load a leaf node and filter entries
    fn load_leaf(&mut self, leaf_id: PageId, start_key: Option<&K>) -> Result<()> {
        let guard = self.buffer_pool.fetch_page(leaf_id)?;
        let page = guard.page();
        let page_ref = page
            .as_ref()
            .ok_or(crate::error::TdbError::PageNotFound(leaf_id))?;

        let node = BTreeNode::<K, V>::deserialize_from_page(page_ref)?;

        if let BTreeNode::Leaf(leaf) = node {
            // Clone all entries (inefficient but simple for now)
            self.current_entries = leaf.entries.clone();

            // Find starting position
            if let Some(start) = start_key {
                match self.current_entries.binary_search_by(|(k, _)| k.cmp(start)) {
                    Ok(idx) => self.current_index = idx,
                    Err(idx) => self.current_index = idx,
                }
            } else {
                self.current_index = 0;
            }

            // Update next leaf pointer
            self.current_leaf = leaf.next;
        } else {
            return Err(crate::error::TdbError::Other(
                "Expected leaf node in iterator".to_string(),
            ));
        }

        Ok(())
    }

    /// Advance to next entry
    fn advance(&mut self) -> Result<bool> {
        self.current_index += 1;

        // Check if we've exhausted current leaf
        if self.current_index >= self.current_entries.len() {
            // Try to move to next leaf
            if let Some(next_leaf) = self.current_leaf {
                self.load_leaf(next_leaf, None)?;
                return Ok(!self.current_entries.is_empty());
            }
            return Ok(false);
        }

        Ok(true)
    }
}

impl<K, V> Iterator for BTreeIterator<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    type Item = Result<(K, V)>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have a current entry
        if self.current_index >= self.current_entries.len() {
            return None;
        }

        let entry = &self.current_entries[self.current_index];

        // Check end bound
        if let Some(ref end) = self.end_key {
            if &entry.0 >= end {
                return None;
            }
        }

        let result = (entry.0.clone(), entry.1.clone());

        // Advance to next entry
        match self.advance() {
            Ok(has_more) => {
                if !has_more && self.current_index >= self.current_entries.len() {
                    // No more entries
                }
            }
            Err(e) => return Some(Err(e)),
        }

        Some(Ok(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Allocator, FileManager, PageType, PAGE_SIZE};
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_iterator_single_leaf() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true)?);
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));

        // Create a leaf node with test data
        let leaf = LeafNode {
            entries: vec![
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string()),
                (4, "four".to_string()),
                (5, "five".to_string()),
            ],
            next: None,
        };

        // Write to page
        let guard = buffer_pool.new_page(PageType::BTreeLeaf)?;
        let leaf_id = guard.page_id();
        {
            let mut page = guard.page_mut();
            let page = page.as_mut().unwrap();
            let node = BTreeNode::Leaf(leaf);
            node.serialize_to_page(page)?;
        }
        drop(guard);

        // Iterate all entries
        let iter = BTreeIterator::new(buffer_pool.clone(), leaf_id, None, None)?;
        let results: Result<Vec<_>> = iter.collect();
        let results = results?;

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], (1, "one".to_string()));
        assert_eq!(results[4], (5, "five".to_string()));

        Ok(())
    }

    #[test]
    fn test_iterator_with_range() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let file_manager = Arc::new(FileManager::open(&db_path, true)?);
        let buffer_pool = Arc::new(BufferPool::new(10, file_manager));

        // Create a leaf node
        let leaf = LeafNode {
            entries: vec![
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string()),
                (4, "four".to_string()),
                (5, "five".to_string()),
            ],
            next: None,
        };

        let guard = buffer_pool.new_page(PageType::BTreeLeaf)?;
        let leaf_id = guard.page_id();
        {
            let mut page = guard.page_mut();
            let page = page.as_mut().unwrap();
            let node = BTreeNode::Leaf(leaf);
            node.serialize_to_page(page)?;
        }
        drop(guard);

        // Range scan from 2 to 4
        let iter = BTreeIterator::new(buffer_pool.clone(), leaf_id, Some(2), Some(4))?;
        let results: Result<Vec<_>> = iter.collect();
        let results = results?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (2, "two".to_string()));
        assert_eq!(results[1], (3, "three".to_string()));

        Ok(())
    }
}
