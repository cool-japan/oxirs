//! B+Tree implementation for key-value storage
//!
//! This module implements a disk-based B+Tree for efficient range queries
//! and sequential scans. The tree uses the buffer pool for page management.

pub mod iterator;
pub mod node;

use crate::error::{Result, TdbError};
use crate::storage::{BufferPool, PageId, PageType};
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
            let page = guard.page_checked()?;
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
                let mut page = guard.page_mut_checked()?;
                let page = page.as_mut().ok_or(TdbError::PageNotFound(root_id))?;
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

        let root_id = self.root_page.ok_or_else(|| {
            TdbError::Other("B+Tree root page vanished after existence check".to_string())
        })?;

        // Try insert, handle split if needed
        match self.insert_recursive(root_id, key.clone(), value.clone())? {
            InsertResult::Ok(old_value) => Ok(old_value),
            InsertResult::Split(split_key, new_page_id) => {
                // Root split - create new root
                let new_root_guard = self.buffer_pool.new_page(PageType::BTreeInternal)?;
                let new_root_id = new_root_guard.page_id();
                {
                    let mut page = new_root_guard.page_mut_checked()?;
                    let page = page.as_mut().ok_or(TdbError::PageNotFound(new_root_id))?;
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
            let page = guard.page_checked()?;
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
                            let mut page = guard.page_mut_checked()?;
                            let page = page.as_mut().ok_or(TdbError::PageNotFound(page_id))?;
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
                    let mut page = guard.page_mut_checked()?;
                    let page = page.as_mut().ok_or(TdbError::PageNotFound(page_id))?;
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

        // Maintain the leaf linked list: the new right sibling takes over the
        // left leaf's old successor, and the left leaf now points at the right
        // sibling. Without this link, range scans (and full-index scans used
        // for reopen/bloom-rebuild) silently skip every split-off right leaf.
        right_leaf.next = leaf.next;
        leaf.next = Some(right_id);

        {
            let mut page = right_guard.page_mut_checked()?;
            let page = page.as_mut().ok_or(TdbError::PageNotFound(right_id))?;
            let right_node = BTreeNode::Leaf(right_leaf);
            right_node.serialize_to_page(page)?;
        }
        drop(right_guard);

        // Write left leaf back
        let left_guard = self.buffer_pool.fetch_page(page_id)?;
        {
            let mut page = left_guard.page_mut_checked()?;
            let page = page.as_mut().ok_or(TdbError::PageNotFound(page_id))?;
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
            let mut page = right_guard.page_mut_checked()?;
            let page = page.as_mut().ok_or(TdbError::PageNotFound(right_id))?;
            let right_node: BTreeNode<K, V> = BTreeNode::Internal(right_internal);
            right_node.serialize_to_page(page)?;
        }
        drop(right_guard);

        // Write left internal node back
        let left_guard = self.buffer_pool.fetch_page(page_id)?;
        {
            let mut page = left_guard.page_mut_checked()?;
            let page = page.as_mut().ok_or(TdbError::PageNotFound(page_id))?;
            let left_node: BTreeNode<K, V> = BTreeNode::Internal(internal);
            left_node.serialize_to_page(page)?;
        }
        drop(left_guard);

        Ok(InsertResult::Split(split_key, right_id))
    }

    /// Load and deserialize the B+Tree node stored at `page_id`.
    fn load_node(&self, page_id: PageId) -> Result<BTreeNode<K, V>> {
        let guard = self.buffer_pool.fetch_page(page_id)?;
        let page = guard.page_checked()?;
        let page_ref = page.as_ref().ok_or(TdbError::PageNotFound(page_id))?;
        BTreeNode::<K, V>::deserialize_from_page(page_ref)
    }

    /// Serialize `node` back into its `page_id` (marks the page dirty).
    fn write_node(&self, page_id: PageId, node: &BTreeNode<K, V>) -> Result<()> {
        let guard = self.buffer_pool.fetch_page(page_id)?;
        {
            let mut page = guard.page_mut_checked()?;
            let page = page.as_mut().ok_or(TdbError::PageNotFound(page_id))?;
            node.serialize_to_page(page)?;
        }
        Ok(())
    }

    /// Return an emptied page to the on-disk free list, safely.
    ///
    /// [`FileManager::free_page`](crate::storage::FileManager::free_page) writes a
    /// free-list node directly through the file manager, bypassing the buffer
    /// pool. Any cached copy of the page must therefore be evicted first, or a
    /// later `flush_all()` would write the stale page back over the free-list
    /// node and corrupt the free list. `discard_page` closes that hazard.
    fn free_page_cache_safe(&self, page_id: PageId) -> Result<()> {
        if !self.buffer_pool.discard_page(page_id) {
            return Err(TdbError::Other(format!(
                "cannot free B+Tree page {page_id}: still pinned in the buffer pool"
            )));
        }
        self.buffer_pool.file_manager().free_page(page_id)
    }

    /// Delete a key from the tree, reclaiming any page that becomes empty.
    ///
    /// Unlike a leave-it-behind delete, an emptied leaf (and any internal node
    /// that loses its last child as a result) is unlinked from its parent and
    /// returned to the allocator via [`Self::free_page_cache_safe`], so
    /// delete/update-heavy workloads no longer grow the data file without bound.
    /// The singly-linked leaf list used by range scans is repaired so no live
    /// leaf's `next` is left pointing at a freed page.
    pub fn delete(&mut self, key: &K) -> Result<Option<V>> {
        let root_page = match self.root_page {
            Some(id) => id,
            None => return Ok(None),
        };

        let outcome = self.delete_recursive(root_page, key, true)?;

        // If a leaf page was reclaimed, repair its predecessor's `next` pointer
        // so range scans never traverse into the freed (and possibly reused)
        // page. Done after all structural changes, and no page is allocated in
        // between, so the freed page id cannot have been recycled yet.
        if let Some((freed_leaf, freed_next)) = outcome.freed_leaf {
            self.fix_leaf_predecessor(freed_leaf, freed_next)?;
        }

        Ok(outcome.value)
    }

    /// Recursive delete helper with empty-page reclamation.
    ///
    /// Returns the removed value plus two structural signals: `emptied` tells the
    /// caller (the parent) that this node became empty and its page should be
    /// unlinked and freed; `freed_leaf` carries the `(page_id, next)` of a leaf
    /// that was reclaimed so the top-level [`Self::delete`] can repair the leaf
    /// linked list exactly once.
    fn delete_recursive(
        &mut self,
        page_id: PageId,
        key: &K,
        is_root: bool,
    ) -> Result<DeleteOutcome<V>> {
        let mut node = self.load_node(page_id)?;

        match node {
            BTreeNode::Leaf(ref mut leaf) => {
                let old_value = leaf.remove(key);
                if old_value.is_none() {
                    return Ok(DeleteOutcome::not_found());
                }

                if leaf.entries.is_empty() && !is_root {
                    // Signal the parent to unlink and free this leaf. Do NOT
                    // rewrite the soon-to-be-freed page; carry its `next` so the
                    // leaf list can be repaired.
                    return Ok(DeleteOutcome {
                        value: old_value,
                        emptied: true,
                        freed_leaf: Some((page_id, leaf.next)),
                    });
                }

                // Persist the modified leaf (non-empty, or the empty root leaf,
                // which is retained as an empty tree root).
                self.write_node(page_id, &node)?;
                Ok(DeleteOutcome::removed(old_value))
            }
            BTreeNode::Internal(ref mut internal) => {
                let child_idx = internal.find_child(key);
                let child_id = *internal.children.get(child_idx).ok_or_else(|| {
                    TdbError::Other("B+Tree internal child index out of range".to_string())
                })?;

                let child = self.delete_recursive(child_id, key, false)?;
                if !child.emptied {
                    return Ok(DeleteOutcome {
                        value: child.value,
                        emptied: false,
                        freed_leaf: child.freed_leaf,
                    });
                }

                // The child emptied out: reclaim its page and drop the pointer +
                // one separator key (keeping keys.len() == children.len() - 1).
                self.free_page_cache_safe(child_id)?;
                internal.children.remove(child_idx);
                if child_idx > 0 {
                    internal.keys.remove(child_idx - 1);
                } else if !internal.keys.is_empty() {
                    internal.keys.remove(0);
                }

                if is_root {
                    if internal.children.len() == 1 {
                        // Collapse a root with a single child into that child.
                        let new_root = *internal.children.first().ok_or_else(|| {
                            TdbError::Other("B+Tree root collapse: missing child".to_string())
                        })?;
                        self.free_page_cache_safe(page_id)?;
                        self.root_page = Some(new_root);
                        return Ok(DeleteOutcome {
                            value: child.value,
                            emptied: false,
                            freed_leaf: child.freed_leaf,
                        });
                    }
                    if internal.children.is_empty() {
                        // The whole tree is now empty.
                        self.free_page_cache_safe(page_id)?;
                        self.root_page = None;
                        return Ok(DeleteOutcome {
                            value: child.value,
                            emptied: false,
                            freed_leaf: child.freed_leaf,
                        });
                    }
                    self.write_node(page_id, &node)?;
                    return Ok(DeleteOutcome {
                        value: child.value,
                        emptied: false,
                        freed_leaf: child.freed_leaf,
                    });
                }

                // Non-root internal node: if it lost its last child, signal the
                // grandparent to free it too; otherwise persist it.
                if internal.children.is_empty() {
                    return Ok(DeleteOutcome {
                        value: child.value,
                        emptied: true,
                        freed_leaf: child.freed_leaf,
                    });
                }
                self.write_node(page_id, &node)?;
                Ok(DeleteOutcome {
                    value: child.value,
                    emptied: false,
                    freed_leaf: child.freed_leaf,
                })
            }
        }
    }

    /// Repair the singly-linked leaf list after `freed_leaf` was reclaimed:
    /// find the live leaf whose `next` points at it and re-point it at
    /// `freed_next`. If `freed_leaf` was the leftmost leaf (nothing points at
    /// it) this is a no-op.
    fn fix_leaf_predecessor(&self, freed_leaf: PageId, freed_next: Option<PageId>) -> Result<()> {
        let root = match self.root_page {
            Some(r) => r,
            None => return Ok(()), // tree empty; nothing links to the freed leaf
        };

        let mut current = self.find_leftmost_leaf(root)?;
        // Bound the walk by the page count so a corrupt cycle cannot loop forever.
        let max_iters = self.buffer_pool.file_manager().num_pages() as usize + 1;
        for _ in 0..max_iters {
            let node = self.load_node(current)?;
            let leaf = match node {
                BTreeNode::Leaf(leaf) => leaf,
                BTreeNode::Internal(_) => {
                    return Err(TdbError::Other(
                        "leaf-list repair reached an internal node".to_string(),
                    ));
                }
            };

            if leaf.next == Some(freed_leaf) {
                let mut fixed = leaf;
                fixed.next = freed_next;
                self.write_node(current, &BTreeNode::Leaf(fixed))?;
                return Ok(());
            }

            match leaf.next {
                Some(next) => current = next,
                None => return Ok(()), // freed leaf had no predecessor (leftmost)
            }
        }

        Ok(())
    }

    /// Range scan from start_key (inclusive) to end_key (exclusive).
    ///
    /// Returns an empty iterator when the tree has no entries yet.
    pub fn range_scan(
        &self,
        start_key: Option<K>,
        end_key: Option<K>,
    ) -> Result<BTreeIterator<K, V>> {
        let root_page = match self.root_page {
            Some(id) => id,
            // Empty tree: return an iterator that yields nothing
            None => return Ok(BTreeIterator::empty(self.buffer_pool.clone())),
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
            let page = guard.page_checked()?;
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
            let page = guard.page_checked()?;
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

/// Result of a recursive delete, threaded back up so parents can reclaim emptied
/// child pages and the top-level delete can repair the leaf linked list.
struct DeleteOutcome<V> {
    /// The removed value, if the key was present.
    value: Option<V>,
    /// True when the visited node became empty and its page was NOT rewritten;
    /// the parent must unlink the child pointer and free the page.
    emptied: bool,
    /// `(page_id, next)` of a leaf that was reclaimed during this delete, so the
    /// top-level [`BTree::delete`] can re-point that leaf's predecessor. At most
    /// one leaf is freed per single-key delete.
    freed_leaf: Option<(PageId, Option<PageId>)>,
}

impl<V> DeleteOutcome<V> {
    /// The key was not present; nothing changed.
    fn not_found() -> Self {
        Self {
            value: None,
            emptied: false,
            freed_leaf: None,
        }
    }

    /// The key was removed and the node persisted in place (no reclamation).
    fn removed(value: Option<V>) -> Self {
        Self {
            value,
            emptied: false,
            freed_leaf: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::FileManager;
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
    fn regression_delete_reclaims_empty_pages() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        // Build a multi-level tree (ORDER = 64) so there are many leaves to free.
        let n = 500;
        for i in 0..n {
            btree.insert(i, format!("v{i}"))?;
        }
        assert!(btree.root_page.is_some());
        assert!(
            btree.buffer_pool.file_manager().free_list_head().is_none(),
            "no free pages before any delete"
        );

        // Delete every key.
        for i in 0..n {
            assert_eq!(btree.delete(&i)?, Some(format!("v{i}")));
        }
        for i in 0..n {
            assert_eq!(btree.search(&i)?, None);
        }

        // Emptied leaf/internal pages must have been returned to the free list:
        // free_page() is now exercised in production, so the file no longer grows
        // monotonically under delete-heavy workloads.
        assert!(
            btree.buffer_pool.file_manager().free_list_head().is_some(),
            "emptied pages must be reclaimed onto the free list"
        );

        // Reinserting reuses freed pages instead of growing the file unbounded.
        let pages_after_delete = btree.buffer_pool.file_manager().num_pages();
        for i in 0..n {
            btree.insert(i, format!("w{i}"))?;
        }
        for i in 0..n {
            assert_eq!(btree.search(&i)?, Some(format!("w{i}")));
        }
        let pages_after_reinsert = btree.buffer_pool.file_manager().num_pages();
        assert!(
            pages_after_reinsert <= pages_after_delete + 4,
            "reinsertion should reuse freed pages ({pages_after_delete} -> {pages_after_reinsert})"
        );

        Ok(())
    }

    #[test]
    fn regression_range_scan_intact_after_reclaiming_deletes() -> Result<()> {
        let (_temp_dir, mut btree) = create_test_btree();

        for i in 0..300 {
            btree.insert(i, format!("v{i}"))?;
        }

        // Delete a contiguous block large enough to empty whole leaves, which
        // reclaims their pages and must repair the leaf linked list.
        for i in 100..200 {
            assert!(btree.delete(&i)?.is_some());
        }

        // A full scan must yield exactly the survivors, in order, never
        // traversing into a freed (and possibly reused) leaf page.
        let got: Vec<_> = btree.range_scan(None, None)?.collect::<Result<Vec<_>>>()?;
        let expected: Vec<_> = (0..300)
            .filter(|i| !(100..200).contains(i))
            .map(|i| (i, format!("v{i}")))
            .collect();
        assert_eq!(got, expected);

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
