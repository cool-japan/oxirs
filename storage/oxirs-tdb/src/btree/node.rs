//! B+Tree node implementations

use crate::error::{Result, TdbError};
use crate::storage::{Page, PageId, PageType, PAGE_USABLE_SIZE};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Debug;

/// Maximum number of keys in a node (order)
pub const ORDER: usize = 64;

/// B+Tree node type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BTreeNode<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone + Debug,
{
    /// Internal node with keys and child pointers
    Internal(InternalNode<K>),
    /// Leaf node with key-value pairs
    Leaf(LeafNode<K, V>),
}

/// Internal node containing keys and child page IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalNode<K>
where
    K: Ord + Clone + Debug,
{
    /// Keys for routing (keys\[i\] is the smallest key in children\[i+1\])
    pub keys: Vec<K>,
    /// Child page IDs (always has keys.len() + 1 children)
    pub children: Vec<PageId>,
}

/// Leaf node containing key-value pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafNode<K, V>
where
    K: Ord + Clone + Debug,
    V: Clone + Debug,
{
    /// Key-value pairs sorted by key
    pub entries: Vec<(K, V)>,
    /// Next leaf page ID for range scans (None if last leaf)
    pub next: Option<PageId>,
}

impl<K, V> BTreeNode<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    /// Create a new empty leaf node
    pub fn new_leaf() -> Self {
        BTreeNode::Leaf(LeafNode {
            entries: Vec::new(),
            next: None,
        })
    }

    /// Create a new empty internal node
    pub fn new_internal() -> Self {
        BTreeNode::Internal(InternalNode {
            keys: Vec::new(),
            children: Vec::new(),
        })
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        matches!(self, BTreeNode::Leaf(_))
    }

    /// Check if this is an internal node
    pub fn is_internal(&self) -> bool {
        matches!(self, BTreeNode::Internal(_))
    }

    /// Get the number of keys in this node
    pub fn len(&self) -> usize {
        match self {
            BTreeNode::Internal(node) => node.keys.len(),
            BTreeNode::Leaf(node) => node.entries.len(),
        }
    }

    /// Check if this node is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this node is full (needs splitting)
    pub fn is_full(&self) -> bool {
        self.len() >= ORDER
    }

    /// Check if this node is underfull (needs merging)
    pub fn is_underfull(&self) -> bool {
        self.len() < ORDER / 2
    }

    /// Serialize node to page
    pub fn serialize_to_page(&self, page: &mut Page) -> Result<()> {
        let data = bincode::serialize(self).map_err(|e| TdbError::Serialization(e.to_string()))?;

        if data.len() > PAGE_USABLE_SIZE {
            return Err(TdbError::NodeTooLarge {
                size: data.len(),
                max: PAGE_USABLE_SIZE,
            });
        }

        page.write_at(0, &data)?;
        Ok(())
    }

    /// Deserialize node from page
    pub fn deserialize_from_page(page: &Page) -> Result<Self> {
        let data = page.read_at(0, PAGE_USABLE_SIZE)?;
        bincode::deserialize(data).map_err(|e| TdbError::Deserialization(e.to_string()))
    }
}

impl<K> InternalNode<K>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    /// Find the child index for a given key
    pub fn find_child(&self, key: &K) -> usize {
        // Binary search for the appropriate child
        match self.keys.binary_search(key) {
            Ok(idx) => idx + 1, // Key found, go to right child
            Err(idx) => idx,    // Key not found, idx is insertion point
        }
    }

    /// Insert a new child pointer and separator key
    pub fn insert_child(&mut self, key: K, child_id: PageId, index: usize) {
        if index == 0 {
            // Inserting as leftmost child
            self.children.insert(0, child_id);
        } else {
            // Insert separator key and right child
            self.keys.insert(index - 1, key);
            self.children.insert(index, child_id);
        }
    }

    /// Split this internal node into two
    pub fn split(&mut self) -> (K, InternalNode<K>) {
        let mid = self.keys.len() / 2;

        // Split key (promoted to parent)
        let split_key = self.keys[mid].clone();

        // Create new right sibling
        let right_keys = self.keys.split_off(mid + 1);
        let right_children = self.children.split_off(mid + 1);

        // Remove the promoted key from left node
        self.keys.pop();

        let right = InternalNode {
            keys: right_keys,
            children: right_children,
        };

        (split_key, right)
    }
}

impl<K, V> LeafNode<K, V>
where
    K: Ord + Clone + Serialize + for<'de> Deserialize<'de> + Debug,
    V: Clone + Serialize + for<'de> Deserialize<'de> + Debug,
{
    /// Search for a key in this leaf node
    pub fn search(&self, key: &K) -> Option<&V> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Insert or update a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(idx) => {
                // Key exists, update value
                let old_value = self.entries[idx].1.clone();
                self.entries[idx].1 = value;
                Some(old_value)
            }
            Err(idx) => {
                // Key doesn't exist, insert new entry
                self.entries.insert(idx, (key, value));
                None
            }
        }
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.entries.binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(idx) => Some(self.entries.remove(idx).1),
            Err(_) => None,
        }
    }

    /// Split this leaf node into two
    pub fn split(&mut self) -> (K, LeafNode<K, V>) {
        let mid = self.entries.len() / 2;

        let right_entries = self.entries.split_off(mid);
        let split_key = right_entries[0].0.clone();

        let right = LeafNode {
            entries: right_entries,
            next: self.next,
        };

        (split_key, right)
    }

    /// Merge with a right sibling
    pub fn merge(&mut self, right: &mut LeafNode<K, V>) {
        self.entries.append(&mut right.entries);
        self.next = right.next;
    }

    /// Range scan starting from a key
    pub fn range_from(&self, start: &K) -> impl Iterator<Item = (&K, &V)> {
        let start_idx = match self.entries.binary_search_by(|(k, _)| k.cmp(start)) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        self.entries[start_idx..].iter().map(|(k, v)| (k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_nodes() {
        let leaf: BTreeNode<i32, String> = BTreeNode::new_leaf();
        assert!(leaf.is_leaf());
        assert!(!leaf.is_internal());
        assert_eq!(leaf.len(), 0);

        let internal: BTreeNode<i32, String> = BTreeNode::new_internal();
        assert!(internal.is_internal());
        assert!(!internal.is_leaf());
        assert_eq!(internal.len(), 0);
    }

    #[test]
    fn test_leaf_insert_search() {
        let mut node = BTreeNode::new_leaf();

        if let BTreeNode::Leaf(ref mut leaf) = node {
            assert_eq!(leaf.insert(5, "five".to_string()), None);
            assert_eq!(leaf.insert(3, "three".to_string()), None);
            assert_eq!(leaf.insert(7, "seven".to_string()), None);

            assert_eq!(leaf.search(&5), Some(&"five".to_string()));
            assert_eq!(leaf.search(&3), Some(&"three".to_string()));
            assert_eq!(leaf.search(&7), Some(&"seven".to_string()));
            assert_eq!(leaf.search(&1), None);

            // Update existing key
            assert_eq!(leaf.insert(5, "FIVE".to_string()), Some("five".to_string()));
            assert_eq!(leaf.search(&5), Some(&"FIVE".to_string()));
        }
    }

    #[test]
    fn test_leaf_remove() {
        let mut node = BTreeNode::new_leaf();

        if let BTreeNode::Leaf(ref mut leaf) = node {
            leaf.insert(5, "five".to_string());
            leaf.insert(3, "three".to_string());
            leaf.insert(7, "seven".to_string());

            assert_eq!(leaf.remove(&5), Some("five".to_string()));
            assert_eq!(leaf.remove(&5), None);
            assert_eq!(leaf.search(&5), None);
            assert_eq!(leaf.entries.len(), 2);
        }
    }

    #[test]
    fn test_leaf_split() {
        let mut node = BTreeNode::new_leaf();

        if let BTreeNode::Leaf(ref mut leaf) = node {
            for i in 0..10 {
                leaf.insert(i, format!("value{}", i));
            }

            let (split_key, right) = leaf.split();

            assert_eq!(split_key, 5);
            assert_eq!(leaf.entries.len(), 5);
            assert_eq!(right.entries.len(), 5);

            // Verify all keys are in correct nodes
            for i in 0..5 {
                assert!(leaf.search(&i).is_some());
                assert!(right.search(&i).is_none());
            }
            for i in 5..10 {
                assert!(leaf.search(&i).is_none());
                assert!(right.search(&i).is_some());
            }
        }
    }

    #[test]
    fn test_internal_find_child() {
        let mut node: BTreeNode<i32, String> = BTreeNode::new_internal();

        if let BTreeNode::Internal(ref mut internal) = node {
            // Setup: keys [10, 20, 30] with 4 children
            internal.keys = vec![10, 20, 30];
            internal.children = vec![1, 2, 3, 4];

            // Values < 10 should go to child 0
            assert_eq!(internal.find_child(&5), 0);
            assert_eq!(internal.find_child(&10), 1);
            assert_eq!(internal.find_child(&15), 1);
            assert_eq!(internal.find_child(&20), 2);
            assert_eq!(internal.find_child(&25), 2);
            assert_eq!(internal.find_child(&30), 3);
            assert_eq!(internal.find_child(&35), 3);
        }
    }

    #[test]
    fn test_internal_split() {
        let mut node: BTreeNode<i32, String> = BTreeNode::new_internal();

        if let BTreeNode::Internal(ref mut internal) = node {
            // Setup: 7 keys with 8 children
            internal.keys = vec![10, 20, 30, 40, 50, 60, 70];
            internal.children = vec![1, 2, 3, 4, 5, 6, 7, 8];

            let (split_key, right) = internal.split();

            // Split key should be middle key (40)
            assert_eq!(split_key, 40);

            // Left node: keys [10, 20, 30], children [1, 2, 3, 4]
            assert_eq!(internal.keys.len(), 3);
            assert_eq!(internal.children.len(), 4);

            // Right node: keys [50, 60, 70], children [5, 6, 7, 8]
            assert_eq!(right.keys.len(), 3);
            assert_eq!(right.children.len(), 4);
        }
    }

    #[test]
    fn test_node_fullness() {
        let mut node = BTreeNode::new_leaf();

        assert!(!node.is_full());
        assert!(node.is_underfull());

        if let BTreeNode::Leaf(ref mut leaf) = node {
            // Fill to half capacity
            for i in 0..(ORDER / 2) {
                leaf.insert(i, format!("value{}", i));
            }
        }

        assert!(!node.is_underfull());
        assert!(!node.is_full());

        if let BTreeNode::Leaf(ref mut leaf) = node {
            // Fill to capacity
            for i in (ORDER / 2)..ORDER {
                leaf.insert(i, format!("value{}", i));
            }
        }

        assert!(node.is_full());
    }

    #[test]
    fn test_leaf_range_from() {
        let mut node = BTreeNode::new_leaf();

        if let BTreeNode::Leaf(ref mut leaf) = node {
            for i in [2, 4, 6, 8, 10] {
                leaf.insert(i, format!("value{}", i));
            }

            // Range from existing key
            let range: Vec<_> = leaf.range_from(&6).collect();
            assert_eq!(range.len(), 3);
            assert_eq!(range[0], (&6, &"value6".to_string()));
            assert_eq!(range[1], (&8, &"value8".to_string()));
            assert_eq!(range[2], (&10, &"value10".to_string()));

            // Range from non-existing key (should start from next key)
            let range: Vec<_> = leaf.range_from(&5).collect();
            assert_eq!(range.len(), 3);
            assert_eq!(range[0], (&6, &"value6".to_string()));
        }
    }
}
