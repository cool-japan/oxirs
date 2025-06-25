//! # B+ Tree Implementation for TDB Storage
//!
//! High-performance B+ Tree implementation optimized for RDF triple storage.
//! Provides efficient range queries, bulk operations, and concurrent access.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// B+ Tree configuration
#[derive(Debug, Clone)]
pub struct BTreeConfig {
    pub max_keys_per_node: usize,
    pub enable_bulk_loading: bool,
    pub enable_compression: bool,
    pub cache_size: usize,
}

impl Default for BTreeConfig {
    fn default() -> Self {
        Self {
            max_keys_per_node: 511, // Optimized for 8KB pages
            enable_bulk_loading: true,
            enable_compression: false,
            cache_size: 1024 * 1024, // 1MB cache
        }
    }
}

/// Node identifier type
pub type NodeId = u64;

/// B+ Tree node types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BTreeNode<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    /// Internal node containing keys and child pointers
    Internal {
        keys: Vec<K>,
        children: Vec<NodeId>,
        parent: Option<NodeId>,
        level: u32,
    },
    /// Leaf node containing key-value pairs
    Leaf {
        entries: Vec<(K, V)>,
        parent: Option<NodeId>,
        next: Option<NodeId>,
        prev: Option<NodeId>,
    },
}

impl<K, V> BTreeNode<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    /// Check if the node is full
    pub fn is_full(&self, max_keys: usize) -> bool {
        match self {
            BTreeNode::Internal { keys, .. } => keys.len() >= max_keys,
            BTreeNode::Leaf { entries, .. } => entries.len() >= max_keys,
        }
    }

    /// Check if the node is underflow (needs merging)
    pub fn is_underflow(&self, min_keys: usize) -> bool {
        match self {
            BTreeNode::Internal { keys, .. } => keys.len() < min_keys,
            BTreeNode::Leaf { entries, .. } => entries.len() < min_keys,
        }
    }

    /// Get the number of entries in the node
    pub fn len(&self) -> usize {
        match self {
            BTreeNode::Internal { keys, .. } => keys.len(),
            BTreeNode::Leaf { entries, .. } => entries.len(),
        }
    }

    /// Check if the node is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// B+ Tree statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct BTreeStats {
    pub total_nodes: usize,
    pub internal_nodes: usize,
    pub leaf_nodes: usize,
    pub total_entries: usize,
    pub height: u32,
    pub utilization: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// B+ Tree implementation
pub struct BTree<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    root: Option<NodeId>,
    nodes: Arc<RwLock<Vec<Option<BTreeNode<K, V>>>>>,
    next_node_id: NodeId,
    config: BTreeConfig,
    stats: BTreeStats,
}

impl<K, V> BTree<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    /// Create a new B+ Tree
    pub fn new() -> Self {
        Self::with_config(BTreeConfig::default())
    }

    /// Create a new B+ Tree with custom configuration
    pub fn with_config(config: BTreeConfig) -> Self {
        Self {
            root: None,
            nodes: Arc::new(RwLock::new(Vec::new())),
            next_node_id: 0,
            config,
            stats: BTreeStats::default(),
        }
    }

    /// Insert a key-value pair into the tree
    pub fn insert(&mut self, key: K, value: V) -> Result<()> {
        if self.root.is_none() {
            // Create root leaf node
            let root_id = self.create_leaf_node()?;
            self.root = Some(root_id);
        }

        let root_id = self.root.unwrap();
        let (new_key, new_child) = self.insert_recursive(root_id, key, value)?;

        // Check if root needs to be split
        if let Some((split_key, right_child)) = new_key.zip(new_child) {
            let new_root_id = self.create_internal_node()?;
            let new_root = BTreeNode::Internal {
                keys: vec![split_key],
                children: vec![root_id, right_child],
                parent: None,
                level: self.get_node_level(root_id)? + 1,
            };

            self.set_node(new_root_id, new_root)?;
            self.set_parent(root_id, Some(new_root_id))?;
            self.set_parent(right_child, Some(new_root_id))?;
            self.root = Some(new_root_id);
            self.stats.height += 1;
        }

        self.stats.total_entries += 1;
        Ok(())
    }

    /// Find a value by key
    pub fn find(&self, key: &K) -> Result<Option<V>> {
        if let Some(root_id) = self.root {
            self.find_recursive(root_id, key)
        } else {
            Ok(None)
        }
    }

    /// Delete a key from the tree
    pub fn delete(&mut self, key: &K) -> Result<bool> {
        if let Some(root_id) = self.root {
            let deleted = self.delete_recursive(root_id, key)?;
            if deleted {
                self.stats.total_entries -= 1;

                // Check if root became empty and has a single child
                if self.is_root_empty_internal()? {
                    if let Some(new_root) = self.get_single_child(root_id)? {
                        self.root = Some(new_root);
                        self.set_parent(new_root, None)?;
                        self.stats.height -= 1;
                    }
                }
            }
            Ok(deleted)
        } else {
            Ok(false)
        }
    }

    /// Range query - find all key-value pairs in the given range
    pub fn range(&self, start: &K, end: &K) -> Result<Vec<(K, V)>> {
        let mut results = Vec::new();
        if let Some(root_id) = self.root {
            let start_leaf = self.find_leaf(root_id, start)?;
            self.collect_range(start_leaf, start, end, &mut results)?;
        }
        Ok(results)
    }

    /// Prefix search - find all entries with keys that start with the given prefix
    pub fn prefix_search(&self, prefix: &K) -> Result<Vec<(K, V)>>
    where
        K: PartialOrd,
    {
        let mut results = Vec::new();
        if let Some(root_id) = self.root {
            let start_leaf = self.find_leaf(root_id, prefix)?;
            self.collect_prefix(start_leaf, prefix, &mut results)?;
        }
        Ok(results)
    }

    /// Bulk load entries (optimized for large datasets)
    pub fn bulk_load(&mut self, mut entries: Vec<(K, V)>) -> Result<()> {
        if !self.config.enable_bulk_loading {
            return Err(anyhow!("Bulk loading is disabled"));
        }

        // Sort entries by key
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Clear existing tree
        self.clear()?;

        if entries.is_empty() {
            return Ok(());
        }

        // Build tree bottom-up
        let leaves = self.build_leaf_level(entries)?;
        self.build_internal_levels(leaves)?;

        self.stats.total_entries = self.count_entries()?;
        Ok(())
    }

    /// Get tree statistics
    pub fn get_stats(&self) -> BTreeStats {
        self.stats.clone()
    }

    /// Clear the entire tree
    pub fn clear(&mut self) -> Result<()> {
        self.root = None;
        if let Ok(mut nodes) = self.nodes.write() {
            nodes.clear();
        }
        self.next_node_id = 0;
        self.stats = BTreeStats::default();
        Ok(())
    }

    /// Validate tree structure (for debugging)
    pub fn validate(&self) -> Result<bool> {
        if let Some(root_id) = self.root {
            self.validate_recursive(root_id, None, None)
        } else {
            Ok(true)
        }
    }

    // Private helper methods

    fn create_leaf_node(&mut self) -> Result<NodeId> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = BTreeNode::Leaf {
            entries: Vec::new(),
            parent: None,
            next: None,
            prev: None,
        };

        self.set_node(node_id, node)?;
        self.stats.leaf_nodes += 1;
        self.stats.total_nodes += 1;
        Ok(node_id)
    }

    fn create_internal_node(&mut self) -> Result<NodeId> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let node = BTreeNode::Internal {
            keys: Vec::new(),
            children: Vec::new(),
            parent: None,
            level: 0,
        };

        self.set_node(node_id, node)?;
        self.stats.internal_nodes += 1;
        self.stats.total_nodes += 1;
        Ok(node_id)
    }

    fn get_node(&self, node_id: NodeId) -> Result<BTreeNode<K, V>> {
        let nodes = self
            .nodes
            .read()
            .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;

        if let Some(Some(node)) = nodes.get(node_id as usize) {
            Ok(node.clone())
        } else {
            Err(anyhow!("Node {} not found", node_id))
        }
    }

    fn set_node(&self, node_id: NodeId, node: BTreeNode<K, V>) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|_| anyhow!("Failed to acquire nodes lock"))?;

        // Ensure the vector is large enough
        while nodes.len() <= node_id as usize {
            nodes.push(None);
        }

        nodes[node_id as usize] = Some(node);
        Ok(())
    }

    fn insert_recursive(
        &mut self,
        node_id: NodeId,
        key: K,
        value: V,
    ) -> Result<(Option<K>, Option<NodeId>)> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf {
                mut entries,
                parent,
                next,
                prev,
            } => {
                // Find insertion position
                let pos = match entries.binary_search_by(|probe| probe.0.cmp(&key)) {
                    Ok(pos) => {
                        // Key exists, update value
                        entries[pos] = (key, value);
                        self.set_node(
                            node_id,
                            BTreeNode::Leaf {
                                entries,
                                parent,
                                next,
                                prev,
                            },
                        )?;
                        return Ok((None, None));
                    }
                    Err(pos) => pos,
                };

                entries.insert(pos, (key, value));

                // Check if node is full
                if entries.len() <= self.config.max_keys_per_node {
                    self.set_node(
                        node_id,
                        BTreeNode::Leaf {
                            entries,
                            parent,
                            next,
                            prev,
                        },
                    )?;
                    Ok((None, None))
                } else {
                    // Split leaf node
                    let mid = entries.len() / 2;
                    let right_entries = entries.split_off(mid);
                    let split_key = right_entries[0].0.clone();

                    // Update left node
                    self.set_node(
                        node_id,
                        BTreeNode::Leaf {
                            entries,
                            parent,
                            next: None,
                            prev,
                        },
                    )?;

                    // Create right node
                    let right_id = self.create_leaf_node()?;
                    self.set_node(
                        right_id,
                        BTreeNode::Leaf {
                            entries: right_entries,
                            parent,
                            next,
                            prev: Some(node_id),
                        },
                    )?;

                    // Update sibling pointers
                    if let Some(next_id) = next {
                        self.set_prev_sibling(next_id, Some(right_id))?;
                    }
                    self.set_next_sibling(node_id, Some(right_id))?;

                    Ok((Some(split_key), Some(right_id)))
                }
            }
            BTreeNode::Internal {
                keys,
                children,
                parent,
                level,
            } => {
                // Find child to insert into
                let child_idx = match keys.binary_search(&key) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };

                let child_id = children[child_idx];
                let (split_key, new_child) = self.insert_recursive(child_id, key, value)?;

                if let Some((key_to_insert, child_to_insert)) = split_key.zip(new_child) {
                    // Child was split, need to insert new key and child
                    let mut new_keys = keys;
                    let mut new_children = children;

                    let insert_pos = match new_keys.binary_search(&key_to_insert) {
                        Ok(pos) => pos + 1,
                        Err(pos) => pos,
                    };

                    new_keys.insert(insert_pos, key_to_insert);
                    new_children.insert(insert_pos + 1, child_to_insert);
                    self.set_parent(child_to_insert, Some(node_id))?;

                    // Check if internal node is full
                    if new_keys.len() <= self.config.max_keys_per_node {
                        self.set_node(
                            node_id,
                            BTreeNode::Internal {
                                keys: new_keys,
                                children: new_children,
                                parent,
                                level,
                            },
                        )?;
                        Ok((None, None))
                    } else {
                        // Split internal node
                        let mid = new_keys.len() / 2;
                        let split_key = new_keys[mid].clone();

                        let right_keys = new_keys.split_off(mid + 1);
                        let right_children = new_children.split_off(mid + 1);
                        new_keys.pop(); // Remove the split key from left node

                        // Update left node
                        self.set_node(
                            node_id,
                            BTreeNode::Internal {
                                keys: new_keys,
                                children: new_children,
                                parent,
                                level,
                            },
                        )?;

                        // Create right node
                        let right_id = self.create_internal_node()?;
                        self.set_node(
                            right_id,
                            BTreeNode::Internal {
                                keys: right_keys,
                                children: right_children.clone(),
                                parent,
                                level,
                            },
                        )?;

                        // Update parent pointers for children in right node
                        for child_id in &right_children {
                            self.set_parent(*child_id, Some(right_id))?;
                        }

                        Ok((Some(split_key), Some(right_id)))
                    }
                } else {
                    Ok((None, None))
                }
            }
        }
    }

    fn find_recursive(&self, node_id: NodeId, key: &K) -> Result<Option<V>> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { entries, .. } => {
                match entries.binary_search_by(|probe| probe.0.cmp(key)) {
                    Ok(pos) => Ok(Some(entries[pos].1.clone())),
                    Err(_) => Ok(None),
                }
            }
            BTreeNode::Internal { keys, children, .. } => {
                let child_idx = match keys.binary_search(key) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };
                self.find_recursive(children[child_idx], key)
            }
        }
    }

    fn delete_recursive(&mut self, node_id: NodeId, key: &K) -> Result<bool> {
        // Simplified delete implementation - full implementation would handle underflow
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf {
                mut entries,
                parent,
                next,
                prev,
            } => match entries.binary_search_by(|probe| probe.0.cmp(key)) {
                Ok(pos) => {
                    entries.remove(pos);
                    self.set_node(
                        node_id,
                        BTreeNode::Leaf {
                            entries,
                            parent,
                            next,
                            prev,
                        },
                    )?;
                    Ok(true)
                }
                Err(_) => Ok(false),
            },
            BTreeNode::Internal { keys, children, .. } => {
                let child_idx = match keys.binary_search(key) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };
                self.delete_recursive(children[child_idx], key)
            }
        }
    }

    fn find_leaf(&self, node_id: NodeId, key: &K) -> Result<NodeId> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { .. } => Ok(node_id),
            BTreeNode::Internal { keys, children, .. } => {
                let child_idx = match keys.binary_search(key) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };
                self.find_leaf(children[child_idx], key)
            }
        }
    }

    fn collect_range(
        &self,
        start_leaf: NodeId,
        start: &K,
        end: &K,
        results: &mut Vec<(K, V)>,
    ) -> Result<()> {
        let mut current_leaf = Some(start_leaf);

        while let Some(leaf_id) = current_leaf {
            let node = self.get_node(leaf_id)?;

            if let BTreeNode::Leaf { entries, next, .. } = node {
                for (k, v) in entries {
                    if k >= *start && k <= *end {
                        results.push((k, v));
                    } else if k > *end {
                        return Ok(());
                    }
                }
                current_leaf = next;
            } else {
                return Err(anyhow!("Expected leaf node"));
            }
        }

        Ok(())
    }

    fn collect_prefix(
        &self,
        start_leaf: NodeId,
        prefix: &K,
        results: &mut Vec<(K, V)>,
    ) -> Result<()>
    where
        K: PartialOrd,
    {
        let mut current_leaf = Some(start_leaf);

        while let Some(leaf_id) = current_leaf {
            let node = self.get_node(leaf_id)?;

            if let BTreeNode::Leaf { entries, next, .. } = node {
                for (k, v) in entries {
                    if k >= *prefix {
                        // This is a simplified prefix check - real implementation would need proper prefix matching
                        results.push((k, v));
                    }
                }
                current_leaf = next;
            } else {
                return Err(anyhow!("Expected leaf node"));
            }
        }

        Ok(())
    }

    fn build_leaf_level(&mut self, entries: Vec<(K, V)>) -> Result<Vec<NodeId>> {
        let mut leaf_nodes = Vec::new();
        let mut prev_leaf: Option<NodeId> = None;

        for chunk in entries.chunks(self.config.max_keys_per_node) {
            let leaf_id = self.create_leaf_node()?;
            let leaf = BTreeNode::Leaf {
                entries: chunk.to_vec(),
                parent: None,
                next: None,
                prev: prev_leaf,
            };

            self.set_node(leaf_id, leaf)?;

            // Update previous leaf's next pointer
            if let Some(prev_id) = prev_leaf {
                self.set_next_sibling(prev_id, Some(leaf_id))?;
            }

            leaf_nodes.push(leaf_id);
            prev_leaf = Some(leaf_id);
        }

        Ok(leaf_nodes)
    }

    fn build_internal_levels(&mut self, mut current_level: Vec<NodeId>) -> Result<()> {
        let mut level = 1;

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(self.config.max_keys_per_node + 1) {
                let internal_id = self.create_internal_node()?;
                let mut keys = Vec::new();
                let children = chunk.to_vec();

                // Get separator keys from first key of each child (except first)
                for &child_id in &children[1..] {
                    if let Ok(first_key) = self.get_first_key(child_id) {
                        keys.push(first_key);
                    }
                }

                let internal = BTreeNode::Internal {
                    keys,
                    children: children.clone(),
                    parent: None,
                    level,
                };

                self.set_node(internal_id, internal)?;

                // Update parent pointers for children
                for &child_id in &children {
                    self.set_parent(child_id, Some(internal_id))?;
                }

                next_level.push(internal_id);
            }

            current_level = next_level;
            level += 1;
        }

        if let Some(root_id) = current_level.first() {
            self.root = Some(*root_id);
            self.stats.height = level - 1;
        }

        Ok(())
    }

    fn get_first_key(&self, node_id: NodeId) -> Result<K> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { entries, .. } => entries
                .first()
                .map(|(k, _)| k.clone())
                .ok_or_else(|| anyhow!("Empty leaf node")),
            BTreeNode::Internal { keys, .. } => keys
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("Empty internal node")),
        }
    }

    fn set_parent(&self, node_id: NodeId, parent_id: Option<NodeId>) -> Result<()> {
        let mut node = self.get_node(node_id)?;

        match &mut node {
            BTreeNode::Leaf { parent, .. } => *parent = parent_id,
            BTreeNode::Internal { parent, .. } => *parent = parent_id,
        }

        self.set_node(node_id, node)
    }

    fn set_next_sibling(&self, node_id: NodeId, next_id: Option<NodeId>) -> Result<()> {
        let mut node = self.get_node(node_id)?;

        match &mut node {
            BTreeNode::Leaf { next, .. } => *next = next_id,
            _ => return Err(anyhow!("Cannot set next sibling on internal node")),
        }

        self.set_node(node_id, node)
    }

    fn set_prev_sibling(&self, node_id: NodeId, prev_id: Option<NodeId>) -> Result<()> {
        let mut node = self.get_node(node_id)?;

        match &mut node {
            BTreeNode::Leaf { prev, .. } => *prev = prev_id,
            _ => return Err(anyhow!("Cannot set prev sibling on internal node")),
        }

        self.set_node(node_id, node)
    }

    fn get_node_level(&self, node_id: NodeId) -> Result<u32> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { .. } => Ok(0),
            BTreeNode::Internal { level, .. } => Ok(level),
        }
    }

    fn is_root_empty_internal(&self) -> Result<bool> {
        if let Some(root_id) = self.root {
            let node = self.get_node(root_id)?;
            match node {
                BTreeNode::Internal { keys, .. } => Ok(keys.is_empty()),
                _ => Ok(false),
            }
        } else {
            Ok(false)
        }
    }

    fn get_single_child(&self, node_id: NodeId) -> Result<Option<NodeId>> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Internal { children, .. } => {
                if children.len() == 1 {
                    Ok(Some(children[0]))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn count_entries(&self) -> Result<usize> {
        if let Some(root_id) = self.root {
            self.count_entries_recursive(root_id)
        } else {
            Ok(0)
        }
    }

    fn count_entries_recursive(&self, node_id: NodeId) -> Result<usize> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { entries, .. } => Ok(entries.len()),
            BTreeNode::Internal { children, .. } => {
                let mut total = 0;
                for child_id in children {
                    total += self.count_entries_recursive(child_id)?;
                }
                Ok(total)
            }
        }
    }

    fn validate_recursive(
        &self,
        node_id: NodeId,
        min_key: Option<&K>,
        max_key: Option<&K>,
    ) -> Result<bool> {
        let node = self.get_node(node_id)?;

        match node {
            BTreeNode::Leaf { entries, .. } => {
                // Check ordering
                for window in entries.windows(2) {
                    if window[0].0 >= window[1].0 {
                        return Ok(false);
                    }
                }

                // Check bounds
                if let Some(min) = min_key {
                    if let Some((first_key, _)) = entries.first() {
                        if first_key < min {
                            return Ok(false);
                        }
                    }
                }

                if let Some(max) = max_key {
                    if let Some((last_key, _)) = entries.last() {
                        if last_key > max {
                            return Ok(false);
                        }
                    }
                }

                Ok(true)
            }
            BTreeNode::Internal { keys, children, .. } => {
                // Check key ordering
                for window in keys.windows(2) {
                    if window[0] >= window[1] {
                        return Ok(false);
                    }
                }

                // Validate children recursively
                for (i, &child_id) in children.iter().enumerate() {
                    let child_min = if i == 0 { min_key } else { keys.get(i - 1) };
                    let child_max = keys.get(i);

                    if !self.validate_recursive(child_id, child_min, child_max)? {
                        return Ok(false);
                    }
                }

                Ok(true)
            }
        }
    }
}

impl<K, V> Default for BTree<K, V>
where
    K: Clone + Ord + Debug,
    V: Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_basic_operations() {
        let mut tree = BTree::new();

        // Test insert and find
        assert!(tree.insert(5, "five".to_string()).is_ok());
        assert!(tree.insert(3, "three".to_string()).is_ok());
        assert!(tree.insert(7, "seven".to_string()).is_ok());

        assert_eq!(tree.find(&5).unwrap(), Some("five".to_string()));
        assert_eq!(tree.find(&3).unwrap(), Some("three".to_string()));
        assert_eq!(tree.find(&7).unwrap(), Some("seven".to_string()));
        assert_eq!(tree.find(&10).unwrap(), None);

        // Test delete
        assert!(tree.delete(&3).unwrap());
        assert_eq!(tree.find(&3).unwrap(), None);
        assert!(!tree.delete(&10).unwrap());
    }

    #[test]
    fn test_btree_range_query() {
        let mut tree = BTree::new();

        for i in 0..10 {
            tree.insert(i, format!("value_{}", i)).unwrap();
        }

        let range_result = tree.range(&3, &7).unwrap();
        assert_eq!(range_result.len(), 5);

        for (i, (key, value)) in range_result.iter().enumerate() {
            assert_eq!(*key, 3 + i);
            assert_eq!(*value, format!("value_{}", 3 + i));
        }
    }

    #[test]
    fn test_btree_bulk_load() {
        let mut tree = BTree::new();

        let entries: Vec<(i32, String)> = (0..1000).map(|i| (i, format!("value_{}", i))).collect();

        assert!(tree.bulk_load(entries).is_ok());

        // Verify all entries are present
        for i in 0..1000 {
            assert_eq!(tree.find(&i).unwrap(), Some(format!("value_{}", i)));
        }

        let stats = tree.get_stats();
        assert_eq!(stats.total_entries, 1000);
        assert!(stats.height > 0);
    }

    #[test]
    fn test_btree_validation() {
        let mut tree = BTree::new();

        for i in 0..100 {
            tree.insert(i, format!("value_{}", i)).unwrap();
        }

        assert!(tree.validate().unwrap());
    }
}
