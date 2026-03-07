//! In-memory B+ tree index for RDF triple patterns with SPO, POS, OSP orderings.
//!
//! This module provides a fully in-memory B+ tree index designed for high-throughput
//! triple pattern matching without requiring disk I/O. It complements the disk-based
//! B+Tree in `crate::btree` by providing a cache-friendly in-process index.
//!
//! # Design
//!
//! - Order-16 B+ tree (up to 16 keys per node)
//! - Leaf nodes form a doubly-linked chain for efficient range scans
//! - Three orderings maintained: SPO, POS, OSP
//! - All operations are O(log n)

use crate::error::{Result, TdbError};

/// Order of the B+ tree (maximum keys per node before splitting).
const BTREE_ORDER: usize = 16;

/// An encoded triple using integer IDs (dictionary-encoded).
///
/// All three components are 64-bit node IDs obtained from the dictionary layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EncodedTriple {
    /// Subject node ID
    pub s: u64,
    /// Predicate node ID
    pub p: u64,
    /// Object node ID
    pub o: u64,
}

impl EncodedTriple {
    /// Create a new encoded triple from raw node IDs.
    pub fn new(s: u64, p: u64, o: u64) -> Self {
        Self { s, p, o }
    }
}

/// Ordering strategy for a B+ tree triple index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripleOrdering {
    /// Subject → Predicate → Object (default ordering)
    Spo,
    /// Predicate → Object → Subject (efficient for predicate-bound queries)
    Pos,
    /// Object → Subject → Predicate (efficient for object-bound queries)
    Osp,
}

impl TripleOrdering {
    /// Transform a triple into its ordered (first, second, third) key for this ordering.
    pub fn key_for(self, triple: &EncodedTriple) -> (u64, u64, u64) {
        match self {
            Self::Spo => (triple.s, triple.p, triple.o),
            Self::Pos => (triple.p, triple.o, triple.s),
            Self::Osp => (triple.o, triple.s, triple.p),
        }
    }

    /// Reconstruct a canonical [`EncodedTriple`] from an ordered key.
    pub fn triple_from_key(self, k0: u64, k1: u64, k2: u64) -> EncodedTriple {
        match self {
            Self::Spo => EncodedTriple::new(k0, k1, k2),
            Self::Pos => EncodedTriple::new(k2, k0, k1),
            Self::Osp => EncodedTriple::new(k1, k2, k0),
        }
    }
}

/// A composite triple key ordered for a specific [`TripleOrdering`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TripleKey(u64, u64, u64);

impl TripleKey {
    fn from_triple(ordering: TripleOrdering, triple: &EncodedTriple) -> Self {
        let (a, b, c) = ordering.key_for(triple);
        Self(a, b, c)
    }
}

// ---------------------------------------------------------------------------
// Internal B+ tree nodes
// ---------------------------------------------------------------------------

/// An internal (non-leaf) node in the B+ tree.
#[derive(Debug, Clone)]
struct InternalNode {
    /// Separator keys; `keys[i]` separates `children[i]` and `children[i+1]`.
    keys: Vec<TripleKey>,
    /// Child node indices into the node arena.
    children: Vec<usize>,
}

/// A leaf node in the B+ tree.
#[derive(Debug, Clone)]
struct LeafNode {
    /// Sorted triple keys stored in this leaf.
    keys: Vec<TripleKey>,
    /// Index of the next leaf node (for range scans); `usize::MAX` means no next.
    next_leaf: usize,
}

impl LeafNode {
    fn new() -> Self {
        Self {
            keys: Vec::with_capacity(BTREE_ORDER),
            next_leaf: usize::MAX,
        }
    }
}

impl InternalNode {
    fn new() -> Self {
        Self {
            keys: Vec::with_capacity(BTREE_ORDER),
            children: Vec::with_capacity(BTREE_ORDER + 1),
        }
    }
}

/// A node in the B+ tree arena (either internal or leaf).
#[derive(Debug, Clone)]
enum ArenaNode {
    Internal(InternalNode),
    Leaf(LeafNode),
}

// ---------------------------------------------------------------------------
// B+ tree arena
// ---------------------------------------------------------------------------

/// Memory arena that stores all B+ tree nodes by index.
///
/// Using an arena avoids recursive ownership issues and allows the leaf-chain
/// to be implemented with simple integer indices.
#[derive(Debug)]
struct NodeArena {
    nodes: Vec<ArenaNode>,
}

impl NodeArena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn alloc(&mut self, node: ArenaNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    fn get(&self, idx: usize) -> Option<&ArenaNode> {
        self.nodes.get(idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut ArenaNode> {
        self.nodes.get_mut(idx)
    }
}

// ---------------------------------------------------------------------------
// Split result
// ---------------------------------------------------------------------------

/// Result of a node split operation.
struct SplitResult {
    /// The promoted key that rises into the parent internal node.
    promoted_key: TripleKey,
    /// The index of the newly created right sibling node.
    new_node_idx: usize,
}

// ---------------------------------------------------------------------------
// BTreeTripleIndex
// ---------------------------------------------------------------------------

/// A single in-memory B+ tree index for RDF triples with a fixed ordering.
///
/// Keys are stored in the order defined by [`TripleOrdering`] so that prefix
/// scans (e.g., all triples with a given subject in SPO order) are O(log n +
/// result size).
pub struct BTreeTripleIndex {
    ordering: TripleOrdering,
    arena: NodeArena,
    root: usize,
    size: usize,
}

impl BTreeTripleIndex {
    /// Create a new empty B+ tree index with the specified ordering.
    pub fn new(ordering: TripleOrdering) -> Self {
        let mut arena = NodeArena::new();
        let root = arena.alloc(ArenaNode::Leaf(LeafNode::new()));
        Self {
            ordering,
            arena,
            root,
            size: 0,
        }
    }

    /// Return the number of triples stored in this index.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Return `true` if the index contains no triples.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check whether the given triple is present.
    pub fn contains(&self, triple: &EncodedTriple) -> bool {
        let key = TripleKey::from_triple(self.ordering, triple);
        self.search(self.root, &key)
    }

    /// Insert a triple. No-op if the triple is already present.
    pub fn insert(&mut self, triple: EncodedTriple) {
        let key = TripleKey::from_triple(self.ordering, &triple);
        if let Some(split) = self.insert_recursive(self.root, key) {
            // Root was split – create a new root
            let mut new_root = InternalNode::new();
            new_root.keys.push(split.promoted_key);
            new_root.children.push(self.root);
            new_root.children.push(split.new_node_idx);
            let new_root_idx = self.arena.alloc(ArenaNode::Internal(new_root));
            self.root = new_root_idx;
        }
    }

    /// Delete a triple. Returns `true` if the triple was present and removed.
    pub fn delete(&mut self, triple: &EncodedTriple) -> bool {
        let key = TripleKey::from_triple(self.ordering, triple);
        let removed = self.delete_from_leaf(self.root, &key);
        if removed {
            self.size = self.size.saturating_sub(1);
        }
        removed
    }

    /// Scan all triples matching a pattern where `None` means wildcard.
    ///
    /// Pattern components `(s, p, o)` are in canonical triple order regardless
    /// of the index ordering.
    pub fn scan_pattern(
        &self,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
    ) -> Vec<EncodedTriple> {
        // Build lower-bound key using the index ordering
        let (bound_a, bound_b, bound_c) = match self.ordering {
            TripleOrdering::Spo => {
                let a = s.unwrap_or(0);
                let b = if s.is_some() { p.unwrap_or(0) } else { 0 };
                let c = if s.is_some() && p.is_some() {
                    o.unwrap_or(0)
                } else {
                    0
                };
                (a, b, c)
            }
            TripleOrdering::Pos => {
                let a = p.unwrap_or(0);
                let b = if p.is_some() { o.unwrap_or(0) } else { 0 };
                let c = if p.is_some() && o.is_some() {
                    s.unwrap_or(0)
                } else {
                    0
                };
                (a, b, c)
            }
            TripleOrdering::Osp => {
                let a = o.unwrap_or(0);
                let b = if o.is_some() { s.unwrap_or(0) } else { 0 };
                let c = if o.is_some() && s.is_some() {
                    p.unwrap_or(0)
                } else {
                    0
                };
                (a, b, c)
            }
        };

        let start_key = TripleKey(bound_a, bound_b, bound_c);
        let leaf_idx = self.find_leaf(self.root, &start_key);
        let mut results = Vec::new();
        let mut current_idx = leaf_idx;

        loop {
            if current_idx == usize::MAX {
                break;
            }
            let node = match self.arena.get(current_idx) {
                Some(ArenaNode::Leaf(leaf)) => leaf,
                _ => break,
            };

            for &key in &node.keys {
                if key < start_key {
                    continue;
                }
                let triple = self.ordering.triple_from_key(key.0, key.1, key.2);
                if self.matches_pattern(&triple, s, p, o) {
                    // Early termination: if the key's first component changed and
                    // the first constraint is bound, stop scanning.
                    results.push(triple);
                } else {
                    // If the first bound component no longer matches, we can stop
                    let stops = self.should_stop_scan(key, s, p, o);
                    if stops {
                        return results;
                    }
                }
            }

            current_idx = node.next_leaf;
        }

        results
    }

    /// Return all triples stored in this index, in key order.
    pub fn all_triples(&self) -> Vec<EncodedTriple> {
        let mut results = Vec::new();
        let mut current_idx = self.find_leftmost_leaf(self.root);

        loop {
            if current_idx == usize::MAX {
                break;
            }
            let node = match self.arena.get(current_idx) {
                Some(ArenaNode::Leaf(leaf)) => leaf,
                _ => break,
            };
            for &key in &node.keys {
                results.push(self.ordering.triple_from_key(key.0, key.1, key.2));
            }
            current_idx = node.next_leaf;
        }

        results
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn search(&self, node_idx: usize, key: &TripleKey) -> bool {
        match self.arena.get(node_idx) {
            Some(ArenaNode::Leaf(leaf)) => leaf.keys.binary_search(key).is_ok(),
            Some(ArenaNode::Internal(internal)) => {
                let child = self.find_child_index(internal, key);
                self.search(internal.children[child], key)
            }
            None => false,
        }
    }

    /// Insert into the subtree rooted at `node_idx`.
    /// Returns `Some(SplitResult)` if a split propagates upward.
    fn insert_recursive(&mut self, node_idx: usize, key: TripleKey) -> Option<SplitResult> {
        let node_kind = match self.arena.get(node_idx) {
            Some(ArenaNode::Leaf(_)) => NodeKind::Leaf,
            Some(ArenaNode::Internal(_)) => NodeKind::Internal,
            None => return None,
        };

        match node_kind {
            NodeKind::Leaf => {
                // Insert into leaf
                let leaf = match self.arena.get_mut(node_idx) {
                    Some(ArenaNode::Leaf(l)) => l,
                    _ => return None,
                };

                match leaf.keys.binary_search(&key) {
                    Ok(_) => {
                        // Already present – nothing to do
                        return None;
                    }
                    Err(pos) => {
                        leaf.keys.insert(pos, key);
                        self.size += 1;
                    }
                }

                // Check if split needed
                if self.arena.get(node_idx).map_or(0, |n| match n {
                    ArenaNode::Leaf(l) => l.keys.len(),
                    _ => 0,
                }) < BTREE_ORDER
                {
                    None
                } else {
                    Some(self.split_leaf(node_idx))
                }
            }

            NodeKind::Internal => {
                // Find child index
                let child_idx = {
                    let internal = match self.arena.get(node_idx) {
                        Some(ArenaNode::Internal(n)) => n,
                        _ => return None,
                    };
                    let pos = self.find_child_index(internal, &key);
                    internal.children[pos]
                };

                // Recurse into child
                let split = self.insert_recursive(child_idx, key)?;

                // Insert promoted key into this internal node
                let internal = match self.arena.get_mut(node_idx) {
                    Some(ArenaNode::Internal(n)) => n,
                    _ => return None,
                };

                let pos = internal
                    .keys
                    .binary_search(&split.promoted_key)
                    .unwrap_or_else(|p| p);
                internal.keys.insert(pos, split.promoted_key);
                internal.children.insert(pos + 1, split.new_node_idx);

                if internal.keys.len() < BTREE_ORDER {
                    None
                } else {
                    Some(self.split_internal(node_idx))
                }
            }
        }
    }

    fn split_leaf(&mut self, node_idx: usize) -> SplitResult {
        let (right_keys, right_next, promoted_key) = {
            let leaf = match self.arena.get_mut(node_idx) {
                Some(ArenaNode::Leaf(l)) => l,
                _ => panic!("expected leaf node"),
            };
            let mid = leaf.keys.len() / 2;
            let promoted_key = leaf.keys[mid];
            let right_keys: Vec<TripleKey> = leaf.keys.split_off(mid);
            let right_next = leaf.next_leaf;
            (right_keys, right_next, promoted_key)
        };

        let mut new_leaf = LeafNode::new();
        new_leaf.keys = right_keys;
        new_leaf.next_leaf = right_next;
        let new_idx = self.arena.alloc(ArenaNode::Leaf(new_leaf));

        // Link left leaf to the new right leaf
        if let Some(ArenaNode::Leaf(l)) = self.arena.get_mut(node_idx) {
            l.next_leaf = new_idx;
        }

        SplitResult {
            promoted_key,
            new_node_idx: new_idx,
        }
    }

    fn split_internal(&mut self, node_idx: usize) -> SplitResult {
        let (right_keys, right_children, promoted_key) = {
            let internal = match self.arena.get_mut(node_idx) {
                Some(ArenaNode::Internal(n)) => n,
                _ => panic!("expected internal node"),
            };
            let mid = internal.keys.len() / 2;
            let promoted_key = internal.keys[mid];
            let right_keys: Vec<TripleKey> = internal.keys.split_off(mid + 1);
            internal.keys.truncate(mid);
            let right_children: Vec<usize> = internal.children.split_off(mid + 1);
            (right_keys, right_children, promoted_key)
        };

        let mut new_internal = InternalNode::new();
        new_internal.keys = right_keys;
        new_internal.children = right_children;
        let new_idx = self.arena.alloc(ArenaNode::Internal(new_internal));

        SplitResult {
            promoted_key,
            new_node_idx: new_idx,
        }
    }

    fn delete_from_leaf(&mut self, node_idx: usize, key: &TripleKey) -> bool {
        match self.arena.get(node_idx) {
            Some(ArenaNode::Leaf(_)) => {
                let leaf = match self.arena.get_mut(node_idx) {
                    Some(ArenaNode::Leaf(l)) => l,
                    _ => return false,
                };
                match leaf.keys.binary_search(key) {
                    Ok(pos) => {
                        leaf.keys.remove(pos);
                        true
                    }
                    Err(_) => false,
                }
            }
            Some(ArenaNode::Internal(_)) => {
                let child_idx = {
                    let internal = match self.arena.get(node_idx) {
                        Some(ArenaNode::Internal(n)) => n,
                        _ => return false,
                    };
                    let pos = self.find_child_index(internal, key);
                    internal.children[pos]
                };
                self.delete_from_leaf(child_idx, key)
            }
            None => false,
        }
    }

    fn find_child_index(&self, internal: &InternalNode, key: &TripleKey) -> usize {
        match internal.keys.binary_search(key) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        }
    }

    fn find_leaf(&self, node_idx: usize, key: &TripleKey) -> usize {
        match self.arena.get(node_idx) {
            Some(ArenaNode::Leaf(_)) => node_idx,
            Some(ArenaNode::Internal(internal)) => {
                let pos = self.find_child_index(internal, key);
                let child = internal.children[pos];
                self.find_leaf(child, key)
            }
            None => usize::MAX,
        }
    }

    fn find_leftmost_leaf(&self, node_idx: usize) -> usize {
        match self.arena.get(node_idx) {
            Some(ArenaNode::Leaf(_)) => node_idx,
            Some(ArenaNode::Internal(internal)) => {
                let first_child = internal.children[0];
                self.find_leftmost_leaf(first_child)
            }
            None => usize::MAX,
        }
    }

    fn matches_pattern(
        &self,
        triple: &EncodedTriple,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
    ) -> bool {
        s.map_or(true, |sv| triple.s == sv)
            && p.map_or(true, |pv| triple.p == pv)
            && o.map_or(true, |ov| triple.o == ov)
    }

    /// Return `true` if scanning forward cannot produce more matches.
    fn should_stop_scan(
        &self,
        key: TripleKey,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
    ) -> bool {
        match self.ordering {
            TripleOrdering::Spo => {
                if let Some(sv) = s {
                    if key.0 > sv {
                        return true;
                    }
                    if let Some(pv) = p {
                        if key.0 == sv && key.1 > pv {
                            return true;
                        }
                    }
                }
                false
            }
            TripleOrdering::Pos => {
                if let Some(pv) = p {
                    if key.0 > pv {
                        return true;
                    }
                    if let Some(ov) = o {
                        if key.0 == pv && key.1 > ov {
                            return true;
                        }
                    }
                }
                false
            }
            TripleOrdering::Osp => {
                if let Some(ov) = o {
                    if key.0 > ov {
                        return true;
                    }
                    if let Some(sv) = s {
                        if key.0 == ov && key.1 > sv {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }
}

/// Tiny enum to avoid multiple borrows.
enum NodeKind {
    Leaf,
    Internal,
}

// ---------------------------------------------------------------------------
// TripleIndexSet
// ---------------------------------------------------------------------------

/// A set of three B+ tree indexes (SPO, POS, OSP) that together provide
/// efficient lookup for any triple pattern binding.
///
/// All three indexes are kept in sync; every insert/delete touches all three.
pub struct TripleIndexSet {
    spo: BTreeTripleIndex,
    pos: BTreeTripleIndex,
    osp: BTreeTripleIndex,
}

impl Default for TripleIndexSet {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleIndexSet {
    /// Create a new, empty triple index set.
    pub fn new() -> Self {
        Self {
            spo: BTreeTripleIndex::new(TripleOrdering::Spo),
            pos: BTreeTripleIndex::new(TripleOrdering::Pos),
            osp: BTreeTripleIndex::new(TripleOrdering::Osp),
        }
    }

    /// Insert a triple into all three indexes.
    pub fn insert(&mut self, triple: EncodedTriple) {
        self.spo.insert(triple);
        self.pos.insert(triple);
        self.osp.insert(triple);
    }

    /// Delete a triple from all three indexes. Returns `true` if it was present.
    pub fn delete(&mut self, triple: &EncodedTriple) -> bool {
        let found = self.spo.delete(triple);
        if found {
            self.pos.delete(triple);
            self.osp.delete(triple);
        }
        found
    }

    /// Return `true` if the triple is present (uses SPO index).
    pub fn contains(&self, triple: &EncodedTriple) -> bool {
        self.spo.contains(triple)
    }

    /// Number of triples in the store (as tracked by the SPO index).
    pub fn len(&self) -> usize {
        self.spo.len()
    }

    /// Return `true` if there are no triples.
    pub fn is_empty(&self) -> bool {
        self.spo.is_empty()
    }

    /// Find all triples with the given subject (uses SPO index).
    pub fn find_by_subject(&self, s: u64) -> Vec<EncodedTriple> {
        self.spo.scan_pattern(Some(s), None, None)
    }

    /// Find all triples with the given predicate (uses POS index).
    pub fn find_by_predicate(&self, p: u64) -> Vec<EncodedTriple> {
        self.pos.scan_pattern(None, Some(p), None)
    }

    /// Find all triples with the given object (uses OSP index).
    pub fn find_by_object(&self, o: u64) -> Vec<EncodedTriple> {
        self.osp.scan_pattern(None, None, Some(o))
    }

    /// Find all triples with given subject and predicate (uses SPO index).
    pub fn find_by_sp(&self, s: u64, p: u64) -> Vec<EncodedTriple> {
        self.spo.scan_pattern(Some(s), Some(p), None)
    }

    /// Find all triples with given predicate and object (uses POS index).
    pub fn find_by_po(&self, p: u64, o: u64) -> Vec<EncodedTriple> {
        self.pos.scan_pattern(None, Some(p), Some(o))
    }

    /// Find all triples with given subject and object (uses OSP index).
    pub fn find_by_so(&self, s: u64, o: u64) -> Vec<EncodedTriple> {
        self.osp.scan_pattern(Some(s), None, Some(o))
    }

    /// Find all triples matching a full pattern (any component `None` = wildcard).
    pub fn find_pattern(
        &self,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
    ) -> Result<Vec<EncodedTriple>> {
        let results = match (s, p, o) {
            (Some(_), Some(_), Some(_)) => {
                // Exact lookup
                let candidate = EncodedTriple::new(s.unwrap_or(0), p.unwrap_or(0), o.unwrap_or(0));
                if self.contains(&candidate) {
                    vec![candidate]
                } else {
                    vec![]
                }
            }
            (Some(_), Some(_), None) => self.spo.scan_pattern(s, p, o),
            (Some(_), None, None) => self.spo.scan_pattern(s, p, o),
            (None, Some(_), Some(_)) => self.pos.scan_pattern(s, p, o),
            (None, Some(_), None) => self.pos.scan_pattern(s, p, o),
            (Some(_), None, Some(_)) => self.osp.scan_pattern(s, p, o),
            (None, None, Some(_)) => self.osp.scan_pattern(s, p, o),
            (None, None, None) => self.spo.all_triples(),
        };
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: u64, p: u64, o: u64) -> EncodedTriple {
        EncodedTriple::new(s, p, o)
    }

    #[test]
    fn test_single_index_insert_contains() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        let t = make_triple(1, 2, 3);
        assert!(!idx.contains(&t));
        idx.insert(t);
        assert!(idx.contains(&t));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_single_index_delete() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        let t = make_triple(10, 20, 30);
        idx.insert(t);
        assert!(idx.contains(&t));
        assert!(idx.delete(&t));
        assert!(!idx.contains(&t));
        assert!(!idx.delete(&t)); // Second delete returns false
    }

    #[test]
    fn test_duplicate_insert_ignored() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        let t = make_triple(5, 6, 7);
        idx.insert(t);
        idx.insert(t);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_many_inserts_trigger_split() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        // Insert more than BTREE_ORDER triples to force splits
        for i in 0..100u64 {
            idx.insert(make_triple(i, i + 1, i + 2));
        }
        assert_eq!(idx.len(), 100);
        for i in 0..100u64 {
            assert!(idx.contains(&make_triple(i, i + 1, i + 2)));
        }
    }

    #[test]
    fn test_scan_pattern_by_subject() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        idx.insert(make_triple(1, 10, 100));
        idx.insert(make_triple(1, 20, 200));
        idx.insert(make_triple(2, 10, 100));
        let results = idx.scan_pattern(Some(1), None, None);
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.s, 1);
        }
    }

    #[test]
    fn test_scan_pattern_by_sp() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        idx.insert(make_triple(1, 10, 100));
        idx.insert(make_triple(1, 10, 200));
        idx.insert(make_triple(1, 20, 300));
        let results = idx.scan_pattern(Some(1), Some(10), None);
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.s, 1);
            assert_eq!(t.p, 10);
        }
    }

    #[test]
    fn test_triple_index_set_basic() {
        let mut set = TripleIndexSet::new();
        let t = make_triple(1, 2, 3);
        assert!(!set.contains(&t));
        set.insert(t);
        assert!(set.contains(&t));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_triple_index_set_delete() {
        let mut set = TripleIndexSet::new();
        let t = make_triple(5, 5, 5);
        set.insert(t);
        assert!(set.delete(&t));
        assert!(!set.contains(&t));
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_triple_index_set_find_by_subject() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(1, 20, 200));
        set.insert(make_triple(2, 10, 100));
        let results = set.find_by_subject(1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_triple_index_set_find_by_predicate() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(2, 10, 200));
        set.insert(make_triple(3, 20, 300));
        let results = set.find_by_predicate(10);
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.p, 10);
        }
    }

    #[test]
    fn test_triple_index_set_find_by_object() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(2, 20, 100));
        set.insert(make_triple(3, 30, 200));
        let results = set.find_by_object(100);
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.o, 100);
        }
    }

    #[test]
    fn test_triple_index_set_find_by_sp() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(1, 10, 200));
        set.insert(make_triple(1, 20, 300));
        let results = set.find_by_sp(1, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_triple_index_set_find_by_po() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(2, 10, 100));
        set.insert(make_triple(3, 10, 200));
        let results = set.find_by_po(10, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_triple_index_set_find_by_so() {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 10, 100));
        set.insert(make_triple(1, 20, 100));
        set.insert(make_triple(2, 10, 100));
        let results = set.find_by_so(1, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_all_three_indexes_consistent() {
        let mut set = TripleIndexSet::new();
        let t = make_triple(100, 200, 300);
        set.insert(t);
        // All three underlying indexes must see the triple
        assert!(set.spo.contains(&t));
        assert!(set.pos.contains(&t));
        assert!(set.osp.contains(&t));
        set.delete(&t);
        assert!(!set.spo.contains(&t));
        assert!(!set.pos.contains(&t));
        assert!(!set.osp.contains(&t));
    }

    #[test]
    fn test_find_pattern_exact() -> Result<()> {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 2, 3));
        let results = set.find_pattern(Some(1), Some(2), Some(3))?;
        assert_eq!(results.len(), 1);
        let results_miss = set.find_pattern(Some(1), Some(2), Some(99))?;
        assert_eq!(results_miss.len(), 0);
        Ok(())
    }

    #[test]
    fn test_find_pattern_wildcard_all() -> Result<()> {
        let mut set = TripleIndexSet::new();
        set.insert(make_triple(1, 2, 3));
        set.insert(make_triple(4, 5, 6));
        let results = set.find_pattern(None, None, None)?;
        assert_eq!(results.len(), 2);
        Ok(())
    }

    #[test]
    fn test_ordering_pos_index() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Pos);
        idx.insert(make_triple(1, 10, 100));
        idx.insert(make_triple(2, 10, 200));
        idx.insert(make_triple(3, 20, 100));
        // Scan by predicate using POS index
        let results = idx.scan_pattern(None, Some(10), None);
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.p, 10);
        }
    }

    #[test]
    fn test_ordering_osp_index() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Osp);
        idx.insert(make_triple(1, 10, 100));
        idx.insert(make_triple(2, 20, 100));
        idx.insert(make_triple(3, 30, 200));
        // Scan by object using OSP index
        let results = idx.scan_pattern(None, None, Some(100));
        assert_eq!(results.len(), 2);
        for t in &results {
            assert_eq!(t.o, 100);
        }
    }

    #[test]
    fn test_all_triples_iteration() {
        let mut idx = BTreeTripleIndex::new(TripleOrdering::Spo);
        let expected: Vec<EncodedTriple> = (0..50u64).map(|i| make_triple(i, i, i)).collect();
        for &t in &expected {
            idx.insert(t);
        }
        let mut all = idx.all_triples();
        all.sort_by_key(|t| (t.s, t.p, t.o));
        assert_eq!(all.len(), expected.len());
    }
}
