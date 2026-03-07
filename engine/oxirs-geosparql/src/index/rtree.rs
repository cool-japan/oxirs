//! Pure Rust R*-tree spatial index (Beckmann et al. 1990)
//!
//! This module implements the R*-tree variant of R-tree with:
//! - **Forced reinsertion** on overflow (reduces tree restructuring)
//! - **Overlap + area + perimeter minimization** split criterion
//! - **Generic value type** `T` so any payload can be indexed
//!
//! The implementation is self-contained (no external spatial crates) and
//! operates on 2D axis-aligned bounding boxes ([`BoundingBox`]).
//!
//! # References
//!
//! - Guttman (1984) "R-Trees: A Dynamic Index Structure for Spatial Searching"
//! - Beckmann et al. (1990) "The R*-tree: An Efficient and Robust Access Method"
//!
//! # Usage
//!
//! ```rust
//! use oxirs_geosparql::index::rtree::{RTree, BoundingBox};
//!
//! let mut tree: RTree<&str> = RTree::new();
//!
//! tree.insert(BoundingBox::new(0.0, 0.0, 1.0, 1.0), "A");
//! tree.insert(BoundingBox::new(2.0, 2.0, 3.0, 3.0), "B");
//!
//! let results = tree.search(&BoundingBox::new(0.5, 0.5, 1.5, 1.5));
//! assert_eq!(results.len(), 1);
//! assert_eq!(*results[0], "A");
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Constants (R* tree parameters)
// ---------------------------------------------------------------------------

/// Maximum number of entries per node (M).
const MAX_ENTRIES: usize = 9;

/// Minimum number of entries per non-root node (m = ceil(0.4 * M)).
const MIN_ENTRIES: usize = 4;

/// Number of entries to reinsert during overflow (p = ceil(0.3 * M)).
const REINSERT_COUNT: usize = 3;

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// 2D axis-aligned bounding box used for spatial indexing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Minimum X coordinate
    pub min_x: f64,
    /// Minimum Y coordinate
    pub min_y: f64,
    /// Maximum X coordinate
    pub max_x: f64,
    /// Maximum Y coordinate
    pub max_y: f64,
}

impl BoundingBox {
    /// Create a new bounding box. Panics in debug mode if min > max.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        debug_assert!(
            min_x <= max_x && min_y <= max_y,
            "BoundingBox: min must be <= max"
        );
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Create a degenerate (point) bounding box.
    pub fn from_point(x: f64, y: f64) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        }
    }

    /// Area of this bounding box.
    #[inline]
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Half-perimeter of this bounding box (margin in R* terminology).
    #[inline]
    pub fn margin(&self) -> f64 {
        (self.max_x - self.min_x) + (self.max_y - self.min_y)
    }

    /// Area of intersection with another bounding box (0 if non-overlapping).
    #[inline]
    pub fn overlap_area(&self, other: &BoundingBox) -> f64 {
        let ix = (self.max_x.min(other.max_x) - self.min_x.max(other.min_x)).max(0.0);
        let iy = (self.max_y.min(other.max_y) - self.min_y.max(other.min_y)).max(0.0);
        ix * iy
    }

    /// Test if two bounding boxes have any overlap (inclusive boundaries).
    #[inline]
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Test if this bounding box fully contains another.
    #[inline]
    pub fn contains(&self, other: &BoundingBox) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    /// Test if a point lies inside (inclusive).
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Minimum bounding box that contains both `self` and `other`.
    #[inline]
    pub fn expand_to_include(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// How much the area of `self` would increase if `other` were added.
    #[inline]
    pub fn expansion_needed(&self, other: &BoundingBox) -> f64 {
        self.expand_to_include(other).area() - self.area()
    }

    /// Center of this bounding box.
    #[inline]
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// Minimum squared distance from a point `(px, py)` to this box.
    ///
    /// Returns 0 if the point is inside.
    #[inline]
    pub fn min_sq_dist_to_point(&self, px: f64, py: f64) -> f64 {
        let dx = if px < self.min_x {
            self.min_x - px
        } else if px > self.max_x {
            px - self.max_x
        } else {
            0.0
        };
        let dy = if py < self.min_y {
            self.min_y - py
        } else if py > self.max_y {
            py - self.max_y
        } else {
            0.0
        };
        dx * dx + dy * dy
    }

    /// Expand by a uniform `delta` in all directions.
    #[inline]
    pub fn expand_by(&self, delta: f64) -> Self {
        Self {
            min_x: self.min_x - delta,
            min_y: self.min_y - delta,
            max_x: self.max_x + delta,
            max_y: self.max_y + delta,
        }
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BOX({} {},{} {})",
            self.min_x, self.min_y, self.max_x, self.max_y
        )
    }
}

// ---------------------------------------------------------------------------
// Internal node types
// ---------------------------------------------------------------------------

/// A leaf entry: bounding box + stored value.
#[derive(Debug, Clone)]
struct LeafEntry<T> {
    bbox: BoundingBox,
    value: T,
}

/// An internal (branch) entry: bounding box + child node index.
#[derive(Debug, Clone)]
struct BranchEntry {
    bbox: BoundingBox,
    child: usize, // index into RTree::nodes
}

/// A node in the R* tree.
#[derive(Debug, Clone)]
enum Node<T> {
    /// Leaf node: holds actual data entries.
    Leaf { entries: Vec<LeafEntry<T>> },
    /// Internal node: holds pointers to child nodes.
    Internal { entries: Vec<BranchEntry> },
}

impl<T> Node<T> {
    /// Compute the MBR of this node's entries.
    fn mbr(&self) -> Option<BoundingBox> {
        match self {
            Node::Leaf { entries } => entries
                .iter()
                .map(|e| e.bbox)
                .reduce(|a, b| a.expand_to_include(&b)),
            Node::Internal { entries } => entries
                .iter()
                .map(|e| e.bbox)
                .reduce(|a, b| a.expand_to_include(&b)),
        }
    }

    #[allow(dead_code)]
    fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    fn entry_count(&self) -> usize {
        match self {
            Node::Leaf { entries } => entries.len(),
            Node::Internal { entries } => entries.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// RTree
// ---------------------------------------------------------------------------

/// Pure Rust R*-tree spatial index.
///
/// Type parameter `T` is the stored value. `T` must implement `Clone`.
pub struct RTree<T: Clone> {
    /// Flat arena of nodes; index 0 is always the root.
    nodes: Vec<Node<T>>,
    /// Total number of leaf entries.
    size: usize,
    /// Height of the tree (0 = single leaf node).
    height: usize,
}

impl<T: Clone> RTree<T> {
    /// Create a new, empty R*-tree.
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::Leaf {
                entries: Vec::new(),
            }],
            size: 0,
            height: 0,
        }
    }

    /// Number of entries in the tree.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert a new entry with the given bounding box and value.
    pub fn insert(&mut self, bbox: BoundingBox, value: T) {
        let mut reinsertions_at_level: Vec<bool> = vec![false; self.height + 1];
        self.insert_at_level(bbox, value, self.height, &mut reinsertions_at_level);
        self.size += 1;
    }

    /// Internal insert: insert a leaf entry at the given tree level.
    fn insert_at_level(
        &mut self,
        bbox: BoundingBox,
        value: T,
        target_level: usize,
        did_reinsert: &mut Vec<bool>,
    ) {
        // Find the leaf node to insert into (path from root)
        let path = self.choose_subtree_path(bbox, target_level);
        let leaf_idx = *path.last().expect("path non-empty");

        // Insert the entry
        if let Node::Leaf { entries } = &mut self.nodes[leaf_idx] {
            entries.push(LeafEntry { bbox, value });
        }

        // Handle overflows upwards
        self.handle_overflow_path(path, did_reinsert);
    }

    /// Choose a path from root to the target level by minimizing bounding box extension.
    fn choose_subtree_path(&self, bbox: BoundingBox, target_level: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(self.height + 1);
        let mut node_idx = 0; // root
        path.push(node_idx);

        for level in 0..target_level {
            let is_leaf_children = level + 1 == self.height;
            node_idx = self.choose_child(node_idx, bbox, is_leaf_children);
            path.push(node_idx);
        }
        path
    }

    /// Choose which child of an internal node to descend into.
    ///
    /// When the children are leaf nodes, use overlap + area minimization (R* criterion).
    /// Otherwise, use area enlargement minimization (Guttman criterion).
    fn choose_child(&self, node_idx: usize, bbox: BoundingBox, children_are_leaves: bool) -> usize {
        if let Node::Internal { entries } = &self.nodes[node_idx] {
            if children_are_leaves {
                // R* criterion: minimize overlap enlargement, then area
                let mut best_idx = 0;
                let mut best_overlap_increase = f64::INFINITY;
                let mut best_area_increase = f64::INFINITY;
                let mut best_area = f64::INFINITY;

                for (i, entry) in entries.iter().enumerate() {
                    let new_bbox = entry.bbox.expand_to_include(&bbox);

                    // Compute overlap change with all siblings
                    let mut current_overlap = 0.0f64;
                    let mut new_overlap = 0.0f64;
                    for (j, other) in entries.iter().enumerate() {
                        if i != j {
                            current_overlap += entry.bbox.overlap_area(&other.bbox);
                            new_overlap += new_bbox.overlap_area(&other.bbox);
                        }
                    }
                    let overlap_increase = new_overlap - current_overlap;
                    let area_increase = new_bbox.area() - entry.bbox.area();

                    if overlap_increase < best_overlap_increase
                        || (overlap_increase == best_overlap_increase
                            && area_increase < best_area_increase)
                        || (overlap_increase == best_overlap_increase
                            && area_increase == best_area_increase
                            && entry.bbox.area() < best_area)
                    {
                        best_overlap_increase = overlap_increase;
                        best_area_increase = area_increase;
                        best_area = entry.bbox.area();
                        best_idx = i;
                    }
                }

                entries[best_idx].child
            } else {
                // Guttman: minimize area enlargement
                let mut best_child = entries[0].child;
                let mut best_increase = f64::INFINITY;
                let mut best_area = f64::INFINITY;

                for entry in entries {
                    let increase = entry.bbox.expansion_needed(&bbox);
                    if increase < best_increase
                        || (increase == best_increase && entry.bbox.area() < best_area)
                    {
                        best_increase = increase;
                        best_area = entry.bbox.area();
                        best_child = entry.child;
                    }
                }
                best_child
            }
        } else {
            panic!("choose_child called on leaf node")
        }
    }

    /// Handle overflows along a path from leaf to root.
    fn handle_overflow_path(&mut self, path: Vec<usize>, did_reinsert: &mut Vec<bool>) {
        let tree_height = self.height;

        for (level, &node_idx) in path.iter().rev().enumerate() {
            let actual_level = tree_height - level;
            if self.nodes[node_idx].entry_count() <= MAX_ENTRIES {
                // No overflow; update MBRs upwards
                self.update_mbr_upwards(&path[..path.len() - level]);
                return;
            }

            // Overflow treatment: forced reinsert first time at this level
            if !did_reinsert[actual_level] {
                did_reinsert[actual_level] = true;
                self.forced_reinsert(node_idx, actual_level, did_reinsert);
                self.update_mbr_upwards(&path[..path.len() - level]);
                return;
            }

            // Split
            let (new_node, new_bbox) = self.split_node(node_idx);
            let new_node_idx = self.nodes.len();
            self.nodes.push(new_node);

            if level + 1 == path.len() {
                // We're at the root — need to grow the tree
                let old_root_bbox = self.nodes[0]
                    .mbr()
                    .unwrap_or(BoundingBox::from_point(0.0, 0.0));
                let old_root = std::mem::replace(
                    &mut self.nodes[0],
                    Node::Internal {
                        entries: Vec::new(),
                    },
                );

                let old_idx = self.nodes.len();
                self.nodes.push(old_root);

                if let Node::Internal { entries } = &mut self.nodes[0] {
                    entries.push(BranchEntry {
                        bbox: old_root_bbox,
                        child: old_idx,
                    });
                    entries.push(BranchEntry {
                        bbox: new_bbox,
                        child: new_node_idx,
                    });
                }

                did_reinsert.push(false);
                self.height += 1;
                return;
            } else {
                // Add new node to the parent
                let parent_idx = path[path.len() - level - 2];
                if let Node::Internal { entries } = &mut self.nodes[parent_idx] {
                    entries.push(BranchEntry {
                        bbox: new_bbox,
                        child: new_node_idx,
                    });
                }
            }
        }

        // Update MBRs from root
        self.update_mbr_upwards(&path);
    }

    /// Update MBR values from the leaf back up to the root.
    fn update_mbr_upwards(&mut self, path: &[usize]) {
        if path.len() < 2 {
            return;
        }
        for i in (0..path.len() - 1).rev() {
            let child_idx = path[i + 1];
            let child_mbr = self.nodes[child_idx].mbr();
            let parent_idx = path[i];
            if let Node::Internal { entries } = &mut self.nodes[parent_idx] {
                for e in entries.iter_mut() {
                    if e.child == child_idx {
                        if let Some(mbr) = child_mbr {
                            e.bbox = mbr;
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Forced reinsert: remove `p` entries from `node_idx` and reinsert them.
    ///
    /// Entries furthest from the center are removed and reinserted (R* strategy).
    fn forced_reinsert(&mut self, node_idx: usize, level: usize, did_reinsert: &mut Vec<bool>) {
        // Compute node center
        let center = self.nodes[node_idx]
            .mbr()
            .map(|b| b.center())
            .unwrap_or((0.0, 0.0));

        // Determine which entries to reinsert, releasing the borrow before calling self methods.
        let is_leaf = matches!(self.nodes[node_idx], Node::Leaf { .. });

        if is_leaf {
            // Extract and reinsert leaf entries furthest from center.
            let to_reinsert_owned: Vec<(BoundingBox, T)> = {
                if let Node::Leaf { entries } = &mut self.nodes[node_idx] {
                    entries.sort_by(|a, b| {
                        let ca = a.bbox.center();
                        let cb = b.bbox.center();
                        let da = (ca.0 - center.0).powi(2) + (ca.1 - center.1).powi(2);
                        let db = (cb.0 - center.0).powi(2) + (cb.1 - center.1).powi(2);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let to_reinsert: Vec<LeafEntry<T>> =
                        entries.split_off(entries.len().saturating_sub(REINSERT_COUNT));
                    to_reinsert.into_iter().map(|e| (e.bbox, e.value)).collect()
                } else {
                    Vec::new()
                }
            }; // borrow of self.nodes[node_idx] ends here

            for (bbox, value) in to_reinsert_owned {
                self.insert_at_level(bbox, value, level, did_reinsert);
                // These entries were already in the tree (moved, not added);
                // do NOT touch self.size here — size is managed only by the public insert().
            }
        } else {
            // Extract child indices to reinsert, releasing the borrow before calling self methods.
            let children_to_reinsert: Vec<usize> = {
                if let Node::Internal { entries } = &mut self.nodes[node_idx] {
                    entries.sort_by(|a, b| {
                        let ca = a.bbox.center();
                        let cb = b.bbox.center();
                        let da = (ca.0 - center.0).powi(2) + (ca.1 - center.1).powi(2);
                        let db = (cb.0 - center.0).powi(2) + (cb.1 - center.1).powi(2);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let to_reinsert: Vec<BranchEntry> =
                        entries.split_off(entries.len().saturating_sub(REINSERT_COUNT));
                    to_reinsert.into_iter().map(|br| br.child).collect()
                } else {
                    Vec::new()
                }
            }; // borrow of self.nodes[node_idx] ends here

            // Reinsert subtrees - self is now freely borrowable again.
            // These leaf entries were already counted; do NOT adjust self.size.
            for child_idx in children_to_reinsert {
                let leaves = self.collect_leaves(child_idx);
                for (bbox, value) in leaves {
                    self.insert_at_level(bbox, value, level, did_reinsert);
                }
            }
        }
    }

    /// Recursively collect all leaf entries under a subtree.
    fn collect_leaves(&self, node_idx: usize) -> Vec<(BoundingBox, T)> {
        match &self.nodes[node_idx] {
            Node::Leaf { entries } => entries.iter().map(|e| (e.bbox, e.value.clone())).collect(),
            Node::Internal { entries } => entries
                .iter()
                .flat_map(|e| self.collect_leaves(e.child))
                .collect(),
        }
    }

    // ------------------------------------------------------------------
    // Split (R* split algorithm)
    // ------------------------------------------------------------------

    /// Split an overflow node into two nodes.
    ///
    /// Returns the new node and its MBR.
    fn split_node(&mut self, node_idx: usize) -> (Node<T>, BoundingBox) {
        match &self.nodes[node_idx] {
            Node::Leaf { .. } => self.split_leaf(node_idx),
            Node::Internal { .. } => self.split_internal(node_idx),
        }
    }

    /// Split a leaf node.
    fn split_leaf(&mut self, node_idx: usize) -> (Node<T>, BoundingBox) {
        let entries = if let Node::Leaf { entries } = &self.nodes[node_idx] {
            entries.clone()
        } else {
            unreachable!()
        };

        let (split_axis, split_idx, sorted) = choose_leaf_split(&entries);
        let _ = split_axis; // axis is determined internally

        let (group1, group2): (Vec<_>, Vec<_>) = sorted
            .into_iter()
            .enumerate()
            .partition(|(i, _)| *i < split_idx);

        let group1_entries: Vec<LeafEntry<T>> = group1.into_iter().map(|(_, e)| e).collect();
        let group2_entries: Vec<LeafEntry<T>> = group2.into_iter().map(|(_, e)| e).collect();

        let new_mbr = group2_entries
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b))
            .unwrap_or(BoundingBox::from_point(0.0, 0.0));

        let new_node = Node::Leaf {
            entries: group2_entries,
        };
        self.nodes[node_idx] = Node::Leaf {
            entries: group1_entries,
        };

        (new_node, new_mbr)
    }

    /// Split an internal node.
    fn split_internal(&mut self, node_idx: usize) -> (Node<T>, BoundingBox) {
        let entries = if let Node::Internal { entries } = &self.nodes[node_idx] {
            entries.clone()
        } else {
            unreachable!()
        };

        let (split_axis, split_idx, sorted) = choose_branch_split(&entries);
        let _ = split_axis;

        let (group1, group2): (Vec<_>, Vec<_>) = sorted
            .into_iter()
            .enumerate()
            .partition(|(i, _)| *i < split_idx);

        let group1_entries: Vec<BranchEntry> = group1.into_iter().map(|(_, e)| e).collect();
        let group2_entries: Vec<BranchEntry> = group2.into_iter().map(|(_, e)| e).collect();

        let new_mbr = group2_entries
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b))
            .unwrap_or(BoundingBox::from_point(0.0, 0.0));

        let new_node = Node::Internal {
            entries: group2_entries,
        };
        self.nodes[node_idx] = Node::Internal {
            entries: group1_entries,
        };

        (new_node, new_mbr)
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    /// Return references to all stored values whose bounding boxes intersect `query`.
    pub fn search(&self, query: &BoundingBox) -> Vec<&T> {
        let mut results = Vec::new();
        self.search_node(0, query, &mut results);
        results
    }

    fn search_node<'a>(&'a self, node_idx: usize, query: &BoundingBox, results: &mut Vec<&'a T>) {
        match &self.nodes[node_idx] {
            Node::Leaf { entries } => {
                for e in entries {
                    if e.bbox.intersects(query) {
                        results.push(&e.value);
                    }
                }
            }
            Node::Internal { entries } => {
                for e in entries {
                    if e.bbox.intersects(query) {
                        self.search_node(e.child, query, results);
                    }
                }
            }
        }
    }

    /// Count entries whose bounding boxes intersect `query`.
    pub fn count_within(&self, query: &BoundingBox) -> usize {
        self.count_node(0, query)
    }

    fn count_node(&self, node_idx: usize, query: &BoundingBox) -> usize {
        match &self.nodes[node_idx] {
            Node::Leaf { entries } => entries.iter().filter(|e| e.bbox.intersects(query)).count(),
            Node::Internal { entries } => entries
                .iter()
                .filter(|e| e.bbox.intersects(query))
                .map(|e| self.count_node(e.child, query))
                .sum(),
        }
    }

    // ------------------------------------------------------------------
    // k-NN
    // ------------------------------------------------------------------

    /// Find the `k` nearest stored values to point `(px, py)`.
    ///
    /// Returns a vector of `(&value, distance)` sorted by ascending distance.
    pub fn nearest_neighbor(&self, px: f64, py: f64, k: usize) -> Vec<(&T, f64)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        // Best-first priority queue: (min_dist_sq, node_or_leaf)
        // We use a manual sorted list for simplicity; for large k, a BinaryHeap would be better.
        let mut candidates: Vec<(&T, f64)> = Vec::new();
        self.knn_node(0, px, py, k, &mut candidates);
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    fn knn_node<'a>(
        &'a self,
        node_idx: usize,
        px: f64,
        py: f64,
        k: usize,
        best: &mut Vec<(&'a T, f64)>,
    ) {
        // Compute current worst distance among k-best
        let worst_dist_sq = if best.len() < k {
            f64::INFINITY
        } else {
            best.iter().map(|(_, d)| d * d).fold(0.0f64, f64::max)
        };

        match &self.nodes[node_idx] {
            Node::Leaf { entries } => {
                for e in entries {
                    let min_dsq = e.bbox.min_sq_dist_to_point(px, py);
                    if min_dsq <= worst_dist_sq {
                        let dist = min_dsq.sqrt();
                        best.push((&e.value, dist));
                    }
                }
                // Prune to k best
                best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                best.truncate(k);
            }
            Node::Internal { entries } => {
                // Sort children by min distance and prune
                let mut sorted_children: Vec<(usize, f64)> = entries
                    .iter()
                    .map(|e| {
                        let dsq = e.bbox.min_sq_dist_to_point(px, py);
                        (e.child, dsq)
                    })
                    .collect();
                sorted_children
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                for (child_idx, child_min_dsq) in sorted_children {
                    let current_worst = if best.len() < k {
                        f64::INFINITY
                    } else {
                        let w = best.iter().map(|(_, d)| d).cloned().fold(0.0f64, f64::max);
                        w * w
                    };
                    if child_min_dsq > current_worst {
                        break; // prune remaining children
                    }
                    self.knn_node(child_idx, px, py, k, best);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Delete
    // ------------------------------------------------------------------

    /// Remove the first entry whose bounding box intersects `bbox` and whose value equals `value`.
    ///
    /// Returns `true` if an entry was removed.
    pub fn delete(&mut self, bbox: &BoundingBox, value: &T) -> bool
    where
        T: PartialEq,
    {
        let removed = self.delete_from_node(0, bbox, value);
        if removed {
            self.size = self.size.saturating_sub(1);
            self.condense_tree();
        }
        removed
    }

    fn delete_from_node(&mut self, node_idx: usize, bbox: &BoundingBox, value: &T) -> bool
    where
        T: PartialEq,
    {
        match &mut self.nodes[node_idx] {
            Node::Leaf { entries } => {
                if let Some(pos) = entries
                    .iter()
                    .position(|e| e.bbox.intersects(bbox) && e.value == *value)
                {
                    entries.remove(pos);
                    return true;
                }
                false
            }
            Node::Internal { entries } => {
                let children: Vec<usize> = entries
                    .iter()
                    .filter(|e| e.bbox.intersects(bbox))
                    .map(|e| e.child)
                    .collect();
                for child_idx in children {
                    if self.delete_from_node(child_idx, bbox, value) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Condense tree: remove underfull nodes and reinsert their entries.
    fn condense_tree(&mut self) {
        // Simple approach: collect underfull leaf entries and reinsert them
        let mut orphans: Vec<(BoundingBox, T)> = Vec::new();
        self.collect_underfull_leaves(0, &mut orphans);

        let n = orphans.len();
        if n > 0 {
            // Size was already decremented for the deleted item;
            // we need to account for reinserted orphans
            self.size += n;
            for (bbox, value) in orphans {
                self.insert(bbox, value);
                self.size -= 1; // insert increments; compensate
            }
        }
    }

    fn collect_underfull_leaves(&mut self, node_idx: usize, orphans: &mut Vec<(BoundingBox, T)>) {
        let children: Vec<usize> = match &self.nodes[node_idx] {
            Node::Internal { entries } => entries.iter().map(|e| e.child).collect(),
            Node::Leaf { .. } => return,
        };

        for child in children {
            if self.nodes[child].entry_count() < MIN_ENTRIES {
                let leaves = self.collect_leaves(child);
                orphans.extend(leaves);
                // Remove from parent
                if let Node::Internal { entries } = &mut self.nodes[node_idx] {
                    entries.retain(|e| e.child != child);
                }
            } else {
                self.collect_underfull_leaves(child, orphans);
            }
        }
    }
}

impl<T: Clone> Default for RTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// R* split heuristic helpers (axis + split index selection)
// ---------------------------------------------------------------------------

/// Determine best split axis and index for leaf entries.
///
/// Returns `(axis, split_index, sorted_entries)`.
fn choose_leaf_split<T: Clone>(entries: &[LeafEntry<T>]) -> (usize, usize, Vec<LeafEntry<T>>) {
    let axis = choose_split_axis_leaf(entries);
    let mut sorted = entries.to_vec();

    if axis == 0 {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_x
                .partial_cmp(&b.bbox.min_x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_y
                .partial_cmp(&b.bbox.min_y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let split_idx = choose_split_index_leaf(&sorted);
    (axis, split_idx, sorted)
}

/// Choose the split axis for leaf entries (minimum total margin sum).
fn choose_split_axis_leaf<T: Clone>(entries: &[LeafEntry<T>]) -> usize {
    let margin_x = compute_margin_sum_leaf(entries, true);
    let margin_y = compute_margin_sum_leaf(entries, false);
    if margin_x <= margin_y {
        0
    } else {
        1
    }
}

fn compute_margin_sum_leaf<T: Clone>(entries: &[LeafEntry<T>], by_x: bool) -> f64 {
    let mut sorted = entries.to_vec();
    if by_x {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_x
                .partial_cmp(&b.bbox.min_x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_y
                .partial_cmp(&b.bbox.min_y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut margin_sum = 0.0;
    let n = sorted.len();
    if n < MIN_ENTRIES * 2 {
        return 0.0;
    }
    for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
        let g1_mbr = sorted[..k]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        let g2_mbr = sorted[k..]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        if let (Some(b1), Some(b2)) = (g1_mbr, g2_mbr) {
            margin_sum += b1.margin() + b2.margin();
        }
    }
    margin_sum
}

/// Choose the split index that minimizes overlap, then area.
fn choose_split_index_leaf<T: Clone>(sorted: &[LeafEntry<T>]) -> usize {
    let mut best_idx = MIN_ENTRIES;
    let mut best_overlap = f64::INFINITY;
    let mut best_area = f64::INFINITY;

    let n = sorted.len();
    if n < MIN_ENTRIES * 2 {
        return MIN_ENTRIES.min(n / 2).max(1);
    }

    for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
        let g1 = sorted[..k]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        let g2 = sorted[k..]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        if let (Some(b1), Some(b2)) = (g1, g2) {
            let overlap = b1.overlap_area(&b2);
            let area = b1.area() + b2.area();
            if overlap < best_overlap || (overlap == best_overlap && area < best_area) {
                best_overlap = overlap;
                best_area = area;
                best_idx = k;
            }
        }
    }
    best_idx
}

// Branch split analogues

fn choose_branch_split(entries: &[BranchEntry]) -> (usize, usize, Vec<BranchEntry>) {
    let axis = choose_split_axis_branch(entries);
    let mut sorted = entries.to_vec();

    if axis == 0 {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_x
                .partial_cmp(&b.bbox.min_x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_y
                .partial_cmp(&b.bbox.min_y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let split_idx = choose_split_index_branch(&sorted);
    (axis, split_idx, sorted)
}

fn choose_split_axis_branch(entries: &[BranchEntry]) -> usize {
    let mx = compute_margin_sum_branch(entries, true);
    let my = compute_margin_sum_branch(entries, false);
    if mx <= my {
        0
    } else {
        1
    }
}

fn compute_margin_sum_branch(entries: &[BranchEntry], by_x: bool) -> f64 {
    let mut sorted = entries.to_vec();
    if by_x {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_x
                .partial_cmp(&b.bbox.min_x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted.sort_by(|a, b| {
            a.bbox
                .min_y
                .partial_cmp(&b.bbox.min_y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut margin_sum = 0.0;
    let n = sorted.len();
    if n < MIN_ENTRIES * 2 {
        return 0.0;
    }
    for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
        let g1 = sorted[..k]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        let g2 = sorted[k..]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        if let (Some(b1), Some(b2)) = (g1, g2) {
            margin_sum += b1.margin() + b2.margin();
        }
    }
    margin_sum
}

fn choose_split_index_branch(sorted: &[BranchEntry]) -> usize {
    let mut best_idx = MIN_ENTRIES;
    let mut best_overlap = f64::INFINITY;
    let mut best_area = f64::INFINITY;

    let n = sorted.len();
    if n < MIN_ENTRIES * 2 {
        return MIN_ENTRIES.min(n / 2).max(1);
    }

    for k in MIN_ENTRIES..=(n - MIN_ENTRIES) {
        let g1 = sorted[..k]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        let g2 = sorted[k..]
            .iter()
            .map(|e| e.bbox)
            .reduce(|a, b| a.expand_to_include(&b));
        if let (Some(b1), Some(b2)) = (g1, g2) {
            let overlap = b1.overlap_area(&b2);
            let area = b1.area() + b2.area();
            if overlap < best_overlap || (overlap == best_overlap && area < best_area) {
                best_overlap = overlap;
                best_area = area;
                best_idx = k;
            }
        }
    }
    best_idx
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(x: f64, y: f64) -> BoundingBox {
        BoundingBox::new(x, y, x + 1.0, y + 1.0)
    }

    // ---- BoundingBox ----

    #[test]
    fn test_bbox_area() {
        let b = BoundingBox::new(0.0, 0.0, 3.0, 4.0);
        assert!((b.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BoundingBox::new(1.0, 1.0, 3.0, 3.0);
        assert!(a.intersects(&b));
        let c = BoundingBox::new(3.0, 3.0, 5.0, 5.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bbox_contains() {
        let outer = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let inner = BoundingBox::new(2.0, 2.0, 8.0, 8.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_bbox_overlap_area() {
        let a = BoundingBox::new(0.0, 0.0, 4.0, 4.0);
        let b = BoundingBox::new(2.0, 2.0, 6.0, 6.0);
        assert!((a.overlap_area(&b) - 4.0).abs() < 1e-10); // 2x2 overlap
    }

    #[test]
    fn test_bbox_expand_to_include() {
        let a = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BoundingBox::new(2.0, 2.0, 3.0, 3.0);
        let union = a.expand_to_include(&b);
        assert!((union.min_x - 0.0).abs() < 1e-10);
        assert!((union.max_x - 3.0).abs() < 1e-10);
        assert!((union.max_y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_min_sq_dist_inside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.min_sq_dist_to_point(5.0, 5.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_min_sq_dist_outside() {
        let b = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        // Point at (2, 0): distance = 1
        assert!((b.min_sq_dist_to_point(2.0, 0.0) - 1.0).abs() < 1e-10);
        // Point at (2, 2): distance = sqrt(2), sq = 2
        assert!((b.min_sq_dist_to_point(2.0, 2.0) - 2.0).abs() < 1e-10);
    }

    // ---- RTree basic ----

    #[test]
    fn test_insert_and_len() {
        let mut tree: RTree<i32> = RTree::new();
        assert!(tree.is_empty());
        tree.insert(unit_box(0.0, 0.0), 1);
        tree.insert(unit_box(5.0, 5.0), 2);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_search_hit() {
        let mut tree: RTree<i32> = RTree::new();
        tree.insert(unit_box(0.0, 0.0), 100);
        tree.insert(unit_box(10.0, 10.0), 200);

        let results = tree.search(&BoundingBox::new(0.5, 0.5, 1.5, 1.5));
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0], 100);
    }

    #[test]
    fn test_search_miss() {
        let mut tree: RTree<i32> = RTree::new();
        tree.insert(unit_box(0.0, 0.0), 1);

        let results = tree.search(&BoundingBox::new(5.0, 5.0, 6.0, 6.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_count_within() {
        let mut tree: RTree<i32> = RTree::new();
        for i in 0..5 {
            tree.insert(unit_box(i as f64, 0.0), i);
        }
        let count = tree.count_within(&BoundingBox::new(0.0, 0.0, 2.5, 1.0));
        // boxes at x=0,1,2 overlap with query
        assert_eq!(count, 3);
    }

    #[test]
    fn test_bulk_insert_many() {
        let mut tree: RTree<usize> = RTree::new();
        for i in 0..100 {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            tree.insert(unit_box(x, y), i);
        }
        assert_eq!(tree.len(), 100);
        // Query centre of grid
        let found = tree.search(&BoundingBox::new(4.0, 4.0, 6.0, 6.0));
        // Should include boxes at (4,4),(5,4),(4,5),(5,5) plus their extents
        assert!(!found.is_empty());
    }

    #[test]
    fn test_nearest_neighbor_single() {
        let mut tree: RTree<&str> = RTree::new();
        tree.insert(BoundingBox::from_point(3.0, 4.0), "A");
        tree.insert(BoundingBox::from_point(10.0, 10.0), "B");

        let nn = tree.nearest_neighbor(0.0, 0.0, 1);
        assert_eq!(nn.len(), 1);
        assert_eq!(*nn[0].0, "A");
    }

    #[test]
    fn test_nearest_neighbor_k() {
        let mut tree: RTree<i32> = RTree::new();
        tree.insert(BoundingBox::from_point(1.0, 0.0), 1);
        tree.insert(BoundingBox::from_point(2.0, 0.0), 2);
        tree.insert(BoundingBox::from_point(10.0, 0.0), 10);

        let nn = tree.nearest_neighbor(0.0, 0.0, 2);
        assert_eq!(nn.len(), 2);
        let values: Vec<i32> = nn.iter().map(|(v, _)| **v).collect();
        assert!(values.contains(&1));
        assert!(values.contains(&2));
    }

    #[test]
    fn test_delete() {
        let mut tree: RTree<i32> = RTree::new();
        tree.insert(unit_box(0.0, 0.0), 42);
        assert_eq!(tree.len(), 1);
        let removed = tree.delete(&unit_box(0.0, 0.0), &42);
        assert!(removed);
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_delete_not_found() {
        let mut tree: RTree<i32> = RTree::new();
        tree.insert(unit_box(0.0, 0.0), 1);
        let removed = tree.delete(&unit_box(5.0, 5.0), &1);
        assert!(!removed);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_search_all_types() {
        let mut tree: RTree<String> = RTree::new();
        let items = ["alpha", "beta", "gamma", "delta", "epsilon"];
        for (i, name) in items.iter().enumerate() {
            tree.insert(unit_box(i as f64 * 3.0, 0.0), name.to_string());
        }
        // Wide query should find all
        let results = tree.search(&BoundingBox::new(-1.0, -1.0, 100.0, 100.0));
        assert_eq!(results.len(), items.len());
    }
}
