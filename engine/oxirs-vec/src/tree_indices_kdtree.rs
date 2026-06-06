//! KD-Tree implementation for nearest-neighbor search.
//!
//! A KD-tree is a classic space-partitioning binary tree that splits points
//! along alternating axis-aligned dimensions.

use crate::tree_indices_types::{SearchResult, TreeIndexConfig};
use crate::Vector;
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// KD-Tree implementation
pub struct KdTree {
    pub(crate) root: Option<Box<KdNode>>,
    pub(crate) data: Vec<(String, Vector)>,
    pub(crate) config: TreeIndexConfig,
}

pub(crate) struct KdNode {
    /// Split dimension
    split_dim: usize,
    /// Split value
    split_value: f32,
    /// Left child (values <= split_value)
    left: Option<Box<KdNode>>,
    /// Right child (values > split_value)
    right: Option<Box<KdNode>>,
    /// Indices for leaf nodes
    indices: Vec<usize>,
}

impl KdTree {
    pub fn new(config: TreeIndexConfig) -> Self {
        Self {
            root: None,
            data: Vec::new(),
            config,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let indices: Vec<usize> = (0..self.data.len()).collect();
        let points: Vec<Vec<f32>> = self.data.iter().map(|(_, v)| v.as_f32()).collect();

        self.root = Some(Box::new(self.build_node(&points, indices, 0)?));
        Ok(())
    }

    fn build_node(&self, points: &[Vec<f32>], indices: Vec<usize>, depth: usize) -> Result<KdNode> {
        // Reasonable stack overflow prevention with proper depth limit
        let max_depth = if !self.data.is_empty() {
            ((self.data.len() as f32).log2() * 2.0) as usize + 10
        } else {
            50
        };

        if indices.len() <= self.config.max_leaf_size || indices.len() <= 1 || depth >= max_depth {
            return Ok(KdNode {
                split_dim: 0,
                split_value: 0.0,
                left: None,
                right: None,
                indices,
            });
        }

        let dimensions = points[0].len();
        let split_dim = depth % dimensions;

        // Find median along split dimension
        let mut values: Vec<(f32, usize)> = indices
            .iter()
            .map(|&idx| (points[idx][split_dim], idx))
            .collect();

        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let median_idx = values.len() / 2;
        let split_value = values[median_idx].0;

        let left_indices: Vec<usize> = values[..median_idx].iter().map(|(_, idx)| *idx).collect();

        let right_indices: Vec<usize> = values[median_idx..].iter().map(|(_, idx)| *idx).collect();

        // Prevent creating empty partitions - create leaf instead
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(KdNode {
                split_dim: 0,
                split_value: 0.0,
                left: None,
                right: None,
                indices,
            });
        }

        let left = Some(Box::new(self.build_node(
            points,
            left_indices,
            depth + 1,
        )?));

        let right = Some(Box::new(self.build_node(
            points,
            right_indices,
            depth + 1,
        )?));

        Ok(KdNode {
            split_dim,
            split_value,
            left,
            right,
            indices: Vec::new(),
        })
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut heap = BinaryHeap::new();
        self.search_node(
            self.root
                .as_ref()
                .expect("tree should have root after build"),
            query,
            k,
            &mut heap,
        );

        let mut results: Vec<(usize, f32)> =
            heap.into_iter().map(|r| (r.index, r.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn search_node(
        &self,
        node: &KdNode,
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<SearchResult>,
    ) {
        if !node.indices.is_empty() {
            // Leaf node
            for &idx in &node.indices {
                let point = &self.data[idx].1.as_f32();
                let dist = self.config.distance_metric.distance(query, point);

                if heap.len() < k {
                    heap.push(SearchResult {
                        index: idx,
                        distance: dist,
                    });
                } else if dist < heap.peek().expect("heap should have k elements").distance {
                    heap.pop();
                    heap.push(SearchResult {
                        index: idx,
                        distance: dist,
                    });
                }
            }
            return;
        }

        // Determine which side to search first
        let go_left = query[node.split_dim] <= node.split_value;

        let (first, second) = if go_left {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the nearer side first
        if let Some(child) = first {
            self.search_node(child, query, k, heap);
        }

        // Check if we need to search the other side
        if heap.len() < k || {
            let split_dist = (query[node.split_dim] - node.split_value).abs();
            split_dist < heap.peek().expect("heap should have k elements").distance
        } {
            if let Some(child) = second {
                self.search_node(child, query, k, heap);
            }
        }
    }
}
