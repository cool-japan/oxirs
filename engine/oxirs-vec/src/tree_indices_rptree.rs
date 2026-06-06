//! Random Projection Tree implementation for approximate nearest-neighbor search.
//!
//! A random projection tree recursively partitions points using randomly
//! generated projection vectors, providing fast approximate search.

use crate::tree_indices_types::{SearchResult, TreeIndexConfig};
use crate::Vector;
use anyhow::Result;
use oxirs_core::simd::SimdOps;
use scirs2_core::random::{Random, Rng, RngExt};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Random Projection Tree implementation
pub struct RandomProjectionTree {
    pub(crate) root: Option<Box<RpNode>>,
    pub(crate) data: Vec<(String, Vector)>,
    pub(crate) config: TreeIndexConfig,
}

pub(crate) struct RpNode {
    /// Random projection vector
    projection: Vec<f32>,
    /// Projection threshold
    threshold: f32,
    /// Left child (projection <= threshold)
    left: Option<Box<RpNode>>,
    /// Right child (projection > threshold)
    right: Option<Box<RpNode>>,
    /// Indices for leaf nodes
    indices: Vec<usize>,
}

impl RandomProjectionTree {
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
        let dimensions = self.data[0].1.dimensions;

        let mut rng = if let Some(seed) = self.config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        self.root = Some(Box::new(self.build_node(indices, dimensions, &mut rng)?));
        Ok(())
    }

    fn build_node<R: Rng>(
        &self,
        indices: Vec<usize>,
        dimensions: usize,
        rng: &mut R,
    ) -> Result<RpNode> {
        self.build_node_safe(indices, dimensions, rng, 0)
    }

    #[allow(deprecated)]
    fn build_node_safe<R: Rng>(
        &self,
        indices: Vec<usize>,
        dimensions: usize,
        rng: &mut R,
        depth: usize,
    ) -> Result<RpNode> {
        // Very strict stack overflow prevention - similar to BallTree approach
        if indices.len() <= self.config.max_leaf_size || indices.len() <= 2 || depth >= 5 {
            return Ok(RpNode {
                projection: Vec::new(),
                threshold: 0.0,
                left: None,
                right: None,
                indices,
            });
        }

        // Generate random projection vector
        let projection: Vec<f32> = (0..dimensions)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        // Normalize projection vector
        let norm = (projection.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        let projection: Vec<f32> = if norm > 0.0 {
            projection.iter().map(|&x| x / norm).collect()
        } else {
            projection
        };

        // Project all points
        let mut projections: Vec<(f32, usize)> = indices
            .iter()
            .map(|&idx| {
                let point = &self.data[idx].1.as_f32();
                let proj_val = f32::dot(point, &projection);
                (proj_val, idx)
            })
            .collect();

        projections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Choose median as threshold
        let median_idx = projections.len() / 2;
        let threshold = projections[median_idx].0;

        let left_indices: Vec<usize> = projections[..median_idx]
            .iter()
            .map(|(_, idx)| *idx)
            .collect();

        let right_indices: Vec<usize> = projections[median_idx..]
            .iter()
            .map(|(_, idx)| *idx)
            .collect();

        // Prevent creating empty partitions - create leaf instead
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(RpNode {
                projection: Vec::new(),
                threshold: 0.0,
                left: None,
                right: None,
                indices,
            });
        }

        let left = Some(Box::new(self.build_node_safe(
            left_indices,
            dimensions,
            rng,
            depth + 1,
        )?));
        let right = Some(Box::new(self.build_node_safe(
            right_indices,
            dimensions,
            rng,
            depth + 1,
        )?));

        Ok(RpNode {
            projection,
            threshold,
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
        node: &RpNode,
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

        // Project query
        let query_projection = f32::dot(query, &node.projection);

        // Determine which side to search first
        let go_left = query_projection <= node.threshold;

        let (first, second) = if go_left {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search both sides (random projections don't provide distance bounds)
        if let Some(child) = first {
            self.search_node(child, query, k, heap);
        }

        if let Some(child) = second {
            self.search_node(child, query, k, heap);
        }
    }
}
