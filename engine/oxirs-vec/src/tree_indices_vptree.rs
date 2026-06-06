//! VP-Tree (Vantage Point Tree) implementation for metric-space search.
//!
//! A VP-tree partitions points by their distance from a randomly chosen
//! vantage point, supporting efficient search in arbitrary metric spaces.

use crate::tree_indices_types::{SearchResult, TreeIndexConfig};
use crate::Vector;
use anyhow::Result;
use scirs2_core::random::{Random, Rng, RngExt};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// VP-Tree (Vantage Point Tree) implementation
pub struct VpTree {
    pub(crate) root: Option<Box<VpNode>>,
    pub(crate) data: Vec<(String, Vector)>,
    pub(crate) config: TreeIndexConfig,
}

pub(crate) struct VpNode {
    /// Vantage point index
    vantage_point: usize,
    /// Median distance from vantage point
    median_distance: f32,
    /// Points closer than median
    inside: Option<Box<VpNode>>,
    /// Points farther than median
    outside: Option<Box<VpNode>>,
    /// Indices for leaf nodes
    indices: Vec<usize>,
}

impl VpTree {
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
        let mut rng = if let Some(seed) = self.config.random_seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        self.root = Some(Box::new(self.build_node(indices, &mut rng)?));
        Ok(())
    }

    fn build_node<R: Rng>(&self, indices: Vec<usize>, rng: &mut R) -> Result<VpNode> {
        self.build_node_safe(indices, rng, 0)
    }

    #[allow(deprecated)]
    fn build_node_safe<R: Rng>(
        &self,
        mut indices: Vec<usize>,
        rng: &mut R,
        depth: usize,
    ) -> Result<VpNode> {
        // Note: Using manual random selection instead of SliceRandom

        // CRITICAL: Extremely strict depth and size limits to prevent stack overflow
        // For very small datasets or deep recursion, immediately create leaf nodes
        let max_depth = 30; // Conservative depth limit

        // Aggressive leaf node creation for small datasets
        if indices.len() <= self.config.max_leaf_size
            || indices.len() <= 2  // Changed from <= 1 to <= 2 for extra safety
            || depth >= max_depth
        {
            return Ok(VpNode {
                vantage_point: if indices.is_empty() { 0 } else { indices[0] },
                median_distance: 0.0,
                inside: None,
                outside: None,
                indices,
            });
        }

        // Choose random vantage point - simplified to avoid potential issues
        let vp_idx = if indices.len() > 1 {
            rng.random_range(0..indices.len())
        } else {
            0
        };
        let vantage_point = indices[vp_idx];
        indices.remove(vp_idx);

        // Calculate distances from vantage point
        let vp_data = &self.data[vantage_point].1.as_f32();
        let mut distances: Vec<(f32, usize)> = indices
            .iter()
            .map(|&idx| {
                let point = &self.data[idx].1.as_f32();
                let dist = self.config.distance_metric.distance(vp_data, point);
                (dist, idx)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let median_idx = distances.len() / 2;
        let median_distance = distances[median_idx].0;

        let inside_indices: Vec<usize> = distances[..median_idx]
            .iter()
            .map(|(_, idx)| *idx)
            .collect();

        let outside_indices: Vec<usize> = distances[median_idx..]
            .iter()
            .map(|(_, idx)| *idx)
            .collect();

        // Prevent creating empty partitions - create leaf instead
        if inside_indices.is_empty() || outside_indices.is_empty() {
            return Ok(VpNode {
                vantage_point: if indices.is_empty() { 0 } else { indices[0] },
                median_distance: 0.0,
                inside: None,
                outside: None,
                indices,
            });
        }

        let inside = Some(Box::new(self.build_node_safe(
            inside_indices,
            rng,
            depth + 1,
        )?));
        let outside = Some(Box::new(self.build_node_safe(
            outside_indices,
            rng,
            depth + 1,
        )?));

        Ok(VpNode {
            vantage_point,
            median_distance,
            inside,
            outside,
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
            f32::INFINITY,
        );

        let mut results: Vec<(usize, f32)> =
            heap.into_iter().map(|r| (r.index, r.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn search_node(
        &self,
        node: &VpNode,
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<SearchResult>,
        tau: f32,
    ) -> f32 {
        let mut tau = tau;

        if !node.indices.is_empty() {
            // Leaf node
            for &idx in &node.indices {
                let point = &self.data[idx].1.as_f32();
                let dist = self.config.distance_metric.distance(query, point);

                if dist < tau {
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

                    if heap.len() >= k {
                        tau = heap.peek().expect("heap should have k elements").distance;
                    }
                }
            }
            return tau;
        }

        // Calculate distance to vantage point
        let vp_data = &self.data[node.vantage_point].1.as_f32();
        let dist_to_vp = self.config.distance_metric.distance(query, vp_data);

        // Consider vantage point itself
        if dist_to_vp < tau {
            if heap.len() < k {
                heap.push(SearchResult {
                    index: node.vantage_point,
                    distance: dist_to_vp,
                });
            } else if dist_to_vp < heap.peek().expect("heap should have k elements").distance {
                heap.pop();
                heap.push(SearchResult {
                    index: node.vantage_point,
                    distance: dist_to_vp,
                });
            }

            if heap.len() >= k {
                tau = heap.peek().expect("heap should have k elements").distance;
            }
        }

        // Search children
        if dist_to_vp < node.median_distance {
            // Search inside first
            if let Some(inside) = &node.inside {
                tau = self.search_node(inside, query, k, heap, tau);
            }

            // Check if we need to search outside
            if dist_to_vp + tau >= node.median_distance {
                if let Some(outside) = &node.outside {
                    tau = self.search_node(outside, query, k, heap, tau);
                }
            }
        } else {
            // Search outside first
            if let Some(outside) = &node.outside {
                tau = self.search_node(outside, query, k, heap, tau);
            }

            // Check if we need to search inside
            if dist_to_vp - tau <= node.median_distance {
                if let Some(inside) = &node.inside {
                    tau = self.search_node(inside, query, k, heap, tau);
                }
            }
        }

        tau
    }
}
