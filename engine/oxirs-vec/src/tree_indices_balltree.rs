//! Ball Tree implementation for nearest-neighbor search.
//!
//! A ball tree recursively partitions points into nested hyperspheres ("balls"),
//! making it efficient for arbitrary distance metrics.

use crate::tree_indices_types::{SearchResult, TreeIndexConfig};
use crate::Vector;
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Ball Tree implementation
pub struct BallTree {
    pub(crate) root: Option<Box<BallNode>>,
    pub(crate) data: Vec<(String, Vector)>,
    pub(crate) config: TreeIndexConfig,
}

#[derive(Clone)]
pub(crate) struct BallNode {
    /// Center of the ball
    center: Vec<f32>,
    /// Radius of the ball
    radius: f32,
    /// Left child
    left: Option<Box<BallNode>>,
    /// Right child
    right: Option<Box<BallNode>>,
    /// Indices of points in this node (for leaf nodes)
    indices: Vec<usize>,
}

impl BallTree {
    pub fn new(config: TreeIndexConfig) -> Self {
        Self {
            root: None,
            data: Vec::new(),
            config,
        }
    }

    /// Build the tree from data with conservative depth limits to prevent stack overflow
    ///
    /// Note: Tree indices work best with moderate dataset sizes (< 100K points).
    /// For larger datasets, consider using HNSW, IVF, or LSH indices instead.
    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let indices: Vec<usize> = (0..self.data.len()).collect();
        let points: Vec<Vec<f32>> = self.data.iter().map(|(_, v)| v.as_f32()).collect();

        self.root = Some(Box::new(self.build_node_safe(&points, indices, 0)?));
        Ok(())
    }

    /// Conservative recursive construction with strict depth limits
    fn build_node_safe(
        &self,
        points: &[Vec<f32>],
        indices: Vec<usize>,
        depth: usize,
    ) -> Result<BallNode> {
        // VERY conservative depth limit to prevent stack overflow
        // Limit depth to 20 for safety (can handle ~1M points with leaf_size=10)
        const MAX_DEPTH: usize = 20;

        // Force leaf creation if:
        // 1. At or below leaf size
        // 2. Only 1 or 2 points left
        // 3. Reached maximum safe depth
        if indices.len() <= self.config.max_leaf_size || indices.len() <= 2 || depth >= MAX_DEPTH {
            let center = self.compute_centroid(points, &indices);
            let radius = self.compute_radius(points, &indices, &center);
            return Ok(BallNode {
                center,
                radius,
                left: None,
                right: None,
                indices,
            });
        }

        // Find split dimension
        let split_dim = self.find_split_dimension(points, &indices);
        let (left_indices, right_indices) = self.partition_indices(points, &indices, split_dim);

        // Prevent empty partitions - create leaf instead
        if left_indices.is_empty() || right_indices.is_empty() {
            let center = self.compute_centroid(points, &indices);
            let radius = self.compute_radius(points, &indices, &center);
            return Ok(BallNode {
                center,
                radius,
                left: None,
                right: None,
                indices,
            });
        }

        // Recursively build children (limited by MAX_DEPTH)
        let left_node = self.build_node_safe(points, left_indices, depth + 1)?;
        let right_node = self.build_node_safe(points, right_indices, depth + 1)?;

        // Compute bounding ball
        let all_centers = vec![left_node.center.clone(), right_node.center.clone()];
        let center = self.compute_centroid_of_centers(&all_centers);
        let radius = left_node.radius.max(right_node.radius)
            + self
                .config
                .distance_metric
                .distance(&center, &left_node.center);

        Ok(BallNode {
            center,
            radius,
            left: Some(Box::new(left_node)),
            right: Some(Box::new(right_node)),
            indices: Vec::new(),
        })
    }

    fn compute_centroid(&self, points: &[Vec<f32>], indices: &[usize]) -> Vec<f32> {
        let dim = points[0].len();
        let mut centroid = vec![0.0; dim];

        for &idx in indices {
            for (i, &val) in points[idx].iter().enumerate() {
                centroid[i] += val;
            }
        }

        let n = indices.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }

        centroid
    }

    fn compute_radius(&self, points: &[Vec<f32>], indices: &[usize], center: &[f32]) -> f32 {
        indices
            .iter()
            .map(|&idx| self.config.distance_metric.distance(&points[idx], center))
            .fold(0.0f32, f32::max)
    }

    fn find_split_dimension(&self, points: &[Vec<f32>], indices: &[usize]) -> usize {
        let dim = points[0].len();
        let mut max_spread = 0.0;
        let mut split_dim = 0;

        // We need the dimension index `d` to access the d-th component of each point
        #[allow(clippy::needless_range_loop)]
        for d in 0..dim {
            let values: Vec<f32> = indices.iter().map(|&idx| points[idx][d]).collect();

            let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let spread = max_val - min_val;

            if spread > max_spread {
                max_spread = spread;
                split_dim = d;
            }
        }

        split_dim
    }

    fn partition_indices(
        &self,
        points: &[Vec<f32>],
        indices: &[usize],
        dim: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut values: Vec<(f32, usize)> =
            indices.iter().map(|&idx| (points[idx][dim], idx)).collect();

        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mid = values.len() / 2;
        let left_indices: Vec<usize> = values[..mid].iter().map(|(_, idx)| *idx).collect();
        let right_indices: Vec<usize> = values[mid..].iter().map(|(_, idx)| *idx).collect();

        (left_indices, right_indices)
    }

    fn compute_centroid_of_centers(&self, centers: &[Vec<f32>]) -> Vec<f32> {
        let dim = centers[0].len();
        let mut centroid = vec![0.0; dim];

        for center in centers {
            for (i, &val) in center.iter().enumerate() {
                centroid[i] += val;
            }
        }

        let n = centers.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }

        centroid
    }

    /// Search for k nearest neighbors using iterative algorithm
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::new();
        let mut stack: Vec<&BallNode> = vec![self
            .root
            .as_ref()
            .expect("tree should have root after build")];

        while let Some(node) = stack.pop() {
            // Check if we need to explore this node
            let dist_to_center = self.config.distance_metric.distance(query, &node.center);

            if heap.len() >= k {
                let worst_dist = heap.peek().expect("heap should have k elements").distance;
                if dist_to_center - node.radius > worst_dist {
                    continue; // Prune this branch
                }
            }

            if node.indices.is_empty() {
                // Internal node - add children to stack
                if let (Some(left), Some(right)) = (&node.left, &node.right) {
                    let left_dist = self.config.distance_metric.distance(query, &left.center);
                    let right_dist = self.config.distance_metric.distance(query, &right.center);

                    // Add in order so closer one is processed first
                    if left_dist < right_dist {
                        stack.push(right);
                        stack.push(left);
                    } else {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            } else {
                // Leaf node - check all points
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
            }
        }

        let mut results: Vec<(usize, f32)> =
            heap.into_iter().map(|r| (r.index, r.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }
}
