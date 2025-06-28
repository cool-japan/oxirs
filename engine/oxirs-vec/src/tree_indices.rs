//! Tree-based indices for efficient nearest neighbor search
//!
//! This module implements various tree data structures optimized for
//! high-dimensional vector search:
//! - Ball Tree: Efficient for arbitrary metrics
//! - KD-Tree: Classic space partitioning tree
//! - VP-Tree: Vantage point tree for metric spaces
//! - Cover Tree: Navigating nets with provable bounds
//! - Random Projection Trees: Randomized space partitioning

use crate::{Vector, VectorIndex};
use anyhow::Result;
use oxirs_core::parallel::*;
use oxirs_core::simd::SimdOps;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for tree-based indices
#[derive(Debug, Clone)]
pub struct TreeIndexConfig {
    /// Type of tree to use
    pub tree_type: TreeType,
    /// Maximum leaf size before splitting
    pub max_leaf_size: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Enable parallel construction
    pub parallel_construction: bool,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

impl Default for TreeIndexConfig {
    fn default() -> Self {
        Self {
            tree_type: TreeType::BallTree,
            max_leaf_size: 40,
            random_seed: None,
            parallel_construction: true,
            distance_metric: DistanceMetric::Euclidean,
        }
    }
}

/// Available tree types
#[derive(Debug, Clone, Copy)]
pub enum TreeType {
    BallTree,
    KdTree,
    VpTree,
    CoverTree,
    RandomProjectionTree,
}

/// Distance metrics
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Minkowski(f32),
}

impl DistanceMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => f32::euclidean_distance(a, b),
            DistanceMetric::Manhattan => f32::manhattan_distance(a, b),
            DistanceMetric::Cosine => f32::cosine_distance(a, b),
            DistanceMetric::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs().powf(*p))
                .sum::<f32>()
                .powf(1.0 / p),
        }
    }
}

/// Search result with distance
#[derive(Debug, Clone)]
struct SearchResult {
    index: usize,
    distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Ball Tree implementation
pub struct BallTree {
    root: Option<Box<BallNode>>,
    data: Vec<(String, Vector)>,
    config: TreeIndexConfig,
}

struct BallNode {
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

    /// Build the tree from data
    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        let indices: Vec<usize> = (0..self.data.len()).collect();
        let points: Vec<Vec<f32>> = self.data.iter().map(|(_, v)| v.as_f32()).collect();

        self.root = Some(Box::new(self.build_node(&points, indices)?));
        Ok(())
    }

    fn build_node(&self, points: &[Vec<f32>], indices: Vec<usize>) -> Result<BallNode> {
        if indices.len() <= self.config.max_leaf_size {
            // Leaf node
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

        // Find the dimension with maximum spread
        let split_dim = self.find_split_dimension(points, &indices);

        // Partition points along the split dimension
        let (left_indices, right_indices) = self.partition_indices(points, &indices, split_dim);

        // Recursively build child nodes
        let left_node = self.build_node(points, left_indices)?;
        let right_node = self.build_node(points, right_indices)?;

        // Compute bounding ball for this node
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

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut heap = BinaryHeap::new();
        self.search_node(self.root.as_ref().unwrap(), query, k, &mut heap);

        let mut results: Vec<(usize, f32)> =
            heap.into_iter().map(|r| (r.index, r.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn search_node(
        &self,
        node: &BallNode,
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<SearchResult>,
    ) {
        // Check if we need to explore this node
        let dist_to_center = self.config.distance_metric.distance(query, &node.center);

        if heap.len() >= k {
            let worst_dist = heap.peek().unwrap().distance;
            if dist_to_center - node.radius > worst_dist {
                return; // Prune this branch
            }
        }

        if node.indices.is_empty() {
            // Internal node - search children
            if let (Some(left), Some(right)) = (&node.left, &node.right) {
                let left_dist = self.config.distance_metric.distance(query, &left.center);
                let right_dist = self.config.distance_metric.distance(query, &right.center);

                if left_dist < right_dist {
                    self.search_node(left, query, k, heap);
                    self.search_node(right, query, k, heap);
                } else {
                    self.search_node(right, query, k, heap);
                    self.search_node(left, query, k, heap);
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
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(SearchResult {
                        index: idx,
                        distance: dist,
                    });
                }
            }
        }
    }
}

/// KD-Tree implementation
pub struct KdTree {
    root: Option<Box<KdNode>>,
    data: Vec<(String, Vector)>,
    config: TreeIndexConfig,
}

struct KdNode {
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
        if indices.len() <= self.config.max_leaf_size {
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

        let left = if !left_indices.is_empty() {
            Some(Box::new(self.build_node(
                points,
                left_indices,
                depth + 1,
            )?))
        } else {
            None
        };

        let right = if !right_indices.is_empty() {
            Some(Box::new(self.build_node(
                points,
                right_indices,
                depth + 1,
            )?))
        } else {
            None
        };

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
        self.search_node(self.root.as_ref().unwrap(), query, k, &mut heap);

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
                } else if dist < heap.peek().unwrap().distance {
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
            split_dist < heap.peek().unwrap().distance
        } {
            if let Some(child) = second {
                self.search_node(child, query, k, heap);
            }
        }
    }
}

/// VP-Tree (Vantage Point Tree) implementation
pub struct VpTree {
    root: Option<Box<VpNode>>,
    data: Vec<(String, Vector)>,
    config: TreeIndexConfig,
}

struct VpNode {
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
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        self.root = Some(Box::new(self.build_node(indices, &mut rng)?));
        Ok(())
    }

    fn build_node(&self, mut indices: Vec<usize>, rng: &mut impl rand::Rng) -> Result<VpNode> {
        use rand::seq::SliceRandom;

        if indices.len() <= self.config.max_leaf_size {
            return Ok(VpNode {
                vantage_point: 0,
                median_distance: 0.0,
                inside: None,
                outside: None,
                indices,
            });
        }

        // Choose random vantage point
        let vp_idx = indices.len() - 1;
        indices.shuffle(rng);
        let vantage_point = indices[vp_idx];
        indices.truncate(vp_idx);

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

        let inside = if !inside_indices.is_empty() {
            Some(Box::new(self.build_node(inside_indices, rng)?))
        } else {
            None
        };

        let outside = if !outside_indices.is_empty() {
            Some(Box::new(self.build_node(outside_indices, rng)?))
        } else {
            None
        };

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
            self.root.as_ref().unwrap(),
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
                    } else if dist < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(SearchResult {
                            index: idx,
                            distance: dist,
                        });
                    }

                    if heap.len() >= k {
                        tau = heap.peek().unwrap().distance;
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
            } else if dist_to_vp < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(SearchResult {
                    index: node.vantage_point,
                    distance: dist_to_vp,
                });
            }

            if heap.len() >= k {
                tau = heap.peek().unwrap().distance;
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

/// Cover Tree implementation
pub struct CoverTree {
    root: Option<Box<CoverNode>>,
    data: Vec<(String, Vector)>,
    config: TreeIndexConfig,
    base: f32,
}

struct CoverNode {
    /// Point index
    point: usize,
    /// Level in the tree
    level: i32,
    /// Children at the same or lower level
    children: Vec<Box<CoverNode>>,
}

impl CoverTree {
    pub fn new(config: TreeIndexConfig) -> Self {
        Self {
            root: None,
            data: Vec::new(),
            config,
            base: 2.0, // Base for the covering constant
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        // Initialize with first point
        self.root = Some(Box::new(CoverNode {
            point: 0,
            level: self.get_level(0),
            children: Vec::new(),
        }));

        // Insert remaining points
        for idx in 1..self.data.len() {
            self.insert(idx)?;
        }

        Ok(())
    }

    fn get_level(&self, point_idx: usize) -> i32 {
        // Simple heuristic for initial level
        ((self.data.len() as f32).log2() as i32).max(0)
    }

    fn insert(&mut self, point_idx: usize) -> Result<()> {
        // Simplified insert - in practice, this would be more complex
        // to maintain the cover tree invariants
        let level = self.get_level(point_idx);
        if let Some(root) = &mut self.root {
            root.children.push(Box::new(CoverNode {
                point: point_idx,
                level,
                children: Vec::new(),
            }));
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut results = Vec::new();
        self.search_node(self.root.as_ref().unwrap(), query, k, &mut results);

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn search_node(
        &self,
        node: &CoverNode,
        query: &[f32],
        k: usize,
        results: &mut Vec<(usize, f32)>,
    ) {
        let point_data = &self.data[node.point].1.as_f32();
        let dist = self.config.distance_metric.distance(query, point_data);

        results.push((node.point, dist));

        // Search children
        for child in &node.children {
            self.search_node(child, query, k, results);
        }
    }
}

/// Random Projection Tree implementation
pub struct RandomProjectionTree {
    root: Option<Box<RpNode>>,
    data: Vec<(String, Vector)>,
    config: TreeIndexConfig,
}

struct RpNode {
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
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        self.root = Some(Box::new(self.build_node(indices, dimensions, &mut rng)?));
        Ok(())
    }

    fn build_node(
        &self,
        indices: Vec<usize>,
        dimensions: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<RpNode> {
        use rand::distributions::{Distribution, Standard};

        if indices.len() <= self.config.max_leaf_size {
            return Ok(RpNode {
                projection: Vec::new(),
                threshold: 0.0,
                left: None,
                right: None,
                indices,
            });
        }

        // Generate random projection vector
        let projection: Vec<f32> = Standard.sample_iter(&mut *rng).take(dimensions).collect();

        // Normalize projection vector
        let norm = f32::norm(&projection);
        let projection: Vec<f32> = projection.iter().map(|&x| x / norm).collect();

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

        let left = if !left_indices.is_empty() {
            Some(Box::new(self.build_node(left_indices, dimensions, rng)?))
        } else {
            None
        };

        let right = if !right_indices.is_empty() {
            Some(Box::new(self.build_node(right_indices, dimensions, rng)?))
        } else {
            None
        };

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
        self.search_node(self.root.as_ref().unwrap(), query, k, &mut heap);

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
                } else if dist < heap.peek().unwrap().distance {
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

/// Unified tree index interface
pub struct TreeIndex {
    tree_type: TreeType,
    ball_tree: Option<BallTree>,
    kd_tree: Option<KdTree>,
    vp_tree: Option<VpTree>,
    cover_tree: Option<CoverTree>,
    rp_tree: Option<RandomProjectionTree>,
}

impl TreeIndex {
    pub fn new(config: TreeIndexConfig) -> Self {
        let tree_type = config.tree_type;

        let (ball_tree, kd_tree, vp_tree, cover_tree, rp_tree) = match tree_type {
            TreeType::BallTree => (Some(BallTree::new(config)), None, None, None, None),
            TreeType::KdTree => (None, Some(KdTree::new(config)), None, None, None),
            TreeType::VpTree => (None, None, Some(VpTree::new(config)), None, None),
            TreeType::CoverTree => (None, None, None, Some(CoverTree::new(config)), None),
            TreeType::RandomProjectionTree => (
                None,
                None,
                None,
                None,
                Some(RandomProjectionTree::new(config)),
            ),
        };

        Self {
            tree_type,
            ball_tree,
            kd_tree,
            vp_tree,
            cover_tree,
            rp_tree,
        }
    }

    fn build(&mut self) -> Result<()> {
        match self.tree_type {
            TreeType::BallTree => self.ball_tree.as_mut().unwrap().build(),
            TreeType::KdTree => self.kd_tree.as_mut().unwrap().build(),
            TreeType::VpTree => self.vp_tree.as_mut().unwrap().build(),
            TreeType::CoverTree => self.cover_tree.as_mut().unwrap().build(),
            TreeType::RandomProjectionTree => self.rp_tree.as_mut().unwrap().build(),
        }
    }

    fn search_internal(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self.tree_type {
            TreeType::BallTree => self.ball_tree.as_ref().unwrap().search(query, k),
            TreeType::KdTree => self.kd_tree.as_ref().unwrap().search(query, k),
            TreeType::VpTree => self.vp_tree.as_ref().unwrap().search(query, k),
            TreeType::CoverTree => self.cover_tree.as_ref().unwrap().search(query, k),
            TreeType::RandomProjectionTree => self.rp_tree.as_ref().unwrap().search(query, k),
        }
    }
}

impl VectorIndex for TreeIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        let data = match self.tree_type {
            TreeType::BallTree => &mut self.ball_tree.as_mut().unwrap().data,
            TreeType::KdTree => &mut self.kd_tree.as_mut().unwrap().data,
            TreeType::VpTree => &mut self.vp_tree.as_mut().unwrap().data,
            TreeType::CoverTree => &mut self.cover_tree.as_mut().unwrap().data,
            TreeType::RandomProjectionTree => &mut self.rp_tree.as_mut().unwrap().data,
        };

        data.push((uri, vector));
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let results = self.search_internal(&query_f32, k);

        let data = match self.tree_type {
            TreeType::BallTree => &self.ball_tree.as_ref().unwrap().data,
            TreeType::KdTree => &self.kd_tree.as_ref().unwrap().data,
            TreeType::VpTree => &self.vp_tree.as_ref().unwrap().data,
            TreeType::CoverTree => &self.cover_tree.as_ref().unwrap().data,
            TreeType::RandomProjectionTree => &self.rp_tree.as_ref().unwrap().data,
        };

        Ok(results
            .into_iter()
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let all_results = self.search_internal(&query_f32, 1000); // Search more broadly

        let data = match self.tree_type {
            TreeType::BallTree => &self.ball_tree.as_ref().unwrap().data,
            TreeType::KdTree => &self.kd_tree.as_ref().unwrap().data,
            TreeType::VpTree => &self.vp_tree.as_ref().unwrap().data,
            TreeType::CoverTree => &self.cover_tree.as_ref().unwrap().data,
            TreeType::RandomProjectionTree => &self.rp_tree.as_ref().unwrap().data,
        };

        Ok(all_results
            .into_iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        let data = match self.tree_type {
            TreeType::BallTree => &self.ball_tree.as_ref().unwrap().data,
            TreeType::KdTree => &self.kd_tree.as_ref().unwrap().data,
            TreeType::VpTree => &self.vp_tree.as_ref().unwrap().data,
            TreeType::CoverTree => &self.cover_tree.as_ref().unwrap().data,
            TreeType::RandomProjectionTree => &self.rp_tree.as_ref().unwrap().data,
        };

        data.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }
}

// Add rand to dependencies for VP-Tree and Random Projection Tree
use rand;

// Placeholder for async task spawning - integrate with oxirs-core::parallel
async fn spawn_task<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // In practice, this would use oxirs-core::parallel's task spawning
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_tree() {
        let config = TreeIndexConfig {
            tree_type: TreeType::BallTree,
            ..Default::default()
        };

        let mut index = TreeIndex::new(config);

        // Insert test vectors
        for i in 0..100 {
            let vector = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![50.0, 100.0, 150.0]);
        let results = index.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, "vec_50"); // Exact match
    }

    #[test]
    fn test_kd_tree() {
        let config = TreeIndexConfig {
            tree_type: TreeType::KdTree,
            ..Default::default()
        };

        let mut index = TreeIndex::new(config);

        // Insert test vectors
        for i in 0..50 {
            let vector = Vector::new(vec![i as f32, (50 - i) as f32]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![25.0, 25.0]);
        let results = index.search_knn(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vp_tree() {
        let config = TreeIndexConfig {
            tree_type: TreeType::VpTree,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut index = TreeIndex::new(config);

        // Insert test vectors
        for i in 0..30 {
            let angle = (i as f32) * std::f32::consts::PI / 15.0;
            let vector = Vector::new(vec![angle.cos(), angle.sin()]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        index.build().unwrap();

        // Search for nearest neighbors
        let query = Vector::new(vec![1.0, 0.0]);
        let results = index.search_knn(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
    }
}
