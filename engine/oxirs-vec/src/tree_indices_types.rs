//! Shared configuration and helper types for tree-based indices.
//!
//! Defines [`TreeIndexConfig`], [`TreeType`], [`DistanceMetric`], and the
//! crate-internal `SearchResult` heap entry used by every tree implementation.

use oxirs_core::simd::SimdOps;
use std::cmp::Ordering;

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
            max_leaf_size: 16, // Larger leaf size to prevent deep recursion and stack overflow
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
    pub(crate) fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) struct SearchResult {
    pub(crate) index: usize,
    pub(crate) distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
