//! Tree-based indices for efficient nearest neighbor search
//!
//! **EXPERIMENTAL**: These tree implementations are currently experimental
//! and have known limitations with large datasets or specific configurations.
//! For production use, prefer HNSW, IVF, or LSH indices instead.
//!
//! ## Known Limitations
//! - Tree construction uses recursion with conservative depth limits (20 levels)
//! - Best suited for moderate-sized datasets (< 100K vectors)
//! - May encounter stack overflow on systems with very small stack sizes
//! - Performance degrades in high-dimensional spaces (> 128 dimensions)
//!
//! ## Recommended Alternatives
//! - For most use cases: Use `HnswIndex` (Hierarchical Navigable Small World)
//! - For very large datasets: Use `IVFIndex` (Inverted File Index)
//! - For approximate search: Use `LSHIndex` (Locality Sensitive Hashing)
//!
//! This module implements various tree data structures:
//! - Ball Tree: Efficient for arbitrary metrics
//! - KD-Tree: Classic space partitioning tree
//! - VP-Tree: Vantage point tree for metric spaces
//! - Cover Tree: Navigating nets with provable bounds
//! - Random Projection Trees: Randomized space partitioning
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::tree_indices_types`]      — [`TreeIndexConfig`], [`TreeType`], [`DistanceMetric`].
//! - [`crate::tree_indices_balltree`]   — the [`BallTree`].
//! - [`crate::tree_indices_kdtree`]     — the [`KdTree`].
//! - [`crate::tree_indices_vptree`]     — the [`VpTree`].
//! - [`crate::tree_indices_covertree`]  — the [`CoverTree`].
//! - [`crate::tree_indices_rptree`]     — the [`RandomProjectionTree`].
//! - [`crate::tree_indices_unified`]    — the unified [`TreeIndex`] dispatcher.

pub use crate::tree_indices_balltree::BallTree;
pub use crate::tree_indices_covertree::CoverTree;
pub use crate::tree_indices_kdtree::KdTree;
pub use crate::tree_indices_rptree::RandomProjectionTree;
pub use crate::tree_indices_types::{DistanceMetric, TreeIndexConfig, TreeType};
pub use crate::tree_indices_unified::TreeIndex;
pub use crate::tree_indices_vptree::VpTree;
