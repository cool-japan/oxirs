//! HNSW (Hierarchical Navigable Small World) implementation
//!
//! This module provides a pure Rust implementation of the HNSW algorithm
//! for approximate nearest neighbor search with optional GPU acceleration
//! and SIMD optimizations via SciRS2-Core.

pub mod adaptive_search;
pub mod batch;
pub mod config;
pub mod construction;
pub mod index;
pub mod optimization;
pub mod parallel_construction;
pub mod parallel_search;
pub mod query_cache;
pub mod search;
pub mod simd_distance;
pub mod stats;
pub mod types;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export main types for convenience
pub use adaptive_search::{
    AdaptiveSearchConfig, AdaptiveSearchStats, AdaptiveSearchTuner, QueryMetrics,
};
pub use batch::*;
pub use config::*;
pub use index::*;
pub use parallel_construction::{
    ParallelConstructionConfig, ParallelConstructionStats, ParallelHnswBuilder,
    ParallelHnswIndexBuilder,
};
pub use simd_distance::*;
pub use stats::*;
pub use types::*;
