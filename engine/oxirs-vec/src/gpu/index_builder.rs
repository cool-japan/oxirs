//! GPU-accelerated HNSW index construction
//!
//! This module implements GPU-based HNSW graph construction using CUDA kernels
//! (feature-gated behind the `cuda` feature). When CUDA is not available, it
//! falls back to an efficient CPU implementation.
//!
//! # Architecture
//!
//! The GPU index builder works in phases:
//! 1. **Vector Upload**: Transfer vectors to GPU memory in batches
//! 2. **Distance Matrix Computation**: Compute all-pairs distances via CUDA kernels
//! 3. **Neighbor Selection**: Apply heuristic neighbor selection on GPU
//! 4. **Graph Assembly**: Assemble the HNSW graph structure from GPU results
//!
//! # Pure Rust Policy
//!
//! All CUDA code is gated with `#[cfg(feature = "cuda")]` so the default build
//! is 100% Pure Rust.
//!
//! # Module Layout
//!
//! - [`crate::gpu::index_builder_types`]: Config types, distance metrics, graph structures
//! - [`crate::gpu::index_builder_phases`]: Builder implementations and pipeline stages

// Re-export all public types from sub-modules so the public API is unchanged.
pub use crate::gpu::index_builder_phases::{
    BatchSizeCalculator, ComputedBatch, GpuBatchDistanceComputer, GpuHnswIndexBuilder,
    GpuIndexOptimizer, GpuMemoryBudget, IncrementalGpuIndexBuilder, IndexedBatch,
    PipelinedIndexBuilder, PreparedBatch,
};
pub use crate::gpu::index_builder_types::{
    ComputationCache, GpuDistanceMetric, GpuIndexBuildStats, GpuIndexBuilderConfig, HnswGraph,
    HnswNode,
};
