//! HNSW (Hierarchical Navigable Small World) implementation
//!
//! This module provides a pure Rust implementation of the HNSW algorithm
//! for approximate nearest neighbor search with optional GPU acceleration.

pub mod config;
pub mod types;
pub mod stats;
pub mod index;
pub mod search;
pub mod construction;
pub mod optimization;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export main types for convenience
pub use config::*;
pub use types::*;
pub use stats::*;
pub use index::*;