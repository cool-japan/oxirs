//! HNSW (Hierarchical Navigable Small World) implementation
//!
//! This module provides a pure Rust implementation of the HNSW algorithm
//! for approximate nearest neighbor search with optional GPU acceleration.

pub mod config;
pub mod construction;
pub mod index;
pub mod optimization;
pub mod search;
pub mod stats;
pub mod types;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export main types for convenience
pub use config::*;
pub use index::*;
pub use stats::*;
pub use types::*;
