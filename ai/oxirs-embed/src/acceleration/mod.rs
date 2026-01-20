//! Hardware acceleration modules for embedding computations
//!
//! This module provides various acceleration strategies including GPU acceleration
//! through CUDA, OpenCL, ROCm, and Metal backends via scirs2-linalg integration.

pub mod gpu;

// Re-export main types for convenience
pub use gpu::{AdaptiveEmbeddingAccelerator, GpuEmbeddingAccelerator};
