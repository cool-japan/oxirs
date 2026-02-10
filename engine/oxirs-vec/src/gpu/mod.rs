//! GPU acceleration for vector operations using CUDA
//!
//! This module provides GPU acceleration for:
//! - Distance calculations (cosine, euclidean, etc.)
//! - Batch vector operations
//! - Parallel search algorithms
//! - Matrix operations for embeddings
//!
//! # CUDA Feature Gating (Pure Rust Policy)
//!
//! GPU acceleration is **optional** and properly feature-gated:
//! - **Default build**: 100% Pure Rust, no CUDA required, CPU implementations only
//! - **With `cuda` feature**: GPU acceleration when CUDA toolkit is installed
//! - **With `cuda` feature but no toolkit**: Graceful fallback to CPU implementations
//!
//! All CUDA-dependent code is gated with `#[cfg(all(feature = "cuda", cuda_runtime_available))]`
//! to ensure the crate builds successfully regardless of CUDA availability.

pub mod accelerator;
pub mod buffer;
pub mod config;
pub mod device;
pub mod index;
pub mod kernels;
pub mod memory_pool;
pub mod performance;
pub mod runtime;
pub mod types;

// Re-export key types for convenience
pub use accelerator::{
    create_default_accelerator, create_memory_optimized_accelerator,
    create_performance_accelerator, is_gpu_available, GpuAccelerator,
};
pub use buffer::GpuBuffer;
pub use config::{GpuConfig, OptimizationLevel, PrecisionMode};
pub use device::GpuDevice;
pub use index::{AdvancedGpuVectorIndex, BatchVectorProcessor, GpuVectorIndex};
pub use kernels::*;
pub use memory_pool::GpuMemoryPool;
pub use performance::GpuPerformanceStats;
pub use types::GpuExecutionConfig;
