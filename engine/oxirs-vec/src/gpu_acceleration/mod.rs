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

pub mod device;
pub mod buffer;
pub mod config;
pub mod accelerator;
pub mod vector_index;
pub mod memory_pool;
pub mod batch_processor;

// Re-export main types for convenience
pub use device::{GpuDevice, query_gpu_devices, get_best_gpu_device};
pub use buffer::GpuBuffer;
pub use config::{GpuConfig, GpuOptimization, GpuPrecision};
pub use accelerator::GpuAccelerator;
pub use vector_index::{GpuVectorIndex, AdvancedGpuVectorIndex, GpuOperationType, GpuOperationResult};
pub use memory_pool::GpuMemoryPool;
pub use batch_processor::{BatchVectorProcessor, GpuPerformanceReport};

use anyhow::Result;
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize GPU acceleration system
pub fn initialize_gpu() -> Result<()> {
    INIT.call_once(|| {
        // Initialize CUDA runtime
        // This would contain CUDA initialization code
    });
    Ok(())
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    // Check CUDA availability
    query_gpu_devices().map(|devices| !devices.is_empty()).unwrap_or(false)
}