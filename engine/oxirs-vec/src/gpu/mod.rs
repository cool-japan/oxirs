//! GPU acceleration for vector operations using CUDA
//!
//! This module provides GPU acceleration for:
//! - Distance calculations (cosine, euclidean, etc.)
//! - Batch vector operations
//! - Parallel search algorithms
//! - Matrix operations for embeddings

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
pub use runtime::GpuExecutionConfig;
pub use types::*;
