//! GPU acceleration abstractions for vector operations
//!
//! This module provides GPU acceleration abstractions for:
//! - Distance calculations (cosine, euclidean, etc.)
//! - Batch vector operations
//! - Parallel search algorithms
//! - Matrix operations for embeddings
//!
//! # Pure Rust Policy (COOLJAPAN Pure Rust Policy v2)
//!
//! This module is **100% Pure Rust**: all buffers, devices, streams and kernels
//! here are CPU-backed reference implementations with no C FFI. Real NVIDIA CUDA
//! acceleration (`cuda-runtime-sys`) is quarantined in the companion
//! `oxirs-vec-adapter-cuda` crate (publish = false), which depends on this crate
//! and reuses these public types. This keeps oxirs-vec's published
//! `--all-features` surface free of `cuda-runtime-sys`.

pub mod accelerator;
pub mod buffer;
pub mod config;
pub mod device;
pub mod index;
pub mod index_builder;
pub mod index_builder_phases;
#[cfg(test)]
mod index_builder_tests;
pub mod index_builder_types;
pub mod kernels;
pub mod load_balancer;
pub mod memory_pool;
pub mod multi_gpu;
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
pub use index_builder::{
    BatchSizeCalculator, ComputedBatch, GpuBatchDistanceComputer, GpuDistanceMetric,
    GpuHnswIndexBuilder, GpuIndexBuildStats, GpuIndexBuilderConfig, GpuIndexOptimizer,
    GpuMemoryBudget, HnswGraph, HnswNode, IncrementalGpuIndexBuilder, IndexedBatch,
    PipelinedIndexBuilder, PreparedBatch,
};
pub use kernels::*;
pub use load_balancer::{GpuLoadBalancer, SimpleGpuDevice, WorkloadChunk, WorkloadDistributor};
pub use memory_pool::GpuMemoryPool;
pub use multi_gpu::{
    GpuDeviceMetrics, GpuTaskOutput, GpuTaskResult, LoadBalancingStrategy, MultiGpuConfig,
    MultiGpuConfigFactory, MultiGpuManager, MultiGpuStats, MultiGpuTask, TaskPriority,
};
pub use performance::GpuPerformanceStats;
pub use types::GpuExecutionConfig;
