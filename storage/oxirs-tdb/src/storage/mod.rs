//! Low-level storage layer
//!
//! This module provides the foundation for disk-based storage:
//! - Page management (4KB fixed-size pages)
//! - File I/O with memory-mapped files
//! - Free space allocation
//! - Buffer pool with LRU caching

pub mod adaptive_tuning;
pub mod allocator;
pub mod buffer_pool;
// pub mod buffer_pool_tuner; // TODO(v0.1.0-rc.1): Update to scirs2-core v0.1.0-rc.2+ metrics API
pub mod direct_io;
pub mod file_manager;
pub mod memory_optimization;
pub mod page;
pub mod zero_copy;

// Re-exports
pub use adaptive_tuning::{AdaptiveTuner, AdaptiveTuningConfig};
pub use allocator::Allocator;
pub use buffer_pool::{BufferPool, BufferPoolStats, PageGuard};
// pub use buffer_pool_tuner::{
//     AccessPattern, BufferPoolTuner, BufferPoolTunerConfig, EvictionPolicy,
//     PerformanceReport, TuningRecommendation,
// };
pub use direct_io::{
    AlignedBuffer, DirectIOConfig, DirectIOFile, DirectIOFileStats, DIRECT_IO_ALIGNMENT,
};
pub use file_manager::FileManager;
pub use memory_optimization::{
    MemoryOptimizationConfig, MemoryOptimizer, MemoryPoolStats, ReadaheadStrategy,
};
pub use page::{Page, PageHeader, PageId, PageType, PAGE_SIZE, PAGE_USABLE_SIZE};
pub use zero_copy::{BatchView, VectoredIO, ZeroCopyBuffer, ZeroCopyStats};
