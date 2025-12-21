//! Low-level storage layer
//!
//! This module provides the foundation for disk-based storage:
//! - Page management (4KB fixed-size pages)
//! - File I/O with memory-mapped files
//! - Free space allocation
//! - Buffer pool with LRU caching

pub mod adaptive_tuning;
pub mod allocator;
pub mod async_io;
pub mod buffer_pool;
pub mod buffer_pool_tuner; // ✅ Updated to scirs2-core v0.1.0-rc.2+ metrics API
pub mod columnar_analytics; // Columnar analytics storage for SPARQL analytical queries
pub mod direct_io;
pub mod file_manager;
pub mod lsm_tree; // LSM-tree storage engine
pub mod memory_optimization;
pub mod mmap_optimizer; // Memory-mapped file optimization with OS-level hints
pub mod numa_allocator; // NUMA-aware memory management
pub mod page;
pub mod partitioning; // Database partitioning for horizontal scaling
pub mod zero_copy;

// Re-exports
pub use adaptive_tuning::{AdaptiveTuner, AdaptiveTuningConfig};
pub use allocator::Allocator;
pub use async_io::{AsyncFileHandle, AsyncIoBackend, AsyncIoBatch, AsyncIoStats};
pub use buffer_pool::{BufferPool, BufferPoolStats, PageGuard};
pub use buffer_pool_tuner::{
    AccessPattern, BufferPoolTuner, BufferPoolTunerConfig, EvictionPolicy, PerformanceReport,
    TuningRecommendation,
};
pub use columnar_analytics::{AggregateFunction, AggregateResult, ColumnarStats, ColumnarStore};
pub use direct_io::{
    AlignedBuffer, DirectIOConfig, DirectIOFile, DirectIOFileStats, DIRECT_IO_ALIGNMENT,
};
pub use file_manager::FileManager;
pub use lsm_tree::{CompactionStrategy, LevelStats, LsmConfig, LsmStats, LsmTree};
pub use memory_optimization::{
    MemoryOptimizationConfig, MemoryOptimizer, MemoryPoolStats, ReadaheadStrategy,
};
pub use mmap_optimizer::{
    AccessPattern as MmapAccessPattern, MmapOptimizer, MmapOptimizerConfig, MmapStats,
};
pub use numa_allocator::{NumaAllocator, NumaNode, NumaPolicy, NumaStats, NumaTopology};
pub use page::{Page, PageHeader, PageId, PageType, PAGE_SIZE, PAGE_USABLE_SIZE};
pub use partitioning::{
    PartitionId, PartitionManager, PartitionMetadata, PartitionStats, PartitionStrategy,
    TriplePattern,
};
pub use zero_copy::{BatchView, VectoredIO, ZeroCopyBuffer, ZeroCopyStats};
