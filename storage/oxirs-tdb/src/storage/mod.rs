//! Low-level storage layer
//!
//! This module provides the foundation for disk-based storage:
//! - Page management (4KB fixed-size pages)
//! - File I/O with memory-mapped files
//! - Free space allocation
//! - Buffer pool with LRU caching

// Hardened durability core (allocator, buffer_pool, file_manager, page,
// superblock) compiles strictly; peripheral storage engines keep a scoped
// dead-code allow (they predate this durability pass and are out of scope here).
#[allow(dead_code, unused_imports, unused_variables)]
pub mod adaptive_tuning;
pub mod allocator;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod async_io;
pub mod buffer_pool;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod buffer_pool_tuner; // ✅ Updated to scirs2-core v0.1.0+ metrics API
#[allow(dead_code, unused_imports, unused_variables)]
pub mod columnar_analytics; // Columnar analytics storage for SPARQL analytical queries
#[allow(dead_code, unused_imports, unused_variables)]
pub mod direct_io;
pub mod file_manager;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod lsm_tree; // LSM-tree storage engine
#[allow(dead_code, unused_imports, unused_variables)]
pub mod memory_optimization;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod mmap_optimizer; // Memory-mapped file optimization with OS-level hints
#[allow(dead_code, unused_imports, unused_variables)]
pub mod numa_allocator; // NUMA-aware memory management
pub mod page;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod partitioning; // Database partitioning for horizontal scaling
pub mod superblock; // On-disk catalog (page 0): durable B+Tree roots + free-list head
#[allow(dead_code, unused_imports, unused_variables)]
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
pub use superblock::{Superblock, SUPERBLOCK_FORMAT_VERSION, SUPERBLOCK_MAGIC, SUPERBLOCK_PAGE_ID};
pub use zero_copy::{BatchView, VectoredIO, ZeroCopyBuffer, ZeroCopyStats};
