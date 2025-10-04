//! Low-level storage layer
//!
//! This module provides the foundation for disk-based storage:
//! - Page management (4KB fixed-size pages)
//! - File I/O with memory-mapped files
//! - Free space allocation
//! - Buffer pool with LRU caching

pub mod page;
pub mod file_manager;
pub mod allocator;
pub mod buffer_pool;

// Re-exports
pub use page::{Page, PageId, PageType, PageHeader, PAGE_SIZE, PAGE_USABLE_SIZE};
pub use file_manager::FileManager;
pub use allocator::Allocator;
pub use buffer_pool::{BufferPool, BufferPoolStats, PageGuard};
