//! Memory management and pooling for performance optimization
//!
//! This module provides memory pooling capabilities to reduce allocation overhead
//! and improve performance for high-throughput streaming operations.

use super::config::MemoryPoolConfig;
use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Memory pool for efficient allocation and deallocation
pub struct MemoryPool {
    config: MemoryPoolConfig,
    available_blocks: Arc<RwLock<VecDeque<MemoryBlock>>>,
    stats: Arc<MemoryPoolStats>,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    last_compaction: Arc<RwLock<Instant>>,
}

/// Memory block structure
#[derive(Debug)]
pub struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
    allocated_at: Instant,
}

// Safety: MemoryBlock is safe to send between threads as it represents
// owned memory that won't be accessed concurrently
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Memory pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStats {
    /// Total blocks allocated
    pub total_allocated: usize,
    /// Total blocks freed
    pub total_freed: usize,
    /// Currently allocated blocks
    pub current_allocated: usize,
    /// Peak allocated blocks
    pub peak_allocated: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Total bytes freed
    pub total_bytes_freed: usize,
    /// Current bytes allocated
    pub current_bytes_allocated: usize,
    /// Peak bytes allocated
    pub peak_bytes_allocated: usize,
    /// Pool hits (reused blocks)
    pub pool_hits: usize,
    /// Pool misses (new allocations)
    pub pool_misses: usize,
    /// Fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Last compaction time (seconds since Unix epoch)
    pub last_compaction: Option<u64>,
}

impl Default for MemoryPoolStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_freed: 0,
            current_allocated: 0,
            peak_allocated: 0,
            total_bytes_allocated: 0,
            total_bytes_freed: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
            pool_hits: 0,
            pool_misses: 0,
            fragmentation_ratio: 0.0,
            average_allocation_size: 0.0,
            last_compaction: None,
        }
    }
}

impl MemoryPool {
    /// Create a new memory pool with the given configuration
    pub fn new(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            available_blocks: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(MemoryPoolStats::default()),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            last_compaction: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Allocate a memory block of the given size
    pub fn allocate(&self, size: usize) -> Result<MemoryHandle> {
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|e| anyhow::anyhow!("Invalid layout: {}", e))?;

        // Try to reuse an existing block
        if let Some(block) = self.try_reuse_block(size) {
            self.update_stats_on_hit();
            return Ok(MemoryHandle {
                block,
                pool: self.available_blocks.clone(),
            });
        }

        // Allocate new block
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow::anyhow!("Failed to allocate memory"));
        }

        let block = MemoryBlock {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
            allocated_at: Instant::now(),
        };

        self.update_stats_on_miss(size);

        Ok(MemoryHandle {
            block,
            pool: self.available_blocks.clone(),
        })
    }

    /// Try to reuse an existing memory block
    fn try_reuse_block(&self, size: usize) -> Option<MemoryBlock> {
        let mut available = self.available_blocks.write();

        // Find a suitable block (first fit strategy)
        for (i, block) in available.iter().enumerate() {
            if block.size >= size {
                return available.remove(i);
            }
        }

        None
    }

    /// Update statistics on cache hit
    fn update_stats_on_hit(&self) {
        // In a real implementation, this would update atomic counters
        // For now, we'll use a simplified approach
        debug!("Memory pool hit");
    }

    /// Update statistics on cache miss
    fn update_stats_on_miss(&self, size: usize) {
        let current = self.total_allocated.fetch_add(1, Ordering::Relaxed) + 1;
        let peak = self.peak_allocated.load(Ordering::Relaxed);

        if current > peak {
            self.peak_allocated.store(current, Ordering::Relaxed);
        }

        debug!("Memory pool miss, allocated block of size: {}", size);
    }

    /// Return a memory block to the pool
    pub fn deallocate(&self, block: MemoryBlock) {
        let mut available = self.available_blocks.write();

        // Check if we should return the block to the pool or deallocate it
        if available.len() < self.config.max_size / block.size {
            available.push_back(block);
            debug!("Returned block to pool");
        } else {
            // Pool is full, deallocate the block
            unsafe {
                dealloc(block.ptr.as_ptr(), block.layout);
            }
            debug!("Deallocated block (pool full)");
        }

        self.total_allocated.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        let available = self.available_blocks.read();
        let current_allocated = self.total_allocated.load(Ordering::Relaxed);
        let peak_allocated = self.peak_allocated.load(Ordering::Relaxed);

        MemoryPoolStats {
            total_allocated: current_allocated + available.len(),
            total_freed: 0, // Would be tracked in a real implementation
            current_allocated,
            peak_allocated,
            total_bytes_allocated: 0, // Would be tracked in a real implementation
            total_bytes_freed: 0,
            current_bytes_allocated: 0,
            peak_bytes_allocated: 0,
            pool_hits: 0,
            pool_misses: 0,
            fragmentation_ratio: self.calculate_fragmentation(),
            average_allocation_size: 0.0,
            last_compaction: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            ),
        }
    }

    /// Calculate fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let available = self.available_blocks.read();
        if available.is_empty() {
            return 0.0;
        }

        let total_size: usize = available.iter().map(|b| b.size).sum();
        let largest_block = available.iter().map(|b| b.size).max().unwrap_or(0);

        if total_size == 0 {
            0.0
        } else {
            1.0 - (largest_block as f64 / total_size as f64)
        }
    }

    /// Compact the memory pool
    pub fn compact(&self) -> Result<()> {
        let mut available = self.available_blocks.write();
        let mut last_compaction = self.last_compaction.write();

        // Sort blocks by size for better allocation patterns
        let mut blocks: Vec<_> = available.drain(..).collect();
        blocks.sort_by_key(|b| b.size);

        // Remove very old blocks to prevent memory leaks
        let now = Instant::now();
        let threshold = std::time::Duration::from_secs(300); // 5 minutes

        blocks.retain(|block| now.duration_since(block.allocated_at) < threshold);

        // Put blocks back
        available.extend(blocks);
        *last_compaction = now;

        info!(
            "Memory pool compacted, {} blocks remaining",
            available.len()
        );
        Ok(())
    }

    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        let last_compaction = self.last_compaction.read();
        let elapsed = last_compaction.elapsed();

        elapsed.as_secs() > self.config.compaction_interval
    }

    /// Get the current pool size
    pub fn pool_size(&self) -> usize {
        self.available_blocks.read().len()
    }

    /// Get the total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }
}

/// Handle for allocated memory
pub struct MemoryHandle {
    block: MemoryBlock,
    pool: Arc<RwLock<VecDeque<MemoryBlock>>>,
}

impl MemoryHandle {
    /// Get a raw pointer to the allocated memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.block.ptr.as_ptr()
    }

    /// Get the size of the allocated memory
    pub fn size(&self) -> usize {
        self.block.size
    }

    /// Get a slice view of the allocated memory
    ///
    /// # Safety
    /// The caller must ensure that the memory block is properly initialized
    /// and that the returned slice is not accessed after the block is deallocated
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.block.ptr.as_ptr(), self.block.size)
    }

    /// Get a mutable slice view of the allocated memory
    ///
    /// # Safety
    /// The caller must ensure that the memory block is properly initialized
    /// and that the returned slice is not accessed after the block is deallocated
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.block.ptr.as_ptr(), self.block.size)
    }
}

impl Drop for MemoryHandle {
    fn drop(&mut self) {
        // Return the block to the pool
        let block = MemoryBlock {
            ptr: self.block.ptr,
            size: self.block.size,
            layout: self.block.layout,
            allocated_at: self.block.allocated_at,
        };

        let mut available = self.pool.write();
        available.push_back(block);
    }
}

unsafe impl Send for MemoryHandle {}
unsafe impl Sync for MemoryHandle {}

/// Memory allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AllocationStrategy {
    /// First fit - use the first block that fits
    #[default]
    FirstFit,
    /// Best fit - use the smallest block that fits
    BestFit,
    /// Worst fit - use the largest block that fits
    WorstFit,
    /// Next fit - start searching from the last allocated position
    NextFit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);

        assert_eq!(pool.pool_size(), 0);
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_memory_allocation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);

        let handle = pool.allocate(1024).unwrap();
        assert_eq!(handle.size(), 1024);
        assert_eq!(pool.total_allocated(), 1);
    }

    #[test]
    fn test_memory_pool_stats() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);

        let stats = pool.stats();
        assert_eq!(stats.current_allocated, 0);
        assert_eq!(stats.peak_allocated, 0);
    }

    #[test]
    fn test_memory_pool_compaction() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);

        assert!(pool.compact().is_ok());
    }
}
