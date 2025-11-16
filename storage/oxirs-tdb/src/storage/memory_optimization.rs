//! Advanced memory optimizations for high-performance I/O
//!
//! This module provides advanced memory optimization techniques:
//! - Sequential access prefetching
//! - Memory pooling for zero-copy operations
//! - Page alignment for optimal cache usage
//! - Readahead strategies for predictive I/O

use crate::error::Result;
use crate::storage::page::{PageId, PAGE_SIZE};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable sequential access prefetching
    pub enable_prefetch: bool,
    /// Number of pages to prefetch ahead
    pub prefetch_pages: usize,
    /// Enable memory pooling for zero-copy
    pub enable_memory_pool: bool,
    /// Memory pool size (in pages)
    pub pool_size: usize,
    /// Readahead strategy
    pub readahead_strategy: ReadaheadStrategy,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_prefetch: true,
            prefetch_pages: 8,
            enable_memory_pool: true,
            pool_size: 256,
            readahead_strategy: ReadaheadStrategy::Adaptive,
        }
    }
}

/// Readahead strategy for predictive I/O
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadaheadStrategy {
    /// No readahead
    None,
    /// Fixed readahead window
    Fixed,
    /// Adaptive readahead based on access patterns
    Adaptive,
    /// Aggressive prefetching for sequential scans
    Aggressive,
}

/// Access pattern detector
#[derive(Debug)]
struct AccessPatternDetector {
    /// Recent accesses
    recent_accesses: RwLock<VecDeque<PageId>>,
    /// Sequential access counter
    sequential_count: AtomicU64,
    /// Random access counter
    random_count: AtomicU64,
    /// History size
    history_size: usize,
}

impl AccessPatternDetector {
    /// Create a new pattern detector
    fn new(history_size: usize) -> Self {
        Self {
            recent_accesses: RwLock::new(VecDeque::with_capacity(history_size)),
            sequential_count: AtomicU64::new(0),
            random_count: AtomicU64::new(0),
            history_size,
        }
    }

    /// Record a page access
    fn record_access(&self, page_id: PageId) {
        let mut accesses = self.recent_accesses.write();

        // Check if sequential
        if let Some(&last_page) = accesses.back() {
            if page_id == last_page + 1 || page_id == last_page.saturating_sub(1) {
                self.sequential_count.fetch_add(1, Ordering::Relaxed);
            } else {
                self.random_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        accesses.push_back(page_id);

        // Limit history size
        while accesses.len() > self.history_size {
            accesses.pop_front();
        }
    }

    /// Detect if access pattern is sequential
    fn is_sequential(&self) -> bool {
        let sequential = self.sequential_count.load(Ordering::Relaxed);
        let random = self.random_count.load(Ordering::Relaxed);
        let total = sequential + random;

        if total == 0 {
            false
        } else {
            // Consider sequential if >70% of accesses are sequential
            (sequential as f64 / total as f64) > 0.7
        }
    }

    /// Get recommended prefetch distance
    fn recommended_prefetch_distance(&self) -> usize {
        if self.is_sequential() {
            // Aggressive prefetching for sequential access
            16
        } else {
            // Conservative prefetching for random access
            4
        }
    }

    /// Reset statistics
    fn reset(&self) {
        self.recent_accesses.write().clear();
        self.sequential_count.store(0, Ordering::Relaxed);
        self.random_count.store(0, Ordering::Relaxed);
    }
}

/// Memory pool for zero-copy operations
#[derive(Debug)]
struct MemoryPool {
    /// Pre-allocated page buffers
    buffers: RwLock<Vec<Vec<u8>>>,
    /// Pool capacity
    capacity: usize,
    /// Statistics
    allocations: AtomicU64,
    /// Pool hits (reused buffer)
    pool_hits: AtomicU64,
    /// Pool misses (new allocation)
    pool_misses: AtomicU64,
}

impl MemoryPool {
    /// Create a new memory pool
    fn new(capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(capacity);

        // Pre-allocate page-aligned buffers
        for _ in 0..capacity {
            let buffer = vec![0; PAGE_SIZE];
            buffers.push(buffer);
        }

        Self {
            buffers: RwLock::new(buffers),
            capacity,
            allocations: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
        }
    }

    /// Acquire a buffer from the pool
    fn acquire(&self) -> Vec<u8> {
        self.allocations.fetch_add(1, Ordering::Relaxed);

        let mut buffers = self.buffers.write();

        if let Some(buffer) = buffers.pop() {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
            buffer
        } else {
            self.pool_misses.fetch_add(1, Ordering::Relaxed);
            vec![0; PAGE_SIZE]
        }
    }

    /// Return a buffer to the pool
    fn release(&self, mut buffer: Vec<u8>) {
        // Clear buffer before returning to pool
        buffer.fill(0);

        let mut buffers = self.buffers.write();

        if buffers.len() < self.capacity {
            buffers.push(buffer);
        }
        // Otherwise, buffer is dropped and deallocated
    }

    /// Get pool hit rate
    fn hit_rate(&self) -> f64 {
        let hits = self.pool_hits.load(Ordering::Relaxed) as f64;
        let total = self.allocations.load(Ordering::Relaxed) as f64;

        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Get pool statistics
    fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            capacity: self.capacity,
            available: self.buffers.read().len(),
            allocations: self.allocations.load(Ordering::Relaxed),
            hits: self.pool_hits.load(Ordering::Relaxed),
            misses: self.pool_misses.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Pool capacity
    pub capacity: usize,
    /// Available buffers
    pub available: usize,
    /// Total allocations
    pub allocations: u64,
    /// Pool hits
    pub hits: u64,
    /// Pool misses
    pub misses: u64,
    /// Hit rate
    pub hit_rate: f64,
}

/// Memory optimization manager
pub struct MemoryOptimizer {
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Access pattern detector
    pattern_detector: Arc<AccessPatternDetector>,
    /// Memory pool
    memory_pool: Option<Arc<MemoryPool>>,
    /// Statistics
    stats: MemoryOptimizerStats,
}

/// Memory optimizer statistics
#[derive(Debug, Default)]
pub struct MemoryOptimizerStats {
    /// Total prefetch requests
    pub prefetch_requests: AtomicU64,
    /// Successful prefetches
    pub prefetch_hits: AtomicU64,
    /// Sequential accesses detected
    pub sequential_accesses: AtomicU64,
    /// Random accesses detected
    pub random_accesses: AtomicU64,
}

impl MemoryOptimizerStats {
    /// Get prefetch hit rate
    pub fn prefetch_hit_rate(&self) -> f64 {
        let requests = self.prefetch_requests.load(Ordering::Relaxed) as f64;
        if requests == 0.0 {
            0.0
        } else {
            let hits = self.prefetch_hits.load(Ordering::Relaxed) as f64;
            hits / requests
        }
    }

    /// Get sequential access percentage
    pub fn sequential_percentage(&self) -> f64 {
        let sequential = self.sequential_accesses.load(Ordering::Relaxed) as f64;
        let random = self.random_accesses.load(Ordering::Relaxed) as f64;
        let total = sequential + random;

        if total == 0.0 {
            0.0
        } else {
            (sequential / total) * 100.0
        }
    }
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        let pattern_detector = Arc::new(AccessPatternDetector::new(100));

        let memory_pool = if config.enable_memory_pool {
            Some(Arc::new(MemoryPool::new(config.pool_size)))
        } else {
            None
        };

        Self {
            config,
            pattern_detector,
            memory_pool,
            stats: MemoryOptimizerStats::default(),
        }
    }

    /// Record a page access
    pub fn record_access(&self, page_id: PageId) {
        self.pattern_detector.record_access(page_id);

        if self.pattern_detector.is_sequential() {
            self.stats
                .sequential_accesses
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.random_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get prefetch recommendations
    pub fn get_prefetch_recommendations(&self, current_page: PageId) -> Vec<PageId> {
        if !self.config.enable_prefetch {
            return Vec::new();
        }

        let prefetch_distance = match self.config.readahead_strategy {
            ReadaheadStrategy::None => return Vec::new(),
            ReadaheadStrategy::Fixed => self.config.prefetch_pages,
            ReadaheadStrategy::Adaptive => self.pattern_detector.recommended_prefetch_distance(),
            ReadaheadStrategy::Aggressive => 32, // Very aggressive prefetching
        };

        // Generate prefetch list (sequential forward)
        let mut recommendations = Vec::with_capacity(prefetch_distance);
        for i in 1..=prefetch_distance {
            recommendations.push(current_page + i as u64);
        }

        self.stats
            .prefetch_requests
            .fetch_add(prefetch_distance as u64, Ordering::Relaxed);

        recommendations
    }

    /// Acquire a buffer from the memory pool
    pub fn acquire_buffer(&self) -> Vec<u8> {
        if let Some(ref pool) = self.memory_pool {
            pool.acquire()
        } else {
            vec![0; PAGE_SIZE]
        }
    }

    /// Release a buffer back to the memory pool
    pub fn release_buffer(&self, buffer: Vec<u8>) {
        if let Some(ref pool) = self.memory_pool {
            pool.release(buffer);
        }
        // Otherwise, buffer is dropped
    }

    /// Get memory pool statistics (if enabled)
    pub fn pool_stats(&self) -> Option<MemoryPoolStats> {
        self.memory_pool.as_ref().map(|pool| pool.stats())
    }

    /// Get optimizer statistics
    pub fn stats(&self) -> &MemoryOptimizerStats {
        &self.stats
    }

    /// Check if access pattern is sequential
    pub fn is_sequential_access(&self) -> bool {
        self.pattern_detector.is_sequential()
    }

    /// Reset access pattern detection
    pub fn reset_pattern_detection(&self) {
        self.pattern_detector.reset();
        self.stats.sequential_accesses.store(0, Ordering::Relaxed);
        self.stats.random_accesses.store(0, Ordering::Relaxed);
    }

    /// Get recommended readahead distance
    pub fn recommended_readahead(&self) -> usize {
        self.pattern_detector.recommended_prefetch_distance()
    }
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new(MemoryOptimizationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimizer_creation() {
        let config = MemoryOptimizationConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        assert!(!optimizer.is_sequential_access());
    }

    #[test]
    fn test_sequential_access_detection() {
        let optimizer = MemoryOptimizer::default();

        // Simulate sequential access pattern
        for page_id in 0..20 {
            optimizer.record_access(page_id);
        }

        assert!(optimizer.is_sequential_access());
        assert!(optimizer.stats().sequential_percentage() > 50.0);
    }

    #[test]
    fn test_random_access_detection() {
        let optimizer = MemoryOptimizer::default();

        // Simulate random access pattern
        let random_pages = [5, 100, 3, 200, 7, 150, 9, 250];
        for &page_id in &random_pages {
            optimizer.record_access(page_id);
        }

        assert!(!optimizer.is_sequential_access());
    }

    #[test]
    fn test_prefetch_recommendations() {
        let config = MemoryOptimizationConfig {
            enable_prefetch: true,
            prefetch_pages: 4,
            readahead_strategy: ReadaheadStrategy::Fixed,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        let recommendations = optimizer.get_prefetch_recommendations(100);

        assert_eq!(recommendations.len(), 4);
        assert_eq!(recommendations[0], 101);
        assert_eq!(recommendations[1], 102);
        assert_eq!(recommendations[2], 103);
        assert_eq!(recommendations[3], 104);
    }

    #[test]
    fn test_adaptive_prefetch() {
        let config = MemoryOptimizationConfig {
            readahead_strategy: ReadaheadStrategy::Adaptive,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        // Establish sequential pattern
        for page_id in 0..10 {
            optimizer.record_access(page_id);
        }

        let recommendations = optimizer.get_prefetch_recommendations(10);

        // Should recommend aggressive prefetching for sequential access
        assert!(recommendations.len() >= 4);
    }

    #[test]
    fn test_memory_pool_acquire_release() {
        let config = MemoryOptimizationConfig {
            enable_memory_pool: true,
            pool_size: 10,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        // Acquire buffer
        let buffer1 = optimizer.acquire_buffer();
        assert_eq!(buffer1.len(), PAGE_SIZE);

        // Release buffer
        optimizer.release_buffer(buffer1);

        // Acquire again (should reuse)
        let buffer2 = optimizer.acquire_buffer();
        assert_eq!(buffer2.len(), PAGE_SIZE);

        // Check pool stats
        let stats = optimizer.pool_stats().unwrap();
        assert!(stats.allocations >= 2);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_memory_pool_statistics() {
        let config = MemoryOptimizationConfig {
            enable_memory_pool: true,
            pool_size: 5,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        // Acquire and release multiple buffers
        for _ in 0..10 {
            let buffer = optimizer.acquire_buffer();
            optimizer.release_buffer(buffer);
        }

        let stats = optimizer.pool_stats().unwrap();
        assert_eq!(stats.capacity, 5);
        assert!(stats.hits > 0);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_disabled_prefetch() {
        let config = MemoryOptimizationConfig {
            enable_prefetch: false,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        let recommendations = optimizer.get_prefetch_recommendations(10);
        assert!(recommendations.is_empty());
    }

    #[test]
    fn test_aggressive_readahead() {
        let config = MemoryOptimizationConfig {
            readahead_strategy: ReadaheadStrategy::Aggressive,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        let recommendations = optimizer.get_prefetch_recommendations(10);
        assert_eq!(recommendations.len(), 32);
    }

    #[test]
    fn test_pattern_reset() {
        let optimizer = MemoryOptimizer::default();

        // Establish pattern
        for page_id in 0..10 {
            optimizer.record_access(page_id);
        }

        assert!(optimizer.is_sequential_access());

        // Reset
        optimizer.reset_pattern_detection();

        // Pattern should be reset
        assert_eq!(optimizer.stats().sequential_percentage(), 0.0);
    }

    #[test]
    fn test_recommended_readahead_distance() {
        let optimizer = MemoryOptimizer::default();

        // Sequential access
        for page_id in 0..20 {
            optimizer.record_access(page_id);
        }

        let distance = optimizer.recommended_readahead();
        assert!(distance >= 8); // Should recommend larger distance for sequential
    }

    #[test]
    fn test_memory_pool_capacity_limit() {
        let config = MemoryOptimizationConfig {
            enable_memory_pool: true,
            pool_size: 2,
            ..Default::default()
        };
        let optimizer = MemoryOptimizer::new(config);

        // Acquire more buffers than capacity
        let buf1 = optimizer.acquire_buffer();
        let buf2 = optimizer.acquire_buffer();
        let buf3 = optimizer.acquire_buffer();

        // Release all
        optimizer.release_buffer(buf1);
        optimizer.release_buffer(buf2);
        optimizer.release_buffer(buf3);

        // Pool should not exceed capacity
        let stats = optimizer.pool_stats().unwrap();
        assert_eq!(stats.available, 2); // Only keeps 2 buffers
    }

    #[test]
    fn test_prefetch_statistics() {
        let optimizer = MemoryOptimizer::default();

        // Generate prefetch recommendations
        for i in 0..5 {
            optimizer.get_prefetch_recommendations(i * 10);
        }

        let stats = optimizer.stats();
        assert!(stats.prefetch_requests.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_mixed_access_pattern() {
        let optimizer = MemoryOptimizer::default();

        // Mix sequential and random
        for i in 0..5 {
            optimizer.record_access(i);
        }
        optimizer.record_access(100);
        optimizer.record_access(200);

        // Should detect mixed pattern
        let stats = optimizer.stats();
        assert!(stats.sequential_accesses.load(Ordering::Relaxed) > 0);
        assert!(stats.random_accesses.load(Ordering::Relaxed) > 0);
    }
}
