//! Performance statistics for HNSW operations

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Performance statistics for HNSW operations
#[derive(Debug)]
pub struct HnswPerformanceStats {
    pub total_searches: AtomicU64,
    pub total_insertions: AtomicU64,
    pub avg_search_time_us: AtomicU64, // Store as microseconds, will convert to f64 when needed
    pub avg_distance_calculations: AtomicU64, // Store as integer, will convert to f64 when needed
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub simd_operations: AtomicU64,
    pub parallel_searches: AtomicU64,
    pub parallel_operations: AtomicU64,
    pub prefetch_operations: AtomicU64,
    pub memory_allocations: AtomicU64,
    pub lock_contentions: AtomicU64,
}

impl Default for HnswPerformanceStats {
    fn default() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            total_insertions: AtomicU64::new(0),
            avg_search_time_us: AtomicU64::new(0),
            avg_distance_calculations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            simd_operations: AtomicU64::new(0),
            parallel_searches: AtomicU64::new(0),
            parallel_operations: AtomicU64::new(0),
            prefetch_operations: AtomicU64::new(0),
            memory_allocations: AtomicU64::new(0),
            lock_contentions: AtomicU64::new(0),
        }
    }
}

impl Clone for HnswPerformanceStats {
    fn clone(&self) -> Self {
        Self {
            total_searches: AtomicU64::new(self.total_searches.load(AtomicOrdering::Relaxed)),
            total_insertions: AtomicU64::new(self.total_insertions.load(AtomicOrdering::Relaxed)),
            avg_search_time_us: AtomicU64::new(
                self.avg_search_time_us.load(AtomicOrdering::Relaxed),
            ),
            avg_distance_calculations: AtomicU64::new(
                self.avg_distance_calculations.load(AtomicOrdering::Relaxed),
            ),
            cache_hits: AtomicU64::new(self.cache_hits.load(AtomicOrdering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(AtomicOrdering::Relaxed)),
            simd_operations: AtomicU64::new(self.simd_operations.load(AtomicOrdering::Relaxed)),
            parallel_searches: AtomicU64::new(self.parallel_searches.load(AtomicOrdering::Relaxed)),
            parallel_operations: AtomicU64::new(
                self.parallel_operations.load(AtomicOrdering::Relaxed),
            ),
            prefetch_operations: AtomicU64::new(
                self.prefetch_operations.load(AtomicOrdering::Relaxed),
            ),
            memory_allocations: AtomicU64::new(
                self.memory_allocations.load(AtomicOrdering::Relaxed),
            ),
            lock_contentions: AtomicU64::new(self.lock_contentions.load(AtomicOrdering::Relaxed)),
        }
    }
}

impl HnswPerformanceStats {
    /// Get total searches as u64
    pub fn get_total_searches(&self) -> u64 {
        self.total_searches.load(AtomicOrdering::Relaxed)
    }

    /// Get average search time as f64 microseconds
    pub fn get_avg_search_time_us(&self) -> f64 {
        self.avg_search_time_us.load(AtomicOrdering::Relaxed) as f64
    }

    /// Get average distance calculations as f64
    pub fn get_avg_distance_calculations(&self) -> f64 {
        self.avg_distance_calculations.load(AtomicOrdering::Relaxed) as f64
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(AtomicOrdering::Relaxed);
        let misses = self.cache_misses.load(AtomicOrdering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }

    /// Get average search time in microseconds
    pub fn avg_search_time(&self) -> u64 {
        self.avg_search_time_us.load(AtomicOrdering::Relaxed)
    }

    /// Get parallel operation efficiency ratio
    pub fn parallel_efficiency(&self) -> f64 {
        let total = self.total_searches.load(AtomicOrdering::Relaxed);
        let parallel = self.parallel_operations.load(AtomicOrdering::Relaxed);
        if total == 0 {
            0.0
        } else {
            parallel as f64 / total as f64
        }
    }
}