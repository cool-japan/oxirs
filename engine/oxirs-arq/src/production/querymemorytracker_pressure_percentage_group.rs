//! # QueryMemoryTracker - pressure_percentage_group Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::types::MemoryStats;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Get memory pressure as a percentage
    pub fn pressure_percentage(&self) -> f64 {
        let current = self.current_usage.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);
        if limit == 0 {
            0.0
        } else {
            (current as f64 / limit as f64) * 100.0
        }
    }
    /// Get detailed memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            memory_limit: self.memory_limit.load(Ordering::Relaxed),
            active_queries: self.query_allocations.read().expect("lock poisoned").len(),
            pressure_percentage: self.pressure_percentage(),
        }
    }
}
