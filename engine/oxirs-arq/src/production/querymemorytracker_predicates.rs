//! # QueryMemoryTracker - predicates Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Check if under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        let current = self.current_usage.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);
        let threshold = *self.pressure_threshold.read().expect("lock poisoned");
        (current as f64 / limit as f64) > threshold
    }
}
