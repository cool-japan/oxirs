//! # QueryMemoryTracker - reset_stats_group Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Reset statistics
    pub fn reset_stats(&self) {
        self.peak_usage.store(
            self.current_usage.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}
