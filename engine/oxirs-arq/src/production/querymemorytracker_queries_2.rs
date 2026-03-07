//! # QueryMemoryTracker - queries Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    pub fn set_per_query_limit(&self, limit: usize) {
        self.per_query_limit.store(limit, Ordering::Relaxed);
    }
}
