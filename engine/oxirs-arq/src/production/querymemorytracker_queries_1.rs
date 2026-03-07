//! # QueryMemoryTracker - queries Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Get usage for a specific query
    pub fn query_usage(&self, query_id: u64) -> usize {
        self.query_allocations
            .read()
            .expect("lock poisoned")
            .get(&query_id)
            .copied()
            .unwrap_or(0)
    }
}
