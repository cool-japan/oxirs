//! # QueryMemoryTracker - queries Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Free all memory for a completed query
    pub fn free_query(&self, query_id: u64) -> usize {
        let mut allocations = self.query_allocations.write().expect("lock poisoned");
        if let Some(freed) = allocations.remove(&query_id) {
            self.current_usage.fetch_sub(freed, Ordering::SeqCst);
            freed
        } else {
            0
        }
    }
}
