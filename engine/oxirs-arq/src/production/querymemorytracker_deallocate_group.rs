//! # QueryMemoryTracker - deallocate_group Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Deallocate memory for a query
    pub fn deallocate(&self, query_id: u64, bytes: usize) {
        let mut allocations = self.query_allocations.write().expect("lock poisoned");
        if let Some(query_usage) = allocations.get_mut(&query_id) {
            let to_free = bytes.min(*query_usage);
            *query_usage -= to_free;
            self.current_usage.fetch_sub(to_free, Ordering::SeqCst);
            if *query_usage == 0 {
                allocations.remove(&query_id);
            }
        }
    }
}
