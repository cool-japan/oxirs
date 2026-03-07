//! # QueryMemoryTracker - new_group Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::AtomicUsize;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    pub fn new(memory_limit: usize, per_query_limit: usize) -> Self {
        Self {
            memory_limit: AtomicUsize::new(memory_limit),
            per_query_limit: AtomicUsize::new(per_query_limit),
            ..Default::default()
        }
    }
}
