//! # QueryMemoryTracker - Trait Implementations
//!
//! This module contains trait implementations for `QueryMemoryTracker`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;

use super::querymemorytracker_type::QueryMemoryTracker;

impl Default for QueryMemoryTracker {
    fn default() -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            memory_limit: AtomicUsize::new(1 << 30),
            per_query_limit: AtomicUsize::new(256 << 20),
            query_allocations: RwLock::new(HashMap::new()),
            pressure_threshold: RwLock::new(0.8),
        }
    }
}
