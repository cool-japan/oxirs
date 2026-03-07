//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;

/// Memory usage tracker for queries
///
/// Tracks memory allocation and provides pressure-based throttling.
pub struct QueryMemoryTracker {
    pub(super) current_usage: AtomicUsize,
    pub(super) peak_usage: AtomicUsize,
    pub(super) memory_limit: AtomicUsize,
    pub(super) per_query_limit: AtomicUsize,
    pub(super) query_allocations: RwLock<HashMap<u64, usize>>,
    pub(super) pressure_threshold: RwLock<f64>,
}
