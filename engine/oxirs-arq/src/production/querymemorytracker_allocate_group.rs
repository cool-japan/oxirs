//! # QueryMemoryTracker - allocate_group Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};
use std::sync::atomic::Ordering;

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    /// Allocate memory for a query
    pub fn allocate(&self, query_id: u64, bytes: usize) -> Result<()> {
        let per_query_limit = self.per_query_limit.load(Ordering::Relaxed);
        let memory_limit = self.memory_limit.load(Ordering::Relaxed);
        let mut allocations = self.query_allocations.write().expect("lock poisoned");
        let current_query_usage = allocations.get(&query_id).copied().unwrap_or(0);
        if current_query_usage + bytes > per_query_limit {
            return Err(anyhow!(
                "Query {} memory allocation {} would exceed per-query limit of {}",
                query_id,
                current_query_usage + bytes,
                per_query_limit
            ));
        }
        let current = self.current_usage.load(Ordering::Relaxed);
        if current + bytes > memory_limit {
            return Err(anyhow!(
                "Global memory limit {} would be exceeded (current: {}, requested: {})",
                memory_limit,
                current,
                bytes
            ));
        }
        *allocations.entry(query_id).or_insert(0) += bytes;
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::SeqCst) + bytes;
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                new_usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        Ok(())
    }
}
