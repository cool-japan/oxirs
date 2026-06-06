//! Parallel execution types, statistics, and traits.

use crate::algebra::{Binding, Solution};
use anyhow::Result;

/// Parallel execution statistics
#[derive(Debug, Default)]
pub struct ParallelStats {
    pub parallel_operations: usize,
    pub work_items_processed: usize,
    pub thread_utilization: f64,
    pub parallel_speedup: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Parallel iterator for solution processing
pub trait ParallelSolutionIterator: Send + Sync {
    fn par_process<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> Option<Binding> + Send + Sync;

    fn par_filter<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> bool + Send + Sync;

    fn par_extend<F>(&self, f: F) -> Result<Solution>
    where
        F: Fn(&Binding) -> Vec<Binding> + Send + Sync;
}
