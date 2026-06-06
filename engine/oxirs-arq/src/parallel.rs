//! Parallel Query Execution — facade re-exporting all parallel sub-modules.
//!
//! Implementation is split across:
//! - [`crate::parallel_types`]   — statistics structs and iterator traits
//! - [`crate::parallel_executor`] — thread pool, query executor, work-stealing queue
//! - [`crate::parallel_planner`]  — partition strategies and cost-based plan helpers

pub use crate::parallel_executor::{
    ParallelExecutor, ParallelQueryExecutor, ParallelScanIterator, WorkStealingQueue,
};
pub use crate::parallel_planner::{
    compute_chunk_size, select_bgp_partition_strategy, BgpPartitionStrategy, ParallelPlanCost,
};
pub use crate::parallel_types::{ParallelSolutionIterator, ParallelStats};
