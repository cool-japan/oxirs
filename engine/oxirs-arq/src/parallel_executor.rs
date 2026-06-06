//! Parallel query executor, scan iterator, and work-stealing queue.
//!
//! This module provides `ParallelQueryExecutor` for multi-threaded SPARQL
//! algebra evaluation, `ParallelScanIterator` for partitioned dataset scans,
//! and `WorkStealingQueue<T>` for dynamic load-balancing across worker threads.
//!
//! This module is a thin facade. The implementation is split across:
//! - [`crate::parallel_executor_engine`] — the [`ParallelQueryExecutor`] type,
//!   thread pool, algebra dispatch, and BGP/join/union/filter/order-by/group-by.
//! - [`crate::parallel_executor_ops`]    — property paths, optional/minus joins,
//!   federation, projection, and slicing operators.
//! - [`crate::parallel_executor_queue`]  — [`ParallelScanIterator`] and
//!   [`WorkStealingQueue`].

pub use crate::parallel_executor_engine::{ParallelExecutor, ParallelQueryExecutor};
pub use crate::parallel_executor_queue::{ParallelScanIterator, WorkStealingQueue};
