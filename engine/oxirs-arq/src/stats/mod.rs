//! Runtime statistics collection for adaptive query feedback.
//!
//! This module exposes `RuntimeStatsCollector`, `QueryExecutionStats`, and
//! `PatternStats` for recording per-query execution data and feeding it back
//! to the optimizer.

pub mod runtime_stats;

pub use runtime_stats::{PatternStats, QueryExecutionStats, RuntimeStatsCollector};
