//! # Watermark- and exactly-once-aware aggregation operators
//!
//! Complements the wall-clock based aggregation in
//! [`crate::processing::aggregation`] with operators that integrate with
//! the operator-parallelism + checkpointing model:
//!
//! * [`exactly_once::ExactlyOnceAggregator`] — wraps a per-partition
//!   aggregation state with the deduplication / transaction primitives in
//!   [`crate::state::exactly_once`] so that re-deliveries do not double-count.

pub mod exactly_once;

pub use exactly_once::{
    ExactlyOnceAggregator, ExactlyOnceAggregatorConfig, ExactlyOnceAggregatorStats,
    PartitionAggregateState, PartitionAggregateValue,
};
