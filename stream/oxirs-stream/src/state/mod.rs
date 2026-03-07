//! # State Management for Stream Operators
//!
//! Provides distributed, fault-tolerant state stores for stateful stream
//! processing.
//!
//! ## Modules
//!
//! - [`legacy`]: Original `StateStore` trait, `MemoryStateStore`,
//!   `StateProcessor`, and helper patterns from the v0.1 release.
//! - [`distributed_state`]: Low-level `StateBackend` trait plus
//!   `InMemoryStateBackend`, `KeyedStateStore`, and `AggregatingState`.
//! - [`exactly_once`]: Exactly-once processing via deduplication log and
//!   atomic transactions.

pub mod distributed;
pub mod distributed_state;
pub mod exactly_once;
pub mod legacy;

pub use distributed::{
    DistributedStateStore, PartitionStateValue, StateAggregator, StateCoordinator, StatePartition,
};
pub use distributed_state::{
    AggregatingState, InMemoryStateBackend, KeyedStateStore,
    StateBackend as DistributedStateBackend, StateBackendStats, StatePartitionKey,
};
pub use exactly_once::{
    DeduplicationConfig, DeduplicationLog, ExactlyOnceProcessor, ExactlyOnceStats,
    ExactlyOnceTransaction, MessageId,
};

// Re-export legacy API for backwards compatibility
pub use legacy::{
    MemoryStateStore, StateBackend, StateConfig, StateOperation, StateOperationType,
    StateProcessor, StateProcessorBuilder, StateStatistics, StateStore, StateValue,
};
