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
//! - [`raft_state`]: W3-S11 Raft-backed operator state via the cluster sink
//!   bridge ([`oxirs_cluster::streaming::cluster_sink::StreamSink`]).
//! - [`linearizable_reader`]: W3-S11 linearizable reads on top of Raft state.

pub mod distributed;
pub mod distributed_state;
pub mod exactly_once;
pub mod legacy;
pub mod linearizable_reader;
pub mod raft_state;

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
pub use linearizable_reader::{
    LinearizableReadConfig, LinearizableReadError, LinearizableReadResult, LinearizableReader,
};
pub use raft_state::{
    RaftBackedOperatorState, RaftStateConfig, RaftStateError, RaftStateResult, RaftStateStats,
    RaftStateStatsSnapshot, StateValue as RaftStateValue,
};

// Re-export legacy API for backwards compatibility
pub use legacy::{
    MemoryStateStore, StateBackend, StateConfig, StateOperation, StateOperationType,
    StateProcessor, StateProcessorBuilder, StateStatistics, StateStore, StateValue,
};
