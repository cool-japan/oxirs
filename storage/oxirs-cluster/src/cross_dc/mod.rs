//! Cross-datacenter replication
//!
//! Provides async replication with configurable consistency levels,
//! replication stream management, and acknowledgement tracking.
//!
//! The `enhanced` sub-module adds topology-aware placement, synchronisation
//! orchestration, and bulk batch replication support.

pub mod enhanced;
pub mod replication;

pub use enhanced::{
    BatchReplicator, CrossDcReplicationPolicy, CrossDcSyncManager, DcHealthStatus, DcTopology,
    FlushStats, SyncResult,
};
pub use replication::{
    ConsistencyLevel, CrossDcReplicationManager, ReplicationOp, ReplicationStream, ReplicationUnit,
    StreamState,
};
