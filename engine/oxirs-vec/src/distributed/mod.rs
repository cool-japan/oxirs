//! Distributed vector index infrastructure
//!
//! This module provides:
//! - Raft-based consensus for distributed index management
//! - Cross-datacenter async replication with configurable lag tolerance
//! - Conflict resolution for divergent cross-DC writes
//! - Delta synchronisation for bandwidth-efficient replication

pub mod conflict_resolver;
pub mod cross_dc;
pub mod delta_sync;
pub mod raft_index;

// Re-export key types
pub use conflict_resolver::{
    ConflictPolicy, ConflictResolver, IndexVersion, MergedIndex, Resolution,
};
pub use cross_dc::{
    ConflictRecord, ConflictResolutionStrategy, CrossDcConfig, CrossDcCoordinator, CrossDcStats,
    PrimaryDcManager, ReplicaDcManager, ReplicaStatus, ReplicationEntry, ReplicationHealth,
    ReplicationOperation, ReplicationSeq,
};
pub use delta_sync::{
    DeltaSync, IndexDelta, IndexSnapshot, ReplicationAlert, ReplicationLag,
    VectorEntry as DeltaVectorEntry,
};
pub use raft_index::{
    AppendEntriesRequest, AppendEntriesResponse, ClusterSimulator, IndexCommand, LogEntry, NodeId,
    NodeRole, RaftConfig, RaftIndexNode, RaftStats, RequestVoteRequest, RequestVoteResponse, Term,
    VectorEntry,
};
