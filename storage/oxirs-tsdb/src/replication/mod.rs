//! Raft-based replication for distributed TSDB
//!
//! This module implements the Raft consensus protocol state machine for
//! replicating time-series write operations across a cluster.
//!
//! ## Modules
//!
//! - `raft_state` -- Core Raft state machine (pure logic, no I/O)
//! - `replication_group` -- Simulated multi-node Raft cluster for testing
//!   and embedded deployments
//! - `wal_replicator` -- WAL → Raft bridge: converts TSDB WAL ops into Raft
//!   log entries and ships them to the cluster via an in-process channel
//! - `snapshot` -- Raft snapshot format, install/restore helpers, and
//!   in-memory snapshot store for log compaction
//!
//! ## References
//!
//! - Ongaro, D. & Ousterhout, J. (2014). *In Search of an Understandable
//!   Consensus Algorithm*. USENIX ATC '14.
//!   <https://raft.github.io/raft.pdf>

pub mod raft_state;
pub mod replication_group;
pub mod snapshot;
pub mod wal_replicator;

pub use raft_state::{
    AppendEntriesArgs, AppendEntriesReply, LogEntry, RaftError, RaftResult, RaftRole, RaftState,
    RequestVoteArgs, RequestVoteReply, SeriesMetadata, TsdbCommand,
};

pub use replication_group::{ReplicationGroup, TsdbRaftNode, WriteEntry};

pub use wal_replicator::{
    ApplyError, ReplicationError, TsdbRaftOp, TsdbStateMachine, WalReplicator,
};

pub use snapshot::{SnapshotDataPoint, SnapshotError, SnapshotStore, TsdbRaftSnapshot};
