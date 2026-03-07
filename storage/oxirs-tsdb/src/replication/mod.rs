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
//!
//! ## References
//!
//! - Ongaro, D. & Ousterhout, J. (2014). *In Search of an Understandable
//!   Consensus Algorithm*. USENIX ATC '14.
//!   <https://raft.github.io/raft.pdf>

pub mod raft_state;
pub mod replication_group;

pub use raft_state::{
    AppendEntriesArgs, AppendEntriesReply, LogEntry, RaftError, RaftResult, RaftRole, RaftState,
    RequestVoteArgs, RequestVoteReply, SeriesMetadata, TsdbCommand,
};

pub use replication_group::{ReplicationGroup, TsdbRaftNode, WriteEntry};
