//! Distributed storage and consensus modules
//!
//! This module contains distributed system components for OxiRS:
//! - Raft consensus with optimized log compaction
//! - Multi-region active-active replication
//! - CRDTs for conflict-free replicated RDF

pub mod crdt;
pub mod raft;
pub mod replication;

pub use crdt::{CrdtConfig, RdfCrdt};
pub use raft::{RaftConfig, RaftNode};
pub use replication::{ReplicationConfig, ReplicationManager};
