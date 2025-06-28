//! Distributed storage and consensus modules
//!
//! This module contains distributed system components for OxiRS:
//! - Raft consensus with optimized log compaction
//! - Multi-region active-active replication
//! - CRDTs for conflict-free replicated RDF
//! - Byzantine fault tolerance for untrusted environments
//! - Semantic-aware sharding for distributed storage
//! - Cross-shard transactions with 2PC optimization

pub mod bft;
pub mod crdt;
pub mod raft;
pub mod replication;
pub mod sharding;
pub mod transaction;

pub use bft::{BftConfig, BftNode};
pub use crdt::{CrdtConfig, RdfCrdt};
pub use raft::{RaftConfig, RaftNode};
pub use replication::{ReplicationConfig, ReplicationManager};
pub use sharding::{ShardManager, ShardRouter, ShardingConfig, ShardingStrategy};
pub use transaction::{TransactionConfig, TransactionCoordinator, TransactionId};
