//! Fault tolerance modules for OxiRS vector index.
//!
//! - [`replica_manager`]: Shard replica tracking, promotion, and failover.

pub mod replica_manager;

pub use replica_manager::{ReplicaManager, ReplicaState, ReplicationStatus, ShardReplica};
