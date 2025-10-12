//! Distributed storage implementation for Raft consensus
//!
//! This module provides persistent storage for Raft with:
//! - Write-ahead logging (WAL) for durability
//! - Snapshot support for log compaction
//! - Crash recovery and integrity verification
//! - Checksumming for data integrity

pub mod config;
pub mod error;
#[cfg(any(test, debug_assertions))]
pub mod mock;
pub mod persistent;
pub mod recovery;
pub mod stats;
pub mod types;

// Re-export main types
pub use config::StorageConfig;
pub use error::StorageError;
pub use persistent::{PersistentStorage, StorageBackend};
pub use recovery::{
    CorruptionReport, LogConsistencyReport, LogInconsistency, RecoveryReport,
    StateConsistencyReport,
};
pub use stats::StorageStats;
pub use types::{ChecksummedData, RaftState, SnapshotMetadata, WalEntry, WalOperation};
