//! Transaction support for TDB storage engine
//!
//! Provides ACID transaction support with:
//! - Write-Ahead Logging (WAL) for durability
//! - Two-Phase Locking (2PL) for isolation
//! - Transaction context for atomicity and consistency
//! - Group commit optimization for improved throughput

/// Conflict resolution and deadlock detection
pub mod conflict;
/// Group commit optimization for WAL
pub mod group_commit;
/// Lock management for transaction isolation
pub mod lock_manager;
/// Transaction context and manager
pub mod txn_context;
/// Write-Ahead Log for durability
pub mod wal;

pub use conflict::{ConflictManager, ConflictStrategy, DeadlockDetection};
pub use group_commit::{GroupCommitConfig, GroupCommitCoordinator, GroupCommitStats};
pub use lock_manager::{LockManager, LockMode};
pub use txn_context::{Transaction, TransactionManager, TxnState};
pub use wal::{LogEntry, LogRecord, Lsn, TxnId, WriteAheadLog};
