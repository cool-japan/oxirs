//! Transaction support for TDB storage engine
//!
//! Provides ACID transaction support with:
//! - Write-Ahead Logging (WAL) for durability
//! - Two-Phase Locking (2PL) for isolation
//! - Transaction context for atomicity and consistency

/// Lock management for transaction isolation
pub mod lock_manager;
/// Transaction context and manager
pub mod transaction;
/// Write-Ahead Log for durability
pub mod wal;

pub use lock_manager::{LockManager, LockMode};
pub use transaction::{Transaction, TransactionManager, TxnState};
pub use wal::{LogEntry, LogRecord, Lsn, TxnId, WriteAheadLog};
