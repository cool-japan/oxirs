//! Transaction support for TDB storage engine
//!
//! Provides ACID transaction support with:
//! - Write-Ahead Logging (WAL) for durability
//! - Two-Phase Locking (2PL) for isolation
//! - Transaction context for atomicity and consistency
//! - Group commit optimization for improved throughput
//! - Two-Phase Commit (2PC) for distributed transactions
//! - Three-Phase Commit (3PC) for enhanced reliability

/// Conflict resolution and deadlock detection
pub mod conflict;
/// Group commit optimization for WAL
pub mod group_commit;
/// Lock management for transaction isolation
pub mod lock_manager;
/// Three-Phase Commit protocol for distributed transactions
pub mod three_phase_commit;
/// Two-Phase Commit protocol for distributed transactions
pub mod two_phase_commit;
/// Transaction context and manager
pub mod txn_context;
/// Write-Ahead Log for durability
pub mod wal;
/// WAL optimizer with batching, compression, and group commit
pub mod wal_optimizer;

pub use conflict::{ConflictManager, ConflictStrategy, DeadlockDetection};
pub use group_commit::{GroupCommitConfig, GroupCommitCoordinator, GroupCommitStats};
pub use lock_manager::{LockManager, LockMode};
pub use three_phase_commit::{
    ThreePhaseCoordinator, ThreePhaseParticipant, ThreePhaseStats, TpcPhase,
};
pub use two_phase_commit::{
    Participant, TpcCoordinatorStats, TpcParticipantStats, TpcState, TwoPhaseCoordinator,
    TwoPhaseParticipant, Vote,
};
pub use txn_context::{Transaction, TransactionManager, TxnState};
pub use wal::{LogEntry, LogRecord, Lsn, TxnId, WriteAheadLog};
pub use wal_optimizer::{WalOptimizer, WalOptimizerConfig, WalOptimizerStats};
