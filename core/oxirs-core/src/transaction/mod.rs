//! ACID transaction support with Write-Ahead Logging
//!
//! This module provides full ACID (Atomicity, Consistency, Isolation, Durability)
//! transaction support for RDF operations using:
//!
//! - **Atomicity**: All-or-nothing commit semantics
//! - **Consistency**: Validation before commit
//! - **Isolation**: Snapshot isolation with MVCC
//! - **Durability**: Write-Ahead Logging (WAL) for crash recovery
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_core::transaction::{TransactionManager, IsolationLevel};
//! use oxirs_core::model::Quad;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! let mut tx_mgr = TransactionManager::new("./wal")?;
//!
//! // Begin a transaction with snapshot isolation
//! let mut tx = tx_mgr.begin(IsolationLevel::Snapshot)?;
//!
//! // Perform operations
//! // tx.insert(quad)?;
//! // tx.remove(quad)?;
//!
//! // Commit with durability guarantee
//! tx.commit()?;
//! # Ok(())
//! # }
//! ```

pub mod acid_transaction;
pub mod named_graph;
pub mod snapshot;
pub mod wal;

pub use acid_transaction::{AcidTransaction, TransactionId, TransactionState};
pub use named_graph::{GraphStats, NamedGraphTransaction};
pub use snapshot::{MvccSnapshot, SnapshotManager, VersionedQuad};
pub use wal::{WalEntry, WalRecovery, WalValidation, WriteAheadLog};

use crate::model::{GraphName, Quad};
use crate::OxirsError;
use ahash::AHashSet;
use scirs2_core::metrics::{Counter, Timer};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Isolation level for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted (no isolation)
    ReadUncommitted,
    /// Read committed (prevents dirty reads)
    ReadCommitted,
    /// Repeatable read (prevents non-repeatable reads)
    RepeatableRead,
    /// Snapshot isolation (MVCC-based, prevents all anomalies)
    Snapshot,
    /// Serializable (strongest isolation)
    Serializable,
}

/// Transaction manager with ACID guarantees
pub struct TransactionManager {
    /// Next transaction ID
    next_tx_id: Arc<AtomicU64>,
    /// Write-Ahead Log
    wal: Arc<RwLock<WriteAheadLog>>,
    /// Snapshot manager for MVCC
    snapshot_mgr: Arc<RwLock<SnapshotManager>>,
    /// Active transactions
    active_transactions: Arc<RwLock<Vec<TransactionId>>>,
    /// Graph-level locks for named graph transactions
    graph_locks: Arc<RwLock<AHashSet<GraphName>>>,
    /// Metrics
    commit_counter: Arc<Counter>,
    abort_counter: Arc<Counter>,
    commit_timer: Arc<Timer>,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new(wal_dir: impl AsRef<Path>) -> Result<Self, OxirsError> {
        let wal = WriteAheadLog::new(wal_dir)?;

        Ok(Self {
            next_tx_id: Arc::new(AtomicU64::new(1)),
            wal: Arc::new(RwLock::new(wal)),
            snapshot_mgr: Arc::new(RwLock::new(SnapshotManager::new())),
            active_transactions: Arc::new(RwLock::new(Vec::new())),
            graph_locks: Arc::new(RwLock::new(AHashSet::new())),
            commit_counter: Arc::new(Counter::new("transaction.commits".to_string())),
            abort_counter: Arc::new(Counter::new("transaction.aborts".to_string())),
            commit_timer: Arc::new(Timer::new("transaction.commit_time".to_string())),
        })
    }

    /// Begin a new transaction with specified isolation level
    pub fn begin(&mut self, isolation: IsolationLevel) -> Result<AcidTransaction, OxirsError> {
        let tx_id = TransactionId(self.next_tx_id.fetch_add(1, Ordering::SeqCst));

        // Create snapshot for MVCC
        let snapshot = match isolation {
            IsolationLevel::Snapshot | IsolationLevel::Serializable => {
                let mut snapshot_mgr = self
                    .snapshot_mgr
                    .write()
                    .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;
                Some(snapshot_mgr.create_snapshot(tx_id))
            }
            _ => None,
        };

        // Register as active transaction
        let mut active = self
            .active_transactions
            .write()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;
        active.push(tx_id);

        Ok(AcidTransaction::new(
            tx_id,
            isolation,
            snapshot,
            self.wal.clone(),
            self.commit_counter.clone(),
            self.abort_counter.clone(),
            self.commit_timer.clone(),
        ))
    }

    /// Begin a new named graph transaction with specified isolation level
    ///
    /// This provides graph-specific operations with ACID guarantees:
    /// - Atomic multi-graph operations
    /// - Graph-level isolation
    /// - MVCC snapshot isolation per graph
    pub fn begin_named_graph_transaction(
        &mut self,
        isolation: IsolationLevel,
    ) -> Result<NamedGraphTransaction, OxirsError> {
        let inner = self.begin(isolation)?;
        Ok(NamedGraphTransaction::new(inner, self.graph_locks.clone()))
    }

    /// Recover from WAL after crash
    ///
    /// This method replays committed transactions from the Write-Ahead Log
    /// using callback functions to apply operations to the store.
    ///
    /// # Arguments
    ///
    /// * `insert_fn` - Callback to insert a quad into the store
    /// * `delete_fn` - Callback to delete a quad from the store
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tx_mgr = TransactionManager::new("./wal")?;
    /// let mut store = RdfStore::new()?;
    ///
    /// tx_mgr.recover(
    ///     |quad| store.insert_quad(quad),
    ///     |quad| store.remove_quad(quad)
    /// )?;
    /// ```
    pub fn recover<F, G>(&mut self, insert_fn: F, delete_fn: G) -> Result<usize, OxirsError>
    where
        F: FnMut(Quad) -> Result<bool, OxirsError>,
        G: FnMut(&Quad) -> Result<bool, OxirsError>,
    {
        // Create a recovery handler from the WAL path
        let wal_guard = self
            .wal
            .read()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        // Get the WAL path to create a recovery handler
        let wal_path = wal_guard.path().to_path_buf();
        drop(wal_guard);

        let recovery = WalRecovery::new(&wal_path)?;
        recovery.replay(insert_fn, delete_fn)
    }

    /// Checkpoint WAL (write buffered entries to disk)
    pub fn checkpoint(&mut self) -> Result<(), OxirsError> {
        let mut wal = self
            .wal
            .write()
            .map_err(|_| OxirsError::ConcurrencyError("Lock poisoned".to_string()))?;

        wal.checkpoint()
    }

    /// Get transaction statistics
    pub fn stats(&self) -> TransactionStats {
        TransactionStats {
            total_commits: self.commit_counter.get(),
            total_aborts: self.abort_counter.get(),
            active_count: self
                .active_transactions
                .read()
                .ok()
                .map(|a| a.len())
                .unwrap_or(0),
            avg_commit_time_ms: {
                let timer_stats = self.commit_timer.get_stats();
                timer_stats.mean * 1000.0
            },
        }
    }
}

/// Transaction statistics
#[derive(Debug, Clone)]
pub struct TransactionStats {
    /// Total committed transactions
    pub total_commits: u64,
    /// Total aborted transactions
    pub total_aborts: u64,
    /// Active transactions count
    pub active_count: usize,
    /// Average commit time in milliseconds
    pub avg_commit_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_transaction_manager_creation() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let tx_mgr = TransactionManager::new(dir.path())?;

        let stats = tx_mgr.stats();
        assert_eq!(stats.total_commits, 0);
        assert_eq!(stats.total_aborts, 0);

        Ok(())
    }

    #[test]
    fn test_begin_transaction() -> Result<(), OxirsError> {
        let dir = tempdir().map_err(|e| OxirsError::Io(e.to_string()))?;
        let mut tx_mgr = TransactionManager::new(dir.path())?;

        let tx = tx_mgr.begin(IsolationLevel::Snapshot)?;
        assert_eq!(tx.id().0, 1);

        Ok(())
    }
}
