//! Read/write transaction management for OxiRS Fuseki.
//!
//! Provides [`TransactionManager`] which tracks active transactions,
//! enforces read/write isolation, limits concurrency, and expires timed-out
//! transactions.

use std::collections::HashMap;

/// Distinguishes how a transaction accesses the dataset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransactionMode {
    /// Read-only — no writes allowed.
    Read,
    /// Write — mutations allowed.
    Write,
    /// Read-write — mutations allowed (alias for [`Write`](TransactionMode::Write)).
    ReadWrite,
}

impl TransactionMode {
    /// Returns `true` if this mode permits write operations.
    pub fn allows_writes(&self) -> bool {
        matches!(self, TransactionMode::Write | TransactionMode::ReadWrite)
    }
}

/// Current lifecycle state of a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransactionState {
    /// In progress.
    Active,
    /// Successfully committed.
    Committed,
    /// Rolled back by the client.
    Aborted,
    /// Killed by the timeout policy.
    TimedOut,
}

/// A write operation staged inside a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteOp {
    /// Insert a triple, optionally into a named graph.
    Insert(String, String, String, Option<String>),
    /// Delete a triple, optionally from a named graph.
    Delete(String, String, String, Option<String>),
    /// Clear an entire graph (`None` = default graph).
    Clear(Option<String>),
}

/// A single in-progress or completed transaction.
#[derive(Clone, Debug)]
pub struct Transaction {
    /// Unique transaction identifier.
    pub id: u64,
    /// Access mode.
    pub mode: TransactionMode,
    /// Lifecycle state.
    pub state: TransactionState,
    /// Unix timestamp (milliseconds) when the transaction started.
    pub started_at: u64,
    /// Maximum duration in milliseconds before the transaction times out.
    pub timeout_ms: u64,
    writes: Vec<WriteOp>,
}

impl Transaction {
    fn new(id: u64, mode: TransactionMode, started_at: u64, timeout_ms: u64) -> Self {
        Self {
            id,
            mode,
            state: TransactionState::Active,
            started_at,
            timeout_ms,
            writes: Vec::new(),
        }
    }

    /// Returns `true` if this transaction is still [`Active`](TransactionState::Active).
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }

    /// Number of staged write operations.
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }

    /// Check whether `current_time_ms` has exceeded the transaction's deadline.
    pub fn is_timed_out(&self, current_time_ms: u64) -> bool {
        self.timeout_ms > 0 && current_time_ms.saturating_sub(self.started_at) >= self.timeout_ms
    }
}

/// Errors returned by [`TransactionManager`] operations.
#[derive(Debug, PartialEq, Eq)]
pub enum TxError {
    /// No transaction exists with the given id.
    NotFound(u64),
    /// The transaction was already committed.
    AlreadyCommitted(u64),
    /// The transaction was already aborted.
    AlreadyAborted(u64),
    /// A write was attempted on a read-only transaction.
    ReadOnly(u64),
    /// Concurrent write limit reached.
    MaxConcurrentWritesExceeded,
    /// A write conflict was detected (reserved for future SSI).
    WriteConflict,
    /// The transaction timed out before the operation was attempted.
    TimedOut(u64),
}

impl std::fmt::Display for TxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TxError::NotFound(id) => write!(f, "Transaction {} not found", id),
            TxError::AlreadyCommitted(id) => write!(f, "Transaction {} already committed", id),
            TxError::AlreadyAborted(id) => write!(f, "Transaction {} already aborted", id),
            TxError::ReadOnly(id) => write!(f, "Transaction {} is read-only", id),
            TxError::MaxConcurrentWritesExceeded => {
                write!(f, "Maximum concurrent write transactions exceeded")
            }
            TxError::WriteConflict => write!(f, "Write conflict detected"),
            TxError::TimedOut(id) => write!(f, "Transaction {} timed out", id),
        }
    }
}

impl std::error::Error for TxError {}

/// Manages the lifecycle of read and write transactions.
pub struct TransactionManager {
    next_id: u64,
    active: HashMap<u64, Transaction>,
    max_concurrent_writes: usize,
    default_timeout_ms: u64,
}

impl TransactionManager {
    /// Create a new manager.
    ///
    /// * `max_concurrent_writes` — maximum number of simultaneous write
    ///   transactions (0 means unlimited).
    /// * `default_timeout_ms` — default deadline in milliseconds (0 means
    ///   no timeout).
    pub fn new(max_concurrent_writes: usize, default_timeout_ms: u64) -> Self {
        Self {
            next_id: 1,
            active: HashMap::new(),
            max_concurrent_writes,
            default_timeout_ms,
        }
    }

    /// Begin a new transaction with the given mode.
    ///
    /// The `started_at` timestamp is taken as `0` (callers should use
    /// `expire_timed_out` with a real clock).  Returns the new transaction id.
    pub fn begin(&mut self, mode: TransactionMode) -> Result<u64, TxError> {
        self.begin_at(mode, 0)
    }

    /// Begin a transaction anchored to a specific timestamp (Unix ms).
    pub fn begin_at(&mut self, mode: TransactionMode, current_time_ms: u64) -> Result<u64, TxError> {
        // Enforce concurrent-writes limit.
        if mode.allows_writes() && self.max_concurrent_writes > 0 {
            let active_writes = self.write_count();
            if active_writes >= self.max_concurrent_writes {
                return Err(TxError::MaxConcurrentWritesExceeded);
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        let tx = Transaction::new(id, mode, current_time_ms, self.default_timeout_ms);
        self.active.insert(id, tx);
        Ok(id)
    }

    /// Commit the transaction, returning the staged write operations to apply.
    pub fn commit(&mut self, tx_id: u64) -> Result<Vec<WriteOp>, TxError> {
        let tx = self.active.get_mut(&tx_id).ok_or(TxError::NotFound(tx_id))?;

        match tx.state {
            TransactionState::Active => {}
            TransactionState::Committed => return Err(TxError::AlreadyCommitted(tx_id)),
            TransactionState::Aborted => return Err(TxError::AlreadyAborted(tx_id)),
            TransactionState::TimedOut => return Err(TxError::TimedOut(tx_id)),
        }

        tx.state = TransactionState::Committed;
        let ops = tx.writes.drain(..).collect();
        Ok(ops)
    }

    /// Abort the transaction, discarding all staged writes.
    pub fn abort(&mut self, tx_id: u64) -> Result<(), TxError> {
        let tx = self.active.get_mut(&tx_id).ok_or(TxError::NotFound(tx_id))?;

        match tx.state {
            TransactionState::Active => {}
            TransactionState::Committed => return Err(TxError::AlreadyCommitted(tx_id)),
            TransactionState::Aborted => return Err(TxError::AlreadyAborted(tx_id)),
            TransactionState::TimedOut => return Err(TxError::TimedOut(tx_id)),
        }

        tx.state = TransactionState::Aborted;
        tx.writes.clear();
        Ok(())
    }

    /// Stage a write operation inside the transaction.
    pub fn add_write(&mut self, tx_id: u64, op: WriteOp) -> Result<(), TxError> {
        let tx = self.active.get_mut(&tx_id).ok_or(TxError::NotFound(tx_id))?;

        match tx.state {
            TransactionState::Active => {}
            TransactionState::Committed => return Err(TxError::AlreadyCommitted(tx_id)),
            TransactionState::Aborted => return Err(TxError::AlreadyAborted(tx_id)),
            TransactionState::TimedOut => return Err(TxError::TimedOut(tx_id)),
        }

        if !tx.mode.allows_writes() {
            return Err(TxError::ReadOnly(tx_id));
        }

        tx.writes.push(op);
        Ok(())
    }

    /// Get a shared reference to a transaction by id.
    pub fn get_transaction(&self, tx_id: u64) -> Option<&Transaction> {
        self.active.get(&tx_id)
    }

    /// Number of active (not yet committed / aborted) transactions.
    pub fn active_count(&self) -> usize {
        self.active
            .values()
            .filter(|tx| tx.state == TransactionState::Active)
            .count()
    }

    /// Number of active write transactions.
    pub fn write_count(&self) -> usize {
        self.active
            .values()
            .filter(|tx| tx.state == TransactionState::Active && tx.mode.allows_writes())
            .count()
    }

    /// Mark all active transactions whose deadline has passed as
    /// [`TimedOut`](TransactionState::TimedOut).
    ///
    /// Returns the ids of transactions that were expired.
    pub fn expire_timed_out(&mut self, current_time_ms: u64) -> Vec<u64> {
        let mut expired = Vec::new();
        for tx in self.active.values_mut() {
            if tx.state == TransactionState::Active && tx.is_timed_out(current_time_ms) {
                tx.state = TransactionState::TimedOut;
                tx.writes.clear();
                expired.push(tx.id);
            }
        }
        expired
    }

    /// Returns `true` if a transaction with `tx_id` exists and is
    /// [`Active`](TransactionState::Active).
    pub fn is_active(&self, tx_id: u64) -> bool {
        self.active
            .get(&tx_id)
            .is_some_and(|tx| tx.state == TransactionState::Active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> TransactionManager {
        TransactionManager::new(10, 0)
    }

    // ── begin ───────────────────────────────────────────────────────────────

    #[test]
    fn test_begin_read() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        assert_eq!(id, 1);
        assert!(m.is_active(id));
    }

    #[test]
    fn test_begin_write() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        assert!(m.is_active(id));
    }

    #[test]
    fn test_begin_readwrite() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::ReadWrite).unwrap();
        assert!(m.is_active(id));
    }

    #[test]
    fn test_begin_increments_id() {
        let mut m = mgr();
        let id1 = m.begin(TransactionMode::Read).unwrap();
        let id2 = m.begin(TransactionMode::Read).unwrap();
        let id3 = m.begin(TransactionMode::Write).unwrap();
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[test]
    fn test_active_count() {
        let mut m = mgr();
        assert_eq!(m.active_count(), 0);
        m.begin(TransactionMode::Read).unwrap();
        m.begin(TransactionMode::Write).unwrap();
        assert_eq!(m.active_count(), 2);
    }

    // ── commit ──────────────────────────────────────────────────────────────

    #[test]
    fn test_commit_read_transaction() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        let ops = m.commit(id).unwrap();
        assert!(ops.is_empty());
        assert!(!m.is_active(id));
    }

    #[test]
    fn test_commit_returns_write_ops() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        )
        .unwrap();
        let ops = m.commit(id).unwrap();
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn test_commit_already_committed() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        m.commit(id).unwrap();
        let result = m.commit(id);
        assert_eq!(result, Err(TxError::AlreadyCommitted(id)));
    }

    #[test]
    fn test_commit_already_aborted() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        let result = m.commit(id);
        assert_eq!(result, Err(TxError::AlreadyAborted(id)));
    }

    #[test]
    fn test_commit_not_found() {
        let mut m = mgr();
        let result = m.commit(999);
        assert_eq!(result, Err(TxError::NotFound(999)));
    }

    // ── abort ───────────────────────────────────────────────────────────────

    #[test]
    fn test_abort_active_transaction() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        )
        .unwrap();
        m.abort(id).unwrap();
        assert!(!m.is_active(id));
        // Commit after abort is an error.
        assert_eq!(m.commit(id), Err(TxError::AlreadyAborted(id)));
    }

    #[test]
    fn test_abort_already_aborted() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        m.abort(id).unwrap();
        assert_eq!(m.abort(id), Err(TxError::AlreadyAborted(id)));
    }

    #[test]
    fn test_abort_already_committed() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        m.commit(id).unwrap();
        assert_eq!(m.abort(id), Err(TxError::AlreadyCommitted(id)));
    }

    #[test]
    fn test_abort_not_found() {
        let mut m = mgr();
        assert_eq!(m.abort(999), Err(TxError::NotFound(999)));
    }

    // ── add_write ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_write_to_write_transaction() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        )
        .unwrap();
        let tx = m.get_transaction(id).unwrap();
        assert_eq!(tx.write_count(), 1);
    }

    #[test]
    fn test_add_write_to_read_transaction_fails() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        let result = m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        );
        assert_eq!(result, Err(TxError::ReadOnly(id)));
    }

    #[test]
    fn test_add_write_multiple_ops() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        for i in 0..5u32 {
            m.add_write(
                id,
                WriteOp::Insert(
                    format!("s{}", i),
                    "p".into(),
                    format!("o{}", i),
                    None,
                ),
            )
            .unwrap();
        }
        let tx = m.get_transaction(id).unwrap();
        assert_eq!(tx.write_count(), 5);
    }

    #[test]
    fn test_add_write_clear_op() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(id, WriteOp::Clear(None)).unwrap();
        let ops = m.commit(id).unwrap();
        assert_eq!(ops[0], WriteOp::Clear(None));
    }

    #[test]
    fn test_add_write_delete_op() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(
            id,
            WriteOp::Delete("s".into(), "p".into(), "o".into(), Some("g".into())),
        )
        .unwrap();
        let ops = m.commit(id).unwrap();
        assert_eq!(
            ops[0],
            WriteOp::Delete("s".into(), "p".into(), "o".into(), Some("g".into()))
        );
    }

    #[test]
    fn test_add_write_after_commit_fails() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.commit(id).unwrap();
        let result = m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        );
        assert_eq!(result, Err(TxError::AlreadyCommitted(id)));
    }

    // ── max concurrent writes ───────────────────────────────────────────────

    #[test]
    fn test_max_concurrent_writes_respected() {
        let mut m = TransactionManager::new(2, 0);
        m.begin(TransactionMode::Write).unwrap();
        m.begin(TransactionMode::Write).unwrap();
        let result = m.begin(TransactionMode::Write);
        assert_eq!(result, Err(TxError::MaxConcurrentWritesExceeded));
    }

    #[test]
    fn test_read_allowed_when_writes_at_limit() {
        let mut m = TransactionManager::new(1, 0);
        m.begin(TransactionMode::Write).unwrap();
        // A read transaction should still be allowed.
        let id = m.begin(TransactionMode::Read).unwrap();
        assert!(m.is_active(id));
    }

    #[test]
    fn test_write_allowed_after_commit_frees_slot() {
        let mut m = TransactionManager::new(1, 0);
        let id = m.begin(TransactionMode::Write).unwrap();
        m.commit(id).unwrap();
        // Slot freed — should succeed.
        let id2 = m.begin(TransactionMode::Write).unwrap();
        assert!(m.is_active(id2));
    }

    #[test]
    fn test_write_count() {
        let mut m = mgr();
        let r = m.begin(TransactionMode::Read).unwrap();
        let w = m.begin(TransactionMode::Write).unwrap();
        assert_eq!(m.write_count(), 1);
        m.commit(w).unwrap();
        assert_eq!(m.write_count(), 0);
        m.commit(r).unwrap();
    }

    #[test]
    fn test_unlimited_writes_when_max_zero() {
        let mut m = TransactionManager::new(0, 0);
        for _ in 0..50 {
            m.begin(TransactionMode::Write).unwrap();
        }
        assert_eq!(m.write_count(), 50);
    }

    // ── timeout ─────────────────────────────────────────────────────────────

    #[test]
    fn test_expire_timed_out() {
        let mut m = TransactionManager::new(10, 1000);
        let id = m.begin_at(TransactionMode::Write, 0).unwrap();
        // Expire at t=2000 (past 1000ms deadline).
        let expired = m.expire_timed_out(2000);
        assert_eq!(expired, vec![id]);
        assert!(!m.is_active(id));
    }

    #[test]
    fn test_no_expiry_before_deadline() {
        let mut m = TransactionManager::new(10, 5000);
        let _id = m.begin_at(TransactionMode::Write, 0).unwrap();
        let expired = m.expire_timed_out(4999);
        assert!(expired.is_empty());
    }

    #[test]
    fn test_expire_timed_out_returns_multiple() {
        let mut m = TransactionManager::new(10, 500);
        let id1 = m.begin_at(TransactionMode::Write, 0).unwrap();
        let id2 = m.begin_at(TransactionMode::Read, 0).unwrap();
        let mut expired = m.expire_timed_out(1000);
        expired.sort();
        assert!(expired.contains(&id1));
        assert!(expired.contains(&id2));
    }

    #[test]
    fn test_commit_after_timeout_fails() {
        let mut m = TransactionManager::new(10, 500);
        let id = m.begin_at(TransactionMode::Write, 0).unwrap();
        m.expire_timed_out(1000);
        assert_eq!(m.commit(id), Err(TxError::TimedOut(id)));
    }

    #[test]
    fn test_no_timeout_when_zero() {
        let mut m = TransactionManager::new(10, 0);
        let id = m.begin_at(TransactionMode::Write, 0).unwrap();
        let expired = m.expire_timed_out(u64::MAX);
        assert!(expired.is_empty());
        assert!(m.is_active(id));
    }

    // ── get_transaction ─────────────────────────────────────────────────────

    #[test]
    fn test_get_transaction_exists() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        let tx = m.get_transaction(id);
        assert!(tx.is_some());
        assert_eq!(tx.unwrap().id, id);
    }

    #[test]
    fn test_get_transaction_missing() {
        let m = mgr();
        assert!(m.get_transaction(999).is_none());
    }

    #[test]
    fn test_transaction_mode_preserved() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::ReadWrite).unwrap();
        let tx = m.get_transaction(id).unwrap();
        assert_eq!(tx.mode, TransactionMode::ReadWrite);
    }

    // ── is_active ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_active_after_commit() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        m.commit(id).unwrap();
        assert!(!m.is_active(id));
    }

    #[test]
    fn test_is_active_unknown_id() {
        let m = mgr();
        assert!(!m.is_active(42));
    }

    // ── display ─────────────────────────────────────────────────────────────

    #[test]
    fn test_tx_error_display() {
        assert!(!TxError::NotFound(1).to_string().is_empty());
        assert!(!TxError::AlreadyCommitted(1).to_string().is_empty());
        assert!(!TxError::AlreadyAborted(1).to_string().is_empty());
        assert!(!TxError::ReadOnly(1).to_string().is_empty());
        assert!(!TxError::MaxConcurrentWritesExceeded.to_string().is_empty());
        assert!(!TxError::WriteConflict.to_string().is_empty());
        assert!(!TxError::TimedOut(1).to_string().is_empty());
    }

    // ── additional corner cases ──────────────────────────────────────────────

    #[test]
    fn test_abort_after_timeout() {
        let mut m = TransactionManager::new(10, 500);
        let id = m.begin_at(TransactionMode::Write, 0).unwrap();
        m.expire_timed_out(1000);
        assert_eq!(m.abort(id), Err(TxError::TimedOut(id)));
    }

    #[test]
    fn test_add_write_after_abort() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        let result = m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        );
        assert_eq!(result, Err(TxError::AlreadyAborted(id)));
    }

    #[test]
    fn test_add_write_not_found() {
        let mut m = mgr();
        let result = m.add_write(
            999,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        );
        assert_eq!(result, Err(TxError::NotFound(999)));
    }

    #[test]
    fn test_is_active_after_abort() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        assert!(!m.is_active(id));
    }

    #[test]
    fn test_active_count_decrements_after_commit() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        assert_eq!(m.active_count(), 1);
        m.commit(id).unwrap();
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_active_count_decrements_after_abort() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_write_count_after_abort() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        assert_eq!(m.write_count(), 0);
    }

    #[test]
    fn test_transaction_state_active() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        assert_eq!(
            m.get_transaction(id).unwrap().state,
            TransactionState::Active
        );
    }

    #[test]
    fn test_transaction_state_committed() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Read).unwrap();
        m.commit(id).unwrap();
        assert_eq!(
            m.get_transaction(id).unwrap().state,
            TransactionState::Committed
        );
    }

    #[test]
    fn test_transaction_state_aborted() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.abort(id).unwrap();
        assert_eq!(
            m.get_transaction(id).unwrap().state,
            TransactionState::Aborted
        );
    }

    #[test]
    fn test_transaction_mode_allows_writes() {
        assert!(TransactionMode::Write.allows_writes());
        assert!(TransactionMode::ReadWrite.allows_writes());
        assert!(!TransactionMode::Read.allows_writes());
    }

    #[test]
    fn test_commit_clears_write_ops() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        for i in 0..3u32 {
            m.add_write(
                id,
                WriteOp::Insert(format!("s{}", i), "p".into(), "o".into(), None),
            )
            .unwrap();
        }
        let ops = m.commit(id).unwrap();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn test_abort_clears_write_ops() {
        let mut m = mgr();
        let id = m.begin(TransactionMode::Write).unwrap();
        m.add_write(
            id,
            WriteOp::Insert("s".into(), "p".into(), "o".into(), None),
        )
        .unwrap();
        m.abort(id).unwrap();
        assert_eq!(m.get_transaction(id).unwrap().write_count(), 0);
    }
}
