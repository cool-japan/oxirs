use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use crate::transaction::lock_manager::{LockManager, LockMode};
use crate::transaction::wal::{LogRecord, Lsn, TxnId, WriteAheadLog};
use std::sync::{Arc, RwLock};

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    /// Transaction is active and executing
    Active,
    /// Transaction has been committed
    Committed,
    /// Transaction has been aborted/rolled back
    Aborted,
}

/// Transaction context
pub struct Transaction {
    /// Transaction ID
    txn_id: TxnId,
    /// Transaction state
    state: RwLock<TxnState>,
    /// WAL reference
    wal: Arc<WriteAheadLog>,
    /// Lock manager reference
    lock_manager: Arc<LockManager>,
    /// Begin LSN
    begin_lsn: Lsn,
    /// Read-only flag
    read_only: bool,
}

impl Transaction {
    /// Begin a new transaction
    pub fn begin(
        txn_id: TxnId,
        wal: Arc<WriteAheadLog>,
        lock_manager: Arc<LockManager>,
    ) -> Result<Self> {
        Self::begin_internal(txn_id, wal, lock_manager, false)
    }

    /// Begin a new read-only transaction
    pub fn begin_read_only(
        txn_id: TxnId,
        wal: Arc<WriteAheadLog>,
        lock_manager: Arc<LockManager>,
    ) -> Result<Self> {
        Self::begin_internal(txn_id, wal, lock_manager, true)
    }

    /// Internal method to begin a transaction
    fn begin_internal(
        txn_id: TxnId,
        wal: Arc<WriteAheadLog>,
        lock_manager: Arc<LockManager>,
        read_only: bool,
    ) -> Result<Self> {
        // For read-only transactions, we don't need to write a Begin record
        let begin_lsn = if read_only {
            Lsn::new(0) // Placeholder LSN for read-only transactions
        } else {
            wal.append(LogRecord::Begin { txn_id })?
        };

        Ok(Self {
            txn_id,
            state: RwLock::new(TxnState::Active),
            wal,
            lock_manager,
            begin_lsn,
            read_only,
        })
    }

    /// Get transaction ID
    pub fn id(&self) -> TxnId {
        self.txn_id
    }

    /// Get transaction state
    pub fn state(&self) -> TxnState {
        *self
            .state
            .read()
            .expect("transaction state lock should not be poisoned")
    }

    /// Get begin LSN
    pub fn begin_lsn(&self) -> Lsn {
        self.begin_lsn
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state() == TxnState::Active
    }

    /// Check if transaction is read-only
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Acquire a shared lock
    pub fn lock_shared(&self, page_id: PageId) -> Result<()> {
        if !self.is_active() {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        self.lock_manager
            .lock(self.txn_id, page_id, LockMode::Shared)
    }

    /// Acquire an exclusive lock
    pub fn lock_exclusive(&self, page_id: PageId) -> Result<()> {
        if !self.is_active() {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // Read-only transactions cannot acquire exclusive locks
        if self.read_only {
            return Err(TdbError::Other(format!(
                "Read-only transaction {} cannot acquire exclusive locks",
                self.txn_id.as_u64()
            )));
        }

        self.lock_manager
            .lock(self.txn_id, page_id, LockMode::Exclusive)
    }

    /// Log a page update
    pub fn log_update(
        &self,
        page_id: PageId,
        before_image: Vec<u8>,
        after_image: Vec<u8>,
    ) -> Result<Lsn> {
        if !self.is_active() {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // Read-only transactions cannot log updates
        if self.read_only {
            return Err(TdbError::Other(format!(
                "Read-only transaction {} cannot perform updates",
                self.txn_id.as_u64()
            )));
        }

        self.wal.append(LogRecord::Update {
            txn_id: self.txn_id,
            page_id,
            before_image,
            after_image,
        })
    }

    /// Commit the transaction
    pub fn commit(&self) -> Result<Lsn> {
        let mut state = self
            .state
            .write()
            .expect("transaction state lock should not be poisoned");

        if *state != TxnState::Active {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // For read-only transactions, skip WAL operations
        let commit_lsn = if self.read_only {
            Lsn::new(0) // Placeholder LSN for read-only transactions
        } else {
            // Write commit log
            let lsn = self.wal.append(LogRecord::Commit {
                txn_id: self.txn_id,
            })?;

            // Flush WAL (ensure durability)
            self.wal.flush()?;

            lsn
        };

        // Release all locks
        self.lock_manager.release_all(self.txn_id)?;

        // Update state
        *state = TxnState::Committed;

        Ok(commit_lsn)
    }

    /// Abort the transaction
    pub fn abort(&self) -> Result<Lsn> {
        let mut state = self
            .state
            .write()
            .expect("transaction state lock should not be poisoned");

        if *state != TxnState::Active {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // For read-only transactions, skip WAL operations
        let abort_lsn = if self.read_only {
            Lsn::new(0) // Placeholder LSN for read-only transactions
        } else {
            // Write abort log
            let lsn = self.wal.append(LogRecord::Abort {
                txn_id: self.txn_id,
            })?;

            // Flush WAL
            self.wal.flush()?;

            lsn
        };

        // Release all locks
        self.lock_manager.release_all(self.txn_id)?;

        // Update state
        *state = TxnState::Aborted;

        Ok(abort_lsn)
    }
}

/// Transaction Manager
pub struct TransactionManager {
    /// WAL
    wal: Arc<WriteAheadLog>,
    /// Lock manager
    lock_manager: Arc<LockManager>,
    /// Next transaction ID
    next_txn_id: RwLock<TxnId>,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new(wal: Arc<WriteAheadLog>, lock_manager: Arc<LockManager>) -> Self {
        Self {
            wal,
            lock_manager,
            next_txn_id: RwLock::new(TxnId::new(1)),
        }
    }

    /// Begin a new transaction
    pub fn begin(&self) -> Result<Transaction> {
        let mut next_txn_id = self
            .next_txn_id
            .write()
            .expect("next_txn_id lock should not be poisoned");
        let txn_id = *next_txn_id;
        *next_txn_id = next_txn_id.next();

        Transaction::begin(
            txn_id,
            Arc::clone(&self.wal),
            Arc::clone(&self.lock_manager),
        )
    }

    /// Begin a new read-only transaction
    pub fn begin_read(&self) -> Result<Transaction> {
        let mut next_txn_id = self
            .next_txn_id
            .write()
            .expect("next_txn_id lock should not be poisoned");
        let txn_id = *next_txn_id;
        *next_txn_id = next_txn_id.next();

        Transaction::begin_read_only(
            txn_id,
            Arc::clone(&self.wal),
            Arc::clone(&self.lock_manager),
        )
    }

    /// Checkpoint (write active transactions to WAL)
    pub fn checkpoint(&self, active_txns: Vec<TxnId>) -> Result<Lsn> {
        let lsn = self.wal.append(LogRecord::Checkpoint { active_txns })?;
        self.wal.flush()?;
        Ok(lsn)
    }

    /// Get WAL reference
    pub fn wal(&self) -> &Arc<WriteAheadLog> {
        &self.wal
    }

    /// Get lock manager reference
    pub fn lock_manager(&self) -> &Arc<LockManager> {
        &self.lock_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_transaction_begin() {
        let temp_dir = env::temp_dir().join("oxirs_txn_begin");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin(TxnId::new(1), wal, lock_manager).unwrap();

        assert_eq!(txn.id().as_u64(), 1);
        assert_eq!(txn.state(), TxnState::Active);
        assert!(txn.is_active());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_commit() {
        let temp_dir = env::temp_dir().join("oxirs_txn_commit");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin(TxnId::new(1), wal, lock_manager).unwrap();
        txn.commit().unwrap();

        assert_eq!(txn.state(), TxnState::Committed);
        assert!(!txn.is_active());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_abort() {
        let temp_dir = env::temp_dir().join("oxirs_txn_abort");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin(TxnId::new(1), wal, lock_manager).unwrap();
        txn.abort().unwrap();

        assert_eq!(txn.state(), TxnState::Aborted);
        assert!(!txn.is_active());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_locks() {
        let temp_dir = env::temp_dir().join("oxirs_txn_locks");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn =
            Transaction::begin(TxnId::new(1), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();

        let page1: PageId = 1;
        txn.lock_shared(page1).unwrap();

        assert!(lock_manager.has_locks(txn.id()));

        txn.commit().unwrap();

        assert!(!lock_manager.has_locks(txn.id()));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_log_update() {
        let temp_dir = env::temp_dir().join("oxirs_txn_log_update");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin(TxnId::new(1), wal, lock_manager).unwrap();

        let page1: PageId = 1;
        let before = vec![1, 2, 3];
        let after = vec![4, 5, 6];

        let lsn = txn.log_update(page1, before, after).unwrap();
        assert!(lsn.as_u64() > 0);

        txn.commit().unwrap();

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_manager() {
        let temp_dir = env::temp_dir().join("oxirs_txn_manager");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let tm = TransactionManager::new(wal, lock_manager);

        let txn1 = tm.begin().unwrap();
        let txn2 = tm.begin().unwrap();

        assert_eq!(txn1.id().as_u64(), 1);
        assert_eq!(txn2.id().as_u64(), 2);

        txn1.commit().unwrap();
        txn2.commit().unwrap();

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_manager_checkpoint() {
        let temp_dir = env::temp_dir().join("oxirs_txn_checkpoint");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let tm = TransactionManager::new(wal, lock_manager);

        let txn1 = tm.begin().unwrap();
        let txn2 = tm.begin().unwrap();

        let active = vec![txn1.id(), txn2.id()];
        tm.checkpoint(active).unwrap();

        txn1.commit().unwrap();
        txn2.commit().unwrap();

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_not_active_error() {
        let temp_dir = env::temp_dir().join("oxirs_txn_not_active");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin(TxnId::new(1), wal, lock_manager).unwrap();
        txn.commit().unwrap();

        // Try to commit again
        let result = txn.commit();
        assert!(result.is_err());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_isolation() {
        let temp_dir = env::temp_dir().join("oxirs_txn_isolation");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn1 =
            Transaction::begin(TxnId::new(1), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();
        let txn2 =
            Transaction::begin(TxnId::new(2), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();

        let page1: PageId = 1;

        // Txn1 gets exclusive lock
        txn1.lock_exclusive(page1).unwrap();

        // Txn2 cannot get lock (will timeout or wait)
        let result = lock_manager.try_lock(txn2.id(), page1, LockMode::Shared);
        assert!(result.is_ok());
        assert!(!result.unwrap());

        txn1.commit().unwrap();
        txn2.commit().unwrap();

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_read_only_transaction() {
        let temp_dir = env::temp_dir().join("oxirs_txn_readonly");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin_read_only(TxnId::new(1), wal, lock_manager).unwrap();

        assert_eq!(txn.id().as_u64(), 1);
        assert_eq!(txn.state(), TxnState::Active);
        assert!(txn.is_active());
        assert!(txn.is_read_only());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_read_only_transaction_cannot_acquire_exclusive_lock() {
        let temp_dir = env::temp_dir().join("oxirs_txn_readonly_lock");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin_read_only(TxnId::new(1), wal, lock_manager).unwrap();

        let page1: PageId = 1;

        // Shared lock should work
        txn.lock_shared(page1).unwrap();

        // Exclusive lock should fail
        let result = txn.lock_exclusive(page1);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Read-only transaction"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_read_only_transaction_cannot_log_updates() {
        let temp_dir = env::temp_dir().join("oxirs_txn_readonly_update");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin_read_only(TxnId::new(1), wal, lock_manager).unwrap();

        let page1: PageId = 1;
        let before = vec![1, 2, 3];
        let after = vec![4, 5, 6];

        // Log update should fail
        let result = txn.log_update(page1, before, after);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Read-only transaction"));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_read_only_transaction_commit() {
        let temp_dir = env::temp_dir().join("oxirs_txn_readonly_commit");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let txn = Transaction::begin_read_only(TxnId::new(1), wal, lock_manager).unwrap();

        // Commit should succeed without writing to WAL
        txn.commit().unwrap();
        assert_eq!(txn.state(), TxnState::Committed);
        assert!(!txn.is_active());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_transaction_manager_begin_read() {
        let temp_dir = env::temp_dir().join("oxirs_txn_manager_read");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let wal = Arc::new(WriteAheadLog::new(&temp_dir).unwrap());
        let lock_manager = Arc::new(LockManager::new());

        let tm = TransactionManager::new(wal, lock_manager);

        let txn1 = tm.begin_read().unwrap();
        let txn2 = tm.begin_read().unwrap();

        assert_eq!(txn1.id().as_u64(), 1);
        assert_eq!(txn2.id().as_u64(), 2);
        assert!(txn1.is_read_only());
        assert!(txn2.is_read_only());

        txn1.commit().unwrap();
        txn2.commit().unwrap();

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
