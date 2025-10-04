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
}

impl Transaction {
    /// Begin a new transaction
    pub fn begin(
        txn_id: TxnId,
        wal: Arc<WriteAheadLog>,
        lock_manager: Arc<LockManager>,
    ) -> Result<Self> {
        let begin_lsn = wal.append(LogRecord::Begin { txn_id })?;

        Ok(Self {
            txn_id,
            state: RwLock::new(TxnState::Active),
            wal,
            lock_manager,
            begin_lsn,
        })
    }

    /// Get transaction ID
    pub fn id(&self) -> TxnId {
        self.txn_id
    }

    /// Get transaction state
    pub fn state(&self) -> TxnState {
        *self.state.read().unwrap()
    }

    /// Get begin LSN
    pub fn begin_lsn(&self) -> Lsn {
        self.begin_lsn
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state() == TxnState::Active
    }

    /// Acquire a shared lock
    pub fn lock_shared(&self, page_id: PageId) -> Result<()> {
        if !self.is_active() {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        self.lock_manager.lock(self.txn_id, page_id, LockMode::Shared)
    }

    /// Acquire an exclusive lock
    pub fn lock_exclusive(&self, page_id: PageId) -> Result<()> {
        if !self.is_active() {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
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

        self.wal.append(LogRecord::Update {
            txn_id: self.txn_id,
            page_id,
            before_image,
            after_image,
        })
    }

    /// Commit the transaction
    pub fn commit(&self) -> Result<Lsn> {
        let mut state = self.state.write().unwrap();

        if *state != TxnState::Active {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // Write commit log
        let commit_lsn = self.wal.append(LogRecord::Commit {
            txn_id: self.txn_id,
        })?;

        // Flush WAL (ensure durability)
        self.wal.flush()?;

        // Release all locks
        self.lock_manager.release_all(self.txn_id)?;

        // Update state
        *state = TxnState::Committed;

        Ok(commit_lsn)
    }

    /// Abort the transaction
    pub fn abort(&self) -> Result<Lsn> {
        let mut state = self.state.write().unwrap();

        if *state != TxnState::Active {
            return Err(TdbError::TransactionNotActive {
                txn_id: self.txn_id.as_u64(),
            });
        }

        // Write abort log
        let abort_lsn = self.wal.append(LogRecord::Abort {
            txn_id: self.txn_id,
        })?;

        // Flush WAL
        self.wal.flush()?;

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
        let mut next_txn_id = self.next_txn_id.write().unwrap();
        let txn_id = *next_txn_id;
        *next_txn_id = next_txn_id.next();

        Transaction::begin(txn_id, Arc::clone(&self.wal), Arc::clone(&self.lock_manager))
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

        let txn = Transaction::begin(TxnId::new(1), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();

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

        let txn1 = Transaction::begin(TxnId::new(1), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();
        let txn2 = Transaction::begin(TxnId::new(2), Arc::clone(&wal), Arc::clone(&lock_manager)).unwrap();

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
}
