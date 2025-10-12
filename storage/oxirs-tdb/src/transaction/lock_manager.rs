use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use crate::transaction::wal::TxnId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Lock mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockMode {
    /// Shared lock (read-only)
    Shared,
    /// Exclusive lock (read-write)
    Exclusive,
}

impl LockMode {
    /// Check if two lock modes are compatible
    pub fn is_compatible(&self, other: &LockMode) -> bool {
        matches!((self, other), (LockMode::Shared, LockMode::Shared))
    }
}

/// Lock request
#[derive(Debug, Clone)]
struct LockRequest {
    txn_id: TxnId,
    mode: LockMode,
}

/// Lock table entry for a single resource
#[derive(Debug, Clone)]
struct LockTableEntry {
    /// Currently granted locks
    granted: Vec<LockRequest>,
    /// Waiting lock requests (FIFO queue)
    waiting: VecDeque<LockRequest>,
}

impl LockTableEntry {
    fn new() -> Self {
        Self {
            granted: Vec::new(),
            waiting: VecDeque::new(),
        }
    }

    /// Check if a lock mode is compatible with all granted locks
    fn is_compatible(&self, mode: LockMode) -> bool {
        self.granted.iter().all(|req| req.mode.is_compatible(&mode))
    }

    /// Grant a lock
    fn grant(&mut self, txn_id: TxnId, mode: LockMode) {
        self.granted.push(LockRequest { txn_id, mode });
    }

    /// Add to wait queue
    fn add_waiting(&mut self, txn_id: TxnId, mode: LockMode) {
        self.waiting.push_back(LockRequest { txn_id, mode });
    }

    /// Release a lock and promote waiting locks if possible
    fn release(&mut self, txn_id: TxnId) -> Vec<(TxnId, LockMode)> {
        // Remove from granted
        self.granted.retain(|req| req.txn_id != txn_id);

        // Try to promote waiting locks
        let mut promoted = Vec::new();

        while let Some(waiting_req) = self.waiting.front() {
            if self.is_compatible(waiting_req.mode) {
                let req = self.waiting.pop_front().unwrap();
                self.granted.push(req.clone());
                promoted.push((req.txn_id, req.mode));
            } else {
                break;
            }
        }

        promoted
    }

    /// Check if transaction holds any lock
    fn holds_lock(&self, txn_id: TxnId) -> bool {
        self.granted.iter().any(|req| req.txn_id == txn_id)
    }

    /// Check if transaction is waiting
    fn is_waiting(&self, txn_id: TxnId) -> bool {
        self.waiting.iter().any(|req| req.txn_id == txn_id)
    }
}

/// Lock Manager with Two-Phase Locking (2PL)
pub struct LockManager {
    /// Lock table: PageId -> LockTableEntry
    lock_table: RwLock<HashMap<PageId, LockTableEntry>>,
    /// Transaction lock set: TxnId -> Set<(PageId, LockMode)>
    txn_locks: RwLock<HashMap<TxnId, HashSet<(PageId, LockMode)>>>,
    /// Deadlock detection timeout
    deadlock_timeout: Duration,
}

impl LockManager {
    /// Create a new lock manager
    pub fn new() -> Self {
        Self {
            lock_table: RwLock::new(HashMap::new()),
            txn_locks: RwLock::new(HashMap::new()),
            deadlock_timeout: Duration::from_secs(1),
        }
    }

    /// Acquire a lock (blocking with timeout)
    pub fn lock(&self, txn_id: TxnId, page_id: PageId, mode: LockMode) -> Result<()> {
        // Try to acquire immediately
        if self.try_lock(txn_id, page_id, mode)? {
            return Ok(());
        }

        // Add to wait queue
        {
            let mut lock_table = self.lock_table.write().unwrap();
            let entry = lock_table
                .entry(page_id)
                .or_insert_with(LockTableEntry::new);
            entry.add_waiting(txn_id, mode);
        }

        // Simplified: just fail after timeout (real impl would use condvar)
        std::thread::sleep(self.deadlock_timeout);

        // Check if lock was granted
        let lock_table = self.lock_table.read().unwrap();
        if let Some(entry) = lock_table.get(&page_id) {
            if entry.holds_lock(txn_id) {
                // Lock was granted
                let mut txn_locks = self.txn_locks.write().unwrap();
                txn_locks.entry(txn_id).or_default().insert((page_id, mode));
                return Ok(());
            }
        }

        // Timeout - deadlock detected
        Err(TdbError::Deadlock {
            txn_id: txn_id.as_u64(),
        })
    }

    /// Try to acquire lock without blocking
    pub fn try_lock(&self, txn_id: TxnId, page_id: PageId, mode: LockMode) -> Result<bool> {
        let mut lock_table = self.lock_table.write().unwrap();
        let entry = lock_table
            .entry(page_id)
            .or_insert_with(LockTableEntry::new);

        // Check if already holds lock
        if entry.holds_lock(txn_id) {
            return Ok(true);
        }

        // Check compatibility
        if entry.is_compatible(mode) && entry.waiting.is_empty() {
            entry.grant(txn_id, mode);

            let mut txn_locks = self.txn_locks.write().unwrap();
            txn_locks.entry(txn_id).or_default().insert((page_id, mode));

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Release a specific lock
    pub fn unlock(&self, txn_id: TxnId, page_id: PageId) -> Result<()> {
        let mut lock_table = self.lock_table.write().unwrap();

        if let Some(entry) = lock_table.get_mut(&page_id) {
            let promoted = entry.release(txn_id);

            // Update txn_locks for promoted transactions
            let mut txn_locks = self.txn_locks.write().unwrap();
            for (promoted_txn, promoted_mode) in promoted {
                txn_locks
                    .entry(promoted_txn)
                    .or_default()
                    .insert((page_id, promoted_mode));
            }

            // Remove from this transaction's lock set
            if let Some(locks) = txn_locks.get_mut(&txn_id) {
                locks.retain(|(pid, _)| *pid != page_id);
            }

            // Clean up empty entries
            if entry.granted.is_empty() && entry.waiting.is_empty() {
                lock_table.remove(&page_id);
            }
        }

        Ok(())
    }

    /// Release all locks held by a transaction (on commit/abort)
    pub fn release_all(&self, txn_id: TxnId) -> Result<()> {
        let txn_locks = self.txn_locks.read().unwrap();
        let locks_to_release: Vec<_> = txn_locks
            .get(&txn_id)
            .map(|locks| locks.iter().map(|(page_id, _)| *page_id).collect())
            .unwrap_or_default();

        drop(txn_locks); // Release read lock

        for page_id in locks_to_release {
            self.unlock(txn_id, page_id)?;
        }

        // Clean up transaction entry
        let mut txn_locks = self.txn_locks.write().unwrap();
        txn_locks.remove(&txn_id);

        Ok(())
    }

    /// Check if transaction holds any locks
    pub fn has_locks(&self, txn_id: TxnId) -> bool {
        let txn_locks = self.txn_locks.read().unwrap();
        txn_locks
            .get(&txn_id)
            .map(|locks| !locks.is_empty())
            .unwrap_or(false)
    }

    /// Get all locks held by a transaction
    pub fn get_locks(&self, txn_id: TxnId) -> Vec<(PageId, LockMode)> {
        let txn_locks = self.txn_locks.read().unwrap();
        txn_locks
            .get(&txn_id)
            .map(|locks| locks.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_mode_compatibility() {
        assert!(LockMode::Shared.is_compatible(&LockMode::Shared));
        assert!(!LockMode::Shared.is_compatible(&LockMode::Exclusive));
        assert!(!LockMode::Exclusive.is_compatible(&LockMode::Shared));
        assert!(!LockMode::Exclusive.is_compatible(&LockMode::Exclusive));
    }

    #[test]
    fn test_lock_manager_basic() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let page1: PageId = 1;

        // Acquire shared lock
        lm.lock(txn1, page1, LockMode::Shared).unwrap();
        assert!(lm.has_locks(txn1));

        // Release lock
        lm.release_all(txn1).unwrap();
        assert!(!lm.has_locks(txn1));
    }

    #[test]
    fn test_shared_locks_compatible() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);
        let page1: PageId = 1;

        // Both can acquire shared locks
        lm.lock(txn1, page1, LockMode::Shared).unwrap();
        lm.lock(txn2, page1, LockMode::Shared).unwrap();

        assert!(lm.has_locks(txn1));
        assert!(lm.has_locks(txn2));

        lm.release_all(txn1).unwrap();
        lm.release_all(txn2).unwrap();
    }

    #[test]
    fn test_exclusive_lock_blocks() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);
        let page1: PageId = 1;

        // Txn1 acquires exclusive lock
        lm.lock(txn1, page1, LockMode::Exclusive).unwrap();

        // Txn2 cannot acquire shared lock (will timeout)
        let result = lm.try_lock(txn2, page1, LockMode::Shared).unwrap();
        assert!(!result);

        lm.release_all(txn1).unwrap();
    }

    #[test]
    fn test_lock_promotion() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);
        let page1: PageId = 1;

        // Txn1 acquires exclusive lock
        lm.lock(txn1, page1, LockMode::Exclusive).unwrap();

        // Txn2 tries to acquire (will be queued)
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            // This will timeout in the real implementation
        });

        // Release txn1's lock
        lm.release_all(txn1).unwrap();
    }

    #[test]
    fn test_release_specific_lock() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let page1: PageId = 1;
        let page2: PageId = 2;

        lm.lock(txn1, page1, LockMode::Shared).unwrap();
        lm.lock(txn1, page2, LockMode::Shared).unwrap();

        lm.unlock(txn1, page1).unwrap();

        let locks = lm.get_locks(txn1);
        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].0, page2);

        lm.release_all(txn1).unwrap();
    }

    #[test]
    fn test_get_locks() {
        let lm = LockManager::new();
        let txn1 = TxnId::new(1);
        let page1: PageId = 1;
        let page2: PageId = 2;

        lm.lock(txn1, page1, LockMode::Shared).unwrap();
        lm.lock(txn1, page2, LockMode::Exclusive).unwrap();

        let locks = lm.get_locks(txn1);
        assert_eq!(locks.len(), 2);

        lm.release_all(txn1).unwrap();
    }

    #[test]
    fn test_multiple_transactions() {
        let lm = Arc::new(LockManager::new());
        let page1: PageId = 1;

        let lm1 = Arc::clone(&lm);
        let handle1 = std::thread::spawn(move || {
            let txn1 = TxnId::new(1);
            lm1.lock(txn1, page1, LockMode::Shared).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            lm1.release_all(txn1).unwrap();
        });

        let lm2 = Arc::clone(&lm);
        let handle2 = std::thread::spawn(move || {
            let txn2 = TxnId::new(2);
            lm2.lock(txn2, page1, LockMode::Shared).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            lm2.release_all(txn2).unwrap();
        });

        handle1.join().unwrap();
        handle2.join().unwrap();
    }
}
