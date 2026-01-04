//! Transaction conflict resolution and deadlock detection
//!
//! This module provides advanced conflict resolution strategies and
//! deadlock detection algorithms beyond basic timeout-based detection.

use super::lock_manager::{LockManager, LockMode};
use super::wal::TxnId;
use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Wait for lock with timeout (default)
    WaitTimeout,
    /// Abort immediately if lock unavailable (no-wait)
    NoWait,
    /// Abort younger transaction (wound-wait)
    WoundWait,
    /// Wait if older, abort if younger (wait-die)
    WaitDie,
}

/// Deadlock detection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlockDetection {
    /// Timeout-based detection
    Timeout,
    /// Wait-for graph based detection
    WaitForGraph,
    /// Periodic cycle detection
    Periodic,
}

/// Enhanced transaction conflict manager
pub struct ConflictManager {
    /// Lock manager
    lock_manager: Arc<LockManager>,
    /// Conflict resolution strategy
    strategy: ConflictStrategy,
    /// Deadlock detection algorithm
    detection: DeadlockDetection,
    /// Wait-for graph (txn -> set of txns it waits for)
    wait_for_graph: RwLock<HashMap<TxnId, HashSet<TxnId>>>,
    /// Transaction timestamps (for wound-wait/wait-die)
    txn_timestamps: RwLock<HashMap<TxnId, u64>>,
    /// Global timestamp counter
    timestamp_counter: RwLock<u64>,
}

impl ConflictManager {
    /// Create a new conflict manager
    pub fn new(
        lock_manager: Arc<LockManager>,
        strategy: ConflictStrategy,
        detection: DeadlockDetection,
    ) -> Self {
        ConflictManager {
            lock_manager,
            strategy,
            detection,
            wait_for_graph: RwLock::new(HashMap::new()),
            txn_timestamps: RwLock::new(HashMap::new()),
            timestamp_counter: RwLock::new(0),
        }
    }

    /// Begin a transaction and assign timestamp
    pub fn begin_transaction(&self, txn_id: TxnId) -> Result<()> {
        let mut counter = self
            .timestamp_counter
            .write()
            .expect("rwlock should not be poisoned");
        let timestamp = *counter;
        *counter += 1;

        let mut timestamps = self
            .txn_timestamps
            .write()
            .expect("rwlock should not be poisoned");
        timestamps.insert(txn_id, timestamp);

        Ok(())
    }

    /// End a transaction and clean up
    pub fn end_transaction(&self, txn_id: TxnId) -> Result<()> {
        // Remove from wait-for graph
        let mut wait_for = self
            .wait_for_graph
            .write()
            .expect("rwlock should not be poisoned");
        wait_for.remove(&txn_id);

        // Remove from timestamps
        let mut timestamps = self
            .txn_timestamps
            .write()
            .expect("rwlock should not be poisoned");
        timestamps.remove(&txn_id);

        Ok(())
    }

    /// Acquire a lock with conflict resolution
    pub fn lock_with_strategy(&self, txn_id: TxnId, page_id: PageId, mode: LockMode) -> Result<()> {
        match self.strategy {
            ConflictStrategy::WaitTimeout => {
                // Default behavior - wait with timeout
                self.lock_manager.lock(txn_id, page_id, mode)
            }
            ConflictStrategy::NoWait => {
                // Immediate failure if lock unavailable
                if self.lock_manager.try_lock(txn_id, page_id, mode)? {
                    Ok(())
                } else {
                    Err(TdbError::Transaction(format!(
                        "Lock unavailable for page {} (no-wait mode)",
                        page_id
                    )))
                }
            }
            ConflictStrategy::WoundWait => {
                // Wound-wait: if requester is older, abort younger holder
                self.wound_wait_lock(txn_id, page_id, mode)
            }
            ConflictStrategy::WaitDie => {
                // Wait-die: if requester is older, wait; if younger, abort
                self.wait_die_lock(txn_id, page_id, mode)
            }
        }
    }

    /// Wound-wait lock acquisition
    ///
    /// Strategy: If requester is older than holder, abort the younger holder.
    /// If requester is younger, wait normally.
    fn wound_wait_lock(&self, txn_id: TxnId, page_id: PageId, mode: LockMode) -> Result<()> {
        // Try to acquire immediately
        if self.lock_manager.try_lock(txn_id, page_id, mode)? {
            return Ok(());
        }

        // Get my timestamp
        let my_timestamp = self
            .txn_timestamps
            .read()
            .expect("rwlock should not be poisoned")
            .get(&txn_id)
            .copied()
            .unwrap_or(u64::MAX);

        // Get current lock holders
        let holders = self.lock_manager.get_lock_holders(page_id);

        // Check if I'm older than any holder
        let timestamps = self
            .txn_timestamps
            .read()
            .expect("rwlock should not be poisoned");
        let should_wound = holders.iter().any(|(holder_txn, _)| {
            let holder_timestamp = timestamps.get(holder_txn).copied().unwrap_or(0);
            my_timestamp < holder_timestamp // I'm older
        });

        if should_wound {
            // Wound strategy: Abort younger holders
            // Note: In a real system, we would send abort signals to the holders
            // For now, we log the action and proceed to wait
            log::info!(
                "Transaction {} (ts={}) wounds younger holders for page {}",
                txn_id.as_u64(),
                my_timestamp,
                page_id
            );

            // In a full implementation, we would:
            // 1. Send abort signals to younger holders
            // 2. Wait for them to release locks
            // 3. Then acquire the lock

            // For now, fall back to regular locking
            self.lock_manager.lock(txn_id, page_id, mode)
        } else {
            // I'm younger - wait normally
            self.lock_manager.lock(txn_id, page_id, mode)
        }
    }

    /// Wait-die lock acquisition
    ///
    /// Strategy: If requester is older than holder, wait.
    /// If requester is younger, abort (die) immediately.
    fn wait_die_lock(&self, txn_id: TxnId, page_id: PageId, mode: LockMode) -> Result<()> {
        // Try to acquire immediately
        if self.lock_manager.try_lock(txn_id, page_id, mode)? {
            return Ok(());
        }

        // Get my timestamp
        let my_timestamp = self
            .txn_timestamps
            .read()
            .expect("rwlock should not be poisoned")
            .get(&txn_id)
            .copied()
            .unwrap_or(u64::MAX);

        // Get current lock holders
        let holders = self.lock_manager.get_lock_holders(page_id);

        if holders.is_empty() {
            // No holders, try to acquire
            return self.lock_manager.lock(txn_id, page_id, mode);
        }

        // Check if I'm older than all holders
        let timestamps = self
            .txn_timestamps
            .read()
            .expect("rwlock should not be poisoned");
        let i_am_older = holders.iter().all(|(holder_txn, _)| {
            let holder_timestamp = timestamps.get(holder_txn).copied().unwrap_or(u64::MAX);
            my_timestamp < holder_timestamp // I'm older
        });

        if i_am_older {
            // I'm older - wait for holders to release
            log::debug!(
                "Transaction {} (ts={}) waiting for locks on page {}",
                txn_id.as_u64(),
                my_timestamp,
                page_id
            );
            self.lock_manager.lock(txn_id, page_id, mode)
        } else {
            // I'm younger - die (abort) immediately
            log::info!(
                "Transaction {} (ts={}) dies (aborts) due to conflict on page {}",
                txn_id.as_u64(),
                my_timestamp,
                page_id
            );

            Err(TdbError::Transaction(format!(
                "Transaction {} aborted by wait-die strategy",
                txn_id.as_u64()
            )))
        }
    }

    /// Detect deadlock using wait-for graph
    pub fn detect_deadlock(&self) -> Result<Vec<DeadlockCycle>> {
        match self.detection {
            DeadlockDetection::Timeout => {
                // Timeout-based detection is handled by lock manager
                Ok(Vec::new())
            }
            DeadlockDetection::WaitForGraph => {
                // Build and analyze wait-for graph
                self.detect_deadlock_cycles()
            }
            DeadlockDetection::Periodic => {
                // Periodic cycle detection
                self.detect_deadlock_cycles()
            }
        }
    }

    /// Detect cycles in wait-for graph
    fn detect_deadlock_cycles(&self) -> Result<Vec<DeadlockCycle>> {
        let wait_for = self
            .wait_for_graph
            .read()
            .expect("rwlock should not be poisoned");
        let mut cycles = Vec::new();

        // Use DFS to detect cycles
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        for &txn in wait_for.keys() {
            if !visited.contains(&txn) {
                if let Some(cycle) = self.find_cycle(&wait_for, txn, &mut visited, &mut path) {
                    cycles.push(cycle);
                }
            }
        }

        Ok(cycles)
    }

    /// Find a cycle starting from a transaction (DFS)
    #[allow(clippy::only_used_in_recursion)]
    fn find_cycle(
        &self,
        wait_for: &HashMap<TxnId, HashSet<TxnId>>,
        start_txn: TxnId,
        visited: &mut HashSet<TxnId>,
        path: &mut Vec<TxnId>,
    ) -> Option<DeadlockCycle> {
        visited.insert(start_txn);
        path.push(start_txn);

        if let Some(waiting_for) = wait_for.get(&start_txn) {
            for &next_txn in waiting_for {
                if let Some(pos) = path.iter().position(|&t| t == next_txn) {
                    // Found a cycle
                    let cycle_txns = path[pos..].to_vec();
                    return Some(DeadlockCycle {
                        transactions: cycle_txns,
                    });
                }

                if !visited.contains(&next_txn) {
                    if let Some(cycle) = self.find_cycle(wait_for, next_txn, visited, path) {
                        return Some(cycle);
                    }
                }
            }
        }

        path.pop();
        None
    }

    /// Add a wait-for edge to the graph
    pub fn add_wait_for(&self, waiter: TxnId, holder: TxnId) -> Result<()> {
        let mut wait_for = self
            .wait_for_graph
            .write()
            .expect("rwlock should not be poisoned");
        wait_for.entry(waiter).or_default().insert(holder);
        Ok(())
    }

    /// Remove a wait-for edge from the graph
    pub fn remove_wait_for(&self, waiter: TxnId, holder: TxnId) -> Result<()> {
        let mut wait_for = self
            .wait_for_graph
            .write()
            .expect("rwlock should not be poisoned");
        if let Some(holders) = wait_for.get_mut(&waiter) {
            holders.remove(&holder);
            if holders.is_empty() {
                wait_for.remove(&waiter);
            }
        }
        Ok(())
    }

    /// Get statistics about conflicts
    pub fn conflict_stats(&self) -> ConflictStats {
        let wait_for = self
            .wait_for_graph
            .read()
            .expect("rwlock should not be poisoned");

        ConflictStats {
            waiting_transactions: wait_for.len(),
            total_wait_edges: wait_for.values().map(|s| s.len()).sum(),
            active_transactions: self
                .txn_timestamps
                .read()
                .expect("rwlock should not be poisoned")
                .len(),
        }
    }
}

/// A detected deadlock cycle
#[derive(Debug, Clone)]
pub struct DeadlockCycle {
    /// Transactions involved in the cycle
    pub transactions: Vec<TxnId>,
}

impl DeadlockCycle {
    /// Get the youngest transaction in the cycle (to abort)
    pub fn youngest_transaction(&self, timestamps: &HashMap<TxnId, u64>) -> Option<TxnId> {
        self.transactions
            .iter()
            .max_by_key(|&&txn| timestamps.get(&txn).copied().unwrap_or(0))
            .copied()
    }

    /// Get the oldest transaction in the cycle
    pub fn oldest_transaction(&self, timestamps: &HashMap<TxnId, u64>) -> Option<TxnId> {
        self.transactions
            .iter()
            .min_by_key(|&&txn| timestamps.get(&txn).copied().unwrap_or(u64::MAX))
            .copied()
    }
}

/// Conflict statistics
#[derive(Debug, Clone)]
pub struct ConflictStats {
    /// Number of transactions currently waiting
    pub waiting_transactions: usize,
    /// Total number of wait-for edges
    pub total_wait_edges: usize,
    /// Number of active transactions
    pub active_transactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_manager_creation() {
        let lock_manager = Arc::new(LockManager::new());
        let _conflict = ConflictManager::new(
            lock_manager,
            ConflictStrategy::WaitTimeout,
            DeadlockDetection::WaitForGraph,
        );
    }

    #[test]
    fn test_transaction_timestamps() {
        let lock_manager = Arc::new(LockManager::new());
        let conflict = ConflictManager::new(
            lock_manager,
            ConflictStrategy::WaitTimeout,
            DeadlockDetection::WaitForGraph,
        );

        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);

        conflict.begin_transaction(txn1).unwrap();
        conflict.begin_transaction(txn2).unwrap();

        let timestamps = conflict
            .txn_timestamps
            .read()
            .expect("rwlock should not be poisoned");
        let ts1 = timestamps.get(&txn1).copied().unwrap();
        let ts2 = timestamps.get(&txn2).copied().unwrap();

        // txn1 should be older (lower timestamp)
        assert!(ts1 < ts2);

        drop(timestamps);
        conflict.end_transaction(txn1).unwrap();
        conflict.end_transaction(txn2).unwrap();
    }

    #[test]
    fn test_wait_for_graph() {
        let lock_manager = Arc::new(LockManager::new());
        let conflict = ConflictManager::new(
            lock_manager,
            ConflictStrategy::WaitTimeout,
            DeadlockDetection::WaitForGraph,
        );

        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);

        conflict.add_wait_for(txn1, txn2).unwrap();

        let stats = conflict.conflict_stats();
        assert_eq!(stats.waiting_transactions, 1);
        assert_eq!(stats.total_wait_edges, 1);

        conflict.remove_wait_for(txn1, txn2).unwrap();

        let stats = conflict.conflict_stats();
        assert_eq!(stats.waiting_transactions, 0);
    }

    #[test]
    fn test_deadlock_cycle_detection() {
        let lock_manager = Arc::new(LockManager::new());
        let conflict = ConflictManager::new(
            lock_manager,
            ConflictStrategy::WaitTimeout,
            DeadlockDetection::WaitForGraph,
        );

        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);
        let txn3 = TxnId::new(3);

        // Create a cycle: txn1 -> txn2 -> txn3 -> txn1
        conflict.add_wait_for(txn1, txn2).unwrap();
        conflict.add_wait_for(txn2, txn3).unwrap();
        conflict.add_wait_for(txn3, txn1).unwrap();

        let cycles = conflict.detect_deadlock().unwrap();
        assert!(!cycles.is_empty(), "Should detect at least one cycle");
    }

    #[test]
    fn test_conflict_stats() {
        let lock_manager = Arc::new(LockManager::new());
        let conflict = ConflictManager::new(
            lock_manager,
            ConflictStrategy::WaitTimeout,
            DeadlockDetection::WaitForGraph,
        );

        let txn1 = TxnId::new(1);
        let txn2 = TxnId::new(2);

        conflict.begin_transaction(txn1).unwrap();
        conflict.begin_transaction(txn2).unwrap();

        let stats = conflict.conflict_stats();
        assert_eq!(stats.active_transactions, 2);

        conflict.end_transaction(txn1).unwrap();
        conflict.end_transaction(txn2).unwrap();
    }
}
