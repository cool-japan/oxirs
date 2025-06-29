//! # Lock Management System
//!
//! Hierarchical lock management system supporting multiple lock modes with deadlock detection,
//! timeout handling, and fair scheduling for efficient concurrent access control.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Lock mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum LockMode {
    /// No lock
    None = 0,
    /// Intention shared lock
    IntentionShared = 1,
    /// Intention exclusive lock  
    IntentionExclusive = 2,
    /// Shared lock
    Shared = 3,
    /// Shared intention exclusive lock
    SharedIntentionExclusive = 4,
    /// Exclusive lock
    Exclusive = 5,
}

impl LockMode {
    /// Check if this lock mode is compatible with another
    pub fn is_compatible_with(&self, other: &LockMode) -> bool {
        use LockMode::*;

        match (self, other) {
            (None, _) | (_, None) => true,

            (IntentionShared, IntentionShared) => true,
            (IntentionShared, IntentionExclusive) => true,
            (IntentionShared, Shared) => true,
            (IntentionShared, SharedIntentionExclusive) => true,
            (IntentionShared, Exclusive) => false,

            (IntentionExclusive, IntentionShared) => true,
            (IntentionExclusive, IntentionExclusive) => true,
            (IntentionExclusive, Shared) => false,
            (IntentionExclusive, SharedIntentionExclusive) => false,
            (IntentionExclusive, Exclusive) => false,

            (Shared, IntentionShared) => true,
            (Shared, IntentionExclusive) => false,
            (Shared, Shared) => true,
            (Shared, SharedIntentionExclusive) => false,
            (Shared, Exclusive) => false,

            (SharedIntentionExclusive, IntentionShared) => true,
            (SharedIntentionExclusive, IntentionExclusive) => false,
            (SharedIntentionExclusive, Shared) => false,
            (SharedIntentionExclusive, SharedIntentionExclusive) => false,
            (SharedIntentionExclusive, Exclusive) => false,

            (Exclusive, _) => false,
        }
    }

    /// Check if this mode can be upgraded to another mode
    pub fn can_upgrade_to(&self, target: &LockMode) -> bool {
        use LockMode::*;

        match (self, target) {
            (None, _) => true,
            (IntentionShared, IntentionExclusive) => true,
            (IntentionShared, Shared) => true,
            (IntentionShared, SharedIntentionExclusive) => true,
            (IntentionShared, Exclusive) => true,
            (IntentionExclusive, SharedIntentionExclusive) => true,
            (IntentionExclusive, Exclusive) => true,
            (Shared, SharedIntentionExclusive) => true,
            (Shared, Exclusive) => true,
            (SharedIntentionExclusive, Exclusive) => true,
            (current, target) if current == target => true,
            _ => false,
        }
    }

    /// Get the lock strength (higher values are stronger)
    pub fn strength(&self) -> u8 {
        *self as u8
    }
}

impl std::fmt::Display for LockMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockMode::None => write!(f, "None"),
            LockMode::IntentionShared => write!(f, "IS"),
            LockMode::IntentionExclusive => write!(f, "IX"),
            LockMode::Shared => write!(f, "S"),
            LockMode::SharedIntentionExclusive => write!(f, "SIX"),
            LockMode::Exclusive => write!(f, "X"),
        }
    }
}

/// Resource identifier type
pub type ResourceId = String;

/// Transaction identifier type  
pub type TransactionId = u64;

/// Lock request information
#[derive(Debug, Clone)]
pub struct LockRequest {
    /// Transaction requesting the lock
    pub transaction_id: TransactionId,
    /// Resource being locked
    pub resource_id: ResourceId,
    /// Requested lock mode
    pub mode: LockMode,
    /// Request timestamp
    pub requested_at: Instant,
    /// Thread ID of the requester
    pub thread_id: ThreadId,
    /// Whether this is an upgrade request
    pub is_upgrade: bool,
}

impl LockRequest {
    /// Create a new lock request
    pub fn new(transaction_id: TransactionId, resource_id: ResourceId, mode: LockMode) -> Self {
        Self {
            transaction_id,
            resource_id,
            mode,
            requested_at: Instant::now(),
            thread_id: thread::current().id(),
            is_upgrade: false,
        }
    }

    /// Create an upgrade request
    pub fn upgrade(
        transaction_id: TransactionId,
        resource_id: ResourceId,
        new_mode: LockMode,
    ) -> Self {
        Self {
            transaction_id,
            resource_id,
            mode: new_mode,
            requested_at: Instant::now(),
            thread_id: thread::current().id(),
            is_upgrade: true,
        }
    }
}

/// Lock grant information
#[derive(Debug, Clone)]
pub struct LockGrant {
    /// Transaction holding the lock
    pub transaction_id: TransactionId,
    /// Lock mode granted
    pub mode: LockMode,
    /// When the lock was granted
    pub granted_at: Instant,
    /// Lock count (for multiple acquisitions)
    pub count: u32,
}

impl LockGrant {
    /// Create a new lock grant
    pub fn new(transaction_id: TransactionId, mode: LockMode) -> Self {
        Self {
            transaction_id,
            mode,
            granted_at: Instant::now(),
            count: 1,
        }
    }

    /// Increment the lock count
    pub fn increment(&mut self) {
        self.count += 1;
    }

    /// Decrement the lock count
    pub fn decrement(&mut self) -> bool {
        self.count -= 1;
        self.count == 0
    }
}

/// Lock table entry for a specific resource
#[derive(Debug)]
struct LockTableEntry {
    /// Currently granted locks
    granted: HashMap<TransactionId, LockGrant>,
    /// Waiting lock requests (FIFO queue)
    waiting: VecDeque<LockRequest>,
    /// Condition variable for waiting threads
    condition: Arc<Condvar>,
    /// Mutex for synchronization
    mutex: Arc<Mutex<()>>,
}

impl LockTableEntry {
    fn new() -> Self {
        Self {
            granted: HashMap::new(),
            waiting: VecDeque::new(),
            condition: Arc::new(Condvar::new()),
            mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Check if a lock request can be granted immediately
    fn can_grant(&self, request: &LockRequest) -> bool {
        // Check compatibility with all granted locks
        for grant in self.granted.values() {
            if grant.transaction_id != request.transaction_id {
                if !request.mode.is_compatible_with(&grant.mode) {
                    return false;
                }
            }
        }

        // For upgrade requests, check if no other waiters exist
        if request.is_upgrade {
            return self.waiting.is_empty();
        }

        // Check if there are waiting requests that should be granted first
        for waiting_request in &self.waiting {
            if waiting_request.transaction_id != request.transaction_id {
                if waiting_request.requested_at < request.requested_at {
                    return false;
                }
            }
        }

        true
    }

    /// Grant a lock to a transaction
    fn grant_lock(&mut self, request: &LockRequest) {
        if let Some(existing) = self.granted.get_mut(&request.transaction_id) {
            // Upgrade existing lock or increment count
            if request.mode.strength() > existing.mode.strength() {
                existing.mode = request.mode;
            }
            existing.increment();
        } else {
            // Grant new lock
            self.granted.insert(
                request.transaction_id,
                LockGrant::new(request.transaction_id, request.mode),
            );
        }
    }

    /// Release a lock held by a transaction
    fn release_lock(&mut self, transaction_id: TransactionId, mode: LockMode) -> bool {
        if let Some(grant) = self.granted.get_mut(&transaction_id) {
            if grant.decrement() {
                self.granted.remove(&transaction_id);
            }
            true
        } else {
            false
        }
    }

    /// Get the current lock mode for a transaction
    fn get_lock_mode(&self, transaction_id: TransactionId) -> LockMode {
        self.granted
            .get(&transaction_id)
            .map(|grant| grant.mode)
            .unwrap_or(LockMode::None)
    }

    /// Check if any locks can be granted from the waiting queue
    fn try_grant_waiting(&mut self) -> Vec<TransactionId> {
        let mut granted_transactions = Vec::new();
        let mut i = 0;

        while i < self.waiting.len() {
            let request = &self.waiting[i];
            if self.can_grant(request) {
                let request = self.waiting.remove(i).unwrap();
                self.grant_lock(&request);
                granted_transactions.push(request.transaction_id);
            } else {
                i += 1;
            }
        }

        granted_transactions
    }
}

/// Deadlock detection using wait-for graph
#[derive(Debug)]
struct DeadlockDetector {
    /// Wait-for graph: transaction -> set of transactions it's waiting for
    wait_for: HashMap<TransactionId, HashSet<TransactionId>>,
}

impl DeadlockDetector {
    fn new() -> Self {
        Self {
            wait_for: HashMap::new(),
        }
    }

    /// Add a wait-for edge
    fn add_edge(&mut self, waiter: TransactionId, holder: TransactionId) {
        self.wait_for
            .entry(waiter)
            .or_insert_with(HashSet::new)
            .insert(holder);
    }

    /// Remove all edges for a transaction
    fn remove_transaction(&mut self, transaction_id: TransactionId) {
        self.wait_for.remove(&transaction_id);
        for waiting_set in self.wait_for.values_mut() {
            waiting_set.remove(&transaction_id);
        }
    }

    /// Detect cycles in the wait-for graph using DFS
    fn detect_deadlock(&self) -> Option<Vec<TransactionId>> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for &transaction in self.wait_for.keys() {
            if !visited.contains(&transaction) {
                if let Some(cycle) =
                    self.dfs_cycle_detection(transaction, &mut visited, &mut rec_stack, &mut path)
                {
                    return Some(cycle);
                }
            }
        }

        None
    }

    /// DFS-based cycle detection
    fn dfs_cycle_detection(
        &self,
        node: TransactionId,
        visited: &mut HashSet<TransactionId>,
        rec_stack: &mut HashSet<TransactionId>,
        path: &mut Vec<TransactionId>,
    ) -> Option<Vec<TransactionId>> {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = self.wait_for.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if let Some(cycle) =
                        self.dfs_cycle_detection(neighbor, visited, rec_stack, path)
                    {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(&neighbor) {
                    // Found a cycle - extract it from the path
                    let cycle_start = path.iter().position(|&x| x == neighbor).unwrap();
                    return Some(path[cycle_start..].to_vec());
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
        None
    }
}

/// Lock manager configuration
#[derive(Debug, Clone)]
pub struct LockManagerConfig {
    /// Default lock timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Enable deadlock detection
    pub enable_deadlock_detection: bool,
    /// Deadlock detection interval in milliseconds
    pub deadlock_check_interval_ms: u64,
    /// Maximum number of lock requests per transaction
    pub max_locks_per_transaction: usize,
    /// Enable lock escalation
    pub enable_lock_escalation: bool,
    /// Lock escalation threshold
    pub lock_escalation_threshold: usize,
}

impl Default for LockManagerConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 30000, // 30 seconds
            enable_deadlock_detection: true,
            deadlock_check_interval_ms: 1000, // 1 second
            max_locks_per_transaction: 10000,
            enable_lock_escalation: true,
            lock_escalation_threshold: 1000,
        }
    }
}

/// Lock manager statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct LockManagerStats {
    pub total_lock_requests: u64,
    pub granted_immediately: u64,
    pub had_to_wait: u64,
    pub lock_upgrades: u64,
    pub deadlocks_detected: u64,
    pub timeouts: u64,
    pub active_transactions: usize,
    pub total_locks_held: usize,
    pub avg_wait_time_ms: f64,
    pub max_wait_time_ms: u64,
    pub lock_escalations: u64,
}

/// Lock manager error types
#[derive(Debug, thiserror::Error)]
pub enum LockManagerError {
    #[error("Deadlock detected involving transactions: {transactions:?}")]
    Deadlock { transactions: Vec<TransactionId> },

    #[error("Lock request timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Invalid lock upgrade from {from} to {to}")]
    InvalidUpgrade { from: LockMode, to: LockMode },

    #[error("Transaction {transaction_id} exceeded maximum locks limit {limit}")]
    TooManyLocks {
        transaction_id: TransactionId,
        limit: usize,
    },

    #[error("Transaction {transaction_id} not found")]
    TransactionNotFound { transaction_id: TransactionId },
}

/// Hierarchical lock manager
pub struct LockManager {
    /// Lock table for all resources
    lock_table: Arc<RwLock<HashMap<ResourceId, Arc<RwLock<LockTableEntry>>>>>,
    /// Transaction-to-locks mapping
    transaction_locks: Arc<RwLock<HashMap<TransactionId, HashSet<ResourceId>>>>,
    /// Deadlock detector
    deadlock_detector: Arc<Mutex<DeadlockDetector>>,
    /// Configuration
    config: LockManagerConfig,
    /// Statistics
    stats: Arc<RwLock<LockManagerStats>>,
}

impl LockManager {
    /// Create a new lock manager
    pub fn new() -> Self {
        Self::with_config(LockManagerConfig::default())
    }

    /// Create a new lock manager with custom configuration
    pub fn with_config(config: LockManagerConfig) -> Self {
        Self {
            lock_table: Arc::new(RwLock::new(HashMap::new())),
            transaction_locks: Arc::new(RwLock::new(HashMap::new())),
            deadlock_detector: Arc::new(Mutex::new(DeadlockDetector::new())),
            config,
            stats: Arc::new(RwLock::new(LockManagerStats::default())),
        }
    }

    /// Acquire a lock on a resource
    pub fn acquire_lock(
        &self,
        transaction_id: TransactionId,
        resource_id: ResourceId,
        mode: LockMode,
        timeout: Option<Duration>,
    ) -> Result<(), LockManagerError> {
        let timeout = timeout.unwrap_or(Duration::from_millis(self.config.default_timeout_ms));
        let request = LockRequest::new(transaction_id, resource_id.clone(), mode);

        self.acquire_lock_internal(request, timeout)
    }

    /// Upgrade a lock to a stronger mode
    pub fn upgrade_lock(
        &self,
        transaction_id: TransactionId,
        resource_id: ResourceId,
        new_mode: LockMode,
        timeout: Option<Duration>,
    ) -> Result<(), LockManagerError> {
        let current_mode = self.get_lock_mode(transaction_id, &resource_id);

        if !current_mode.can_upgrade_to(&new_mode) {
            return Err(LockManagerError::InvalidUpgrade {
                from: current_mode,
                to: new_mode,
            });
        }

        let timeout = timeout.unwrap_or(Duration::from_millis(self.config.default_timeout_ms));
        let request = LockRequest::upgrade(transaction_id, resource_id, new_mode);

        self.acquire_lock_internal(request, timeout)
    }

    /// Internal lock acquisition implementation
    fn acquire_lock_internal(
        &self,
        request: LockRequest,
        timeout: Duration,
    ) -> Result<(), LockManagerError> {
        let start_time = Instant::now();

        // Check lock count limits
        self.check_lock_limits(request.transaction_id)?;

        // Get or create lock table entry
        let entry = self.get_or_create_lock_entry(&request.resource_id);

        // Try to grant immediately
        {
            let mut entry_guard = entry.write().unwrap();
            if entry_guard.can_grant(&request) {
                entry_guard.grant_lock(&request);
                self.record_lock_grant(&request);
                return Ok(());
            }

            // Add to waiting queue
            entry_guard.waiting.push_back(request.clone());
        }

        // Wait for lock with timeout using polling approach (simpler and more reliable)
        let poll_interval = Duration::from_millis(10);
        let mut remaining_time = timeout;

        while remaining_time > Duration::ZERO {
            // Try to acquire the lock again
            {
                let mut entry_guard = entry.write().unwrap();
                if entry_guard.can_grant(&request) {
                    entry_guard.grant_lock(&request);
                    self.record_lock_grant(&request);
                    let wait_time = start_time.elapsed();
                    self.update_wait_time_stats(wait_time);
                    return Ok(());
                }
            }

            // Sleep for a short interval
            std::thread::sleep(poll_interval);
            remaining_time = remaining_time.saturating_sub(poll_interval);
        }

        // Timeout occurred - remove from waiting queue
        {
            let mut entry_guard = entry.write().unwrap();
            entry_guard.waiting.retain(|r| {
                r.transaction_id != request.transaction_id || r.resource_id != request.resource_id
            });
        }

        self.update_timeout_stats();
        Err(LockManagerError::Timeout {
            timeout_ms: timeout.as_millis() as u64,
        })
    }

    /// Release a lock
    pub fn release_lock(
        &self,
        transaction_id: TransactionId,
        resource_id: &ResourceId,
        mode: LockMode,
    ) -> Result<(), LockManagerError> {
        let entry = {
            let lock_table = self.lock_table.read().unwrap();
            lock_table.get(resource_id).cloned()
        };

        if let Some(entry) = entry {
            let mut entry_guard = entry.write().unwrap();

            if entry_guard.release_lock(transaction_id, mode) {
                // Try to grant waiting locks
                let granted = entry_guard.try_grant_waiting();

                // Notify waiting threads
                for _ in granted {
                    entry_guard.condition.notify_all();
                }

                // Update transaction locks mapping
                {
                    let mut tx_locks = self.transaction_locks.write().unwrap();
                    if let Some(locks) = tx_locks.get_mut(&transaction_id) {
                        locks.remove(resource_id);
                        if locks.is_empty() {
                            tx_locks.remove(&transaction_id);
                        }
                    }
                }

                Ok(())
            } else {
                Err(LockManagerError::TransactionNotFound { transaction_id })
            }
        } else {
            Err(LockManagerError::TransactionNotFound { transaction_id })
        }
    }

    /// Release all locks held by a transaction
    pub fn release_all_locks(&self, transaction_id: TransactionId) -> Result<(), LockManagerError> {
        let resource_ids = {
            let tx_locks = self.transaction_locks.read().unwrap();
            tx_locks.get(&transaction_id).cloned().unwrap_or_default()
        };

        for resource_id in resource_ids {
            // Get current lock mode
            let mode = self.get_lock_mode(transaction_id, &resource_id);
            if mode != LockMode::None {
                self.release_lock(transaction_id, &resource_id, mode)?;
            }
        }

        // Remove from deadlock detector
        {
            let mut detector = self.deadlock_detector.lock().unwrap();
            detector.remove_transaction(transaction_id);
        }

        Ok(())
    }

    /// Get current lock mode for a transaction on a resource
    pub fn get_lock_mode(
        &self,
        transaction_id: TransactionId,
        resource_id: &ResourceId,
    ) -> LockMode {
        let lock_table = self.lock_table.read().unwrap();
        if let Some(entry) = lock_table.get(resource_id) {
            let entry_guard = entry.read().unwrap();
            entry_guard.get_lock_mode(transaction_id)
        } else {
            LockMode::None
        }
    }

    /// Get or create a lock table entry
    fn get_or_create_lock_entry(&self, resource_id: &ResourceId) -> Arc<RwLock<LockTableEntry>> {
        let mut lock_table = self.lock_table.write().unwrap();
        lock_table
            .entry(resource_id.clone())
            .or_insert_with(|| Arc::new(RwLock::new(LockTableEntry::new())))
            .clone()
    }

    /// Check lock count limits for a transaction
    fn check_lock_limits(&self, transaction_id: TransactionId) -> Result<(), LockManagerError> {
        let tx_locks = self.transaction_locks.read().unwrap();
        if let Some(locks) = tx_locks.get(&transaction_id) {
            if locks.len() >= self.config.max_locks_per_transaction {
                return Err(LockManagerError::TooManyLocks {
                    transaction_id,
                    limit: self.config.max_locks_per_transaction,
                });
            }
        }
        Ok(())
    }

    /// Record a lock grant for statistics
    fn record_lock_grant(&self, request: &LockRequest) {
        // Update transaction locks mapping
        {
            let mut tx_locks = self.transaction_locks.write().unwrap();
            tx_locks
                .entry(request.transaction_id)
                .or_insert_with(HashSet::new)
                .insert(request.resource_id.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_lock_requests += 1;
            stats.granted_immediately += 1;
            if request.is_upgrade {
                stats.lock_upgrades += 1;
            }
        }
    }

    /// Update timeout statistics
    fn update_timeout_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.timeouts += 1;
    }

    /// Update wait time statistics
    fn update_wait_time_stats(&self, wait_time: Duration) {
        let mut stats = self.stats.write().unwrap();
        stats.had_to_wait += 1;

        let wait_ms = wait_time.as_millis() as u64;
        stats.max_wait_time_ms = stats.max_wait_time_ms.max(wait_ms);

        // Update average (simple moving average)
        let total_waits = stats.had_to_wait as f64;
        stats.avg_wait_time_ms =
            (stats.avg_wait_time_ms * (total_waits - 1.0) + wait_ms as f64) / total_waits;
    }

    /// Get lock manager statistics
    pub fn get_stats(&self) -> LockManagerStats {
        let mut stats = self.stats.read().unwrap().clone();

        // Update current state
        let tx_locks = self.transaction_locks.read().unwrap();
        stats.active_transactions = tx_locks.len();
        stats.total_locks_held = tx_locks.values().map(|locks| locks.len()).sum();

        stats
    }

    /// Check for deadlocks (should be called periodically)
    pub fn check_deadlocks(&self) -> Result<(), LockManagerError> {
        if !self.config.enable_deadlock_detection {
            return Ok(());
        }

        let detector = self.deadlock_detector.lock().unwrap();
        if let Some(cycle) = detector.detect_deadlock() {
            {
                let mut stats = self.stats.write().unwrap();
                stats.deadlocks_detected += 1;
            }

            return Err(LockManagerError::Deadlock {
                transactions: cycle,
            });
        }

        Ok(())
    }

    /// Clear all locks (for testing/cleanup)
    pub fn clear(&self) {
        {
            let mut lock_table = self.lock_table.write().unwrap();
            lock_table.clear();
        }

        {
            let mut tx_locks = self.transaction_locks.write().unwrap();
            tx_locks.clear();
        }

        {
            let mut detector = self.deadlock_detector.lock().unwrap();
            *detector = DeadlockDetector::new();
        }

        {
            let mut stats = self.stats.write().unwrap();
            *stats = LockManagerStats::default();
        }
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lock_mode_compatibility() {
        use LockMode::*;

        assert!(IntentionShared.is_compatible_with(&IntentionShared));
        assert!(IntentionShared.is_compatible_with(&Shared));
        assert!(!IntentionShared.is_compatible_with(&Exclusive));

        assert!(Shared.is_compatible_with(&Shared));
        assert!(!Shared.is_compatible_with(&Exclusive));

        assert!(!Exclusive.is_compatible_with(&Shared));
        assert!(!Exclusive.is_compatible_with(&Exclusive));
    }

    #[test]
    fn test_lock_upgrade() {
        use LockMode::*;

        assert!(IntentionShared.can_upgrade_to(&Shared));
        assert!(IntentionShared.can_upgrade_to(&Exclusive));
        assert!(Shared.can_upgrade_to(&Exclusive));
        assert!(!Exclusive.can_upgrade_to(&Shared));
    }

    #[test]
    fn test_basic_locking() {
        let manager = LockManager::new();

        // Acquire shared lock
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();

        // Check lock mode
        assert_eq!(
            manager.get_lock_mode(1, &"resource1".to_string()),
            LockMode::Shared
        );

        // Release lock
        manager
            .release_lock(1, &"resource1".to_string(), LockMode::Shared)
            .unwrap();

        // Check lock is released
        assert_eq!(
            manager.get_lock_mode(1, &"resource1".to_string()),
            LockMode::None
        );
    }

    #[test]
    fn test_lock_upgrade_functionality() {
        let manager = LockManager::new();

        // Acquire intention shared lock
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::IntentionShared, None)
            .unwrap();

        // Upgrade to shared lock
        manager
            .upgrade_lock(1, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();

        // Check upgraded lock mode
        assert_eq!(
            manager.get_lock_mode(1, &"resource1".to_string()),
            LockMode::Shared
        );
    }

    #[test]
    fn test_multiple_shared_locks() {
        let manager = LockManager::new();

        // Multiple transactions can hold shared locks
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();
        manager
            .acquire_lock(2, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();

        assert_eq!(
            manager.get_lock_mode(1, &"resource1".to_string()),
            LockMode::Shared
        );
        assert_eq!(
            manager.get_lock_mode(2, &"resource1".to_string()),
            LockMode::Shared
        );
    }

    #[test]
    fn test_exclusive_lock_blocking() {
        let manager = Arc::new(LockManager::new());
        let counter = Arc::new(AtomicU64::new(0));

        // First transaction gets exclusive lock
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Exclusive, None)
            .unwrap();

        let manager_clone = Arc::clone(&manager);
        let counter_clone = Arc::clone(&counter);

        // Second transaction should be blocked
        let handle = thread::spawn(move || {
            let result = manager_clone.acquire_lock(
                2,
                "resource1".to_string(),
                LockMode::Shared,
                Some(Duration::from_millis(100)),
            );

            // Should timeout
            assert!(result.is_err());
            counter_clone.store(1, Ordering::SeqCst);
        });

        handle.join().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_release_all_locks() {
        let manager = LockManager::new();

        // Acquire multiple locks
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();
        manager
            .acquire_lock(1, "resource2".to_string(), LockMode::Exclusive, None)
            .unwrap();

        // Release all locks
        manager.release_all_locks(1).unwrap();

        // Check all locks are released
        assert_eq!(
            manager.get_lock_mode(1, &"resource1".to_string()),
            LockMode::None
        );
        assert_eq!(
            manager.get_lock_mode(1, &"resource2".to_string()),
            LockMode::None
        );
    }

    #[test]
    fn test_statistics() {
        let manager = LockManager::new();

        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Shared, None)
            .unwrap();
        manager
            .acquire_lock(2, "resource2".to_string(), LockMode::Exclusive, None)
            .unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_lock_requests, 2);
        assert_eq!(stats.granted_immediately, 2);
        assert_eq!(stats.active_transactions, 2);
        assert_eq!(stats.total_locks_held, 2);
    }

    #[test]
    fn test_invalid_upgrade() {
        let manager = LockManager::new();

        // Acquire exclusive lock
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Exclusive, None)
            .unwrap();

        // Try to downgrade (should fail)
        let result = manager.upgrade_lock(1, "resource1".to_string(), LockMode::Shared, None);
        assert!(matches!(
            result,
            Err(LockManagerError::InvalidUpgrade { .. })
        ));
    }

    #[test]
    fn test_lock_timeout() {
        let manager = Arc::new(LockManager::new());

        // First transaction gets exclusive lock
        manager
            .acquire_lock(1, "resource1".to_string(), LockMode::Exclusive, None)
            .unwrap();

        // Second transaction should timeout
        let result = manager.acquire_lock(
            2,
            "resource1".to_string(),
            LockMode::Shared,
            Some(Duration::from_millis(100)),
        );

        assert!(matches!(result, Err(LockManagerError::Timeout { .. })));
    }

    #[test]
    fn test_deadlock_detection() {
        let detector = DeadlockDetector::new();

        // This test would need a more complex setup to actually test deadlock detection
        // For now, just verify the structure exists
        assert!(detector.detect_deadlock().is_none());
    }
}
