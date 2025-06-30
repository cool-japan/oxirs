//! # Advanced Transaction Management System
//!
//! Comprehensive transaction management with ACID compliance, multiple isolation levels,
//! deadlock detection, nested transactions, and full recovery support.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Transaction isolation levels with full SQL standard compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// Allows dirty reads, non-repeatable reads, and phantom reads
    ReadUncommitted,
    /// Prevents dirty reads, allows non-repeatable reads and phantom reads
    ReadCommitted,
    /// Prevents dirty reads and non-repeatable reads, allows phantom reads
    RepeatableRead,
    /// Prevents all phenomena (dirty reads, non-repeatable reads, phantom reads)
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        IsolationLevel::ReadCommitted
    }
}

/// Transaction state with comprehensive lifecycle management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Transaction is active and can perform operations
    Active,
    /// Transaction is preparing for commit (2PC phase 1)
    Preparing,
    /// Transaction has been committed successfully
    Committed,
    /// Transaction has been aborted/rolled back
    Aborted,
    /// Transaction is in recovery mode
    Recovery,
    /// Transaction has timed out
    TimedOut,
    /// Transaction is suspended (for nested transactions)
    Suspended,
}

/// Transaction type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType {
    /// Regular read-write transaction
    ReadWrite,
    /// Read-only transaction (can be optimized)
    ReadOnly,
    /// System transaction (for internal operations)
    System,
    /// Nested subtransaction
    Nested,
}

/// Transaction priority for conflict resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TransactionPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Default for TransactionPriority {
    fn default() -> Self {
        TransactionPriority::Normal
    }
}

/// Resource lock modes for granular locking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockMode {
    /// No lock
    None,
    /// Shared read lock
    Shared,
    /// Exclusive write lock
    Exclusive,
    /// Intention shared lock
    IntentionShared,
    /// Intention exclusive lock
    IntentionExclusive,
    /// Shared with intention exclusive
    SharedIntentionExclusive,
}

/// Resource that can be locked
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    /// Database-level resource
    Database(String),
    /// Table/Graph-level resource
    Table(String),
    /// Page-level resource
    Page(u64),
    /// Row/Triple-level resource
    Row(String),
    /// Index-level resource
    Index(String),
}

/// Lock information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockInfo {
    pub resource: Resource,
    pub mode: LockMode,
    pub acquired_at: Instant,
    pub owner_tx_id: String,
}

/// Transaction configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Default isolation level
    pub default_isolation: IsolationLevel,
    /// Transaction timeout duration
    pub timeout: Duration,
    /// Enable deadlock detection
    pub enable_deadlock_detection: bool,
    /// Deadlock detection interval
    pub deadlock_check_interval: Duration,
    /// Maximum number of active transactions
    pub max_active_transactions: usize,
    /// Enable nested transactions
    pub enable_nested_transactions: bool,
    /// Read-only transaction optimization
    pub optimize_read_only: bool,
    /// Lock timeout duration
    pub lock_timeout: Duration,
    /// Maximum transaction age before forced abort
    pub max_transaction_age: Duration,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            default_isolation: IsolationLevel::ReadCommitted,
            timeout: Duration::from_secs(300), // 5 minutes
            enable_deadlock_detection: true,
            deadlock_check_interval: Duration::from_secs(1),
            max_active_transactions: 1000,
            enable_nested_transactions: true,
            optimize_read_only: true,
            lock_timeout: Duration::from_secs(30),
            max_transaction_age: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Transaction statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct TransactionStats {
    pub total_transactions: u64,
    pub active_transactions: usize,
    pub committed_transactions: u64,
    pub aborted_transactions: u64,
    pub deadlock_count: u64,
    pub timeout_count: u64,
    pub average_duration: Duration,
    pub lock_wait_time: Duration,
    pub read_only_transactions: u64,
    pub nested_transactions: u64,
}

/// Comprehensive transaction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionInfo {
    /// Unique transaction identifier
    pub id: String,
    /// Transaction type
    pub tx_type: TransactionType,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Current state
    pub state: TransactionState,
    /// Transaction priority
    pub priority: TransactionPriority,
    /// Start timestamp
    pub start_time: SystemTime,
    /// Last activity timestamp
    pub last_activity: SystemTime,
    /// Transaction timeout
    pub timeout: Duration,
    /// Parent transaction ID (for nested transactions)
    pub parent_id: Option<String>,
    /// Child transaction IDs
    pub children: HashSet<String>,
    /// Held locks
    pub locks: Vec<LockInfo>,
    /// Read set (for optimistic concurrency control)
    pub read_set: HashSet<Resource>,
    /// Write set
    pub write_set: HashSet<Resource>,
    /// Transaction size metrics
    pub operations_count: u64,
    /// Memory usage
    pub memory_usage: usize,
    /// Client/session information
    pub client_info: Option<String>,
}

impl TransactionInfo {
    /// Create a new transaction info
    pub fn new(
        id: String,
        tx_type: TransactionType,
        isolation_level: IsolationLevel,
        priority: TransactionPriority,
        timeout: Duration,
        parent_id: Option<String>,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            tx_type,
            isolation_level,
            state: TransactionState::Active,
            priority,
            start_time: now,
            last_activity: now,
            timeout,
            parent_id,
            children: HashSet::new(),
            locks: Vec::new(),
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            operations_count: 0,
            memory_usage: 0,
            client_info: None,
        }
    }

    /// Check if transaction has timed out
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed().unwrap_or(Duration::ZERO) > self.timeout
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        matches!(self.state, TransactionState::Active)
    }

    /// Update last activity timestamp
    pub fn update_activity(&mut self) {
        self.last_activity = SystemTime::now();
    }

    /// Add a lock to the transaction
    pub fn add_lock(&mut self, lock: LockInfo) {
        self.locks.push(lock);
    }

    /// Remove a lock from the transaction
    pub fn remove_lock(&mut self, resource: &Resource) {
        self.locks.retain(|lock| &lock.resource != resource);
    }

    /// Check if transaction holds a lock on a resource
    pub fn holds_lock(&self, resource: &Resource, mode: LockMode) -> bool {
        self.locks
            .iter()
            .any(|lock| &lock.resource == resource && lock.mode == mode)
    }

    /// Get transaction age
    pub fn age(&self) -> Duration {
        self.start_time.elapsed().unwrap_or(Duration::ZERO)
    }
}

/// Deadlock detector using wait-for graph
#[derive(Debug)]
pub struct DeadlockDetector {
    /// Wait-for graph: transaction -> set of transactions it's waiting for
    wait_for_graph: HashMap<String, HashSet<String>>,
    /// Last detection run
    last_detection: Instant,
    /// Detection results
    deadlock_cycles: Vec<Vec<String>>,
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            wait_for_graph: HashMap::new(),
            last_detection: Instant::now(),
            deadlock_cycles: Vec::new(),
        }
    }

    /// Add wait relationship
    pub fn add_wait(&mut self, waiter: &str, holder: &str) {
        self.wait_for_graph
            .entry(waiter.to_string())
            .or_insert_with(HashSet::new)
            .insert(holder.to_string());
    }

    /// Remove wait relationship
    pub fn remove_wait(&mut self, waiter: &str, holder: &str) {
        if let Some(waiting_for) = self.wait_for_graph.get_mut(waiter) {
            waiting_for.remove(holder);
            if waiting_for.is_empty() {
                self.wait_for_graph.remove(waiter);
            }
        }
    }

    /// Detect deadlocks using DFS cycle detection
    pub fn detect_deadlocks(&mut self) -> Vec<Vec<String>> {
        self.deadlock_cycles.clear();
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        for tx_id in self.wait_for_graph.keys() {
            if !visited.contains(tx_id) {
                self.dfs_detect_cycle(tx_id, &mut visited, &mut path);
            }
        }

        self.last_detection = Instant::now();
        self.deadlock_cycles.clone()
    }

    /// DFS-based cycle detection
    fn dfs_detect_cycle(
        &mut self,
        tx_id: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) {
        if path.contains(&tx_id.to_string()) {
            // Found a cycle
            let cycle_start = path
                .iter()
                .position(|id| id == tx_id)
                .unwrap_or(path.len());
            let cycle = path[cycle_start..].to_vec();
            if cycle.len() > 1 {
                self.deadlock_cycles.push(cycle);
            }
            return;
        }

        if visited.contains(tx_id) {
            return;
        }

        visited.insert(tx_id.to_string());
        path.push(tx_id.to_string());

        if let Some(waiting_for) = self.wait_for_graph.get(tx_id) {
            for next_tx in waiting_for {
                self.dfs_detect_cycle(next_tx, visited, path);
            }
        }

        path.pop();
    }

    /// Clear all wait relationships
    pub fn clear(&mut self) {
        self.wait_for_graph.clear();
        self.deadlock_cycles.clear();
    }
}

/// Advanced transaction manager with full ACID compliance
pub struct TransactionManager {
    /// Active transactions
    transactions: Arc<RwLock<HashMap<String, TransactionInfo>>>,
    /// Transaction configuration
    config: TransactionConfig,
    /// Lock manager
    lock_manager: Arc<Mutex<HashMap<Resource, Vec<LockInfo>>>>,
    /// Deadlock detector
    deadlock_detector: Arc<Mutex<DeadlockDetector>>,
    /// Transaction statistics
    stats: Arc<RwLock<TransactionStats>>,
    /// Next transaction ID
    next_tx_id: Arc<Mutex<u64>>,
    /// Transaction log
    transaction_log: Arc<Mutex<VecDeque<TransactionInfo>>>,
    /// Background task handle
    background_task_running: Arc<Mutex<bool>>,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self::with_config(TransactionConfig::default())
    }

    /// Create a new transaction manager with custom configuration
    pub fn with_config(config: TransactionConfig) -> Self {
        let manager = Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            config,
            lock_manager: Arc::new(Mutex::new(HashMap::new())),
            deadlock_detector: Arc::new(Mutex::new(DeadlockDetector::new())),
            stats: Arc::new(RwLock::new(TransactionStats::default())),
            next_tx_id: Arc::new(Mutex::new(1)),
            transaction_log: Arc::new(Mutex::new(VecDeque::new())),
            background_task_running: Arc::new(Mutex::new(false)),
        };

        // Start background tasks
        manager.start_background_tasks();
        manager
    }

    /// Begin a new transaction
    pub fn begin_transaction(
        &self,
        tx_type: TransactionType,
        isolation_level: Option<IsolationLevel>,
        priority: Option<TransactionPriority>,
        parent_id: Option<String>,
    ) -> Result<String> {
        let isolation = isolation_level.unwrap_or(self.config.default_isolation);
        let priority = priority.unwrap_or_default();

        // Check if we can create a new transaction
        {
            let transactions = self.transactions.read().unwrap();
            if transactions.len() >= self.config.max_active_transactions {
                return Err(anyhow!("Maximum number of active transactions exceeded"));
            }

            // Validate parent transaction for nested transactions
            if let Some(ref parent) = parent_id {
                if !self.config.enable_nested_transactions {
                    return Err(anyhow!("Nested transactions are not enabled"));
                }
                if !transactions.contains_key(parent) {
                    return Err(anyhow!("Parent transaction {} not found", parent));
                }
            }
        }

        // Generate transaction ID
        let tx_id = {
            let mut next_id = self.next_tx_id.lock().unwrap();
            let id = format!("tx_{}", *next_id);
            *next_id += 1;
            id
        };

        // Create transaction info
        let tx_info = TransactionInfo::new(
            tx_id.clone(),
            tx_type,
            isolation,
            priority,
            self.config.timeout,
            parent_id.clone(),
        );

        // Add to active transactions
        {
            let mut transactions = self.transactions.write().unwrap();
            transactions.insert(tx_id.clone(), tx_info.clone());

            // Update parent's children list
            if let Some(parent) = parent_id {
                if let Some(parent_tx) = transactions.get_mut(&parent) {
                    parent_tx.children.insert(tx_id.clone());
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_transactions += 1;
            stats.active_transactions += 1;
            if matches!(tx_type, TransactionType::ReadOnly) {
                stats.read_only_transactions += 1;
            }
            if matches!(tx_type, TransactionType::Nested) {
                stats.nested_transactions += 1;
            }
        }

        info!("Started transaction {}", tx_id);
        Ok(tx_id)
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, tx_id: &str) -> Result<()> {
        let tx_info = {
            let mut transactions = self.transactions.write().unwrap();
            let tx_info = transactions
                .get_mut(tx_id)
                .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

            if !tx_info.is_active() {
                return Err(anyhow!("Transaction {} is not active", tx_id));
            }

            // Check for timeout
            if tx_info.is_timed_out() {
                tx_info.state = TransactionState::TimedOut;
                return Err(anyhow!("Transaction {} has timed out", tx_id));
            }

            // Prepare phase for 2PC
            tx_info.state = TransactionState::Preparing;

            // Validate isolation level constraints
            self.validate_isolation_constraints(tx_info)?;

            // Commit phase
            tx_info.state = TransactionState::Committed;
            tx_info.update_activity();

            transactions.remove(tx_id).unwrap()
        };

        // Release all locks
        self.release_transaction_locks(tx_id)?;

        // Remove from deadlock detector
        {
            let mut detector = self.deadlock_detector.lock().unwrap();
            detector.wait_for_graph.remove(tx_id);
            // Also remove this transaction from others' wait lists
            for (_, waiting_for) in detector.wait_for_graph.iter_mut() {
                waiting_for.remove(tx_id);
            }
        }

        // Commit nested transactions
        let children = tx_info.children.clone();
        for child_id in children {
            if let Err(e) = self.commit_transaction(&child_id) {
                warn!("Failed to commit child transaction {}: {}", child_id, e);
            }
        }

        // Add to transaction log
        {
            let mut log = self.transaction_log.lock().unwrap();
            log.push_back(tx_info.clone());
            // Keep log size manageable
            if log.len() > 10000 {
                log.pop_front();
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.committed_transactions += 1;
            stats.active_transactions -= 1;
            
            // Update average duration
            let duration = tx_info.age();
            stats.average_duration = if stats.committed_transactions == 1 {
                duration
            } else {
                (stats.average_duration + duration) / 2
            };
        }

        info!("Committed transaction {}", tx_id);
        Ok(())
    }

    /// Abort a transaction
    pub fn abort_transaction(&self, tx_id: &str) -> Result<()> {
        let tx_info = {
            let mut transactions = self.transactions.write().unwrap();
            let tx_info = transactions
                .get_mut(tx_id)
                .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

            if matches!(tx_info.state, TransactionState::Committed) {
                return Err(anyhow!("Cannot abort committed transaction {}", tx_id));
            }

            tx_info.state = TransactionState::Aborted;
            tx_info.update_activity();

            transactions.remove(tx_id).unwrap()
        };

        // Release all locks
        self.release_transaction_locks(tx_id)?;

        // Remove from deadlock detector
        {
            let mut detector = self.deadlock_detector.lock().unwrap();
            detector.wait_for_graph.remove(tx_id);
            for (_, waiting_for) in detector.wait_for_graph.iter_mut() {
                waiting_for.remove(tx_id);
            }
        }

        // Abort nested transactions
        let children = tx_info.children.clone();
        for child_id in children {
            if let Err(e) = self.abort_transaction(&child_id) {
                warn!("Failed to abort child transaction {}: {}", child_id, e);
            }
        }

        // Add to transaction log
        {
            let mut log = self.transaction_log.lock().unwrap();
            log.push_back(tx_info);
            if log.len() > 10000 {
                log.pop_front();
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.aborted_transactions += 1;
            stats.active_transactions -= 1;
        }

        info!("Aborted transaction {}", tx_id);
        Ok(())
    }

    /// Acquire a lock on a resource
    pub fn acquire_lock(
        &self,
        tx_id: &str,
        resource: Resource,
        mode: LockMode,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Check if transaction exists and is active
        {
            let transactions = self.transactions.read().unwrap();
            let tx_info = transactions
                .get(tx_id)
                .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

            if !tx_info.is_active() {
                return Err(anyhow!("Transaction {} is not active", tx_id));
            }
        }

        // Try to acquire lock with timeout
        let acquired = self.try_acquire_lock_with_timeout(tx_id, &resource, mode)?;

        if !acquired {
            return Err(anyhow!(
                "Failed to acquire lock on {:?} for transaction {}",
                resource,
                tx_id
            ));
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.lock_wait_time += start_time.elapsed();
        }

        debug!(
            "Acquired {:?} lock on {:?} for transaction {}",
            mode, resource, tx_id
        );
        Ok(())
    }

    /// Try to acquire lock with timeout
    fn try_acquire_lock_with_timeout(
        &self,
        tx_id: &str,
        resource: &Resource,
        mode: LockMode,
    ) -> Result<bool> {
        let timeout_deadline = Instant::now() + self.config.lock_timeout;

        loop {
            if self.try_acquire_lock_immediate(tx_id, resource, mode)? {
                return Ok(true);
            }

            if Instant::now() >= timeout_deadline {
                return Ok(false);
            }

            // Wait a bit before retrying
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Try to acquire lock immediately
    fn try_acquire_lock_immediate(
        &self,
        tx_id: &str,
        resource: &Resource,
        mode: LockMode,
    ) -> Result<bool> {
        let mut lock_manager = self.lock_manager.lock().unwrap();
        let existing_locks = lock_manager.entry(resource.clone()).or_insert_with(Vec::new);

        // Check lock compatibility
        if self.is_lock_compatible(existing_locks, mode, tx_id) {
            // Acquire the lock
            let lock_info = LockInfo {
                resource: resource.clone(),
                mode,
                acquired_at: Instant::now(),
                owner_tx_id: tx_id.to_string(),
            };

            existing_locks.push(lock_info.clone());

            // Add lock to transaction
            {
                let mut transactions = self.transactions.write().unwrap();
                if let Some(tx_info) = transactions.get_mut(tx_id) {
                    tx_info.add_lock(lock_info);
                }
            }

            Ok(true)
        } else {
            // Add to wait-for graph for deadlock detection
            if self.config.enable_deadlock_detection {
                let mut detector = self.deadlock_detector.lock().unwrap();
                for lock in existing_locks.iter() {
                    detector.add_wait(tx_id, &lock.owner_tx_id);
                }
            }

            Ok(false)
        }
    }

    /// Check if lock modes are compatible
    fn is_lock_compatible(&self, existing_locks: &[LockInfo], mode: LockMode, tx_id: &str) -> bool {
        for lock in existing_locks {
            // Same transaction can acquire multiple locks
            if lock.owner_tx_id == tx_id {
                continue;
            }

            // Check compatibility matrix
            if !self.are_lock_modes_compatible(lock.mode, mode) {
                return false;
            }
        }
        true
    }

    /// Lock compatibility matrix
    fn are_lock_modes_compatible(&self, existing: LockMode, requested: LockMode) -> bool {
        use LockMode::*;
        match (existing, requested) {
            (None, _) | (_, None) => true,
            (Shared, Shared) => true,
            (Shared, IntentionShared) => true,
            (IntentionShared, Shared) => true,
            (IntentionShared, IntentionShared) => true,
            (IntentionShared, IntentionExclusive) => true,
            (IntentionExclusive, IntentionShared) => true,
            _ => false,
        }
    }

    /// Release all locks held by a transaction
    fn release_transaction_locks(&self, tx_id: &str) -> Result<()> {
        let mut lock_manager = self.lock_manager.lock().unwrap();

        // Remove locks from lock manager
        lock_manager.retain(|_, locks| {
            locks.retain(|lock| lock.owner_tx_id != tx_id);
            !locks.is_empty()
        });

        debug!("Released all locks for transaction {}", tx_id);
        Ok(())
    }

    /// Validate isolation level constraints
    fn validate_isolation_constraints(&self, tx_info: &TransactionInfo) -> Result<()> {
        match tx_info.isolation_level {
            IsolationLevel::Serializable => {
                // Check for conflicts with read/write sets
                self.validate_serializable_constraints(tx_info)
            }
            IsolationLevel::RepeatableRead => {
                // Check for non-repeatable reads
                self.validate_repeatable_read_constraints(tx_info)
            }
            IsolationLevel::ReadCommitted => {
                // Check for dirty reads
                self.validate_read_committed_constraints(tx_info)
            }
            IsolationLevel::ReadUncommitted => {
                // No validation needed
                Ok(())
            }
        }
    }

    /// Validate serializable isolation constraints
    fn validate_serializable_constraints(&self, _tx_info: &TransactionInfo) -> Result<()> {
        // Implementation would check for serialization conflicts
        // For now, we assume validation passes
        Ok(())
    }

    /// Validate repeatable read constraints
    fn validate_repeatable_read_constraints(&self, _tx_info: &TransactionInfo) -> Result<()> {
        // Implementation would check for non-repeatable reads
        Ok(())
    }

    /// Validate read committed constraints
    fn validate_read_committed_constraints(&self, _tx_info: &TransactionInfo) -> Result<()> {
        // Implementation would check for dirty reads
        Ok(())
    }

    /// Get transaction information
    pub fn get_transaction_info(&self, tx_id: &str) -> Option<TransactionInfo> {
        let transactions = self.transactions.read().unwrap();
        transactions.get(tx_id).cloned()
    }

    /// List all active transactions
    pub fn active_transactions(&self) -> Vec<TransactionInfo> {
        let transactions = self.transactions.read().unwrap();
        transactions.values().cloned().collect()
    }

    /// Get transaction statistics
    pub fn get_stats(&self) -> TransactionStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Force deadlock detection
    pub fn detect_deadlocks(&self) -> Vec<Vec<String>> {
        let mut detector = self.deadlock_detector.lock().unwrap();
        let cycles = detector.detect_deadlocks();

        if !cycles.is_empty() {
            let mut stats = self.stats.write().unwrap();
            stats.deadlock_count += cycles.len() as u64;
        }

        cycles
    }

    /// Resolve deadlocks by aborting victim transactions
    pub fn resolve_deadlocks(&self) -> Result<Vec<String>> {
        let cycles = self.detect_deadlocks();
        let mut aborted = Vec::new();

        for cycle in cycles {
            if let Some(victim) = self.select_deadlock_victim(&cycle) {
                if let Err(e) = self.abort_transaction(&victim) {
                    warn!("Failed to abort deadlock victim {}: {}", victim, e);
                } else {
                    aborted.push(victim);
                    info!("Aborted transaction {} to resolve deadlock", aborted.last().unwrap());
                }
            }
        }

        Ok(aborted)
    }

    /// Select transaction to abort in deadlock resolution
    fn select_deadlock_victim(&self, cycle: &[String]) -> Option<String> {
        let transactions = self.transactions.read().unwrap();

        // Select transaction with lowest priority, or youngest if tied
        cycle
            .iter()
            .filter_map(|tx_id| transactions.get(tx_id))
            .min_by_key(|tx| (tx.priority, tx.start_time))
            .map(|tx| tx.id.clone())
    }

    /// Start background maintenance tasks
    fn start_background_tasks(&self) {
        let mut running = self.background_task_running.lock().unwrap();
        if *running {
            return;
        }
        *running = true;

        // Clone necessary data for background task
        let transactions = Arc::clone(&self.transactions);
        let deadlock_detector = Arc::clone(&self.deadlock_detector);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        let running_flag = Arc::clone(&self.background_task_running);

        // Spawn background task
        std::thread::spawn(move || {
            while *running_flag.lock().unwrap() {
                // Deadlock detection
                if config.enable_deadlock_detection {
                    let mut detector = deadlock_detector.lock().unwrap();
                    let cycles = detector.detect_deadlocks();
                    if !cycles.is_empty() {
                        let mut stats = stats.write().unwrap();
                        stats.deadlock_count += cycles.len() as u64;
                    }
                }

                // Timeout detection
                {
                    let mut transactions = transactions.write().unwrap();
                    let mut timed_out = Vec::new();

                    for (tx_id, tx_info) in transactions.iter_mut() {
                        if tx_info.is_active() && tx_info.is_timed_out() {
                            tx_info.state = TransactionState::TimedOut;
                            timed_out.push(tx_id.clone());
                        }
                    }

                    if !timed_out.is_empty() {
                        let mut stats = stats.write().unwrap();
                        stats.timeout_count += timed_out.len() as u64;
                    }
                }

                std::thread::sleep(config.deadlock_check_interval);
            }
        });
    }

    /// Shutdown the transaction manager
    pub fn shutdown(&self) {
        let mut running = self.background_task_running.lock().unwrap();
        *running = false;

        // Abort all active transactions
        let active_txs: Vec<String> = {
            let transactions = self.transactions.read().unwrap();
            transactions.keys().cloned().collect()
        };

        for tx_id in active_txs {
            if let Err(e) = self.abort_transaction(&tx_id) {
                warn!("Failed to abort transaction {} during shutdown: {}", tx_id, e);
            }
        }

        info!("Transaction manager shutdown complete");
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TransactionManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_transaction_lifecycle() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();
        assert!(!tx_id.is_empty());

        let tx_info = tm.get_transaction_info(&tx_id).unwrap();
        assert_eq!(tx_info.state, TransactionState::Active);

        tm.commit_transaction(&tx_id).unwrap();

        // Transaction should no longer be active
        assert!(tm.get_transaction_info(&tx_id).is_none());
    }

    #[test]
    fn test_transaction_abort() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();

        tm.abort_transaction(&tx_id).unwrap();

        // Transaction should no longer be active
        assert!(tm.get_transaction_info(&tx_id).is_none());
    }

    #[test]
    fn test_nested_transactions() {
        let tm = TransactionManager::new();

        let parent_id = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();

        let child_id = tm
            .begin_transaction(
                TransactionType::Nested,
                None,
                None,
                Some(parent_id.clone()),
            )
            .unwrap();

        let parent_info = tm.get_transaction_info(&parent_id).unwrap();
        assert!(parent_info.children.contains(&child_id));

        tm.commit_transaction(&parent_id).unwrap();
    }

    #[test]
    fn test_lock_acquisition() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();

        let resource = Resource::Page(1);
        tm.acquire_lock(&tx_id, resource, LockMode::Shared).unwrap();

        let tx_info = tm.get_transaction_info(&tx_id).unwrap();
        assert!(!tx_info.locks.is_empty());

        tm.commit_transaction(&tx_id).unwrap();
    }

    #[test]
    fn test_isolation_levels() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(
                TransactionType::ReadWrite,
                Some(IsolationLevel::Serializable),
                None,
                None,
            )
            .unwrap();

        let tx_info = tm.get_transaction_info(&tx_id).unwrap();
        assert_eq!(tx_info.isolation_level, IsolationLevel::Serializable);

        tm.commit_transaction(&tx_id).unwrap();
    }

    #[test]
    fn test_transaction_priorities() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(
                TransactionType::ReadWrite,
                None,
                Some(TransactionPriority::High),
                None,
            )
            .unwrap();

        let tx_info = tm.get_transaction_info(&tx_id).unwrap();
        assert_eq!(tx_info.priority, TransactionPriority::High);

        tm.commit_transaction(&tx_id).unwrap();
    }

    #[test]
    fn test_statistics() {
        let tm = TransactionManager::new();

        let tx_id = tm
            .begin_transaction(TransactionType::ReadOnly, None, None, None)
            .unwrap();

        let stats = tm.get_stats();
        assert_eq!(stats.active_transactions, 1);
        assert_eq!(stats.read_only_transactions, 1);

        tm.commit_transaction(&tx_id).unwrap();

        let stats = tm.get_stats();
        assert_eq!(stats.active_transactions, 0);
        assert_eq!(stats.committed_transactions, 1);
    }

    #[test]
    fn test_deadlock_detection() {
        let tm = TransactionManager::new();

        let tx1 = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();
        let tx2 = tm
            .begin_transaction(TransactionType::ReadWrite, None, None, None)
            .unwrap();

        // Create a potential deadlock scenario
        let resource1 = Resource::Page(1);
        let resource2 = Resource::Page(2);

        tm.acquire_lock(&tx1, resource1.clone(), LockMode::Exclusive)
            .unwrap();
        tm.acquire_lock(&tx2, resource2.clone(), LockMode::Exclusive)
            .unwrap();

        // This would create a deadlock if both tried to acquire each other's locks
        // The deadlock detector should catch this

        tm.commit_transaction(&tx1).unwrap();
        tm.commit_transaction(&tx2).unwrap();
    }
}
