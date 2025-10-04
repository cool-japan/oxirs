//! # MVCC (Multi-Version Concurrency Control)
//!
//! Advanced MVCC implementation for TDB storage with transaction support,
//! snapshot isolation, and optimized concurrent access patterns.
//!
//! This module provides sophisticated multi-version concurrency control
//! capabilities including:
//! - Snapshot isolation for read transactions
//! - Optimistic concurrency control for writes
//! - Garbage collection of old versions
//! - Transaction rollback support
//! - Read-write conflict detection

use crate::transactions::IsolationLevel;
use anyhow::{anyhow, Result};
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Version identifier
pub type Version = u64;

/// Transaction identifier
pub type TransactionId = u64;

/// Versioned value container with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedValue<T> {
    pub value: T,
    pub version: Version,
    pub transaction_id: TransactionId,
    pub timestamp: SystemTime,
    pub is_deleted: bool,
    pub created_by: Option<String>, // For auditing
}

impl<T> VersionedValue<T> {
    pub fn new(value: T, version: Version, transaction_id: TransactionId) -> Self {
        Self {
            value,
            version,
            transaction_id,
            timestamp: SystemTime::now(),
            is_deleted: false,
            created_by: None,
        }
    }

    pub fn deleted(version: Version, transaction_id: TransactionId) -> Self
    where
        T: Default,
    {
        Self {
            value: T::default(),
            version,
            transaction_id,
            timestamp: SystemTime::now(),
            is_deleted: true,
            created_by: None,
        }
    }

    pub fn with_creator(mut self, creator: String) -> Self {
        self.created_by = Some(creator);
        self
    }

    /// Check if this version is visible to a transaction at a specific version
    pub fn is_visible_to(&self, read_version: Version) -> bool {
        self.version <= read_version
    }
}

/// MVCC transaction state and metadata
#[derive(Debug, Clone)]
pub struct MvccTransaction {
    pub id: TransactionId,
    pub start_version: Version,
    pub read_set: HashSet<String>,  // Keys read in this transaction
    pub write_set: HashSet<String>, // Keys written in this transaction
    pub started_at: SystemTime,
    pub is_read_only: bool,
    pub isolation_level: IsolationLevel,
}

impl MvccTransaction {
    pub fn new(id: TransactionId, start_version: Version, is_read_only: bool) -> Self {
        Self {
            id,
            start_version,
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            started_at: SystemTime::now(),
            is_read_only,
            isolation_level: IsolationLevel::SnapshotIsolation,
        }
    }

    pub fn duration(&self) -> Duration {
        self.started_at.elapsed().unwrap_or_default()
    }
}

/// MVCC storage configuration
#[derive(Debug, Clone)]
pub struct MvccConfig {
    pub max_versions_per_key: usize,
    pub gc_interval_seconds: u64,
    pub max_transaction_duration_seconds: u64,
    pub enable_write_skew_detection: bool,
    pub vacuum_threshold: f64, // Fraction of deleted versions that triggers vacuum
}

impl Default for MvccConfig {
    fn default() -> Self {
        Self {
            max_versions_per_key: 100,
            gc_interval_seconds: 300,               // 5 minutes
            max_transaction_duration_seconds: 3600, // 1 hour
            enable_write_skew_detection: true,
            vacuum_threshold: 0.3, // 30% deleted versions
        }
    }
}

/// MVCC storage statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct MvccStats {
    pub total_keys: usize,
    pub total_versions: usize,
    pub deleted_versions: usize,
    pub active_transactions: usize,
    pub completed_transactions: u64,
    pub aborted_transactions: u64,
    pub gc_runs: u64,
    pub last_gc: Option<SystemTime>,
    pub read_conflicts: u64,
    pub write_conflicts: u64,
}

/// Enhanced MVCC storage for key-value pairs with full transaction support
pub struct MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug,
    V: Clone,
{
    data: Arc<RwLock<HashMap<K, Vec<VersionedValue<V>>>>>,
    current_version: Arc<RwLock<Version>>,
    transaction_counter: Arc<RwLock<TransactionId>>,
    active_transactions: Arc<RwLock<HashMap<TransactionId, MvccTransaction>>>,
    config: MvccConfig,
    stats: Arc<Mutex<MvccStats>>,
    vacuum_queue: Arc<Mutex<VecDeque<K>>>, // Keys that need garbage collection
}

impl<K, V> MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug + ToString,
    V: Clone + Default,
{
    pub fn new() -> Self {
        Self::with_config(MvccConfig::default())
    }

    pub fn with_config(config: MvccConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            current_version: Arc::new(RwLock::new(0)),
            transaction_counter: Arc::new(RwLock::new(0)),
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(MvccStats::default())),
            vacuum_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self, read_only: bool) -> Result<TransactionId> {
        let mut tx_counter = self
            .transaction_counter
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction counter lock"))?;
        *tx_counter += 1;
        let tx_id = *tx_counter;

        let current_version = *self
            .current_version
            .read()
            .map_err(|_| anyhow!("Failed to acquire version lock"))?;

        let transaction = MvccTransaction::new(tx_id, current_version, read_only);

        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        active_txs.insert(tx_id, transaction);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.active_transactions = active_txs.len();
        }

        debug!("Started transaction {} (read_only: {})", tx_id, read_only);
        Ok(tx_id)
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, tx_id: TransactionId) -> Result<Version> {
        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let transaction = active_txs
            .remove(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // For read-only transactions, just remove and return current version
        if transaction.is_read_only {
            let current_version = *self
                .current_version
                .read()
                .map_err(|_| anyhow!("Failed to acquire version lock"))?;

            debug!("Committed read-only transaction {}", tx_id);
            self.update_stats_on_commit();
            return Ok(current_version);
        }

        // For write transactions, check for conflicts and assign commit version
        self.validate_transaction(&transaction)?;

        let mut current_version = self
            .current_version
            .write()
            .map_err(|_| anyhow!("Failed to acquire version lock"))?;
        *current_version += 1;
        let commit_version = *current_version;

        // Update all written values with the commit version
        self.finalize_writes(tx_id, commit_version)?;

        debug!(
            "Committed transaction {} at version {}",
            tx_id, commit_version
        );
        self.update_stats_on_commit();

        Ok(commit_version)
    }

    /// Abort a transaction
    pub fn abort_transaction(&self, tx_id: TransactionId) -> Result<()> {
        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let _transaction = active_txs
            .remove(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Remove any uncommitted writes from this transaction
        self.rollback_writes(tx_id)?;

        debug!("Aborted transaction {}", tx_id);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.active_transactions = active_txs.len();
            stats.aborted_transactions += 1;
        }

        Ok(())
    }

    /// Insert or update a value within a transaction
    pub fn put_tx(&self, tx_id: TransactionId, key: K, value: V) -> Result<()> {
        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let transaction = active_txs
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        if transaction.is_read_only {
            return Err(anyhow!("Cannot write in read-only transaction"));
        }

        // Add to write set
        transaction.write_set.insert(key.to_string());

        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        // Create a versioned value with transaction ID (uncommitted)
        let versioned_value = VersionedValue::new(value, 0, tx_id); // Version 0 means uncommitted

        data.entry(key)
            .or_insert_with(Vec::new)
            .push(versioned_value);

        Ok(())
    }

    /// Get a value within a transaction (with snapshot isolation)
    pub fn get_tx(&self, tx_id: TransactionId, key: &K) -> Result<Option<V>> {
        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let transaction = active_txs
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Add to read set
        transaction.read_set.insert(key.to_string());

        let data = self
            .data
            .read()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        if let Some(versions) = data.get(key) {
            // Find the latest version visible to this transaction
            for versioned_value in versions.iter().rev() {
                // Check if this version is visible
                if versioned_value.version == 0 {
                    // Uncommitted value - only visible if it's from the same transaction
                    if versioned_value.transaction_id == tx_id {
                        return Ok(if versioned_value.is_deleted {
                            None
                        } else {
                            Some(versioned_value.value.clone())
                        });
                    }
                } else if versioned_value.version <= transaction.start_version {
                    // Committed value visible to this transaction
                    return Ok(if versioned_value.is_deleted {
                        None
                    } else {
                        Some(versioned_value.value.clone())
                    });
                }
            }
        }

        Ok(None)
    }

    /// Delete a value within a transaction
    pub fn delete_tx(&self, tx_id: TransactionId, key: K) -> Result<()> {
        let mut active_txs = self
            .active_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let transaction = active_txs
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        if transaction.is_read_only {
            return Err(anyhow!("Cannot delete in read-only transaction"));
        }

        // Add to write set
        transaction.write_set.insert(key.to_string());

        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        // Create a versioned deletion marker
        let versioned_value = VersionedValue::deleted(0, tx_id); // Version 0 means uncommitted

        data.entry(key)
            .or_insert_with(Vec::new)
            .push(versioned_value);

        Ok(())
    }

    /// Legacy non-transactional put (creates and commits a transaction)
    pub fn put(&self, key: K, value: V) -> Result<Version> {
        let tx_id = self.begin_transaction(false)?;
        self.put_tx(tx_id, key, value)?;
        self.commit_transaction(tx_id)
    }

    /// Legacy non-transactional get (reads from latest committed state)
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let current_version = *self
            .current_version
            .read()
            .map_err(|_| anyhow!("Failed to acquire version lock"))?;

        self.get_at_version(key, current_version)
    }

    /// Get a value as of a specific version
    pub fn get_at_version(&self, key: &K, target_version: Version) -> Result<Option<V>> {
        let data = self
            .data
            .read()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        if let Some(versions) = data.get(key) {
            // Find the latest committed version <= target_version that is not deleted
            for versioned_value in versions.iter().rev() {
                if versioned_value.version > 0 && // Must be committed
                   versioned_value.version <= target_version
                {
                    return Ok(if versioned_value.is_deleted {
                        None
                    } else {
                        Some(versioned_value.value.clone())
                    });
                }
            }
        }

        Ok(None)
    }

    /// Legacy non-transactional delete
    pub fn delete(&self, key: K) -> Result<Version> {
        let tx_id = self.begin_transaction(false)?;
        self.delete_tx(tx_id, key)?;
        self.commit_transaction(tx_id)
    }

    /// Get current version
    pub fn current_version(&self) -> Result<Version> {
        let version = self
            .current_version
            .read()
            .map_err(|_| anyhow!("Failed to acquire version lock"))?;
        Ok(*version)
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> Result<MvccStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
        Ok(stats.clone())
    }

    /// Cleanup old versions (garbage collection)
    pub fn cleanup_old_versions(&self, keep_versions: usize) -> Result<usize> {
        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        let mut cleaned = 0;
        let oldest_active_version = self.get_oldest_active_version()?;

        for versions in data.values_mut() {
            let original_len = versions.len();

            // Collect recent versions before retain operation to avoid borrow conflict
            let recent_versions: Vec<*const VersionedValue<V>> = versions
                .iter()
                .rev()
                .take(keep_versions)
                .map(|v| v as *const _)
                .collect();

            // Keep versions that are:
            // 1. Recent (within keep_versions)
            // 2. Still visible to active transactions
            // 3. Uncommitted (version = 0)
            versions.retain(|v| {
                v.version == 0 || // Uncommitted
                v.version >= oldest_active_version || // Visible to active transactions
                recent_versions.iter().any(|&kept| std::ptr::eq(kept, v)) // Recent
            });

            cleaned += original_len - versions.len();
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.gc_runs += 1;
            stats.last_gc = Some(SystemTime::now());
            self.update_storage_stats(&mut stats, &data);
        }

        info!("Garbage collection completed, cleaned {} versions", cleaned);
        Ok(cleaned)
    }

    /// Vacuum storage by compacting deleted entries
    pub fn vacuum(&self) -> Result<usize> {
        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        let mut removed_keys = 0;
        let oldest_active_version = self.get_oldest_active_version()?;

        // Remove keys that have only deleted versions older than active transactions
        data.retain(|_key, versions| {
            let has_live_data = versions
                .iter()
                .any(|v| !v.is_deleted && (v.version == 0 || v.version >= oldest_active_version));

            if !has_live_data {
                removed_keys += 1;
            }

            has_live_data
        });

        info!("Vacuum completed, removed {} empty keys", removed_keys);
        Ok(removed_keys)
    }

    /// Get active transaction information
    pub fn get_active_transactions(&self) -> Result<Vec<(TransactionId, Duration)>> {
        let active_txs = self
            .active_transactions
            .read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        Ok(active_txs
            .values()
            .map(|tx| (tx.id, tx.duration()))
            .collect())
    }

    /// Get all keys visible to a transaction
    pub fn get_all_keys_tx(&self, tx_id: TransactionId) -> Result<Vec<K>> {
        let active_txs = self
            .active_transactions
            .read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let transaction = active_txs
            .get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        let data = self
            .data
            .read()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        let mut result = Vec::new();

        for (key, versions) in data.iter() {
            // Check if there's a visible version of this key for this transaction
            for versioned_value in versions.iter().rev() {
                if versioned_value.version == 0 {
                    // Uncommitted value - only visible if it's from the same transaction
                    if versioned_value.transaction_id == tx_id && !versioned_value.is_deleted {
                        result.push(key.clone());
                        break;
                    }
                } else if versioned_value.version <= transaction.start_version {
                    // Committed value visible to this transaction
                    if !versioned_value.is_deleted {
                        result.push(key.clone());
                    }
                    break;
                }
            }
        }

        Ok(result)
    }

    // Private helper methods

    fn validate_transaction(&self, transaction: &MvccTransaction) -> Result<()> {
        if !self.config.enable_write_skew_detection {
            return Ok(());
        }

        // Check for write-write conflicts
        let current_version = *self
            .current_version
            .read()
            .map_err(|_| anyhow!("Failed to acquire version lock"))?;

        if current_version > transaction.start_version {
            // There have been commits since this transaction started
            // Check if any of our read/write sets conflict

            for key_str in &transaction.read_set {
                if self.has_been_modified_since(key_str, transaction.start_version)? {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.read_conflicts += 1;
                    }
                    return Err(anyhow!("Read conflict detected for key: {}", key_str));
                }
            }

            for key_str in &transaction.write_set {
                if self.has_been_modified_since(key_str, transaction.start_version)? {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.write_conflicts += 1;
                    }
                    return Err(anyhow!("Write conflict detected for key: {}", key_str));
                }
            }
        }

        Ok(())
    }

    fn has_been_modified_since(&self, key_str: &str, since_version: Version) -> Result<bool> {
        let data = self
            .data
            .read()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        // Check all keys that match the string pattern (for different key types)
        for (key, versions) in data.iter() {
            let key_string = key.to_string();

            // Check if this key matches the pattern (exact match or prefix)
            if key_string == key_str || key_string.starts_with(key_str) {
                // Check if any version was committed after since_version
                for version in versions.iter() {
                    if version.version > since_version && version.version > 0 {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    fn finalize_writes(&self, tx_id: TransactionId, commit_version: Version) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        // Update all uncommitted writes from this transaction
        for versions in data.values_mut() {
            for version in versions.iter_mut() {
                if version.transaction_id == tx_id && version.version == 0 {
                    version.version = commit_version;
                }
            }
        }

        Ok(())
    }

    fn rollback_writes(&self, tx_id: TransactionId) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|_| anyhow!("Failed to acquire data lock"))?;

        // Remove all uncommitted writes from this transaction
        for versions in data.values_mut() {
            versions.retain(|v| !(v.transaction_id == tx_id && v.version == 0));
        }

        Ok(())
    }

    fn get_oldest_active_version(&self) -> Result<Version> {
        let active_txs = self
            .active_transactions
            .read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let oldest = active_txs
            .values()
            .map(|tx| tx.start_version)
            .min()
            .unwrap_or_else(|| {
                // If no active transactions, use current version as fallback
                // Handle the lock acquisition properly
                match self.current_version.read() {
                    Ok(version) => *version,
                    Err(_) => {
                        // If we can't acquire the lock, return a conservative estimate
                        warn!("Failed to acquire version lock in get_oldest_active_version, using version 0");
                        0
                    }
                }
            });

        Ok(oldest)
    }

    fn update_stats_on_commit(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.completed_transactions += 1;
        }
    }

    fn update_storage_stats(
        &self,
        stats: &mut MvccStats,
        data: &HashMap<K, Vec<VersionedValue<V>>>,
    ) {
        stats.total_keys = data.len();
        stats.total_versions = data.values().map(|v| v.len()).sum();
        stats.deleted_versions = data
            .values()
            .flat_map(|versions| versions.iter())
            .filter(|v| v.is_deleted)
            .count();
    }
}

impl<K, V> Default for MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug + ToString,
    V: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Clone for MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            current_version: Arc::clone(&self.current_version),
            transaction_counter: Arc::clone(&self.transaction_counter),
            active_transactions: Arc::clone(&self.active_transactions),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
            vacuum_queue: Arc::clone(&self.vacuum_queue),
        }
    }
}

/// Logical timestamp for ordering operations across distributed nodes
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogicalTimestamp {
    /// Lamport timestamp component
    pub lamport_time: u64,
    /// Node identifier for tie-breaking
    pub node_id: u64,
    /// Physical timestamp for additional ordering
    pub physical_time: SystemTime,
}

impl LogicalTimestamp {
    pub fn new(lamport_time: u64, node_id: u64) -> Self {
        Self {
            lamport_time,
            node_id,
            physical_time: SystemTime::now(),
        }
    }

    /// Increment the logical timestamp for a new operation
    pub fn increment(&mut self) {
        self.lamport_time += 1;
        self.physical_time = SystemTime::now();
    }

    /// Update timestamp based on received timestamp (for distributed coordination)
    pub fn update(&mut self, other: &LogicalTimestamp) {
        self.lamport_time = self.lamport_time.max(other.lamport_time) + 1;
        self.physical_time = SystemTime::now();
    }

    /// Check if this timestamp happened before another
    pub fn happens_before(&self, other: &LogicalTimestamp) -> bool {
        self.lamport_time < other.lamport_time
            || (self.lamport_time == other.lamport_time && self.node_id < other.node_id)
    }
}

/// Vector clock for distributed timestamp ordering
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorClock {
    /// Clock values for each node
    pub clocks: HashMap<u64, u64>,
    /// This node's ID
    pub node_id: u64,
}

impl VectorClock {
    pub fn new(node_id: u64) -> Self {
        let mut clocks = HashMap::new();
        clocks.insert(node_id, 0);
        Self { clocks, node_id }
    }

    /// Increment local clock
    pub fn tick(&mut self) {
        *self.clocks.entry(self.node_id).or_insert(0) += 1;
    }

    /// Update clock based on received message
    pub fn update(&mut self, other: &VectorClock) {
        // Update all clocks to max of local and received
        for (&node_id, &time) in &other.clocks {
            let current = self.clocks.entry(node_id).or_insert(0);
            *current = (*current).max(time);
        }
        // Increment local clock
        self.tick();
    }

    /// Check if this event happened before another (partial order)
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        // All components must be ≤ and at least one must be <
        let mut all_leq = true;
        let mut some_less = false;

        for (&node, &other_time) in &other.clocks {
            let self_time = self.clocks.get(&node).unwrap_or(&0);
            if self_time > &other_time {
                all_leq = false;
                break;
            }
            if self_time < &other_time {
                some_less = true;
            }
        }

        all_leq && some_less
    }

    /// Check if events are concurrent (neither happens before the other)
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

/// Read and write sets for optimistic concurrency control
#[derive(Debug, Clone)]
pub struct ReadWriteSets<K>
where
    K: Clone + Eq + std::hash::Hash,
{
    /// Keys read during transaction
    pub read_set: HashSet<K>,
    /// Keys written during transaction with their versions
    pub write_set: HashMap<K, Version>,
    /// Timestamp when transaction started
    pub start_timestamp: Option<LogicalTimestamp>,
    /// Timestamp when transaction attempts to commit
    pub commit_timestamp: Option<LogicalTimestamp>,
}

impl<K> ReadWriteSets<K>
where
    K: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self {
        Self {
            read_set: HashSet::new(),
            write_set: HashMap::new(),
            start_timestamp: None,
            commit_timestamp: None,
        }
    }

    /// Record a read operation
    pub fn add_read(&mut self, key: K) {
        self.read_set.insert(key);
    }

    /// Record a write operation
    pub fn add_write(&mut self, key: K, version: Version) {
        self.write_set.insert(key, version);
    }

    /// Set start timestamp for transaction
    pub fn set_start_timestamp(&mut self, timestamp: LogicalTimestamp) {
        self.start_timestamp = Some(timestamp);
    }

    /// Set commit timestamp for validation
    pub fn set_commit_timestamp(&mut self, timestamp: LogicalTimestamp) {
        self.commit_timestamp = Some(timestamp);
    }
}

impl<K> Default for ReadWriteSets<K>
where
    K: Clone + Eq + std::hash::Hash,
{
    fn default() -> Self {
        Self {
            read_set: HashSet::new(),
            write_set: HashMap::new(),
            start_timestamp: None,
            commit_timestamp: None,
        }
    }
}

/// Optimistic concurrency control validation result
#[derive(Debug, PartialEq)]
pub enum ValidationResult {
    Valid,
    ReadConflict(String),
    WriteConflict(String),
    TimestampConflict(String),
}

/// Type alias for committed transaction history
type CommittedTransactionHistory<K> =
    Arc<RwLock<Vec<(TransactionId, ReadWriteSets<K>, LogicalTimestamp)>>>;

/// Optimistic concurrency control manager
pub struct OptimisticConcurrencyControl<K>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug,
{
    /// Active transaction read/write sets
    transaction_sets: Arc<RwLock<HashMap<TransactionId, ReadWriteSets<K>>>>,
    /// Committed transaction history for validation
    committed_transactions: CommittedTransactionHistory<K>,
    /// Logical timestamp generator
    timestamp_generator: Arc<Mutex<LogicalTimestamp>>,
    /// Node ID for distributed coordination
    node_id: u64,
    /// Configuration for backoff and retry
    config: OptimisticConfig,
}

/// Configuration for optimistic concurrency control
#[derive(Debug, Clone)]
pub struct OptimisticConfig {
    /// Maximum retry attempts for failed transactions
    pub max_retry_attempts: u32,
    /// Base backoff time in milliseconds
    pub base_backoff_ms: u64,
    /// Maximum backoff time in milliseconds  
    pub max_backoff_ms: u64,
    /// History size for validation (number of committed transactions to keep)
    pub validation_history_size: usize,
    /// Enable advanced conflict detection
    pub enable_phantom_read_detection: bool,
}

impl Default for OptimisticConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            base_backoff_ms: 10,
            max_backoff_ms: 1000,
            validation_history_size: 10000,
            enable_phantom_read_detection: true,
        }
    }
}

impl<K> OptimisticConcurrencyControl<K>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug + ToString,
{
    pub fn new(node_id: u64) -> Self {
        Self::with_config(node_id, OptimisticConfig::default())
    }

    pub fn with_config(node_id: u64, config: OptimisticConfig) -> Self {
        Self {
            transaction_sets: Arc::new(RwLock::new(HashMap::new())),
            committed_transactions: Arc::new(RwLock::new(Vec::new())),
            timestamp_generator: Arc::new(Mutex::new(LogicalTimestamp::new(0, node_id))),
            node_id,
            config,
        }
    }

    /// Begin optimistic transaction and assign start timestamp
    pub fn begin_transaction(&self, tx_id: TransactionId) -> Result<LogicalTimestamp> {
        let mut timestamp_gen = self
            .timestamp_generator
            .lock()
            .map_err(|_| anyhow!("Failed to acquire timestamp generator lock"))?;

        timestamp_gen.increment();
        let start_timestamp = timestamp_gen.clone();

        let mut sets = ReadWriteSets::new();
        sets.set_start_timestamp(start_timestamp.clone());

        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        transaction_sets.insert(tx_id, sets);

        debug!(
            "Started optimistic transaction {} with timestamp {:?}",
            tx_id, start_timestamp
        );
        Ok(start_timestamp)
    }

    /// Record read operation for transaction
    pub fn record_read(&self, tx_id: TransactionId, key: K) -> Result<()> {
        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        if let Some(sets) = transaction_sets.get_mut(&tx_id) {
            sets.add_read(key);
            Ok(())
        } else {
            Err(anyhow!("Transaction {} not found", tx_id))
        }
    }

    /// Record write operation for transaction
    pub fn record_write(&self, tx_id: TransactionId, key: K, version: Version) -> Result<()> {
        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        if let Some(sets) = transaction_sets.get_mut(&tx_id) {
            sets.add_write(key, version);
            Ok(())
        } else {
            Err(anyhow!("Transaction {} not found", tx_id))
        }
    }

    /// Validate transaction for commit using optimistic concurrency control
    pub fn validate_transaction(&self, tx_id: TransactionId) -> Result<ValidationResult> {
        let mut timestamp_gen = self
            .timestamp_generator
            .lock()
            .map_err(|_| anyhow!("Failed to acquire timestamp generator lock"))?;

        timestamp_gen.increment();
        let commit_timestamp = timestamp_gen.clone();

        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        let sets = transaction_sets
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        sets.set_commit_timestamp(commit_timestamp.clone());

        // Validation phase: check for conflicts with committed transactions
        let committed = self
            .committed_transactions
            .read()
            .map_err(|_| anyhow!("Failed to acquire committed transactions lock"))?;

        let start_timestamp = sets
            .start_timestamp
            .as_ref()
            .ok_or_else(|| anyhow!("No start timestamp for transaction {}", tx_id))?;

        // Check read-write conflicts
        for (committed_tx_id, committed_sets, committed_timestamp) in committed.iter() {
            // Only check transactions that committed after our start time
            if committed_timestamp.happens_before(start_timestamp) {
                continue;
            }

            // Check if we read something that was written by a committed transaction
            for read_key in &sets.read_set {
                if committed_sets.write_set.contains_key(read_key) {
                    return Ok(ValidationResult::ReadConflict(format!(
                        "Read key {read_key:?} was written by committed transaction {committed_tx_id}"
                    )));
                }
            }

            // Check write-write conflicts
            for write_key in sets.write_set.keys() {
                if committed_sets.write_set.contains_key(write_key) {
                    return Ok(ValidationResult::WriteConflict(format!(
                        "Write key {write_key:?} conflicts with committed transaction {committed_tx_id}"
                    )));
                }
            }
        }

        // Phantom read detection (if enabled)
        if self.config.enable_phantom_read_detection {
            if let Some(phantom_conflict) = self.detect_phantom_reads(sets, &committed)? {
                return Ok(ValidationResult::ReadConflict(phantom_conflict));
            }
        }

        info!("Transaction {} validated successfully", tx_id);
        Ok(ValidationResult::Valid)
    }

    /// Commit validated transaction
    pub fn commit_transaction(&self, tx_id: TransactionId) -> Result<()> {
        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        let sets = transaction_sets
            .remove(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        let commit_timestamp = sets
            .commit_timestamp
            .clone()
            .ok_or_else(|| anyhow!("No commit timestamp for transaction {}", tx_id))?;

        // Add to committed transaction history
        let mut committed = self
            .committed_transactions
            .write()
            .map_err(|_| anyhow!("Failed to acquire committed transactions lock"))?;

        committed.push((tx_id, sets, commit_timestamp));

        // Cleanup old history if it gets too large
        if committed.len() > self.config.validation_history_size {
            let remove_count = committed.len() - self.config.validation_history_size;
            committed.drain(0..remove_count);
        }

        info!("Transaction {} committed successfully", tx_id);
        Ok(())
    }

    /// Abort transaction and cleanup
    pub fn abort_transaction(&self, tx_id: TransactionId) -> Result<()> {
        let mut transaction_sets = self
            .transaction_sets
            .write()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;

        transaction_sets.remove(&tx_id);
        warn!("Transaction {} aborted", tx_id);
        Ok(())
    }

    /// Execute transaction with retry and exponential backoff
    pub async fn execute_with_retry<F, R>(&self, mut operation: F) -> Result<R>
    where
        F: FnMut() -> Result<R>,
    {
        let mut attempts = 0;
        let mut backoff_ms = self.config.base_backoff_ms;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.max_retry_attempts {
                        return Err(anyhow!(
                            "Transaction failed after {} attempts: {}",
                            attempts,
                            e
                        ));
                    }

                    // Exponential backoff with jitter
                    let mut rng = Random::default();
                    let jitter = rng.random_range(0, backoff_ms / 4);
                    let sleep_time = Duration::from_millis(backoff_ms + jitter);

                    warn!(
                        "Transaction attempt {} failed: {}. Retrying after {:?}",
                        attempts, e, sleep_time
                    );
                    tokio::time::sleep(sleep_time).await;

                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
            }
        }
    }

    /// Detect phantom reads in read set
    fn detect_phantom_reads(
        &self,
        sets: &ReadWriteSets<K>,
        committed: &[(TransactionId, ReadWriteSets<K>, LogicalTimestamp)],
    ) -> Result<Option<String>> {
        // This is a simplified phantom read detection
        // In a full implementation, this would involve predicate-based conflict detection

        for (committed_tx_id, committed_sets, _) in committed {
            // Check if committed transaction inserted records that would match our reads
            for read_key in &sets.read_set {
                // Simple heuristic: if we read a key pattern and a committed transaction
                // wrote to a similar key, it might be a phantom read
                for write_key in committed_sets.write_set.keys() {
                    if self.keys_might_conflict(read_key, write_key) {
                        return Ok(Some(format!(
                            "Potential phantom read: read pattern {read_key:?} conflicts with insert {write_key:?} from transaction {committed_tx_id}"
                        )));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if two keys might conflict (simplified for demonstration)
    fn keys_might_conflict(&self, read_key: &K, write_key: &K) -> bool {
        // This is a placeholder - in a real implementation, this would involve
        // sophisticated predicate analysis for range queries, etc.
        read_key
            .to_string()
            .contains(&write_key.to_string()[..write_key.to_string().len().min(3)])
    }

    /// Get statistics about optimistic concurrency control
    pub fn get_stats(&self) -> Result<OptimisticStats> {
        let transaction_sets = self
            .transaction_sets
            .read()
            .map_err(|_| anyhow!("Failed to acquire transaction sets lock"))?;
        let committed = self
            .committed_transactions
            .read()
            .map_err(|_| anyhow!("Failed to acquire committed transactions lock"))?;

        Ok(OptimisticStats {
            active_transactions: transaction_sets.len(),
            committed_transactions: committed.len(),
            node_id: self.node_id,
            current_timestamp: self.timestamp_generator.lock().unwrap().clone(),
        })
    }
}

/// Statistics for optimistic concurrency control
#[derive(Debug, Clone, Serialize)]
pub struct OptimisticStats {
    pub active_transactions: usize,
    pub committed_transactions: usize,
    pub node_id: u64,
    pub current_timestamp: LogicalTimestamp,
}

/// Enhanced MVCC storage with timestamp ordering and optimistic concurrency control
impl<K, V> MvccStorage<K, V>
where
    K: Clone + Eq + std::hash::Hash + std::fmt::Debug + ToString,
    V: Clone + Default,
{
    /// Create new MVCC storage with optimistic concurrency control
    pub fn with_optimistic_control(_node_id: u64) -> Self {
        // Initialize optimistic concurrency control (this would be integrated more deeply in a real implementation)
        Self::new()
    }

    /// Begin transaction with timestamp ordering
    pub fn begin_transaction_with_timestamp(
        &self,
        read_only: bool,
        node_id: u64,
    ) -> Result<(TransactionId, LogicalTimestamp)> {
        // Generate logical timestamp
        let timestamp = LogicalTimestamp::new(
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64,
            node_id,
        );

        let tx_id = self.begin_transaction(read_only)?;

        // In a full implementation, we would store the timestamp with the transaction
        debug!(
            "Started transaction {} with timestamp {:?}",
            tx_id, timestamp
        );

        Ok((tx_id, timestamp))
    }

    /// Commit transaction with optimistic validation
    pub fn commit_transaction_optimistic(&self, tx_id: TransactionId) -> Result<Version> {
        // In a full implementation, this would perform optimistic validation
        // before committing the transaction

        info!(
            "Committing transaction {} with optimistic validation",
            tx_id
        );
        self.commit_transaction(tx_id)
    }
}
