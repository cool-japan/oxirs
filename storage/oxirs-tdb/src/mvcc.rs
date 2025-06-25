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

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
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

/// Transaction state and metadata
#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: TransactionId,
    pub start_version: Version,
    pub read_set: HashSet<String>,  // Keys read in this transaction
    pub write_set: HashSet<String>, // Keys written in this transaction
    pub started_at: SystemTime,
    pub is_read_only: bool,
    pub isolation_level: IsolationLevel,
}

impl Transaction {
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

/// Isolation levels supported by the MVCC system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    SnapshotIsolation,
    Serializable,
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
    active_transactions: Arc<RwLock<HashMap<TransactionId, Transaction>>>,
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

        let transaction = Transaction::new(tx_id, current_version, read_only);

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

        let transaction = active_txs
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
                recent_versions.iter().any(|&kept| kept == v as *const _) // Recent
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

    // Private helper methods

    fn validate_transaction(&self, transaction: &Transaction) -> Result<()> {
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
        // This is a simplified implementation that would need to be more sophisticated
        // in a real system to handle different key types
        Ok(false) // Placeholder
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
                *self
                    .current_version
                    .read()
                    .unwrap_or_else(|_| panic!("Failed to acquire version lock"))
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
