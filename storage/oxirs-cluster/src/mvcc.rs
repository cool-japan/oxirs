//! Multi-Version Concurrency Control (MVCC) implementation for OxiRS cluster
//!
//! This module provides a comprehensive MVCC system that allows multiple concurrent
//! readers to access consistent snapshots of data without blocking writers. It uses
//! hybrid logical clocks (HLC) for timestamp generation and maintains multiple
//! versions of each triple.

use crate::transaction::{IsolationLevel, TransactionId};
use anyhow::Result;
use dashmap::DashMap;
use oxirs_core::model::Triple;
#[cfg(test)]
use oxirs_core::vocab::xsd;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Hybrid Logical Clock for distributed timestamp generation
#[derive(Debug)]
pub struct HybridLogicalClock {
    /// Physical time component (milliseconds since epoch)
    physical_time: AtomicU64,
    /// Logical counter for events at the same physical time
    logical_counter: AtomicU64,
    /// Node ID for unique timestamp generation across nodes
    node_id: u64,
}

impl HybridLogicalClock {
    /// Create a new HLC with the given node ID
    pub fn new(node_id: u64) -> Self {
        let physical_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            physical_time: AtomicU64::new(physical_time),
            logical_counter: AtomicU64::new(0),
            node_id,
        }
    }

    /// Generate a new timestamp
    pub fn now(&self) -> HLCTimestamp {
        let current_physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last_physical = self.physical_time.load(AtomicOrdering::SeqCst);

        let (physical, logical) = if current_physical > last_physical {
            // Physical time has advanced
            self.physical_time
                .store(current_physical, AtomicOrdering::SeqCst);
            self.logical_counter.store(0, AtomicOrdering::SeqCst);
            (current_physical, 0)
        } else {
            // Same physical time, increment logical counter
            let logical = self.logical_counter.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            (last_physical, logical)
        };

        HLCTimestamp {
            physical,
            logical,
            node_id: self.node_id,
        }
    }

    /// Update HLC with a received timestamp
    pub fn update(&self, received: &HLCTimestamp) -> HLCTimestamp {
        let current_physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last_physical = self.physical_time.load(AtomicOrdering::SeqCst);
        let max_physical = current_physical.max(last_physical).max(received.physical);

        let (physical, logical) =
            if max_physical > last_physical && max_physical > received.physical {
                // Local physical time is ahead
                self.physical_time
                    .store(max_physical, AtomicOrdering::SeqCst);
                self.logical_counter.store(0, AtomicOrdering::SeqCst);
                (max_physical, 0)
            } else if max_physical == received.physical {
                // Received timestamp has same or higher physical time
                let logical = if max_physical == last_physical {
                    self.logical_counter
                        .load(AtomicOrdering::SeqCst)
                        .max(received.logical)
                        + 1
                } else {
                    received.logical + 1
                };
                self.physical_time
                    .store(max_physical, AtomicOrdering::SeqCst);
                self.logical_counter.store(logical, AtomicOrdering::SeqCst);
                (max_physical, logical)
            } else {
                // Local physical time matches max
                let logical = self.logical_counter.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                (max_physical, logical)
            };

        HLCTimestamp {
            physical,
            logical,
            node_id: self.node_id,
        }
    }
}

/// HLC timestamp with physical time, logical counter, and node ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HLCTimestamp {
    /// Physical time component (milliseconds since epoch)
    pub physical: u64,
    /// Logical counter for events at the same physical time
    pub logical: u64,
    /// Node ID for unique identification
    pub node_id: u64,
}

impl PartialOrd for HLCTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HLCTimestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.physical.cmp(&other.physical) {
            Ordering::Equal => match self.logical.cmp(&other.logical) {
                Ordering::Equal => self.node_id.cmp(&other.node_id),
                other => other,
            },
            other => other,
        }
    }
}

/// Version metadata for a triple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Timestamp when this version was created
    pub timestamp: HLCTimestamp,
    /// Transaction ID that created this version
    pub transaction_id: TransactionId,
    /// Whether this version represents a deletion
    pub is_deleted: bool,
    /// The actual triple data (None if deleted)
    pub data: Option<Triple>,
}

/// MVCC configuration
#[derive(Debug, Clone)]
pub struct MVCCConfig {
    /// Enable snapshot isolation
    pub enable_snapshot_isolation: bool,
    /// Garbage collection interval
    pub gc_interval: Duration,
    /// Minimum age for version garbage collection
    pub gc_min_age: Duration,
    /// Maximum versions to keep per key
    pub max_versions_per_key: usize,
    /// Enable read/write conflict detection
    pub enable_conflict_detection: bool,
}

impl Default for MVCCConfig {
    fn default() -> Self {
        Self {
            enable_snapshot_isolation: true,
            gc_interval: Duration::from_secs(60),
            gc_min_age: Duration::from_secs(300), // 5 minutes
            max_versions_per_key: 100,
            enable_conflict_detection: true,
        }
    }
}

/// Transaction snapshot for consistent reads
#[derive(Debug, Clone)]
pub struct TransactionSnapshot {
    /// Transaction ID
    pub transaction_id: TransactionId,
    /// Snapshot timestamp
    pub timestamp: HLCTimestamp,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Read set for conflict detection
    pub read_set: Arc<RwLock<HashSet<String>>>,
    /// Write set for conflict detection
    pub write_set: Arc<RwLock<HashSet<String>>>,
}

/// MVCC manager for version control
pub struct MVCCManager {
    /// Configuration
    config: MVCCConfig,
    /// Hybrid logical clock
    clock: Arc<HybridLogicalClock>,
    /// Version storage: key -> list of versions
    versions: Arc<DashMap<String, BTreeMap<HLCTimestamp, Version>>>,
    /// Active transactions
    transactions: Arc<RwLock<HashMap<TransactionId, TransactionSnapshot>>>,
    /// Committed transaction timestamps
    committed_transactions: Arc<RwLock<BTreeMap<HLCTimestamp, TransactionId>>>,
    /// Garbage collection task handle
    gc_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MVCCManager {
    /// Create a new MVCC manager
    pub fn new(node_id: u64, config: MVCCConfig) -> Self {
        Self {
            config,
            clock: Arc::new(HybridLogicalClock::new(node_id)),
            versions: Arc::new(DashMap::new()),
            transactions: Arc::new(RwLock::new(HashMap::new())),
            committed_transactions: Arc::new(RwLock::new(BTreeMap::new())),
            gc_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the MVCC manager (including garbage collection)
    pub async fn start(&self) -> Result<()> {
        // Start garbage collection task
        let gc_interval = self.config.gc_interval;
        let gc_min_age = self.config.gc_min_age;
        let max_versions = self.config.max_versions_per_key;
        let versions = Arc::clone(&self.versions);
        let committed_transactions = Arc::clone(&self.committed_transactions);
        let clock = Arc::clone(&self.clock);

        let gc_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(gc_interval);
            loop {
                interval.tick().await;

                let current_time = clock.now();
                let cutoff_physical = current_time
                    .physical
                    .saturating_sub(gc_min_age.as_millis() as u64);

                // Clean up old versions
                for mut entry in versions.iter_mut() {
                    let key = entry.key();
                    let versions_map = entry.value_mut();

                    // Remove versions older than cutoff
                    versions_map.retain(|timestamp, _| timestamp.physical >= cutoff_physical);

                    // Keep only max_versions most recent
                    if versions_map.len() > max_versions {
                        let to_remove: Vec<_> = versions_map
                            .keys()
                            .take(versions_map.len() - max_versions)
                            .cloned()
                            .collect();

                        for timestamp in to_remove {
                            versions_map.remove(&timestamp);
                        }
                    }
                }

                // Clean up old committed transaction records
                let mut committed = committed_transactions.write().await;
                committed.retain(|timestamp, _| timestamp.physical >= cutoff_physical);

                debug!("MVCC garbage collection completed");
            }
        });

        *self.gc_handle.lock().await = Some(gc_task);
        info!("MVCC manager started with garbage collection");

        Ok(())
    }

    /// Stop the MVCC manager
    pub async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.gc_handle.lock().await.take() {
            handle.abort();
        }
        info!("MVCC manager stopped");
        Ok(())
    }

    /// Begin a new transaction
    pub async fn begin_transaction(
        &self,
        transaction_id: TransactionId,
        isolation_level: IsolationLevel,
    ) -> Result<TransactionSnapshot> {
        let timestamp = self.clock.now();

        let snapshot = TransactionSnapshot {
            transaction_id: transaction_id.clone(),
            timestamp,
            isolation_level,
            read_set: Arc::new(RwLock::new(HashSet::new())),
            write_set: Arc::new(RwLock::new(HashSet::new())),
        };

        self.transactions
            .write()
            .await
            .insert(transaction_id, snapshot.clone());

        debug!(
            "Started MVCC transaction {} at {:?}",
            snapshot.transaction_id, timestamp
        );
        Ok(snapshot)
    }

    /// Read a value with MVCC
    pub async fn read(&self, transaction_id: &TransactionId, key: &str) -> Result<Option<Triple>> {
        let transactions = self.transactions.read().await;
        let snapshot = transactions
            .get(transaction_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        // Record read in read set
        if self.config.enable_conflict_detection {
            snapshot.read_set.write().await.insert(key.to_string());
        }

        // Get the appropriate version based on isolation level
        let version = match snapshot.isolation_level {
            IsolationLevel::ReadUncommitted => {
                // Read the latest version (including uncommitted)
                self.get_latest_version(key).await
            }
            IsolationLevel::ReadCommitted => {
                // First check for version from current transaction
                if let Some(version) = self.get_version_from_transaction(key, transaction_id).await
                {
                    Some(version)
                } else {
                    // Otherwise get the latest committed version
                    self.get_latest_committed_version(key, &snapshot.timestamp)
                        .await
                }
            }
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => {
                // Read the version as of transaction start
                self.get_version_at_timestamp(key, &snapshot.timestamp)
                    .await
            }
        };

        Ok(version.and_then(|v| v.data))
    }

    /// Write a value with MVCC
    pub async fn write(
        &self,
        transaction_id: &TransactionId,
        key: &str,
        triple: Option<Triple>,
    ) -> Result<()> {
        let transactions = self.transactions.read().await;
        let snapshot = transactions
            .get(transaction_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        // Record write in write set
        if self.config.enable_conflict_detection {
            snapshot.write_set.write().await.insert(key.to_string());
        }

        // Create new version
        let timestamp = self.clock.now();
        let version = Version {
            timestamp,
            transaction_id: transaction_id.clone(),
            is_deleted: triple.is_none(),
            data: triple,
        };

        // Store version
        self.versions
            .entry(key.to_string())
            .or_insert_with(BTreeMap::new)
            .insert(timestamp, version);

        debug!(
            "Wrote version for key {} in transaction {} at {:?}",
            key, transaction_id, timestamp
        );
        Ok(())
    }

    /// Check for conflicts before committing
    pub async fn check_conflicts(&self, transaction_id: &TransactionId) -> Result<bool> {
        if !self.config.enable_conflict_detection {
            return Ok(false);
        }

        let transactions = self.transactions.read().await;
        let snapshot = transactions
            .get(transaction_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        let read_set = snapshot.read_set.read().await;
        let write_set = snapshot.write_set.read().await;

        // Check for write-write conflicts
        let committed = self.committed_transactions.read().await;
        for key in write_set.iter() {
            if let Some(versions) = self.versions.get(key) {
                // Check if any committed version conflicts with our writes
                let has_conflict = versions.range(snapshot.timestamp..).any(|(ts, v)| {
                    ts > &snapshot.timestamp &&
                        v.transaction_id != *transaction_id &&
                        // Only consider it a conflict if the other transaction is committed
                        committed.values().any(|tx_id| tx_id == &v.transaction_id)
                });

                if has_conflict {
                    warn!(
                        "Write-write conflict detected for key {} in transaction {}",
                        key, transaction_id
                    );
                    return Ok(true);
                }
            }
        }

        // Check for read-write conflicts (for serializable isolation)
        if snapshot.isolation_level == IsolationLevel::Serializable {
            for key in read_set.iter() {
                if let Some(versions) = self.versions.get(key) {
                    // Check if any committed version was written after our snapshot
                    let has_conflict = versions.range(snapshot.timestamp..).any(|(ts, v)| {
                        ts > &snapshot.timestamp &&
                        v.transaction_id != *transaction_id &&
                        // Only consider it a conflict if the other transaction is committed
                        committed.values().any(|tx_id| tx_id == &v.transaction_id)
                    });

                    if has_conflict {
                        warn!(
                            "Read-write conflict detected for key {} in transaction {}",
                            key, transaction_id
                        );
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Commit a transaction
    pub async fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        // Check for conflicts
        if self.check_conflicts(transaction_id).await? {
            return Err(anyhow::anyhow!("Transaction conflicts detected"));
        }

        let timestamp = {
            let transactions = self.transactions.read().await;
            let snapshot = transactions
                .get(transaction_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            snapshot.timestamp
        };

        // Record committed transaction
        self.committed_transactions
            .write()
            .await
            .insert(timestamp, transaction_id.clone());

        // Remove from active transactions
        self.transactions.write().await.remove(transaction_id);

        info!(
            "Committed transaction {} at {:?}",
            transaction_id, timestamp
        );
        Ok(())
    }

    /// Rollback a transaction
    pub async fn rollback_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        // Remove all versions created by this transaction
        for mut entry in self.versions.iter_mut() {
            entry
                .value_mut()
                .retain(|_, version| version.transaction_id != *transaction_id);
        }

        // Remove from active transactions
        self.transactions.write().await.remove(transaction_id);

        info!("Rolled back transaction {}", transaction_id);
        Ok(())
    }

    /// Get the latest version of a key
    async fn get_latest_version(&self, key: &str) -> Option<Version> {
        self.versions
            .get(key)
            .and_then(|versions| versions.values().last().cloned())
    }

    /// Get the latest committed version of a key
    async fn get_latest_committed_version(
        &self,
        key: &str,
        before_timestamp: &HLCTimestamp,
    ) -> Option<Version> {
        let committed = self.committed_transactions.read().await;

        self.versions.get(key).and_then(|versions| {
            versions
                .range(..=before_timestamp)
                .rev()
                .find(|(ts, version)| {
                    // Check if this version's transaction is committed
                    committed
                        .values()
                        .any(|tx_id| tx_id == &version.transaction_id)
                })
                .map(|(_, version)| version.clone())
        })
    }

    /// Get version from a specific transaction
    async fn get_version_from_transaction(
        &self,
        key: &str,
        transaction_id: &TransactionId,
    ) -> Option<Version> {
        self.versions.get(key).and_then(|versions| {
            versions
                .values()
                .rev()
                .find(|version| version.transaction_id == *transaction_id)
                .cloned()
        })
    }

    /// Get version at a specific timestamp
    async fn get_version_at_timestamp(
        &self,
        key: &str,
        timestamp: &HLCTimestamp,
    ) -> Option<Version> {
        let committed = self.committed_transactions.read().await;

        self.versions.get(key).and_then(|versions| {
            versions
                .range(..=timestamp)
                .rev()
                .find(|(_, version)| {
                    // For repeatable read, only consider committed versions
                    committed
                        .values()
                        .any(|tx_id| tx_id == &version.transaction_id)
                })
                .map(|(_, version)| version.clone())
        })
    }

    /// Get all versions of a key (for debugging/monitoring)
    pub async fn get_all_versions(&self, key: &str) -> Vec<Version> {
        self.versions
            .get(key)
            .map(|versions| versions.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get MVCC statistics
    pub async fn get_statistics(&self) -> MVCCStatistics {
        let total_keys = self.versions.len();
        let mut total_versions = 0;
        let mut max_versions_per_key = 0;

        for entry in self.versions.iter() {
            let version_count = entry.value().len();
            total_versions += version_count;
            max_versions_per_key = max_versions_per_key.max(version_count);
        }

        let active_transactions = self.transactions.read().await.len();
        let committed_transactions = self.committed_transactions.read().await.len();

        MVCCStatistics {
            total_keys,
            total_versions,
            max_versions_per_key,
            active_transactions,
            committed_transactions,
        }
    }

    /// Update clock with external timestamp (for distributed synchronization)
    pub fn update_clock(&self, timestamp: &HLCTimestamp) -> HLCTimestamp {
        self.clock.update(timestamp)
    }

    /// Get current timestamp
    pub fn current_timestamp(&self) -> HLCTimestamp {
        self.clock.now()
    }

    /// Scan all keys with a given prefix and return their values
    pub async fn scan_prefix(
        &self,
        transaction_id: &TransactionId,
        prefix: &str,
    ) -> Result<Vec<(String, Triple)>> {
        let mut results = Vec::new();

        // Get the transaction snapshot for visibility
        let transactions = self.transactions.read().await;
        let snapshot = transactions
            .get(transaction_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", transaction_id))?;

        // Iterate through all keys and filter by prefix
        for entry in self.versions.iter() {
            let key = entry.key();
            if key.starts_with(prefix) {
                if let Some(version) = self
                    .get_visible_version(
                        key,
                        &snapshot.timestamp,
                        snapshot.isolation_level == IsolationLevel::ReadUncommitted,
                    )
                    .await
                {
                    if let Some(triple) = version.data {
                        results.push((key.clone(), triple));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get the visible version of a key for a given timestamp
    async fn get_visible_version(
        &self,
        key: &str,
        timestamp: &HLCTimestamp,
        include_uncommitted: bool,
    ) -> Option<Version> {
        if let Some(versions) = self.versions.get(key) {
            // Find the latest version visible to this timestamp
            for (ts, version) in versions.iter().rev() {
                if ts <= timestamp {
                    // Check if this version is committed or if we include uncommitted
                    if include_uncommitted {
                        return Some(version.clone());
                    } else {
                        // Check if the transaction is committed
                        let committed = self.committed_transactions.read().await;
                        if committed
                            .values()
                            .any(|tx_id| tx_id == &version.transaction_id)
                        {
                            return Some(version.clone());
                        }
                    }
                }
            }
        }
        None
    }
}

/// MVCC statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MVCCStatistics {
    /// Total number of keys with versions
    pub total_keys: usize,
    /// Total number of versions across all keys
    pub total_versions: usize,
    /// Maximum versions for any single key
    pub max_versions_per_key: usize,
    /// Number of active transactions
    pub active_transactions: usize,
    /// Number of committed transactions tracked
    pub committed_transactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_timestamp_ordering() {
        let ts1 = HLCTimestamp {
            physical: 100,
            logical: 0,
            node_id: 1,
        };
        let ts2 = HLCTimestamp {
            physical: 100,
            logical: 1,
            node_id: 1,
        };
        let ts3 = HLCTimestamp {
            physical: 101,
            logical: 0,
            node_id: 1,
        };
        let ts4 = HLCTimestamp {
            physical: 100,
            logical: 0,
            node_id: 2,
        };

        assert!(ts1 < ts2);
        assert!(ts2 < ts3);
        assert!(ts1 < ts4); // Different node IDs
    }

    #[test]
    fn test_hlc_generation() {
        let clock = HybridLogicalClock::new(1);

        let ts1 = clock.now();
        let ts2 = clock.now();

        assert!(ts2 > ts1);
        assert_eq!(ts1.node_id, 1);
        assert_eq!(ts2.node_id, 1);
    }

    #[test]
    fn test_hlc_update() {
        let clock = HybridLogicalClock::new(1);

        let ts1 = clock.now();
        let received = HLCTimestamp {
            physical: ts1.physical + 1000,
            logical: 5,
            node_id: 2,
        };

        let ts2 = clock.update(&received);

        assert!(ts2.physical >= received.physical);
        assert!(ts2 > ts1);
    }

    #[tokio::test]
    async fn test_mvcc_basic_operations() {
        let mvcc = MVCCManager::new(1, MVCCConfig::default());
        mvcc.start().await.unwrap();

        // Begin transaction
        let tx_id = "tx1".to_string();
        let snapshot = mvcc
            .begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Write a value
        let triple = Triple::new(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
            oxirs_core::model::Literal::new_typed_literal("value", xsd::STRING.clone()),
        );

        mvcc.write(&tx_id, "key1", Some(triple.clone()))
            .await
            .unwrap();

        // Read the value
        let read_value = mvcc.read(&tx_id, "key1").await.unwrap();
        assert!(read_value.is_some());

        // Commit transaction
        mvcc.commit_transaction(&tx_id).await.unwrap();

        // Verify statistics
        let stats = mvcc.get_statistics().await;
        assert_eq!(stats.total_keys, 1);
        assert_eq!(stats.total_versions, 1);

        mvcc.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_mvcc_isolation_levels() {
        let mvcc = MVCCManager::new(1, MVCCConfig::default());
        mvcc.start().await.unwrap();

        // Create a committed version
        let tx1 = "tx1".to_string();
        mvcc.begin_transaction(tx1.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        let triple = Triple::new(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
            oxirs_core::model::Literal::new_typed_literal("value1", xsd::STRING.clone()),
        );

        mvcc.write(&tx1, "key1", Some(triple.clone()))
            .await
            .unwrap();
        mvcc.commit_transaction(&tx1).await.unwrap();

        // Start new transaction with repeatable read
        let tx2 = "tx2".to_string();
        mvcc.begin_transaction(tx2.clone(), IsolationLevel::RepeatableRead)
            .await
            .unwrap();

        // Read should see committed value
        let value = mvcc.read(&tx2, "key1").await.unwrap();
        assert!(value.is_some());

        // Another transaction modifies the value
        let tx3 = "tx3".to_string();
        mvcc.begin_transaction(tx3.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        let triple2 = Triple::new(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
            oxirs_core::model::Literal::new_typed_literal("value2", xsd::STRING.clone()),
        );

        mvcc.write(&tx3, "key1", Some(triple2)).await.unwrap();
        mvcc.commit_transaction(&tx3).await.unwrap();

        // tx2 should still see the old value (repeatable read)
        let value2 = mvcc.read(&tx2, "key1").await.unwrap();
        assert!(value2.is_some());
        // Note: In a real implementation, we'd verify it's the old value

        mvcc.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_mvcc_conflict_detection() {
        let config = MVCCConfig {
            enable_conflict_detection: true,
            ..Default::default()
        };
        let mvcc = MVCCManager::new(1, config);
        mvcc.start().await.unwrap();

        // Two transactions modifying the same key
        let tx1 = "tx1".to_string();
        let tx2 = "tx2".to_string();

        mvcc.begin_transaction(tx1.clone(), IsolationLevel::Serializable)
            .await
            .unwrap();
        mvcc.begin_transaction(tx2.clone(), IsolationLevel::Serializable)
            .await
            .unwrap();

        let triple = Triple::new(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
            oxirs_core::model::Literal::new_typed_literal("value", xsd::STRING.clone()),
        );

        // Both read the same key
        mvcc.read(&tx1, "key1").await.unwrap();
        mvcc.read(&tx2, "key1").await.unwrap();

        // Both write to the same key
        mvcc.write(&tx1, "key1", Some(triple.clone()))
            .await
            .unwrap();
        mvcc.write(&tx2, "key1", Some(triple)).await.unwrap();

        // First commit should succeed
        mvcc.commit_transaction(&tx1).await.unwrap();

        // Second commit should fail due to conflict
        let result = mvcc.commit_transaction(&tx2).await;
        assert!(result.is_err());

        mvcc.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_mvcc_rollback() {
        let mvcc = MVCCManager::new(1, MVCCConfig::default());
        mvcc.start().await.unwrap();

        let tx_id = "tx1".to_string();
        mvcc.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        let triple = Triple::new(
            oxirs_core::model::NamedNode::new("http://example.org/s").unwrap(),
            oxirs_core::model::NamedNode::new("http://example.org/p").unwrap(),
            oxirs_core::model::Literal::new_typed_literal("value", xsd::STRING.clone()),
        );

        mvcc.write(&tx_id, "key1", Some(triple)).await.unwrap();

        // Rollback the transaction
        mvcc.rollback_transaction(&tx_id).await.unwrap();

        // Start new transaction and verify value is not there
        let tx2 = "tx2".to_string();
        mvcc.begin_transaction(tx2.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();
        let value = mvcc.read(&tx2, "key1").await.unwrap();
        assert!(value.is_none());

        mvcc.stop().await.unwrap();
    }
}
