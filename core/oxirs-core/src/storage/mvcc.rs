//! Multi-Version Concurrency Control (MVCC) for RDF storage
//!
//! This module implements MVCC to enable high-concurrency read/write operations
//! by maintaining multiple timestamped versions of RDF triples.

use crate::model::{Object, Predicate, Subject, Triple};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Transaction timestamp (logical timestamp)
pub type Timestamp = u64;

/// Transaction ID
pub type TransactionId = u64;

/// Version identifier
pub type VersionId = u64;

/// MVCC configuration
#[derive(Debug, Clone)]
pub struct MvccConfig {
    /// Maximum number of versions to keep per triple
    pub max_versions_per_triple: usize,

    /// Garbage collection interval
    pub gc_interval: Duration,

    /// Minimum age for version cleanup (prevents cleaning active versions)
    pub min_version_age: Duration,

    /// Enable snapshot isolation
    pub enable_snapshot_isolation: bool,

    /// Enable read-your-writes consistency
    pub enable_read_your_writes: bool,

    /// Conflict detection strategy
    pub conflict_detection: ConflictDetection,
}

impl Default for MvccConfig {
    fn default() -> Self {
        Self {
            max_versions_per_triple: 100,
            gc_interval: Duration::from_secs(60),
            min_version_age: Duration::from_secs(300), // 5 minutes
            enable_snapshot_isolation: true,
            enable_read_your_writes: true,
            conflict_detection: ConflictDetection::Optimistic,
        }
    }
}

/// Conflict detection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictDetection {
    /// Optimistic concurrency control
    Optimistic,
    /// Optimistic two-phase locking
    OptimisticTwoPhase,
    /// Pessimistic locking
    Pessimistic,
    /// Multi-version timestamp ordering
    TimestampOrdering,
}

/// MVCC store for RDF triples
pub struct MvccStore {
    /// Configuration
    config: MvccConfig,

    /// Version storage (triple hash -> versions)
    versions: Arc<DashMap<TripleKey, VersionChain>>,

    /// Active transactions
    transactions: Arc<DashMap<TransactionId, TransactionState>>,

    /// Global timestamp counter
    timestamp_counter: Arc<AtomicU64>,

    /// Transaction ID counter
    transaction_counter: Arc<AtomicU64>,

    /// Snapshot registry
    snapshots: Arc<RwLock<BTreeMap<Timestamp, SnapshotInfo>>>,

    /// Garbage collection state
    gc_state: Arc<Mutex<GarbageCollectionState>>,

    /// Index for efficient queries
    indexes: Arc<MvccIndexes>,
}

/// Key for identifying a triple
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TripleKey {
    subject: String,
    predicate: String,
    object: String,
}

impl TripleKey {
    fn from_triple(triple: &Triple) -> Self {
        Self {
            subject: triple.subject().to_string(),
            predicate: triple.predicate().to_string(),
            object: triple.object().to_string(),
        }
    }
}

/// Version chain for a triple
#[derive(Debug, Clone)]
pub struct VersionChain {
    /// Versions sorted by timestamp (newest first)
    versions: Vec<Version>,
}

impl VersionChain {
    fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Add a new version
    fn add_version(&mut self, version: Version) {
        // Insert in sorted order (newest first)
        let pos = self
            .versions
            .binary_search_by_key(&std::cmp::Reverse(version.timestamp), |v| {
                std::cmp::Reverse(v.timestamp)
            })
            .unwrap_or_else(|pos| pos);
        self.versions.insert(pos, version);
    }

    /// Get visible version for a timestamp
    fn get_visible_version(&self, timestamp: Timestamp) -> Option<&Version> {
        self.versions
            .iter()
            .find(|v| v.timestamp <= timestamp && v.is_visible_at(timestamp))
    }

    /// Garbage collect old versions
    fn gc_versions(&mut self, min_timestamp: Timestamp, max_versions: usize) {
        // Keep at least one version
        if self.versions.len() <= 1 {
            return;
        }

        // Remove old deleted versions
        self.versions
            .retain(|v| !(v.deleted && v.timestamp < min_timestamp));

        // Limit number of versions
        if self.versions.len() > max_versions {
            self.versions.truncate(max_versions);
        }
    }
}

/// A version of a triple
#[derive(Debug, Clone)]
pub struct Version {
    /// Version ID
    pub id: VersionId,

    /// Creation timestamp
    pub timestamp: Timestamp,

    /// Transaction that created this version
    pub transaction_id: TransactionId,

    /// Whether this version represents a deletion
    pub deleted: bool,

    /// The triple data (None if deleted)
    pub triple: Option<Triple>,

    /// Commit timestamp (None if not yet committed)
    pub commit_timestamp: Option<Timestamp>,
}

impl Version {
    /// Check if this version is visible at a given timestamp
    fn is_visible_at(&self, timestamp: Timestamp) -> bool {
        if let Some(commit_ts) = self.commit_timestamp {
            commit_ts <= timestamp
        } else {
            false // Uncommitted versions are not visible
        }
    }
}

/// Transaction state
#[derive(Debug, Clone)]
pub struct TransactionState {
    /// Transaction ID
    pub id: TransactionId,

    /// Start timestamp
    pub start_timestamp: Timestamp,

    /// Commit timestamp (if committed)
    pub commit_timestamp: Option<Timestamp>,

    /// Transaction status
    pub status: TransactionStatus,

    /// Read set (for conflict detection)
    pub read_set: HashSet<TripleKey>,

    /// Write set
    pub write_set: HashMap<TripleKey, WriteOperation>,

    /// Isolation level
    pub isolation_level: IsolationLevel,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionStatus {
    Active,
    Preparing,
    Committed,
    Aborted,
}

/// Write operation type
#[derive(Debug, Clone)]
pub enum WriteOperation {
    Insert(Triple),
    Delete,
}

/// Isolation level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted (dirty reads allowed)
    ReadUncommitted,
    /// Read committed (no dirty reads)
    ReadCommitted,
    /// Repeatable read (no dirty or non-repeatable reads)
    RepeatableRead,
    /// Serializable (full isolation)
    Serializable,
    /// Snapshot isolation
    Snapshot,
    /// Snapshot isolation (alias for Snapshot)
    SnapshotIsolation,
}

/// Snapshot information
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    /// Snapshot timestamp
    pub timestamp: Timestamp,

    /// Active transactions at snapshot time
    pub active_transactions: HashSet<TransactionId>,

    /// Reference count
    pub ref_count: usize,
}

/// Garbage collection state
#[derive(Debug)]
pub struct GarbageCollectionState {
    /// Last GC run time
    last_gc: Instant,

    /// Number of versions collected
    versions_collected: u64,

    /// Number of GC runs
    gc_runs: u64,
}

/// MVCC indexes for efficient queries
pub struct MvccIndexes {
    /// Subject index
    subject_index: DashMap<String, HashSet<TripleKey>>,

    /// Predicate index
    predicate_index: DashMap<String, HashSet<TripleKey>>,

    /// Object index
    object_index: DashMap<String, HashSet<TripleKey>>,
}

impl MvccStore {
    /// Create a new MVCC store
    pub fn new(config: MvccConfig) -> Self {
        Self {
            config,
            versions: Arc::new(DashMap::new()),
            transactions: Arc::new(DashMap::new()),
            timestamp_counter: Arc::new(AtomicU64::new(1)),
            transaction_counter: Arc::new(AtomicU64::new(1)),
            snapshots: Arc::new(RwLock::new(BTreeMap::new())),
            gc_state: Arc::new(Mutex::new(GarbageCollectionState {
                last_gc: Instant::now(),
                versions_collected: 0,
                gc_runs: 0,
            })),
            indexes: Arc::new(MvccIndexes {
                subject_index: DashMap::new(),
                predicate_index: DashMap::new(),
                object_index: DashMap::new(),
            }),
        }
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<TransactionId> {
        let tx_id = self.transaction_counter.fetch_add(1, Ordering::SeqCst);
        let start_timestamp = self.get_next_timestamp();

        let tx_state = TransactionState {
            id: tx_id,
            start_timestamp,
            commit_timestamp: None,
            status: TransactionStatus::Active,
            read_set: HashSet::new(),
            write_set: HashMap::new(),
            isolation_level,
        };

        self.transactions.insert(tx_id, tx_state);

        // Create snapshot if needed
        if isolation_level == IsolationLevel::Snapshot {
            self.create_snapshot(start_timestamp)?;
        }

        Ok(tx_id)
    }

    /// Insert a triple in a transaction
    pub fn insert(&self, tx_id: TransactionId, triple: Triple) -> Result<()> {
        let mut tx = self.get_active_transaction(tx_id)?;

        let key = TripleKey::from_triple(&triple);

        // Check for write-write conflicts
        if self.config.conflict_detection == ConflictDetection::Pessimistic {
            self.check_write_conflict(&key, tx_id)?;
        }

        // Add to write set
        tx.write_set
            .insert(key.clone(), WriteOperation::Insert(triple.clone()));

        // Update indexes
        self.update_indexes_for_insert(&key);

        Ok(())
    }

    /// Delete a triple in a transaction
    pub fn delete(&self, tx_id: TransactionId, triple: &Triple) -> Result<()> {
        let mut tx = self.get_active_transaction(tx_id)?;

        let key = TripleKey::from_triple(triple);

        // Check if triple exists
        if !self.exists_at_timestamp(&key, tx.start_timestamp)? {
            return Err(anyhow!("Triple does not exist"));
        }

        // Add to write set
        tx.write_set.insert(key.clone(), WriteOperation::Delete);

        // Update indexes
        self.update_indexes_for_delete(&key);

        Ok(())
    }

    /// Query triples at transaction's snapshot
    pub fn query(
        &self,
        tx_id: TransactionId,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        let mut tx = self.get_active_transaction(tx_id)?;
        let timestamp = tx.start_timestamp;

        // Get candidate keys from indexes
        let candidates = self.get_candidate_keys(subject, predicate, object);

        let mut results = Vec::new();
        let mut processed_keys = HashSet::new();

        for key in candidates {
            processed_keys.insert(key.clone());

            // Add to read set for conflict detection
            tx.read_set.insert(key.clone());

            // Check if visible at timestamp
            if let Some(version_chain) = self.versions.get(&key) {
                if let Some(version) = version_chain.get_visible_version(timestamp) {
                    if !version.deleted {
                        if let Some(triple) = &version.triple {
                            // Apply predicate filters
                            if self.matches_pattern(triple, subject, predicate, object) {
                                results.push(triple.clone());
                            }
                        }
                    }
                }
            }

            // Check uncommitted writes in same transaction
            if self.config.enable_read_your_writes {
                if let Some(write_op) = tx.write_set.get(&key) {
                    match write_op {
                        WriteOperation::Insert(triple) => {
                            if self.matches_pattern(triple, subject, predicate, object) {
                                results.push(triple.clone());
                            }
                        }
                        WriteOperation::Delete => {
                            // Remove from results if it was added
                            results.retain(|t| TripleKey::from_triple(t) != key);
                        }
                    }
                }
            }
        }

        // Also check write set for new keys not in main indexes (read-your-writes)
        if self.config.enable_read_your_writes {
            for (key, write_op) in &tx.write_set {
                if !processed_keys.contains(key) {
                    match write_op {
                        WriteOperation::Insert(triple) => {
                            if self.matches_pattern(triple, subject, predicate, object) {
                                results.push(triple.clone());
                            }
                        }
                        WriteOperation::Delete => {
                            // Already handled above
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Commit a transaction
    pub fn commit_transaction(&self, tx_id: TransactionId) -> Result<()> {
        let mut tx = self.get_active_transaction(tx_id)?;

        // Change status to preparing
        tx.status = TransactionStatus::Preparing;

        // Validate transaction
        self.validate_transaction(&tx)?;

        // Get commit timestamp
        let commit_timestamp = self.get_next_timestamp();

        // Apply writes
        for (key, operation) in &tx.write_set {
            let version = match operation {
                WriteOperation::Insert(triple) => Version {
                    id: self.get_next_timestamp(), // Use timestamp as version ID
                    timestamp: commit_timestamp,
                    transaction_id: tx_id,
                    deleted: false,
                    triple: Some(triple.clone()),
                    commit_timestamp: Some(commit_timestamp),
                },
                WriteOperation::Delete => Version {
                    id: self.get_next_timestamp(),
                    timestamp: commit_timestamp,
                    transaction_id: tx_id,
                    deleted: true,
                    triple: None,
                    commit_timestamp: Some(commit_timestamp),
                },
            };

            self.versions
                .entry(key.clone())
                .or_insert_with(VersionChain::new)
                .add_version(version);
        }

        // Update transaction state
        tx.commit_timestamp = Some(commit_timestamp);
        tx.status = TransactionStatus::Committed;

        // Trigger GC if needed
        self.maybe_run_gc();

        Ok(())
    }

    /// Abort a transaction
    pub fn abort_transaction(&self, tx_id: TransactionId) -> Result<()> {
        if let Some(mut tx) = self.transactions.get_mut(&tx_id) {
            tx.status = TransactionStatus::Aborted;

            // Clean up any locks or resources
            // In optimistic mode, no cleanup needed
        }

        Ok(())
    }

    /// Validate transaction for conflicts
    fn validate_transaction(&self, tx: &TransactionState) -> Result<()> {
        match self.config.conflict_detection {
            ConflictDetection::Optimistic => {
                // Check read set for modifications
                for key in &tx.read_set {
                    if let Some(version_chain) = self.versions.get(key) {
                        if let Some(latest) = version_chain.versions.first() {
                            if latest.timestamp > tx.start_timestamp {
                                return Err(anyhow!("Read conflict detected"));
                            }
                        }
                    }
                }

                // Check write-write conflicts
                for key in tx.write_set.keys() {
                    if let Some(version_chain) = self.versions.get(key) {
                        if let Some(latest) = version_chain.versions.first() {
                            if latest.timestamp > tx.start_timestamp
                                && latest.transaction_id != tx.id
                            {
                                return Err(anyhow!("Write conflict detected"));
                            }
                        }
                    }
                }
            }

            ConflictDetection::Pessimistic => {
                // Conflicts already prevented during operations
            }

            ConflictDetection::TimestampOrdering => {
                // Ensure timestamp ordering is maintained
                for key in tx.write_set.keys() {
                    if let Some(version_chain) = self.versions.get(key) {
                        for version in &version_chain.versions {
                            if version.transaction_id != tx.id
                                && version.timestamp > tx.start_timestamp
                                && version.commit_timestamp.is_some()
                            {
                                return Err(anyhow!("Timestamp ordering violation"));
                            }
                        }
                    }
                }
            }
            ConflictDetection::OptimisticTwoPhase => {
                // Two-phase optimistic validation
                // Phase 1: Read validation (similar to optimistic)
                for key in &tx.read_set {
                    if let Some(version_chain) = self.versions.get(key) {
                        if let Some(latest) = version_chain.versions.first() {
                            if latest.timestamp > tx.start_timestamp {
                                return Err(anyhow!("Read conflict detected in phase 1"));
                            }
                        }
                    }
                }
                
                // Phase 2: Write validation 
                for key in tx.write_set.keys() {
                    if let Some(version_chain) = self.versions.get(key) {
                        for version in &version_chain.versions {
                            if version.transaction_id != tx.id
                                && version.timestamp > tx.start_timestamp
                                && version.commit_timestamp.is_some()
                            {
                                return Err(anyhow!("Write conflict detected in phase 2"));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get active transaction
    fn get_active_transaction(
        &self,
        tx_id: TransactionId,
    ) -> Result<dashmap::mapref::one::RefMut<TransactionId, TransactionState>> {
        let tx = self
            .transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        if tx.status != TransactionStatus::Active {
            return Err(anyhow!("Transaction is not active"));
        }

        Ok(tx)
    }

    /// Check if triple exists at timestamp
    fn exists_at_timestamp(&self, key: &TripleKey, timestamp: Timestamp) -> Result<bool> {
        if let Some(version_chain) = self.versions.get(key) {
            if let Some(version) = version_chain.get_visible_version(timestamp) {
                return Ok(!version.deleted);
            }
        }
        Ok(false)
    }

    /// Get candidate keys from indexes
    fn get_candidate_keys(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> HashSet<TripleKey> {
        let mut candidates = HashSet::new();

        // Use most selective index
        if let Some(subj) = subject {
            if let Some(keys) = self.indexes.subject_index.get(&subj.to_string()) {
                candidates.extend(keys.iter().cloned());
            }
        } else if let Some(pred) = predicate {
            if let Some(keys) = self.indexes.predicate_index.get(&pred.to_string()) {
                candidates.extend(keys.iter().cloned());
            }
        } else if let Some(obj) = object {
            if let Some(keys) = self.indexes.object_index.get(&obj.to_string()) {
                candidates.extend(keys.iter().cloned());
            }
        } else {
            // Full scan
            for entry in self.versions.iter() {
                candidates.insert(entry.key().clone());
            }
        }

        candidates
    }

    /// Check if triple matches pattern
    fn matches_pattern(
        &self,
        triple: &Triple,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> bool {
        if let Some(s) = subject {
            if triple.subject() != s {
                return false;
            }
        }
        if let Some(p) = predicate {
            if triple.predicate() != p {
                return false;
            }
        }
        if let Some(o) = object {
            if triple.object() != o {
                return false;
            }
        }
        true
    }

    /// Update indexes for insert
    fn update_indexes_for_insert(&self, key: &TripleKey) {
        self.indexes
            .subject_index
            .entry(key.subject.clone())
            .or_default()
            .insert(key.clone());

        self.indexes
            .predicate_index
            .entry(key.predicate.clone())
            .or_default()
            .insert(key.clone());

        self.indexes
            .object_index
            .entry(key.object.clone())
            .or_default()
            .insert(key.clone());
    }

    /// Update indexes for delete
    fn update_indexes_for_delete(&self, _key: &TripleKey) {
        // Note: We don't actually remove from indexes during transaction
        // This happens during GC when versions are cleaned up
    }

    /// Check for write conflicts (pessimistic mode)
    fn check_write_conflict(&self, _key: &TripleKey, _tx_id: TransactionId) -> Result<()> {
        // In a real implementation, would check locks
        Ok(())
    }

    /// Create a snapshot
    fn create_snapshot(&self, timestamp: Timestamp) -> Result<()> {
        let active_txs: HashSet<TransactionId> = self
            .transactions
            .iter()
            .filter(|entry| entry.value().status == TransactionStatus::Active)
            .map(|entry| *entry.key())
            .collect();

        let snapshot = SnapshotInfo {
            timestamp,
            active_transactions: active_txs,
            ref_count: 1,
        };

        self.snapshots.write().insert(timestamp, snapshot);

        Ok(())
    }

    /// Get current timestamp
    fn get_current_timestamp(&self) -> Timestamp {
        self.timestamp_counter.load(Ordering::SeqCst)
    }

    /// Get next timestamp
    fn get_next_timestamp(&self) -> Timestamp {
        self.timestamp_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Maybe run garbage collection
    fn maybe_run_gc(&self) {
        let mut gc_state = self.gc_state.lock();

        if gc_state.last_gc.elapsed() >= self.config.gc_interval {
            // Run GC in background
            let versions = self.versions.clone();
            let config = self.config.clone();
            let min_timestamp = self.calculate_min_timestamp();

            std::thread::spawn(move || {
                Self::run_gc_internal(versions, config, min_timestamp);
            });

            gc_state.last_gc = Instant::now();
            gc_state.gc_runs += 1;
        }
    }

    /// Run garbage collection
    fn run_gc_internal(
        versions: Arc<DashMap<TripleKey, VersionChain>>,
        config: MvccConfig,
        min_timestamp: Timestamp,
    ) {
        for mut entry in versions.iter_mut() {
            entry
                .value_mut()
                .gc_versions(min_timestamp, config.max_versions_per_triple);
        }
    }

    /// Calculate minimum timestamp to keep
    fn calculate_min_timestamp(&self) -> Timestamp {
        // Keep versions needed by active transactions
        let min_active = self
            .transactions
            .iter()
            .filter(|entry| entry.value().status == TransactionStatus::Active)
            .map(|entry| entry.value().start_timestamp)
            .min()
            .unwrap_or(self.get_current_timestamp());

        // Keep versions needed by snapshots
        let min_snapshot = self
            .snapshots
            .read()
            .keys()
            .next()
            .copied()
            .unwrap_or(self.get_current_timestamp());

        min_active.min(min_snapshot)
    }

    /// Run garbage collection manually
    pub fn garbage_collect(&self) -> Result<()> {
        let min_timestamp = self.calculate_min_timestamp();
        Self::run_gc_internal(self.versions.clone(), self.config.clone(), min_timestamp);
        Ok(())
    }

    /// Get store statistics
    pub fn get_stats(&self) -> MvccStats {
        let total_versions = self
            .versions
            .iter()
            .map(|entry| entry.value().versions.len())
            .sum();

        let gc_state = self.gc_state.lock();

        MvccStats {
            total_triples: self.versions.len(),
            total_versions,
            active_transactions: self
                .transactions
                .iter()
                .filter(|entry| entry.value().status == TransactionStatus::Active)
                .count(),
            gc_runs: gc_state.gc_runs,
            versions_collected: gc_state.versions_collected,
        }
    }
}

/// MVCC statistics
#[derive(Debug, Clone)]
pub struct MvccStats {
    pub total_triples: usize,
    pub total_versions: usize,
    pub active_transactions: usize,
    pub gc_runs: u64,
    pub versions_collected: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_basic_mvcc_operations() {
        let config = MvccConfig::default();
        let store = MvccStore::new(config);

        // Begin transaction
        let tx1 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();

        // Insert triple
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("value"),
        );

        store.insert(tx1, triple.clone()).unwrap();

        // Query should see the triple (read-your-writes)
        let results = store.query(tx1, None, None, None).unwrap();
        assert_eq!(results.len(), 1);

        // Commit transaction
        store.commit_transaction(tx1).unwrap();

        // New transaction should see committed data
        let tx2 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let results = store.query(tx2, None, None, None).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_concurrent_transactions() {
        let config = MvccConfig::default();
        let store = MvccStore::new(config);

        // Insert initial data
        let tx0 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("initial"),
        );
        store.insert(tx0, triple.clone()).unwrap();
        store.commit_transaction(tx0).unwrap();

        // Start two concurrent transactions
        let tx1 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let tx2 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();

        // Both should see initial data
        assert_eq!(store.query(tx1, None, None, None).unwrap().len(), 1);
        assert_eq!(store.query(tx2, None, None, None).unwrap().len(), 1);

        // TX1 modifies data
        store.delete(tx1, &triple).unwrap();
        let new_triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("modified"),
        );
        store.insert(tx1, new_triple).unwrap();

        // TX2 shouldn't see TX1's changes yet
        let tx2_results = store.query(tx2, None, None, None).unwrap();
        assert_eq!(tx2_results.len(), 1);
        assert_eq!(tx2_results[0].object().to_string(), "\"initial\"");

        // Commit TX1
        store.commit_transaction(tx1).unwrap();

        // TX2 still sees snapshot
        let tx2_results = store.query(tx2, None, None, None).unwrap();
        assert_eq!(tx2_results.len(), 1);
        assert_eq!(tx2_results[0].object().to_string(), "\"initial\"");

        // New transaction sees committed changes
        let tx3 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let tx3_results = store.query(tx3, None, None, None).unwrap();
        assert_eq!(tx3_results.len(), 1);
        assert_eq!(tx3_results[0].object().to_string(), "\"modified\"");
    }

    #[test]
    fn test_write_conflict_detection() {
        let config = MvccConfig {
            conflict_detection: ConflictDetection::Optimistic,
            ..Default::default()
        };
        let store = MvccStore::new(config);

        // Insert initial data
        let tx0 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("initial"),
        );
        store.insert(tx0, triple.clone()).unwrap();
        store.commit_transaction(tx0).unwrap();

        // Start two transactions
        let tx1 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();
        let tx2 = store.begin_transaction(IsolationLevel::Snapshot).unwrap();

        // Both modify the same triple
        store.delete(tx1, &triple).unwrap();
        store.delete(tx2, &triple).unwrap();

        // First commit succeeds
        assert!(store.commit_transaction(tx1).is_ok());

        // Second commit should fail due to conflict
        assert!(store.commit_transaction(tx2).is_err());
    }

    #[test]
    fn test_version_chain() {
        let mut chain = VersionChain::new();

        // Add versions
        for i in 0..5 {
            let version = Version {
                id: i,
                timestamp: i * 10,
                transaction_id: i,
                deleted: false,
                triple: None,
                commit_timestamp: Some(i * 10 + 5),
            };
            chain.add_version(version);
        }

        // Test visibility
        assert_eq!(chain.get_visible_version(25).unwrap().id, 2);
        assert_eq!(chain.get_visible_version(45).unwrap().id, 4);

        // Test GC
        chain.gc_versions(20, 3);
        assert!(chain.versions.len() <= 3);
    }
}
