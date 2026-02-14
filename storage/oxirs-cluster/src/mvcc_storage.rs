//! Version-aware storage backend for MVCC
//!
//! This module provides storage implementations that integrate with the MVCC system,
//! including multi-version indices, efficient version lookup, and compaction strategies.

use crate::mvcc::{HLCTimestamp, MVCCManager, TransactionSnapshot};
use crate::shard::ShardId;
use crate::storage::StorageBackend;
use crate::transaction::{IsolationLevel, TransactionId};
use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
#[allow(unused_imports)]
use oxirs_core::model::{
    BlankNode, Literal, NamedNode, Object, Predicate, QuotedTriple, Subject, Triple, Variable,
};
#[cfg(test)]
use oxirs_core::vocab::xsd;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Index key type for efficient lookups
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKey {
    /// Subject index key
    Subject(String),
    /// Predicate index key
    Predicate(String),
    /// Object index key
    Object(String),
    /// Subject-Predicate composite key
    SubjectPredicate(String, String),
    /// Predicate-Object composite key
    PredicateObject(String, String),
    /// Subject-Object composite key
    SubjectObject(String, String),
    /// Full triple key
    Triple(String, String, String),
}

impl IndexKey {
    /// Create index key from triple components
    pub fn from_triple_pattern(
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Self {
        match (subject, predicate, object) {
            (Some(s), Some(p), Some(o)) => {
                IndexKey::Triple(s.to_string(), p.to_string(), o.to_string())
            }
            (Some(s), Some(p), None) => IndexKey::SubjectPredicate(s.to_string(), p.to_string()),
            (Some(s), None, Some(o)) => IndexKey::SubjectObject(s.to_string(), o.to_string()),
            (None, Some(p), Some(o)) => IndexKey::PredicateObject(p.to_string(), o.to_string()),
            (Some(s), None, None) => IndexKey::Subject(s.to_string()),
            (None, Some(p), None) => IndexKey::Predicate(p.to_string()),
            (None, None, Some(o)) => IndexKey::Object(o.to_string()),
            (None, None, None) => IndexKey::Subject("*".to_string()), // Wildcard
        }
    }

    /// Convert to storage key
    pub fn to_storage_key(&self) -> String {
        match self {
            IndexKey::Subject(s) => format!("s:{s}"),
            IndexKey::Predicate(p) => format!("p:{p}"),
            IndexKey::Object(o) => format!("o:{o}"),
            IndexKey::SubjectPredicate(s, p) => format!("sp:{s}:{p}"),
            IndexKey::PredicateObject(p, o) => format!("po:{p}:{o}"),
            IndexKey::SubjectObject(s, o) => format!("so:{s}:{o}"),
            IndexKey::Triple(s, p, o) => format!("spo:{s}:{p}:{o}"),
        }
    }
}

/// Multi-version index for efficient lookups
pub struct MVCCIndex {
    /// Primary index: IndexKey -> Set of triple keys with versions
    primary_index: Arc<DashMap<IndexKey, BTreeMap<HLCTimestamp, HashSet<String>>>>,
    /// Reverse index: Triple key -> IndexKeys
    reverse_index: Arc<DashMap<String, HashSet<IndexKey>>>,
}

impl Default for MVCCIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MVCCIndex {
    /// Create a new MVCC index
    pub fn new() -> Self {
        Self {
            primary_index: Arc::new(DashMap::new()),
            reverse_index: Arc::new(DashMap::new()),
        }
    }

    /// Index a triple version
    pub fn index_triple(&self, triple: &Triple, timestamp: HLCTimestamp, triple_key: &str) {
        let subject = subject_to_string(triple.subject());
        let predicate = predicate_to_string(triple.predicate());
        let object = object_to_string(triple.object());

        // Generate all index keys for this triple
        let index_keys = vec![
            IndexKey::Subject(subject.clone()),
            IndexKey::Predicate(predicate.clone()),
            IndexKey::Object(object.clone()),
            IndexKey::SubjectPredicate(subject.clone(), predicate.clone()),
            IndexKey::PredicateObject(predicate.clone(), object.clone()),
            IndexKey::SubjectObject(subject.clone(), object.clone()),
            IndexKey::Triple(subject, predicate, object),
        ];

        // Update primary index
        for key in &index_keys {
            self.primary_index
                .entry(key.clone())
                .or_default()
                .entry(timestamp)
                .or_default()
                .insert(triple_key.to_string());
        }

        // Update reverse index
        self.reverse_index
            .entry(triple_key.to_string())
            .or_default()
            .extend(index_keys);
    }

    /// Remove a triple from indices
    pub fn remove_triple(&self, triple_key: &str, timestamp: HLCTimestamp) {
        if let Some(index_keys) = self.reverse_index.get(triple_key) {
            for key in index_keys.value() {
                if let Some(mut entry) = self.primary_index.get_mut(key) {
                    if let Some(keys_at_timestamp) = entry.get_mut(&timestamp) {
                        keys_at_timestamp.remove(triple_key);
                        if keys_at_timestamp.is_empty() {
                            entry.remove(&timestamp);
                        }
                    }
                }
            }
        }
    }

    /// Query index with version awareness
    pub fn query(
        &self,
        index_key: &IndexKey,
        timestamp: &HLCTimestamp,
        include_uncommitted: bool,
    ) -> HashSet<String> {
        let mut results = HashSet::new();

        if let Some(versions) = self.primary_index.get(index_key) {
            // Get all versions up to the given timestamp
            for (ts, keys) in versions.range(..=timestamp) {
                if include_uncommitted || ts <= timestamp {
                    results.extend(keys.clone());
                }
            }
        }

        results
    }

    /// Get index statistics
    pub fn get_statistics(&self) -> IndexStatistics {
        let total_index_entries = self.primary_index.len();
        let total_triple_keys = self.reverse_index.len();

        let mut max_versions_per_index = 0;
        for entry in self.primary_index.iter() {
            max_versions_per_index = max_versions_per_index.max(entry.value().len());
        }

        IndexStatistics {
            total_index_entries,
            total_triple_keys,
            max_versions_per_index,
        }
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub total_index_entries: usize,
    pub total_triple_keys: usize,
    pub max_versions_per_index: usize,
}

/// Version compaction strategy
#[derive(Debug, Clone, Copy)]
pub enum CompactionStrategy {
    /// Keep all versions
    None,
    /// Keep only the latest N versions
    KeepLatest(usize),
    /// Keep versions newer than a certain age
    TimeBasedRetention(std::time::Duration),
    /// Hybrid: keep latest N or within time window
    Hybrid {
        max_versions: usize,
        retention_period: std::time::Duration,
    },
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        CompactionStrategy::Hybrid {
            max_versions: 100,
            retention_period: std::time::Duration::from_secs(86400), // 24 hours
        }
    }
}

/// MVCC-aware storage backend
pub struct MVCCStorage {
    /// MVCC manager
    mvcc: Arc<MVCCManager>,
    /// Multi-version index
    index: Arc<MVCCIndex>,
    /// Base storage path
    base_path: String,
    /// Compaction strategy
    compaction_strategy: CompactionStrategy,
    /// Statistics
    stats: Arc<RwLock<StorageStatistics>>,
}

impl MVCCStorage {
    /// Create a new MVCC storage backend
    pub fn new(node_id: u64, base_path: String, compaction_strategy: CompactionStrategy) -> Self {
        let mvcc = Arc::new(MVCCManager::new(
            node_id,
            crate::mvcc::MVCCConfig::default(),
        ));

        Self {
            mvcc,
            index: Arc::new(MVCCIndex::new()),
            base_path,
            compaction_strategy,
            stats: Arc::new(RwLock::new(StorageStatistics::default())),
        }
    }

    /// Start the storage backend
    pub async fn start(&self) -> Result<()> {
        // Start MVCC manager
        self.mvcc.start().await?;

        // Create base directory if needed
        tokio::fs::create_dir_all(&self.base_path).await?;

        info!("MVCC storage started at {}", self.base_path);
        Ok(())
    }

    /// Stop the storage backend
    pub async fn stop(&self) -> Result<()> {
        self.mvcc.stop().await?;
        info!("MVCC storage stopped");
        Ok(())
    }

    /// Begin a new transaction
    pub async fn begin_transaction(
        &self,
        transaction_id: TransactionId,
        isolation_level: IsolationLevel,
    ) -> Result<TransactionSnapshot> {
        self.mvcc
            .begin_transaction(transaction_id, isolation_level)
            .await
    }

    /// Insert a triple within a transaction
    pub async fn insert_triple(
        &self,
        transaction_id: &TransactionId,
        triple: Triple,
    ) -> Result<()> {
        // For simplicity in transactions, use shard 0
        // In production, this would determine the appropriate shard
        let shard_id = 0;
        let key = self.shard_triple_to_key(shard_id, &triple);

        // Write to MVCC
        self.mvcc
            .write(transaction_id, &key, Some(triple.clone()))
            .await?;

        // Update index (this would be done at commit time in production)
        let timestamp = self.mvcc.current_timestamp();
        self.index.index_triple(&triple, timestamp, &key);

        // Update statistics
        self.stats.write().await.total_inserts += 1;

        Ok(())
    }

    /// Delete a triple within a transaction
    pub async fn delete_triple(
        &self,
        transaction_id: &TransactionId,
        triple: Triple,
    ) -> Result<()> {
        let key = self.triple_to_key(&triple);

        // Write deletion marker to MVCC
        self.mvcc.write(transaction_id, &key, None).await?;

        // Update statistics
        self.stats.write().await.total_deletes += 1;

        Ok(())
    }

    /// Query triples with MVCC
    pub async fn query_triples(
        &self,
        transaction_id: &TransactionId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        let index_key = IndexKey::from_triple_pattern(subject, predicate, object);

        // Get transaction snapshot
        let snapshot = self
            .mvcc
            .begin_transaction(transaction_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        // Query index
        let triple_keys = self.index.query(
            &index_key,
            &snapshot.timestamp,
            snapshot.isolation_level == IsolationLevel::ReadUncommitted,
        );

        // Read each triple with MVCC
        let mut results = Vec::new();
        for key in triple_keys {
            if let Some(triple) = self.mvcc.read(transaction_id, &key).await? {
                // Apply pattern matching
                if matches_pattern(&triple, subject, predicate, object) {
                    results.push(triple);
                }
            }
        }

        // Update statistics
        self.stats.write().await.total_queries += 1;

        Ok(results)
    }

    /// Commit a transaction
    pub async fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        self.mvcc.commit_transaction(transaction_id).await?;
        self.stats.write().await.total_commits += 1;
        Ok(())
    }

    /// Rollback a transaction
    pub async fn rollback_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        self.mvcc.rollback_transaction(transaction_id).await?;
        self.stats.write().await.total_rollbacks += 1;
        Ok(())
    }

    /// Run compaction based on configured strategy
    pub async fn compact(&self) -> Result<CompactionResult> {
        let start_time = std::time::Instant::now();
        let versions_removed = 0;
        let keys_processed = 0;

        match self.compaction_strategy {
            CompactionStrategy::None => {
                // No compaction
                return Ok(CompactionResult {
                    duration: start_time.elapsed(),
                    versions_removed: 0,
                    keys_processed: 0,
                });
            }
            CompactionStrategy::KeepLatest(_n) => {
                // Keep only the latest N versions
                // This would be implemented by calling MVCC manager's internal methods
                warn!("KeepLatest compaction not yet implemented");
            }
            CompactionStrategy::TimeBasedRetention(_retention) => {
                // Remove versions older than retention period
                warn!("TimeBasedRetention compaction not yet implemented");
            }
            CompactionStrategy::Hybrid {
                max_versions: _,
                retention_period: _,
            } => {
                // Hybrid compaction
                warn!("Hybrid compaction not yet implemented");
            }
        }

        Ok(CompactionResult {
            duration: start_time.elapsed(),
            versions_removed,
            keys_processed,
        })
    }

    /// Get storage statistics
    pub async fn get_statistics(&self) -> StorageStatistics {
        let stats = self.stats.read().await.clone();
        stats
    }

    /// Convert triple to storage key
    fn triple_to_key(&self, triple: &Triple) -> String {
        format!(
            "{}:{}:{}",
            subject_to_string(triple.subject()),
            predicate_to_string(triple.predicate()),
            object_to_string(triple.object())
        )
    }

    fn shard_triple_to_key(&self, shard_id: ShardId, triple: &Triple) -> String {
        format!("shard:{}:{}", shard_id, self.triple_to_key(triple))
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStatistics {
    pub total_inserts: u64,
    pub total_deletes: u64,
    pub total_queries: u64,
    pub total_commits: u64,
    pub total_rollbacks: u64,
}

/// Compaction result
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub duration: std::time::Duration,
    pub versions_removed: usize,
    pub keys_processed: usize,
}

/// MVCC storage backend trait implementation
#[async_trait]
impl StorageBackend for MVCCStorage {
    async fn create_shard(&self, _shard_id: ShardId) -> Result<()> {
        // For MVCC storage, shards are logical partitions
        // No physical creation needed
        Ok(())
    }

    async fn delete_shard(&self, _shard_id: ShardId) -> Result<()> {
        // Mark all triples in the shard as deleted
        // This would be implemented via a shard-wide deletion marker
        Ok(())
    }

    async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()> {
        // Create a temporary transaction for non-transactional inserts
        let tx_id = format!("shard_{}_insert_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        // Use shard-prefixed key
        let key = format!("shard:{}:{}", shard_id, self.triple_to_key(&triple));
        self.mvcc.write(&tx_id, &key, Some(triple.clone())).await?;

        // Update index (same as regular insert)
        let timestamp = self.mvcc.current_timestamp();
        self.index.index_triple(&triple, timestamp, &key);

        // Update statistics
        self.stats.write().await.total_inserts += 1;

        self.commit_transaction(&tx_id).await
    }

    async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()> {
        // Create a temporary transaction for non-transactional deletes
        let tx_id = format!("shard_{}_delete_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        // Use shard-prefixed key
        let key = format!("shard:{}:{}", shard_id, self.triple_to_key(triple));
        self.mvcc.write(&tx_id, &key, None).await?;

        self.commit_transaction(&tx_id).await
    }

    async fn query_shard(
        &self,
        shard_id: ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        // Create a temporary transaction for queries
        let tx_id = format!("shard_{}_query_{}", shard_id, uuid::Uuid::new_v4());

        // Begin transaction to get snapshot
        let _snapshot = self
            .mvcc
            .begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        // For pattern queries on shards, we need to scan with the shard prefix
        // This is a simplified implementation
        let mut results = Vec::new();

        // For now, let's use the original query_triples approach but filter for shard keys
        let index_key = IndexKey::from_triple_pattern(subject, predicate, object);
        let triple_keys = self.index.query(
            &index_key,
            &_snapshot.timestamp,
            _snapshot.isolation_level == IsolationLevel::ReadUncommitted,
        );

        let shard_prefix = format!("shard:{shard_id}:");

        for key in triple_keys {
            // Only consider keys that belong to this shard
            if key.starts_with(&shard_prefix) {
                if let Some(triple) = self.mvcc.read(&tx_id, &key).await? {
                    if matches_pattern(&triple, subject, predicate, object) {
                        results.push(triple);
                    }
                }
            }
        }

        Ok(results)
    }

    async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64> {
        // Estimate based on triple count and average size
        let count = self.get_shard_triple_count(shard_id).await?;
        Ok((count * 100) as u64) // Estimate 100 bytes per triple
    }

    async fn get_shard_triple_count(&self, _shard_id: ShardId) -> Result<usize> {
        // Count triples in the shard
        // This would need proper implementation with shard filtering
        let stats = self.mvcc.get_statistics().await;
        Ok(stats.total_keys / 10) // Simplified: assume even distribution
    }

    async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        // Export all triples from a shard
        self.query_shard(shard_id, None, None, None).await
    }

    async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        // Import triples into a shard
        let tx_id = format!("shard_{}_import_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        for triple in triples {
            let key = format!("shard:{}:{}", shard_id, self.triple_to_key(&triple));
            self.mvcc.write(&tx_id, &key, Some(triple)).await?;
        }

        self.commit_transaction(&tx_id).await
    }

    async fn get_shard_triples(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        // Query all triples from a specific shard
        let tx_id = format!("shard_{}_get_triples_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        let prefix = format!("shard:{shard_id}:");
        let results = self.mvcc.scan_prefix(&tx_id, &prefix).await?;

        let mut triples = Vec::new();
        for (_, triple) in results {
            triples.push(triple);
        }

        self.commit_transaction(&tx_id).await?;
        Ok(triples)
    }

    async fn insert_triples_to_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        // Insert multiple triples into a shard
        let tx_id = format!("shard_{}_insert_bulk_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        for triple in triples {
            let key = format!("shard:{}:{}", shard_id, self.triple_to_key(&triple));
            self.mvcc.write(&tx_id, &key, Some(triple)).await?;
        }

        self.commit_transaction(&tx_id).await
    }

    async fn mark_shard_for_deletion(&self, shard_id: ShardId) -> Result<()> {
        // Mark a shard for deletion by creating a deletion marker
        let tx_id = format!("shard_{}_mark_delete_{}", shard_id, uuid::Uuid::new_v4());
        self.begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await?;

        let deletion_marker_key = format!("shard:{shard_id}:__MARKED_FOR_DELETION__");
        let marker_triple = Triple::new(
            Subject::NamedNode(
                NamedNode::new("urn:oxirs:shard:deleted").expect("valid static URI"),
            ),
            Predicate::NamedNode(
                NamedNode::new("urn:oxirs:prop:deletionMarker").expect("valid static URI"),
            ),
            Object::Literal(Literal::new_simple_literal("true")),
        );

        self.mvcc
            .write(&tx_id, &deletion_marker_key, Some(marker_triple))
            .await?;
        self.commit_transaction(&tx_id).await
    }
}

// Helper functions

fn subject_to_string(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(n) => n.as_str().to_string(),
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(t) => format!(
            "<<{} {} {}>>",
            subject_to_string(t.subject()),
            predicate_to_string(t.predicate()),
            object_to_string(t.object())
        ),
    }
}

fn predicate_to_string(predicate: &Predicate) -> String {
    match predicate {
        Predicate::NamedNode(n) => n.as_str().to_string(),
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    }
}

fn object_to_string(object: &Object) -> String {
    match object {
        Object::NamedNode(n) => n.as_str().to_string(),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(l) => {
            if let Some(lang) = l.language() {
                format!("\"{}\"@{}", l.value(), lang)
            } else {
                let dt = l.datatype();
                format!("\"{}\"^^<{}>", l.value(), dt.as_str())
            }
        }
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(t) => format!(
            "<<{} {} {}>>",
            subject_to_string(t.subject()),
            predicate_to_string(t.predicate()),
            object_to_string(t.object())
        ),
    }
}

fn matches_pattern(
    triple: &Triple,
    subject: Option<&str>,
    predicate: Option<&str>,
    object: Option<&str>,
) -> bool {
    if let Some(s) = subject {
        if subject_to_string(triple.subject()) != s {
            return false;
        }
    }
    if let Some(p) = predicate {
        if predicate_to_string(triple.predicate()) != p {
            return false;
        }
    }
    if let Some(o) = object {
        if object_to_string(triple.object()) != o {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_key_creation() {
        let key1 = IndexKey::from_triple_pattern(Some("s"), Some("p"), Some("o"));
        assert!(matches!(key1, IndexKey::Triple(_, _, _)));

        let key2 = IndexKey::from_triple_pattern(Some("s"), Some("p"), None);
        assert!(matches!(key2, IndexKey::SubjectPredicate(_, _)));

        let key3 = IndexKey::from_triple_pattern(Some("s"), None, None);
        assert!(matches!(key3, IndexKey::Subject(_)));
    }

    #[test]
    fn test_index_key_to_storage_key() {
        let key = IndexKey::Triple("s".to_string(), "p".to_string(), "o".to_string());
        assert_eq!(key.to_storage_key(), "spo:s:p:o");

        let key = IndexKey::Subject("s".to_string());
        assert_eq!(key.to_storage_key(), "s:s");
    }

    #[tokio::test]
    async fn test_mvcc_storage_basic() {
        let storage = MVCCStorage::new(1, "/tmp/mvcc_test".to_string(), CompactionStrategy::None);
        storage.start().await.unwrap();

        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new_typed_literal("value", xsd::STRING.clone()),
        );

        // Insert triple using StorageBackend trait
        storage
            .insert_triple_to_shard(0, triple.clone())
            .await
            .unwrap();

        // Query triple using StorageBackend trait
        let results = storage
            .query_shard(0, Some("http://example.org/s"), None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        // Check statistics
        let stats = storage.get_statistics().await;
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.total_commits, 1);

        storage.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_mvcc_storage_transaction() {
        let storage =
            MVCCStorage::new(1, "/tmp/mvcc_test_tx".to_string(), CompactionStrategy::None);
        storage.start().await.unwrap();

        let tx_id = "test_tx".to_string();
        storage
            .begin_transaction(tx_id.clone(), IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new_typed_literal("value", xsd::STRING.clone()),
        );

        // Insert within transaction
        storage.insert_triple(&tx_id, triple.clone()).await.unwrap();

        // Commit transaction
        storage.commit_transaction(&tx_id).await.unwrap();

        // Verify triple is persisted
        let results = storage
            .query_shard(0, Some("http://example.org/s"), None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        storage.stop().await.unwrap();
    }

    #[test]
    fn test_compaction_strategy() {
        let strategy = CompactionStrategy::default();
        match strategy {
            CompactionStrategy::Hybrid {
                max_versions,
                retention_period,
            } => {
                assert_eq!(max_versions, 100);
                assert_eq!(retention_period.as_secs(), 86400);
            }
            _ => panic!("Expected hybrid strategy"),
        }
    }
}
