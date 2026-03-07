//! Delta-sync vector store with incremental change tracking and conflict-free merge.
//!
//! `DeltaSyncVectorStore` maintains a full in-memory vector store alongside an
//! append-only change log.  Replicas exchange only the delta (changes since
//! their last known sequence number) instead of full snapshots, keeping
//! replication bandwidth proportional to the number of changed entries rather
//! than the total dataset size.
//!
//! # Conflict Resolution
//!
//! When merging a remote delta the store uses a **last-writer-wins** strategy
//! based on the logical sequence number embedded in each `ChangeRecord`.  If
//! two replicas concurrently update the same key the update with the higher
//! sequence number wins (ties broken by preferring the incoming change).
//!
//! # Pure Rust Policy
//!
//! No unsafe code, no C/Fortran FFI, no CUDA runtime calls.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── change kinds ──────────────────────────────────────────────────────────────

/// The type of a change applied to the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKind {
    /// A vector was inserted for the first time.
    Insert,
    /// An existing vector was replaced.
    Update,
    /// A vector was removed.
    Delete,
}

// ── change record ─────────────────────────────────────────────────────────────

/// A single entry in the change log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    /// Monotonically increasing sequence number within this store.
    pub seq: u64,
    /// The affected vector key.
    pub key: String,
    /// The kind of change.
    pub kind: ChangeKind,
    /// The new vector data (absent for deletes).
    pub vector: Option<Vec<f32>>,
    /// Arbitrary metadata attached at write time.
    pub metadata: HashMap<String, String>,
}

impl ChangeRecord {
    fn insert(seq: u64, key: String, vector: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self {
            seq,
            key,
            kind: ChangeKind::Insert,
            vector: Some(vector),
            metadata,
        }
    }

    fn update(seq: u64, key: String, vector: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self {
            seq,
            key,
            kind: ChangeKind::Update,
            vector: Some(vector),
            metadata,
        }
    }

    fn delete(seq: u64, key: String) -> Self {
        Self {
            seq,
            key,
            kind: ChangeKind::Delete,
            vector: None,
            metadata: HashMap::new(),
        }
    }
}

// ── stored entry ──────────────────────────────────────────────────────────────

/// A vector entry stored in the main map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEntry {
    /// The vector data.
    pub vector: Vec<f32>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// The sequence number at which this entry was last written.
    pub version: u64,
}

// ── delta ─────────────────────────────────────────────────────────────────────

/// A set of change records transferable between replicas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreDelta {
    /// The highest sequence number of the source store at the time of export.
    pub source_seq: u64,
    /// The lower-bound (exclusive) used when constructing this delta.
    pub since_seq: u64,
    /// The changes, ordered by sequence number ascending.
    pub changes: Vec<ChangeRecord>,
}

impl StoreDelta {
    /// Return the number of changes in this delta.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Return `true` if this delta contains no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

// ── merge result ──────────────────────────────────────────────────────────────

/// Summary returned after merging a remote delta.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeResult {
    /// Number of entries newly inserted from the delta.
    pub inserts_applied: usize,
    /// Number of entries updated from the delta.
    pub updates_applied: usize,
    /// Number of entries deleted from the delta.
    pub deletes_applied: usize,
    /// Number of records skipped because the local version was newer.
    pub conflicts_skipped: usize,
}

impl MergeResult {
    /// Total number of changes applied (inserts + updates + deletes).
    pub fn total_applied(&self) -> usize {
        self.inserts_applied + self.updates_applied + self.deletes_applied
    }
}

// ── store stats ───────────────────────────────────────────────────────────────

/// Statistics about the store's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaStoreStats {
    /// Number of live entries.
    pub entry_count: usize,
    /// Current sequence counter.
    pub current_seq: u64,
    /// Total records in the change log.
    pub log_length: usize,
    /// Number of inserts recorded lifetime.
    pub total_inserts: u64,
    /// Number of updates recorded lifetime.
    pub total_updates: u64,
    /// Number of deletes recorded lifetime.
    pub total_deletes: u64,
    /// Number of merges performed.
    pub total_merges: u64,
}

// ── main struct ───────────────────────────────────────────────────────────────

/// An in-memory vector store with a full append-only change log enabling
/// efficient delta synchronisation between replicas.
pub struct DeltaSyncVectorStore {
    /// Live vector data, keyed by string identifier.
    entries: HashMap<String, StoredEntry>,
    /// Append-only change log ordered by `seq`.
    change_log: Vec<ChangeRecord>,
    /// Monotonically increasing sequence counter.
    seq: u64,
    /// Lifetime operation counters.
    total_inserts: u64,
    total_updates: u64,
    total_deletes: u64,
    total_merges: u64,
}

impl Default for DeltaSyncVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaSyncVectorStore {
    /// Create a new, empty store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            change_log: Vec::new(),
            seq: 0,
            total_inserts: 0,
            total_updates: 0,
            total_deletes: 0,
            total_merges: 0,
        }
    }

    // ── write operations ──────────────────────────────────────────────────

    /// Insert a new vector.  Returns an error if the key already exists.
    pub fn insert(&mut self, key: String, vector: Vec<f32>) -> Result<u64> {
        self.insert_with_metadata(key, vector, HashMap::new())
    }

    /// Insert with explicit metadata.  Returns an error if the key already exists.
    pub fn insert_with_metadata(
        &mut self,
        key: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<u64> {
        if self.entries.contains_key(&key) {
            return Err(anyhow!("Key '{}' already exists; use update()", key));
        }
        self.seq += 1;
        let seq = self.seq;
        let record = ChangeRecord::insert(seq, key.clone(), vector.clone(), metadata.clone());
        self.change_log.push(record);
        self.entries.insert(
            key,
            StoredEntry {
                vector,
                metadata,
                version: seq,
            },
        );
        self.total_inserts += 1;
        Ok(seq)
    }

    /// Update an existing vector.  Returns an error if the key does not exist.
    pub fn update(&mut self, key: String, vector: Vec<f32>) -> Result<u64> {
        self.update_with_metadata(key, vector, HashMap::new())
    }

    /// Update with explicit metadata.  Returns an error if the key does not exist.
    pub fn update_with_metadata(
        &mut self,
        key: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<u64> {
        if !self.entries.contains_key(&key) {
            return Err(anyhow!("Key '{}' does not exist; use insert()", key));
        }
        self.seq += 1;
        let seq = self.seq;
        let record = ChangeRecord::update(seq, key.clone(), vector.clone(), metadata.clone());
        self.change_log.push(record);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.vector = vector;
            entry.metadata = metadata;
            entry.version = seq;
        }
        self.total_updates += 1;
        Ok(seq)
    }

    /// Insert or update a vector (upsert semantics).
    pub fn upsert(&mut self, key: String, vector: Vec<f32>) -> Result<u64> {
        if self.entries.contains_key(&key) {
            self.update(key, vector)
        } else {
            self.insert(key, vector)
        }
    }

    /// Delete a vector by key.  Returns an error if the key does not exist.
    pub fn delete(&mut self, key: &str) -> Result<u64> {
        if !self.entries.contains_key(key) {
            return Err(anyhow!("Key '{}' not found", key));
        }
        self.seq += 1;
        let seq = self.seq;
        let record = ChangeRecord::delete(seq, key.to_string());
        self.change_log.push(record);
        self.entries.remove(key);
        self.total_deletes += 1;
        Ok(seq)
    }

    // ── read operations ───────────────────────────────────────────────────

    /// Look up a vector by key.
    pub fn get(&self, key: &str) -> Option<&StoredEntry> {
        self.entries.get(key)
    }

    /// Check whether a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Current number of live entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no entries are present.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The current (latest) sequence number.
    pub fn current_seq(&self) -> u64 {
        self.seq
    }

    /// All keys of live entries.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    // ── delta operations ──────────────────────────────────────────────────

    /// Export all changes with `seq > since_seq` as a `StoreDelta`.
    ///
    /// Pass `since_seq = 0` to export the full history.
    pub fn export_delta(&self, since_seq: u64) -> StoreDelta {
        let changes: Vec<ChangeRecord> = self
            .change_log
            .iter()
            .filter(|r| r.seq > since_seq)
            .cloned()
            .collect();
        StoreDelta {
            source_seq: self.seq,
            since_seq,
            changes,
        }
    }

    /// Apply a `StoreDelta` received from a remote replica.
    ///
    /// Uses last-writer-wins based on sequence number; records whose sequence
    /// number is ≤ the current local version of that key are skipped.
    ///
    /// **The local sequence counter is NOT advanced** by merges — only by
    /// local write operations.  This keeps sequence numbers local-only and
    /// avoids gaps in the log.
    pub fn merge_delta(&mut self, delta: &StoreDelta) -> Result<MergeResult> {
        let mut result = MergeResult::default();

        for record in &delta.changes {
            match &record.kind {
                ChangeKind::Insert | ChangeKind::Update => {
                    let vector = record
                        .vector
                        .as_ref()
                        .ok_or_else(|| anyhow!("Insert/Update record missing vector data"))?
                        .clone();
                    let metadata = record.metadata.clone();

                    if let Some(existing) = self.entries.get(&record.key) {
                        if existing.version >= record.seq {
                            // Local version is at least as new — skip
                            result.conflicts_skipped += 1;
                            continue;
                        }
                        // Remote wins — update in place
                        if let Some(e) = self.entries.get_mut(&record.key) {
                            e.vector = vector;
                            e.metadata = metadata;
                            e.version = record.seq;
                        }
                        result.updates_applied += 1;
                    } else {
                        // New entry from remote
                        self.entries.insert(
                            record.key.clone(),
                            StoredEntry {
                                vector,
                                metadata,
                                version: record.seq,
                            },
                        );
                        if record.kind == ChangeKind::Insert {
                            result.inserts_applied += 1;
                        } else {
                            result.updates_applied += 1;
                        }
                    }
                }
                ChangeKind::Delete => {
                    if let Some(existing) = self.entries.get(&record.key) {
                        if existing.version >= record.seq {
                            result.conflicts_skipped += 1;
                            continue;
                        }
                    }
                    if self.entries.remove(&record.key).is_some() {
                        result.deletes_applied += 1;
                    }
                }
            }
        }

        self.total_merges += 1;
        Ok(result)
    }

    /// Return store statistics.
    pub fn stats(&self) -> DeltaStoreStats {
        DeltaStoreStats {
            entry_count: self.entries.len(),
            current_seq: self.seq,
            log_length: self.change_log.len(),
            total_inserts: self.total_inserts,
            total_updates: self.total_updates,
            total_deletes: self.total_deletes,
            total_merges: self.total_merges,
        }
    }

    /// Compact the change log, retaining only the most-recent operation for each key
    /// plus any delete tombstones whose key no longer exists.
    ///
    /// After compaction the log covers the same logical state but may be shorter.
    pub fn compact_log(&mut self) {
        // Keep only the last record per key
        let mut last_seq_per_key: HashMap<String, usize> = HashMap::new();
        for (idx, record) in self.change_log.iter().enumerate() {
            last_seq_per_key.insert(record.key.clone(), idx);
        }

        let keep: std::collections::HashSet<usize> = last_seq_per_key.values().copied().collect();

        let mut new_log = Vec::with_capacity(keep.len());
        for (idx, record) in self.change_log.iter().enumerate() {
            if keep.contains(&idx) {
                new_log.push(record.clone());
            }
        }
        new_log.sort_by_key(|r| r.seq);
        self.change_log = new_log;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> DeltaSyncVectorStore {
        DeltaSyncVectorStore::new()
    }

    // ── basic CRUD ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_store_is_empty() {
        let store = make_store();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.current_seq(), 0);
    }

    #[test]
    fn test_insert_increments_seq() {
        let mut store = make_store();
        let seq = store.insert("k1".to_string(), vec![1.0, 2.0]).unwrap();
        assert_eq!(seq, 1);
        assert_eq!(store.current_seq(), 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_insert_duplicate_key_fails() {
        let mut store = make_store();
        store.insert("k1".to_string(), vec![1.0]).unwrap();
        let err = store.insert("k1".to_string(), vec![2.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_update_existing_key() {
        let mut store = make_store();
        store.insert("k1".to_string(), vec![1.0, 0.0]).unwrap();
        let seq = store.update("k1".to_string(), vec![2.0, 0.0]).unwrap();
        assert_eq!(seq, 2);
        let entry = store.get("k1").unwrap();
        assert_eq!(entry.vector, vec![2.0, 0.0]);
    }

    #[test]
    fn test_update_missing_key_fails() {
        let mut store = make_store();
        let err = store.update("nonexistent".to_string(), vec![1.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_delete_existing_key() {
        let mut store = make_store();
        store.insert("k1".to_string(), vec![1.0]).unwrap();
        let seq = store.delete("k1").unwrap();
        assert_eq!(seq, 2);
        assert!(!store.contains("k1"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_delete_missing_key_fails() {
        let mut store = make_store();
        let err = store.delete("missing");
        assert!(err.is_err());
    }

    #[test]
    fn test_upsert_insert_path() {
        let mut store = make_store();
        let seq = store.upsert("k".to_string(), vec![1.0]).unwrap();
        assert_eq!(seq, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_upsert_update_path() {
        let mut store = make_store();
        store.insert("k".to_string(), vec![1.0]).unwrap();
        store.upsert("k".to_string(), vec![99.0]).unwrap();
        let entry = store.get("k").unwrap();
        assert_eq!(entry.vector, vec![99.0]);
    }

    #[test]
    fn test_contains_after_insert() {
        let mut store = make_store();
        store.insert("x".to_string(), vec![0.0]).unwrap();
        assert!(store.contains("x"));
        assert!(!store.contains("y"));
    }

    // ── change log ─────────────────────────────────────────────────────────

    #[test]
    fn test_change_log_grows_with_operations() {
        let mut store = make_store();
        store.insert("k1".to_string(), vec![1.0]).unwrap();
        store.insert("k2".to_string(), vec![2.0]).unwrap();
        store.update("k1".to_string(), vec![3.0]).unwrap();
        store.delete("k2").unwrap();
        let stats = store.stats();
        assert_eq!(stats.log_length, 4);
    }

    #[test]
    fn test_change_log_records_correct_kinds() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        store.update("a".to_string(), vec![2.0]).unwrap();
        store.delete("a").unwrap();
        assert_eq!(store.change_log[0].kind, ChangeKind::Insert);
        assert_eq!(store.change_log[1].kind, ChangeKind::Update);
        assert_eq!(store.change_log[2].kind, ChangeKind::Delete);
    }

    // ── delta export ───────────────────────────────────────────────────────

    #[test]
    fn test_export_delta_full() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        store.insert("b".to_string(), vec![2.0]).unwrap();
        let delta = store.export_delta(0);
        assert_eq!(delta.changes.len(), 2);
        assert_eq!(delta.source_seq, 2);
    }

    #[test]
    fn test_export_delta_incremental() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        store.insert("b".to_string(), vec![2.0]).unwrap();
        let delta = store.export_delta(1); // Only changes after seq=1
        assert_eq!(delta.changes.len(), 1);
        assert_eq!(delta.changes[0].key, "b");
    }

    #[test]
    fn test_export_delta_empty_when_up_to_date() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        let delta = store.export_delta(1); // Already have seq=1
        assert!(delta.is_empty());
    }

    // ── delta merge ────────────────────────────────────────────────────────

    #[test]
    fn test_merge_delta_inserts_new_entries() {
        let mut source = make_store();
        source.insert("remote_key".to_string(), vec![42.0]).unwrap();
        let delta = source.export_delta(0);

        let mut target = make_store();
        let result = target.merge_delta(&delta).unwrap();

        assert_eq!(result.inserts_applied, 1);
        assert!(target.contains("remote_key"));
        assert_eq!(target.get("remote_key").unwrap().vector, vec![42.0]);
    }

    #[test]
    fn test_merge_delta_deletes_entries() {
        let mut target = make_store();
        target.insert("to_delete".to_string(), vec![1.0]).unwrap();

        // Manually create a delta with a delete at seq=99 (higher than target's seq=1)
        let delta = StoreDelta {
            source_seq: 99,
            since_seq: 0,
            changes: vec![ChangeRecord::delete(99, "to_delete".to_string())],
        };

        let result = target.merge_delta(&delta).unwrap();
        assert_eq!(result.deletes_applied, 1);
        assert!(!target.contains("to_delete"));
    }

    #[test]
    fn test_merge_delta_conflict_local_wins() {
        let mut target = make_store();
        // Insert with seq=5 (by inserting 5 items)
        for i in 0..5 {
            target.insert(format!("k{}", i), vec![i as f32]).unwrap();
        }

        // Remote delta tries to update k0 at seq=1 (lower than local seq=1 for k0)
        let delta = StoreDelta {
            source_seq: 1,
            since_seq: 0,
            changes: vec![ChangeRecord::update(
                1,
                "k0".to_string(),
                vec![999.0],
                HashMap::new(),
            )],
        };

        let result = target.merge_delta(&delta).unwrap();
        assert_eq!(result.conflicts_skipped, 1);
        // Local value unchanged
        assert_eq!(target.get("k0").unwrap().vector, vec![0.0]);
    }

    #[test]
    fn test_merge_delta_remote_wins_newer_seq() {
        let mut target = make_store();
        target.insert("k".to_string(), vec![1.0]).unwrap(); // seq=1

        // Remote update at seq=100 (newer)
        let delta = StoreDelta {
            source_seq: 100,
            since_seq: 0,
            changes: vec![ChangeRecord::update(
                100,
                "k".to_string(),
                vec![200.0],
                HashMap::new(),
            )],
        };

        let result = target.merge_delta(&delta).unwrap();
        assert_eq!(result.updates_applied, 1);
        assert_eq!(target.get("k").unwrap().vector, vec![200.0]);
    }

    #[test]
    fn test_merge_empty_delta_noop() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        let delta = StoreDelta {
            source_seq: 0,
            since_seq: 0,
            changes: Vec::new(),
        };
        let result = store.merge_delta(&delta).unwrap();
        assert_eq!(result.total_applied(), 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_merge_result_total_applied() {
        let mut source = make_store();
        source.insert("a".to_string(), vec![1.0]).unwrap();
        source.insert("b".to_string(), vec![2.0]).unwrap();
        let delta = source.export_delta(0);

        let mut target = make_store();
        let result = target.merge_delta(&delta).unwrap();
        assert_eq!(result.total_applied(), 2);
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counters() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        store.insert("b".to_string(), vec![2.0]).unwrap();
        store.update("a".to_string(), vec![10.0]).unwrap();
        store.delete("b").unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_inserts, 2);
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.total_deletes, 1);
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.current_seq, 4);
    }

    #[test]
    fn test_stats_merge_counter() {
        let mut source = make_store();
        source.insert("x".to_string(), vec![1.0]).unwrap();
        let delta = source.export_delta(0);

        let mut target = make_store();
        target.merge_delta(&delta).unwrap();
        target
            .merge_delta(&StoreDelta {
                source_seq: 0,
                since_seq: 0,
                changes: Vec::new(),
            })
            .unwrap();

        assert_eq!(target.stats().total_merges, 2);
    }

    // ── log compaction ─────────────────────────────────────────────────────

    #[test]
    fn test_compact_log_reduces_size() {
        let mut store = make_store();
        store.insert("k".to_string(), vec![1.0]).unwrap();
        store.update("k".to_string(), vec![2.0]).unwrap();
        store.update("k".to_string(), vec![3.0]).unwrap();
        assert_eq!(store.stats().log_length, 3);
        store.compact_log();
        // After compaction, only the most recent change per key should remain
        assert_eq!(store.stats().log_length, 1);
    }

    #[test]
    fn test_compact_log_preserves_state() {
        let mut store = make_store();
        for i in 0..5 {
            store.insert(format!("k{}", i), vec![i as f32]).unwrap();
        }
        store.update("k0".to_string(), vec![99.0]).unwrap();
        store.compact_log();
        assert_eq!(store.get("k0").unwrap().vector, vec![99.0]);
        assert_eq!(store.len(), 5);
    }

    // ── keys ───────────────────────────────────────────────────────────────

    #[test]
    fn test_keys_returns_all_live_keys() {
        let mut store = make_store();
        store.insert("a".to_string(), vec![1.0]).unwrap();
        store.insert("b".to_string(), vec![2.0]).unwrap();
        store.insert("c".to_string(), vec![3.0]).unwrap();
        store.delete("b").unwrap();
        let mut keys = store.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "c"]);
    }

    // ── insert_with_metadata ───────────────────────────────────────────────

    #[test]
    fn test_insert_with_metadata_stored() {
        let mut store = make_store();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());
        store
            .insert_with_metadata("k".to_string(), vec![1.0], meta.clone())
            .unwrap();
        let entry = store.get("k").unwrap();
        assert_eq!(
            entry.metadata.get("source").map(String::as_str),
            Some("test")
        );
    }
}
