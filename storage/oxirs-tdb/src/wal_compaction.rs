//! # WAL Compaction and Point-in-Time Recovery
//!
//! This module provides write-ahead log (WAL) segment compaction and
//! point-in-time recovery (PITR) for the TDB storage engine.
//!
//! ## WAL Compaction
//!
//! Over time the WAL accumulates segments. Compaction merges older segments,
//! discarding entries that have been superseded by later writes. The result
//! is a single compact segment covering the same logical time range.
//!
//! ## Point-in-Time Recovery
//!
//! PITR allows recovering the database state to any past timestamp by
//! replaying WAL entries up to (but not beyond) a specified recovery
//! point. Combined with periodic base snapshots, this provides
//! fine-grained disaster-recovery capability.
//!
//! ## Architecture
//!
//! ```text
//! WAL Segments:  [seg-001] [seg-002] [seg-003] [seg-004]
//!                   |___________|  compacted into [seg-compact-001]
//!
//! PITR Timeline: snap-001 -----seg-001-----seg-002-----seg-003------
//!                              ^                        ^
//!                         recovery_point           current
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::error::{Result, TdbError};

// ---------------------------------------------------------------------------
// WAL entry types
// ---------------------------------------------------------------------------

/// The type of a WAL record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalRecordType {
    /// Insert a triple / quad.
    Insert,
    /// Delete a triple / quad.
    Delete,
    /// Transaction begin marker.
    TxnBegin,
    /// Transaction commit marker.
    TxnCommit,
    /// Transaction abort marker.
    TxnAbort,
    /// Checkpoint / snapshot marker.
    Checkpoint,
    /// Schema change.
    SchemaChange,
}

/// A single WAL record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Log Sequence Number (monotonically increasing).
    pub lsn: u64,
    /// Record type.
    pub record_type: WalRecordType,
    /// Transaction ID (0 for non-transactional records).
    pub txn_id: u64,
    /// Timestamp of the record.
    pub timestamp: u64,
    /// Key being affected (e.g. serialized triple).
    pub key: Vec<u8>,
    /// Value being written (empty for deletes).
    pub value: Vec<u8>,
    /// CRC32 checksum of key + value.
    pub checksum: u32,
}

impl WalRecord {
    /// Create a new WAL record.
    pub fn new(
        lsn: u64,
        record_type: WalRecordType,
        txn_id: u64,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let checksum = Self::compute_checksum(&key, &value);
        Self {
            lsn,
            record_type,
            txn_id,
            timestamp,
            key,
            value,
            checksum,
        }
    }

    /// Compute CRC32 checksum.
    fn compute_checksum(key: &[u8], value: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(key);
        hasher.update(value);
        hasher.finalize()
    }

    /// Verify the checksum.
    pub fn verify_checksum(&self) -> bool {
        let expected = Self::compute_checksum(&self.key, &self.value);
        self.checksum == expected
    }

    /// Byte size estimate.
    pub fn estimated_size(&self) -> usize {
        8 + 1 + 8 + 8 + self.key.len() + self.value.len() + 4
    }
}

// ---------------------------------------------------------------------------
// WAL segment
// ---------------------------------------------------------------------------

/// A WAL segment is a contiguous sequence of WAL records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalSegment {
    /// Segment ID.
    pub segment_id: u64,
    /// Records in this segment, ordered by LSN.
    pub records: Vec<WalRecord>,
    /// First LSN in this segment.
    pub first_lsn: u64,
    /// Last LSN in this segment.
    pub last_lsn: u64,
    /// Whether this segment has been compacted.
    pub compacted: bool,
    /// Creation time.
    pub created_at: u64,
}

impl WalSegment {
    /// Create a new segment.
    pub fn new(segment_id: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            segment_id,
            records: Vec::new(),
            first_lsn: 0,
            last_lsn: 0,
            compacted: false,
            created_at: now,
        }
    }

    /// Add a record to the segment.
    pub fn add_record(&mut self, record: WalRecord) {
        if self.records.is_empty() {
            self.first_lsn = record.lsn;
        }
        self.last_lsn = record.lsn;
        self.records.push(record);
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Estimated size in bytes.
    pub fn estimated_size(&self) -> usize {
        self.records.iter().map(|r| r.estimated_size()).sum()
    }

    /// Get the timestamp range of this segment.
    pub fn timestamp_range(&self) -> Option<(u64, u64)> {
        if self.records.is_empty() {
            return None;
        }
        let first_ts = self.records.first().map(|r| r.timestamp).unwrap_or(0);
        let last_ts = self.records.last().map(|r| r.timestamp).unwrap_or(0);
        Some((first_ts, last_ts))
    }
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// Configuration for WAL compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Minimum number of segments before compaction triggers.
    pub min_segments_to_compact: usize,
    /// Maximum number of segments to compact in one pass.
    pub max_segments_per_pass: usize,
    /// Whether to remove superseded insert+delete pairs.
    pub remove_dead_entries: bool,
    /// Whether to preserve transaction boundaries.
    pub preserve_transactions: bool,
    /// Whether to verify checksums during compaction.
    pub verify_checksums: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_segments_to_compact: 3,
            max_segments_per_pass: 10,
            remove_dead_entries: true,
            preserve_transactions: true,
            verify_checksums: true,
        }
    }
}

/// Statistics from a compaction run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Number of segments compacted.
    pub segments_compacted: usize,
    /// Number of records before compaction.
    pub records_before: usize,
    /// Number of records after compaction.
    pub records_after: usize,
    /// Number of dead entries removed.
    pub dead_entries_removed: usize,
    /// Bytes saved.
    pub bytes_saved: usize,
    /// Duration of compaction.
    pub duration_ms: u64,
    /// Number of checksum failures detected.
    pub checksum_failures: usize,
}

/// WAL compaction engine.
pub struct WalCompactor {
    config: CompactionConfig,
}

impl WalCompactor {
    /// Create a new compactor with the given configuration.
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Compact a set of WAL segments into a single segment.
    pub fn compact(&self, segments: &[WalSegment]) -> Result<(WalSegment, CompactionStats)> {
        let start = std::time::Instant::now();
        let mut stats = CompactionStats::default();

        if segments.is_empty() {
            return Ok((WalSegment::new(0), stats));
        }

        stats.segments_compacted = segments.len();

        // Collect all records in LSN order
        let mut all_records: Vec<&WalRecord> =
            segments.iter().flat_map(|s| s.records.iter()).collect();
        all_records.sort_by_key(|r| r.lsn);

        stats.records_before = all_records.len();

        // Verify checksums if configured
        if self.config.verify_checksums {
            for record in &all_records {
                if !record.verify_checksum() {
                    stats.checksum_failures += 1;
                }
            }
        }

        // Build a map of key -> latest state to identify dead entries
        let mut key_state: HashMap<Vec<u8>, WalRecordType> = HashMap::new();
        let mut live_lsns: std::collections::HashSet<u64> = std::collections::HashSet::new();

        if self.config.remove_dead_entries {
            // First pass: determine which keys are still live
            for record in &all_records {
                match record.record_type {
                    WalRecordType::Insert => {
                        key_state.insert(record.key.clone(), WalRecordType::Insert);
                    }
                    WalRecordType::Delete => {
                        key_state.insert(record.key.clone(), WalRecordType::Delete);
                    }
                    _ => {}
                }
            }

            // Second pass: mark records for live keys and txn records
            for record in &all_records {
                let keep = match record.record_type {
                    WalRecordType::Insert => {
                        // Keep only if the key's final state is Insert
                        matches!(key_state.get(&record.key), Some(WalRecordType::Insert))
                    }
                    WalRecordType::Delete => {
                        // Remove: insert+delete pairs cancel out
                        false
                    }
                    WalRecordType::TxnBegin
                    | WalRecordType::TxnCommit
                    | WalRecordType::TxnAbort => self.config.preserve_transactions,
                    WalRecordType::Checkpoint | WalRecordType::SchemaChange => true,
                };

                if keep {
                    live_lsns.insert(record.lsn);
                }
            }
        } else {
            // Keep all records
            for record in &all_records {
                live_lsns.insert(record.lsn);
            }
        }

        // Build the compacted segment
        let first_segment_id = segments.first().map(|s| s.segment_id).unwrap_or(0);
        let mut compacted = WalSegment::new(first_segment_id);
        compacted.compacted = true;

        let size_before: usize = all_records.iter().map(|r| r.estimated_size()).sum();

        for record in &all_records {
            if live_lsns.contains(&record.lsn) {
                compacted.add_record((*record).clone());
            }
        }

        stats.records_after = compacted.len();
        stats.dead_entries_removed = stats.records_before - stats.records_after;

        let size_after: usize = compacted.records.iter().map(|r| r.estimated_size()).sum();
        stats.bytes_saved = size_before.saturating_sub(size_after);
        stats.duration_ms = start.elapsed().as_millis() as u64;

        Ok((compacted, stats))
    }

    /// Check if compaction should be triggered.
    pub fn should_compact(&self, segment_count: usize) -> bool {
        segment_count >= self.config.min_segments_to_compact
    }

    /// Get configuration.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Point-in-Time Recovery
// ---------------------------------------------------------------------------

/// A base snapshot reference for PITR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseSnapshot {
    /// Snapshot ID.
    pub snapshot_id: u64,
    /// LSN at which the snapshot was taken.
    pub lsn: u64,
    /// Timestamp when the snapshot was taken.
    pub timestamp: u64,
    /// Path to the snapshot file.
    pub path: PathBuf,
    /// Size of the snapshot in bytes.
    pub size_bytes: u64,
}

/// Configuration for point-in-time recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitrConfig {
    /// Whether PITR is enabled.
    pub enabled: bool,
    /// How often to take base snapshots (in records).
    pub snapshot_interval_records: u64,
    /// Maximum number of snapshots to retain.
    pub max_snapshots: usize,
    /// Whether to verify checksums during recovery.
    pub verify_checksums: bool,
}

impl Default for PitrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval_records: 100_000,
            max_snapshots: 5,
            verify_checksums: true,
        }
    }
}

/// A recovery plan produced by the PITR engine.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Base snapshot to start from.
    pub base_snapshot: Option<BaseSnapshot>,
    /// WAL segments to replay (in order).
    pub segments_to_replay: Vec<u64>,
    /// Maximum LSN to replay up to.
    pub target_lsn: Option<u64>,
    /// Maximum timestamp to replay up to.
    pub target_timestamp: Option<u64>,
    /// Estimated number of records to replay.
    pub estimated_records: u64,
}

impl RecoveryPlan {
    /// Whether this plan requires a base snapshot.
    pub fn needs_snapshot(&self) -> bool {
        self.base_snapshot.is_some()
    }

    /// Whether this plan requires WAL replay.
    pub fn needs_wal_replay(&self) -> bool {
        !self.segments_to_replay.is_empty()
    }
}

/// PITR recovery engine.
pub struct PitrEngine {
    config: PitrConfig,
    /// Known base snapshots, ordered by LSN.
    snapshots: BTreeMap<u64, BaseSnapshot>,
    /// Known WAL segments, ordered by segment ID.
    segments: BTreeMap<u64, WalSegment>,
}

impl PitrEngine {
    /// Create a new PITR engine.
    pub fn new(config: PitrConfig) -> Self {
        Self {
            config,
            snapshots: BTreeMap::new(),
            segments: BTreeMap::new(),
        }
    }

    /// Register a base snapshot.
    pub fn register_snapshot(&mut self, snapshot: BaseSnapshot) {
        self.snapshots.insert(snapshot.lsn, snapshot);

        // Enforce max_snapshots
        while self.snapshots.len() > self.config.max_snapshots {
            if let Some((&oldest_lsn, _)) = self.snapshots.iter().next() {
                self.snapshots.remove(&oldest_lsn);
            }
        }
    }

    /// Register a WAL segment.
    pub fn register_segment(&mut self, segment: WalSegment) {
        self.segments.insert(segment.segment_id, segment);
    }

    /// Plan recovery to a specific LSN.
    pub fn plan_recovery_to_lsn(&self, target_lsn: u64) -> Result<RecoveryPlan> {
        // Find the best base snapshot (latest one with LSN <= target)
        let base_snapshot = self
            .snapshots
            .range(..=target_lsn)
            .next_back()
            .map(|(_, s)| s.clone());

        let start_lsn = base_snapshot.as_ref().map(|s| s.lsn).unwrap_or(0);

        // Find segments needed for replay
        let segments_to_replay: Vec<u64> = self
            .segments
            .values()
            .filter(|s| s.last_lsn > start_lsn && s.first_lsn <= target_lsn)
            .map(|s| s.segment_id)
            .collect();

        // Estimate records to replay
        let estimated_records: u64 = self
            .segments
            .values()
            .filter(|s| segments_to_replay.contains(&s.segment_id))
            .map(|s| s.records.len() as u64)
            .sum();

        Ok(RecoveryPlan {
            base_snapshot,
            segments_to_replay,
            target_lsn: Some(target_lsn),
            target_timestamp: None,
            estimated_records,
        })
    }

    /// Plan recovery to a specific timestamp.
    pub fn plan_recovery_to_timestamp(&self, target_timestamp: u64) -> Result<RecoveryPlan> {
        // Find the best base snapshot (latest one with timestamp <= target)
        let base_snapshot = self
            .snapshots
            .values()
            .rfind(|s| s.timestamp <= target_timestamp)
            .cloned();

        let start_lsn = base_snapshot.as_ref().map(|s| s.lsn).unwrap_or(0);

        // Find segments that overlap the recovery window
        let segments_to_replay: Vec<u64> = self
            .segments
            .values()
            .filter(|s| {
                if let Some((_, last_ts)) = s.timestamp_range() {
                    last_ts
                        > base_snapshot
                            .as_ref()
                            .map(|snap| snap.timestamp)
                            .unwrap_or(0)
                        && s.first_lsn > start_lsn
                } else {
                    false
                }
            })
            .map(|s| s.segment_id)
            .collect();

        let estimated_records: u64 = self
            .segments
            .values()
            .filter(|s| segments_to_replay.contains(&s.segment_id))
            .map(|s| s.records.len() as u64)
            .sum();

        Ok(RecoveryPlan {
            base_snapshot,
            segments_to_replay,
            target_lsn: None,
            target_timestamp: Some(target_timestamp),
            estimated_records,
        })
    }

    /// Execute a recovery plan: filter WAL records up to the target.
    pub fn execute_plan(&self, plan: &RecoveryPlan) -> Result<Vec<WalRecord>> {
        let mut recovered_records = Vec::new();

        for seg_id in &plan.segments_to_replay {
            if let Some(segment) = self.segments.get(seg_id) {
                for record in &segment.records {
                    // Check LSN bound
                    if let Some(target_lsn) = plan.target_lsn {
                        if record.lsn > target_lsn {
                            break;
                        }
                    }

                    // Check timestamp bound
                    if let Some(target_ts) = plan.target_timestamp {
                        if record.timestamp > target_ts {
                            break;
                        }
                    }

                    // Verify checksum if configured
                    if self.config.verify_checksums && !record.verify_checksum() {
                        return Err(TdbError::Wal(format!(
                            "Checksum verification failed for record LSN {}",
                            record.lsn
                        )));
                    }

                    recovered_records.push(record.clone());
                }
            }
        }

        Ok(recovered_records)
    }

    /// Get the number of registered snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the number of registered segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get latest snapshot LSN.
    pub fn latest_snapshot_lsn(&self) -> Option<u64> {
        self.snapshots.keys().next_back().copied()
    }

    /// Get configuration.
    pub fn config(&self) -> &PitrConfig {
        &self.config
    }

    /// Check if a snapshot should be taken based on the current LSN.
    pub fn should_take_snapshot(&self, current_lsn: u64) -> bool {
        if !self.config.enabled {
            return false;
        }
        let latest_lsn = self.latest_snapshot_lsn().unwrap_or(0);
        current_lsn.saturating_sub(latest_lsn) >= self.config.snapshot_interval_records
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(lsn: u64, rt: WalRecordType, key: &[u8], value: &[u8]) -> WalRecord {
        WalRecord::new(lsn, rt, 1, key.to_vec(), value.to_vec())
    }

    fn make_segment(id: u64, records: Vec<WalRecord>) -> WalSegment {
        let mut seg = WalSegment::new(id);
        for r in records {
            seg.add_record(r);
        }
        seg
    }

    // --- WalRecord ---

    #[test]
    fn test_wal_record_creation() {
        let rec = make_record(1, WalRecordType::Insert, b"key1", b"val1");
        assert_eq!(rec.lsn, 1);
        assert_eq!(rec.record_type, WalRecordType::Insert);
        assert_eq!(rec.key, b"key1");
        assert_eq!(rec.value, b"val1");
    }

    #[test]
    fn test_wal_record_checksum_valid() {
        let rec = make_record(1, WalRecordType::Insert, b"key", b"value");
        assert!(rec.verify_checksum());
    }

    #[test]
    fn test_wal_record_checksum_invalid() {
        let mut rec = make_record(1, WalRecordType::Insert, b"key", b"value");
        rec.key = b"modified".to_vec(); // Tamper with the key
        assert!(!rec.verify_checksum());
    }

    #[test]
    fn test_wal_record_estimated_size() {
        let rec = make_record(1, WalRecordType::Insert, b"key", b"value");
        assert!(rec.estimated_size() > 0);
    }

    // --- WalSegment ---

    #[test]
    fn test_segment_new() {
        let seg = WalSegment::new(1);
        assert_eq!(seg.segment_id, 1);
        assert!(seg.is_empty());
        assert_eq!(seg.len(), 0);
    }

    #[test]
    fn test_segment_add_records() {
        let mut seg = WalSegment::new(1);
        seg.add_record(make_record(1, WalRecordType::Insert, b"k1", b"v1"));
        seg.add_record(make_record(2, WalRecordType::Insert, b"k2", b"v2"));
        assert_eq!(seg.len(), 2);
        assert_eq!(seg.first_lsn, 1);
        assert_eq!(seg.last_lsn, 2);
    }

    #[test]
    fn test_segment_estimated_size() {
        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Insert, b"k2", b"v2"),
            ],
        );
        assert!(seg.estimated_size() > 0);
    }

    #[test]
    fn test_segment_timestamp_range() {
        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Insert, b"k2", b"v2"),
            ],
        );
        let range = seg.timestamp_range();
        assert!(range.is_some());
        let (first, last) = range.expect("should have range");
        assert!(last >= first);
    }

    #[test]
    fn test_segment_timestamp_range_empty() {
        let seg = WalSegment::new(1);
        assert!(seg.timestamp_range().is_none());
    }

    // --- CompactionConfig ---

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionConfig::default();
        assert_eq!(config.min_segments_to_compact, 3);
        assert_eq!(config.max_segments_per_pass, 10);
        assert!(config.remove_dead_entries);
        assert!(config.preserve_transactions);
        assert!(config.verify_checksums);
    }

    // --- WalCompactor ---

    #[test]
    fn test_compactor_should_compact() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        assert!(!compactor.should_compact(1));
        assert!(!compactor.should_compact(2));
        assert!(compactor.should_compact(3));
        assert!(compactor.should_compact(10));
    }

    #[test]
    fn test_compact_empty_segments() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let (result, stats) = compactor.compact(&[]).expect("should succeed");
        assert!(result.is_empty());
        assert_eq!(stats.segments_compacted, 0);
    }

    #[test]
    fn test_compact_single_segment() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Insert, b"k2", b"v2"),
            ],
        );

        let (result, stats) = compactor.compact(&[seg]).expect("should succeed");
        assert_eq!(stats.segments_compacted, 1);
        assert_eq!(stats.records_before, 2);
        assert_eq!(stats.records_after, 2);
    }

    #[test]
    fn test_compact_removes_dead_entries() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg1 = make_segment(1, vec![make_record(1, WalRecordType::Insert, b"k1", b"v1")]);
        let seg2 = make_segment(2, vec![make_record(2, WalRecordType::Delete, b"k1", b"")]);

        let (result, stats) = compactor.compact(&[seg1, seg2]).expect("should succeed");
        assert_eq!(stats.segments_compacted, 2);
        assert_eq!(stats.records_before, 2);
        // Insert + Delete for same key = both removed
        assert_eq!(stats.dead_entries_removed, 2);
        assert_eq!(stats.records_after, 0);
    }

    #[test]
    fn test_compact_preserves_live_entries() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg1 = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Insert, b"k2", b"v2"),
            ],
        );
        let seg2 = make_segment(2, vec![make_record(3, WalRecordType::Delete, b"k1", b"")]);

        let (result, stats) = compactor.compact(&[seg1, seg2]).expect("should succeed");
        // k1: insert then delete -> dead. k2: insert only -> live
        assert_eq!(stats.records_after, 1);
    }

    #[test]
    fn test_compact_preserves_txn_records() {
        let config = CompactionConfig {
            preserve_transactions: true,
            ..Default::default()
        };
        let compactor = WalCompactor::new(config);

        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::TxnBegin, b"", b""),
                make_record(2, WalRecordType::Insert, b"k1", b"v1"),
                make_record(3, WalRecordType::TxnCommit, b"", b""),
            ],
        );

        let (result, stats) = compactor.compact(&[seg]).expect("should succeed");
        // TxnBegin + Insert + TxnCommit all preserved
        assert_eq!(stats.records_after, 3);
    }

    #[test]
    fn test_compact_without_dead_entry_removal() {
        let config = CompactionConfig {
            remove_dead_entries: false,
            ..Default::default()
        };
        let compactor = WalCompactor::new(config);

        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Delete, b"k1", b""),
            ],
        );

        let (_, stats) = compactor.compact(&[seg]).expect("should succeed");
        assert_eq!(stats.records_after, 2); // Nothing removed
        assert_eq!(stats.dead_entries_removed, 0);
    }

    #[test]
    fn test_compact_checksum_failures() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let mut rec = make_record(1, WalRecordType::Insert, b"k1", b"v1");
        rec.checksum = 0xDEADBEEF; // Corrupt checksum

        let seg = make_segment(1, vec![rec]);
        let (_, stats) = compactor.compact(&[seg]).expect("should succeed");
        assert_eq!(stats.checksum_failures, 1);
    }

    #[test]
    fn test_compact_preserves_checkpoint() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg = make_segment(
            1,
            vec![make_record(1, WalRecordType::Checkpoint, b"snap", b"data")],
        );

        let (result, stats) = compactor.compact(&[seg]).expect("should succeed");
        assert_eq!(stats.records_after, 1);
    }

    #[test]
    fn test_compact_preserves_schema_change() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg = make_segment(
            1,
            vec![make_record(
                1,
                WalRecordType::SchemaChange,
                b"schema",
                b"data",
            )],
        );

        let (_, stats) = compactor.compact(&[seg]).expect("should succeed");
        assert_eq!(stats.records_after, 1);
    }

    #[test]
    fn test_compact_stats_bytes_saved() {
        let compactor = WalCompactor::new(CompactionConfig::default());
        let seg1 = make_segment(
            1,
            vec![make_record(1, WalRecordType::Insert, b"key", b"val")],
        );
        let seg2 = make_segment(2, vec![make_record(2, WalRecordType::Delete, b"key", b"")]);

        let (_, stats) = compactor.compact(&[seg1, seg2]).expect("should succeed");
        assert!(stats.bytes_saved > 0);
    }

    // --- PitrConfig ---

    #[test]
    fn test_pitr_config_default() {
        let config = PitrConfig::default();
        assert!(config.enabled);
        assert_eq!(config.snapshot_interval_records, 100_000);
        assert_eq!(config.max_snapshots, 5);
    }

    // --- PitrEngine ---

    #[test]
    fn test_pitr_engine_creation() {
        let engine = PitrEngine::new(PitrConfig::default());
        assert_eq!(engine.snapshot_count(), 0);
        assert_eq!(engine.segment_count(), 0);
    }

    #[test]
    fn test_pitr_register_snapshot() {
        let mut engine = PitrEngine::new(PitrConfig::default());
        engine.register_snapshot(BaseSnapshot {
            snapshot_id: 1,
            lsn: 100,
            timestamp: 1000,
            path: PathBuf::from("/tmp/snap-1"),
            size_bytes: 1024,
        });
        assert_eq!(engine.snapshot_count(), 1);
        assert_eq!(engine.latest_snapshot_lsn(), Some(100));
    }

    #[test]
    fn test_pitr_register_segment() {
        let mut engine = PitrEngine::new(PitrConfig::default());
        let seg = make_segment(1, vec![make_record(1, WalRecordType::Insert, b"k", b"v")]);
        engine.register_segment(seg);
        assert_eq!(engine.segment_count(), 1);
    }

    #[test]
    fn test_pitr_max_snapshots_enforced() {
        let config = PitrConfig {
            max_snapshots: 2,
            ..Default::default()
        };
        let mut engine = PitrEngine::new(config);

        for i in 0..5u64 {
            engine.register_snapshot(BaseSnapshot {
                snapshot_id: i,
                lsn: i * 100,
                timestamp: i * 1000,
                path: PathBuf::from(format!("/tmp/snap-{i}")),
                size_bytes: 1024,
            });
        }

        assert_eq!(engine.snapshot_count(), 2);
    }

    #[test]
    fn test_pitr_plan_recovery_to_lsn() {
        let mut engine = PitrEngine::new(PitrConfig::default());

        engine.register_snapshot(BaseSnapshot {
            snapshot_id: 1,
            lsn: 100,
            timestamp: 1000,
            path: PathBuf::from("/tmp/snap-1"),
            size_bytes: 1024,
        });

        let seg = make_segment(
            1,
            vec![
                make_record(101, WalRecordType::Insert, b"k1", b"v1"),
                make_record(150, WalRecordType::Insert, b"k2", b"v2"),
                make_record(200, WalRecordType::Insert, b"k3", b"v3"),
            ],
        );
        engine.register_segment(seg);

        let plan = engine.plan_recovery_to_lsn(150).expect("should succeed");
        assert!(plan.needs_snapshot());
        assert!(plan.needs_wal_replay());
        assert_eq!(plan.target_lsn, Some(150));
    }

    #[test]
    fn test_pitr_plan_recovery_no_snapshot() {
        let mut engine = PitrEngine::new(PitrConfig::default());

        let seg = make_segment(1, vec![make_record(1, WalRecordType::Insert, b"k1", b"v1")]);
        engine.register_segment(seg);

        let plan = engine.plan_recovery_to_lsn(1).expect("should succeed");
        assert!(!plan.needs_snapshot());
        assert!(plan.needs_wal_replay());
    }

    #[test]
    fn test_pitr_execute_plan_lsn_bounded() {
        let mut engine = PitrEngine::new(PitrConfig::default());

        let seg = make_segment(
            1,
            vec![
                make_record(1, WalRecordType::Insert, b"k1", b"v1"),
                make_record(2, WalRecordType::Insert, b"k2", b"v2"),
                make_record(3, WalRecordType::Insert, b"k3", b"v3"),
            ],
        );
        engine.register_segment(seg);

        let plan = engine.plan_recovery_to_lsn(2).expect("should succeed");
        let records = engine.execute_plan(&plan).expect("should succeed");

        // Should replay records with LSN <= 2
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].lsn, 1);
        assert_eq!(records[1].lsn, 2);
    }

    #[test]
    fn test_pitr_execute_plan_checksum_failure() {
        let config = PitrConfig {
            verify_checksums: true,
            ..Default::default()
        };
        let mut engine = PitrEngine::new(config);

        let mut rec = make_record(1, WalRecordType::Insert, b"k", b"v");
        rec.checksum = 0xBAD; // Corrupt

        let seg = make_segment(1, vec![rec]);
        engine.register_segment(seg);

        let plan = engine.plan_recovery_to_lsn(1).expect("should succeed");
        let result = engine.execute_plan(&plan);
        assert!(result.is_err());
    }

    #[test]
    fn test_pitr_should_take_snapshot() {
        let config = PitrConfig {
            snapshot_interval_records: 100,
            ..Default::default()
        };
        let mut engine = PitrEngine::new(config);

        assert!(engine.should_take_snapshot(100));
        assert!(!engine.should_take_snapshot(50));

        engine.register_snapshot(BaseSnapshot {
            snapshot_id: 1,
            lsn: 50,
            timestamp: 1000,
            path: PathBuf::from("/tmp/snap"),
            size_bytes: 0,
        });

        assert!(!engine.should_take_snapshot(100)); // 100 - 50 = 50 < 100
        assert!(engine.should_take_snapshot(150)); // 150 - 50 = 100 >= 100
    }

    #[test]
    fn test_pitr_disabled() {
        let config = PitrConfig {
            enabled: false,
            ..Default::default()
        };
        let engine = PitrEngine::new(config);
        assert!(!engine.should_take_snapshot(999999));
    }

    // --- RecoveryPlan ---

    #[test]
    fn test_recovery_plan_needs() {
        let plan = RecoveryPlan {
            base_snapshot: None,
            segments_to_replay: vec![1],
            target_lsn: Some(100),
            target_timestamp: None,
            estimated_records: 50,
        };
        assert!(!plan.needs_snapshot());
        assert!(plan.needs_wal_replay());

        let plan_with_snap = RecoveryPlan {
            base_snapshot: Some(BaseSnapshot {
                snapshot_id: 1,
                lsn: 50,
                timestamp: 1000,
                path: PathBuf::from("/tmp/snap"),
                size_bytes: 0,
            }),
            segments_to_replay: vec![],
            target_lsn: Some(50),
            target_timestamp: None,
            estimated_records: 0,
        };
        assert!(plan_with_snap.needs_snapshot());
        assert!(!plan_with_snap.needs_wal_replay());
    }

    #[test]
    fn test_compactor_config_access() {
        let config = CompactionConfig {
            min_segments_to_compact: 5,
            ..Default::default()
        };
        let compactor = WalCompactor::new(config);
        assert_eq!(compactor.config().min_segments_to_compact, 5);
    }

    #[test]
    fn test_pitr_engine_config_access() {
        let config = PitrConfig {
            max_snapshots: 10,
            ..Default::default()
        };
        let engine = PitrEngine::new(config);
        assert_eq!(engine.config().max_snapshots, 10);
    }
}
