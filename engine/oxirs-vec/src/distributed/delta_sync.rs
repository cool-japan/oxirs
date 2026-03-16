//! Efficient delta synchronisation for distributed vector indexes.
//!
//! Instead of shipping entire index snapshots on every sync cycle, this module
//! computes and applies **deltas** — the minimal set of additions, removals and
//! modifications required to transform one index snapshot into another.
//!
//! # Key Types
//!
//! - `IndexSnapshot` — an in-memory view of the current index state.
//! - `VectorEntry` — a single vector with its ID, data and metadata.
//! - `IndexDelta` — the minimal diff between two snapshots.
//! - `DeltaSync` — stateless helper for computing and applying deltas.
//! - `ReplicationLag` — measures and evaluates sync lag between datacenters.
//! - `ReplicationAlert` — generated when lag exceeds a configured threshold.
//!
//! # Pure Rust Policy
//!
//! No CUDA runtime calls or FFI.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// VectorEntry
// ============================================================

/// A single vector stored in the index.
///
/// Note: a `VectorEntry` type also exists in `raft_index`; this one is
/// intentionally local to the delta-sync domain (it includes `version` for
/// conflict detection).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorEntry {
    /// Globally unique vector identifier
    pub id: u64,
    /// Optional human-readable label
    pub label: String,
    /// The vector data
    pub vector: Vec<f32>,
    /// Arbitrary key-value metadata
    pub metadata: HashMap<String, String>,
    /// Monotonically increasing version counter (updated on every write)
    pub version: u64,
}

impl VectorEntry {
    /// Create a new entry with the given `id`, `vector` and `version`.
    pub fn new(id: u64, vector: Vec<f32>, version: u64) -> Self {
        Self {
            id,
            label: String::new(),
            vector,
            metadata: HashMap::new(),
            version,
        }
    }

    /// Approximate serialised byte size (id + version + label bytes + vector f32s + metadata).
    pub fn approx_bytes(&self) -> u64 {
        let meta_bytes: u64 = self
            .metadata
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum();
        8 + 8 + self.label.len() as u64 + (self.vector.len() as u64 * 4) + meta_bytes
    }
}

// ============================================================
// IndexSnapshot
// ============================================================

/// An immutable snapshot of the current index state.
///
/// Internally stored as a map from vector ID to `VectorEntry` for O(1) lookups.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// All entries in this snapshot, keyed by vector ID
    pub entries: HashMap<u64, VectorEntry>,
    /// Sequence number associated with this snapshot
    pub seq: u64,
}

impl IndexSnapshot {
    /// Create an empty snapshot at sequence 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a snapshot with the given entries and sequence number.
    pub fn from_entries(entries: Vec<VectorEntry>, seq: u64) -> Self {
        Self {
            entries: entries.into_iter().map(|e| (e.id, e)).collect(),
            seq,
        }
    }

    /// Insert or update an entry.
    pub fn upsert(&mut self, entry: VectorEntry) {
        self.entries.insert(entry.id, entry);
    }

    /// Remove an entry by ID.  Returns `true` if the entry was present.
    pub fn remove(&mut self, id: u64) -> bool {
        self.entries.remove(&id).is_some()
    }

    /// Return the entry for the given ID, if present.
    pub fn get(&self, id: u64) -> Option<&VectorEntry> {
        self.entries.get(&id)
    }

    /// Number of entries in the snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the snapshot contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================
// IndexDelta
// ============================================================

/// The minimal diff between two index snapshots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexDelta {
    /// Entries that are new or have been updated in the newer snapshot
    pub added: Vec<VectorEntry>,
    /// IDs of entries that have been removed in the newer snapshot
    pub removed: Vec<u64>,
    /// Entries whose vector data or metadata has changed
    pub modified: Vec<VectorEntry>,
}

impl IndexDelta {
    /// Returns `true` if the delta contains no changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Total number of change operations in this delta.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }
}

// ============================================================
// DeltaSync
// ============================================================

/// Stateless helper for computing and applying index deltas.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaSync;

impl DeltaSync {
    /// Create a new `DeltaSync` instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute the delta required to transform `old_index` into `new_index`.
    ///
    /// An entry is considered **added** if it exists only in `new_index`.
    /// An entry is **removed** if it exists only in `old_index`.
    /// An entry is **modified** if it exists in both but the version number has
    /// increased in `new_index`.
    pub fn compute_delta(
        &self,
        old_index: &IndexSnapshot,
        new_index: &IndexSnapshot,
    ) -> IndexDelta {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        // Entries in new that are absent or newer in old
        for (id, new_entry) in &new_index.entries {
            match old_index.entries.get(id) {
                None => added.push(new_entry.clone()),
                Some(old_entry) => {
                    if new_entry.version > old_entry.version {
                        modified.push(new_entry.clone());
                    }
                }
            }
        }

        // Entries in old that are absent in new
        for id in old_index.entries.keys() {
            if !new_index.entries.contains_key(id) {
                removed.push(*id);
            }
        }

        IndexDelta {
            added,
            removed,
            modified,
        }
    }

    /// Apply `delta` to `base`, mutating it in place.
    ///
    /// Additions and modifications are upserted; removals are deleted.
    /// Returns an error if a removal targets an ID that does not exist in `base`
    /// (indicates a logic error in delta computation).
    pub fn apply_delta(&self, base: &mut IndexSnapshot, delta: &IndexDelta) -> Result<()> {
        for entry in &delta.added {
            base.upsert(entry.clone());
        }
        for entry in &delta.modified {
            base.upsert(entry.clone());
        }
        for &id in &delta.removed {
            if !base.remove(id) {
                return Err(anyhow!(
                    "Delta removal of ID {} failed: entry not found in base snapshot",
                    id
                ));
            }
        }
        Ok(())
    }

    /// Estimate the serialised byte size of a delta.
    ///
    /// Approximates: each removed ID costs 8 bytes; each added/modified entry
    /// uses `VectorEntry::approx_bytes()`.
    pub fn delta_size_bytes(&self, delta: &IndexDelta) -> u64 {
        let added_bytes: u64 = delta.added.iter().map(|e| e.approx_bytes()).sum();
        let modified_bytes: u64 = delta.modified.iter().map(|e| e.approx_bytes()).sum();
        let removed_bytes: u64 = (delta.removed.len() as u64) * 8;
        added_bytes + modified_bytes + removed_bytes
    }
}

// ============================================================
// ReplicationAlert
// ============================================================

/// Alert generated when replication lag exceeds a configured threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationAlert {
    /// Source datacenter
    pub dc_a: String,
    /// Destination datacenter
    pub dc_b: String,
    /// Measured lag in milliseconds
    pub measured_lag_ms: u64,
    /// The threshold that was exceeded
    pub threshold_ms: u64,
    /// Human-readable description
    pub message: String,
}

// ============================================================
// ReplicationLag
// ============================================================

/// Measures and evaluates replication lag between pairs of datacenters.
///
/// Lag measurements are stored in-memory; in production these would be
/// populated from heartbeat timestamps.
#[derive(Debug, Clone, Default)]
pub struct ReplicationLag {
    /// Measured lag per DC pair: key = (dc_a, dc_b) in sorted order
    measurements_ms: HashMap<(String, String), u64>,
}

impl ReplicationLag {
    /// Create a new lag tracker with no measurements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a lag measurement between `dc_a` and `dc_b` (directional).
    pub fn record(&mut self, dc_a: impl Into<String>, dc_b: impl Into<String>, lag_ms: u64) {
        self.measurements_ms
            .insert((dc_a.into(), dc_b.into()), lag_ms);
    }

    /// Retrieve the most recently recorded lag between `dc_a` and `dc_b` in
    /// milliseconds.  Returns 0 if no measurement has been recorded.
    pub fn lag_ms(&self, dc_a: &str, dc_b: &str) -> u64 {
        self.measurements_ms
            .get(&(dc_a.to_string(), dc_b.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Returns `true` if `lag_ms` is within the acceptable `sla_ms` bound.
    pub fn is_acceptable(&self, lag_ms: u64, sla_ms: u64) -> bool {
        lag_ms <= sla_ms
    }

    /// Generate an alert if `lag_ms` exceeds `threshold_ms`, otherwise returns
    /// `None`.
    pub fn alert_if_excessive(
        &self,
        dc_a: &str,
        dc_b: &str,
        lag_ms: u64,
        threshold_ms: u64,
    ) -> Option<ReplicationAlert> {
        if lag_ms > threshold_ms {
            Some(ReplicationAlert {
                dc_a: dc_a.to_string(),
                dc_b: dc_b.to_string(),
                measured_lag_ms: lag_ms,
                threshold_ms,
                message: format!(
                    "Replication lag from {} to {} is {} ms, exceeding threshold {} ms",
                    dc_a, dc_b, lag_ms, threshold_ms
                ),
            })
        } else {
            None
        }
    }

    /// Check recorded lag between `dc_a` and `dc_b` against a threshold and
    /// produce an alert if it is excessive.
    pub fn check_and_alert(
        &self,
        dc_a: &str,
        dc_b: &str,
        threshold_ms: u64,
    ) -> Option<ReplicationAlert> {
        let lag = self.lag_ms(dc_a, dc_b);
        self.alert_if_excessive(dc_a, dc_b, lag, threshold_ms)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn make_entry(id: u64, version: u64) -> VectorEntry {
        VectorEntry::new(id, vec![id as f32, version as f32], version)
    }

    fn make_snapshot(entries: Vec<(u64, u64)>) -> IndexSnapshot {
        let seq = entries.iter().map(|(_, v)| *v).max().unwrap_or(0);
        IndexSnapshot::from_entries(
            entries
                .into_iter()
                .map(|(id, ver)| make_entry(id, ver))
                .collect(),
            seq,
        )
    }

    // ---- IndexSnapshot ----

    #[test]
    fn test_snapshot_upsert_and_get() {
        let mut snap = IndexSnapshot::new();
        snap.upsert(make_entry(1, 1));
        assert!(snap.get(1).is_some());
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn test_snapshot_remove_existing() {
        let mut snap = make_snapshot(vec![(1, 1), (2, 1)]);
        assert!(snap.remove(1));
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn test_snapshot_remove_nonexistent() {
        let mut snap = IndexSnapshot::new();
        assert!(!snap.remove(99));
    }

    #[test]
    fn test_snapshot_is_empty() {
        let snap = IndexSnapshot::new();
        assert!(snap.is_empty());
    }

    // ---- VectorEntry ----

    #[test]
    fn test_vector_entry_approx_bytes_basic() {
        let e = make_entry(1, 1); // 2-element f32 vector
        let bytes = e.approx_bytes();
        // 8 (id) + 8 (version) + 0 (empty label) + 8 (2 * 4) + 0 (no meta) = 24
        assert_eq!(bytes, 24);
    }

    #[test]
    fn test_vector_entry_with_metadata_bytes() {
        let mut e = make_entry(1, 1);
        e.metadata.insert("key".into(), "value".into()); // 3 + 5 = 8 bytes
        let bytes = e.approx_bytes();
        assert_eq!(bytes, 32);
    }

    // ---- DeltaSync::compute_delta ----

    #[test]
    fn test_compute_delta_empty_to_empty() {
        let ds = DeltaSync::new();
        let old = IndexSnapshot::new();
        let new = IndexSnapshot::new();
        let delta = ds.compute_delta(&old, &new);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_compute_delta_all_added() {
        let ds = DeltaSync::new();
        let old = IndexSnapshot::new();
        let new = make_snapshot(vec![(1, 1), (2, 1), (3, 1)]);
        let delta = ds.compute_delta(&old, &new);
        assert_eq!(delta.added.len(), 3);
        assert!(delta.removed.is_empty());
        assert!(delta.modified.is_empty());
    }

    #[test]
    fn test_compute_delta_all_removed() {
        let ds = DeltaSync::new();
        let old = make_snapshot(vec![(1, 1), (2, 1)]);
        let new = IndexSnapshot::new();
        let delta = ds.compute_delta(&old, &new);
        assert_eq!(delta.removed.len(), 2);
        assert!(delta.added.is_empty());
        assert!(delta.modified.is_empty());
    }

    #[test]
    fn test_compute_delta_modifications() {
        let ds = DeltaSync::new();
        let old = make_snapshot(vec![(1, 1), (2, 1)]);
        let new = make_snapshot(vec![(1, 2), (2, 1)]); // entry 1 updated
        let delta = ds.compute_delta(&old, &new);
        assert_eq!(delta.modified.len(), 1);
        assert_eq!(delta.modified[0].id, 1);
        assert!(delta.added.is_empty());
        assert!(delta.removed.is_empty());
    }

    #[test]
    fn test_compute_delta_mixed() {
        let ds = DeltaSync::new();
        let old = make_snapshot(vec![(1, 1), (2, 1), (3, 1)]);
        // entry 2 updated, entry 3 removed, entry 4 added
        let new = make_snapshot(vec![(1, 1), (2, 2), (4, 1)]);
        let delta = ds.compute_delta(&old, &new);
        assert_eq!(delta.added.len(), 1); // entry 4
        assert_eq!(delta.removed.len(), 1); // entry 3
        assert_eq!(delta.modified.len(), 1); // entry 2
    }

    #[test]
    fn test_compute_delta_no_change_no_diff() {
        let ds = DeltaSync::new();
        let snap = make_snapshot(vec![(1, 5), (2, 3)]);
        let delta = ds.compute_delta(&snap, &snap);
        assert!(delta.is_empty());
    }

    // ---- DeltaSync::apply_delta ----

    #[test]
    fn test_apply_delta_add() -> Result<()> {
        let ds = DeltaSync::new();
        let mut base = IndexSnapshot::new();
        let delta = IndexDelta {
            added: vec![make_entry(1, 1)],
            removed: vec![],
            modified: vec![],
        };
        ds.apply_delta(&mut base, &delta)?;
        assert!(base.get(1).is_some());
        Ok(())
    }

    #[test]
    fn test_apply_delta_remove() -> Result<()> {
        let ds = DeltaSync::new();
        let mut base = make_snapshot(vec![(1, 1), (2, 1)]);
        let delta = IndexDelta {
            added: vec![],
            removed: vec![1],
            modified: vec![],
        };
        ds.apply_delta(&mut base, &delta)?;
        assert!(base.get(1).is_none());
        assert_eq!(base.len(), 1);
        Ok(())
    }

    #[test]
    fn test_apply_delta_modify() -> Result<()> {
        let ds = DeltaSync::new();
        let mut base = make_snapshot(vec![(1, 1)]);
        let updated = make_entry(1, 2);
        let delta = IndexDelta {
            added: vec![],
            removed: vec![],
            modified: vec![updated.clone()],
        };
        ds.apply_delta(&mut base, &delta)?;
        assert_eq!(base.get(1).expect("test value").version, 2);
        Ok(())
    }

    #[test]
    fn test_apply_delta_remove_nonexistent_errors() {
        let ds = DeltaSync::new();
        let mut base = IndexSnapshot::new();
        let delta = IndexDelta {
            added: vec![],
            removed: vec![99],
            modified: vec![],
        };
        assert!(ds.apply_delta(&mut base, &delta).is_err());
    }

    #[test]
    fn test_apply_delta_roundtrip() -> Result<()> {
        let ds = DeltaSync::new();
        let old = make_snapshot(vec![(1, 1), (2, 1), (3, 1)]);
        let new = make_snapshot(vec![(1, 2), (2, 1), (4, 1)]);
        let delta = ds.compute_delta(&old, &new);
        let mut applied = old.clone();
        ds.apply_delta(&mut applied, &delta)?;
        // applied should now match new
        assert_eq!(applied.len(), new.len());
        for (id, entry) in &new.entries {
            assert_eq!(applied.get(*id).map(|e| e.version), Some(entry.version));
        }
        Ok(())
    }

    // ---- DeltaSync::delta_size_bytes ----

    #[test]
    fn test_delta_size_bytes_empty() {
        let ds = DeltaSync::new();
        let delta = IndexDelta::default();
        assert_eq!(ds.delta_size_bytes(&delta), 0);
    }

    #[test]
    fn test_delta_size_bytes_removed_only() {
        let ds = DeltaSync::new();
        let delta = IndexDelta {
            added: vec![],
            removed: vec![1, 2, 3],
            modified: vec![],
        };
        assert_eq!(ds.delta_size_bytes(&delta), 24); // 3 * 8 bytes
    }

    #[test]
    fn test_delta_size_bytes_added() {
        let ds = DeltaSync::new();
        let entry = make_entry(1, 1); // 2-element vector => 24 bytes
        let expected = entry.approx_bytes();
        let delta = IndexDelta {
            added: vec![entry],
            removed: vec![],
            modified: vec![],
        };
        assert_eq!(ds.delta_size_bytes(&delta), expected);
    }

    // ---- ReplicationLag ----

    #[test]
    fn test_lag_ms_unknown_pair_is_zero() {
        let lag = ReplicationLag::new();
        assert_eq!(lag.lag_ms("dc-a", "dc-b"), 0);
    }

    #[test]
    fn test_lag_ms_after_record() {
        let mut lag = ReplicationLag::new();
        lag.record("dc-a", "dc-b", 500);
        assert_eq!(lag.lag_ms("dc-a", "dc-b"), 500);
    }

    #[test]
    fn test_is_acceptable_within_sla() {
        let lag = ReplicationLag::new();
        assert!(lag.is_acceptable(100, 500));
    }

    #[test]
    fn test_is_acceptable_equals_sla() {
        let lag = ReplicationLag::new();
        assert!(lag.is_acceptable(500, 500));
    }

    #[test]
    fn test_is_acceptable_exceeds_sla() {
        let lag = ReplicationLag::new();
        assert!(!lag.is_acceptable(501, 500));
    }

    #[test]
    fn test_alert_if_excessive_below_threshold() {
        let lag = ReplicationLag::new();
        let alert = lag.alert_if_excessive("dc-a", "dc-b", 100, 500);
        assert!(alert.is_none());
    }

    #[test]
    fn test_alert_if_excessive_above_threshold() -> Result<()> {
        let lag = ReplicationLag::new();
        let alert = lag.alert_if_excessive("dc-a", "dc-b", 1000, 500);
        assert!(alert.is_some());
        let a = alert.expect("alert was None");
        assert_eq!(a.measured_lag_ms, 1000);
        assert_eq!(a.threshold_ms, 500);
        assert!(!a.message.is_empty());
        Ok(())
    }

    #[test]
    fn test_check_and_alert_uses_recorded_lag() -> Result<()> {
        let mut lag = ReplicationLag::new();
        lag.record("dc-a", "dc-b", 999);
        let alert = lag.check_and_alert("dc-a", "dc-b", 500);
        assert!(alert.is_some());
        assert_eq!(alert.expect("test value").measured_lag_ms, 999);
        Ok(())
    }

    #[test]
    fn test_check_and_alert_no_alert_when_below() {
        let mut lag = ReplicationLag::new();
        lag.record("dc-a", "dc-b", 50);
        let alert = lag.check_and_alert("dc-a", "dc-b", 500);
        assert!(alert.is_none());
    }

    #[test]
    fn test_delta_change_count() {
        let delta = IndexDelta {
            added: vec![make_entry(1, 1)],
            removed: vec![2],
            modified: vec![make_entry(3, 2), make_entry(4, 3)],
        };
        assert_eq!(delta.change_count(), 4);
    }
}
