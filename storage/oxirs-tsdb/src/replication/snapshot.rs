//! Raft snapshot support for TSDB high-availability replication.
//!
//! In the Raft protocol, snapshots allow a leader to send the current
//! committed state to slow or newly-joined followers without replaying the
//! entire log.  This module provides the snapshot format and install/restore
//! helpers used by the TSDB Raft integration.
//!
//! ## Snapshot format
//!
//! A [`TsdbRaftSnapshot`] contains:
//!
//! - `last_included_index` — the log index up to which all entries are
//!   compacted into this snapshot (§7 of the Raft paper).
//! - `last_included_term` — the Raft term of that log entry.
//! - `data` — an opaque byte payload (JSON array of serialised
//!   [`SnapshotDataPoint`] records in the current implementation).
//!
//! The JSON payload was chosen for portability; a production system might use
//! a more compact binary format (e.g. the existing Gorilla-encoded chunks) but
//! that would couple this module tightly to the storage layer.
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_tsdb::replication::snapshot::{SnapshotDataPoint, TsdbRaftSnapshot};
//!
//! let points = vec![
//!     SnapshotDataPoint {
//!         series_id: "cpu.usage".into(),
//!         timestamp: 1_714_300_000_000,
//!         value: 42.7,
//!     },
//! ];
//! let snapshot = TsdbRaftSnapshot::from_data_points(1, 1, &points)
//!     .expect("create snapshot");
//! assert_eq!(snapshot.last_included_index, 1);
//! assert_eq!(snapshot.last_included_term, 1);
//!
//! let restored = snapshot.to_data_points().expect("restore snapshot");
//! assert_eq!(restored.len(), 1);
//! assert_eq!(restored[0].series_id, "cpu.usage");
//! ```

use serde::{Deserialize, Serialize};

// ────────────────────────────────────────────────────────────────────────────
// SnapshotDataPoint
// ────────────────────────────────────────────────────────────────────────────

/// A single time-series observation included in a Raft snapshot payload.
///
/// This is the portable, serialisable representation used inside the snapshot
/// byte buffer.  It intentionally avoids references to the internal chunk or
/// columnar storage structures so that the snapshot module has no compile-time
/// dependency on the storage internals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotDataPoint {
    /// Series / metric identifier.
    pub series_id: String,
    /// Observation timestamp in milliseconds since Unix epoch.
    pub timestamp: i64,
    /// Measured floating-point value.
    pub value: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// SnapshotError
// ────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during snapshot creation or restoration.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    /// The snapshot payload could not be serialised to or from JSON.
    #[error("Snapshot serialisation error: {0}")]
    Serialization(String),
    /// The snapshot metadata is invalid (e.g. index == 0 when expecting ≥ 1).
    #[error("Invalid snapshot metadata: {0}")]
    InvalidMetadata(String),
}

// ────────────────────────────────────────────────────────────────────────────
// TsdbRaftSnapshot
// ────────────────────────────────────────────────────────────────────────────

/// A Raft snapshot for the TSDB state machine.
///
/// Contains the log index and term of the last compacted entry together with
/// an opaque byte payload that encodes the current committed TSDB state.
///
/// Create via [`TsdbRaftSnapshot::from_data_points`]; restore via
/// [`TsdbRaftSnapshot::to_data_points`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsdbRaftSnapshot {
    /// Raft log index of the last entry included in this snapshot (1-based).
    pub last_included_index: u64,
    /// Raft term of `last_included_index`.
    pub last_included_term: u64,
    /// Serialised TSDB state (JSON array of [`SnapshotDataPoint`]).
    pub data: Vec<u8>,
}

impl TsdbRaftSnapshot {
    /// Create a snapshot from a slice of [`SnapshotDataPoint`] values.
    ///
    /// `last_included_index` must be ≥ 1 (the Raft log is 1-indexed).
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotError::InvalidMetadata`] if `last_included_index == 0`,
    /// or [`SnapshotError::Serialization`] if JSON encoding fails.
    pub fn from_data_points(
        last_included_index: u64,
        last_included_term: u64,
        points: &[SnapshotDataPoint],
    ) -> Result<Self, SnapshotError> {
        if last_included_index == 0 {
            return Err(SnapshotError::InvalidMetadata(
                "last_included_index must be ≥ 1".into(),
            ));
        }
        let data =
            serde_json::to_vec(points).map_err(|e| SnapshotError::Serialization(e.to_string()))?;
        Ok(TsdbRaftSnapshot {
            last_included_index,
            last_included_term,
            data,
        })
    }

    /// Deserialise the snapshot payload back to a `Vec<SnapshotDataPoint>`.
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotError::Serialization`] if the embedded JSON is
    /// malformed.
    pub fn to_data_points(&self) -> Result<Vec<SnapshotDataPoint>, SnapshotError> {
        serde_json::from_slice(&self.data).map_err(|e| SnapshotError::Serialization(e.to_string()))
    }

    /// Return the byte size of the snapshot payload.
    pub fn data_size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the snapshot supersedes (is newer than) `other`.
    ///
    /// A snapshot A supersedes B iff `A.last_included_index > B.last_included_index`,
    /// or the indices are equal but `A.last_included_term > B.last_included_term`.
    pub fn supersedes(&self, other: &TsdbRaftSnapshot) -> bool {
        self.last_included_index > other.last_included_index
            || (self.last_included_index == other.last_included_index
                && self.last_included_term > other.last_included_term)
    }

    /// Serialise the entire [`TsdbRaftSnapshot`] (metadata + payload) to bytes
    /// for network transfer in an `InstallSnapshot` RPC.
    pub fn to_wire_bytes(&self) -> Result<Vec<u8>, SnapshotError> {
        serde_json::to_vec(self).map_err(|e| SnapshotError::Serialization(e.to_string()))
    }

    /// Deserialise a [`TsdbRaftSnapshot`] from bytes received via an
    /// `InstallSnapshot` RPC.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        serde_json::from_slice(bytes).map_err(|e| SnapshotError::Serialization(e.to_string()))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SnapshotStore — simple in-memory snapshot registry for single-node testing
// ────────────────────────────────────────────────────────────────────────────

/// Minimal in-memory snapshot store used during testing and single-node
/// embedded deployments.
///
/// A production implementation would persist the snapshot to disk (e.g.
/// alongside the WAL segment files) so that it survives a process restart.
#[derive(Debug, Default)]
pub struct SnapshotStore {
    current: Option<TsdbRaftSnapshot>,
}

impl SnapshotStore {
    /// Create an empty snapshot store.
    pub fn new() -> Self {
        SnapshotStore { current: None }
    }

    /// Store a snapshot, replacing the previous one if the new snapshot
    /// supersedes it.  Returns `true` if the snapshot was accepted.
    pub fn install(&mut self, snapshot: TsdbRaftSnapshot) -> bool {
        match &self.current {
            None => {
                self.current = Some(snapshot);
                true
            }
            Some(existing) if snapshot.supersedes(existing) => {
                self.current = Some(snapshot);
                true
            }
            _ => false,
        }
    }

    /// Return a reference to the most recently installed snapshot, if any.
    pub fn latest(&self) -> Option<&TsdbRaftSnapshot> {
        self.current.as_ref()
    }

    /// Return the `last_included_index` of the current snapshot, or `0` if no
    /// snapshot has been installed yet.
    pub fn last_included_index(&self) -> u64 {
        self.current
            .as_ref()
            .map(|s| s.last_included_index)
            .unwrap_or(0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<SnapshotDataPoint> {
        vec![
            SnapshotDataPoint {
                series_id: "cpu.usage".into(),
                timestamp: 1_714_300_000_000,
                value: 42.7,
            },
            SnapshotDataPoint {
                series_id: "mem.free".into(),
                timestamp: 1_714_300_060_000,
                value: 1024.0,
            },
        ]
    }

    // ── round-trip ────────────────────────────────────────────────────────

    #[test]
    fn snapshot_round_trips_data_points() {
        let points = sample_points();
        let snap = TsdbRaftSnapshot::from_data_points(5, 2, &points).expect("create");
        let restored = snap.to_data_points().expect("restore");
        assert_eq!(restored, points);
    }

    #[test]
    fn snapshot_round_trips_wire_bytes() {
        let points = sample_points();
        let snap = TsdbRaftSnapshot::from_data_points(3, 1, &points).expect("create");
        let wire = snap.to_wire_bytes().expect("to wire");
        let decoded = TsdbRaftSnapshot::from_wire_bytes(&wire).expect("from wire");
        assert_eq!(decoded.last_included_index, 3);
        assert_eq!(decoded.last_included_term, 1);
        let decoded_points = decoded.to_data_points().expect("decode points");
        assert_eq!(decoded_points, points);
    }

    // ── metadata ──────────────────────────────────────────────────────────

    #[test]
    fn snapshot_zero_index_is_invalid() {
        let result = TsdbRaftSnapshot::from_data_points(0, 1, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_data_size_non_empty() {
        let snap = TsdbRaftSnapshot::from_data_points(1, 1, &sample_points()).expect("create");
        assert!(snap.data_size_bytes() > 0);
    }

    #[test]
    fn empty_snapshot_is_valid() {
        let snap = TsdbRaftSnapshot::from_data_points(1, 1, &[]).expect("create");
        let points = snap.to_data_points().expect("restore");
        assert!(points.is_empty());
    }

    // ── supersedes ────────────────────────────────────────────────────────

    #[test]
    fn supersedes_higher_index() {
        let old = TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap();
        let new = TsdbRaftSnapshot::from_data_points(2, 1, &[]).unwrap();
        assert!(new.supersedes(&old));
        assert!(!old.supersedes(&new));
    }

    #[test]
    fn supersedes_same_index_higher_term() {
        let old = TsdbRaftSnapshot::from_data_points(3, 1, &[]).unwrap();
        let new = TsdbRaftSnapshot::from_data_points(3, 2, &[]).unwrap();
        assert!(new.supersedes(&old));
        assert!(!old.supersedes(&new));
    }

    #[test]
    fn same_index_same_term_not_superseded() {
        let a = TsdbRaftSnapshot::from_data_points(3, 1, &[]).unwrap();
        let b = TsdbRaftSnapshot::from_data_points(3, 1, &[]).unwrap();
        assert!(!a.supersedes(&b));
        assert!(!b.supersedes(&a));
    }

    // ── SnapshotStore ─────────────────────────────────────────────────────

    #[test]
    fn snapshot_store_installs_first_snapshot() {
        let mut store = SnapshotStore::new();
        let snap = TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap();
        assert!(store.install(snap));
        assert_eq!(store.last_included_index(), 1);
    }

    #[test]
    fn snapshot_store_installs_superseding_snapshot() {
        let mut store = SnapshotStore::new();
        store.install(TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap());
        let accepted = store.install(TsdbRaftSnapshot::from_data_points(2, 1, &[]).unwrap());
        assert!(accepted);
        assert_eq!(store.last_included_index(), 2);
    }

    #[test]
    fn snapshot_store_rejects_older_snapshot() {
        let mut store = SnapshotStore::new();
        store.install(TsdbRaftSnapshot::from_data_points(5, 2, &[]).unwrap());
        let accepted = store.install(TsdbRaftSnapshot::from_data_points(3, 1, &[]).unwrap());
        assert!(!accepted);
        assert_eq!(store.last_included_index(), 5);
    }

    #[test]
    fn snapshot_store_latest_returns_none_initially() {
        let store = SnapshotStore::new();
        assert!(store.latest().is_none());
    }

    #[test]
    fn snapshot_store_last_included_index_zero_initially() {
        let store = SnapshotStore::new();
        assert_eq!(store.last_included_index(), 0);
    }
}
