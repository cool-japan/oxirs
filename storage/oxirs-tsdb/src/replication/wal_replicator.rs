//! WAL → Raft bridge for TSDB high-availability replication.
//!
//! [`WalReplicator`] converts a TSDB WAL operation into a serialised byte
//! stream and ships it to a Raft peer through an in-process channel.  The
//! channel endpoint is owned by the node that manages the Raft log; the
//! replicator side only needs a `Sender<Vec<u8>>` handle.
//!
//! ## Data model
//!
//! [`TsdbRaftOp`] is the application-level command that can appear inside a
//! Raft log entry.  It covers the three mutating operations that the TSDB
//! write path can produce:
//!
//! - `Insert` — write a single (series, timestamp, value, tags) tuple
//! - `Delete` — remove all data in a half-open timestamp range from a series
//! - `Truncate` — drop all data for a series
//!
//! The existing [`super::raft_state::TsdbCommand`] covers the same ground at
//! the Raft state-machine level; `TsdbRaftOp` is the WAL-facing counterpart
//! that carries richer metadata (tag map, timestamp range).
//!
//! ## Wire format
//!
//! Operations are serialised to JSON via [`serde_json`].  JSON was chosen over
//! binary formats because:
//!
//! - It is human-readable and easy to inspect during debugging.
//! - It avoids the `bincode` / `oxicode` choice for now (pure-text transport).
//! - The extra bytes are negligible compared to the Raft framing overhead.
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_tsdb::replication::wal_replicator::{TsdbRaftOp, WalReplicator};
//! use std::sync::mpsc;
//!
//! let (tx, rx) = mpsc::channel::<Vec<u8>>();
//! let replicator = WalReplicator::new(tx);
//!
//! let op = TsdbRaftOp::Insert {
//!     series_id: "cpu.usage".into(),
//!     timestamp: 1_714_300_000_000,
//!     value: 42.7,
//!     tags: Default::default(),
//! };
//! replicator.replicate(&op).expect("replication should succeed");
//!
//! let bytes = rx.recv().expect("should receive bytes");
//! let decoded: TsdbRaftOp = serde_json::from_slice(&bytes).expect("must decode");
//! assert!(matches!(decoded, TsdbRaftOp::Insert { .. }));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc;

// ────────────────────────────────────────────────────────────────────────────
// TsdbRaftOp — WAL-level operation carried inside a Raft log entry
// ────────────────────────────────────────────────────────────────────────────

/// An application-level mutation that the TSDB WAL can produce and that Raft
/// must replicate to every node in the cluster before it is considered durable.
///
/// Each variant corresponds to one class of write in the TSDB write path:
///
/// | Variant | WAL equivalent |
/// |---------|----------------|
/// | `Insert` | `WalEntry::Write` |
/// | `Delete` | `WalEntry::Delete` |
/// | `Truncate` | `WalEntry::Truncate` |
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TsdbRaftOp {
    /// Write a single data point to the named series.
    Insert {
        /// Series / metric identifier (UUID-style or human-readable string).
        series_id: String,
        /// Observation timestamp in milliseconds since Unix epoch.
        timestamp: i64,
        /// Measured floating-point value.
        value: f64,
        /// Arbitrary key-value tag pairs (e.g. `{"host": "srv-1"}`).
        tags: HashMap<String, String>,
    },
    /// Delete all data in the half-open range `[from_ts, to_ts)` for the
    /// named series.
    Delete {
        /// Series / metric identifier.
        series_id: String,
        /// Start of deletion window (inclusive, milliseconds since epoch).
        from_ts: i64,
        /// End of deletion window (exclusive, milliseconds since epoch).
        to_ts: i64,
    },
    /// Drop every data point associated with the named series, equivalent to
    /// a full series reset.
    Truncate {
        /// Series / metric identifier.
        series_id: String,
    },
}

impl TsdbRaftOp {
    /// Return the series ID referenced by any variant of this op.
    pub fn series_id(&self) -> &str {
        match self {
            TsdbRaftOp::Insert { series_id, .. }
            | TsdbRaftOp::Delete { series_id, .. }
            | TsdbRaftOp::Truncate { series_id } => series_id,
        }
    }

    /// Serialise this op to JSON bytes for transport through the Raft channel.
    ///
    /// Returns an error only if serialisation fails (in practice this never
    /// happens for well-formed ops, but the signature is fallible to match the
    /// `serde_json` API).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplicationError> {
        serde_json::to_vec(self).map_err(|e| ReplicationError::Serialization(e.to_string()))
    }

    /// Deserialise a `TsdbRaftOp` from JSON bytes produced by [`TsdbRaftOp::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplicationError> {
        serde_json::from_slice(bytes).map_err(|e| ReplicationError::Serialization(e.to_string()))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ReplicationError
// ────────────────────────────────────────────────────────────────────────────

/// Errors that the [`WalReplicator`] can produce.
#[derive(Debug, thiserror::Error)]
pub enum ReplicationError {
    /// Serialisation of a [`TsdbRaftOp`] to JSON failed.
    #[error("Replication serialisation error: {0}")]
    Serialization(String),
    /// The receiving end of the Raft channel has been dropped; the node is
    /// gone or the system is shutting down.
    #[error("Replication channel closed — receiving Raft node has dropped")]
    ChannelClosed,
}

// ────────────────────────────────────────────────────────────────────────────
// WalReplicator
// ────────────────────────────────────────────────────────────────────────────

/// Ships serialised TSDB WAL operations to a Raft node through an in-process
/// channel.
///
/// The `WalReplicator` is the *sending* side of the WAL→Raft bridge.  It is
/// cheap to clone (it wraps a `mpsc::Sender`) and is designed to be held by
/// the TSDB write path.
///
/// In a real deployment the receiving end would be a Raft node running in a
/// background tokio task; here, for testing, it can simply be an `mpsc::Receiver`.
///
/// # Example
///
/// See [module-level documentation](self) for a full usage example.
#[derive(Clone, Debug)]
pub struct WalReplicator {
    /// Channel sender that delivers serialised ops to the Raft node.
    tx: mpsc::Sender<Vec<u8>>,
}

impl WalReplicator {
    /// Create a new `WalReplicator` that forwards ops through `tx`.
    pub fn new(tx: mpsc::Sender<Vec<u8>>) -> Self {
        WalReplicator { tx }
    }

    /// Serialise `op` to JSON and send it through the Raft channel.
    ///
    /// Returns `Ok(())` on success, or:
    ///
    /// - [`ReplicationError::Serialization`] if JSON serialisation fails
    ///   (should not happen in practice).
    /// - [`ReplicationError::ChannelClosed`] if the Raft node has dropped the
    ///   receiving end of the channel (node crashed or system shutting down).
    pub fn replicate(&self, op: &TsdbRaftOp) -> Result<(), ReplicationError> {
        let bytes = op.to_bytes()?;
        self.tx
            .send(bytes)
            .map_err(|_| ReplicationError::ChannelClosed)
    }

    /// Replicate a batch of operations atomically from the channel's
    /// perspective.  Each op is serialised and sent individually; all ops in
    /// the batch must succeed or the first error is returned.
    pub fn replicate_batch(&self, ops: &[TsdbRaftOp]) -> Result<(), ReplicationError> {
        for op in ops {
            self.replicate(op)?;
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TsdbStateMachine — applies replicated ops to local TSDB storage
// ────────────────────────────────────────────────────────────────────────────

/// Raft state machine that applies [`TsdbRaftOp`] commands to an in-memory
/// TSDB log.
///
/// In a production integration the `apply` method would forward each op to the
/// actual columnar / chunk storage layer.  Here we maintain a simple in-memory
/// event log that tests can inspect.
///
/// ## Design note
///
/// The struct is intentionally storage-agnostic: it does **not** hold a
/// reference to `ColumnarStore` or `WriteBuffer` because those types have
/// complex lifetimes and are not `Send`.  Production callers should sub-class
/// or wrap this type to bridge into the real storage layer.
pub struct TsdbStateMachine {
    /// 0-based index of the last log entry applied to this state machine.
    pub last_applied_index: u64,
    /// Ordered record of every operation applied, for testing and auditing.
    pub applied_ops: Vec<TsdbRaftOp>,
}

impl TsdbStateMachine {
    /// Create an empty state machine with `last_applied_index = 0`.
    pub fn new() -> Self {
        TsdbStateMachine {
            last_applied_index: 0,
            applied_ops: Vec::new(),
        }
    }

    /// Apply a single [`TsdbRaftOp`] from the Raft log at the given `index`.
    ///
    /// The index must be exactly `last_applied_index + 1`; gaps are rejected
    /// to preserve linearisability.
    pub fn apply(&mut self, index: u64, op: TsdbRaftOp) -> Result<(), ApplyError> {
        if index != self.last_applied_index + 1 {
            return Err(ApplyError::IndexGap {
                expected: self.last_applied_index + 1,
                got: index,
            });
        }
        self.applied_ops.push(op);
        self.last_applied_index = index;
        Ok(())
    }

    /// Return the number of ops that have been applied.
    pub fn applied_count(&self) -> usize {
        self.applied_ops.len()
    }
}

impl Default for TsdbStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Error returned when `apply` encounters a consistency violation.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    /// The caller tried to apply index N but the state machine expected a
    /// different index (gap or duplicate).
    #[error("Log index gap: expected {expected}, got {got}")]
    IndexGap {
        /// The next index the state machine expected.
        expected: u64,
        /// The index the caller supplied.
        got: u64,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_insert_op() -> TsdbRaftOp {
        TsdbRaftOp::Insert {
            series_id: "temp.sensor1".into(),
            timestamp: 1_714_300_000_000,
            value: 22.5,
            tags: [("location".to_string(), "room1".to_string())].into(),
        }
    }

    // ── TsdbRaftOp round-trip ─────────────────────────────────────────────

    #[test]
    fn insert_op_round_trips_json() {
        let op = make_insert_op();
        let bytes = op.to_bytes().expect("serialise");
        let decoded = TsdbRaftOp::from_bytes(&bytes).expect("deserialise");
        assert_eq!(op, decoded);
    }

    #[test]
    fn delete_op_round_trips_json() {
        let op = TsdbRaftOp::Delete {
            series_id: "s1".into(),
            from_ts: 1000,
            to_ts: 2000,
        };
        let bytes = op.to_bytes().expect("serialise");
        let decoded = TsdbRaftOp::from_bytes(&bytes).expect("deserialise");
        assert_eq!(op, decoded);
    }

    #[test]
    fn truncate_op_round_trips_json() {
        let op = TsdbRaftOp::Truncate {
            series_id: "s2".into(),
        };
        let bytes = op.to_bytes().expect("serialise");
        let decoded = TsdbRaftOp::from_bytes(&bytes).expect("deserialise");
        assert_eq!(op, decoded);
    }

    #[test]
    fn series_id_accessor_works_for_all_variants() {
        assert_eq!(make_insert_op().series_id(), "temp.sensor1");
        assert_eq!(
            TsdbRaftOp::Delete {
                series_id: "x".into(),
                from_ts: 0,
                to_ts: 1
            }
            .series_id(),
            "x"
        );
        assert_eq!(
            TsdbRaftOp::Truncate {
                series_id: "y".into()
            }
            .series_id(),
            "y"
        );
    }

    // ── WalReplicator ─────────────────────────────────────────────────────

    #[test]
    fn replicator_sends_serialised_bytes() {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let replicator = WalReplicator::new(tx);
        replicator.replicate(&make_insert_op()).expect("replicate");
        let bytes = rx.recv().expect("receive");
        let decoded = TsdbRaftOp::from_bytes(&bytes).expect("decode");
        assert!(matches!(decoded, TsdbRaftOp::Insert { .. }));
    }

    #[test]
    fn replicator_channel_closed_error() {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let replicator = WalReplicator::new(tx);
        drop(rx);
        let result = replicator.replicate(&TsdbRaftOp::Truncate {
            series_id: "s".into(),
        });
        assert!(matches!(result, Err(ReplicationError::ChannelClosed)));
    }

    #[test]
    fn replicator_batch_sends_all_ops() {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let replicator = WalReplicator::new(tx);
        let ops = vec![
            TsdbRaftOp::Insert {
                series_id: "a".into(),
                timestamp: 1,
                value: 1.0,
                tags: Default::default(),
            },
            TsdbRaftOp::Delete {
                series_id: "b".into(),
                from_ts: 0,
                to_ts: 100,
            },
            TsdbRaftOp::Truncate {
                series_id: "c".into(),
            },
        ];
        replicator.replicate_batch(&ops).expect("batch");
        for _ in &ops {
            rx.recv().expect("received");
        }
    }

    // ── TsdbStateMachine ──────────────────────────────────────────────────

    #[test]
    fn state_machine_applies_in_order() {
        let mut sm = TsdbStateMachine::new();
        sm.apply(1, make_insert_op()).expect("apply 1");
        sm.apply(
            2,
            TsdbRaftOp::Delete {
                series_id: "x".into(),
                from_ts: 0,
                to_ts: 99,
            },
        )
        .expect("apply 2");
        assert_eq!(sm.applied_count(), 2);
        assert_eq!(sm.last_applied_index, 2);
    }

    #[test]
    fn state_machine_rejects_gap() {
        let mut sm = TsdbStateMachine::new();
        sm.apply(1, make_insert_op()).expect("apply 1");
        let result = sm.apply(3, make_insert_op()); // skipped index 2
        assert!(result.is_err());
    }

    #[test]
    fn state_machine_rejects_duplicate_index() {
        let mut sm = TsdbStateMachine::new();
        sm.apply(1, make_insert_op()).expect("apply 1");
        let result = sm.apply(1, make_insert_op()); // duplicate
        assert!(result.is_err());
    }
}
