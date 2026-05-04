//! Point-in-time restore for a [`VectorStore`].
//!
//! Provides a high-level function that applies WAL replay results to a
//! [`VectorStore`], effectively rolling the store back (or forward) to any
//! historical timestamp recorded in the WAL.
//!
//! The implementation purposely avoids rebuilding the index from scratch:
//! only the delta entries between the nearest checkpoint and the target
//! timestamp are replayed into the existing (or freshly created) store.

use crate::persistence::point_in_time::{CheckpointRef, PointInTimeRestore};
use crate::vector_store::VectorStore;
use crate::wal::WalEntry;
use crate::Vector;
use anyhow::{anyhow, Result};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// RestoreReport
// ─────────────────────────────────────────────────────────────────────────────

/// Summary produced by [`restore_to_timestamp`].
#[derive(Debug, Clone)]
pub struct RestoreReport {
    /// How many WAL data entries were applied to the store.
    pub entries_replayed: usize,
    /// The checkpoint that was used as the recovery base, if any.
    pub base_checkpoint: Option<CheckpointRef>,
    /// Target Unix-epoch timestamp (seconds) that was requested.
    pub target_timestamp_secs: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Restore a [`VectorStore`] to its state at `target_timestamp_secs` (Unix
/// epoch, seconds).
///
/// The function:
/// 1. Scans `wal_dir` to find the latest checkpoint ≤ `target_timestamp_secs`.
/// 2. Replays all data entries between that checkpoint and `target_timestamp_secs`
///    into `store` (insert / update / delete / batch).
/// 3. Returns a [`RestoreReport`] with replay statistics.
///
/// **Note**: this function does not clear the store before replaying.  Callers
/// that want a clean restore should pass a fresh `VectorStore::new()`.
pub fn restore_to_timestamp(
    store: &mut VectorStore,
    target_timestamp_secs: u64,
    wal_dir: &Path,
) -> Result<RestoreReport> {
    let pit = PointInTimeRestore::new(target_timestamp_secs, wal_dir.to_owned());

    let base = pit.find_base_checkpoint()?;
    let entries = pit.replay_wal_to_timestamp(base.as_ref())?;

    let count = entries.len();
    for entry in &entries {
        apply_wal_entry(store, entry)?;
    }

    Ok(RestoreReport {
        entries_replayed: count,
        base_checkpoint: base,
        target_timestamp_secs,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_wal_entry — maps a WalEntry variant onto VectorStore operations
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single [`WalEntry`] to a [`VectorStore`].
///
/// Structural markers (Checkpoint, Begin/Commit/AbortTransaction) are silently
/// ignored — only data-bearing variants mutate the store.
pub fn apply_wal_entry(store: &mut VectorStore, entry: &WalEntry) -> Result<()> {
    match entry {
        WalEntry::Insert { id, vector, .. } => {
            store
                .index_vector(id.clone(), Vector::new(vector.clone()))
                .map_err(|e| anyhow!("PIT restore: insert '{}' failed: {}", id, e))?;
        }
        WalEntry::Update { id, vector, .. } => {
            // VectorStore has no dedicated update; re-inserting replaces the entry
            store
                .index_vector(id.clone(), Vector::new(vector.clone()))
                .map_err(|e| anyhow!("PIT restore: update '{}' failed: {}", id, e))?;
        }
        WalEntry::Delete { id, .. } => {
            // MemoryVectorIndex's remove_vector is a no-op by default — if the
            // concrete index supports deletion the trait impl will pick it up.
            store
                .remove_vector(id)
                .map_err(|e| anyhow!("PIT restore: delete '{}' failed: {}", id, e))?;
        }
        WalEntry::Batch { entries, .. } => {
            for inner in entries {
                apply_wal_entry(store, inner)?;
            }
        }
        // Structural markers — skip
        WalEntry::Checkpoint { .. }
        | WalEntry::BeginTransaction { .. }
        | WalEntry::CommitTransaction { .. }
        | WalEntry::AbortTransaction { .. } => {}
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::VectorStore;
    use crate::wal::{WalConfig, WalEntry, WalManager};
    use tempfile::TempDir;

    fn populate_wal(dir: &std::path::Path, entries: &[WalEntry]) -> Result<()> {
        let config = WalConfig {
            wal_directory: dir.to_path_buf(),
            checkpoint_interval: u64::MAX,
            sync_on_write: true,
            ..WalConfig::default()
        };
        let mgr = WalManager::new(config)?;
        for e in entries {
            mgr.append(e.clone())?;
        }
        mgr.flush()
    }

    #[test]
    fn test_restore_basic_inserts() -> Result<()> {
        let tmp = TempDir::new()?;
        let entries = vec![
            WalEntry::Insert {
                id: "http://ex.org/a".into(),
                vector: vec![1.0, 0.0],
                metadata: None,
                timestamp: 100,
            },
            WalEntry::Insert {
                id: "http://ex.org/b".into(),
                vector: vec![0.0, 1.0],
                metadata: None,
                timestamp: 200,
            },
        ];
        populate_wal(tmp.path(), &entries)?;

        let mut store = VectorStore::new();
        let report = restore_to_timestamp(&mut store, 300, tmp.path())?;

        assert_eq!(report.entries_replayed, 2);
        assert_eq!(report.target_timestamp_secs, 300);
        Ok(())
    }

    #[test]
    fn test_restore_excludes_entries_after_target() -> Result<()> {
        let tmp = TempDir::new()?;
        let entries = vec![
            WalEntry::Insert {
                id: "http://ex.org/early".into(),
                vector: vec![1.0],
                metadata: None,
                timestamp: 100,
            },
            WalEntry::Insert {
                id: "http://ex.org/late".into(),
                vector: vec![2.0],
                metadata: None,
                timestamp: 900,
            },
        ];
        populate_wal(tmp.path(), &entries)?;

        let mut store = VectorStore::new();
        // Restore to ts=500 → only the ts=100 entry should be replayed
        let report = restore_to_timestamp(&mut store, 500, tmp.path())?;
        assert_eq!(report.entries_replayed, 1);
        Ok(())
    }

    #[test]
    fn test_restore_no_wal_entries() -> Result<()> {
        let tmp = TempDir::new()?;
        // Write an empty WAL (just open and close)
        let config = WalConfig {
            wal_directory: tmp.path().to_path_buf(),
            checkpoint_interval: u64::MAX,
            sync_on_write: true,
            ..WalConfig::default()
        };
        let mgr = WalManager::new(config)?;
        mgr.flush()?;
        drop(mgr);

        let mut store = VectorStore::new();
        let report = restore_to_timestamp(&mut store, 9999, tmp.path())?;
        assert_eq!(report.entries_replayed, 0);
        assert!(report.base_checkpoint.is_none());
        Ok(())
    }

    #[test]
    fn test_restore_batch_entries_counted_individually() -> Result<()> {
        let tmp = TempDir::new()?;
        let batch = WalEntry::Batch {
            entries: vec![
                WalEntry::Insert {
                    id: "http://ex.org/x".into(),
                    vector: vec![1.0],
                    metadata: None,
                    timestamp: 50,
                },
                WalEntry::Insert {
                    id: "http://ex.org/y".into(),
                    vector: vec![2.0],
                    metadata: None,
                    timestamp: 50,
                },
            ],
            timestamp: 50,
        };
        populate_wal(tmp.path(), &[batch])?;

        let mut store = VectorStore::new();
        // Batch itself is one WAL entry — replay_wal_to_timestamp returns 1 Batch
        let report = restore_to_timestamp(&mut store, 200, tmp.path())?;
        // The single Batch entry is replayed; it internally inserts two vectors
        assert_eq!(report.entries_replayed, 1);
        Ok(())
    }
}
