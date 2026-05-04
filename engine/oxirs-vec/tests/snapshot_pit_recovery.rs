//! Integration tests for point-in-time snapshot/restore.

use oxirs_vec::{
    persistence::{restore::restore_to_timestamp, CheckpointRef, PointInTimeRestore},
    vector_store::VectorStore,
    wal::{WalConfig, WalEntry, WalManager},
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

fn write_entries(dir: &std::path::Path, entries: &[WalEntry]) -> anyhow::Result<()> {
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Basic: insert two vectors, restore to a timestamp after both — both replayed.
#[test]
fn snapshot_pit_restore_basic() -> anyhow::Result<()> {
    let tmp = TempDir::new()?;

    let entries = vec![
        WalEntry::Insert {
            id: "http://ex.org/t1".into(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: None,
            timestamp: 1000,
        },
        WalEntry::Insert {
            id: "http://ex.org/t2".into(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: None,
            timestamp: 2000,
        },
    ];
    write_entries(tmp.path(), &entries)?;

    let mut store = VectorStore::new();
    let report = restore_to_timestamp(&mut store, 3000, tmp.path())?;

    assert_eq!(report.entries_replayed, 2);
    assert!(report.base_checkpoint.is_none());
    assert_eq!(report.target_timestamp_secs, 3000);
    Ok(())
}

/// Restore to timestamp before second insert — only first vector replayed.
#[test]
fn snapshot_pit_restore_cuts_off_at_target() -> anyhow::Result<()> {
    let tmp = TempDir::new()?;

    let entries = vec![
        WalEntry::Insert {
            id: "http://ex.org/early".into(),
            vector: vec![1.0],
            metadata: None,
            timestamp: 500,
        },
        WalEntry::Insert {
            id: "http://ex.org/late".into(),
            vector: vec![2.0],
            metadata: None,
            timestamp: 8000,
        },
    ];
    write_entries(tmp.path(), &entries)?;

    let mut store = VectorStore::new();
    // Target = 4000 → only ts=500 entry applies
    let report = restore_to_timestamp(&mut store, 4000, tmp.path())?;

    assert_eq!(report.entries_replayed, 1);
    Ok(())
}

/// When no checkpoint precedes the target, `base_checkpoint` is `None`.
#[test]
fn snapshot_pit_no_earlier_checkpoint_uses_empty_base() -> anyhow::Result<()> {
    let tmp = TempDir::new()?;

    let entries = vec![WalEntry::Insert {
        id: "http://ex.org/only".into(),
        vector: vec![0.5],
        metadata: None,
        timestamp: 100,
    }];
    write_entries(tmp.path(), &entries)?;

    let mut store = VectorStore::new();
    // Very early target → no checkpoint, base is None
    let report = restore_to_timestamp(&mut store, 50, tmp.path())?;
    assert_eq!(report.entries_replayed, 0);
    assert!(report.base_checkpoint.is_none());
    Ok(())
}

/// Create N checkpoints; verify `find_base_checkpoint` returns the correct one.
#[test]
fn snapshot_checkpoint_discovery_ordered() -> anyhow::Result<()> {
    let tmp = TempDir::new()?;

    let ckpts: Vec<WalEntry> = [1000u64, 2000, 3000, 4000, 5000]
        .iter()
        .enumerate()
        .map(|(i, &ts)| WalEntry::Checkpoint {
            sequence_number: i as u64,
            timestamp: ts,
        })
        .collect();
    write_entries(tmp.path(), &ckpts)?;

    // Target = 3500 → best checkpoint is ts=3000 (seq=2)
    let pit = PointInTimeRestore::new(3500, tmp.path().to_path_buf());
    let base = pit.find_base_checkpoint()?;
    let base = base.expect("should find a checkpoint");
    assert_eq!(base.timestamp, 3000);
    assert_eq!(base.sequence_number, 2);
    Ok(())
}

/// Checkpoint at exactly the target timestamp is included.
#[test]
fn snapshot_pit_checkpoint_at_exact_target() -> anyhow::Result<()> {
    let tmp = TempDir::new()?;

    let entries = vec![
        WalEntry::Checkpoint {
            sequence_number: 0,
            timestamp: 5000,
        },
        WalEntry::Insert {
            id: "http://ex.org/after".into(),
            vector: vec![9.0],
            metadata: None,
            timestamp: 6000,
        },
    ];
    write_entries(tmp.path(), &entries)?;

    let pit = PointInTimeRestore::new(5000, tmp.path().to_path_buf());
    let base = pit.find_base_checkpoint()?;
    // The checkpoint at ts=5000 must be selected (≤ target)
    let base = base.expect("checkpoint at ts=5000 should be selected");
    assert_eq!(base.timestamp, 5000);
    Ok(())
}

/// CheckpointRef::default gives zeroed fields.
#[test]
fn checkpoint_ref_default_is_zeroed() {
    let c = CheckpointRef::default();
    assert_eq!(c.sequence_number, 0);
    assert_eq!(c.timestamp, 0);
}
