//! Point-in-time snapshot restore for vector indexes.
//!
//! Enables recovery of a vector store to its exact state at any historical
//! timestamp by:
//! 1. Locating the latest WAL checkpoint whose timestamp precedes the target.
//! 2. Replaying WAL entries from that checkpoint up to (and including) the
//!    target timestamp.
//!
//! WAL timestamps are Unix-epoch seconds (`u64`) matching the format used by
//! [`crate::wal::WalEntry`].

use crate::wal::{WalConfig, WalEntry, WalManager};
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointRef
// ─────────────────────────────────────────────────────────────────────────────

/// Reference to a WAL checkpoint discovered during point-in-time search.
#[derive(Debug, Clone)]
pub struct CheckpointRef {
    /// Sequence number of the checkpoint marker in the WAL.
    pub sequence_number: u64,
    /// Unix-epoch timestamp of the checkpoint (seconds).
    pub timestamp: u64,
}

impl Default for CheckpointRef {
    /// An empty base — use the very beginning of the WAL (sequence 0, epoch 0).
    fn default() -> Self {
        Self {
            sequence_number: 0,
            timestamp: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PointInTimeRestore
// ─────────────────────────────────────────────────────────────────────────────

/// Driver for point-in-time recovery.
///
/// Usage:
/// ```ignore
/// let pit = PointInTimeRestore::new(target_ts, wal_dir);
/// let base = pit.find_base_checkpoint()?;
/// let entries = pit.replay_wal_to_timestamp(base.as_ref())?;
/// // Apply entries to the index...
/// ```
pub struct PointInTimeRestore {
    /// Target Unix-epoch timestamp in seconds.
    pub target_timestamp: u64,
    /// Directory that contains the WAL files.
    pub wal_dir: PathBuf,
}

impl PointInTimeRestore {
    /// Create a new driver.
    ///
    /// `target_timestamp_secs` is seconds since Unix epoch; callers that have a
    /// `std::time::SystemTime` should convert with
    /// `system_time.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()`.
    pub fn new(target_timestamp_secs: u64, wal_dir: PathBuf) -> Self {
        Self {
            target_timestamp: target_timestamp_secs,
            wal_dir,
        }
    }

    /// Find the latest [`CheckpointRef`] whose timestamp is ≤ `target_timestamp`.
    ///
    /// Returns `None` when no checkpoint precedes the target (the caller should
    /// treat that as "start from an empty base").
    pub fn find_base_checkpoint(&self) -> Result<Option<CheckpointRef>> {
        let entries = self.read_all_wal_entries()?;

        let mut best: Option<CheckpointRef> = None;

        for entry in &entries {
            if let WalEntry::Checkpoint {
                sequence_number,
                timestamp,
            } = entry
            {
                if *timestamp <= self.target_timestamp {
                    let candidate = CheckpointRef {
                        sequence_number: *sequence_number,
                        timestamp: *timestamp,
                    };
                    match &best {
                        None => best = Some(candidate),
                        Some(prev) if candidate.timestamp > prev.timestamp => {
                            best = Some(candidate)
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(best)
    }

    /// Replay WAL entries that fall between `base.sequence_number` (exclusive)
    /// and `target_timestamp` (inclusive).
    ///
    /// When `base` is `None` (no prior checkpoint) all entries up to
    /// `target_timestamp` are returned.
    pub fn replay_wal_to_timestamp(&self, base: Option<&CheckpointRef>) -> Result<Vec<WalEntry>> {
        let base_seq = base.map(|b| b.sequence_number).unwrap_or(0);
        let all_entries = self.read_all_indexed_wal_entries()?;

        let mut result = Vec::new();
        for (seq, entry) in all_entries {
            // Skip everything at or before the base checkpoint sequence number
            if seq <= base_seq && base.is_some() {
                continue;
            }
            // Skip entries beyond the target timestamp
            if entry.timestamp() > self.target_timestamp {
                continue;
            }
            // Skip structural markers — they have no data to replay
            if entry.is_checkpoint() {
                continue;
            }
            match &entry {
                WalEntry::BeginTransaction { .. }
                | WalEntry::CommitTransaction { .. }
                | WalEntry::AbortTransaction { .. } => {
                    // Skip transaction bookkeeping; only data entries matter
                }
                _ => result.push(entry),
            }
        }

        Ok(result)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Read all WAL entries in chronological file order.
    fn read_all_wal_entries(&self) -> Result<Vec<WalEntry>> {
        Ok(self
            .read_all_indexed_wal_entries()?
            .into_iter()
            .map(|(_, e)| e)
            .collect())
    }

    /// Read all WAL entries together with their sequence numbers, ordered by
    /// (file name, sequence number).  WAL files are named `wal-<hex_ts>.log`
    /// so lexicographic order equals chronological order.
    fn read_all_indexed_wal_entries(&self) -> Result<Vec<(u64, WalEntry)>> {
        if !self.wal_dir.exists() {
            return Ok(Vec::new());
        }

        // Open a temporary WalManager in read-only mode (just to run recover())
        let config = WalConfig {
            wal_directory: self.wal_dir.clone(),
            // Very large interval so we never auto-trigger a checkpoint during
            // our recovery scan.
            checkpoint_interval: u64::MAX,
            checkpoint_retention: usize::MAX,
            sync_on_write: false,
            ..WalConfig::default()
        };
        let mgr = WalManager::new(config)
            .map_err(|e| anyhow!("Cannot open WAL for PIT recovery: {}", e))?;

        // recover() returns entries *after* the last checkpoint — we need all
        // entries, so we scan the files directly via the WalManager's recover
        // (it skips nothing when checkpoint_retention is huge and the interval
        // is MAX, because no new checkpoint will be written).
        //
        // However recover() still filters on last_checkpoint_seq.  To get
        // *everything*, we need to parse files ourselves.  We borrow the
        // helper below.
        let entries = self.scan_wal_files(&self.wal_dir)?;
        drop(mgr);
        Ok(entries)
    }

    /// Low-level file scanner: returns (sequence_number, WalEntry) for every
    /// parseable record across all `wal-*.log` files in `dir`.
    fn scan_wal_files(&self, dir: &Path) -> Result<Vec<(u64, WalEntry)>> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        const WAL_MAGIC: &[u8; 4] = b"WALV";

        let mut wal_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.starts_with("wal-") && s.ends_with(".log"))
                    .unwrap_or(false)
            })
            .collect();

        // Lexicographic == chronological for hex-timestamp file names
        wal_files.sort_by_key(|e| e.file_name());

        let mut result: Vec<(u64, WalEntry)> = Vec::new();

        for file_entry in wal_files {
            let path = file_entry.path();
            let file = match File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("PIT: cannot open WAL file {:?}: {}", path, e);
                    continue;
                }
            };
            let mut reader = BufReader::new(file);

            // Validate magic
            let mut magic = [0u8; 4];
            if reader.read_exact(&mut magic).is_err() {
                continue;
            }
            if &magic != WAL_MAGIC {
                tracing::warn!("PIT: invalid magic in {:?}", path);
                continue;
            }

            // Version (4 bytes) + file timestamp (8 bytes) — skip both
            let mut skip = [0u8; 12];
            if reader.read_exact(&mut skip).is_err() {
                continue;
            }

            loop {
                // Sequence number
                let mut seq_bytes = [0u8; 8];
                match reader.read_exact(&mut seq_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        tracing::warn!("PIT: read error in {:?}: {}", path, e);
                        break;
                    }
                }
                let seq = u64::from_le_bytes(seq_bytes);

                // Entry length
                let mut len_bytes = [0u8; 4];
                match reader.read_exact(&mut len_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        tracing::warn!("PIT: read error in {:?}: {}", path, e);
                        break;
                    }
                }
                let len = u32::from_le_bytes(len_bytes) as usize;

                // Sanity guard
                if len > 100_000_000 {
                    tracing::warn!(
                        "PIT: suspicious entry length {} at seq {} in {:?}",
                        len,
                        seq,
                        path
                    );
                    break;
                }

                // Entry bytes
                let mut entry_bytes = vec![0u8; len];
                match reader.read_exact(&mut entry_bytes) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        tracing::warn!("PIT: read error in {:?}: {}", path, e);
                        break;
                    }
                }

                match oxicode::serde::decode_from_slice::<WalEntry, _>(
                    &entry_bytes,
                    oxicode::config::standard(),
                ) {
                    Ok((entry, _)) => result.push((seq, entry)),
                    Err(e) => {
                        tracing::warn!("PIT: cannot deserialise entry at seq {}: {}", seq, e);
                    }
                }
            }
        }

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{WalConfig, WalEntry, WalManager};
    use tempfile::TempDir;

    fn write_test_wal(dir: &std::path::Path, entries: &[WalEntry]) -> Result<()> {
        let config = WalConfig {
            wal_directory: dir.to_path_buf(),
            checkpoint_interval: u64::MAX,
            sync_on_write: true,
            ..WalConfig::default()
        };
        let mgr = WalManager::new(config)?;
        for entry in entries {
            mgr.append(entry.clone())?;
        }
        mgr.flush()?;
        Ok(())
    }

    #[test]
    fn test_find_base_checkpoint_no_checkpoint() -> Result<()> {
        let tmp = TempDir::new()?;

        let entries = vec![
            WalEntry::Insert {
                id: "v1".into(),
                vector: vec![1.0],
                metadata: None,
                timestamp: 1000,
            },
            WalEntry::Insert {
                id: "v2".into(),
                vector: vec![2.0],
                metadata: None,
                timestamp: 2000,
            },
        ];
        write_test_wal(tmp.path(), &entries)?;

        let pit = PointInTimeRestore::new(5000, tmp.path().to_path_buf());
        let base = pit.find_base_checkpoint()?;
        assert!(base.is_none(), "expected None when no checkpoint exists");
        Ok(())
    }

    #[test]
    fn test_find_base_checkpoint_selects_latest_before_target() -> Result<()> {
        let tmp = TempDir::new()?;

        let entries = vec![
            WalEntry::Checkpoint {
                sequence_number: 0,
                timestamp: 1000,
            },
            WalEntry::Checkpoint {
                sequence_number: 1,
                timestamp: 3000,
            },
            WalEntry::Checkpoint {
                sequence_number: 2,
                timestamp: 6000,
            },
        ];
        write_test_wal(tmp.path(), &entries)?;

        let pit = PointInTimeRestore::new(4000, tmp.path().to_path_buf());
        let base = pit.find_base_checkpoint()?;
        let base = base.expect("should find a checkpoint");
        assert_eq!(base.timestamp, 3000, "should pick checkpoint at ts=3000");
        Ok(())
    }

    #[test]
    fn test_replay_wal_to_timestamp_filters_correctly() -> Result<()> {
        let tmp = TempDir::new()?;

        let raw = vec![
            WalEntry::Insert {
                id: "v1".into(),
                vector: vec![1.0],
                metadata: None,
                timestamp: 1000,
            },
            WalEntry::Insert {
                id: "v2".into(),
                vector: vec![2.0],
                metadata: None,
                timestamp: 2000,
            },
            WalEntry::Insert {
                id: "v3".into(),
                vector: vec![3.0],
                metadata: None,
                timestamp: 4000,
            },
        ];
        write_test_wal(tmp.path(), &raw)?;

        // Target = 2500 → should replay v1 and v2 but NOT v3
        let pit = PointInTimeRestore::new(2500, tmp.path().to_path_buf());
        let replayed = pit.replay_wal_to_timestamp(None)?;
        assert_eq!(replayed.len(), 2);
        let ids: Vec<_> = replayed
            .iter()
            .filter_map(|e| {
                if let WalEntry::Insert { id, .. } = e {
                    Some(id.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(ids.contains(&"v1"));
        assert!(ids.contains(&"v2"));
        assert!(!ids.contains(&"v3"));

        Ok(())
    }

    #[test]
    fn test_checkpoint_discovery_ordered() -> Result<()> {
        let tmp = TempDir::new()?;

        let timestamps = [500u64, 1500, 2500, 3500, 4500];
        let entries: Vec<WalEntry> = timestamps
            .iter()
            .enumerate()
            .map(|(i, &ts)| WalEntry::Checkpoint {
                sequence_number: i as u64,
                timestamp: ts,
            })
            .collect();
        write_test_wal(tmp.path(), &entries)?;

        // Target = 3000 → best checkpoint is ts=2500 (seq=2)
        let pit = PointInTimeRestore::new(3000, tmp.path().to_path_buf());
        let base = pit.find_base_checkpoint()?.expect("checkpoint expected");
        assert_eq!(base.timestamp, 2500);
        assert_eq!(base.sequence_number, 2);

        Ok(())
    }
}
