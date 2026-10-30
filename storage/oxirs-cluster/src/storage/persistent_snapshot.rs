//! Snapshot files of PersistentStorage.
//!
//! A snapshot is the application state as JSON, compressed into one LZ4
//! frame when `compress_snapshots` is set. Its data file is named after its
//! checksum (`snapshot-<first 16 hex digits of the SHA-256>.dat`) and is
//! written before `snapshot_metadata.json`, which records that checksum;
//! superseded data files are removed afterwards. Whenever a crash happens,
//! the metadata on disk therefore names a data file that exists and matches
//! it. `snapshot.json`, the data file of releases up to 0.4.1, is still read.

use super::config::StorageConfig;
use super::persistent_files::{unix_secs, write_atomic};
use super::types::{RaftState, SnapshotMetadata};
use crate::raft::{OxirsNodeId, RdfApp};

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Metadata of the current snapshot, as JSON.
pub(super) const METADATA_FILE: &str = "snapshot_metadata.json";
/// Data file written by releases up to 0.4.1.
const LEGACY_DATA_FILE: &str = "snapshot.json";
/// Checksum file that releases up to 0.4.1 kept for [`LEGACY_DATA_FILE`].
const LEGACY_CHECKSUM_FILE: &str = "snapshot.json.checksum";
/// Largest snapshot JSON a compressed data file may expand to.
const MAX_SNAPSHOT_LEN: usize = u32::MAX as usize;

/// Name of the data file of a snapshot with `checksum`.
fn data_file_name(checksum: &str) -> String {
    format!("snapshot-{}.dat", checksum.get(..16).unwrap_or(checksum))
}

/// Whether `name` is a data file written by [`write_snapshot`].
fn is_data_file_name(name: &str) -> bool {
    name.strip_prefix("snapshot-")
        .and_then(|rest| rest.strip_suffix(".dat"))
        .is_some_and(|digest| digest.len() == 16 && digest.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Write a snapshot of `app` taken at the end of `raft`'s log.
pub(super) fn write_snapshot(
    dir: &Path,
    node_id: OxirsNodeId,
    raft: &RaftState,
    app: &RdfApp,
    config: &StorageConfig,
) -> Result<SnapshotMetadata> {
    let json = serde_json::to_vec(app)?;
    let bytes = if config.compress_snapshots {
        match oxiarc_lz4::compress(&json) {
            Ok(compressed) => compressed,
            Err(e) => {
                tracing::warn!("snapshot compression failed, writing it uncompressed: {e}");
                json
            }
        }
    } else {
        json
    };

    let checksum = sha256_hex(&bytes);
    let name = data_file_name(&checksum);
    write_atomic(dir, &name, &bytes, config.sync_writes)?;

    let last_entry = raft.log.last();
    let metadata = SnapshotMetadata {
        last_included_index: last_entry.map_or(0, |e| e.index),
        last_included_term: last_entry.map_or(0, |e| e.term),
        configuration: vec![node_id],
        timestamp: unix_secs(),
        size: u64::try_from(bytes.len())?,
        checksum,
    };
    write_atomic(
        dir,
        METADATA_FILE,
        &serde_json::to_vec(&metadata)?,
        config.sync_writes,
    )?;
    remove_superseded(dir, &name);

    tracing::info!(
        "Created snapshot for node {} at index {} with {} bytes",
        node_id,
        metadata.last_included_index,
        metadata.size
    );
    Ok(metadata)
}

/// Remove data files other than `current`, which the metadata no longer
/// names. Failures only leave stale files behind.
fn remove_superseded(dir: &Path, current: &str) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                "could not list {} to remove old snapshots: {e}",
                dir.display()
            );
            return;
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let stale = name == LEGACY_DATA_FILE
            || name == LEGACY_CHECKSUM_FILE
            || (is_data_file_name(name) && name != current);
        if stale {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                tracing::warn!("could not remove old snapshot file {name}: {e}");
            }
        }
    }
}

/// Read the snapshot metadata, if there is any.
fn read_metadata(dir: &Path) -> Result<Option<SnapshotMetadata>> {
    match std::fs::read(dir.join(METADATA_FILE)) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// The data file for `metadata` and its contents: the file named after the
/// checksum, or the data file of an earlier release.
fn read_data(dir: &Path, metadata: &SnapshotMetadata) -> Result<(PathBuf, Vec<u8>)> {
    for name in [
        data_file_name(&metadata.checksum),
        LEGACY_DATA_FILE.to_string(),
    ] {
        let path = dir.join(name);
        match std::fs::read(&path) {
            Ok(bytes) => return Ok((path, bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(anyhow!(
        "the snapshot data file named by {} is missing",
        dir.join(METADATA_FILE).display()
    ))
}

/// Load the current snapshot. With corruption detection enabled, its data
/// must match the checksum in the metadata.
pub(super) fn read_snapshot(
    dir: &Path,
    config: &StorageConfig,
) -> Result<Option<(SnapshotMetadata, RdfApp)>> {
    let Some(metadata) = read_metadata(dir)? else {
        return Ok(None);
    };
    let (path, bytes) = read_data(dir, &metadata)?;
    if config.enable_corruption_detection && sha256_hex(&bytes) != metadata.checksum {
        return Err(anyhow!(
            "snapshot {} does not match the checksum recorded in {METADATA_FILE}",
            path.display()
        ));
    }
    let json = if bytes.starts_with(&oxiarc_lz4::LZ4_FRAME_MAGIC.to_le_bytes()) {
        oxiarc_lz4::decompress(&bytes, MAX_SNAPSHOT_LEN)
            .map_err(|e| anyhow!("snapshot {} does not decompress: {e}", path.display()))?
    } else {
        bytes
    };
    let app = serde_json::from_slice(&json)?;
    Ok(Some((metadata, app)))
}

/// The state of the snapshot files on disk.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum SnapshotCheck {
    /// No snapshot was taken.
    Absent,
    /// The data file matches the metadata.
    Intact,
    /// The snapshot cannot be used, for the reason given.
    Damaged(String),
}

/// Check the snapshot files against each other.
pub(super) fn check_snapshot(dir: &Path) -> SnapshotCheck {
    let metadata = match read_metadata(dir) {
        Ok(Some(metadata)) => metadata,
        Ok(None) => return SnapshotCheck::Absent,
        Err(e) => return SnapshotCheck::Damaged(format!("{METADATA_FILE} is unreadable: {e}")),
    };
    match read_data(dir, &metadata) {
        Ok((_, bytes)) if sha256_hex(&bytes) == metadata.checksum => SnapshotCheck::Intact,
        Ok((path, _)) => SnapshotCheck::Damaged(format!(
            "{} does not match the checksum in {METADATA_FILE}",
            path.display()
        )),
        Err(e) => SnapshotCheck::Damaged(e.to_string()),
    }
}

/// Restore the data file named by the metadata from a backup of this node,
/// if a backup holds a copy that matches the recorded checksum.
pub(super) fn restore_from_backup(
    dir: &Path,
    node_id: OxirsNodeId,
    config: &StorageConfig,
) -> Result<bool> {
    let Some(metadata) = read_metadata(dir)? else {
        return Ok(false);
    };
    let Some(parent) = dir.parent() else {
        return Ok(false);
    };
    let prefix = format!("backup-{node_id}-");
    let mut backups: Vec<PathBuf> = std::fs::read_dir(parent)?
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .map(|entry| entry.path())
        .collect();
    // Newest first: backup directories end in their creation time.
    backups.sort_unstable_by(|a, b| b.cmp(a));

    let name = data_file_name(&metadata.checksum);
    for backup in backups {
        for candidate in [name.as_str(), LEGACY_DATA_FILE] {
            let Ok(bytes) = std::fs::read(backup.join(candidate)) else {
                continue;
            };
            if sha256_hex(&bytes) == metadata.checksum {
                write_atomic(dir, &name, &bytes, config.sync_writes)?;
                tracing::info!("Restored snapshot {name} from backup {}", backup.display());
                return Ok(true);
            }
        }
    }
    Ok(false)
}
