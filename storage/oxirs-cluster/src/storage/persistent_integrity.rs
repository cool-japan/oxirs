//! Integrity verification and crash recovery for PersistentStorage

use super::config::StorageConfig;
use super::persistent_files::{
    read_checksummed, write_checksummed, APP_STATE_FILE, RAFT_STATE_FILE,
};
use super::persistent_snapshot::{self, SnapshotCheck};
use super::persistent_wal::WalWriter;
use super::recovery::*;
use super::types::*;

use anyhow::{anyhow, Result};
use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use crate::raft::{OxirsNodeId, RdfApp};

/// Perform crash recovery checks and repair what can be repaired.
///
/// The WAL itself is recovered before the store opens; `wal_recovered`
/// reports whether that redid a journaled write.
pub(super) fn recover_from_crash(
    config: &StorageConfig,
    data_dir: &Path,
    node_id: OxirsNodeId,
    raft_state: &RwLock<RaftState>,
    wal_recovered: bool,
) -> Result<RecoveryReport> {
    if !config.enable_crash_recovery {
        return Ok(RecoveryReport::new());
    }

    let mut report = RecoveryReport::new();
    report.wal_recovered = wal_recovered;

    if config.enable_corruption_detection {
        let corruption_report = check_file_integrity(config, data_dir, node_id)?;
        report.corrupted_files = corruption_report.corrupted_files;
        report.recovered_files = corruption_report.recovered_files;
    }

    let log_consistency = verify_log_consistency(raft_state);
    if !log_consistency.is_consistent {
        report.log_inconsistencies = log_consistency.issues.len();
        repair_log_inconsistencies(raft_state, log_consistency.issues, data_dir, config)?;
    }

    let state_consistency = verify_state_consistency(raft_state);
    report.state_machine_repaired = state_consistency.repaired;

    tracing::info!(
        "Crash recovery completed for node {}: {:?}",
        node_id,
        report
    );
    Ok(report)
}

/// Check the snapshot against its recorded checksum, and the state files of
/// releases that wrote JSON against their checksum files.
pub(super) fn check_file_integrity(
    config: &StorageConfig,
    data_dir: &Path,
    node_id: OxirsNodeId,
) -> Result<CorruptionReport> {
    let mut report = CorruptionReport::new();

    match persistent_snapshot::check_snapshot(data_dir) {
        SnapshotCheck::Absent | SnapshotCheck::Intact => {}
        SnapshotCheck::Damaged(reason) => {
            tracing::warn!("Snapshot of node {node_id} is damaged: {reason}");
            let name = persistent_snapshot::METADATA_FILE.to_string();
            report.corrupted_files.push(name.clone());
            if persistent_snapshot::restore_from_backup(data_dir, node_id, config)? {
                report.recovered_files.push(name);
            }
        }
    }

    // Legacy JSON state files carry no checksum of their own.
    let files_to_check = [
        ("raft_state.json", "raft_state.json.checksum"),
        ("app_state.json", "app_state.json.checksum"),
    ];

    for (filename, checksum_filename) in files_to_check {
        let file_path = data_dir.join(filename);
        let checksum_path = data_dir.join(checksum_filename);

        if file_path.exists() {
            let integrity_ok = verify_file_checksum(&file_path, &checksum_path)?;
            if !integrity_ok {
                report.corrupted_files.push(filename.to_string());

                if recover_corrupted_file(data_dir, node_id, &file_path)? {
                    report.recovered_files.push(filename.to_string());
                }
            }
        }
    }

    Ok(report)
}

/// Verify file checksum
pub(super) fn verify_file_checksum(file_path: &Path, checksum_path: &Path) -> Result<bool> {
    if !checksum_path.exists() {
        let checksum = calculate_file_checksum(file_path)?;
        fs::write(checksum_path, checksum)?;
        return Ok(true);
    }

    let stored_checksum = fs::read_to_string(checksum_path)?;
    let current_checksum = calculate_file_checksum(file_path)?;

    Ok(stored_checksum.trim() == current_checksum)
}

/// Calculate SHA-256 checksum of a file
pub(super) fn calculate_file_checksum(file_path: &Path) -> Result<String> {
    let data = fs::read(file_path)?;
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    Ok(hex::encode(result))
}

/// Recover corrupted file from the newest backup of this node
pub(super) fn recover_corrupted_file(
    data_dir: &Path,
    node_id: OxirsNodeId,
    file_path: &Path,
) -> Result<bool> {
    let filename = file_path
        .file_name()
        .ok_or_else(|| anyhow!("{} has no file name", file_path.display()))?
        .to_string_lossy();
    let parent_dir = data_dir
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", data_dir.display()))?;
    let backup_prefix = format!("backup-{node_id}-");

    let mut backups = Vec::new();
    for entry in fs::read_dir(parent_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if let Some(name_str) = name.to_str() {
            if name_str.starts_with(&backup_prefix) {
                let backup_file_path = entry.path().join(&*filename);
                if backup_file_path.exists() {
                    backups.push((backup_file_path, name_str.to_string()));
                }
            }
        }
    }

    backups.sort_by(|a, b| b.1.cmp(&a.1));
    let Some((backup_path, _)) = backups.first() else {
        return Ok(false);
    };

    fs::copy(backup_path, file_path)?;

    let checksum_path = file_path.with_extension(format!(
        "{}.checksum",
        file_path.extension().unwrap_or_default().to_string_lossy()
    ));
    let checksum = calculate_file_checksum(file_path)?;
    fs::write(&checksum_path, checksum)?;

    tracing::info!("Recovered corrupted file {} from backup", filename);
    Ok(true)
}

/// Verify log consistency
pub(super) fn verify_log_consistency(raft_state: &RwLock<RaftState>) -> LogConsistencyReport {
    let state = raft_state.read();
    let mut report = LogConsistencyReport::new();

    let mut expected_index = 1u64;
    for entry in &state.log {
        if entry.index != expected_index {
            report.issues.push(LogInconsistency::IndexGap {
                expected: expected_index,
                found: entry.index,
            });
        }
        expected_index = entry.index + 1;
    }

    let mut indices = std::collections::HashSet::new();
    for entry in &state.log {
        if !indices.insert(entry.index) {
            report
                .issues
                .push(LogInconsistency::DuplicateIndex { index: entry.index });
        }
    }

    if state.commit_index > state.log.last().map(|e| e.index).unwrap_or(0) {
        report.issues.push(LogInconsistency::InvalidCommitIndex {
            commit_index: state.commit_index,
            last_log_index: state.log.last().map(|e| e.index).unwrap_or(0),
        });
    }

    report.is_consistent = report.issues.is_empty();
    report
}

/// Repair log inconsistencies
pub(super) fn repair_log_inconsistencies(
    raft_state: &RwLock<RaftState>,
    issues: Vec<LogInconsistency>,
    data_dir: &Path,
    config: &StorageConfig,
) -> Result<()> {
    let mut state = raft_state.write();

    for issue in issues {
        match issue {
            LogInconsistency::IndexGap { expected, found } => {
                tracing::warn!(
                    "Fixing log index gap: expected {}, found {}",
                    expected,
                    found
                );
                state
                    .log
                    .retain(|entry| entry.index < expected || entry.index >= found);
            }
            LogInconsistency::DuplicateIndex { index } => {
                tracing::warn!("Removing duplicate log entry at index {}", index);
                let mut seen = false;
                state.log.retain(|entry| {
                    if entry.index == index {
                        if seen {
                            false
                        } else {
                            seen = true;
                            true
                        }
                    } else {
                        true
                    }
                });
            }
            LogInconsistency::InvalidCommitIndex {
                commit_index,
                last_log_index,
            } => {
                tracing::warn!(
                    "Fixing invalid commit index: {} > {}",
                    commit_index,
                    last_log_index
                );
                state.commit_index = last_log_index;
            }
        }
    }

    write_checksummed(data_dir, RAFT_STATE_FILE, &*state, config)
}

/// Verify data integrity across all files: the state files, the snapshot
/// and the WAL.
pub(super) fn verify_integrity(
    config: &StorageConfig,
    data_dir: &Path,
    wal: &Mutex<Option<WalWriter>>,
) -> Result<bool> {
    let mut all_valid = true;

    let raft_path = data_dir.join(RAFT_STATE_FILE);
    if raft_path.exists() {
        match read_checksummed::<RaftState>(&raft_path, config) {
            Ok(_) => tracing::info!("Raft state integrity verified"),
            Err(e) => {
                tracing::error!("Raft state integrity check failed: {}", e);
                all_valid = false;
            }
        }
    }

    let app_path = data_dir.join(APP_STATE_FILE);
    if app_path.exists() {
        match read_checksummed::<RdfApp>(&app_path, config) {
            Ok(_) => tracing::info!("Application state integrity verified"),
            Err(e) => {
                tracing::error!("Application state integrity check failed: {}", e);
                all_valid = false;
            }
        }
    }

    if let SnapshotCheck::Damaged(reason) = persistent_snapshot::check_snapshot(data_dir) {
        tracing::error!("Snapshot integrity check failed: {reason}");
        all_valid = false;
    }

    if let Some(writer) = wal.lock().as_ref() {
        match writer.verify() {
            Ok(valid_entries) => {
                tracing::info!("WAL integrity verified, {} valid entries", valid_entries)
            }
            Err(e) => {
                tracing::error!("WAL integrity check failed: {}", e);
                all_valid = false;
            }
        }
    }

    Ok(all_valid)
}

/// Verify state machine consistency
pub(super) fn verify_state_consistency(raft_state: &RwLock<RaftState>) -> StateConsistencyReport {
    let state = raft_state.read();
    let mut report = StateConsistencyReport::new();

    if state.last_applied < state.commit_index {
        report.repaired = true;
        tracing::info!(
            "Applying unapplied committed entries: {} to {}",
            state.last_applied + 1,
            state.commit_index
        );
    }

    report
}
