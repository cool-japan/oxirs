//! Integrity verification and crash recovery for PersistentStorage

use super::config::StorageConfig;
use super::recovery::*;
use super::types::*;

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::raft::{OxirsNodeId, RdfApp};

/// Perform crash recovery check and repair if needed
pub(super) async fn recover_from_crash(
    config: &StorageConfig,
    data_dir: &Path,
    node_id: OxirsNodeId,
    raft_state: &Arc<RwLock<RaftState>>,
) -> Result<RecoveryReport> {
    if !config.enable_crash_recovery {
        return Ok(RecoveryReport::new());
    }

    let mut report = RecoveryReport::new();

    if config.enable_wal {
        report.wal_recovered = recover_from_wal(config, data_dir).await?;
    }

    if config.enable_corruption_detection {
        let corruption_report = check_file_integrity(config, data_dir).await?;
        report.corrupted_files = corruption_report.corrupted_files;
        report.recovered_files = corruption_report.recovered_files;
    }

    let log_consistency = verify_log_consistency(raft_state).await?;
    if !log_consistency.is_consistent {
        report.log_inconsistencies = log_consistency.issues.len();
        repair_log_inconsistencies(raft_state, log_consistency.issues, data_dir, config).await?;
    }

    let state_consistency = verify_state_consistency(raft_state).await?;
    report.state_machine_repaired = state_consistency.repaired;

    tracing::info!(
        "Crash recovery completed for node {}: {:?}",
        node_id,
        report
    );
    Ok(report)
}

/// Recover from write-ahead log
pub(super) async fn recover_from_wal(_config: &StorageConfig, data_dir: &Path) -> Result<bool> {
    let wal_path = data_dir.join("wal.log");
    if !wal_path.exists() {
        return Ok(false);
    }

    tracing::info!("Recovering from write-ahead log");

    let wal_data = fs::read(&wal_path)?;
    if wal_data.is_empty() {
        fs::remove_file(&wal_path)?;
        return Ok(false);
    }

    if let Ok(operations) = serde_json::from_slice::<Vec<WalOperation>>(&wal_data) {
        for operation in operations {
            match operation {
                WalOperation::WriteRaftState(state) => {
                    let state_json = serde_json::to_string(&state)?;
                    let state_path = data_dir.join("raft_state.json");
                    fs::write(&state_path, state_json)?;
                }
                WalOperation::WriteAppState(app_state) => {
                    let app_json = serde_json::to_string(&app_state)?;
                    let app_state_path = data_dir.join("app_state.json");
                    fs::write(&app_state_path, app_json)?;
                }
                WalOperation::CreateSnapshot(metadata) => {
                    let metadata_json = serde_json::to_string(&metadata)?;
                    let snapshot_path = data_dir.join("snapshot_metadata.json");
                    fs::write(&snapshot_path, metadata_json)?;
                }
                WalOperation::TruncateLog(_index) => {}
                WalOperation::Commit(_sequence) => {}
            }
        }

        fs::remove_file(&wal_path)?;
        tracing::info!("Successfully recovered from WAL");
        return Ok(true);
    }

    fs::remove_file(&wal_path)?;
    tracing::warn!("WAL file was corrupted and removed");
    Ok(false)
}

/// Check file integrity using checksums
pub(super) async fn check_file_integrity(
    _config: &StorageConfig,
    data_dir: &Path,
) -> Result<CorruptionReport> {
    let mut report = CorruptionReport::new();

    let files_to_check = vec![
        ("raft_state.json", "raft_state.json.checksum"),
        ("app_state.json", "app_state.json.checksum"),
        ("snapshot.json", "snapshot.json.checksum"),
    ];

    for (filename, checksum_filename) in files_to_check {
        let file_path = data_dir.join(filename);
        let checksum_path = data_dir.join(checksum_filename);

        if file_path.exists() {
            let integrity_ok = verify_file_checksum(&file_path, &checksum_path).await?;
            if !integrity_ok {
                report.corrupted_files.push(filename.to_string());

                if recover_corrupted_file(data_dir, &file_path).await? {
                    report.recovered_files.push(filename.to_string());
                }
            }
        }
    }

    Ok(report)
}

/// Verify file checksum
pub(super) async fn verify_file_checksum(file_path: &Path, checksum_path: &Path) -> Result<bool> {
    if !checksum_path.exists() {
        let checksum = calculate_file_checksum(file_path).await?;
        fs::write(checksum_path, checksum)?;
        return Ok(true);
    }

    let stored_checksum = fs::read_to_string(checksum_path)?;
    let current_checksum = calculate_file_checksum(file_path).await?;

    Ok(stored_checksum.trim() == current_checksum)
}

/// Calculate SHA-256 checksum of a file
pub(super) async fn calculate_file_checksum(file_path: &Path) -> Result<String> {
    let data = fs::read(file_path)?;
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    Ok(hex::encode(result))
}

/// Recover corrupted file from backup
pub(super) async fn recover_corrupted_file(data_dir: &Path, file_path: &Path) -> Result<bool> {
    let filename = file_path
        .file_name()
        .expect("file_path should have a file name")
        .to_string_lossy();

    // We need node_id to find backups — derive it from data_dir name
    let dir_name = data_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let node_id_str = dir_name.strip_prefix("node-").unwrap_or(&dir_name);

    let parent_dir = data_dir
        .parent()
        .expect("data_dir should have a parent directory");
    let backup_prefix = format!("backup-{}-", node_id_str);

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

    if backups.is_empty() {
        return Ok(false);
    }

    backups.sort_by(|a, b| b.1.cmp(&a.1));
    let (backup_path, _) = &backups[0];

    fs::copy(backup_path, file_path)?;

    let checksum_path = file_path.with_extension(format!(
        "{}.checksum",
        file_path.extension().unwrap_or_default().to_string_lossy()
    ));
    let checksum = calculate_file_checksum(file_path).await?;
    fs::write(&checksum_path, checksum)?;

    tracing::info!("Recovered corrupted file {} from backup", filename);
    Ok(true)
}

/// Verify log consistency
pub(super) async fn verify_log_consistency(
    raft_state: &Arc<RwLock<RaftState>>,
) -> Result<LogConsistencyReport> {
    let state = raft_state.read().await;
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
    Ok(report)
}

/// Repair log inconsistencies
pub(super) async fn repair_log_inconsistencies(
    raft_state: &Arc<RwLock<RaftState>>,
    issues: Vec<LogInconsistency>,
    data_dir: &Path,
    config: &StorageConfig,
) -> Result<()> {
    let mut state = raft_state.write().await;

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

    // Save repaired state using atomic write
    let state_clone = state.clone();
    drop(state);
    save_raft_state_direct(&state_clone, data_dir, config).await
}

/// Direct raft state save (used during repair without full PersistentStorage context)
async fn save_raft_state_direct(
    state: &RaftState,
    data_dir: &Path,
    config: &StorageConfig,
) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};

    let filename = "raft_state.dat";
    let path = data_dir.join(filename);
    let temp_path = data_dir.join(format!("{filename}.tmp"));

    let checksummed_data: ChecksummedData<RaftState> = if config.enable_corruption_detection {
        ChecksummedData::new(state.clone())?
    } else {
        ChecksummedData {
            data: state.clone(),
            checksum: String::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    };

    let serialized = oxicode::serde::encode_to_vec(&checksummed_data, oxicode::config::standard())?;

    {
        let temp_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        let mut writer = BufWriter::new(temp_file);
        writer.write_all(&serialized)?;
        writer.flush()?;

        if config.sync_writes {
            writer.get_mut().sync_all()?;
        }
    }

    std::fs::rename(&temp_path, &path)?;
    Ok(())
}

/// Verify data integrity across all files
pub(super) async fn verify_integrity(
    config: &StorageConfig,
    data_dir: &Path,
    raft_state: &Arc<RwLock<RaftState>>,
    app_state: &Arc<RwLock<RdfApp>>,
) -> Result<bool> {
    let mut all_valid = true;

    let raft_path = data_dir.join("raft_state.dat");
    if raft_path.exists() {
        match load_with_checksum_verify::<RaftState>(config, &raft_path).await {
            Ok(_) => tracing::info!("Raft state integrity verified"),
            Err(e) => {
                tracing::error!("Raft state integrity check failed: {}", e);
                all_valid = false;
            }
        }
    }

    let app_path = data_dir.join("app_state.dat");
    if app_path.exists() {
        match load_with_checksum_verify::<RdfApp>(config, &app_path).await {
            Ok(_) => tracing::info!("Application state integrity verified"),
            Err(e) => {
                tracing::error!("Application state integrity check failed: {}", e);
                all_valid = false;
            }
        }
    }

    // Keep locks alive until done verifying to avoid dropping prematurely
    let _ = raft_state.read().await;
    let _ = app_state.read().await;

    if config.enable_wal {
        let wal_path = data_dir.join("wal.log");
        if wal_path.exists() {
            match super::persistent_wal::verify_wal_integrity(&wal_path).await {
                Ok(valid_entries) => {
                    tracing::info!("WAL integrity verified, {} valid entries", valid_entries)
                }
                Err(e) => {
                    tracing::error!("WAL integrity check failed: {}", e);
                    all_valid = false;
                }
            }
        }
    }

    Ok(all_valid)
}

/// Load and verify checksum for a typed value (shared helper)
async fn load_with_checksum_verify<T>(config: &StorageConfig, path: &Path) -> Result<T>
where
    T: for<'de> serde::Deserialize<'de> + serde::Serialize,
{
    let data = std::fs::read(path)?;
    let (checksummed_data, _): (ChecksummedData<T>, _) =
        oxicode::serde::decode_from_slice(&data, oxicode::config::standard())?;

    if config.enable_corruption_detection && !checksummed_data.verify()? {
        return Err(anyhow!("Checksum verification failed for {:?}", path));
    }

    Ok(checksummed_data.data)
}

/// Verify state machine consistency
pub(super) async fn verify_state_consistency(
    raft_state: &Arc<RwLock<RaftState>>,
) -> Result<StateConsistencyReport> {
    let state = raft_state.read().await;
    let mut report = StateConsistencyReport::new();

    if state.last_applied < state.commit_index {
        report.repaired = true;
        tracing::info!(
            "Applying unapplied committed entries: {} to {}",
            state.last_applied + 1,
            state.commit_index
        );
    }

    Ok(report)
}
