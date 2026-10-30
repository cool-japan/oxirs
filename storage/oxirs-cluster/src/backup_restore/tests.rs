//! Tests for [`super`]: backup creation, persistence and restore.

use super::*;
use std::env;

#[tokio::test]
async fn test_backup_restore_manager_creation() {
    let temp_dir = env::temp_dir().join("oxirs_backup_test_1");
    let config = BackupConfig {
        backup_dir: temp_dir.clone(),
        ..Default::default()
    };

    let manager = BackupRestoreManager::new(config, 1).await;
    assert!(manager.is_ok());

    // Cleanup
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_full_backup_creation() {
    let temp_dir = env::temp_dir().join("oxirs_backup_test_2");
    let config = BackupConfig {
        backup_dir: temp_dir.clone(),
        verify_after_backup: false,
        ..Default::default()
    };

    let manager = BackupRestoreManager::new(config, 1).await.unwrap();

    let data = vec![
        ("s1".to_string(), "p1".to_string(), "o1".to_string()),
        ("s2".to_string(), "p2".to_string(), "o2".to_string()),
    ];

    let backup_id = manager.create_full_backup(data.clone()).await;
    assert!(backup_id.is_ok());

    let backup_id = backup_id.unwrap();
    let metadata = manager.get_backup_metadata(&backup_id).await;
    assert!(metadata.is_some());

    let metadata = metadata.unwrap();
    assert_eq!(metadata.backup_type, BackupType::Full);
    assert_eq!(metadata.triple_count, 2);

    // Cleanup
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_backup_and_restore() {
    let temp_dir = env::temp_dir().join("oxirs_backup_test_3");
    let config = BackupConfig {
        backup_dir: temp_dir.clone(),
        verify_after_backup: false,
        ..Default::default()
    };

    let manager = BackupRestoreManager::new(config, 1).await.unwrap();

    let original_data = vec![
        ("s1".to_string(), "p1".to_string(), "o1".to_string()),
        ("s2".to_string(), "p2".to_string(), "o2".to_string()),
    ];

    // Create backup
    let backup_id = manager
        .create_full_backup(original_data.clone())
        .await
        .unwrap();

    // Restore backup
    let options = RestoreOptions {
        backup_id,
        skip_verification: true,
        overwrite_existing: true,
        point_in_time: None,
    };

    let restored_data = manager.restore_backup(options).await;
    assert!(restored_data.is_ok());

    let restored_data = restored_data.unwrap();
    assert_eq!(restored_data.len(), original_data.len());

    // Cleanup
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_backup_statistics() {
    let temp_dir = env::temp_dir().join("oxirs_backup_test_4");
    let config = BackupConfig {
        backup_dir: temp_dir.clone(),
        verify_after_backup: false,
        ..Default::default()
    };

    let manager = BackupRestoreManager::new(config, 1).await.unwrap();

    let data = vec![("s1".to_string(), "p1".to_string(), "o1".to_string())];

    let _ = manager.create_full_backup(data).await.unwrap();

    let stats = manager.get_statistics().await;
    assert_eq!(stats.total_backups, 1);
    assert!(stats.total_bytes_backed_up > 0);

    // Cleanup
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

fn write_key_file(dir: &std::path::Path, byte: u8) -> PathBuf {
    let path = dir.join(format!("backup-{byte}.key"));
    std::fs::write(&path, [byte; 32]).expect("write key file");
    path
}

fn encrypted_config(dir: &std::path::Path, key_file: PathBuf) -> BackupConfig {
    BackupConfig {
        backup_dir: dir.join("backups"),
        enable_encryption: true,
        encryption_key_file: Some(key_file),
        ..Default::default()
    }
}

fn restore_options(backup_id: &str, skip_verification: bool) -> RestoreOptions {
    RestoreOptions {
        backup_id: backup_id.to_string(),
        skip_verification,
        overwrite_existing: false,
        point_in_time: None,
    }
}

fn triple(subject: &str) -> (String, String, String) {
    (subject.to_string(), "p".to_string(), "o".to_string())
}

#[tokio::test]
async fn test_encrypted_backup_round_trip() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let key_file = write_key_file(temp_dir.path(), 1);
    let manager = BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file), 1)
        .await
        .expect("manager with key");

    let full = manager
        .create_full_backup(vec![triple("s1")])
        .await
        .expect("full backup");
    let incremental = manager
        .create_incremental_backup(&full, vec![triple("s2")])
        .await
        .expect("incremental backup");

    for id in [&full, &incremental] {
        let metadata = manager.get_backup_metadata(id).await.expect("metadata");
        assert!(metadata.encrypted);
        let bytes = std::fs::read(&metadata.file_path).expect("read backup file");
        assert!(at_rest::is_sealed(&bytes));
    }
    let restored = manager
        .restore_backup(restore_options(&incremental, false))
        .await
        .expect("restore");
    assert_eq!(restored, vec![triple("s1"), triple("s2")]);
}

#[tokio::test]
async fn test_encryption_settings_must_be_consistent() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");

    let flag_only = BackupConfig {
        backup_dir: temp_dir.path().join("a"),
        enable_encryption: true,
        ..Default::default()
    };
    assert!(BackupRestoreManager::new(flag_only, 1).await.is_err());

    let key_only = BackupConfig {
        backup_dir: temp_dir.path().join("b"),
        encryption_key_file: Some(write_key_file(temp_dir.path(), 1)),
        ..Default::default()
    };
    assert!(BackupRestoreManager::new(key_only, 1).await.is_err());

    let missing_key = encrypted_config(temp_dir.path(), temp_dir.path().join("none.key"));
    assert!(BackupRestoreManager::new(missing_key, 1).await.is_err());
}

#[tokio::test]
async fn test_encrypted_backup_rejects_wrong_key_and_tampering() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let key_file = write_key_file(temp_dir.path(), 1);
    let mut manager = BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file), 1)
        .await
        .expect("manager with key");
    let backup_id = manager
        .create_full_backup(vec![triple("s1")])
        .await
        .expect("full backup");

    // Same key id, different key bytes: authentication must fail.
    let original = manager.encryption_key.clone().expect("key configured");
    manager.encryption_key = Some(AtRestKey::new(original.id(), [2u8; 32]).expect("valid key id"));
    assert!(manager
        .restore_backup(restore_options(&backup_id, true))
        .await
        .is_err());

    // A different key altogether.
    manager.encryption_key = Some(AtRestKey::from_key([3u8; 32]));
    assert!(manager
        .restore_backup(restore_options(&backup_id, true))
        .await
        .is_err());

    // Tampered file, with checksum verification skipped.
    manager.encryption_key = Some(original);
    let metadata = manager
        .get_backup_metadata(&backup_id)
        .await
        .expect("metadata");
    let mut bytes = std::fs::read(&metadata.file_path).expect("read backup file");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&metadata.file_path, &bytes).expect("write tampered file");
    assert!(manager
        .restore_backup(restore_options(&backup_id, true))
        .await
        .is_err());
}

// ---------------------------------------------------------------------------
// Persistence across restarts
// ---------------------------------------------------------------------------

fn plain_config(dir: &std::path::Path) -> BackupConfig {
    BackupConfig {
        backup_dir: dir.join("backups"),
        ..Default::default()
    }
}

/// Backup metadata lived only in memory, so after a restart no backup could
/// be restored (or even listed) by its id.
#[tokio::test]
async fn test_backups_survive_a_restart() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let (full, incremental) = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        let full = manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup");
        let incremental = manager
            .create_incremental_backup(&full, vec![triple("s2")])
            .await
            .expect("incremental backup");
        (full, incremental)
    };

    let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
        .await
        .expect("manager after restart");
    assert_eq!(manager.list_backups().await.len(), 2);
    let metadata = manager
        .get_backup_metadata(&full)
        .await
        .expect("full backup metadata");
    assert_eq!(metadata.status, BackupStatus::Verified);
    let restored = manager
        .restore_backup(restore_options(&incremental, false))
        .await
        .expect("restore after restart");
    assert_eq!(restored, vec![triple("s1"), triple("s2")]);

    // Incremental backups can build on a backup made before the restart.
    let next = manager
        .create_incremental_backup(&incremental, vec![triple("s3")])
        .await
        .expect("incremental after restart");
    let restored = manager
        .restore_backup(restore_options(&next, false))
        .await
        .expect("restore chain");
    assert_eq!(restored, vec![triple("s1"), triple("s2"), triple("s3")]);
}

#[tokio::test]
async fn test_encrypted_backups_survive_a_restart() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let key_file = write_key_file(temp_dir.path(), 1);
    let backup_id = {
        let manager =
            BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file.clone()), 1)
                .await
                .expect("manager with key");
        manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup")
    };

    let manager = BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file), 1)
        .await
        .expect("manager after restart");
    let restored = manager
        .restore_backup(restore_options(&backup_id, false))
        .await
        .expect("restore after restart");
    assert_eq!(restored, vec![triple("s1")]);
}

/// Sealed backup files used to share one context, so one could be passed
/// off as another under the same key.
#[tokio::test]
async fn test_swapped_sealed_backups_do_not_open() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let key_file = write_key_file(temp_dir.path(), 1);
    let manager = BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file), 1)
        .await
        .expect("manager with key");
    let first = manager
        .create_full_backup(vec![triple("first")])
        .await
        .expect("first backup");
    let second = manager
        .create_full_backup(vec![triple("second")])
        .await
        .expect("second backup");

    let first_path = manager
        .get_backup_metadata(&first)
        .await
        .expect("metadata")
        .file_path;
    let second_path = manager
        .get_backup_metadata(&second)
        .await
        .expect("metadata")
        .file_path;
    std::fs::copy(&second_path, &first_path).expect("swap backup files");

    assert!(
        manager
            .restore_backup(restore_options(&first, true))
            .await
            .is_err(),
        "a sealed backup opened as a different backup"
    );
}

/// Restore follows what the backup recorded, not today's settings.
#[tokio::test]
async fn test_restore_after_a_settings_change() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let backup_id = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup")
    };

    let changed = BackupConfig {
        format: BackupFormat::Json,
        enable_compression: false,
        ..plain_config(temp_dir.path())
    };
    let manager = BackupRestoreManager::new(changed, 1)
        .await
        .expect("manager after restart");
    let restored = manager
        .restore_backup(restore_options(&backup_id, false))
        .await
        .expect("restore with changed settings");
    assert_eq!(restored, vec![triple("s1")]);
}

/// Path of the metadata file the manager keeps next to backup `metadata`.
fn metadata_file(metadata: &BackupMetadata) -> PathBuf {
    let mut path = metadata.file_path.clone().into_os_string();
    path.push(METADATA_SUFFIX);
    PathBuf::from(path)
}

/// Metadata files are read back from disk, so they must not be able to
/// point the manager at files outside its backup directory: retention and
/// `delete_backup` remove the files that metadata names.
#[tokio::test]
async fn test_tampered_metadata_cannot_redirect_file_operations() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let config = BackupConfig {
        max_backups: 1,
        ..plain_config(temp_dir.path())
    };
    let real = {
        let manager = BackupRestoreManager::new(config.clone(), 1)
            .await
            .expect("manager");
        let id = manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup");
        manager.get_backup_metadata(&id).await.expect("metadata")
    };
    let stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(metadata_file(&real)).expect("read metadata"))
            .expect("metadata is JSON");
    assert_eq!(
        stored["file_path"],
        real.file_path.file_name().unwrap().to_str().unwrap()
    );

    // Old-looking entries that name files outside the backup directory.
    let victim = temp_dir.path().join("victim.txt");
    std::fs::write(&victim, b"keep me").expect("write victim");
    for file_path in [victim.clone(), PathBuf::from("../victim.txt")] {
        let id = uuid::Uuid::new_v4().to_string();
        let forged = BackupMetadata {
            id: id.clone(),
            created_at: SystemTime::UNIX_EPOCH,
            file_path,
            ..real.clone()
        };
        let sidecar = config
            .backup_dir
            .join(format!("backup_full_1_{id}.dat{METADATA_SUFFIX}"));
        std::fs::write(&sidecar, serde_json::to_vec(&forged).expect("encode")).expect("write");
        std::fs::write(
            config.backup_dir.join(format!("backup_full_1_{id}.dat")),
            b"x",
        )
        .expect("write data file");
    }

    let manager = BackupRestoreManager::new(config, 1)
        .await
        .expect("manager after restart");
    assert_eq!(
        manager.list_backups().await.len(),
        1,
        "forged entries were loaded"
    );
    // Creating a backup runs retention (max_backups = 1) over everything listed.
    manager
        .create_full_backup(vec![triple("s2")])
        .await
        .expect("second backup");
    assert_eq!(std::fs::read(&victim).expect("victim survives"), b"keep me");
}

/// With encryption enabled, an unencrypted backup could have been planted
/// by anyone with write access to the directory, so it is not restored.
#[tokio::test]
async fn test_unencrypted_backup_is_refused_when_encryption_is_enabled() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let backup_id = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        manager
            .create_full_backup(vec![triple("planted")])
            .await
            .expect("plaintext backup")
    };

    let key_file = write_key_file(temp_dir.path(), 1);
    let manager = BackupRestoreManager::new(encrypted_config(temp_dir.path(), key_file), 1)
        .await
        .expect("manager with key");
    let error = manager
        .restore_backup(restore_options(&backup_id, false))
        .await
        .expect_err("an unencrypted backup was restored");
    assert!(matches!(error, ClusterError::Encryption(_)), "{error}");
}

/// A parent chain that loops (edited metadata) is an error; restore used to
/// recurse forever.
#[tokio::test]
async fn test_parent_cycle_is_an_error() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let (second, third) = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        let first = manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup");
        let second = manager
            .create_incremental_backup(&first, vec![triple("s2")])
            .await
            .expect("second backup");
        let third = manager
            .create_incremental_backup(&second, vec![triple("s3")])
            .await
            .expect("third backup");
        (
            manager
                .get_backup_metadata(&second)
                .await
                .expect("metadata"),
            third,
        )
    };

    // Point the second backup back at the third one.
    let path = metadata_file(&second);
    let mut stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read metadata")).expect("JSON");
    stored["parent_backup_id"] = serde_json::Value::String(third.clone());
    std::fs::write(&path, serde_json::to_vec(&stored).expect("encode")).expect("write");

    let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
        .await
        .expect("manager after restart");
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        manager.restore_backup(restore_options(&third, true)),
    )
    .await
    .expect("restore did not terminate");
    assert!(result.is_err(), "a looping chain was restored");
}

/// Nodes that share a backup directory keep to their own backups: another
/// node's backups are neither listed nor removed by retention.
#[tokio::test]
async fn test_other_nodes_backups_are_left_alone() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let config = BackupConfig {
        max_backups: 1,
        ..plain_config(temp_dir.path())
    };
    let other = {
        let manager = BackupRestoreManager::new(config.clone(), 2)
            .await
            .expect("node 2 manager");
        let id = manager
            .create_full_backup(vec![triple("node 2")])
            .await
            .expect("node 2 backup");
        manager.get_backup_metadata(&id).await.expect("metadata")
    };

    let manager = BackupRestoreManager::new(config, 1)
        .await
        .expect("node 1 manager");
    assert!(manager.list_backups().await.is_empty());
    for subject in ["a", "b"] {
        manager
            .create_full_backup(vec![triple(subject)])
            .await
            .expect("node 1 backup");
    }
    assert!(
        other.file_path.exists(),
        "retention removed another node's backup"
    );
    assert!(metadata_file(&other).exists());
}

// ---------------------------------------------------------------------------
// Retention keeps every backup that a kept backup needs
// ---------------------------------------------------------------------------

/// Move the recorded creation time of `metadata` back by `days`, as if the
/// backup had been made that long ago.
fn age_backup(metadata: &BackupMetadata, days: u64) {
    let path = metadata_file(metadata);
    let mut stored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read metadata")).expect("JSON");
    let secs = stored["created_at"]["secs_since_epoch"]
        .as_u64()
        .expect("created_at seconds");
    stored["created_at"]["secs_since_epoch"] = serde_json::Value::from(secs - days * 24 * 3600);
    std::fs::write(&path, serde_json::to_vec(&stored).expect("encode")).expect("write");
}

/// Retention counted backups one by one, so the full backup under a chain
/// of incremental backups was deleted first and the chain could no longer
/// be restored.
#[tokio::test]
async fn test_retention_keeps_the_parents_of_kept_backups() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let config = BackupConfig {
        max_backups: 2,
        ..plain_config(temp_dir.path())
    };
    let manager = BackupRestoreManager::new(config.clone(), 1)
        .await
        .expect("manager");
    let full = manager
        .create_full_backup(vec![triple("s1")])
        .await
        .expect("full backup");
    let first = manager
        .create_incremental_backup(&full, vec![triple("s2")])
        .await
        .expect("first incremental");
    let second = manager
        .create_incremental_backup(&first, vec![triple("s3")])
        .await
        .expect("second incremental");
    let other = manager
        .create_full_backup(vec![triple("s4")])
        .await
        .expect("second full backup");

    let expected = vec![triple("s1"), triple("s2"), triple("s3")];
    let restored = manager
        .restore_backup(restore_options(&second, false))
        .await
        .expect("the newest chain must stay restorable");
    assert_eq!(restored, expected);

    let manager = BackupRestoreManager::new(config, 1)
        .await
        .expect("manager after restart");
    let restored = manager
        .restore_backup(restore_options(&second, false))
        .await
        .expect("the chain must stay restorable after a restart");
    assert_eq!(restored, expected);
    assert!(manager.get_backup_metadata(&other).await.is_some());
}

/// An expired full backup is still needed while a younger incremental
/// backup is built on it.
#[tokio::test]
async fn test_age_limit_keeps_a_parent_that_a_young_backup_needs() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let (full, incremental) = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        let full = manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("full backup");
        let incremental = manager
            .create_incremental_backup(&full, vec![triple("s2")])
            .await
            .expect("incremental backup");
        let metadata = manager.get_backup_metadata(&full).await.expect("metadata");
        (metadata, incremental)
    };
    age_backup(&full, 60);

    let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
        .await
        .expect("manager after restart");
    manager
        .create_full_backup(vec![triple("s3")])
        .await
        .expect("another full backup runs retention");
    let restored = manager
        .restore_backup(restore_options(&incremental, false))
        .await
        .expect("the incremental backup lost its expired parent");
    assert_eq!(restored, vec![triple("s1"), triple("s2")]);
    assert!(full.file_path.exists());
}

/// Retention used to run only after full backups.
#[tokio::test]
async fn test_retention_also_runs_after_incremental_backups() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let (expired, current) = {
        let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
            .await
            .expect("manager");
        let expired = manager
            .create_full_backup(vec![triple("old")])
            .await
            .expect("old full backup");
        let current = manager
            .create_full_backup(vec![triple("s1")])
            .await
            .expect("current full backup");
        let metadata = manager
            .get_backup_metadata(&expired)
            .await
            .expect("metadata");
        (metadata, current)
    };
    age_backup(&expired, 60);

    let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
        .await
        .expect("manager after restart");
    manager
        .create_incremental_backup(&current, vec![triple("s2")])
        .await
        .expect("incremental backup");
    assert!(
        manager.get_backup_metadata(&expired.id).await.is_none(),
        "an expired backup survived an incremental backup"
    );
    assert!(!expired.file_path.exists());
}

/// Deleting a backup that an incremental backup is built on would leave
/// that incremental backup impossible to restore.
#[tokio::test]
async fn test_deleting_a_parent_is_refused() {
    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let manager = BackupRestoreManager::new(plain_config(temp_dir.path()), 1)
        .await
        .expect("manager");
    let full = manager
        .create_full_backup(vec![triple("s1")])
        .await
        .expect("full backup");
    let incremental = manager
        .create_incremental_backup(&full, vec![triple("s2")])
        .await
        .expect("incremental backup");

    let error = manager
        .delete_backup(&full)
        .await
        .expect_err("a parent backup was deleted");
    assert!(error.to_string().contains(&incremental), "{error}");
    manager
        .restore_backup(restore_options(&incremental, false))
        .await
        .expect("the chain survives the refused delete");

    manager
        .delete_backup(&incremental)
        .await
        .expect("delete the child");
    manager.delete_backup(&full).await.expect("then the parent");
}
