//! Encryption-at-rest flags must never leave plaintext on disk.
//!
//! Each storage path below used to accept an "encryption enabled" setting and
//! then write the data unencrypted. With encryption enabled, a component must
//! either refuse to write (no key available) or write only sealed bytes; in no
//! case may the plaintext marker appear in any file it created.

use oxirs_cluster::advanced_storage::{AdvancedStorageBackend, StorageConfig};
use oxirs_cluster::backup::{
    destination::DestinationConfig,
    policy::{CronSchedule, EncryptionConfig},
    BackupExecutor, BackupPolicy, RetentionTier,
};
use oxirs_cluster::backup_restore::{BackupConfig, BackupRestoreManager};
use oxirs_cluster::network::LogEntry;
use oxirs_cluster::raft::RdfCommand;
use oxirs_cluster::storage::RaftState;
use std::path::Path;

const MARKER: &[u8] = b"PLAINTEXT-MARKER";

/// Fail if any file under `dir` (recursively) contains [`MARKER`].
fn assert_no_plaintext(dir: &Path) {
    for entry in std::fs::read_dir(dir).expect("failed to list directory") {
        let path = entry.expect("failed to read directory entry").path();
        if path.is_dir() {
            assert_no_plaintext(&path);
            continue;
        }
        let bytes = std::fs::read(&path).expect("failed to read file");
        assert!(
            !bytes.windows(MARKER.len()).any(|w| w == MARKER),
            "{} contains plaintext although encryption is enabled",
            path.display()
        );
    }
}

#[test]
fn backup_policy_with_encryption_never_writes_plaintext() {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let policy = BackupPolicy {
        name: "encrypted".into(),
        schedule: CronSchedule::daily(),
        retention: RetentionTier::minimal(),
        gfs: None,
        encryption: EncryptionConfig::with_key("backup-key"),
        destination: DestinationConfig::Filesystem {
            path: dir.path().to_path_buf(),
        },
    };

    let _ = BackupExecutor::new().execute_backup(&policy, b"PLAINTEXT-MARKER data", dir.path());

    assert_no_plaintext(dir.path());
}

#[tokio::test]
async fn backup_restore_with_encryption_never_writes_plaintext() {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let config = BackupConfig {
        backup_dir: dir.path().to_path_buf(),
        enable_encryption: true,
        enable_compression: false,
        verify_after_backup: false,
        ..Default::default()
    };

    if let Ok(manager) = BackupRestoreManager::new(config, 1).await {
        let triples = vec![(
            "PLAINTEXT-MARKER".to_string(),
            "p".to_string(),
            "o".to_string(),
        )];
        let _ = manager.create_full_backup(triples).await;
    }

    assert_no_plaintext(dir.path());
}

#[tokio::test]
async fn storage_with_encryption_never_writes_plaintext() {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let config = StorageConfig {
        data_dir: dir.path().to_path_buf(),
        enable_encryption: true,
        encryption_key: Some([7u8; 32]),
        enable_compression: false,
        ..Default::default()
    };

    // A key is supplied, so the backend must start and write real data.
    let storage = AdvancedStorageBackend::new(config)
        .await
        .expect("storage with a key must start");
    let state = RaftState {
        current_term: 3,
        voted_for: Some(1),
        log: vec![LogEntry::new(
            1,
            3,
            RdfCommand::Insert {
                subject: "PLAINTEXT-MARKER".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
            },
        )],
        commit_index: 1,
        last_applied: 1,
    };
    storage
        .store_raft_state(state)
        .await
        .expect("store_raft_state");
    storage
        .create_snapshot(1, 3, vec![1], b"PLAINTEXT-MARKER snapshot")
        .await
        .expect("create_snapshot");

    assert_no_plaintext(dir.path());
}
