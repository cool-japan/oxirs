//! Tests for PersistentStorage

use super::config::StorageConfig;
use super::error::StorageError;
use super::persistent::PersistentStorage;

use crate::network::LogEntry;
use crate::raft::RdfCommand;

use tempfile::TempDir;

async fn create_test_storage() -> (PersistentStorage, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        sync_writes: false,
        max_log_entries: 100,
        compress_snapshots: false,
        backup_retention: 2,
        enable_corruption_detection: false,
        enable_crash_recovery: false,
        enable_wal: false,
    };
    let node_id = std::process::id() as u64
        + std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    let storage = PersistentStorage::new(node_id, config).await.unwrap();
    (storage, temp_dir)
}

#[tokio::test]
async fn test_storage_creation() {
    let (storage, _temp_dir) = create_test_storage().await;
    assert!(storage.node_id > 0);
    assert_eq!(storage.get_current_term().await, 0);
    assert_eq!(storage.get_voted_for().await, None);
}

#[tokio::test]
async fn test_term_operations() {
    let (storage, _temp_dir) = create_test_storage().await;

    storage.set_current_term(5).await.unwrap();
    assert_eq!(storage.get_current_term().await, 5);

    assert_eq!(storage.get_voted_for().await, None);
}

#[tokio::test]
async fn test_vote_operations() {
    let (storage, _temp_dir) = create_test_storage().await;

    storage.set_voted_for(Some(2)).await.unwrap();
    assert_eq!(storage.get_voted_for().await, Some(2));

    storage.set_voted_for(None).await.unwrap();
    assert_eq!(storage.get_voted_for().await, None);
}

#[tokio::test]
async fn test_log_operations() {
    let (storage, _temp_dir) = create_test_storage().await;

    let command = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    let entry = LogEntry::new(1, 1, command);

    storage.append_entries(vec![entry.clone()]).await.unwrap();
    assert_eq!(storage.get_last_log_index().await, 1);
    assert_eq!(storage.get_last_log_term().await, 1);

    let retrieved = storage.get_log_entry(1).await.unwrap();
    assert_eq!(retrieved.index, 1);
    assert_eq!(retrieved.term, 1);

    let entries = storage.get_log_entries(1, 2).await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].index, 1);
}

#[tokio::test]
async fn test_commit_operations() {
    let (storage, _temp_dir) = create_test_storage().await;

    storage.set_commit_index(5).await.unwrap();
    assert_eq!(storage.get_commit_index().await, 5);

    storage.set_last_applied(3).await.unwrap();
    assert_eq!(storage.get_last_applied().await, 3);
}

#[tokio::test]
async fn test_application_state() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        sync_writes: false,
        max_log_entries: 100,
        compress_snapshots: false,
        backup_retention: 2,
        enable_corruption_detection: false,
        enable_crash_recovery: false,
        enable_wal: false,
    };
    let node_id = std::process::id() as u64
        + std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

    println!("Creating storage...");
    let storage = PersistentStorage::new(node_id, config).await.unwrap();
    println!("Storage created");

    println!("Getting app state...");
    let app_state = storage.get_app_state().await;
    println!("App state retrieved, length: {}", app_state.len());
    assert_eq!(app_state.len(), 0);

    println!("Modifying app state directly...");
    {
        let mut app_state = storage.app_state.write();
        println!("Got write lock");
        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        app_state.apply_command(&command);
        println!("Applied command to in-memory state");
    }

    println!("Getting updated app state...");
    let app_state = storage.get_app_state().await;
    println!("Updated app state retrieved, length: {}", app_state.len());
    assert_eq!(app_state.len(), 1);
    println!("Test completed successfully");
}

#[tokio::test]
async fn test_log_truncation() {
    let (storage, _temp_dir) = create_test_storage().await;

    for i in 1..=5 {
        let command = RdfCommand::Insert {
            subject: format!("s{i}"),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let entry = LogEntry::new(i, 1, command);
        storage.append_entries(vec![entry]).await.unwrap();
    }

    assert_eq!(storage.get_last_log_index().await, 5);

    storage.truncate_log(3).await.unwrap();
    assert_eq!(storage.get_last_log_index().await, 2);

    assert!(storage.get_log_entry(3).await.is_none());
    assert!(storage.get_log_entry(4).await.is_none());
    assert!(storage.get_log_entry(5).await.is_none());

    assert!(storage.get_log_entry(1).await.is_some());
    assert!(storage.get_log_entry(2).await.is_some());
}

#[tokio::test]
async fn test_snapshot_operations() {
    let (storage, _temp_dir) = create_test_storage().await;

    {
        let mut app_state = storage.app_state.write();
        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        app_state.apply_command(&command);
    }

    let metadata = storage.create_snapshot().await.unwrap();
    assert!(metadata.size > 0);

    let loaded = storage.load_snapshot().await.unwrap();
    assert!(loaded.is_some());

    let (loaded_metadata, loaded_state) = loaded.unwrap();
    assert_eq!(loaded_metadata.size, metadata.size);
    assert_eq!(loaded_state.len(), 1);
}

#[tokio::test]
async fn test_compaction_check() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        max_log_entries: 3,
        ..Default::default()
    };
    let storage = PersistentStorage::new(1, config).await.unwrap();

    assert!(!storage.needs_compaction().await);

    for i in 1..=5 {
        let command = RdfCommand::Insert {
            subject: format!("s{i}"),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let entry = LogEntry::new(i, 1, command);
        storage.append_entries(vec![entry]).await.unwrap();
    }

    assert!(storage.needs_compaction().await);
}

#[tokio::test]
async fn test_storage_stats() {
    let (storage, _temp_dir) = create_test_storage().await;

    let stats = storage.get_stats().await;
    assert!(stats.node_id > 0);
    assert_eq!(stats.log_entries, 0);
    assert_eq!(stats.current_term, 0);
    assert_eq!(stats.triple_count, 0);
}

#[tokio::test]
async fn test_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        sync_writes: false,
        enable_corruption_detection: false,
        enable_crash_recovery: false,
        enable_wal: false,
        ..Default::default()
    };

    {
        let storage = PersistentStorage::new(1, config.clone()).await.unwrap();
        storage.set_current_term(5).await.unwrap();
        storage.set_voted_for(Some(2)).await.unwrap();

        let command = RdfCommand::Insert {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
        };
        let entry = LogEntry::new(1, 1, command.clone());
        storage.append_entries(vec![entry]).await.unwrap();

        {
            let mut app_state = storage.app_state.write();
            app_state.apply_command(&command);
        }
    }

    {
        let storage = PersistentStorage::new(1, config).await.unwrap();
        assert_eq!(storage.get_current_term().await, 5);
        assert_eq!(storage.get_voted_for().await, Some(2));
        assert_eq!(storage.get_last_log_index().await, 1);
    }
}

#[test]
fn test_storage_error_display() {
    let err = StorageError::Corruption {
        file: "log.dat".to_string(),
        message: "checksum mismatch".to_string(),
    };
    assert!(err
        .to_string()
        .contains("Corruption detected in log.dat: checksum mismatch"));

    let err = StorageError::LogEntryNotFound { index: 42 };
    assert!(err.to_string().contains("Log entry not found at index 42"));

    let err = StorageError::InvalidRange { start: 10, end: 5 };
    assert!(err.to_string().contains("Invalid log range: 10 to 5"));
}

// ---------------------------------------------------------------------------
// Durability: the WAL, interrupted saves and snapshots
// ---------------------------------------------------------------------------

use super::persistent::StorageBackend;
use super::types::{RaftState, WalEntry, WalOperation};
use oxirs_core::model::{Literal, NamedNode, Triple};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn durable_config(dir: &TempDir) -> StorageConfig {
    StorageConfig {
        data_dir: dir.path().to_string_lossy().to_string(),
        ..Default::default()
    }
}

fn node_dir(dir: &TempDir, node_id: u64) -> PathBuf {
    dir.path().join(format!("node-{node_id}"))
}

fn wal_file(dir: &TempDir, node_id: u64) -> PathBuf {
    node_dir(dir, node_id).join("wal.log")
}

fn insert(subject: &str) -> RdfCommand {
    RdfCommand::Insert {
        subject: subject.to_string(),
        predicate: "http://example.org/p".to_string(),
        object: "o".to_string(),
    }
}

fn sample_triple() -> Triple {
    Triple::new(
        NamedNode::new("http://example.org/s").expect("IRI"),
        NamedNode::new("http://example.org/p").expect("IRI"),
        Literal::new_simple_literal("o"),
    )
}

/// Change the application state of `storage`, as a committed command would.
async fn apply(storage: &PersistentStorage, subject: &str) {
    storage
        .apply_command(&insert(subject))
        .await
        .expect("apply_command");
}

/// One WAL record in the on-disk format: `u64 LE length | oxicode WalEntry`.
fn wal_record(sequence: u64, operation: WalOperation) -> Vec<u8> {
    let timestamp: u64 = 1_700_000_000;
    let op_bytes = oxicode::serde::encode_to_vec(&operation, oxicode::config::standard())
        .expect("encode operation");
    let mut hasher = Sha256::new();
    hasher.update(&op_bytes);
    hasher.update(sequence.to_le_bytes());
    hasher.update(timestamp.to_le_bytes());
    let entry = WalEntry {
        sequence,
        timestamp,
        operation,
        checksum: hex::encode(hasher.finalize()),
    };
    let body =
        oxicode::serde::encode_to_vec(&entry, oxicode::config::standard()).expect("encode entry");
    let mut record = (body.len() as u64).to_le_bytes().to_vec();
    record.extend(body);
    record
}

fn term_state(term: u64) -> WalOperation {
    WalOperation::WriteRaftState(RaftState {
        current_term: term,
        ..Default::default()
    })
}

fn append_to(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open WAL");
    file.write_all(bytes).expect("append to WAL");
}

fn torn_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .expect("list data dir")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("wal.log.torn-"))
        })
        .collect()
}

/// Every state-machine write held the write lock on the application state
/// and then asked for a read lock on it to save it, so it never returned.
async fn state_machine_writes_return() {
    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    let before = {
        let storage = PersistentStorage::new(5, config.clone()).await.unwrap();
        let writes = async {
            storage
                .apply_command(&insert("http://example.org/a"))
                .await?;
            storage.create_shard(0).await?;
            storage.insert_triple_to_shard(0, sample_triple()).await?;
            storage
                .delete_triple_from_shard(0, &sample_triple())
                .await?;
            storage.import_shard(1, vec![sample_triple()]).await?;
            storage
                .insert_triples_to_shard(1, vec![sample_triple()])
                .await?;
            storage.mark_shard_for_deletion(1).await?;
            storage.delete_shard(0).await
        };
        tokio::time::timeout(Duration::from_secs(10), writes)
            .await
            .expect("a state-machine write never returned")
            .expect("state-machine writes succeed");
        storage.get_app_state().await
    };
    assert_eq!(before.len(), 1);
    assert_eq!(before.shards.len(), 1);
    let storage = PersistentStorage::new(5, config).await.unwrap();
    assert_eq!(
        storage.get_app_state().await,
        before,
        "the writes were not persisted"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_state_machine_writes_return_on_a_current_thread_runtime() {
    state_machine_writes_return().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_state_machine_writes_return_on_a_multi_thread_runtime() {
    state_machine_writes_return().await;
}

/// The WAL was opened and then unlinked during startup, so every journal
/// write went to a file that no longer existed.
#[tokio::test]
async fn test_wal_is_kept_on_disk() {
    let temp_dir = TempDir::new().unwrap();
    let storage = PersistentStorage::new(4, durable_config(&temp_dir))
        .await
        .unwrap();
    storage.set_current_term(2).await.unwrap();
    let wal = wal_file(&temp_dir, 4);
    assert!(wal.exists(), "the WAL is not on disk");
    assert!(
        std::fs::metadata(&wal).unwrap().len() > 0,
        "nothing was journaled"
    );
}

/// A write that reached the WAL but not the state file (a crash in between)
/// is redone on restart. Recovery used to parse the binary WAL as JSON, fail,
/// and delete it.
#[tokio::test]
async fn test_interrupted_save_is_redone_on_restart() {
    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    {
        let storage = PersistentStorage::new(3, config.clone()).await.unwrap();
        storage.set_current_term(1).await.unwrap();
    }
    append_to(&wal_file(&temp_dir, 3), &wal_record(1000, term_state(7)));

    let storage = PersistentStorage::new(3, config).await.unwrap();
    assert_eq!(
        storage.get_current_term().await,
        7,
        "the journaled write was not redone"
    );
}

/// A record cut short by a crash is set aside, and the complete records
/// before it are still redone.
#[tokio::test]
async fn test_torn_wal_tail_is_set_aside() {
    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    {
        let storage = PersistentStorage::new(6, config.clone()).await.unwrap();
        storage.set_current_term(1).await.unwrap();
    }
    let wal = wal_file(&temp_dir, 6);
    append_to(&wal, &wal_record(1000, term_state(7)));
    append_to(&wal, &[0xE8, 0x03, 0, 0, 0, 0, 0, 0, 1, 2, 3]);

    let storage = PersistentStorage::new(6, config).await.unwrap();
    assert_eq!(storage.get_current_term().await, 7);
    assert_eq!(torn_files(&node_dir(&temp_dir, 6)).len(), 1);
}

/// Damage in the middle of the WAL is not a torn tail: refusing to start is
/// better than silently dropping the records after it.
#[tokio::test]
async fn test_mid_log_wal_damage_is_an_error() {
    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    {
        let storage = PersistentStorage::new(8, config.clone()).await.unwrap();
        storage.set_current_term(1).await.unwrap();
    }
    let mut damaged = wal_record(1000, term_state(7));
    let last = damaged.len() - 1;
    damaged[last] ^= 0x01;
    let wal = wal_file(&temp_dir, 8);
    append_to(&wal, &damaged);
    append_to(&wal, &wal_record(1001, term_state(8)));

    assert!(
        PersistentStorage::new(8, config).await.is_err(),
        "a WAL with damage in the middle was accepted"
    );
}

/// A save that fails changes nothing: not the state in memory, not the WAL,
/// and nothing is redone on restart.
#[cfg(unix)]
#[tokio::test]
async fn test_failed_save_changes_nothing() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    let dir = node_dir(&temp_dir, 9);
    {
        let storage = PersistentStorage::new(9, config.clone()).await.unwrap();
        storage.set_current_term(2).await.unwrap();
        let wal_len = std::fs::metadata(wal_file(&temp_dir, 9)).unwrap().len();

        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let outcome = storage.set_current_term(5).await;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();

        assert!(outcome.is_err(), "the save could not have succeeded");
        assert_eq!(
            storage.get_current_term().await,
            2,
            "a failed save changed the state"
        );
        assert_eq!(
            std::fs::metadata(wal_file(&temp_dir, 9)).unwrap().len(),
            wal_len,
            "a failed save left its record in the WAL"
        );
    }
    let storage = PersistentStorage::new(9, config).await.unwrap();
    assert_eq!(storage.get_current_term().await, 2);
}

/// With crash recovery disabled the WAL is still kept well-formed: a torn
/// tail is set aside, and compaction and verification still work.
#[tokio::test]
async fn test_torn_tail_without_crash_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        enable_crash_recovery: false,
        ..durable_config(&temp_dir)
    };
    {
        let storage = PersistentStorage::new(10, config.clone()).await.unwrap();
        storage.set_current_term(3).await.unwrap();
    }
    append_to(
        &wal_file(&temp_dir, 10),
        &[0xE8, 0x03, 0, 0, 0, 0, 0, 0, 1, 2],
    );

    let storage = PersistentStorage::new(10, config).await.unwrap();
    assert_eq!(torn_files(&node_dir(&temp_dir, 10)).len(), 1);
    storage.set_current_term(4).await.unwrap();
    storage.compact_wal().await.expect("compaction");
    assert!(storage.verify_integrity().await.expect("verification"));
    assert_eq!(storage.get_current_term().await, 4);
}

/// The snapshot file the metadata points at.
fn snapshot_data_file(dir: &Path) -> PathBuf {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("list data dir")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("snapshot")
                        && name != "snapshot_metadata.json"
                        && !name.ends_with(".checksum")
                        && !name.ends_with(".tmp")
                })
        })
        .collect();
    assert_eq!(
        files.len(),
        1,
        "expected exactly one snapshot file: {files:?}"
    );
    files.remove(0)
}

/// After a second snapshot, a restart used to judge the snapshot file by a
/// checksum recorded at the first restart, call it corrupted and replace it
/// with an older copy from a backup.
#[tokio::test]
async fn test_snapshot_survives_a_restart_after_a_backup() {
    let temp_dir = TempDir::new().unwrap();
    let config = durable_config(&temp_dir);
    {
        let storage = PersistentStorage::new(7, config.clone()).await.unwrap();
        apply(&storage, "http://example.org/a").await;
        storage.create_snapshot().await.unwrap();
        storage.backup().await.unwrap();
    }
    let expected = {
        let storage = PersistentStorage::new(7, config.clone()).await.unwrap();
        apply(&storage, "http://example.org/b").await;
        apply(&storage, "http://example.org/c").await;
        storage.create_snapshot().await.unwrap();
        storage.get_app_state().await.len()
    };
    let storage = PersistentStorage::new(7, config).await.unwrap();
    let (_, snapshot) = storage
        .load_snapshot()
        .await
        .unwrap()
        .expect("a snapshot exists");
    assert_eq!(snapshot.len(), expected, "the snapshot was rolled back");
}

/// The snapshot checksum is recorded but was never checked on load.
#[tokio::test]
async fn test_tampered_snapshot_is_refused() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        compress_snapshots: false,
        ..durable_config(&temp_dir)
    };
    let storage = PersistentStorage::new(11, config).await.unwrap();
    apply(&storage, "http://example.org/s").await;
    storage.create_snapshot().await.unwrap();

    let path = snapshot_data_file(&node_dir(&temp_dir, 11));
    let original = std::fs::read(&path).unwrap();
    let text = String::from_utf8(original).expect("an uncompressed snapshot is JSON");
    assert!(text.contains("example.org/s"));
    std::fs::write(&path, text.replace("example.org/s", "example.org/x")).unwrap();

    assert!(
        storage.load_snapshot().await.is_err(),
        "a tampered snapshot was loaded"
    );
}

/// `compress_snapshots` (on by default) had no effect.
#[tokio::test]
async fn test_snapshot_compression_setting_is_applied() {
    let temp_dir = TempDir::new().unwrap();
    let storage = PersistentStorage::new(12, durable_config(&temp_dir))
        .await
        .unwrap();
    for subject in ["a", "b", "c"] {
        apply(&storage, &format!("http://example.org/{subject}")).await;
    }
    storage.create_snapshot().await.unwrap();

    let bytes = std::fs::read(snapshot_data_file(&node_dir(&temp_dir, 12))).unwrap();
    assert!(
        bytes.starts_with(&[0x04, 0x22, 0x4D, 0x18]),
        "the snapshot was not compressed"
    );
    let (_, snapshot) = storage.load_snapshot().await.unwrap().expect("snapshot");
    assert_eq!(snapshot.len(), 3);
}

/// Releases up to 0.4.1 with crash recovery disabled kept appending after a
/// torn record, which leaves damage in the middle of the log. The log is
/// never replayed in that mode, so the damage is set aside and the store
/// opens.
#[tokio::test]
async fn test_mid_log_damage_without_crash_recovery_is_set_aside() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig {
        enable_crash_recovery: false,
        ..durable_config(&temp_dir)
    };
    {
        let storage = PersistentStorage::new(13, config.clone()).await.unwrap();
        storage.set_current_term(3).await.unwrap();
    }
    let wal = wal_file(&temp_dir, 13);
    // A torn record, whose length claims more than was written, and a later
    // append behind it.
    append_to(&wal, &[50, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]);
    append_to(&wal, &wal_record(1000, term_state(9)));

    let storage = PersistentStorage::new(13, config)
        .await
        .expect("the store must open");
    assert_eq!(torn_files(&node_dir(&temp_dir, 13)).len(), 1);
    assert_eq!(
        storage.get_current_term().await,
        3,
        "a log that is never replayed changed the state"
    );
    storage.set_current_term(4).await.unwrap();
    assert!(storage.verify_integrity().await.unwrap());
}
