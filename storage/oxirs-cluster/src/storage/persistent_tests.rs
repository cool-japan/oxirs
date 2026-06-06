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
        let mut app_state = storage.app_state.write().await;
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
        let mut app_state = storage.app_state.write().await;
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
            let mut app_state = storage.app_state.write().await;
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
