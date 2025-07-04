use oxirs_chat::*;
use oxirs_core::{Store, ConcreteStore};
use std::sync::Arc;
use tempfile::TempDir;
use tokio;

#[tokio::test]
async fn test_session_creation_and_persistence() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");

    // Create a store for testing
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));

    // Create chat manager with persistence
    let manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Test session creation
    let session_id = "test_session_001".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Verify session exists
    assert_eq!(session.lock().await.id, session_id);

    // Add a message to the session
    {
        let mut session_guard = session.lock().await;
        let response = session_guard
            .process_message("Hello, this is a test message".to_string())
            .await
            .expect("Failed to process message");

        assert_eq!(response.role, MessageRole::Assistant);
        assert!(!response.content.is_empty());
    }

    // Test session statistics
    let stats = manager.get_session_stats().await;
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(stats.active_sessions, 1);

    // Test detailed metrics
    let metrics = manager.get_detailed_metrics().await;
    assert_eq!(metrics.total_sessions, 1);
    assert_eq!(metrics.active_sessions, 1);
    assert!(metrics.total_messages > 0);

    println!("✅ Session persistence test passed!");
}

#[tokio::test]
async fn test_session_backup_and_restore() {
    // Create temporary directories
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");
    let backup_path = temp_dir.path().join("test_backup");

    // Create store and manager
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));
    let mut manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Create a test session with some data
    let session_id = "backup_test_session".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Add some messages to make the session interesting
    {
        let mut session_guard = session.lock().await;
        session_guard
            .process_message("First test message".to_string())
            .await
            .expect("Failed to process first message");

        session_guard
            .process_message("Second test message".to_string())
            .await
            .expect("Failed to process second message");
    }

    // Test backup
    let backup_report = manager
        .backup_sessions(&backup_path)
        .await
        .expect("Failed to backup sessions");

    assert_eq!(backup_report.successful_backups, 1);
    assert_eq!(backup_report.failed_backups, 0);

    // Clear sessions (simulate fresh start)
    manager
        .remove_session(&session_id)
        .await
        .expect("Failed to remove session");

    // Test restore
    let restore_report = manager
        .restore_sessions(&backup_path)
        .await
        .expect("Failed to restore sessions");

    assert_eq!(restore_report.sessions_restored, 1);
    // Note: failed_restorations field not available in RestoreReport

    // Verify the restored session
    let restored_session = manager
        .get_session(&session_id)
        .await
        .expect("Restored session not found");

    let restored_session_guard = restored_session.expect("Restored session should exist").lock().await;
    assert_eq!(restored_session_guard.id, session_id);
    assert!(restored_session_guard.messages.len() >= 2); // Should have the messages we added

    println!("✅ Session backup and restore test passed!");
}

#[tokio::test]
async fn test_session_expiration_and_cleanup() {
    // Create temporary directory
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");

    // Create store and manager
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));
    let manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Create a session with short timeout
    let session_id = "expiration_test_session".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Manually expire the session by setting old timestamp
    {
        let mut session_guard = session.lock().await;
        session_guard.last_activity = chrono::Utc::now() - chrono::Duration::hours(2);
    }

    // Test cleanup
    let cleaned_count = manager
        .cleanup_expired_sessions()
        .await
        .expect("Failed to cleanup expired sessions");

    assert_eq!(cleaned_count, 1);

    // Verify session was removed
    assert!(manager.get_session(&session_id).await.is_none());

    println!("✅ Session expiration and cleanup test passed!");
}
