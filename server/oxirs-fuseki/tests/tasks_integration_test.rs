//! Async Task Management Integration Tests
//!
//! Tests for long-running task tracking and management

use oxirs_fuseki::handlers::tasks::{TaskManager, TaskStatus, TaskType};
use std::sync::Arc;
use std::time::Duration;

/// Test creating a new task
#[tokio::test]
async fn test_create_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Upload, "Test upload task".to_string())
        .unwrap();

    assert!(!task.id.is_empty());
    assert_eq!(task.status, TaskStatus::Pending);
    assert_eq!(task.task_type, TaskType::Upload);
    assert_eq!(task.description, "Test upload task");
}

/// Test listing all tasks
#[tokio::test]
async fn test_list_tasks() {
    let manager = Arc::new(TaskManager::new());

    // Create multiple tasks
    manager
        .create_task(TaskType::Upload, "Upload 1".to_string())
        .unwrap();
    manager
        .create_task(TaskType::Query, "Query 1".to_string())
        .unwrap();
    manager
        .create_task(TaskType::Update, "Update 1".to_string())
        .unwrap();

    let tasks = manager.list_tasks(None).unwrap();
    assert_eq!(tasks.len(), 3);
}

/// Test getting specific task by ID
#[tokio::test]
async fn test_get_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Patch, "Test patch".to_string())
        .unwrap();
    let task_id = task.id.clone();

    let retrieved = manager.get_task(&task_id).unwrap();
    assert!(retrieved.is_some());

    let retrieved_task = retrieved.unwrap();
    assert_eq!(retrieved_task.id, task_id);
    assert_eq!(retrieved_task.task_type, TaskType::Patch);
}

/// Test getting non-existent task returns None
#[tokio::test]
async fn test_get_nonexistent_task() {
    let manager = Arc::new(TaskManager::new());

    let result = manager.get_task("nonexistent-id").unwrap();
    assert!(result.is_none());
}

/// Test updating task status
#[tokio::test]
async fn test_update_task_status() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Query, "Test query".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Update to running
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.status, TaskStatus::Running);
}

/// Test updating task progress
#[tokio::test]
async fn test_update_task_progress() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Upload, "Test upload".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Start task and update progress
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
            t.update_progress(25.0);
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.progress, Some(25.0));

    // Update to 75%
    manager
        .update_task(&task_id, |t| {
            t.update_progress(75.0);
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.progress, Some(75.0));
}

/// Test completing task successfully
#[tokio::test]
async fn test_complete_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Validation, "Test validation".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Complete task
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
            t.mark_completed(Some(serde_json::json!({"valid": true, "count": 100})));
        })
        .unwrap();

    let completed = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(completed.status, TaskStatus::Completed);
    assert_eq!(completed.progress, Some(100.0));
    assert!(completed.result.is_some());
    assert!(completed.completed_at.is_some());
}

/// Test marking task as failed
#[tokio::test]
async fn test_fail_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Update, "Test update".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Fail task
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
            t.mark_failed("Connection timeout".to_string());
        })
        .unwrap();

    let failed = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(failed.status, TaskStatus::Failed);
    assert_eq!(failed.error, Some("Connection timeout".to_string()));
    assert!(failed.completed_at.is_some());
}

/// Test cancelling task
#[tokio::test]
async fn test_cancel_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Query, "Long query".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Start task
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
        })
        .unwrap();

    // Cancel it
    manager.cancel_task(&task_id).unwrap();

    let cancelled = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(cancelled.status, TaskStatus::Cancelled);
    assert!(cancelled.completed_at.is_some());
}

/// Test deleting task
#[tokio::test]
async fn test_delete_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Custom, "Custom task".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Verify it exists
    assert!(manager.get_task(&task_id).unwrap().is_some());

    // Delete it
    let deleted = manager.delete_task(&task_id).unwrap();
    assert!(deleted);

    // Verify it's gone
    assert!(manager.get_task(&task_id).unwrap().is_none());
}

/// Test deleting non-existent task returns false
#[tokio::test]
async fn test_delete_nonexistent_task() {
    let manager = Arc::new(TaskManager::new());

    let deleted = manager.delete_task("nonexistent-id").unwrap();
    assert!(!deleted);
}

/// Test deleting running task cancels it first
#[tokio::test]
async fn test_delete_running_task() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Upload, "Upload task".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Start task
    manager
        .update_task(&task_id, |t| {
            t.mark_running();
        })
        .unwrap();

    // Delete (should cancel first)
    let deleted = manager.delete_task(&task_id).unwrap();
    assert!(deleted);

    // Verify it's gone
    assert!(manager.get_task(&task_id).unwrap().is_none());
}

/// Test task statistics
#[tokio::test]
async fn test_task_statistics() {
    let manager = Arc::new(TaskManager::new());

    // Create various tasks
    let task1 = manager
        .create_task(TaskType::Upload, "Upload 1".to_string())
        .unwrap();
    let task2 = manager
        .create_task(TaskType::Query, "Query 1".to_string())
        .unwrap();
    let task3 = manager
        .create_task(TaskType::Update, "Update 1".to_string())
        .unwrap();

    // Update statuses
    manager
        .update_task(&task1.id, |t| t.mark_running())
        .unwrap();
    manager
        .update_task(&task2.id, |t| t.mark_completed(None))
        .unwrap();
    manager
        .update_task(&task3.id, |t| t.mark_failed("Error".to_string()))
        .unwrap();

    let stats = manager.get_statistics().unwrap();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.running, 1);
    assert_eq!(stats.completed, 1);
    assert_eq!(stats.failed, 1);
}

/// Test filtering tasks by status
#[tokio::test]
async fn test_list_tasks_filtered() {
    let manager = Arc::new(TaskManager::new());

    let task1 = manager
        .create_task(TaskType::Upload, "Upload 1".to_string())
        .unwrap();
    let task2 = manager
        .create_task(TaskType::Query, "Query 1".to_string())
        .unwrap();
    let task3 = manager
        .create_task(TaskType::Update, "Update 1".to_string())
        .unwrap();

    // Set different statuses
    manager
        .update_task(&task1.id, |t| t.mark_running())
        .unwrap();
    manager
        .update_task(&task2.id, |t| t.mark_running())
        .unwrap();
    manager
        .update_task(&task3.id, |t| t.mark_completed(None))
        .unwrap();

    // Filter running tasks
    let running_tasks = manager.list_tasks(Some(TaskStatus::Running)).unwrap();
    assert_eq!(running_tasks.len(), 2);

    // Filter completed tasks
    let completed_tasks = manager.list_tasks(Some(TaskStatus::Completed)).unwrap();
    assert_eq!(completed_tasks.len(), 1);
}

/// Test concurrent task creation
#[tokio::test]
async fn test_concurrent_task_creation() {
    use tokio::task;

    let manager = Arc::new(TaskManager::new());
    let mut handles = vec![];

    // Create 10 tasks concurrently
    for i in 0..10 {
        let manager_clone = manager.clone();
        let handle = task::spawn(async move {
            manager_clone
                .create_task(TaskType::Upload, format!("Upload {}", i))
                .unwrap()
        });
        handles.push(handle);
    }

    // Wait for all
    for handle in handles {
        let task = handle.await.unwrap();
        assert!(!task.id.is_empty());
    }

    // Verify all tasks created
    let tasks = manager.list_tasks(None).unwrap();
    assert_eq!(tasks.len(), 10);
}

/// Test concurrent task updates
#[tokio::test]
async fn test_concurrent_task_updates() {
    use tokio::task;

    let manager = Arc::new(TaskManager::new());

    // Create a task
    let test_task = manager
        .create_task(TaskType::Query, "Shared task".to_string())
        .unwrap();
    let task_id = test_task.id.clone();

    let mut handles = vec![];

    // Update progress from multiple threads
    for i in 0..5 {
        let manager_clone = manager.clone();
        let task_id_clone = task_id.clone();
        let handle = task::spawn(async move {
            manager_clone
                .update_task(&task_id_clone, |t| {
                    t.update_progress((i * 20) as f32);
                })
                .unwrap();
        });
        handles.push(handle);
    }

    // Wait for all updates
    for handle in handles {
        handle.await.unwrap();
    }

    // Task should still be valid
    let task = manager.get_task(&task_id).unwrap();
    assert!(task.is_some());
}

/// Test task limit enforcement
#[tokio::test]
async fn test_task_limit() {
    let manager = Arc::new(TaskManager::with_config(5, Duration::from_secs(3600)));

    // Create 5 tasks (at limit)
    for i in 0..5 {
        manager
            .create_task(TaskType::Upload, format!("Task {}", i))
            .unwrap();
    }

    // 6th task should fail
    let result = manager.create_task(TaskType::Upload, "Task 6".to_string());
    assert!(result.is_err());
}

/// Test old task cleanup
#[tokio::test]
async fn test_old_task_cleanup() {
    let manager = Arc::new(TaskManager::with_config(100, Duration::from_millis(100)));

    // Create and complete a task
    let task = manager
        .create_task(TaskType::Upload, "Old task".to_string())
        .unwrap();
    let task_id = task.id.clone();

    manager
        .update_task(&task_id, |t| t.mark_completed(None))
        .unwrap();

    // Wait for task to become old
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Cleanup should remove it
    let removed = manager.cleanup_old_tasks().unwrap();
    assert_eq!(removed, 1);

    // Task should be gone
    assert!(manager.get_task(&task_id).unwrap().is_none());
}

/// Test task metadata
#[tokio::test]
async fn test_task_metadata() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Custom, "Task with metadata".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Add metadata
    manager
        .update_task(&task_id, |t| {
            t.metadata.insert("key1".to_string(), "value1".to_string());
            t.metadata.insert("key2".to_string(), "value2".to_string());
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.metadata.get("key1"), Some(&"value1".to_string()));
    assert_eq!(updated.metadata.get("key2"), Some(&"value2".to_string()));
}

/// Test task duration calculation
#[tokio::test]
async fn test_task_duration() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Query, "Timed task".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Start task
    manager.update_task(&task_id, |t| t.mark_running()).unwrap();

    // Wait a bit
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Complete task
    manager
        .update_task(&task_id, |t| t.mark_completed(None))
        .unwrap();

    let completed = manager.get_task(&task_id).unwrap().unwrap();
    let duration = completed.duration();
    assert!(duration.is_some());
    assert!(duration.unwrap().as_millis() >= 50);
}

/// Test progress clamping
#[tokio::test]
async fn test_progress_clamping() {
    let manager = Arc::new(TaskManager::new());

    let task = manager
        .create_task(TaskType::Upload, "Progress test".to_string())
        .unwrap();
    let task_id = task.id.clone();

    // Try to set progress > 100
    manager
        .update_task(&task_id, |t| {
            t.update_progress(150.0);
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.progress, Some(100.0)); // Clamped to 100

    // Try to set progress < 0
    manager
        .update_task(&task_id, |t| {
            t.update_progress(-50.0);
        })
        .unwrap();

    let updated = manager.get_task(&task_id).unwrap().unwrap();
    assert_eq!(updated.progress, Some(0.0)); // Clamped to 0
}

/// Test task types
#[tokio::test]
async fn test_all_task_types() {
    let manager = Arc::new(TaskManager::new());

    let types = vec![
        TaskType::Upload,
        TaskType::Query,
        TaskType::Update,
        TaskType::Patch,
        TaskType::Validation,
        TaskType::Custom,
    ];

    for task_type in types {
        let task = manager
            .create_task(task_type.clone(), format!("Test {:?}", task_type))
            .unwrap();
        assert_eq!(task.task_type, task_type);
    }

    let tasks = manager.list_tasks(None).unwrap();
    assert_eq!(tasks.len(), 6);
}

/// Test sorting tasks by creation time
#[tokio::test]
async fn test_task_sorting() {
    let manager = Arc::new(TaskManager::new());

    // Create tasks with delays
    let task1 = manager
        .create_task(TaskType::Upload, "Task 1".to_string())
        .unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;

    let task2 = manager
        .create_task(TaskType::Query, "Task 2".to_string())
        .unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;

    let task3 = manager
        .create_task(TaskType::Update, "Task 3".to_string())
        .unwrap();

    let tasks = manager.list_tasks(None).unwrap();

    // Should be sorted newest first
    assert_eq!(tasks[0].id, task3.id);
    assert_eq!(tasks[1].id, task2.id);
    assert_eq!(tasks[2].id, task1.id);
}
