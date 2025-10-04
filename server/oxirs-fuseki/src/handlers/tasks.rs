//! Async Task Management Handler
//!
//! Provides REST API for managing long-running asynchronous tasks.
//! Based on Apache Jena Fuseki's async task management.
//!
//! GET /$/tasks - List all tasks
//! GET /$/tasks/{id} - Get specific task status
//! POST /$/tasks - Create new async task
//! DELETE /$/tasks/{id} - Cancel/delete task

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Task type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Upload,
    Query,
    Update,
    Patch,
    Validation,
    Custom,
}

/// Task metadata and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,

    /// Task type
    #[serde(rename = "type")]
    pub task_type: TaskType,

    /// Current status
    pub status: TaskStatus,

    /// Task description
    pub description: String,

    /// Progress percentage (0-100)
    pub progress: Option<f32>,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Last update timestamp
    pub updated_at: SystemTime,

    /// Completion timestamp
    pub completed_at: Option<SystemTime>,

    /// Error message if failed
    pub error: Option<String>,

    /// Result data (JSON)
    pub result: Option<serde_json::Value>,

    /// Task metadata (custom fields)
    pub metadata: HashMap<String, String>,
}

impl Task {
    /// Create new pending task
    pub fn new(task_type: TaskType, description: String) -> Self {
        let now = SystemTime::now();
        Self {
            id: Uuid::new_v4().to_string(),
            task_type,
            status: TaskStatus::Pending,
            description,
            progress: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            error: None,
            result: None,
            metadata: HashMap::new(),
        }
    }

    /// Mark task as running
    pub fn mark_running(&mut self) {
        self.status = TaskStatus::Running;
        self.updated_at = SystemTime::now();
    }

    /// Update task progress
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = Some(progress.clamp(0.0, 100.0));
        self.updated_at = SystemTime::now();
    }

    /// Mark task as completed with result
    pub fn mark_completed(&mut self, result: Option<serde_json::Value>) {
        self.status = TaskStatus::Completed;
        self.result = result;
        self.progress = Some(100.0);
        let now = SystemTime::now();
        self.updated_at = now;
        self.completed_at = Some(now);
    }

    /// Mark task as failed with error
    pub fn mark_failed(&mut self, error: String) {
        self.status = TaskStatus::Failed;
        self.error = Some(error);
        let now = SystemTime::now();
        self.updated_at = now;
        self.completed_at = Some(now);
    }

    /// Mark task as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = TaskStatus::Cancelled;
        let now = SystemTime::now();
        self.updated_at = now;
        self.completed_at = Some(now);
    }

    /// Check if task is terminal (completed, failed, or cancelled)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
        )
    }

    /// Get task duration
    pub fn duration(&self) -> Option<Duration> {
        self.completed_at
            .and_then(|completed| completed.duration_since(self.created_at).ok())
    }
}

/// Task manager for tracking and executing async tasks
#[derive(Clone)]
pub struct TaskManager {
    tasks: Arc<RwLock<HashMap<String, Task>>>,
    handles: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
    max_tasks: usize,
    max_age: Duration,
}

impl TaskManager {
    /// Create new task manager
    pub fn new() -> Self {
        Self::with_config(1000, Duration::from_secs(3600))
    }

    /// Create task manager with configuration
    pub fn with_config(max_tasks: usize, max_age: Duration) -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            handles: Arc::new(RwLock::new(HashMap::new())),
            max_tasks,
            max_age,
        }
    }

    /// Create new task
    pub fn create_task(&self, task_type: TaskType, description: String) -> Result<Task, TaskError> {
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        // Check task limit
        if tasks.len() >= self.max_tasks {
            self.cleanup_old_tasks_internal(&mut tasks)?;
            if tasks.len() >= self.max_tasks {
                return Err(TaskError::TooManyTasks(format!(
                    "Maximum {} tasks reached",
                    self.max_tasks
                )));
            }
        }

        let task = Task::new(task_type, description);
        let task_id = task.id.clone();

        tasks.insert(task_id.clone(), task.clone());
        info!("Created task {}: {:?}", task_id, task.task_type);

        Ok(task)
    }

    /// Get task by ID
    pub fn get_task(&self, task_id: &str) -> Result<Option<Task>, TaskError> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        Ok(tasks.get(task_id).cloned())
    }

    /// List all tasks (optionally filtered by status)
    pub fn list_tasks(&self, status_filter: Option<TaskStatus>) -> Result<Vec<Task>, TaskError> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        let mut task_list: Vec<Task> = tasks.values().cloned().collect();

        // Filter by status if requested
        if let Some(status) = status_filter {
            task_list.retain(|t| t.status == status);
        }

        // Sort by creation time (newest first)
        task_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(task_list)
    }

    /// Update task status
    pub fn update_task<F>(&self, task_id: &str, update_fn: F) -> Result<(), TaskError>
    where
        F: FnOnce(&mut Task),
    {
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        let task = tasks
            .get_mut(task_id)
            .ok_or_else(|| TaskError::NotFound(format!("Task '{}' not found", task_id)))?;

        update_fn(task);

        Ok(())
    }

    /// Delete task (and cancel if running)
    pub fn delete_task(&self, task_id: &str) -> Result<bool, TaskError> {
        // Cancel if running
        if let Ok(Some(task)) = self.get_task(task_id) {
            if task.status == TaskStatus::Running {
                self.cancel_task(task_id)?;
            }
        }

        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        let removed = tasks.remove(task_id).is_some();

        if removed {
            info!("Deleted task {}", task_id);
        }

        Ok(removed)
    }

    /// Cancel running task
    pub fn cancel_task(&self, task_id: &str) -> Result<(), TaskError> {
        // Abort the task handle if exists
        {
            let mut handles = self
                .handles
                .write()
                .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

            if let Some(handle) = handles.remove(task_id) {
                handle.abort();
                debug!("Aborted task handle for {}", task_id);
            }
        }

        // Mark as cancelled
        self.update_task(task_id, |task| {
            if !task.is_terminal() {
                task.mark_cancelled();
                info!("Cancelled task {}", task_id);
            }
        })?;

        Ok(())
    }

    /// Register task handle for cancellation support
    pub fn register_handle(&self, task_id: &str, handle: JoinHandle<()>) -> Result<(), TaskError> {
        let mut handles = self
            .handles
            .write()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        handles.insert(task_id.to_string(), handle);

        Ok(())
    }

    /// Cleanup old completed tasks
    pub fn cleanup_old_tasks(&self) -> Result<usize, TaskError> {
        let mut tasks = self
            .tasks
            .write()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        self.cleanup_old_tasks_internal(&mut tasks)
    }

    fn cleanup_old_tasks_internal(
        &self,
        tasks: &mut HashMap<String, Task>,
    ) -> Result<usize, TaskError> {
        let now = SystemTime::now();
        let mut removed = 0;

        tasks.retain(|id, task| {
            if task.is_terminal() {
                if let Ok(elapsed) = now.duration_since(task.updated_at) {
                    if elapsed > self.max_age {
                        debug!("Removing old task {}", id);
                        removed += 1;
                        return false;
                    }
                }
            }
            true
        });

        if removed > 0 {
            info!("Cleaned up {} old tasks", removed);
        }

        Ok(removed)
    }

    /// Get task statistics
    pub fn get_statistics(&self) -> Result<TaskStatistics, TaskError> {
        let tasks = self
            .tasks
            .read()
            .map_err(|e| TaskError::Internal(format!("Lock error: {}", e)))?;

        let mut stats = TaskStatistics {
            total: tasks.len(),
            pending: 0,
            running: 0,
            completed: 0,
            failed: 0,
            cancelled: 0,
        };

        for task in tasks.values() {
            match task.status {
                TaskStatus::Pending => stats.pending += 1,
                TaskStatus::Running => stats.running += 1,
                TaskStatus::Completed => stats.completed += 1,
                TaskStatus::Failed => stats.failed += 1,
                TaskStatus::Cancelled => stats.cancelled += 1,
            }
        }

        Ok(stats)
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Task statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatistics {
    pub total: usize,
    pub pending: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
}

/// Task error types
#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Too many tasks: {0}")]
    TooManyTasks(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl TaskError {
    fn status_code(&self) -> StatusCode {
        match self {
            TaskError::NotFound(_) => StatusCode::NOT_FOUND,
            TaskError::BadRequest(_) => StatusCode::BAD_REQUEST,
            TaskError::TooManyTasks(_) => StatusCode::TOO_MANY_REQUESTS,
            TaskError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for TaskError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();

        (
            status,
            Json(serde_json::json!({
                "error": message,
                "status": status.as_u16(),
            })),
        )
            .into_response()
    }
}

/// Request body for creating task
#[derive(Debug, Clone, Deserialize)]
pub struct CreateTaskRequest {
    #[serde(rename = "type")]
    pub task_type: TaskType,
    pub description: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// List all tasks
///
/// GET /$/tasks
pub async fn list_tasks(State(manager): State<Arc<TaskManager>>) -> Result<Response, TaskError> {
    info!("List tasks request");

    let tasks = manager.list_tasks(None)?;

    debug!("Found {} tasks", tasks.len());

    Ok((StatusCode::OK, Json(tasks)).into_response())
}

/// Get specific task
///
/// GET /$/tasks/{id}
pub async fn get_task(
    Path(task_id): Path<String>,
    State(manager): State<Arc<TaskManager>>,
) -> Result<Response, TaskError> {
    info!("Get task request: {}", task_id);

    match manager.get_task(&task_id)? {
        Some(task) => {
            debug!("Found task '{}' with status {:?}", task_id, task.status);
            Ok((StatusCode::OK, Json(task)).into_response())
        }
        None => {
            debug!("Task '{}' not found", task_id);
            Err(TaskError::NotFound(format!("Task '{}' not found", task_id)))
        }
    }
}

/// Create new async task
///
/// POST /$/tasks
/// Body: { "type": "upload", "description": "Bulk data upload" }
pub async fn create_task(
    State(manager): State<Arc<TaskManager>>,
    Json(req): Json<CreateTaskRequest>,
) -> Result<Response, TaskError> {
    info!(
        "Create task request: {:?} - {}",
        req.task_type, req.description
    );

    let mut task = manager.create_task(req.task_type, req.description)?;

    // Add metadata
    task.metadata = req.metadata;

    debug!("Created task '{}'", task.id);

    Ok((StatusCode::CREATED, Json(task)).into_response())
}

/// Delete task
///
/// DELETE /$/tasks/{id}
pub async fn delete_task(
    Path(task_id): Path<String>,
    State(manager): State<Arc<TaskManager>>,
) -> Result<Response, TaskError> {
    info!("Delete task request: {}", task_id);

    if manager.delete_task(&task_id)? {
        debug!("Deleted task '{}'", task_id);
        Ok((StatusCode::NO_CONTENT, ()).into_response())
    } else {
        debug!("Task '{}' not found", task_id);
        Err(TaskError::NotFound(format!("Task '{}' not found", task_id)))
    }
}

/// Cancel running task
///
/// POST /$/tasks/{id}/cancel
pub async fn cancel_task(
    Path(task_id): Path<String>,
    State(manager): State<Arc<TaskManager>>,
) -> Result<Response, TaskError> {
    info!("Cancel task request: {}", task_id);

    manager.cancel_task(&task_id)?;

    let task = manager
        .get_task(&task_id)?
        .ok_or_else(|| TaskError::NotFound(format!("Task '{}' not found", task_id)))?;

    debug!("Cancelled task '{}'", task_id);

    Ok((StatusCode::OK, Json(task)).into_response())
}

/// Get task statistics
///
/// GET /$/tasks/statistics
pub async fn get_task_statistics(
    State(manager): State<Arc<TaskManager>>,
) -> Result<Response, TaskError> {
    info!("Get task statistics request");

    let stats = manager.get_statistics()?;

    debug!("Task statistics: {:?}", stats);

    Ok((StatusCode::OK, Json(stats)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = Task::new(TaskType::Upload, "Test upload".to_string());

        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.task_type, TaskType::Upload);
        assert_eq!(task.description, "Test upload");
        assert!(!task.id.is_empty());
        assert_eq!(task.progress, None);
    }

    #[test]
    fn test_task_lifecycle() {
        let mut task = Task::new(TaskType::Query, "Test query".to_string());

        // Start running
        task.mark_running();
        assert_eq!(task.status, TaskStatus::Running);

        // Update progress
        task.update_progress(50.0);
        assert_eq!(task.progress, Some(50.0));

        // Complete
        task.mark_completed(Some(serde_json::json!({"count": 100})));
        assert_eq!(task.status, TaskStatus::Completed);
        assert_eq!(task.progress, Some(100.0));
        assert!(task.is_terminal());
        assert!(task.completed_at.is_some());
    }

    #[test]
    fn test_task_failure() {
        let mut task = Task::new(TaskType::Update, "Test update".to_string());

        task.mark_running();
        task.mark_failed("Connection timeout".to_string());

        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error, Some("Connection timeout".to_string()));
        assert!(task.is_terminal());
    }

    #[test]
    fn test_task_cancellation() {
        let mut task = Task::new(TaskType::Patch, "Test patch".to_string());

        task.mark_running();
        task.mark_cancelled();

        assert_eq!(task.status, TaskStatus::Cancelled);
        assert!(task.is_terminal());
    }

    #[tokio::test]
    async fn test_task_manager_create() {
        let manager = TaskManager::new();

        let task = manager
            .create_task(TaskType::Upload, "Test".to_string())
            .unwrap();

        assert!(!task.id.is_empty());
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[tokio::test]
    async fn test_task_manager_get() {
        let manager = TaskManager::new();

        let task = manager
            .create_task(TaskType::Query, "Test".to_string())
            .unwrap();
        let task_id = task.id.clone();

        let retrieved = manager.get_task(&task_id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, task_id);
    }

    #[tokio::test]
    async fn test_task_manager_list() {
        let manager = TaskManager::new();

        manager
            .create_task(TaskType::Upload, "Task 1".to_string())
            .unwrap();
        manager
            .create_task(TaskType::Query, "Task 2".to_string())
            .unwrap();

        let tasks = manager.list_tasks(None).unwrap();
        assert_eq!(tasks.len(), 2);
    }

    #[tokio::test]
    async fn test_task_manager_update() {
        let manager = TaskManager::new();

        let task = manager
            .create_task(TaskType::Update, "Test".to_string())
            .unwrap();
        let task_id = task.id.clone();

        manager
            .update_task(&task_id, |t| {
                t.mark_running();
                t.update_progress(75.0);
            })
            .unwrap();

        let updated = manager.get_task(&task_id).unwrap().unwrap();
        assert_eq!(updated.status, TaskStatus::Running);
        assert_eq!(updated.progress, Some(75.0));
    }

    #[tokio::test]
    async fn test_task_manager_delete() {
        let manager = TaskManager::new();

        let task = manager
            .create_task(TaskType::Patch, "Test".to_string())
            .unwrap();
        let task_id = task.id.clone();

        let deleted = manager.delete_task(&task_id).unwrap();
        assert!(deleted);

        let retrieved = manager.get_task(&task_id).unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_task_manager_statistics() {
        let manager = TaskManager::new();

        let task1 = manager
            .create_task(TaskType::Upload, "Task 1".to_string())
            .unwrap();
        let task2 = manager
            .create_task(TaskType::Query, "Task 2".to_string())
            .unwrap();

        manager
            .update_task(&task1.id, |t| t.mark_running())
            .unwrap();
        manager
            .update_task(&task2.id, |t| t.mark_completed(None))
            .unwrap();

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.running, 1);
        assert_eq!(stats.completed, 1);
    }
}
