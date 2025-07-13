//! Workflow Integration for OxiRS Chat
//!
//! This module provides business process integration including:
//! - Task delegation
//! - Report generation
//! - Data export
//! - Notification systems
//! - Approval workflows
//! - Audit trails

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, info};
use uuid::Uuid;

/// Configuration for workflow integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub enable_task_delegation: bool,
    pub enable_report_generation: bool,
    pub enable_data_export: bool,
    pub enable_notifications: bool,
    pub enable_approval_workflows: bool,
    pub enable_audit_trails: bool,
    pub export_directory: PathBuf,
    pub report_directory: PathBuf,
    pub audit_directory: PathBuf,
    pub max_concurrent_tasks: usize,
    pub task_timeout: Duration,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            enable_task_delegation: true,
            enable_report_generation: true,
            enable_data_export: true,
            enable_notifications: true,
            enable_approval_workflows: true,
            enable_audit_trails: true,
            export_directory: PathBuf::from("./exports"),
            report_directory: PathBuf::from("./reports"),
            audit_directory: PathBuf::from("./audit"),
            max_concurrent_tasks: 10,
            task_timeout: Duration::from_secs(300),
        }
    }
}

/// Workflow manager for business process integration
pub struct WorkflowManager {
    config: WorkflowConfig,
    active_tasks: HashMap<String, Task>,
    approval_queue: Vec<ApprovalRequest>,
    audit_logger: AuditLogger,
    notification_service: NotificationService,
    report_generator: ReportGenerator,
    data_exporter: DataExporter,
}

impl WorkflowManager {
    /// Create a new workflow manager
    pub fn new(config: WorkflowConfig) -> Result<Self> {
        // Ensure directories exist
        std::fs::create_dir_all(&config.export_directory)?;
        std::fs::create_dir_all(&config.report_directory)?;
        std::fs::create_dir_all(&config.audit_directory)?;

        Ok(Self {
            audit_logger: AuditLogger::new(&config),
            notification_service: NotificationService::new(&config),
            report_generator: ReportGenerator::new(&config),
            data_exporter: DataExporter::new(&config),
            config,
            active_tasks: HashMap::new(),
            approval_queue: Vec::new(),
        })
    }

    /// Delegate a task to an external system or user
    pub async fn delegate_task(&mut self, task_request: TaskRequest) -> Result<TaskId> {
        if !self.config.enable_task_delegation {
            return Err(anyhow!("Task delegation is disabled"));
        }

        let task_id = TaskId(Uuid::new_v4().to_string());
        let task = Task {
            id: task_id.clone(),
            request: task_request.clone(),
            status: TaskStatus::Pending,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            assigned_to: task_request.assignee.clone(),
            deadline: task_request.deadline,
            priority: task_request.priority,
            dependencies: task_request.dependencies.clone(),
            metadata: task_request.metadata.clone(),
        };

        // Log task creation
        self.audit_logger.log_task_creation(&task).await?;

        // Send notification to assignee
        if self.config.enable_notifications {
            self.notification_service
                .notify_task_assignment(&task)
                .await?;
        }

        self.active_tasks.insert(task_id.0.clone(), task);

        info!("Task delegated: {} to {}", task_id.0, task_request.assignee);
        Ok(task_id)
    }

    /// Update task status
    pub async fn update_task_status(&mut self, task_id: &TaskId, status: TaskStatus) -> Result<()> {
        let task = self
            .active_tasks
            .get_mut(&task_id.0)
            .ok_or_else(|| anyhow!("Task not found: {}", task_id.0))?;

        let old_status = task.status.clone();
        task.status = status.clone();
        task.updated_at = SystemTime::now();

        // Log status change
        self.audit_logger
            .log_task_status_change(task_id, &old_status, &status)
            .await?;

        // Send notification on completion
        if matches!(status, TaskStatus::Completed) && self.config.enable_notifications {
            self.notification_service
                .notify_task_completion(task)
                .await?;
        }

        info!("Task {} status updated to {:?}", task_id.0, status);
        Ok(())
    }

    /// Generate a report based on session data
    pub async fn generate_report(&self, report_request: ReportRequest) -> Result<ReportResult> {
        if !self.config.enable_report_generation {
            return Err(anyhow!("Report generation is disabled"));
        }

        let report = self
            .report_generator
            .generate_report(report_request)
            .await?;

        // Log report generation
        self.audit_logger.log_report_generation(&report).await?;

        info!("Report generated: {} ({})", report.title, report.format);
        Ok(report)
    }

    /// Export data in various formats
    pub async fn export_data(&self, export_request: ExportRequest) -> Result<ExportResult> {
        if !self.config.enable_data_export {
            return Err(anyhow!("Data export is disabled"));
        }

        let export_result = self.data_exporter.export_data(export_request).await?;

        // Log data export
        self.audit_logger.log_data_export(&export_result).await?;

        info!(
            "Data exported: {} ({})",
            export_result.filename, export_result.format
        );
        Ok(export_result)
    }

    /// Submit a request for approval
    pub async fn submit_for_approval(
        &mut self,
        approval_request: ApprovalRequest,
    ) -> Result<ApprovalId> {
        if !self.config.enable_approval_workflows {
            return Err(anyhow!("Approval workflows are disabled"));
        }

        let approval_id = ApprovalId(Uuid::new_v4().to_string());
        let mut request = approval_request;
        request.id = Some(approval_id.clone());
        request.submitted_at = Some(SystemTime::now());
        request.status = ApprovalStatus::Pending;

        // Log approval request
        self.audit_logger.log_approval_request(&request).await?;

        // Notify approvers
        if self.config.enable_notifications {
            for approver in &request.approvers {
                self.notification_service
                    .notify_approval_needed(&request, approver)
                    .await?;
            }
        }

        self.approval_queue.push(request);

        info!("Approval request submitted: {}", approval_id.0);
        Ok(approval_id)
    }

    /// Process an approval decision
    pub async fn process_approval(
        &mut self,
        approval_id: &ApprovalId,
        decision: ApprovalDecision,
    ) -> Result<()> {
        let request = self
            .approval_queue
            .iter_mut()
            .find(|r| r.id.as_ref() == Some(approval_id))
            .ok_or_else(|| anyhow!("Approval request not found: {}", approval_id.0))?;

        request.status = match decision.approved {
            true => ApprovalStatus::Approved,
            false => ApprovalStatus::Rejected,
        };
        request.decision = Some(decision.clone());
        request.processed_at = Some(SystemTime::now());

        // Log approval decision
        self.audit_logger
            .log_approval_decision(approval_id, &decision)
            .await?;

        // Notify requestor
        if self.config.enable_notifications {
            self.notification_service
                .notify_approval_decision(request)
                .await?;
        }

        info!(
            "Approval processed: {} - {}",
            approval_id.0,
            if decision.approved {
                "Approved"
            } else {
                "Rejected"
            }
        );
        Ok(())
    }

    /// Get audit trail for a specific entity
    pub async fn get_audit_trail(&self, entity_id: &str) -> Result<Vec<AuditEntry>> {
        if !self.config.enable_audit_trails {
            return Err(anyhow!("Audit trails are disabled"));
        }

        self.audit_logger.get_audit_trail(entity_id).await
    }

    /// Send notification
    pub async fn send_notification(&self, notification: Notification) -> Result<()> {
        if !self.config.enable_notifications {
            return Err(anyhow!("Notifications are disabled"));
        }

        self.notification_service
            .send_notification(notification)
            .await
    }

    /// Get active tasks
    pub fn get_active_tasks(&self) -> Vec<&Task> {
        self.active_tasks.values().collect()
    }

    /// Get pending approvals
    pub fn get_pending_approvals(&self) -> Vec<&ApprovalRequest> {
        self.approval_queue
            .iter()
            .filter(|r| matches!(r.status, ApprovalStatus::Pending))
            .collect()
    }
}

/// Task delegation types and structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub title: String,
    pub description: String,
    pub task_type: TaskType,
    pub assignee: String,
    pub deadline: Option<SystemTime>,
    pub priority: TaskPriority,
    pub dependencies: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub request: TaskRequest,
    pub status: TaskStatus,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub assigned_to: String,
    pub deadline: Option<SystemTime>,
    pub priority: TaskPriority,
    pub dependencies: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    DataAnalysis,
    ReportGeneration,
    DataValidation,
    QueryOptimization,
    KnowledgeUpdate,
    UserSupport,
    SystemMaintenance,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Report generation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportRequest {
    pub title: String,
    pub report_type: ReportType,
    pub format: ReportFormat,
    pub time_range: TimeRange,
    pub filters: HashMap<String, String>,
    pub include_charts: bool,
    pub include_raw_data: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResult {
    pub title: String,
    pub format: ReportFormat,
    pub file_path: PathBuf,
    pub generated_at: SystemTime,
    pub size_bytes: u64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    ConversationSummary,
    PerformanceMetrics,
    UsageAnalytics,
    ErrorReport,
    UserSatisfaction,
    QueryAnalysis,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    PDF,
    HTML,
    CSV,
    JSON,
    Excel,
    Markdown,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReportFormat::PDF => write!(f, "PDF"),
            ReportFormat::HTML => write!(f, "HTML"),
            ReportFormat::CSV => write!(f, "CSV"),
            ReportFormat::JSON => write!(f, "JSON"),
            ReportFormat::Excel => write!(f, "Excel"),
            ReportFormat::Markdown => write!(f, "Markdown"),
        }
    }
}

/// Data export types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub data_type: DataType,
    pub format: ExportFormat,
    pub time_range: TimeRange,
    pub filters: HashMap<String, String>,
    pub compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub filename: String,
    pub format: ExportFormat,
    pub file_path: PathBuf,
    pub exported_at: SystemTime,
    pub record_count: usize,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Messages,
    Sessions,
    Analytics,
    AuditLogs,
    UserProfiles,
    QueryHistory,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    Parquet,
    Avro,
    XML,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::JSON => write!(f, "JSON"),
            ExportFormat::CSV => write!(f, "CSV"),
            ExportFormat::Parquet => write!(f, "Parquet"),
            ExportFormat::Avro => write!(f, "Avro"),
            ExportFormat::XML => write!(f, "XML"),
        }
    }
}

/// Approval workflow types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Option<ApprovalId>,
    pub title: String,
    pub description: String,
    pub request_type: ApprovalType,
    pub requester: String,
    pub approvers: Vec<String>,
    pub required_approvals: usize,
    pub status: ApprovalStatus,
    pub submitted_at: Option<SystemTime>,
    pub processed_at: Option<SystemTime>,
    pub decision: Option<ApprovalDecision>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApprovalId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    pub approved: bool,
    pub approver: String,
    pub comments: String,
    pub decided_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalType {
    DataAccess,
    FeatureActivation,
    ConfigurationChange,
    UserPermission,
    DataExport,
    SystemUpgrade,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

/// Notification types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub recipient: String,
    pub notification_type: NotificationType,
    pub title: String,
    pub message: String,
    pub priority: NotificationPriority,
    pub channel: NotificationChannel,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    TaskAssignment,
    TaskCompletion,
    ApprovalRequest,
    ApprovalDecision,
    SystemAlert,
    UserMessage,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Medium,
    High,
    Urgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    SMS,
    InApp,
    Webhook,
    Slack,
    Teams,
}

/// Time range for reports and exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SystemTime,
    pub end: SystemTime,
}

/// Audit trail types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub entity_id: String,
    pub action: AuditAction,
    pub actor: String,
    pub timestamp: SystemTime,
    pub details: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    TaskCreated,
    TaskUpdated,
    TaskCompleted,
    ReportGenerated,
    DataExported,
    ApprovalRequested,
    ApprovalDecided,
    ConfigurationChanged,
    UserAction(String),
}

// Implementation structs for internal services
struct AuditLogger {
    config: WorkflowConfig,
}

impl AuditLogger {
    fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn log_task_creation(&self, task: &Task) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: task.id.0.clone(),
            action: AuditAction::TaskCreated,
            actor: "system".to_string(),
            timestamp: SystemTime::now(),
            details: [
                ("assignee".to_string(), task.assigned_to.clone()),
                ("priority".to_string(), format!("{:?}", task.priority)),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn log_task_status_change(
        &self,
        task_id: &TaskId,
        old_status: &TaskStatus,
        new_status: &TaskStatus,
    ) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: task_id.0.clone(),
            action: AuditAction::TaskUpdated,
            actor: "system".to_string(),
            timestamp: SystemTime::now(),
            details: [
                ("old_status".to_string(), format!("{old_status:?}")),
                ("new_status".to_string(), format!("{new_status:?}")),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn log_report_generation(&self, report: &ReportResult) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: report.title.clone(),
            action: AuditAction::ReportGenerated,
            actor: "system".to_string(),
            timestamp: SystemTime::now(),
            details: [
                ("format".to_string(), format!("{:?}", report.format)),
                ("size".to_string(), report.size_bytes.to_string()),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn log_data_export(&self, export: &ExportResult) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: export.filename.clone(),
            action: AuditAction::DataExported,
            actor: "system".to_string(),
            timestamp: SystemTime::now(),
            details: [
                ("format".to_string(), format!("{:?}", export.format)),
                ("records".to_string(), export.record_count.to_string()),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn log_approval_request(&self, request: &ApprovalRequest) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: request.id.as_ref().unwrap().0.clone(),
            action: AuditAction::ApprovalRequested,
            actor: request.requester.clone(),
            timestamp: SystemTime::now(),
            details: [
                ("type".to_string(), format!("{:?}", request.request_type)),
                ("approvers".to_string(), request.approvers.join(",")),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn log_approval_decision(
        &self,
        approval_id: &ApprovalId,
        decision: &ApprovalDecision,
    ) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: approval_id.0.clone(),
            action: AuditAction::ApprovalDecided,
            actor: decision.approver.clone(),
            timestamp: SystemTime::now(),
            details: [
                ("approved".to_string(), decision.approved.to_string()),
                ("comments".to_string(), decision.comments.clone()),
            ]
            .into(),
            ip_address: None,
            user_agent: None,
        };

        self.write_audit_entry(&entry).await
    }

    async fn write_audit_entry(&self, entry: &AuditEntry) -> Result<()> {
        let filename = format!(
            "audit_{}.jsonl",
            chrono::DateTime::<chrono::Utc>::from(entry.timestamp).format("%Y-%m-%d")
        );
        let filepath = self.config.audit_directory.join(filename);

        let entry_json = serde_json::to_string(entry)?;
        let entry_line = format!("{entry_json}\n");

        fs::write(&filepath, entry_line).await?;
        Ok(())
    }

    async fn get_audit_trail(&self, _entity_id: &str) -> Result<Vec<AuditEntry>> {
        // Implementation to read audit entries for specific entity
        // This would scan audit files and filter by entity_id
        Ok(Vec::new()) // Simplified implementation
    }
}

struct NotificationService {
    #[allow(dead_code)]
    config: WorkflowConfig,
}

impl NotificationService {
    fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn notify_task_assignment(&self, task: &Task) -> Result<()> {
        let notification = Notification {
            recipient: task.assigned_to.clone(),
            notification_type: NotificationType::TaskAssignment,
            title: format!("New task assigned: {}", task.request.title),
            message: format!(
                "You have been assigned a new task: {}",
                task.request.description
            ),
            priority: match task.priority {
                TaskPriority::Critical => NotificationPriority::Urgent,
                TaskPriority::High => NotificationPriority::High,
                TaskPriority::Medium => NotificationPriority::Medium,
                TaskPriority::Low => NotificationPriority::Low,
            },
            channel: NotificationChannel::InApp,
            metadata: HashMap::new(),
        };

        self.send_notification(notification).await
    }

    async fn notify_task_completion(&self, task: &Task) -> Result<()> {
        let notification = Notification {
            recipient: task.assigned_to.clone(),
            notification_type: NotificationType::TaskCompletion,
            title: format!("Task completed: {}", task.request.title),
            message: "Task has been marked as completed".to_string(),
            priority: NotificationPriority::Medium,
            channel: NotificationChannel::InApp,
            metadata: HashMap::new(),
        };

        self.send_notification(notification).await
    }

    async fn notify_approval_needed(
        &self,
        request: &ApprovalRequest,
        approver: &str,
    ) -> Result<()> {
        let notification = Notification {
            recipient: approver.to_string(),
            notification_type: NotificationType::ApprovalRequest,
            title: format!("Approval needed: {}", request.title),
            message: format!("Please review and approve: {}", request.description),
            priority: NotificationPriority::High,
            channel: NotificationChannel::InApp,
            metadata: HashMap::new(),
        };

        self.send_notification(notification).await
    }

    async fn notify_approval_decision(&self, request: &ApprovalRequest) -> Result<()> {
        let decision = request.decision.as_ref().unwrap();
        let notification = Notification {
            recipient: request.requester.clone(),
            notification_type: NotificationType::ApprovalDecision,
            title: format!(
                "Approval {}: {}",
                if decision.approved {
                    "approved"
                } else {
                    "rejected"
                },
                request.title
            ),
            message: format!("Decision: {}", decision.comments),
            priority: NotificationPriority::Medium,
            channel: NotificationChannel::InApp,
            metadata: HashMap::new(),
        };

        self.send_notification(notification).await
    }

    async fn send_notification(&self, notification: Notification) -> Result<()> {
        // Implementation would send notifications via configured channels
        info!(
            "Notification sent to {}: {}",
            notification.recipient, notification.title
        );
        Ok(())
    }
}

struct ReportGenerator {
    config: WorkflowConfig,
}

impl ReportGenerator {
    fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn generate_report(&self, request: ReportRequest) -> Result<ReportResult> {
        let filename = format!(
            "{}_{}.{}",
            request.title.replace(" ", "_"),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"),
            self.format_extension(&request.format)
        );

        let file_path = self.config.report_directory.join(&filename);

        // Generate report content based on type and format
        let content = self.generate_report_content(&request).await?;

        fs::write(&file_path, content).await?;

        let metadata = fs::metadata(&file_path).await?;

        Ok(ReportResult {
            title: request.title,
            format: request.format,
            file_path,
            generated_at: SystemTime::now(),
            size_bytes: metadata.len(),
            metadata: HashMap::new(),
        })
    }

    async fn generate_report_content(&self, request: &ReportRequest) -> Result<Vec<u8>> {
        // Implementation would generate actual report content
        // This is a simplified placeholder
        let content = match request.format {
            ReportFormat::JSON => serde_json::to_string_pretty(&request)?.into_bytes(),
            ReportFormat::CSV => format!(
                "Report: {}\nGenerated at: {}",
                request.title,
                chrono::Utc::now()
            )
            .into_bytes(),
            _ => format!("Report: {}", request.title).into_bytes(),
        };

        Ok(content)
    }

    fn format_extension(&self, format: &ReportFormat) -> &str {
        match format {
            ReportFormat::PDF => "pdf",
            ReportFormat::HTML => "html",
            ReportFormat::CSV => "csv",
            ReportFormat::JSON => "json",
            ReportFormat::Excel => "xlsx",
            ReportFormat::Markdown => "md",
        }
    }
}

struct DataExporter {
    config: WorkflowConfig,
}

impl DataExporter {
    fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    async fn export_data(&self, request: ExportRequest) -> Result<ExportResult> {
        let filename = format!(
            "export_{}_{}.{}",
            format!("{:?}", request.data_type).to_lowercase(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S"),
            self.format_extension(&request.format)
        );

        let file_path = self.config.export_directory.join(&filename);

        // Export data based on type and format
        let (content, record_count) = self.generate_export_content(&request).await?;

        fs::write(&file_path, content).await?;

        let metadata = fs::metadata(&file_path).await?;

        Ok(ExportResult {
            filename,
            format: request.format,
            file_path,
            exported_at: SystemTime::now(),
            record_count,
            size_bytes: metadata.len(),
        })
    }

    async fn generate_export_content(&self, request: &ExportRequest) -> Result<(Vec<u8>, usize)> {
        // Implementation would export actual data
        // This is a simplified placeholder
        let content = match request.format {
            ExportFormat::JSON => serde_json::to_string_pretty(&request)?.into_bytes(),
            ExportFormat::CSV => format!(
                "data_type,exported_at\n{:?},{}",
                request.data_type,
                chrono::Utc::now()
            )
            .into_bytes(),
            _ => format!("Export: {:?}", request.data_type).into_bytes(),
        };

        Ok((content, 1)) // Simplified record count
    }

    fn format_extension(&self, format: &ExportFormat) -> &str {
        match format {
            ExportFormat::JSON => "json",
            ExportFormat::CSV => "csv",
            ExportFormat::Parquet => "parquet",
            ExportFormat::Avro => "avro",
            ExportFormat::XML => "xml",
        }
    }
}

// Enhanced Collaborative Features for Version 1.1

/// Collaborative workspace manager for real-time team coordination
pub struct CollaborativeWorkspaceManager {
    workspaces: HashMap<String, CollaborativeWorkspace>,
    active_sessions: HashMap<String, CollaborativeSession>,
    presence_tracker: PresenceTracker,
    message_bus: CollaborativeMessageBus,
    shared_document_manager: SharedDocumentManager,
    decision_tracker: CollaborativeDecisionTracker,
}

impl Default for CollaborativeWorkspaceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CollaborativeWorkspaceManager {
    /// Create a new collaborative workspace manager
    pub fn new() -> Self {
        Self {
            workspaces: HashMap::new(),
            active_sessions: HashMap::new(),
            presence_tracker: PresenceTracker::new(),
            message_bus: CollaborativeMessageBus::new(),
            shared_document_manager: SharedDocumentManager::new(),
            decision_tracker: CollaborativeDecisionTracker::new(),
        }
    }

    /// Create a new collaborative workspace
    pub async fn create_workspace(
        &mut self,
        request: CreateWorkspaceRequest,
    ) -> Result<WorkspaceId> {
        let workspace_id = WorkspaceId(Uuid::new_v4().to_string());

        let workspace = CollaborativeWorkspace {
            id: workspace_id.clone(),
            name: request.name,
            description: request.description,
            owner: request.owner,
            members: request.initial_members,
            permissions: request.permissions,
            shared_documents: Vec::new(),
            active_collaborations: Vec::new(),
            settings: request.settings.unwrap_or_default(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.workspaces.insert(workspace_id.0.clone(), workspace);

        info!("Created collaborative workspace: {}", workspace_id.0);
        Ok(workspace_id)
    }

    /// Join a collaborative session
    pub async fn join_session(
        &mut self,
        workspace_id: &WorkspaceId,
        user_id: &str,
        user_info: UserInfo,
    ) -> Result<SessionToken> {
        let session_token = SessionToken(Uuid::new_v4().to_string());

        let session = CollaborativeSession {
            token: session_token.clone(),
            workspace_id: workspace_id.clone(),
            user_id: user_id.to_string(),
            user_info,
            joined_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            active_documents: Vec::new(),
            permissions: self.get_user_permissions(workspace_id, user_id)?,
        };

        self.active_sessions
            .insert(session_token.0.clone(), session);

        // Update presence
        self.presence_tracker
            .user_joined(workspace_id, user_id)
            .await?;

        // Notify other users
        self.message_bus
            .broadcast_user_joined(workspace_id, user_id)
            .await?;

        Ok(session_token)
    }

    /// Leave a collaborative session
    pub async fn leave_session(&mut self, session_token: &SessionToken) -> Result<()> {
        if let Some(session) = self.active_sessions.remove(&session_token.0) {
            // Update presence
            self.presence_tracker
                .user_left(&session.workspace_id, &session.user_id)
                .await?;

            // Notify other users
            self.message_bus
                .broadcast_user_left(&session.workspace_id, &session.user_id)
                .await?;

            info!("User {} left session", session.user_id);
        }

        Ok(())
    }

    /// Start collaborative editing on a document
    pub async fn start_collaborative_editing(
        &mut self,
        session_token: &SessionToken,
        document_id: &str,
        document_type: DocumentType,
    ) -> Result<CollaborativeEditingSession> {
        let session = self
            .active_sessions
            .get(session_token.0.as_str())
            .ok_or_else(|| anyhow!("Invalid session token"))?;

        let editing_session = self
            .shared_document_manager
            .start_editing_session(
                &session.workspace_id,
                document_id,
                &session.user_id,
                document_type,
            )
            .await?;

        Ok(editing_session)
    }

    /// Send real-time message to workspace
    pub async fn send_message(
        &mut self,
        session_token: &SessionToken,
        message: CollaborativeMessage,
    ) -> Result<MessageId> {
        let session = self
            .active_sessions
            .get(session_token.0.as_str())
            .ok_or_else(|| anyhow!("Invalid session token"))?;

        let message_id = self
            .message_bus
            .send_message(&session.workspace_id, &session.user_id, message)
            .await?;

        Ok(message_id)
    }

    /// Start a collaborative decision process
    pub async fn start_decision_process(
        &mut self,
        session_token: &SessionToken,
        decision_request: DecisionRequest,
    ) -> Result<DecisionId> {
        let session = self
            .active_sessions
            .get(session_token.0.as_str())
            .ok_or_else(|| anyhow!("Invalid session token"))?;

        let decision_id = self
            .decision_tracker
            .start_decision(&session.workspace_id, &session.user_id, decision_request)
            .await?;

        Ok(decision_id)
    }

    /// Get current workspace presence
    pub async fn get_workspace_presence(
        &self,
        workspace_id: &WorkspaceId,
    ) -> Result<Vec<UserPresence>> {
        self.presence_tracker
            .get_workspace_presence(workspace_id)
            .await
    }

    /// Get workspace activity feed
    pub async fn get_activity_feed(
        &self,
        _workspace_id: &WorkspaceId,
        _since: Option<SystemTime>,
        _limit: usize,
    ) -> Result<Vec<ActivityEvent>> {
        // Implementation would fetch recent activities
        Ok(Vec::new()) // Placeholder
    }

    fn get_user_permissions(
        &self,
        _workspace_id: &WorkspaceId,
        _user_id: &str,
    ) -> Result<UserPermissions> {
        // Implementation would check user permissions
        Ok(UserPermissions::default())
    }
}

/// Collaborative workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeWorkspace {
    pub id: WorkspaceId,
    pub name: String,
    pub description: Option<String>,
    pub owner: String,
    pub members: Vec<WorkspaceMember>,
    pub permissions: WorkspacePermissions,
    pub shared_documents: Vec<SharedDocument>,
    pub active_collaborations: Vec<ActiveCollaboration>,
    pub settings: WorkspaceSettings,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToken(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMember {
    pub user_id: String,
    pub role: WorkspaceRole,
    pub permissions: UserPermissions,
    pub joined_at: SystemTime,
    pub last_active: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkspaceRole {
    Owner,
    Admin,
    Editor,
    Viewer,
    Guest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPermissions {
    pub can_edit_documents: bool,
    pub can_create_documents: bool,
    pub can_delete_documents: bool,
    pub can_invite_users: bool,
    pub can_manage_permissions: bool,
    pub can_start_decisions: bool,
    pub can_vote: bool,
    pub can_moderate: bool,
}

impl Default for UserPermissions {
    fn default() -> Self {
        Self {
            can_edit_documents: true,
            can_create_documents: true,
            can_delete_documents: false,
            can_invite_users: false,
            can_manage_permissions: false,
            can_start_decisions: true,
            can_vote: true,
            can_moderate: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspacePermissions {
    pub public_readable: bool,
    pub allow_anonymous_access: bool,
    pub require_approval_for_members: bool,
    pub default_member_permissions: UserPermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSettings {
    pub enable_real_time_editing: bool,
    pub enable_presence_awareness: bool,
    pub enable_chat: bool,
    pub enable_video_calls: bool,
    pub enable_decision_voting: bool,
    pub auto_save_interval: std::time::Duration,
    pub max_concurrent_editors: usize,
    pub session_timeout: std::time::Duration,
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            enable_real_time_editing: true,
            enable_presence_awareness: true,
            enable_chat: true,
            enable_video_calls: false,
            enable_decision_voting: true,
            auto_save_interval: std::time::Duration::from_secs(10),
            max_concurrent_editors: 50,
            session_timeout: std::time::Duration::from_secs(8 * 3600),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWorkspaceRequest {
    pub name: String,
    pub description: Option<String>,
    pub owner: String,
    pub initial_members: Vec<WorkspaceMember>,
    pub permissions: WorkspacePermissions,
    pub settings: Option<WorkspaceSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub name: String,
    pub email: String,
    pub avatar_url: Option<String>,
    pub timezone: String,
    pub preferences: UserPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub enable_notifications: bool,
    pub notification_types: Vec<NotificationType>,
    pub presence_status: PresenceStatus,
    pub auto_join_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresenceStatus {
    Online,
    Away,
    Busy,
    DoNotDisturb,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeSession {
    pub token: SessionToken,
    pub workspace_id: WorkspaceId,
    pub user_id: String,
    pub user_info: UserInfo,
    pub joined_at: SystemTime,
    pub last_activity: SystemTime,
    pub active_documents: Vec<String>,
    pub permissions: UserPermissions,
}

/// Presence tracking for real-time collaboration awareness
pub struct PresenceTracker {
    workspace_presence: HashMap<String, Vec<UserPresence>>,
}

impl Default for PresenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PresenceTracker {
    pub fn new() -> Self {
        Self {
            workspace_presence: HashMap::new(),
        }
    }

    pub async fn user_joined(&mut self, workspace_id: &WorkspaceId, user_id: &str) -> Result<()> {
        let presence = UserPresence {
            user_id: user_id.to_string(),
            status: PresenceStatus::Online,
            last_seen: SystemTime::now(),
            current_activity: Some("Joined workspace".to_string()),
            cursor_position: None,
            viewing_document: None,
        };

        self.workspace_presence
            .entry(workspace_id.0.clone())
            .or_default()
            .push(presence);

        Ok(())
    }

    pub async fn user_left(&mut self, workspace_id: &WorkspaceId, user_id: &str) -> Result<()> {
        if let Some(users) = self.workspace_presence.get_mut(&workspace_id.0) {
            users.retain(|u| u.user_id != user_id);
        }
        Ok(())
    }

    pub async fn get_workspace_presence(
        &self,
        workspace_id: &WorkspaceId,
    ) -> Result<Vec<UserPresence>> {
        Ok(self
            .workspace_presence
            .get(&workspace_id.0)
            .cloned()
            .unwrap_or_default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user_id: String,
    pub status: PresenceStatus,
    pub last_seen: SystemTime,
    pub current_activity: Option<String>,
    pub cursor_position: Option<CursorPosition>,
    pub viewing_document: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub document_id: String,
    pub line: u32,
    pub column: u32,
    pub selection_start: Option<(u32, u32)>,
    pub selection_end: Option<(u32, u32)>,
}

/// Real-time messaging system for collaboration
pub struct CollaborativeMessageBus {
    message_history: HashMap<String, Vec<CollaborativeMessage>>,
    _subscribers: HashMap<String, Vec<String>>, // workspace_id -> user_ids
}

impl Default for CollaborativeMessageBus {
    fn default() -> Self {
        Self::new()
    }
}

impl CollaborativeMessageBus {
    pub fn new() -> Self {
        Self {
            message_history: HashMap::new(),
            _subscribers: HashMap::new(),
        }
    }

    pub async fn send_message(
        &mut self,
        workspace_id: &WorkspaceId,
        sender_id: &str,
        message: CollaborativeMessage,
    ) -> Result<MessageId> {
        let message_id = MessageId(Uuid::new_v4().to_string());

        let timestamped_message = CollaborativeMessage {
            id: Some(message_id.clone()),
            sender_id: sender_id.to_string(),
            timestamp: Some(SystemTime::now()),
            ..message
        };

        self.message_history
            .entry(workspace_id.0.clone())
            .or_default()
            .push(timestamped_message);

        // Broadcast to subscribers (implementation would use real-time channels)
        self.broadcast_message(workspace_id, &message_id).await?;

        Ok(message_id)
    }

    pub async fn broadcast_user_joined(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &str,
    ) -> Result<()> {
        // Implementation would broadcast presence updates
        debug!(
            "Broadcasting user joined: {} in workspace {}",
            user_id, workspace_id.0
        );
        Ok(())
    }

    pub async fn broadcast_user_left(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &str,
    ) -> Result<()> {
        // Implementation would broadcast presence updates
        debug!(
            "Broadcasting user left: {} in workspace {}",
            user_id, workspace_id.0
        );
        Ok(())
    }

    async fn broadcast_message(
        &self,
        workspace_id: &WorkspaceId,
        message_id: &MessageId,
    ) -> Result<()> {
        // Implementation would use WebSocket or other real-time protocol
        debug!(
            "Broadcasting message {} to workspace {}",
            message_id.0, workspace_id.0
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeMessage {
    pub id: Option<MessageId>,
    pub sender_id: String,
    pub message_type: MessageType,
    pub content: String,
    pub thread_id: Option<String>,
    pub reply_to: Option<MessageId>,
    pub mentions: Vec<String>,
    pub attachments: Vec<MessageAttachment>,
    pub reactions: Vec<MessageReaction>,
    pub timestamp: Option<SystemTime>,
    pub edited_at: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Text,
    System,
    Notification,
    DocumentUpdate,
    VideoCall,
    Decision,
    Poll,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub file_id: String,
    pub filename: String,
    pub file_type: String,
    pub size_bytes: u64,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageReaction {
    pub emoji: String,
    pub user_id: String,
    pub timestamp: SystemTime,
}

/// Shared document management for collaborative editing
pub struct SharedDocumentManager {
    _documents: HashMap<String, SharedDocument>,
    editing_sessions: HashMap<String, Vec<CollaborativeEditingSession>>,
}

impl Default for SharedDocumentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedDocumentManager {
    pub fn new() -> Self {
        Self {
            _documents: HashMap::new(),
            editing_sessions: HashMap::new(),
        }
    }

    pub async fn start_editing_session(
        &mut self,
        workspace_id: &WorkspaceId,
        document_id: &str,
        user_id: &str,
        document_type: DocumentType,
    ) -> Result<CollaborativeEditingSession> {
        let session = CollaborativeEditingSession {
            session_id: Uuid::new_v4().to_string(),
            workspace_id: workspace_id.clone(),
            document_id: document_id.to_string(),
            user_id: user_id.to_string(),
            document_type,
            started_at: SystemTime::now(),
            last_edit: SystemTime::now(),
            cursor_position: None,
            pending_operations: Vec::new(),
        };

        self.editing_sessions
            .entry(document_id.to_string())
            .or_default()
            .push(session.clone());

        Ok(session)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedDocument {
    pub id: String,
    pub name: String,
    pub document_type: DocumentType,
    pub content: String,
    pub version: u64,
    pub created_by: String,
    pub created_at: SystemTime,
    pub last_modified_by: String,
    pub last_modified_at: SystemTime,
    pub collaborators: Vec<String>,
    pub permissions: DocumentPermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    SparqlQuery,
    MarkdownDocument,
    JsonDocument,
    CodeFile { language: String },
    Whiteboard,
    Spreadsheet,
    Presentation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPermissions {
    pub public_readable: bool,
    pub editors: Vec<String>,
    pub viewers: Vec<String>,
    pub allow_comments: bool,
    pub allow_suggestions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborativeEditingSession {
    pub session_id: String,
    pub workspace_id: WorkspaceId,
    pub document_id: String,
    pub user_id: String,
    pub document_type: DocumentType,
    pub started_at: SystemTime,
    pub last_edit: SystemTime,
    pub cursor_position: Option<CursorPosition>,
    pub pending_operations: Vec<EditOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditOperation {
    pub operation_id: String,
    pub operation_type: OperationType,
    pub position: TextPosition,
    pub content: String,
    pub timestamp: SystemTime,
    pub user_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Delete,
    Replace,
    FormatApply,
    Comment,
    Suggestion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPosition {
    pub line: u32,
    pub column: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveCollaboration {
    pub collaboration_id: String,
    pub collaboration_type: CollaborationType,
    pub participants: Vec<String>,
    pub started_by: String,
    pub started_at: SystemTime,
    pub status: CollaborationStatus,
    pub context: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationType {
    DocumentEditing,
    VideoCall,
    ScreenShare,
    Brainstorming,
    DecisionMaking,
    ReviewSession,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationStatus {
    Active,
    Paused,
    Completed,
    Cancelled,
}

/// Collaborative decision-making system
pub struct CollaborativeDecisionTracker {
    active_decisions: HashMap<String, DecisionProcess>,
}

impl Default for CollaborativeDecisionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CollaborativeDecisionTracker {
    pub fn new() -> Self {
        Self {
            active_decisions: HashMap::new(),
        }
    }

    pub async fn start_decision(
        &mut self,
        workspace_id: &WorkspaceId,
        initiator_id: &str,
        request: DecisionRequest,
    ) -> Result<DecisionId> {
        let decision_id = DecisionId(Uuid::new_v4().to_string());

        let decision_process = DecisionProcess {
            id: decision_id.clone(),
            workspace_id: workspace_id.clone(),
            initiator_id: initiator_id.to_string(),
            title: request.title,
            description: request.description,
            decision_type: request.decision_type,
            options: request.options,
            eligible_voters: request.eligible_voters,
            votes: HashMap::new(),
            comments: Vec::new(),
            deadline: request.deadline,
            status: DecisionStatus::Open,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.active_decisions
            .insert(decision_id.0.clone(), decision_process);

        Ok(decision_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub options: Vec<DecisionOption>,
    pub eligible_voters: Vec<String>,
    pub deadline: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionProcess {
    pub id: DecisionId,
    pub workspace_id: WorkspaceId,
    pub initiator_id: String,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub options: Vec<DecisionOption>,
    pub eligible_voters: Vec<String>,
    pub votes: HashMap<String, Vote>, // user_id -> vote
    pub comments: Vec<DecisionComment>,
    pub deadline: Option<SystemTime>,
    pub status: DecisionStatus,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    SingleChoice,
    MultipleChoice,
    Ranking,
    YesNo,
    Consensus,
    Budget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub proposed_by: String,
    pub vote_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter_id: String,
    pub option_ids: Vec<String>,
    pub ranking: Option<Vec<String>>,
    pub comment: Option<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionComment {
    pub id: String,
    pub author_id: String,
    pub content: String,
    pub timestamp: SystemTime,
    pub replies: Vec<DecisionComment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionStatus {
    Open,
    Closed,
    Cancelled,
    Implemented,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEvent {
    pub id: String,
    pub workspace_id: WorkspaceId,
    pub event_type: ActivityEventType,
    pub actor_id: String,
    pub target_id: Option<String>,
    pub description: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityEventType {
    UserJoined,
    UserLeft,
    DocumentCreated,
    DocumentEdited,
    DocumentShared,
    MessageSent,
    DecisionStarted,
    DecisionVoted,
    CollaborationStarted,
    CollaborationEnded,
}
