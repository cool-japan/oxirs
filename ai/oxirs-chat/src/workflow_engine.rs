//! # Workflow Engine
//!
//! Core service implementations: WorkflowManager, AuditLogger, NotificationService,
//! ReportGenerator, DataExporter, and the CollaborativeWorkspaceManager with its
//! supporting services (PresenceTracker, CollaborativeMessageBus, SharedDocumentManager,
//! CollaborativeDecisionTracker).

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::SystemTime;
use tokio::fs;
use tracing::{debug, info};
use uuid::Uuid;

use crate::workflow_types::{
    ActivityEvent, ApprovalDecision, ApprovalId, ApprovalRequest, ApprovalStatus, AuditAction,
    AuditEntry, AuditLogger, CollaborativeDecisionTracker, CollaborativeEditingSession,
    CollaborativeMessage, CollaborativeMessageBus, CollaborativeSession, CollaborativeWorkspace,
    CollaborativeWorkspaceManager, CreateWorkspaceRequest, DataExporter, DecisionId,
    DecisionProcess, DecisionRequest, DecisionStatus, DocumentType, ExportFormat, ExportRequest,
    ExportResult, MessageId, Notification, NotificationChannel, NotificationPriority,
    NotificationService, NotificationType, PresenceStatus, PresenceTracker, ReportFormat,
    ReportGenerator, ReportRequest, ReportResult, SessionToken, SharedDocumentManager, Task,
    TaskId, TaskPriority, TaskRequest, TaskStatus, UserInfo, UserPermissions, UserPresence,
    WorkflowConfig, WorkflowManager, WorkspaceId,
};

// ============================================================================
// WorkflowManager implementation
// ============================================================================

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

// ============================================================================
// AuditLogger implementation
// ============================================================================

impl AuditLogger {
    pub(crate) fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub(crate) async fn log_task_creation(&self, task: &Task) -> Result<()> {
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

    pub(crate) async fn log_task_status_change(
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

    pub(crate) async fn log_report_generation(&self, report: &ReportResult) -> Result<()> {
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

    pub(crate) async fn log_data_export(&self, export: &ExportResult) -> Result<()> {
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

    pub(crate) async fn log_approval_request(&self, request: &ApprovalRequest) -> Result<()> {
        let entry = AuditEntry {
            id: Uuid::new_v4().to_string(),
            entity_id: request
                .id
                .as_ref()
                .expect("request should have an id")
                .0
                .clone(),
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

    pub(crate) async fn log_approval_decision(
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

    pub(crate) async fn get_audit_trail(&self, _entity_id: &str) -> Result<Vec<AuditEntry>> {
        // Implementation to read audit entries for specific entity
        // This would scan audit files and filter by entity_id
        Ok(Vec::new()) // Simplified implementation
    }
}

// ============================================================================
// NotificationService
// ============================================================================

impl NotificationService {
    pub(crate) fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub(crate) async fn notify_task_assignment(&self, task: &Task) -> Result<()> {
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

    pub(crate) async fn notify_task_completion(&self, task: &Task) -> Result<()> {
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

    pub(crate) async fn notify_approval_needed(
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

    pub(crate) async fn notify_approval_decision(&self, request: &ApprovalRequest) -> Result<()> {
        let decision = request
            .decision
            .as_ref()
            .expect("request should have a decision");
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

    pub(crate) async fn send_notification(&self, notification: Notification) -> Result<()> {
        // Implementation would send notifications via configured channels
        info!(
            "Notification sent to {}: {}",
            notification.recipient, notification.title
        );
        Ok(())
    }
}

// ============================================================================
// ReportGenerator
// ============================================================================

impl ReportGenerator {
    pub(crate) fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub(crate) async fn generate_report(&self, request: ReportRequest) -> Result<ReportResult> {
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

// ============================================================================
// DataExporter
// ============================================================================

impl DataExporter {
    pub(crate) fn new(config: &WorkflowConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub(crate) async fn export_data(&self, request: ExportRequest) -> Result<ExportResult> {
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

// ============================================================================
// CollaborativeWorkspaceManager implementation
// ============================================================================

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

// ============================================================================
// PresenceTracker implementation
// ============================================================================

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

// ============================================================================
// CollaborativeMessageBus implementation
// ============================================================================

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

// ============================================================================
// SharedDocumentManager implementation
// ============================================================================

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

// ============================================================================
// CollaborativeDecisionTracker implementation
// ============================================================================

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
