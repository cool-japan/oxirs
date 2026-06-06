//! # Workflow Types
//!
//! All public type definitions, structs, and enums used by the workflow
//! integration system: tasks, reports, exports, approvals, notifications,
//! audit trails, and collaborative workspace types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

// ============================================================================
// Configuration
// ============================================================================

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

// ============================================================================
// Task types
// ============================================================================

/// Task delegation request
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

/// Active task record
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

/// Newtype wrapper for task identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// Classification of a task
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

/// Lifecycle state of a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Urgency level of a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// Report generation types
// ============================================================================

/// Report generation request
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

/// Completed report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportResult {
    pub title: String,
    pub format: ReportFormat,
    pub file_path: PathBuf,
    pub generated_at: SystemTime,
    pub size_bytes: u64,
    pub metadata: HashMap<String, String>,
}

/// Category of the report
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

/// Output format for a report
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

// ============================================================================
// Data export types
// ============================================================================

/// Data export request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequest {
    pub data_type: DataType,
    pub format: ExportFormat,
    pub time_range: TimeRange,
    pub filters: HashMap<String, String>,
    pub compression: bool,
}

/// Completed export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub filename: String,
    pub format: ExportFormat,
    pub file_path: PathBuf,
    pub exported_at: SystemTime,
    pub record_count: usize,
    pub size_bytes: u64,
}

/// Categories of exportable data
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

/// Output format for exported data
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

// ============================================================================
// Approval workflow types
// ============================================================================

/// Approval workflow request
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

/// Newtype wrapper for approval identifiers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApprovalId(pub String);

/// Approval decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    pub approved: bool,
    pub approver: String,
    pub comments: String,
    pub decided_at: SystemTime,
}

/// Category of an approval request
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

/// Lifecycle state of an approval request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

// ============================================================================
// Notification types
// ============================================================================

/// Outbound notification
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

/// Category of a notification
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

/// Urgency level of a notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Delivery channel for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email,
    SMS,
    InApp,
    Webhook,
    Slack,
    Teams,
}

// ============================================================================
// Shared time type
// ============================================================================

/// Time range for reports and exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SystemTime,
    pub end: SystemTime,
}

// ============================================================================
// Audit trail types
// ============================================================================

/// Single audit log entry
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

/// Audit action classification
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

// ============================================================================
// Collaborative workspace types
// ============================================================================

/// Collaborative workspace record
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

/// Newtype wrapper for workspace identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceId(pub String);

/// Session token for collaborative sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToken(pub String);

/// Newtype wrapper for message identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageId(pub String);

/// Newtype wrapper for decision identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionId(pub String);

/// Workspace member record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMember {
    pub user_id: String,
    pub role: WorkspaceRole,
    pub permissions: UserPermissions,
    pub joined_at: SystemTime,
    pub last_active: SystemTime,
}

/// Role of a workspace member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkspaceRole {
    Owner,
    Admin,
    Editor,
    Viewer,
    Guest,
}

/// Fine-grained permissions for a workspace user
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

/// Access-control settings for a workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspacePermissions {
    pub public_readable: bool,
    pub allow_anonymous_access: bool,
    pub require_approval_for_members: bool,
    pub default_member_permissions: UserPermissions,
}

/// Workspace behaviour configuration
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

/// Request to create a new workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWorkspaceRequest {
    pub name: String,
    pub description: Option<String>,
    pub owner: String,
    pub initial_members: Vec<WorkspaceMember>,
    pub permissions: WorkspacePermissions,
    pub settings: Option<WorkspaceSettings>,
}

/// User identity and preferences for a collaborative session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub name: String,
    pub email: String,
    pub avatar_url: Option<String>,
    pub timezone: String,
    pub preferences: UserPreferences,
}

/// User notification and presence preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub enable_notifications: bool,
    pub notification_types: Vec<NotificationType>,
    pub presence_status: PresenceStatus,
    pub auto_join_calls: bool,
}

/// Online presence status for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PresenceStatus {
    Online,
    Away,
    Busy,
    DoNotDisturb,
    Offline,
}

/// Active collaborative session record
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

/// Real-time presence record for a user in a workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    pub user_id: String,
    pub status: PresenceStatus,
    pub last_seen: SystemTime,
    pub current_activity: Option<String>,
    pub cursor_position: Option<CursorPosition>,
    pub viewing_document: Option<String>,
}

/// Cursor position within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub document_id: String,
    pub line: u32,
    pub column: u32,
    pub selection_start: Option<(u32, u32)>,
    pub selection_end: Option<(u32, u32)>,
}

/// Collaborative chat/event message
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

/// Classification of a collaborative message
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

/// File attachment on a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub file_id: String,
    pub filename: String,
    pub file_type: String,
    pub size_bytes: u64,
    pub url: String,
}

/// Emoji reaction on a message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageReaction {
    pub emoji: String,
    pub user_id: String,
    pub timestamp: SystemTime,
}

/// Shared document record
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

/// Classification of a shared document
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

/// Access-control settings for a shared document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPermissions {
    pub public_readable: bool,
    pub editors: Vec<String>,
    pub viewers: Vec<String>,
    pub allow_comments: bool,
    pub allow_suggestions: bool,
}

/// An active collaborative editing session on a document
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

/// Atomic text editing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditOperation {
    pub operation_id: String,
    pub operation_type: OperationType,
    pub position: TextPosition,
    pub content: String,
    pub timestamp: SystemTime,
    pub user_id: String,
}

/// Classification of an edit operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Delete,
    Replace,
    FormatApply,
    Comment,
    Suggestion,
}

/// Position within a text document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPosition {
    pub line: u32,
    pub column: u32,
    pub offset: u32,
}

/// Active collaboration record within a workspace
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

/// Category of a collaboration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationType {
    DocumentEditing,
    VideoCall,
    ScreenShare,
    Brainstorming,
    DecisionMaking,
    ReviewSession,
}

/// Lifecycle state of a collaboration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationStatus {
    Active,
    Paused,
    Completed,
    Cancelled,
}

// ============================================================================
// Decision-making types
// ============================================================================

/// Request to start a collaborative decision process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub options: Vec<DecisionOption>,
    pub eligible_voters: Vec<String>,
    pub deadline: Option<SystemTime>,
}

/// Active decision process record
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
    pub votes: HashMap<String, Vote>,
    pub comments: Vec<DecisionComment>,
    pub deadline: Option<SystemTime>,
    pub status: DecisionStatus,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
}

/// Voting methodology for a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    SingleChoice,
    MultipleChoice,
    Ranking,
    YesNo,
    Consensus,
    Budget,
}

/// A votable option in a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub proposed_by: String,
    pub vote_count: u32,
}

/// A vote cast by a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter_id: String,
    pub option_ids: Vec<String>,
    pub ranking: Option<Vec<String>>,
    pub comment: Option<String>,
    pub timestamp: SystemTime,
}

/// A comment on a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionComment {
    pub id: String,
    pub author_id: String,
    pub content: String,
    pub timestamp: SystemTime,
    pub replies: Vec<DecisionComment>,
}

/// Lifecycle state of a decision process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionStatus {
    Open,
    Closed,
    Cancelled,
    Implemented,
}

// ============================================================================
// Activity feed types
// ============================================================================

/// A workspace activity feed event
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

/// Classification of a workspace activity event
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

// ============================================================================
// Service struct declarations (implementations live in workflow_engine.rs)
// ============================================================================

/// Workflow manager for business process integration
pub struct WorkflowManager {
    pub(crate) config: WorkflowConfig,
    pub(crate) active_tasks: HashMap<String, Task>,
    pub(crate) approval_queue: Vec<ApprovalRequest>,
    pub(crate) audit_logger: AuditLogger,
    pub(crate) notification_service: NotificationService,
    pub(crate) report_generator: ReportGenerator,
    pub(crate) data_exporter: DataExporter,
}

/// Internal audit logging service (implementation in workflow_engine.rs)
pub(crate) struct AuditLogger {
    pub(crate) config: WorkflowConfig,
}

/// Notification delivery service
pub(crate) struct NotificationService {
    #[allow(dead_code)]
    pub(crate) config: WorkflowConfig,
}

/// Report generation service
pub(crate) struct ReportGenerator {
    pub(crate) config: WorkflowConfig,
}

/// Data export service
pub(crate) struct DataExporter {
    pub(crate) config: WorkflowConfig,
}

/// Collaborative workspace manager for real-time team coordination
pub struct CollaborativeWorkspaceManager {
    pub(crate) workspaces: HashMap<String, CollaborativeWorkspace>,
    pub(crate) active_sessions: HashMap<String, CollaborativeSession>,
    pub(crate) presence_tracker: PresenceTracker,
    pub(crate) message_bus: CollaborativeMessageBus,
    pub(crate) shared_document_manager: SharedDocumentManager,
    pub(crate) decision_tracker: CollaborativeDecisionTracker,
}

/// Presence tracking for real-time collaboration awareness
pub struct PresenceTracker {
    pub(crate) workspace_presence: HashMap<String, Vec<UserPresence>>,
}

/// Real-time messaging system for collaboration
pub struct CollaborativeMessageBus {
    pub(crate) message_history: HashMap<String, Vec<CollaborativeMessage>>,
    pub(crate) _subscribers: HashMap<String, Vec<String>>,
}

/// Shared document management for collaborative editing
pub struct SharedDocumentManager {
    pub(crate) _documents: HashMap<String, SharedDocument>,
    pub(crate) editing_sessions: HashMap<String, Vec<CollaborativeEditingSession>>,
}

/// Collaborative decision-making system
pub struct CollaborativeDecisionTracker {
    pub(crate) active_decisions: HashMap<String, DecisionProcess>,
}
