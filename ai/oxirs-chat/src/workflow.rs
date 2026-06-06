//! Workflow Integration for OxiRS Chat
//!
//! This module provides business process integration including:
//! - Task delegation
//! - Report generation
//! - Data export
//! - Notification systems
//! - Approval workflows
//! - Audit trails

// Re-export all public types from sibling modules
pub use crate::workflow_types::{
    // Collaborative workspace types
    ActiveCollaboration,
    ActivityEvent,
    ActivityEventType,
    // Approval types
    ApprovalDecision,
    ApprovalId,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalType,
    // Audit types
    AuditAction,
    AuditEntry,
    CollaborationStatus,
    CollaborationType,
    CollaborativeDecisionTracker,
    CollaborativeEditingSession,
    CollaborativeMessage,
    CollaborativeMessageBus,
    CollaborativeSession,
    CollaborativeWorkspace,
    CollaborativeWorkspaceManager,
    CreateWorkspaceRequest,
    CursorPosition,
    // Export types
    DataType,
    DecisionComment,
    DecisionId,
    DecisionOption,
    DecisionProcess,
    DecisionRequest,
    DecisionStatus,
    DecisionType,
    DocumentPermissions,
    DocumentType,
    EditOperation,
    ExportFormat,
    ExportRequest,
    ExportResult,
    MessageAttachment,
    MessageId,
    MessageReaction,
    MessageType,
    // Notification types
    Notification,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
    OperationType,
    PresenceStatus,
    PresenceTracker,
    // Report types
    ReportFormat,
    ReportRequest,
    ReportResult,
    ReportType,
    SessionToken,
    SharedDocument,
    SharedDocumentManager,
    // Task types
    Task,
    TaskId,
    TaskPriority,
    TaskRequest,
    TaskStatus,
    TaskType,
    TextPosition,
    // Shared
    TimeRange,
    UserInfo,
    UserPermissions,
    UserPreferences,
    UserPresence,
    Vote,
    // Config
    WorkflowConfig,
    WorkflowManager,
    WorkspaceId,
    WorkspaceMember,
    WorkspacePermissions,
    WorkspaceRole,
    WorkspaceSettings,
};
