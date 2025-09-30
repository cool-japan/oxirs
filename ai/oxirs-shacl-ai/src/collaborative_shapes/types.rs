//! Common types and enums for collaborative shapes
//!
//! This module provides the foundational data types used across
//! the collaborative shape development system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::shape::AiShape;
use crate::shape_management::{UserRole, UserPermissions};


/// Collaborative workspace
#[derive(Debug, Clone)]
pub struct Workspace {
    pub workspace_id: String,
    pub name: String,
    pub description: String,
    pub owner: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub settings: WorkspaceSettings,
    pub active_shapes: HashMap<String, CollaborativeShape>,
    pub users: HashSet<String>,
    pub permissions: WorkspacePermissions,
    pub activity_log: Vec<WorkspaceActivity>,
}

/// Collaborative shape with real-time editing capabilities
#[derive(Debug, Clone)]
pub struct CollaborativeShape {
    pub shape_id: String,
    pub base_shape: AiShape,
    pub current_editors: HashSet<String>,
    pub edit_locks: HashMap<String, EditLock>, // Property path -> lock info
    pub pending_changes: Vec<PendingChange>,
    pub collaboration_metadata: CollaborationMetadata,
    pub branches: HashMap<String, ShapeBranch>,
    pub current_branch: String,
}

/// Edit lock information
#[derive(Debug, Clone)]
pub struct EditLock {
    pub locked_by: String,
    pub locked_at: chrono::DateTime<chrono::Utc>,
    pub lock_type: LockType,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// Types of edit locks
#[derive(Debug, Clone)]
pub enum LockType {
    Exclusive,      // Only this user can edit
    Shared,         // Multiple users can edit simultaneously
    ReadOnly,       // No edits allowed
    Advisory,       // Warning but allows edits
}

/// Pending change in collaborative editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChange {
    pub change_id: String,
    pub user_id: String,
    pub change_type: ChangeType,
    pub target_element: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: ChangeStatus,
    pub conflict_info: Option<ConflictInfo>,
}

/// Types of collaborative changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    AddConstraint,
    RemoveConstraint,
    ModifyConstraint,
    AddProperty,
    RemoveProperty,
    ModifyMetadata,
    CreateBranch,
    MergeBranch,
}

/// Status of pending changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeStatus {
    Draft,
    Proposed,
    UnderReview,
    Approved,
    Rejected,
    Conflicted,
    Applied,
}

/// Conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    pub conflict_type: ConflictType,
    pub conflicting_changes: Vec<String>,
    pub severity: ConflictSeverity,
    pub auto_resolvable: bool,
    pub suggested_resolution: Option<String>,
}

/// Types of collaboration conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    ConcurrentEdit,      // Multiple users editing same element
    VersionMismatch,     // Base version changed
    PropertyConflict,    // Conflicting property constraints
    MergeConflict,       // Branch merge conflicts
    PermissionConflict,  // Insufficient permissions
    LockConflict,        // Edit lock violations
}

/// Conflict severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Collaboration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationMetadata {
    pub contributors: Vec<ContributorInfo>,
    pub creation_history: Vec<CreationEvent>,
    pub review_history: Vec<ReviewEvent>,
    pub usage_statistics: CollaborationUsageStats,
    pub quality_metrics: CollaborationQualityMetrics,
}

/// Contributor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorInfo {
    pub user_id: String,
    pub display_name: String,
    pub contribution_count: usize,
    pub last_contribution: chrono::DateTime<chrono::Utc>,
    pub expertise_areas: Vec<String>,
    pub reputation_score: f64,
}

/// Creation event in shape history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreationEvent {
    pub event_id: String,
    pub user_id: String,
    pub event_type: CreationEventType,
    pub description: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub affected_elements: Vec<String>,
}

/// Types of creation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreationEventType {
    InitialCreation,
    ConstraintAdded,
    PropertyAdded,
    MajorRefactoring,
    QualityImprovement,
    PerformanceOptimization,
}

/// Review event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewEvent {
    pub review_id: String,
    pub reviewer_id: String,
    pub review_type: ReviewType,
    pub rating: Option<f64>,
    pub comments: Vec<ReviewComment>,
    pub decision: ReviewDecision,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of reviews
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewType {
    PeerReview,
    QualityAssessment,
    PerformanceReview,
    SecurityReview,
    ComplianceReview,
}

/// Review comment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub comment_id: String,
    pub reviewer_id: String,
    pub comment_type: CommentType,
    pub text: String,
    pub target_element: Option<String>,
    pub severity: CommentSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub resolved: bool,
}

/// Types of review comments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentType {
    Suggestion,
    Issue,
    Question,
    Praise,
    Concern,
    RequiredChange,
}

/// Comment severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentSeverity {
    Info,
    Minor,
    Major,
    Blocking,
}

/// Review decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approved,
    ApprovedWithComments,
    RequestedChanges,
    Rejected,
    NeedsMoreReview,
}

/// Collaboration usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationUsageStats {
    pub total_edits: usize,
    pub concurrent_sessions: usize,
    pub average_session_duration: Duration,
    pub most_active_contributors: Vec<String>,
    pub peak_collaboration_times: Vec<chrono::DateTime<chrono::Utc>>,
}

/// Collaboration quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationQualityMetrics {
    pub conflict_rate: f64,
    pub resolution_time_avg: Duration,
    pub review_coverage: f64,
    pub contributor_satisfaction: f64,
    pub knowledge_sharing_score: f64,
}

/// Shape branch for parallel development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeBranch {
    pub branch_id: String,
    pub branch_name: String,
    pub created_by: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub parent_branch: Option<String>,
    pub base_shape: AiShape,
    pub changes: Vec<PendingChange>,
    pub merge_status: MergeStatus,
}

/// Branch merge status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStatus {
    Open,
    ReadyToMerge,
    MergeConflicts,
    Merged,
    Abandoned,
}

/// User session in workspace
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub workspace_id: String,
    pub connected_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub permissions: UserPermissions,
    pub current_shape: Option<String>,
    pub active_locks: HashSet<String>,
    pub session_state: SessionState,
}

/// Session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionState {
    Active,
    Idle,
    Disconnected,
    TimedOut,
}

/// Connection information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connection_id: String,
    pub user_id: String,
    pub connection_type: ConnectionType,
    pub connected_at: chrono::DateTime<chrono::Utc>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Types of connections
#[derive(Debug, Clone)]
pub enum ConnectionType {
    WebSocket,
    RestApi,
    Desktop,
    Mobile,
}

/// Workspace permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspacePermissions {
    pub owner_permissions: UserPermissions,
    pub member_permissions: UserPermissions,
    pub guest_permissions: UserPermissions,
    pub role_based_permissions: HashMap<UserRole, UserPermissions>,
}

/// Workspace activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceActivity {
    pub activity_id: String,
    pub user_id: String,
    pub activity_type: ActivityType,
    pub description: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub affected_resources: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Types of workspace activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    UserJoined,
    UserLeft,
    ShapeCreated,
    ShapeModified,
    ShapeDeleted,
    BranchCreated,
    BranchMerged,
    ReviewRequested,
    ReviewCompleted,
    ConflictResolved,
}

/// Cursor position in shape editing
#[derive(Debug, Clone)]
pub struct CursorPosition {
    pub shape_id: String,
    pub element_path: String,
    pub position: usize,
}

/// Presence status
#[derive(Debug, Clone)]
pub enum PresenceStatus {
    Online,
    Away,
    Busy,
    Offline,
}