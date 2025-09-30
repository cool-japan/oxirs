//! Workspace management for collaborative shape development

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::RwLock as AsyncRwLock;

use crate::{Result, ShaclAiError};
use crate::shape::AiShape;
use crate::shape_management::UserPermissions; 
use super::types::*;

/// Workspace manager for organizing collaborative sessions
#[derive(Debug)]
pub struct WorkspaceManager {
    workspaces: Arc<AsyncRwLock<HashMap<String, Workspace>>>,
    user_sessions: Arc<AsyncRwLock<HashMap<String, UserSession>>>,
    active_connections: Arc<AsyncRwLock<HashMap<String, ConnectionInfo>>>,
}

impl WorkspaceManager {
    /// Create new workspace manager
    pub fn new() -> Self {
        Self {
            workspaces: Arc::new(AsyncRwLock::new(HashMap::new())),
            user_sessions: Arc::new(AsyncRwLock::new(HashMap::new())),
            active_connections: Arc::new(AsyncRwLock::new(HashMap::new())),
        }
    }

    /// Initialize the workspace manager
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize workspace management components
        tracing::info!("Initializing workspace manager");
        Ok(())
    }

    /// Shutdown the workspace manager
    pub async fn shutdown(&mut self) -> Result<()> {
        // Clean up resources
        tracing::info!("Shutting down workspace manager");
        Ok(())
    }
    
    /// Create a new workspace
    pub async fn create_workspace(&mut self, workspace: Workspace) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;
        workspaces.insert(workspace.workspace_id.clone(), workspace);
        Ok(())
    }
    
    /// Join an existing workspace
    pub async fn join_workspace(
        &mut self,
        workspace_id: String,
        user_id: String,
        permissions: UserPermissions,
    ) -> Result<String> {
        let session_id = format!("session_{}_{}", user_id, chrono::Utc::now().timestamp());
        
        let session = UserSession {
            session_id: session_id.clone(),
            user_id: user_id.clone(),
            workspace_id: workspace_id.clone(),
            connected_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            permissions,
            current_shape: None,
            active_locks: HashSet::new(),
            session_state: SessionState::Active,
        };
        
        let mut sessions = self.user_sessions.write().await;
        sessions.insert(session_id.clone(), session);
        
        // Add user to workspace
        let mut workspaces = self.workspaces.write().await;
        if let Some(workspace) = workspaces.get_mut(&workspace_id) {
            workspace.users.insert(user_id);
        }
        
        Ok(session_id)
    }

    /// Leave a workspace
    pub async fn leave_workspace(&mut self, session_id: String) -> Result<()> {
        let mut sessions = self.user_sessions.write().await;
        if let Some(session) = sessions.remove(&session_id) {
            // Remove user from workspace
            let mut workspaces = self.workspaces.write().await;
            if let Some(workspace) = workspaces.get_mut(&session.workspace_id) {
                workspace.users.remove(&session.user_id);
            }
        }
        Ok(())
    }
    
    /// Start collaborative shape editing
    pub async fn start_shape_editing(
        &mut self,
        workspace_id: String,
        shape_id: String,
        user_id: String,
    ) -> Result<CollaborativeShape> {
        let mut workspaces = self.workspaces.write().await;
        let workspace = workspaces.get_mut(&workspace_id)
            .ok_or_else(|| ShaclAiError::ShapeManagement(format!("Workspace {} not found", workspace_id)))?;
        
        let collaborative_shape = workspace.active_shapes
            .entry(shape_id.clone())
            .or_insert_with(|| CollaborativeShape {
                shape_id: shape_id.clone(),
                base_shape: AiShape::new(shape_id.clone()),
                current_editors: HashSet::new(),
                edit_locks: HashMap::new(),
                pending_changes: Vec::new(),
                collaboration_metadata: CollaborationMetadata {
                    contributors: Vec::new(),
                    creation_history: Vec::new(),
                    review_history: Vec::new(),
                    usage_statistics: CollaborationUsageStats {
                        total_edits: 0,
                        concurrent_sessions: 0,
                        average_session_duration: Duration::from_secs(0),
                        most_active_contributors: Vec::new(),
                        peak_collaboration_times: Vec::new(),
                    },
                    quality_metrics: CollaborationQualityMetrics {
                        conflict_rate: 0.0,
                        resolution_time_avg: Duration::from_secs(0),
                        review_coverage: 0.0,
                        contributor_satisfaction: 0.0,
                        knowledge_sharing_score: 0.0,
                    },
                },
                branches: HashMap::new(),
                current_branch: "main".to_string(),
            });
        
        collaborative_shape.current_editors.insert(user_id);
        
        Ok(collaborative_shape.clone())
    }

    /// Stop shape editing
    pub async fn stop_shape_editing(
        &mut self,
        workspace_id: String,
        shape_id: String,
        user_id: String,
    ) -> Result<()> {
        let mut workspaces = self.workspaces.write().await;
        if let Some(workspace) = workspaces.get_mut(&workspace_id) {
            if let Some(shape) = workspace.active_shapes.get_mut(&shape_id) {
                shape.current_editors.remove(&user_id);
            }
        }
        Ok(())
    }

    /// Get workspace by ID
    pub async fn get_workspace(&self, workspace_id: &str) -> Result<Option<Workspace>> {
        let workspaces = self.workspaces.read().await;
        Ok(workspaces.get(workspace_id).cloned())
    }

    /// List workspaces for user
    pub async fn list_user_workspaces(&self, user_id: &str) -> Result<Vec<Workspace>> {
        let workspaces = self.workspaces.read().await;
        let user_workspaces = workspaces
            .values()
            .filter(|workspace| workspace.users.contains(user_id))
            .cloned()
            .collect();
        Ok(user_workspaces)
    }

    /// Get active user sessions
    pub async fn get_active_sessions(&self, workspace_id: &str) -> Result<Vec<UserSession>> {
        let sessions = self.user_sessions.read().await;
        let active_sessions = sessions
            .values()
            .filter(|session| {
                session.workspace_id == workspace_id &&
                matches!(session.session_state, SessionState::Active)
            })
            .cloned()
            .collect();
        Ok(active_sessions)
    }

    /// Update session activity
    pub async fn update_session_activity(&mut self, session_id: &str) -> Result<()> {
        let mut sessions = self.user_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = chrono::Utc::now();
            session.session_state = SessionState::Active;
        }
        Ok(())
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&mut self, timeout_seconds: u64) -> Result<usize> {
        let mut sessions = self.user_sessions.write().await;
        let now = chrono::Utc::now();
        let timeout = chrono::Duration::seconds(timeout_seconds as i64);
        
        let expired_sessions: Vec<String> = sessions
            .iter()
            .filter(|(_, session)| now - session.last_activity > timeout)
            .map(|(id, _)| id.clone())
            .collect();
        
        let count = expired_sessions.len();
        for session_id in expired_sessions {
            sessions.remove(&session_id);
        }
        
        Ok(count)
    }

    /// Get workspace statistics
    pub async fn get_workspace_statistics(&self, workspace_id: &str) -> Result<WorkspaceStatistics> {
        let workspaces = self.workspaces.read().await;
        let sessions = self.user_sessions.read().await;
        
        if let Some(workspace) = workspaces.get(workspace_id) {
            let active_sessions = sessions
                .values()
                .filter(|session| {
                    session.workspace_id == workspace_id &&
                    matches!(session.session_state, SessionState::Active)
                })
                .count();
            
            Ok(WorkspaceStatistics {
                total_users: workspace.users.len(),
                active_sessions,
                total_shapes: workspace.active_shapes.len(),
                total_activities: workspace.activity_log.len(),
            })
        } else {
            Err(ShaclAiError::ShapeManagement(format!("Workspace {} not found", workspace_id)))
        }
    }
}

impl Default for WorkspaceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WorkspacePermissions {
    fn default() -> Self {
        Self {
            owner_permissions: UserPermissions::all(),
            member_permissions: UserPermissions::default(),
            guest_permissions: UserPermissions::read_only(),
            role_based_permissions: HashMap::new(),
        }
    }
}

/// Workspace usage statistics
#[derive(Debug, Clone)]
pub struct WorkspaceStatistics {
    pub total_users: usize,
    pub active_sessions: usize,
    pub total_shapes: usize,
    pub total_activities: usize,
}