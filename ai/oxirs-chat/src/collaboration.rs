//! Real-Time Collaboration Module
//!
//! Provides real-time collaboration features for multi-user chat sessions including:
//! - Shared session support with multi-user participation
//! - Real-time presence indicators
//! - Live cursor position sharing
//! - Collaborative query building
//! - Synchronized message streaming

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::info;
use uuid::Uuid;

/// Collaborative session manager
pub struct CollaborationManager {
    /// Shared sessions indexed by session ID
    shared_sessions: Arc<RwLock<HashMap<String, SharedSession>>>,
    /// Broadcast channel for real-time updates
    update_tx: broadcast::Sender<CollaborationUpdate>,
    /// Configuration
    config: CollaborationConfig,
}

/// Configuration for collaboration features
#[derive(Debug, Clone)]
pub struct CollaborationConfig {
    /// Maximum users per shared session
    pub max_users_per_session: usize,
    /// Enable cursor position sharing
    pub enable_cursor_sharing: bool,
    /// Enable presence indicators
    pub enable_presence: bool,
    /// Session idle timeout (seconds)
    pub idle_timeout_secs: u64,
    /// Update broadcast buffer size
    pub broadcast_buffer_size: usize,
}

impl Default for CollaborationConfig {
    fn default() -> Self {
        Self {
            max_users_per_session: 10,
            enable_cursor_sharing: true,
            enable_presence: true,
            idle_timeout_secs: 1800, // 30 minutes
            broadcast_buffer_size: 1000,
        }
    }
}

/// Shared session allowing multiple users to collaborate
#[derive(Debug, Clone)]
pub struct SharedSession {
    /// Unique session ID
    pub session_id: String,
    /// Session owner/creator
    pub owner_id: String,
    /// Currently connected participants
    pub participants: HashMap<String, Participant>,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Session metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Access control settings
    pub access_control: AccessControl,
}

/// Participant in a shared session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// User ID
    pub user_id: String,
    /// Display name
    pub display_name: String,
    /// Joined timestamp
    pub joined_at: DateTime<Utc>,
    /// Last seen timestamp
    pub last_seen: DateTime<Utc>,
    /// Current cursor position (if cursor sharing enabled)
    pub cursor_position: Option<CursorPosition>,
    /// User role in the session
    pub role: ParticipantRole,
    /// Current status
    pub status: ParticipantStatus,
    /// User avatar/color for UI display
    pub avatar_color: String,
}

/// Cursor position in collaborative editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    /// Line number (0-indexed)
    pub line: usize,
    /// Column number (0-indexed)
    pub column: usize,
    /// Selected text range (if any)
    pub selection: Option<TextRange>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Text range for selections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRange {
    pub start_line: usize,
    pub start_column: usize,
    pub end_line: usize,
    pub end_column: usize,
}

/// Participant role
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantRole {
    /// Session owner with full control
    Owner,
    /// Editor with write permissions
    Editor,
    /// Viewer with read-only access
    Viewer,
}

/// Participant status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantStatus {
    /// Actively engaged
    Active,
    /// Connected but idle
    Idle,
    /// Temporarily away
    Away,
    /// Disconnected
    Offline,
}

/// Access control for shared sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Is session public or private
    pub is_public: bool,
    /// Allowed user IDs (if private)
    pub allowed_users: HashSet<String>,
    /// Require approval for new participants
    pub require_approval: bool,
    /// Maximum number of participants
    pub max_participants: usize,
}

impl Default for AccessControl {
    fn default() -> Self {
        Self {
            is_public: false,
            allowed_users: HashSet::new(),
            require_approval: false,
            max_participants: 10,
        }
    }
}

/// Collaboration update event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CollaborationUpdate {
    /// User joined session
    #[serde(rename = "user_joined")]
    UserJoined {
        session_id: String,
        participant: Participant,
    },
    /// User left session
    #[serde(rename = "user_left")]
    UserLeft { session_id: String, user_id: String },
    /// Cursor position updated
    #[serde(rename = "cursor_moved")]
    CursorMoved {
        session_id: String,
        user_id: String,
        position: CursorPosition,
    },
    /// Participant status changed
    #[serde(rename = "status_changed")]
    StatusChanged {
        session_id: String,
        user_id: String,
        status: ParticipantStatus,
    },
    /// Session metadata updated
    #[serde(rename = "metadata_updated")]
    MetadataUpdated {
        session_id: String,
        metadata: HashMap<String, serde_json::Value>,
    },
    /// Query being collaboratively built
    #[serde(rename = "query_update")]
    QueryUpdate {
        session_id: String,
        user_id: String,
        query_text: String,
        cursor_position: Option<CursorPosition>,
    },
}

impl CollaborationManager {
    /// Create a new collaboration manager
    pub fn new(config: CollaborationConfig) -> Self {
        let (update_tx, _) = broadcast::channel(config.broadcast_buffer_size);

        Self {
            shared_sessions: Arc::new(RwLock::new(HashMap::new())),
            update_tx,
            config,
        }
    }

    /// Create a new shared session
    pub async fn create_shared_session(
        &self,
        owner_id: String,
        access_control: Option<AccessControl>,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        // Generate a random avatar color for the owner
        let avatar_color = Self::generate_avatar_color();

        let owner = Participant {
            user_id: owner_id.clone(),
            display_name: format!("User {}", &owner_id[..owner_id.len().min(8)]),
            joined_at: now,
            last_seen: now,
            cursor_position: None,
            role: ParticipantRole::Owner,
            status: ParticipantStatus::Active,
            avatar_color,
        };

        let mut participants = HashMap::new();
        participants.insert(owner_id.clone(), owner.clone());

        let session = SharedSession {
            session_id: session_id.clone(),
            owner_id,
            participants,
            created_at: now,
            last_activity: now,
            metadata: HashMap::new(),
            access_control: access_control.unwrap_or_default(),
        };

        let mut sessions = self.shared_sessions.write().await;
        sessions.insert(session_id.clone(), session);

        info!("Created shared session: {}", session_id);

        Ok(session_id)
    }

    /// Add a participant to a shared session
    pub async fn join_session(
        &self,
        session_id: &str,
        user_id: String,
        display_name: Option<String>,
    ) -> Result<()> {
        let mut sessions = self.shared_sessions.write().await;

        let session = sessions.get_mut(session_id).context("Session not found")?;

        // Check access control
        if !session.access_control.is_public
            && !session.access_control.allowed_users.contains(&user_id)
            && session.owner_id != user_id
        {
            anyhow::bail!("Access denied to session");
        }

        // Check participant limit
        if session.participants.len() >= session.access_control.max_participants {
            anyhow::bail!("Session has reached maximum participants");
        }

        let now = Utc::now();
        let avatar_color = Self::generate_avatar_color();

        let participant = Participant {
            user_id: user_id.clone(),
            display_name: display_name
                .unwrap_or_else(|| format!("User {}", &user_id[..user_id.len().min(8)])),
            joined_at: now,
            last_seen: now,
            cursor_position: None,
            role: ParticipantRole::Editor,
            status: ParticipantStatus::Active,
            avatar_color,
        };

        session
            .participants
            .insert(user_id.clone(), participant.clone());
        session.last_activity = now;

        // Broadcast join event
        let _ = self.update_tx.send(CollaborationUpdate::UserJoined {
            session_id: session_id.to_string(),
            participant,
        });

        info!("User {} joined session {}", user_id, session_id);

        Ok(())
    }

    /// Remove a participant from a session
    pub async fn leave_session(&self, session_id: &str, user_id: &str) -> Result<()> {
        let mut sessions = self.shared_sessions.write().await;

        let session = sessions.get_mut(session_id).context("Session not found")?;

        session.participants.remove(user_id);
        session.last_activity = Utc::now();

        // Broadcast leave event
        let _ = self.update_tx.send(CollaborationUpdate::UserLeft {
            session_id: session_id.to_string(),
            user_id: user_id.to_string(),
        });

        info!("User {} left session {}", user_id, session_id);

        // Remove session if empty
        if session.participants.is_empty() {
            sessions.remove(session_id);
            info!("Removed empty session {}", session_id);
        }

        Ok(())
    }

    /// Update cursor position for a user
    pub async fn update_cursor(
        &self,
        session_id: &str,
        user_id: &str,
        position: CursorPosition,
    ) -> Result<()> {
        if !self.config.enable_cursor_sharing {
            return Ok(());
        }

        let mut sessions = self.shared_sessions.write().await;

        let session = sessions.get_mut(session_id).context("Session not found")?;

        if let Some(participant) = session.participants.get_mut(user_id) {
            participant.cursor_position = Some(position.clone());
            participant.last_seen = Utc::now();
            session.last_activity = Utc::now();

            // Broadcast cursor update
            let _ = self.update_tx.send(CollaborationUpdate::CursorMoved {
                session_id: session_id.to_string(),
                user_id: user_id.to_string(),
                position,
            });
        }

        Ok(())
    }

    /// Update participant status
    pub async fn update_status(
        &self,
        session_id: &str,
        user_id: &str,
        status: ParticipantStatus,
    ) -> Result<()> {
        let mut sessions = self.shared_sessions.write().await;

        let session = sessions.get_mut(session_id).context("Session not found")?;

        if let Some(participant) = session.participants.get_mut(user_id) {
            participant.status = status;
            participant.last_seen = Utc::now();
            session.last_activity = Utc::now();

            // Broadcast status update
            let _ = self.update_tx.send(CollaborationUpdate::StatusChanged {
                session_id: session_id.to_string(),
                user_id: user_id.to_string(),
                status,
            });
        }

        Ok(())
    }

    /// Broadcast a collaborative query update
    pub async fn broadcast_query_update(
        &self,
        session_id: &str,
        user_id: &str,
        query_text: String,
        cursor_position: Option<CursorPosition>,
    ) -> Result<()> {
        let sessions = self.shared_sessions.read().await;

        if !sessions.contains_key(session_id) {
            anyhow::bail!("Session not found");
        }

        // Broadcast query update
        let _ = self.update_tx.send(CollaborationUpdate::QueryUpdate {
            session_id: session_id.to_string(),
            user_id: user_id.to_string(),
            query_text,
            cursor_position,
        });

        Ok(())
    }

    /// Get session information
    pub async fn get_session(&self, session_id: &str) -> Option<SharedSession> {
        let sessions = self.shared_sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<String> {
        let sessions = self.shared_sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// Get participants in a session
    pub async fn get_participants(&self, session_id: &str) -> Option<Vec<Participant>> {
        let sessions = self.shared_sessions.read().await;
        sessions
            .get(session_id)
            .map(|s| s.participants.values().cloned().collect())
    }

    /// Subscribe to collaboration updates
    pub fn subscribe(&self) -> broadcast::Receiver<CollaborationUpdate> {
        self.update_tx.subscribe()
    }

    /// Clean up idle sessions
    pub async fn cleanup_idle_sessions(&self) -> usize {
        let mut sessions = self.shared_sessions.write().await;
        let idle_threshold = chrono::Duration::seconds(self.config.idle_timeout_secs as i64);
        let now = Utc::now();

        let mut removed_count = 0;
        let idle_sessions: Vec<String> = sessions
            .iter()
            .filter(|(_, session)| {
                now.signed_duration_since(session.last_activity) > idle_threshold
            })
            .map(|(id, _)| id.clone())
            .collect();

        for session_id in idle_sessions {
            sessions.remove(&session_id);
            removed_count += 1;
            info!("Removed idle session: {}", session_id);
        }

        removed_count
    }

    /// Generate a random avatar color for visual distinction
    fn generate_avatar_color() -> String {
        let colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2",
            "#F8B195", "#C06C84",
        ];

        let index = fastrand::usize(..colors.len());
        colors[index].to_string()
    }
}

/// Statistics for collaboration features
#[derive(Debug, Serialize, Deserialize)]
pub struct CollaborationStats {
    /// Total number of active shared sessions
    pub active_sessions: usize,
    /// Total number of participants across all sessions
    pub total_participants: usize,
    /// Average participants per session
    pub avg_participants_per_session: f64,
    /// Sessions by participant count
    pub sessions_by_size: HashMap<usize, usize>,
}

impl CollaborationManager {
    /// Get collaboration statistics
    pub async fn get_stats(&self) -> CollaborationStats {
        let sessions = self.shared_sessions.read().await;

        let active_sessions = sessions.len();
        let mut total_participants = 0;
        let mut sessions_by_size: HashMap<usize, usize> = HashMap::new();

        for session in sessions.values() {
            let participant_count = session.participants.len();
            total_participants += participant_count;
            *sessions_by_size.entry(participant_count).or_insert(0) += 1;
        }

        let avg_participants_per_session = if active_sessions > 0 {
            total_participants as f64 / active_sessions as f64
        } else {
            0.0
        };

        CollaborationStats {
            active_sessions,
            total_participants,
            avg_participants_per_session,
            sessions_by_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_shared_session() {
        let config = CollaborationConfig::default();
        let manager = CollaborationManager::new(config);

        let session_id = manager
            .create_shared_session("user1".to_string(), None)
            .await
            .unwrap();

        assert!(!session_id.is_empty());

        let session = manager.get_session(&session_id).await.unwrap();
        assert_eq!(session.owner_id, "user1");
        assert_eq!(session.participants.len(), 1);
    }

    #[tokio::test]
    async fn test_join_session() {
        let config = CollaborationConfig::default();
        let manager = CollaborationManager::new(config);

        let session_id = manager
            .create_shared_session(
                "user1".to_string(),
                Some(AccessControl {
                    is_public: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        manager
            .join_session(&session_id, "user2".to_string(), Some("User 2".to_string()))
            .await
            .unwrap();

        let participants = manager.get_participants(&session_id).await.unwrap();
        assert_eq!(participants.len(), 2);
    }

    #[tokio::test]
    async fn test_cursor_update() {
        let config = CollaborationConfig::default();
        let manager = CollaborationManager::new(config);

        let session_id = manager
            .create_shared_session("user1".to_string(), None)
            .await
            .unwrap();

        let position = CursorPosition {
            line: 10,
            column: 5,
            selection: None,
            updated_at: Utc::now(),
        };

        manager
            .update_cursor(&session_id, "user1", position)
            .await
            .unwrap();

        let session = manager.get_session(&session_id).await.unwrap();
        let participant = session.participants.get("user1").unwrap();
        assert!(participant.cursor_position.is_some());
    }

    #[tokio::test]
    async fn test_collaboration_stats() {
        let config = CollaborationConfig::default();
        let manager = CollaborationManager::new(config);

        let _session1 = manager
            .create_shared_session("user1".to_string(), None)
            .await
            .unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.active_sessions, 1);
        assert_eq!(stats.total_participants, 1);
    }
}
