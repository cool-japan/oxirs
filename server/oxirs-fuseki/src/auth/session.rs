//! Session management for authenticated users

use crate::auth::types::{AuthResult, User, UserSession};
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Session manager for handling user sessions
#[derive(Clone)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    timeout_seconds: i64,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(timeout_seconds: i64) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            timeout_seconds,
        }
    }

    /// Create a new session for the user
    pub async fn create_session(&self, user: User) -> FusekiResult<String> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let session = UserSession {
            user: user.clone(),
            session_id: session_id.clone(),
            created_at: now,
            expires_at: now + Duration::seconds(self.timeout_seconds),
            last_activity: now,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);

        info!("Created session for user: {}", user.username);
        Ok(session_id)
    }

    /// Validate session and return user if valid
    pub async fn validate_session(&self, session_id: &str) -> FusekiResult<AuthResult> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            let timeout = Duration::seconds(self.timeout_seconds);

            if Utc::now() > session.expires_at {
                sessions.remove(session_id);
                debug!("Session expired: {}", session_id);
                Ok(AuthResult::Expired)
            } else {
                session.last_activity = Utc::now();
                Ok(AuthResult::Authenticated(session.user.clone()))
            }
        } else {
            debug!("Session not found: {}", session_id);
            Ok(AuthResult::Unauthenticated)
        }
    }

    /// Invalidate a session (logout)
    pub async fn invalidate_session(&self, session_id: &str) -> FusekiResult<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(session_id) {
            info!("User logged out: {}", session.user.username);
        }
        Ok(())
    }

    /// Invalidate all sessions for a user
    pub async fn invalidate_user_sessions(&self, username: &str) -> FusekiResult<()> {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|_, session| session.user.username != username);
        info!("Invalidated all sessions for user: {}", username);
        Ok(())
    }

    /// Get active session count
    pub async fn active_session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired_sessions(&self) -> FusekiResult<usize> {
        let mut sessions = self.sessions.write().await;
        let timeout = Duration::seconds(self.timeout_seconds);
        let now = Utc::now();
        let initial_count = sessions.len();

        sessions.retain(|_, session| now <= session.expires_at);

        let removed_count = initial_count - sessions.len();
        if removed_count > 0 {
            debug!("Cleaned up {} expired sessions", removed_count);
        }
        Ok(removed_count)
    }

    /// Get session information
    pub async fn get_session(&self, session_id: &str) -> Option<UserSession> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// List all active sessions for a user
    pub async fn get_user_sessions(&self, username: &str) -> Vec<UserSession> {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|session| session.user.username == username)
            .cloned()
            .collect()
    }

    /// Create JWT token for user
    #[cfg(feature = "auth")]
    pub fn create_jwt_token(&self, user: &User) -> FusekiResult<String> {
        // TODO: Implement actual JWT token creation
        // For now, return a placeholder token
        Ok(format!("jwt_token_for_{}", user.username))
    }

    /// Create JWT token for user (feature-gated fallback)
    #[cfg(not(feature = "auth"))]
    pub fn create_jwt_token(&self, _user: &User) -> FusekiResult<String> {
        Err(FusekiError::service_unavailable("JWT auth not enabled"))
    }

    /// Validate JWT token
    #[cfg(feature = "auth")]
    pub fn validate_jwt_token(
        &self,
        token: &str,
    ) -> FusekiResult<crate::auth::types::TokenValidation> {
        // TODO: Implement actual JWT token validation
        // For now, return a placeholder validation for valid tokens
        if token.starts_with("jwt_token_for_") {
            let username = token.strip_prefix("jwt_token_for_").unwrap_or("unknown");
            Ok(crate::auth::types::TokenValidation {
                user: User {
                    username: username.to_string(),
                    roles: vec!["user".to_string()], // Default role
                    email: None,
                    full_name: None,
                    last_login: Some(chrono::Utc::now()),
                    permissions: vec![], // Empty permissions for now
                },
                expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
            })
        } else {
            Err(FusekiError::authentication("Invalid JWT token"))
        }
    }

    /// Validate JWT token (feature-gated fallback)
    #[cfg(not(feature = "auth"))]
    pub fn validate_jwt_token(
        &self,
        _token: &str,
    ) -> FusekiResult<crate::auth::types::TokenValidation> {
        Err(FusekiError::service_unavailable("JWT auth not enabled"))
    }
}
