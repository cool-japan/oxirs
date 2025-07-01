//! Comprehensive authentication and authorization system
//!
//! This module provides a modular authentication system supporting multiple
//! authentication methods including local users, LDAP, OAuth2, SAML, and
//! X.509 certificate authentication.

// Re-export all public types and functions from submodules
pub use types::*;
pub use session::SessionManager;
pub use permissions::PermissionChecker;
pub use password::PasswordUtils;
pub use jwt::JwtManager;

// Import submodules
pub mod types;
pub mod session;
pub mod permissions;
pub mod password;
pub mod jwt;
pub mod certificate;
pub mod ldap;
pub mod oauth;

#[cfg(feature = "saml")]
pub mod saml;

// Core imports
use crate::auth::ldap::LdapService;
use crate::auth::oauth::OAuth2Service;
#[cfg(feature = "saml")]
use crate::auth::saml::{SamlConfig, SamlProvider};
use crate::config::{SecurityConfig, UserConfig};
use crate::error::{FusekiError, FusekiResult};

use axum::{
    extract::{FromRequestParts, State},
    http::{request::Parts, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use axum_extra::headers::{authorization::Bearer, Authorization, HeaderMapExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Main authentication service that coordinates all authentication methods
#[derive(Clone)]
pub struct AuthService {
    config: Arc<SecurityConfig>,
    users: Arc<RwLock<HashMap<String, UserConfig>>>,
    session_manager: SessionManager,
    jwt_manager: Option<JwtManager>,
    oauth2_service: Option<OAuth2Service>,
    ldap_service: Option<LdapService>,
    #[cfg(feature = "saml")]
    saml_provider: Option<Arc<SamlProvider>>,
}

impl AuthService {
    /// Create a new authentication service
    pub async fn new(config: SecurityConfig) -> FusekiResult<Self> {
        let users = config.users.clone();

        // Initialize session manager
        let session_manager = SessionManager::new(config.session.timeout_secs as i64);

        // Initialize JWT manager if configured
        let jwt_manager = config.jwt.as_ref()
            .map(|jwt_config| JwtManager::new(jwt_config))
            .transpose()?;

        // Initialize OAuth2 service if configured
        let oauth2_service = config.oauth.as_ref()
            .map(|oauth_config| OAuth2Service::new(oauth_config.clone()));

        // Initialize LDAP service if configured
        let ldap_service = if let Some(ldap_config) = config.ldap.as_ref() {
            Some(LdapService::new(ldap_config.clone()).await?)
        } else {
            None
        };

        #[cfg(feature = "saml")]
        let saml_provider = None; // TODO: Initialize from config when SAML config is added

        Ok(Self {
            config: Arc::new(config),
            users: Arc::new(RwLock::new(users)),
            session_manager,
            jwt_manager,
            oauth2_service,
            ldap_service,
            #[cfg(feature = "saml")]
            saml_provider,
        })
    }

    /// Authenticate user with username/password
    pub async fn authenticate_user(&self, username: &str, password: &str) -> FusekiResult<AuthResult> {
        let users = self.users.read().await;

        // Try local user authentication first
        if let Some(user_config) = users.get(username) {
            // Check if user is enabled
            if !user_config.enabled {
                warn!("Login attempt for disabled user: {}", username);
                return Ok(AuthResult::Forbidden);
            }

            // Check if user is locked
            if let Some(locked_until) = user_config.locked_until {
                if locked_until > chrono::Utc::now() {
                    warn!("Login attempt for locked user: {} (locked until {})", username, locked_until);
                    return Ok(AuthResult::Locked);
                }
            }

            // Verify password
            if PasswordUtils::verify_password(password, &user_config.password_hash)? {
                debug!("Successful local authentication for user: {}", username);

                // Create user object with permissions
                let permissions = PermissionChecker::compute_user_permissions(user_config);
                let user = User {
                    username: username.to_string(),
                    roles: user_config.roles.clone(),
                    email: user_config.email.clone(),
                    full_name: user_config.full_name.clone(),
                    last_login: user_config.last_login,
                    permissions,
                };

                // Update last login time
                self.update_last_login(username).await?;

                return Ok(AuthResult::Authenticated(user));
            } else {
                warn!("Failed local authentication attempt for user: {}", username);
                self.increment_failed_attempts(username).await?;
            }
        }

        // If local authentication failed, try LDAP if enabled
        if let Some(ldap_service) = &self.ldap_service {
            debug!("Trying LDAP authentication for user: {}", username);
            match ldap_service.authenticate(username, password).await {
                Ok(ldap_user) => {
                    info!("Successful LDAP authentication for user: {}", username);
                    return Ok(AuthResult::Authenticated(ldap_user));
                }
                Err(e) => {
                    warn!("LDAP authentication failed for user {}: {}", username, e);
                }
            }
        }

        debug!("Authentication failed for user: {}", username);
        Ok(AuthResult::Unauthenticated)
    }

    /// Create session for authenticated user
    pub async fn create_session(&self, user: User) -> FusekiResult<String> {
        self.session_manager.create_session(user).await
    }

    /// Validate session
    pub async fn validate_session(&self, session_id: &str) -> FusekiResult<AuthResult> {
        self.session_manager.validate_session(session_id).await
    }

    /// Logout user (invalidate session)
    pub async fn logout(&self, session_id: &str) -> FusekiResult<()> {
        self.session_manager.invalidate_session(session_id).await
    }

    /// Check if user has permission
    pub fn has_permission(&self, user: &User, permission: &Permission) -> bool {
        PermissionChecker::has_permission(user, permission)
    }

    /// Generate JWT token for user
    pub fn generate_token(&self, user: &User) -> FusekiResult<String> {
        self.jwt_manager.as_ref()
            .ok_or_else(|| FusekiError::configuration("JWT is not configured".to_string()))?
            .generate_token(user)
    }

    /// Validate JWT token
    pub fn validate_token(&self, token: &str) -> FusekiResult<TokenValidation> {
        self.jwt_manager.as_ref()
            .ok_or_else(|| FusekiError::configuration("JWT is not configured".to_string()))?
            .validate_token(token)
    }

    /// Hash password
    pub fn hash_password(&self, password: &str) -> FusekiResult<String> {
        PasswordUtils::hash_password(password)
    }

    /// Add or update user
    pub async fn upsert_user(&self, username: String, user_config: UserConfig) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        users.insert(username.clone(), user_config);
        info!("User upserted: {}", username);
        Ok(())
    }

    /// Remove user
    pub async fn remove_user(&self, username: &str) -> FusekiResult<bool> {
        let mut users = self.users.write().await;
        let removed = users.remove(username).is_some();
        if removed {
            info!("User removed: {}", username);
            self.session_manager.invalidate_user_sessions(username).await?;
        }
        Ok(removed)
    }

    /// Get user information
    pub async fn get_user(&self, username: &str) -> Option<UserConfig> {
        let users = self.users.read().await;
        users.get(username).cloned()
    }

    /// List all users (for admin)
    pub async fn list_users(&self) -> HashMap<String, UserConfig> {
        let users = self.users.read().await;
        users.clone()
    }

    /// Check if LDAP is enabled
    pub fn is_ldap_enabled(&self) -> bool {
        self.ldap_service.is_some()
    }

    /// Check if OAuth2 is enabled
    pub fn is_oauth2_enabled(&self) -> bool {
        self.oauth2_service.is_some()
    }

    /// Check if JWT is enabled
    pub fn is_jwt_enabled(&self) -> bool {
        self.jwt_manager.is_some()
    }

    // Private helper methods
    async fn update_last_login(&self, username: &str) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.last_login = Some(chrono::Utc::now());
            user.failed_login_attempts = 0;
        }
        Ok(())
    }

    async fn increment_failed_attempts(&self, username: &str) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.failed_login_attempts += 1;
            if user.failed_login_attempts >= 5 {
                user.locked_until = Some(chrono::Utc::now() + chrono::Duration::minutes(15));
                warn!("User locked due to failed login attempts: {}", username);
            }
        }
        Ok(())
    }
}

/// Axum extractor for authenticated users
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
    Arc<AuthService>: axum::extract::FromRef<S>,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let auth_service = Arc::<AuthService>::from_ref(state);

        // Try to extract token from Authorization header
        if let Some(auth_header) = parts.headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if let Some(token) = JwtManager::extract_token_from_header(auth_str) {
                    match auth_service.validate_token(token) {
                        Ok(validation) => {
                            return Ok(AuthUser {
                                user: validation.user,
                            });
                        }
                        Err(e) => {
                            debug!("Token validation failed: {}", e);
                        }
                    }
                }
            }
        }

        // Try to extract session from cookies
        if let Ok(cookies) = parts.extract::<axum_extra::extract::CookieJar>().await {
            if let Some(session_cookie) = cookies.get("session_id") {
                match auth_service.validate_session(session_cookie.value()).await {
                    Ok(AuthResult::Authenticated(user)) => {
                        return Ok(AuthUser { user });
                    }
                    Ok(_) => {
                        debug!("Session validation failed");
                    }
                    Err(e) => {
                        debug!("Session validation error: {}", e);
                    }
                }
            }
        }

        // No valid authentication found
        Err(StatusCode::UNAUTHORIZED.into_response())
    }
}

/// Authenticated user wrapper for Axum
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user: User,
}

impl AuthUser {
    /// Check if the user has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        PermissionChecker::has_permission(&self.user, permission)
    }

    /// Get the username
    pub fn username(&self) -> &str {
        &self.user.username
    }

    /// Get user roles
    pub fn roles(&self) -> &[String] {
        &self.user.roles
    }

    /// Check if user has role
    pub fn has_role(&self, role: &str) -> bool {
        self.user.roles.contains(&role.to_string())
    }
}