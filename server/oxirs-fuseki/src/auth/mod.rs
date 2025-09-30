//! Comprehensive authentication and authorization system
//!
//! This module provides a modular authentication system with support for:
//! - Username/password authentication
//! - X.509 certificate authentication  
//! - Multi-factor authentication (TOTP, SMS, Hardware tokens)
//! - LDAP/Active Directory integration
//! - OAuth2/OIDC support
//! - SAML 2.0 authentication
//! - JWT token management
//! - Session management
//! - Role-based access control

use crate::config::{JwtConfig, LdapConfig, SecurityConfig, UserConfig};
use crate::error::{FusekiError, FusekiResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Module declarations
pub mod certificate;
pub mod ldap;
pub mod oauth;
pub mod password;
pub mod permissions;
#[cfg(feature = "saml")]
pub mod saml;
pub mod session;
pub mod types;

// Re-export key types for easy access
pub use certificate::CertificateAuthService as CertificateAuthenticator;
pub use session::SessionManager;
pub use types::*;

/// Main authentication service that coordinates all authentication methods
#[derive(Clone)]
pub struct AuthService {
    config: Arc<SecurityConfig>,
    users: Arc<RwLock<HashMap<String, UserConfig>>>,
    session_manager: Arc<SessionManager>,
    certificate_auth: Arc<CertificateAuthenticator>,
    oauth2_service: Option<oauth::OAuth2Service>,
    ldap_service: Option<ldap::LdapService>,
    #[cfg(feature = "saml")]
    saml_provider: Option<Arc<saml::SamlProvider>>,
    /// Storage for MFA challenges
    mfa_challenges: Arc<RwLock<HashMap<String, MfaChallenge>>>,
}

impl AuthService {
    /// Create a new authentication service
    pub async fn new(config: SecurityConfig) -> FusekiResult<Self> {
        let users = config.users.clone();
        let config_arc = Arc::new(config);

        // Initialize session manager
        let session_manager = Arc::new(SessionManager::new(config_arc.session.timeout_secs as i64));

        // Initialize certificate authenticator
        let certificate_auth = Arc::new(CertificateAuthenticator::new(config_arc.clone()));

        // Initialize OAuth2 service if configured
        let oauth2_service = config_arc
            .oauth
            .as_ref()
            .map(|oauth_config| oauth::OAuth2Service::new(oauth_config.clone()));

        // Initialize LDAP service if configured
        let ldap_service = if let Some(ldap_config) = config_arc.ldap.as_ref() {
            Some(ldap::LdapService::new(ldap_config.clone()).await?)
        } else {
            None
        };

        Ok(Self {
            config: config_arc,
            users: Arc::new(RwLock::new(users)),
            session_manager,
            certificate_auth,
            oauth2_service,
            ldap_service,
            #[cfg(feature = "saml")]
            saml_provider: None, // TODO: Initialize from config when SAML config is added
            mfa_challenges: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Authenticate user with username/password
    pub async fn authenticate_user(
        &self,
        username: &str,
        password: &str,
    ) -> FusekiResult<AuthResult> {
        let users = self.users.read().await;

        // Try local user authentication first
        if let Some(user_config) = users.get(username) {
            // Check if user is enabled
            if !user_config.enabled {
                info!("Login attempt for disabled user: {}", username);
                return Ok(AuthResult::Forbidden);
            }

            // Verify password using password module
            if password::PasswordUtils::verify_password(password, &user_config.password_hash)? {
                debug!("Successful local authentication for user: {}", username);

                // Create user object with permissions
                let permissions =
                    permissions::PermissionChecker::compute_user_permissions(user_config);
                let user = User {
                    username: username.to_string(),
                    roles: user_config.roles.clone(),
                    email: user_config.email.clone(),
                    full_name: user_config.full_name.clone(),
                    last_login: user_config.last_login,
                    permissions,
                };

                return Ok(AuthResult::Authenticated(user));
            }
        }

        // If local authentication failed, try LDAP if enabled
        if let Some(ldap_service) = &self.ldap_service {
            debug!("Trying LDAP authentication for user: {}", username);
            return ldap_service
                .authenticate_ldap_user(username, password)
                .await;
        }

        Ok(AuthResult::Unauthenticated)
    }

    /// Authenticate using X.509 certificate
    pub async fn authenticate_certificate(&self, cert_data: &[u8]) -> FusekiResult<AuthResult> {
        self.certificate_auth
            .authenticate_certificate(cert_data)
            .await
    }

    /// Create a new session for authenticated user
    pub async fn create_session(&self, user: User) -> FusekiResult<String> {
        self.session_manager.create_session(user).await
    }

    /// Validate an existing session
    pub async fn validate_session(&self, session_id: &str) -> FusekiResult<Option<User>> {
        match self.session_manager.validate_session(session_id).await? {
            AuthResult::Authenticated(user) => Ok(Some(user)),
            _ => Ok(None),
        }
    }

    /// Logout a session
    pub async fn logout(&self, session_id: &str) -> FusekiResult<bool> {
        self.session_manager
            .invalidate_session(session_id)
            .await
            .map(|_| true)
    }

    /// Create JWT token for user
    pub fn create_jwt_token(&self, user: &User) -> FusekiResult<String> {
        self.session_manager.create_jwt_token(user)
    }

    /// Validate JWT token
    pub fn validate_jwt_token(&self, token: &str) -> FusekiResult<TokenValidation> {
        self.session_manager.validate_jwt_token(token)
    }

    /// OAuth2 authentication URL
    pub fn get_oauth2_auth_url(&self, state: &str) -> FusekiResult<String> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .get_auth_url(state)
    }

    /// Complete OAuth2 authentication
    pub async fn complete_oauth2_authentication(
        &self,
        code: &str,
        state: &str,
        redirect_uri: &str,
    ) -> FusekiResult<AuthResult> {
        let oauth2_service = self
            .oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?;

        // Exchange authorization code for access token
        let token = oauth2_service
            .exchange_code_for_token(code, state, redirect_uri)
            .await?;

        // Get user information using the access token
        let user_info = oauth2_service.get_user_info(&token.access_token).await?;

        // Extract user details from OAuth2 user info
        let username = user_info.sub.clone();
        let email = user_info.email.clone();
        let full_name = user_info.name.clone();

        // Create user object with default permissions
        // In a real implementation, you might want to map OAuth2 groups/roles to internal roles
        let roles = vec!["user".to_string()]; // Default role for OAuth2 users

        // Compute permissions for the roles
        let mut permissions = std::collections::HashSet::new();
        for role in &roles {
            if let Some(role_permissions) =
                permissions::PermissionChecker::get_role_permissions(role)
            {
                permissions.extend(role_permissions);
            }
        }
        let permissions: Vec<_> = permissions.into_iter().collect();

        let user = User {
            username: username.clone(),
            roles,
            email,
            full_name,
            last_login: Some(chrono::Utc::now()),
            permissions,
        };

        debug!("Successful OAuth2 authentication for user: {}", username);
        Ok(AuthResult::Authenticated(user))
    }

    /// Check if OAuth2 authentication is enabled
    pub fn is_oauth2_enabled(&self) -> bool {
        self.oauth2_service.is_some()
    }

    /// Generate OAuth2 authorization URL (delegating to OAuth2Service)
    pub async fn generate_oauth2_auth_url(
        &self,
        redirect_uri: &str,
        scopes: &[String],
        use_pkce: bool,
    ) -> FusekiResult<(String, String)> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .generate_authorization_url(redirect_uri, scopes, use_pkce)
            .await
    }

    /// Validate OAuth2 access token (delegating to OAuth2Service)
    pub async fn validate_access_token(&self, access_token: &str) -> FusekiResult<bool> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .validate_access_token(access_token)
            .await
    }

    /// Get OAuth configuration for discovery endpoint
    pub fn get_oauth_config(&self) -> Option<&crate::config::OAuthConfig> {
        self.config.oauth.as_ref()
    }

    /// Get OAuth2 user information (delegating to OAuth2Service)
    pub async fn get_oauth2_user_info(
        &self,
        access_token: &str,
    ) -> FusekiResult<oauth::OIDCUserInfo> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .get_user_info(access_token)
            .await
    }

    /// Refresh OAuth2 access token (delegating to OAuth2Service)
    pub async fn refresh_oauth2_token(
        &self,
        refresh_token: &str,
    ) -> FusekiResult<oauth::OAuth2Token> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .refresh_token(refresh_token)
            .await
    }

    /// SAML authentication methods
    #[cfg(feature = "saml")]
    pub async fn generate_saml_auth_request(
        &self,
        target_url: &str,
        force_authn: bool,
        relay_state: Option<&str>,
    ) -> FusekiResult<String> {
        if let Some(saml_provider) = &self.saml_provider {
            saml_provider
                .generate_auth_request(target_url, force_authn, relay_state)
                .await
        } else {
            Err(FusekiError::configuration("SAML not configured"))
        }
    }

    /// Get configuration reference
    pub fn config(&self) -> &SecurityConfig {
        &self.config
    }

    /// Get session manager reference
    pub fn session_manager(&self) -> &SessionManager {
        &self.session_manager
    }

    /// Get user configuration by username
    pub async fn get_user(&self, username: &str) -> Option<UserConfig> {
        let users = self.users.read().await;
        users.get(username).cloned()
    }

    /// Hash a password using bcrypt
    pub fn hash_password(&self, password: &str) -> FusekiResult<String> {
        #[cfg(feature = "auth")]
        {
            use bcrypt::{hash, DEFAULT_COST};
            hash(password, DEFAULT_COST)
                .map_err(|e| FusekiError::authentication(format!("Failed to hash password: {e}")))
        }
        #[cfg(not(feature = "auth"))]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            password.hash(&mut hasher);
            Ok(format!("hash_{:x}", hasher.finish()))
        }
    }

    /// Verify password against bcrypt hash
    pub fn verify_password(&self, password: &str, hash: &str) -> FusekiResult<bool> {
        #[cfg(feature = "auth")]
        {
            use bcrypt::verify;
            verify(password, hash)
                .map_err(|e| FusekiError::authentication(format!("Failed to verify password: {e}")))
        }
        #[cfg(not(feature = "auth"))]
        {
            let computed_hash = self.hash_password(password)?;
            Ok(computed_hash == hash)
        }
    }

    /// Add or update user
    pub async fn upsert_user(&self, username: String, config: UserConfig) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        users.insert(username, config);
        Ok(())
    }

    /// Remove user by username
    pub async fn remove_user(&self, username: &str) -> FusekiResult<bool> {
        let mut users = self.users.write().await;
        Ok(users.remove(username).is_some())
    }

    /// Check if LDAP authentication is enabled
    pub fn is_ldap_enabled(&self) -> bool {
        self.ldap_service.is_some()
    }

    /// Authenticate user against LDAP
    pub async fn authenticate_ldap(
        &self,
        username: &str,
        password: &str,
    ) -> FusekiResult<AuthResult> {
        if let Some(ref ldap_service) = self.ldap_service {
            ldap_service
                .authenticate_ldap_user(username, password)
                .await
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Get JWT configuration
    pub fn jwt_config(&self) -> Option<&JwtConfig> {
        self.config.jwt.as_ref()
    }

    /// Generate JWT token for user
    pub async fn generate_jwt_token(&self, user: &User) -> FusekiResult<String> {
        self.create_jwt_token(user)
    }

    /// Test LDAP connection
    pub async fn test_ldap_connection(&self) -> FusekiResult<bool> {
        if let Some(ref ldap_service) = self.ldap_service {
            ldap_service.test_connection().await
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Get LDAP configuration
    pub fn ldap_config(&self) -> Option<&LdapConfig> {
        self.config.ldap.as_ref()
    }

    /// Get user LDAP groups
    pub async fn get_ldap_user_groups(&self, username: &str) -> FusekiResult<Vec<String>> {
        if let Some(ref ldap_service) = self.ldap_service {
            let groups = ldap_service.get_user_groups(username).await?;
            Ok(groups.into_iter().map(|group| group.cn).collect())
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Store MFA challenge
    pub async fn store_mfa_challenge(
        &self,
        challenge_id: &str,
        challenge: MfaChallenge,
    ) -> FusekiResult<()> {
        let mut challenges = self.mfa_challenges.write().await;
        challenges.insert(challenge_id.to_string(), challenge);
        debug!("Stored MFA challenge: {}", challenge_id);
        Ok(())
    }

    /// Get MFA challenge
    pub async fn get_mfa_challenge(
        &self,
        challenge_id: &str,
    ) -> FusekiResult<Option<MfaChallenge>> {
        let challenges = self.mfa_challenges.read().await;
        Ok(challenges.get(challenge_id).cloned())
    }

    /// Remove MFA challenge
    pub async fn remove_mfa_challenge(&self, challenge_id: &str) -> FusekiResult<bool> {
        let mut challenges = self.mfa_challenges.write().await;
        let removed = challenges.remove(challenge_id).is_some();
        if removed {
            debug!("Removed MFA challenge: {}", challenge_id);
        }
        Ok(removed)
    }

    /// Store MFA email for user
    pub async fn store_mfa_email(&self, username: &str, _email: &str) -> FusekiResult<()> {
        // TODO: Implement MFA email storage in user profile
        info!("Storing MFA email for user: {}", username);
        Ok(())
    }

    /// Get user's SMS phone number
    pub async fn get_user_sms_phone(&self, _username: &str) -> FusekiResult<Option<String>> {
        // TODO: Implement SMS phone retrieval from user profile
        Ok(Some("+1-555-0123".to_string())) // Placeholder
    }

    /// Get user's MFA email
    pub async fn get_user_mfa_email(&self, _username: &str) -> FusekiResult<Option<String>> {
        // TODO: Implement MFA email retrieval from user profile
        Ok(Some("user@example.com".to_string())) // Placeholder
    }

    /// Store WebAuthn challenge
    pub async fn store_webauthn_challenge(
        &self,
        username: &str,
        _challenge: &str,
    ) -> FusekiResult<()> {
        // TODO: Implement WebAuthn challenge storage
        info!("Storing WebAuthn challenge for user: {}", username);
        Ok(())
    }

    /// Store SMS phone number for user
    pub async fn store_sms_phone(&self, username: &str, _phone: &str) -> FusekiResult<()> {
        // TODO: Implement SMS phone storage in user profile
        info!("Storing SMS phone for user: {}", username);
        Ok(())
    }

    /// Update MFA challenge (placeholder)
    pub async fn update_mfa_challenge(
        &self,
        challenge_id: &str,
        challenge: MfaChallenge,
    ) -> FusekiResult<()> {
        let mut challenges = self.mfa_challenges.write().await;
        challenges.insert(challenge_id.to_string(), challenge);
        debug!("Updated MFA challenge: {}", challenge_id);
        Ok(())
    }

    /// Get user MFA status (placeholder)
    pub async fn get_user_mfa_status(&self, _username: &str) -> FusekiResult<MfaStatus> {
        // TODO: Implement MFA status retrieval
        Ok(MfaStatus {
            enabled: false,
            enrolled_methods: vec![],
            backup_codes_remaining: 0,
            last_used: None,
            expires_at: None,
            message: "MFA disabled".to_string(),
        })
    }

    /// Disable MFA method (placeholder)
    pub async fn disable_mfa_method(
        &self,
        _username: &str,
        _method: MfaMethod,
    ) -> FusekiResult<()> {
        // TODO: Implement MFA method disabling
        Ok(())
    }

    /// Store backup codes (placeholder)
    pub async fn store_backup_codes(
        &self,
        _username: &str,
        _codes: Vec<String>,
    ) -> FusekiResult<()> {
        // TODO: Implement backup code storage
        Ok(())
    }

    /// Store TOTP secret (placeholder)
    pub async fn store_totp_secret(&self, _username: &str, _secret: &str) -> FusekiResult<()> {
        // TODO: Implement TOTP secret storage
        Ok(())
    }

    /// Cleanup LDAP cache
    pub async fn cleanup_ldap_cache(&self) {
        if let Some(ref ldap_service) = self.ldap_service {
            ldap_service.cleanup_expired_cache().await;
        }
    }
}

/// Axum authentication extractor
#[derive(Debug, Clone)]
pub struct AuthUser(pub User);

impl AuthUser {
    pub fn into_inner(self) -> User {
        self.0
    }
}

/// Convert AuthUser to User
impl From<AuthUser> for User {
    fn from(auth_user: AuthUser) -> Self {
        auth_user.0
    }
}

/// Convert User to AuthUser
impl From<User> for AuthUser {
    fn from(user: User) -> Self {
        AuthUser(user)
    }
}

/// Axum extractor implementation for AuthUser
use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};
use axum_extra::headers::{authorization::Bearer, Authorization, HeaderMapExt};

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Try to extract authorization header
        if let Some(auth_header) = parts.headers.typed_get::<Authorization<Bearer>>() {
            // Extract token from header
            let _token = auth_header.token();

            // Get auth service from app state (this would need to be properly implemented)
            // For now, return unauthorized
            return Err(StatusCode::UNAUTHORIZED);
        }

        // Try session-based authentication
        // This would need proper cookie handling implementation
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Permission requirement guard
pub struct RequirePermission(pub Permission);

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Authentication required")]
    AuthenticationRequired,
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Token expired")]
    TokenExpired,
    #[error("Invalid token")]
    InvalidToken,
    #[error("MFA required")]
    MfaRequired,
}

/// Convert AuthError to HTTP response
impl axum::response::IntoResponse for AuthError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            AuthError::AuthenticationRequired => StatusCode::UNAUTHORIZED,
            AuthError::InvalidCredentials => StatusCode::UNAUTHORIZED,
            AuthError::PermissionDenied => StatusCode::FORBIDDEN,
            AuthError::TokenExpired => StatusCode::UNAUTHORIZED,
            AuthError::InvalidToken => StatusCode::UNAUTHORIZED,
            AuthError::MfaRequired => StatusCode::UNAUTHORIZED,
        };

        (status, self.to_string()).into_response()
    }
}

/// Helper function to decode basic auth
#[allow(dead_code)]
fn decode_basic_auth(encoded: &str) -> Result<(String, String), Box<dyn std::error::Error + Send>> {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let decoded = STANDARD
        .decode(encoded)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;
    let credential =
        String::from_utf8(decoded).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send>)?;

    if let Some((username, password)) = credential.split_once(':') {
        Ok((username.to_string(), password.to_string()))
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid basic auth format",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_basic_auth() {
        let encoded = "dGVzdDpwYXNzd29yZA=="; // "test:password" in base64
        let result = decode_basic_auth(encoded).unwrap();
        assert_eq!(result.0, "test");
        assert_eq!(result.1, "password");
    }
}
