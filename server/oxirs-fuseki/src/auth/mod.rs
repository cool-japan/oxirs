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

use crate::config::{JwtConfig, LdapConfig, OAuthConfig, SecurityConfig, UserConfig};
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
pub use types::*;
pub use certificate::CertificateAuth as CertificateAuthenticator;
pub use session::SessionManager;

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
}

impl AuthService {
    /// Create a new authentication service
    pub async fn new(config: SecurityConfig) -> FusekiResult<Self> {
        let users = config.users.clone();
        let config_arc = Arc::new(config);

        // Initialize session manager
        let session_manager = Arc::new(SessionManager::new(config_arc.clone()));

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
            if password::verify_password(password, &user_config.password_hash)? {
                debug!("Successful local authentication for user: {}", username);

                // Create user object with permissions
                let permissions = permissions::compute_user_permissions(user_config).await;
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
            return ldap_service.authenticate(username, password).await;
        }

        Ok(AuthResult::Unauthenticated)
    }

    /// Authenticate using X.509 certificate
    pub async fn authenticate_certificate(&self, cert_data: &[u8]) -> FusekiResult<AuthResult> {
        self.certificate_auth.authenticate_certificate(cert_data).await
    }

    /// Create a new session for authenticated user
    pub async fn create_session(&self, user: User) -> FusekiResult<String> {
        self.session_manager.create_session(user).await
    }

    /// Validate an existing session
    pub async fn validate_session(&self, session_id: &str) -> FusekiResult<Option<User>> {
        self.session_manager.validate_session(session_id).await
    }

    /// Logout a session
    pub async fn logout(&self, session_id: &str) -> FusekiResult<bool> {
        self.session_manager.logout(session_id).await
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
    pub async fn complete_oauth2_authentication(&self, code: &str) -> FusekiResult<AuthResult> {
        self.oauth2_service
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("OAuth2 not configured"))?
            .exchange_code(code)
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
}

/// Axum authentication extractor
#[derive(Clone)]
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
    extract::{FromRequestParts, State},
    http::{request::Parts, StatusCode},
    RequestPartsExt,
};
use axum_extra::headers::{authorization::Bearer, Authorization, HeaderMapExt};

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = StatusCode;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // Try to extract authorization header
        if let Some(auth_header) = parts.headers.typed_get::<Authorization<Bearer>>() {
            // Extract token from header
            let token = auth_header.token();

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
fn decode_basic_auth(encoded: &str) -> Result<(String, String), Box<dyn std::error::Error + Send>> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    
    let decoded = STANDARD.decode(encoded)?;
    let credential = String::from_utf8(decoded)?;
    
    if let Some((username, password)) = credential.split_once(':') {
        Ok((username.to_string(), password.to_string()))
    } else {
        Err("Invalid basic auth format".into())
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