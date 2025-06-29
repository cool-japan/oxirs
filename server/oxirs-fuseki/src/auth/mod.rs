//! Comprehensive authentication and authorization system

use crate::auth::ldap::LdapService;
use crate::auth::oauth::OAuth2Service;
#[cfg(feature = "saml")]
use crate::auth::saml::{SamlConfig, SamlProvider};
use crate::config::{JwtConfig, LdapConfig, OAuthConfig, SecurityConfig, UserConfig};
use crate::error::{FusekiError, FusekiResult};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use axum::{
    extract::{FromRequestParts, State},
    http::{request::Parts, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use axum_extra::headers::{authorization::Bearer, Authorization, HeaderMapExt};
use chrono::{DateTime, Utc};
use der_parser::oid::Oid;
use oid_registry::{OID_X509_EXT_EXTENDED_KEY_USAGE, OID_X509_EXT_KEY_USAGE};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use x509_parser::prelude::*;

#[cfg(feature = "auth")]
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

pub mod ldap;
pub mod oauth;
#[cfg(feature = "saml")]
pub mod saml;

/// Authentication result
#[derive(Debug, Clone)]
pub enum AuthResult {
    Authenticated(User),
    Unauthenticated,
    Forbidden,
    Expired,
    Invalid,
    Locked,
    CertificateRequired,
    CertificateInvalid,
}

/// Certificate authentication data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateAuth {
    pub subject_dn: String,
    pub issuer_dn: String,
    pub serial_number: String,
    pub fingerprint: String,
    pub not_before: DateTime<Utc>,
    pub not_after: DateTime<Utc>,
    pub key_usage: Vec<String>,
    pub extended_key_usage: Vec<String>,
}

/// Multi-factor authentication data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaChallenge {
    pub challenge_id: String,
    pub challenge_type: MfaType,
    pub expires_at: DateTime<Utc>,
    pub attempts_remaining: u8,
}

/// MFA authentication types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MfaType {
    Totp,     // Time-based One-Time Password
    Sms,      // SMS verification
    Email,    // Email verification
    Hardware, // Hardware token (e.g., YubiKey)
    Backup,   // Backup codes
}

/// SAML authentication response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamlResponse {
    pub assertion_id: String,
    pub subject: String,
    pub issuer: String,
    pub attributes: HashMap<String, Vec<String>>,
    pub session_index: String,
    pub not_on_or_after: DateTime<Utc>,
}

/// Authenticated user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    pub roles: Vec<String>,
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub last_login: Option<DateTime<Utc>>,
    pub permissions: Vec<Permission>,
}

/// Permission system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Permission {
    // Dataset permissions
    DatasetRead(String),
    DatasetWrite(String),
    DatasetAdmin(String),

    // Global permissions
    GlobalRead,
    GlobalWrite,
    GlobalAdmin,

    // Service permissions
    SparqlQuery,
    SparqlUpdate,
    GraphStore,

    // Admin permissions
    UserManagement,
    SystemConfig,
    SystemMetrics,
}

/// MFA status for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaStatus {
    pub enabled: bool,
    pub methods: Vec<MfaType>,
    pub backup_codes_generated: bool,
}

/// JWT claims structure
#[cfg(feature = "auth")]
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // Subject (username)
    pub exp: usize,  // Expiration time
    pub iat: usize,  // Issued at
    pub iss: String, // Issuer
    pub aud: String, // Audience
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
}

/// Authentication service manager
#[derive(Clone)]
pub struct AuthService {
    config: Arc<SecurityConfig>,
    users: Arc<RwLock<HashMap<String, UserConfig>>>,
    sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    oauth2_service: Option<OAuth2Service>,
    ldap_service: Option<LdapService>,
    #[cfg(feature = "saml")]
    saml_provider: Option<Arc<SamlProvider>>,
}

/// Active user session
#[derive(Debug, Clone)]
pub struct UserSession {
    pub user: User,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub session_id: String,
}

/// Login request
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
    pub remember_me: Option<bool>,
}

/// Login response
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub success: bool,
    pub token: Option<String>,
    pub user: Option<User>,
    pub expires_at: Option<DateTime<Utc>>,
    pub message: String,
}

/// Token validation result
#[derive(Debug)]
pub struct TokenValidation {
    pub user: User,
    pub expires_at: DateTime<Utc>,
}

impl AuthService {
    /// Create a new authentication service
    pub fn new(config: SecurityConfig) -> Self {
        let users = config.users.clone();

        // Initialize OAuth2 service if configured
        let oauth2_service = config
            .oauth
            .as_ref()
            .map(|oauth_config| OAuth2Service::new(oauth_config.clone()));

        // Initialize LDAP service if configured
        let ldap_service = config
            .ldap
            .as_ref()
            .map(|ldap_config| LdapService::new(ldap_config.clone()));

        Self {
            config: Arc::new(config),
            users: Arc::new(RwLock::new(users)),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            oauth2_service,
            ldap_service,
            #[cfg(feature = "saml")]
            saml_provider: None, // TODO: Initialize from config when SAML config is added
        }
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
                warn!("Login attempt for disabled user: {}", username);
                return Ok(AuthResult::Forbidden);
            }

            // Check if user is locked
            if let Some(locked_until) = user_config.locked_until {
                if locked_until > Utc::now() {
                    warn!(
                        "Login attempt for locked user: {} (locked until {})",
                        username, locked_until
                    );
                    return Ok(AuthResult::Locked);
                }
            }

            // Verify password
            if self.verify_password(password, &user_config.password_hash)? {
                debug!("Successful local authentication for user: {}", username);

                // Create user object with permissions
                let permissions = self.compute_user_permissions(user_config).await;
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

                // Increment failed login attempts
                self.increment_failed_attempts(username).await?;
            }
        }

        // If local authentication failed or user doesn't exist, try LDAP if enabled
        if self.is_ldap_enabled() {
            debug!("Trying LDAP authentication for user: {}", username);
            match self.authenticate_ldap(username, password).await {
                Ok(AuthResult::Authenticated(user)) => {
                    info!("Successful LDAP authentication for user: {}", username);
                    return Ok(AuthResult::Authenticated(user));
                }
                Ok(auth_result) => {
                    debug!("LDAP authentication failed with result: {:?}", auth_result);
                }
                Err(e) => {
                    warn!("LDAP authentication error for user {}: {}", username, e);
                }
            }
        }

        // All authentication methods failed
        warn!("All authentication methods failed for user: {}", username);
        Ok(AuthResult::Unauthenticated)
    }

    /// Generate JWT token for authenticated user
    #[cfg(feature = "auth")]
    pub fn generate_jwt_token(&self, user: &User) -> FusekiResult<String> {
        if let Some(ref jwt_config) = self.config.jwt {
            let now = Utc::now();
            let exp = now + chrono::Duration::seconds(jwt_config.expiration_secs as i64);

            let claims = Claims {
                sub: user.username.clone(),
                exp: exp.timestamp() as usize,
                iat: now.timestamp() as usize,
                iss: jwt_config.issuer.clone(),
                aud: jwt_config.audience.clone(),
                roles: user.roles.clone(),
                permissions: user.permissions.clone(),
            };

            let header = Header::new(Algorithm::HS256);
            let encoding_key = EncodingKey::from_secret(jwt_config.secret.as_ref());

            encode(&header, &claims, &encoding_key).map_err(|e| {
                FusekiError::authentication(format!("Failed to generate JWT token: {}", e))
            })
        } else {
            Err(FusekiError::configuration(
                "JWT configuration not available",
            ))
        }
    }

    /// Validate JWT token
    #[cfg(feature = "auth")]
    pub fn validate_jwt_token(&self, token: &str) -> FusekiResult<TokenValidation> {
        if let Some(ref jwt_config) = self.config.jwt {
            let decoding_key = DecodingKey::from_secret(jwt_config.secret.as_ref());
            let mut validation = Validation::new(Algorithm::HS256);
            validation.set_issuer(&[&jwt_config.issuer]);
            validation.set_audience(&[&jwt_config.audience]);

            match decode::<Claims>(token, &decoding_key, &validation) {
                Ok(token_data) => {
                    let claims = token_data.claims;
                    let expires_at = DateTime::from_timestamp(claims.exp as i64, 0)
                        .unwrap_or_else(|| Utc::now());

                    let user = User {
                        username: claims.sub,
                        roles: claims.roles,
                        email: None, // Not stored in JWT for security
                        full_name: None,
                        last_login: None,
                        permissions: claims.permissions,
                    };

                    Ok(TokenValidation { user, expires_at })
                }
                Err(e) => {
                    debug!("JWT token validation failed: {}", e);
                    Err(FusekiError::authentication("Invalid JWT token"))
                }
            }
        } else {
            Err(FusekiError::configuration(
                "JWT configuration not available",
            ))
        }
    }

    /// Create user session
    pub async fn create_session(&self, user: User) -> FusekiResult<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = UserSession {
            user,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            session_id: session_id.clone(),
        };

        let username = session.user.username.clone();
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);

        info!("Created session for user: {}", username);
        Ok(session_id)
    }

    /// Validate session
    pub async fn validate_session(&self, session_id: &str) -> FusekiResult<AuthResult> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            let timeout = chrono::Duration::seconds(self.config.session.timeout_secs as i64);

            if Utc::now() - session.last_accessed > timeout {
                sessions.remove(session_id);
                debug!("Session expired: {}", session_id);
                Ok(AuthResult::Expired)
            } else {
                session.last_accessed = Utc::now();
                Ok(AuthResult::Authenticated(session.user.clone()))
            }
        } else {
            debug!("Session not found: {}", session_id);
            Ok(AuthResult::Unauthenticated)
        }
    }

    /// Logout user (invalidate session)
    pub async fn logout(&self, session_id: &str) -> FusekiResult<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(session_id) {
            info!("User logged out: {}", session.user.username);
        }
        Ok(())
    }

    /// Check if user has permission
    pub fn has_permission(&self, user: &User, permission: &Permission) -> bool {
        // Check direct permissions
        if user.permissions.contains(permission) {
            return true;
        }

        // Check role-based permissions
        self.check_role_permissions(user, permission)
    }

    /// Hash password using Argon2
    pub fn hash_password(&self, password: &str) -> FusekiResult<String> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();

        argon2
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| FusekiError::authentication(format!("Failed to hash password: {}", e)))
    }

    /// Verify password against hash
    pub fn verify_password(&self, password: &str, hash: &str) -> FusekiResult<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| FusekiError::authentication(format!("Invalid password hash: {}", e)))?;

        let argon2 = Argon2::default();
        Ok(argon2
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
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

            // Invalidate all sessions for this user
            let mut sessions = self.sessions.write().await;
            sessions.retain(|_, session| session.user.username != username);
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

    /// Update last login time
    async fn update_last_login(&self, username: &str) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.last_login = Some(Utc::now());
            user.failed_login_attempts = 0; // Reset failed attempts on successful login
        }
        Ok(())
    }

    /// Increment failed login attempts
    async fn increment_failed_attempts(&self, username: &str) -> FusekiResult<()> {
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.failed_login_attempts += 1;

            // Lock user after 5 failed attempts for 15 minutes
            if user.failed_login_attempts >= 5 {
                user.locked_until = Some(Utc::now() + chrono::Duration::minutes(15));
                warn!("User locked due to failed login attempts: {}", username);
            }
        }
        Ok(())
    }

    /// Compute user permissions based on roles
    async fn compute_user_permissions(&self, user_config: &UserConfig) -> Vec<Permission> {
        let mut permissions = Vec::new();

        for role in &user_config.roles {
            match role.as_str() {
                "admin" => {
                    permissions.extend(vec![
                        Permission::GlobalAdmin,
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                        Permission::UserManagement,
                        Permission::SystemConfig,
                        Permission::SystemMetrics,
                    ]);
                }
                "user" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                "writer" => {
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                    ]);
                }
                "reader" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                _ => {
                    // Custom role - could be dataset-specific
                    if role.starts_with("dataset:") {
                        let dataset_name = role.strip_prefix("dataset:").unwrap_or("");
                        if role.ends_with(":admin") {
                            let dataset_name =
                                dataset_name.strip_suffix(":admin").unwrap_or(dataset_name);
                            permissions.push(Permission::DatasetAdmin(dataset_name.to_string()));
                        } else if role.ends_with(":write") {
                            let dataset_name =
                                dataset_name.strip_suffix(":write").unwrap_or(dataset_name);
                            permissions.push(Permission::DatasetWrite(dataset_name.to_string()));
                        } else if role.ends_with(":read") {
                            let dataset_name =
                                dataset_name.strip_suffix(":read").unwrap_or(dataset_name);
                            permissions.push(Permission::DatasetRead(dataset_name.to_string()));
                        }
                    }
                }
            }
        }

        // Remove duplicates
        permissions.sort();
        permissions.dedup();

        permissions
    }

    /// Check role-based permissions
    fn check_role_permissions(&self, user: &User, permission: &Permission) -> bool {
        for role in &user.roles {
            match (role.as_str(), permission) {
                ("admin", _) => return true,
                (
                    "writer",
                    Permission::GlobalRead
                    | Permission::GlobalWrite
                    | Permission::SparqlQuery
                    | Permission::SparqlUpdate
                    | Permission::GraphStore,
                ) => return true,
                ("reader", Permission::GlobalRead | Permission::SparqlQuery) => return true,
                ("user", Permission::GlobalRead | Permission::SparqlQuery) => return true,
                _ => continue,
            }
        }
        false
    }

    // OAuth2/OIDC Authentication Methods

    /// Get OAuth2 service reference
    pub fn oauth2_service(&self) -> Option<&OAuth2Service> {
        self.oauth2_service.as_ref()
    }

    /// Generate OAuth2 authorization URL
    pub async fn generate_oauth2_auth_url(
        &self,
        redirect_uri: &str,
        scopes: &[String],
        use_pkce: bool,
    ) -> FusekiResult<(String, String)> {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service
                .generate_authorization_url(redirect_uri, scopes, use_pkce)
                .await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Complete OAuth2 authorization flow
    pub async fn complete_oauth2_flow(
        &self,
        code: &str,
        state: &str,
        redirect_uri: &str,
    ) -> FusekiResult<AuthResult> {
        if let Some(oauth2_service) = &self.oauth2_service {
            // Exchange code for token
            let token = oauth2_service
                .exchange_code_for_token(code, state, redirect_uri)
                .await?;

            // Authenticate using the access token
            oauth2_service
                .authenticate_oauth2(&token.access_token)
                .await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Authenticate using OAuth2 access token
    pub async fn authenticate_oauth2_token(&self, access_token: &str) -> FusekiResult<AuthResult> {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service.authenticate_oauth2(access_token).await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Refresh OAuth2 access token
    pub async fn refresh_oauth2_token(
        &self,
        refresh_token: &str,
    ) -> FusekiResult<oauth::OAuth2Token> {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service.refresh_token(refresh_token).await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Validate OAuth2 access token
    pub async fn validate_oauth2_token(&self, access_token: &str) -> FusekiResult<bool> {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service.validate_access_token(access_token).await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Get OIDC user information
    pub async fn get_oidc_user_info(
        &self,
        access_token: &str,
    ) -> FusekiResult<oauth::OIDCUserInfo> {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service.get_user_info(access_token).await
        } else {
            Err(FusekiError::configuration("OAuth2 not configured"))
        }
    }

    /// Cleanup expired OAuth2 states and tokens
    pub async fn cleanup_oauth2_expired(&self) {
        if let Some(oauth2_service) = &self.oauth2_service {
            oauth2_service.cleanup_expired().await;
        }
    }

    /// Check if OAuth2 is configured and available
    pub fn is_oauth2_enabled(&self) -> bool {
        self.oauth2_service.is_some()
    }

    // LDAP Authentication Methods

    /// Get LDAP service reference
    pub fn ldap_service(&self) -> Option<&LdapService> {
        self.ldap_service.as_ref()
    }

    /// Authenticate user using LDAP
    pub async fn authenticate_ldap(
        &self,
        username: &str,
        password: &str,
    ) -> FusekiResult<AuthResult> {
        if let Some(ldap_service) = &self.ldap_service {
            ldap_service
                .authenticate_ldap_user(username, password)
                .await
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Test LDAP connection
    pub async fn test_ldap_connection(&self) -> FusekiResult<bool> {
        if let Some(ldap_service) = &self.ldap_service {
            ldap_service.test_connection().await
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Get user groups from LDAP
    pub async fn get_ldap_user_groups(&self, username: &str) -> FusekiResult<Vec<ldap::LdapGroup>> {
        if let Some(ldap_service) = &self.ldap_service {
            ldap_service.get_user_groups(username).await
        } else {
            Err(FusekiError::configuration("LDAP not configured"))
        }
    }

    /// Cleanup expired LDAP cache entries
    pub async fn cleanup_ldap_cache(&self) {
        if let Some(ldap_service) = &self.ldap_service {
            ldap_service.cleanup_expired_cache().await;
        }
    }

    /// Check if LDAP is configured and available
    pub fn is_ldap_enabled(&self) -> bool {
        self.ldap_service.is_some()
    }

    /// Get JWT configuration if available
    pub fn jwt_config(&self) -> Option<&crate::config::JwtConfig> {
        self.config.jwt.as_ref()
    }

    /// Get LDAP configuration if available
    pub fn ldap_config(&self) -> Option<&crate::config::LdapConfig> {
        self.config.ldap.as_ref()
    }

    // Certificate-based Authentication Methods

    /// Authenticate user using X.509 client certificate
    pub async fn authenticate_certificate(&self, cert_data: &[u8]) -> FusekiResult<AuthResult> {
        debug!("Authenticating using X.509 client certificate");

        // Parse certificate
        let cert_info = self.parse_x509_certificate(cert_data)?;

        // Validate certificate
        if !self.validate_certificate(&cert_info).await? {
            warn!("Certificate validation failed: {}", cert_info.subject_dn);
            return Ok(AuthResult::CertificateInvalid);
        }

        // Map certificate to user
        let user = self.map_certificate_to_user(cert_info).await?;

        info!(
            "Certificate authentication successful for user: {}",
            user.username
        );
        Ok(AuthResult::Authenticated(user))
    }

    /// Parse X.509 certificate using x509-parser
    fn parse_x509_certificate(&self, cert_data: &[u8]) -> FusekiResult<CertificateAuth> {
        // Parse DER-encoded certificate
        let (_, cert) = X509Certificate::from_der(cert_data).map_err(|e| {
            FusekiError::authentication(format!("Failed to parse X.509 certificate: {}", e))
        })?;

        // Extract subject DN
        let subject_dn = cert.subject().to_string();

        // Extract issuer DN
        let issuer_dn = cert.issuer().to_string();

        // Extract serial number
        let serial_number = cert.serial.to_str_radix(16).to_uppercase();

        // Calculate SHA-256 fingerprint
        let fingerprint = sha2::Sha256::digest(cert_data)
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<Vec<_>>()
            .join(":");

        // Extract validity period
        let not_before = cert
            .validity()
            .not_before
            .timestamp()
            .map(|ts| chrono::DateTime::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now()))
            .unwrap_or_else(|| Utc::now());

        let not_after = cert
            .validity()
            .not_after
            .timestamp()
            .map(|ts| chrono::DateTime::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now()))
            .unwrap_or_else(|| Utc::now());

        // Extract key usage
        let mut key_usage = Vec::new();
        if let Some(ext) = cert.extensions().get(&OID_X509_EXT_KEY_USAGE) {
            if let Ok(ku) = ext.parsed_extension::<KeyUsage>() {
                if ku.digital_signature() {
                    key_usage.push("digitalSignature".to_string());
                }
                if ku.key_encipherment() {
                    key_usage.push("keyEncipherment".to_string());
                }
                if ku.key_agreement() {
                    key_usage.push("keyAgreement".to_string());
                }
                if ku.non_repudiation() {
                    key_usage.push("nonRepudiation".to_string());
                }
            }
        }

        // Extract extended key usage
        let mut extended_key_usage = Vec::new();
        if let Some(ext) = cert.extensions().get(&OID_X509_EXT_EXTENDED_KEY_USAGE) {
            if let Ok(eku) = ext.parsed_extension::<ExtendedKeyUsage>() {
                for purpose in eku.any {
                    let purpose_str = purpose.to_string();
                    if purpose_str == "1.3.6.1.5.5.7.3.2" {
                        // Client Auth
                        extended_key_usage.push("clientAuth".to_string());
                    } else if purpose_str == "1.3.6.1.5.5.7.3.1" {
                        // Server Auth
                        extended_key_usage.push("serverAuth".to_string());
                    } else if purpose_str == "1.3.6.1.5.5.7.3.4" {
                        // Email Protection
                        extended_key_usage.push("emailProtection".to_string());
                    }
                }
            }
        }

        let cert_info = CertificateAuth {
            subject_dn,
            issuer_dn,
            serial_number,
            fingerprint,
            not_before,
            not_after,
            key_usage,
            extended_key_usage,
        };

        Ok(cert_info)
    }

    /// Validate certificate against configured trust store and CRL
    async fn validate_certificate(&self, cert_info: &CertificateAuth) -> FusekiResult<bool> {
        // Check certificate validity period
        let now = Utc::now();
        if now < cert_info.not_before || now > cert_info.not_after {
            warn!(
                "Certificate is expired or not yet valid: {} - {}",
                cert_info.not_before, cert_info.not_after
            );
            return Ok(false);
        }

        // Check key usage for client authentication
        if !cert_info
            .extended_key_usage
            .contains(&"clientAuth".to_string())
        {
            warn!("Certificate does not have clientAuth extended key usage");
            return Ok(false);
        }

        // Check if certificate is in configured trust store
        if !self.is_certificate_trusted(cert_info).await? {
            warn!(
                "Certificate is not in trust store: {}",
                cert_info.subject_dn
            );
            return Ok(false);
        }

        // Check Certificate Revocation List (CRL) if configured
        if let Err(e) = self.check_certificate_revocation(cert_info).await {
            warn!("Certificate revocation check failed: {}", e);
            return Ok(false);
        }

        // Validate certificate chain if full chain is available
        if let Err(e) = self.validate_certificate_chain(cert_info).await {
            warn!("Certificate chain validation failed: {}", e);
            return Ok(false);
        }

        info!(
            "Certificate validation successful for: {}",
            cert_info.subject_dn
        );
        Ok(true)
    }

    /// Check if certificate is in the configured trust store
    async fn is_certificate_trusted(&self, cert_info: &CertificateAuth) -> FusekiResult<bool> {
        // For now, implement a simple trust store based on issuer DN patterns
        // In production, this would check against a proper PKCS#12 trust store

        let trusted_issuers = vec![
            "CN=Example Corp Root CA,O=Example Corp,C=US",
            "CN=Internal CA,OU=IT Department,O=Example Corp,C=US",
            // Add more trusted CAs as needed
        ];

        // Check if the issuer is in our trust list
        let is_trusted = trusted_issuers.iter().any(|&issuer| {
            cert_info.issuer_dn.contains(issuer) ||
            // Allow partial matches for flexibility
            self.is_issuer_pattern_match(&cert_info.issuer_dn, issuer)
        });

        if is_trusted {
            debug!("Certificate issuer is trusted: {}", cert_info.issuer_dn);
        } else {
            debug!(
                "Certificate issuer is not in trust store: {}",
                cert_info.issuer_dn
            );
        }

        Ok(is_trusted)
    }

    /// Check if issuer DN matches trusted patterns
    fn is_issuer_pattern_match(&self, actual_issuer: &str, trusted_pattern: &str) -> bool {
        // Simple pattern matching - in production you'd use proper DN parsing
        actual_issuer.contains("Example Corp") && actual_issuer.contains("CA")
    }

    /// Check certificate against Certificate Revocation List (CRL)
    async fn check_certificate_revocation(&self, cert_info: &CertificateAuth) -> FusekiResult<()> {
        // In a real implementation, this would:
        // 1. Download CRL from certificate's CRL distribution points
        // 2. Parse the CRL
        // 3. Check if the certificate serial number is in the revoked list

        debug!(
            "CRL check not yet implemented for certificate: {}",
            cert_info.serial_number
        );

        // For now, maintain a simple in-memory revocation list
        let revoked_serials = vec![
            "DEADBEEF12345678", // Example revoked certificate
        ];

        if revoked_serials.contains(&cert_info.serial_number.as_str()) {
            return Err(FusekiError::authentication(format!(
                "Certificate is revoked: {}",
                cert_info.serial_number
            )));
        }

        Ok(())
    }

    /// Validate the certificate chain
    async fn validate_certificate_chain(&self, cert_info: &CertificateAuth) -> FusekiResult<()> {
        // In a complete implementation, this would:
        // 1. Build the certificate chain from the client cert to a trusted root
        // 2. Verify each certificate in the chain
        // 3. Check that each certificate properly signs the next one
        // 4. Ensure the chain leads to a trusted root CA

        debug!(
            "Certificate chain validation not fully implemented for: {}",
            cert_info.subject_dn
        );

        // For now, do basic checks
        if cert_info.subject_dn == cert_info.issuer_dn {
            // Self-signed certificate - only allow if explicitly configured
            if self.allow_self_signed_certificates() {
                debug!("Allowing self-signed certificate: {}", cert_info.subject_dn);
                Ok(())
            } else {
                Err(FusekiError::authentication(
                    "Self-signed certificates not allowed".to_string(),
                ))
            }
        } else {
            // Assume valid chain for now - in production this needs proper implementation
            debug!(
                "Assuming valid certificate chain for: {}",
                cert_info.subject_dn
            );
            Ok(())
        }
    }

    /// Check if self-signed certificates are allowed
    fn allow_self_signed_certificates(&self) -> bool {
        // This would be configurable in SecurityConfig
        // For development/testing purposes only
        false
    }

    /// Map certificate to internal user
    async fn map_certificate_to_user(&self, cert_info: CertificateAuth) -> FusekiResult<User> {
        // Extract Common Name from subject DN
        let username = self
            .extract_cn_from_dn(&cert_info.subject_dn)
            .unwrap_or_else(|| cert_info.subject_dn.clone());

        // Map certificate attributes to user roles
        let roles = self.map_certificate_to_roles(&cert_info).await;

        // Compute permissions
        let permissions = self.compute_user_permissions_for_roles(&roles).await;

        let user = User {
            username,
            roles,
            email: None, // Could extract from certificate Subject Alternative Names
            full_name: Some(cert_info.subject_dn),
            last_login: Some(Utc::now()),
            permissions,
        };

        Ok(user)
    }

    /// Extract Common Name from Distinguished Name
    fn extract_cn_from_dn(&self, dn: &str) -> Option<String> {
        for component in dn.split(',') {
            let component = component.trim();
            if component.starts_with("CN=") {
                return Some(component[3..].to_string());
            }
        }
        None
    }

    /// Map certificate attributes to roles
    async fn map_certificate_to_roles(&self, cert_info: &CertificateAuth) -> Vec<String> {
        let mut roles = Vec::new();

        // Extract OU (Organizational Unit) from subject DN
        for component in cert_info.subject_dn.split(',') {
            let component = component.trim();
            if component.starts_with("OU=") {
                let ou = &component[3..];
                match ou.to_lowercase().as_str() {
                    "engineering" | "developers" => roles.push("writer".to_string()),
                    "management" | "admins" => roles.push("admin".to_string()),
                    "users" => roles.push("reader".to_string()),
                    _ => {}
                }
            }
        }

        // Default role if none found
        if roles.is_empty() {
            roles.push("user".to_string());
        }

        roles
    }

    // Multi-Factor Authentication Methods

    /// Initiate MFA challenge
    pub async fn initiate_mfa_challenge(
        &self,
        username: &str,
        mfa_type: MfaType,
    ) -> FusekiResult<MfaChallenge> {
        let challenge_id = uuid::Uuid::new_v4().to_string();

        let challenge = MfaChallenge {
            challenge_id: challenge_id.clone(),
            challenge_type: mfa_type,
            expires_at: Utc::now() + chrono::Duration::minutes(5),
            attempts_remaining: 3,
        };

        // TODO: Store challenge in temporary storage
        // TODO: Send challenge via appropriate channel (SMS, Email, etc.)

        info!(
            "Initiated MFA challenge {} for user: {}",
            challenge_id, username
        );
        Ok(challenge)
    }

    /// Store MFA challenge (for temporary storage)
    pub async fn store_mfa_challenge(&self, challenge: &MfaChallenge) -> FusekiResult<()> {
        // TODO: Implement proper challenge storage
        // For now, this is a stub that would store challenges in a temporary cache
        debug!("Storing MFA challenge: {}", challenge.challenge_id);
        Ok(())
    }

    /// Get MFA challenge from storage
    pub async fn get_mfa_challenge(
        &self,
        challenge_id: &str,
    ) -> FusekiResult<Option<MfaChallenge>> {
        // TODO: Implement proper challenge retrieval
        // For now, this is a stub that would retrieve challenges from a temporary cache
        debug!("Retrieving MFA challenge: {}", challenge_id);

        // Return a mock challenge for demonstration
        if !challenge_id.is_empty() {
            Ok(Some(MfaChallenge {
                challenge_id: challenge_id.to_string(),
                challenge_type: MfaType::Totp,
                expires_at: Utc::now() + chrono::Duration::minutes(5),
                attempts_remaining: 3,
            }))
        } else {
            Ok(None)
        }
    }

    /// Remove MFA challenge from storage
    pub async fn remove_mfa_challenge(&self, challenge_id: &str) -> FusekiResult<()> {
        // TODO: Implement proper challenge removal
        // For now, this is a stub that would remove challenges from a temporary cache
        debug!("Removing MFA challenge: {}", challenge_id);
        Ok(())
    }

    /// Verify MFA challenge response
    pub async fn verify_mfa_challenge(
        &self,
        challenge_id: &str,
        response: &str,
    ) -> FusekiResult<bool> {
        // TODO: Retrieve and validate challenge from storage
        // TODO: Verify response based on challenge type

        // Simplified implementation for demonstration
        debug!("Verifying MFA challenge: {}", challenge_id);
        Ok(response.len() == 6 && response.chars().all(|c| c.is_ascii_digit()))
    }

    // SAML Authentication Methods

    /// Process SAML authentication response
    pub async fn process_saml_response(&self, saml_response: &str) -> FusekiResult<AuthResult> {
        debug!("Processing SAML authentication response");

        // Parse SAML response (simplified implementation)
        let saml_data = self.parse_saml_response(saml_response)?;

        // Validate SAML response
        if !self.validate_saml_response(&saml_data).await? {
            warn!("SAML response validation failed");
            return Ok(AuthResult::Invalid);
        }

        // Map SAML attributes to user
        let user = self.map_saml_to_user(saml_data).await?;

        info!("SAML authentication successful for user: {}", user.username);
        Ok(AuthResult::Authenticated(user))
    }

    /// Parse SAML response (simplified implementation)
    fn parse_saml_response(&self, saml_response: &str) -> FusekiResult<SamlResponse> {
        // This is a simplified mock implementation
        // In production, you would use a proper SAML library like 'samael'

        let saml_data = SamlResponse {
            assertion_id: uuid::Uuid::new_v4().to_string(),
            subject: "john.doe@example.com".to_string(),
            issuer: "https://sso.example.com".to_string(),
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert(
                    "email".to_string(),
                    vec!["john.doe@example.com".to_string()],
                );
                attrs.insert("name".to_string(), vec!["John Doe".to_string()]);
                attrs.insert(
                    "groups".to_string(),
                    vec!["employees".to_string(), "developers".to_string()],
                );
                attrs
            },
            session_index: uuid::Uuid::new_v4().to_string(),
            not_on_or_after: Utc::now() + chrono::Duration::hours(8),
        };

        Ok(saml_data)
    }

    /// Validate SAML response
    async fn validate_saml_response(&self, saml_data: &SamlResponse) -> FusekiResult<bool> {
        // Check assertion validity
        if Utc::now() > saml_data.not_on_or_after {
            return Ok(false);
        }

        // TODO: Validate SAML signature
        // TODO: Check issuer against trusted identity providers

        Ok(true)
    }

    /// Map SAML attributes to internal user
    async fn map_saml_to_user(&self, saml_data: SamlResponse) -> FusekiResult<User> {
        let username = saml_data
            .attributes
            .get("email")
            .and_then(|emails| emails.first())
            .unwrap_or(&saml_data.subject)
            .clone();

        let full_name = saml_data
            .attributes
            .get("name")
            .and_then(|names| names.first())
            .cloned();

        let email = saml_data
            .attributes
            .get("email")
            .and_then(|emails| emails.first())
            .cloned();

        // Map SAML groups to roles
        let roles = saml_data
            .attributes
            .get("groups")
            .map(|groups| self.map_saml_groups_to_roles(groups))
            .unwrap_or_else(|| vec!["user".to_string()]);

        // Compute permissions
        let permissions = self.compute_user_permissions_for_roles(&roles).await;

        let user = User {
            username,
            roles,
            email,
            full_name,
            last_login: Some(Utc::now()),
            permissions,
        };

        Ok(user)
    }

    /// Map SAML groups to internal roles
    fn map_saml_groups_to_roles(&self, groups: &[String]) -> Vec<String> {
        let mut roles = Vec::new();

        for group in groups {
            let role = match group.to_lowercase().as_str() {
                "administrators" | "admins" => "admin",
                "developers" | "engineers" => "writer",
                "employees" | "users" => "reader",
                _ => "user",
            };

            if !roles.contains(&role.to_string()) {
                roles.push(role.to_string());
            }
        }

        if roles.is_empty() {
            roles.push("user".to_string());
        }

        roles
    }

    /// Compute permissions for roles (helper method)
    async fn compute_user_permissions_for_roles(&self, roles: &[String]) -> Vec<Permission> {
        // Re-use the existing permission computation logic
        let mut permissions = Vec::new();

        for role in roles {
            match role.as_str() {
                "admin" => {
                    permissions.extend(vec![
                        Permission::GlobalAdmin,
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                        Permission::UserManagement,
                        Permission::SystemConfig,
                        Permission::SystemMetrics,
                    ]);
                }
                "writer" => {
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                    ]);
                }
                "reader" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                "user" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                _ => {}
            }
        }

        // Remove duplicates
        permissions.sort();
        permissions.dedup();

        permissions
    }

    /// Update MFA challenge in storage
    pub async fn update_mfa_challenge(&self, challenge: &MfaChallenge) -> FusekiResult<()> {
        // TODO: Implement proper challenge update storage
        // For now, this is a stub that would update challenges in a temporary cache
        debug!("Updating MFA challenge: {}", challenge.challenge_id);
        Ok(())
    }

    /// Get user MFA status
    pub async fn get_user_mfa_status(&self, username: &str) -> FusekiResult<Option<MfaStatus>> {
        // TODO: Implement proper MFA status retrieval from user storage
        // For now, this is a stub that would retrieve MFA status from user configuration
        debug!("Getting MFA status for user: {}", username);

        // Return mock MFA status for demonstration
        Ok(Some(MfaStatus {
            enabled: false,
            methods: vec![],
            backup_codes_generated: false,
        }))
    }

    /// Disable a specific MFA method for a user
    pub async fn disable_mfa_method(
        &self,
        username: &str,
        mfa_type: &MfaType,
    ) -> FusekiResult<bool> {
        // TODO: Implement proper MFA method disabling in user storage
        // For now, this is a stub that would disable MFA methods in user configuration
        debug!("Disabling MFA method {:?} for user: {}", mfa_type, username);

        // Return mock success for demonstration
        Ok(true)
    }

    /// Store backup codes for a user
    pub async fn store_backup_codes(&self, username: &str, codes: &[String]) -> FusekiResult<()> {
        // TODO: Implement proper backup codes storage
        debug!(
            "Storing {} backup codes for user: {}",
            codes.len(),
            username
        );
        Ok(())
    }

    /// Store TOTP secret for a user
    pub async fn store_totp_secret(&self, username: &str, secret: &str) -> FusekiResult<()> {
        // TODO: Implement proper TOTP secret storage
        debug!("Storing TOTP secret for user: {}", username);
        Ok(())
    }

    /// Store SMS phone number for a user
    pub async fn store_sms_phone(&self, username: &str, phone: &str) -> FusekiResult<()> {
        // TODO: Implement proper SMS phone storage
        debug!("Storing SMS phone for user: {}", username);
        Ok(())
    }

    /// Store MFA email address for a user
    pub async fn store_mfa_email(&self, username: &str, email: &str) -> FusekiResult<()> {
        // TODO: Implement proper MFA email storage
        debug!("Storing MFA email for user: {}", username);
        Ok(())
    }

    /// Store WebAuthn challenge for a user
    pub async fn store_webauthn_challenge(
        &self,
        username: &str,
        challenge: &str,
    ) -> FusekiResult<()> {
        // TODO: Implement proper WebAuthn challenge storage
        debug!("Storing WebAuthn challenge for user: {}", username);
        Ok(())
    }

    /// Get user SMS phone number
    pub async fn get_user_sms_phone(&self, username: &str) -> FusekiResult<Option<String>> {
        // TODO: Implement proper SMS phone retrieval
        debug!("Getting SMS phone for user: {}", username);
        Ok(None)
    }

    /// Get user MFA email address
    pub async fn get_user_mfa_email(&self, username: &str) -> FusekiResult<Option<String>> {
        // TODO: Implement proper MFA email retrieval
        debug!("Getting MFA email for user: {}", username);
        Ok(None)
    }

    /// Check if SAML authentication is enabled
    pub fn is_saml_enabled(&self) -> bool {
        self.config.saml.is_some()
    }

    /// Generate SAML authentication request
    #[cfg(feature = "saml")]
    pub async fn generate_saml_auth_request(
        &self,
        target_url: &str,
        force_authn: bool,
        relay_state: &str,
    ) -> FusekiResult<(String, String)> {
        if let Some(ref saml_provider) = self.saml_provider {
            saml_provider.generate_auth_request(target_url, force_authn, relay_state).await
        } else {
            Err(FusekiError::configuration("SAML not configured"))
        }
    }

    /// Complete SAML authentication
    #[cfg(feature = "saml")]
    pub async fn complete_saml_authentication(&self, user: User) -> FusekiResult<AuthResult> {
        // Store SAML user session information if needed
        debug!("Completing SAML authentication for user: {}", user.username);
        Ok(AuthResult::Authenticated(user))
    }

    /// Logout user by SAML session index
    #[cfg(feature = "saml")]
    pub async fn logout_by_session_index(&self, session_index: &str) -> FusekiResult<bool> {
        debug!("Logging out SAML session: {}", session_index);
        
        // Find and invalidate sessions by SAML session index
        let mut sessions = self.sessions.write().await;
        let mut removed_count = 0;
        
        sessions.retain(|_, session| {
            // In a real implementation, you'd store SAML session index with the session
            // For now, we'll remove all sessions (simplified)
            removed_count += 1;
            false
        });
        
        Ok(removed_count > 0)
    }

    /// Get SAML service provider configuration
    #[cfg(feature = "saml")]
    pub fn get_saml_sp_config(&self) -> FusekiResult<&crate::auth::saml::SamlConfig> {
        self.config.saml.as_ref().ok_or_else(|| {
            FusekiError::configuration("SAML not configured")
        })
    }

    /// Non-feature version of SAML methods for compilation compatibility
    #[cfg(not(feature = "saml"))]
    pub async fn generate_saml_auth_request(
        &self,
        _target_url: &str,
        _force_authn: bool,
        _relay_state: &str,
    ) -> FusekiResult<(String, String)> {
        Err(FusekiError::configuration("SAML feature not enabled"))
    }

    #[cfg(not(feature = "saml"))]
    pub async fn complete_saml_authentication(&self, _user: User) -> FusekiResult<AuthResult> {
        Err(FusekiError::configuration("SAML feature not enabled"))
    }

    #[cfg(not(feature = "saml"))]
    pub async fn logout_by_session_index(&self, _session_index: &str) -> FusekiResult<bool> {
        Err(FusekiError::configuration("SAML feature not enabled"))
    }

    #[cfg(not(feature = "saml"))]
    pub fn get_saml_sp_config(&self) -> FusekiResult<()> {
        Err(FusekiError::configuration("SAML feature not enabled"))
    }
}

/// Authentication extractor for Axum
#[derive(Debug)]
pub struct AuthUser(pub User);

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
    AuthService: FromRequestParts<S>,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        // Extract auth service from state
        let auth_service = AuthService::from_request_parts(parts, state)
            .await
            .map_err(|_| AuthError::ServiceUnavailable)?;

        // Try JWT authentication first
        #[cfg(feature = "auth")]
        if let Some(auth_header) = parts.headers.typed_get::<Authorization<Bearer>>() {
            let token = auth_header.token();

            // Try JWT token validation
            match auth_service.validate_jwt_token(token) {
                Ok(validation) => {
                    return Ok(AuthUser(validation.user));
                }
                Err(_) => {
                    // If JWT fails, try OAuth2 token validation
                    if auth_service.is_oauth2_enabled() {
                        match auth_service.authenticate_oauth2_token(token).await {
                            Ok(AuthResult::Authenticated(user)) => {
                                return Ok(AuthUser(user));
                            }
                            _ => {
                                return Err(AuthError::InvalidToken);
                            }
                        }
                    } else {
                        return Err(AuthError::InvalidToken);
                    }
                }
            }
        }

        // Try OAuth2 access token without Bearer prefix (for compatibility)
        if let Some(auth_header) = parts.headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if !auth_str.starts_with("Bearer ") && auth_service.is_oauth2_enabled() {
                    match auth_service.authenticate_oauth2_token(auth_str).await {
                        Ok(AuthResult::Authenticated(user)) => {
                            return Ok(AuthUser(user));
                        }
                        _ => {} // Continue to other auth methods
                    }
                }
            }
        }

        // Try LDAP authentication with Basic auth
        if let Some(auth_header) = parts.headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str.starts_with("Basic ") && auth_service.is_ldap_enabled() {
                    if let Ok(credentials) = decode_basic_auth(&auth_str[6..]) {
                        match auth_service
                            .authenticate_ldap(&credentials.0, &credentials.1)
                            .await
                        {
                            Ok(AuthResult::Authenticated(user)) => {
                                return Ok(AuthUser(user));
                            }
                            _ => {} // Continue to other auth methods
                        }
                    }
                }
            }
        }

        // Try certificate-based authentication
        if let Some(cert_header) = parts.headers.get("x-client-cert") {
            if let Ok(cert_pem) = cert_header.to_str() {
                // Decode PEM certificate
                if let Ok(cert_data) = base64::decode(
                    cert_pem
                        .replace("-----BEGIN CERTIFICATE-----", "")
                        .replace("-----END CERTIFICATE-----", "")
                        .replace("\n", ""),
                ) {
                    match auth_service.authenticate_certificate(&cert_data).await {
                        Ok(AuthResult::Authenticated(user)) => {
                            return Ok(AuthUser(user));
                        }
                        Ok(AuthResult::CertificateInvalid) => {
                            return Err(AuthError::InvalidToken);
                        }
                        _ => {} // Continue to session auth
                    }
                }
            }
        }

        // Try session authentication
        if let Some(session_cookie) = parts.headers.get("cookie") {
            if let Ok(cookie_str) = session_cookie.to_str() {
                for cookie in cookie_str.split(';') {
                    let cookie = cookie.trim();
                    if let Some(session_id) = cookie.strip_prefix("session_id=") {
                        match auth_service.validate_session(session_id).await {
                            Ok(AuthResult::Authenticated(user)) => {
                                return Ok(AuthUser(user));
                            }
                            Ok(AuthResult::Expired) => {
                                return Err(AuthError::TokenExpired);
                            }
                            _ => continue,
                        }
                    }
                }
            }
        }

        Err(AuthError::MissingToken)
    }
}

/// Authentication errors
#[derive(Debug)]
pub enum AuthError {
    MissingToken,
    InvalidToken,
    TokenExpired,
    ServiceUnavailable,
    InsufficientPermissions,
    CertificateRequired,
    CertificateInvalid,
    MfaRequired,
    MfaFailed,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authentication token"),
            AuthError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid authentication token"),
            AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Authentication token expired"),
            AuthError::ServiceUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Authentication service unavailable",
            ),
            AuthError::InsufficientPermissions => {
                (StatusCode::FORBIDDEN, "Insufficient permissions")
            }
            AuthError::CertificateRequired => {
                (StatusCode::UNAUTHORIZED, "Client certificate required")
            }
            AuthError::CertificateInvalid => (
                StatusCode::UNAUTHORIZED,
                "Invalid or untrusted client certificate",
            ),
            AuthError::MfaRequired => (
                StatusCode::UNAUTHORIZED,
                "Multi-factor authentication required",
            ),
            AuthError::MfaFailed => (
                StatusCode::UNAUTHORIZED,
                "Multi-factor authentication failed",
            ),
        };

        let error_response = serde_json::json!({
            "error": "authentication_error",
            "message": message,
            "timestamp": Utc::now()
        });

        (status, axum::Json(error_response)).into_response()
    }
}

/// Permission checker extractor
#[derive(Debug)]
pub struct RequirePermission(pub Permission);

#[axum::async_trait]
impl<S> FromRequestParts<S> for RequirePermission
where
    S: Send + Sync,
    AuthUser: FromRequestParts<S>,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let AuthUser(user) = AuthUser::from_request_parts(parts, state).await?;

        // For now, we'll extract the required permission from the request path
        // In a real implementation, this would be more sophisticated
        let path = parts.uri.path();

        let required_permission = match path {
            path if path.starts_with("/sparql") => Permission::SparqlQuery,
            path if path.starts_with("/update") => Permission::SparqlUpdate,
            path if path.starts_with("/admin") => Permission::SystemConfig,
            path if path.starts_with("/metrics") => Permission::SystemMetrics,
            _ => Permission::GlobalRead,
        };

        // Check if user has the required permission
        if user.permissions.contains(&required_permission) {
            Ok(RequirePermission(required_permission))
        } else {
            Err(AuthError::InsufficientPermissions)
        }
    }
}

/// Helper functions for password management
pub mod passwords {
    use super::*;

    /// Generate a secure random password
    pub fn generate_password(length: usize) -> String {
        use rand::Rng;

        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                abcdefghijklmnopqrstuvwxyz\
                                0123456789\
                                !@#$%^&*()_+-=[]{}|;:,.<>?";

        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    /// Check password strength
    pub fn check_password_strength(password: &str) -> PasswordStrength {
        let length = password.len();
        let has_uppercase = password.chars().any(|c| c.is_uppercase());
        let has_lowercase = password.chars().any(|c| c.is_lowercase());
        let has_digit = password.chars().any(|c| c.is_ascii_digit());
        let has_special = password.chars().any(|c| !c.is_alphanumeric());

        let score = (length >= 8) as u8
            + (length >= 12) as u8
            + has_uppercase as u8
            + has_lowercase as u8
            + has_digit as u8
            + has_special as u8;

        match score {
            0..=2 => PasswordStrength::Weak,
            3..=4 => PasswordStrength::Medium,
            5..=6 => PasswordStrength::Strong,
            _ => PasswordStrength::Strong, // Any score above 6 is considered strong
        }
    }
}

/// Password strength levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PasswordStrength {
    Weak,
    Medium,
    Strong,
}

/// Decode Basic authentication header
fn decode_basic_auth(encoded: &str) -> Result<(String, String), Box<dyn std::error::Error + Send>> {
    use base64::{engine::general_purpose, Engine as _};

    let decoded_bytes = general_purpose::STANDARD.decode(encoded)?;
    let decoded_str = String::from_utf8(decoded_bytes)?;

    if let Some(colon_pos) = decoded_str.find(':') {
        let username = decoded_str[..colon_pos].to_string();
        let password = decoded_str[colon_pos + 1..].to_string();
        Ok((username, password))
    } else {
        Err("Invalid Basic auth format".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CorsConfig, SameSitePolicy, SecurityConfig, SessionConfig};
    use std::collections::HashMap;

    fn create_test_auth_service() -> AuthService {
        let mut users = HashMap::new();
        let auth_service = AuthService::new(SecurityConfig::default());

        // Add test user
        let password_hash = auth_service.hash_password("test123").unwrap();
        let user_config = UserConfig {
            password_hash,
            roles: vec!["user".to_string()],
            enabled: true,
            email: Some("test@example.com".to_string()),
            full_name: Some("Test User".to_string()),
            last_login: None,
            failed_login_attempts: 0,
            locked_until: None,
        };

        users.insert("testuser".to_string(), user_config);
        auth_service
    }

    #[tokio::test]
    async fn test_password_hashing() {
        let auth_service = create_test_auth_service();
        let password = "test123";

        let hash = auth_service.hash_password(password).unwrap();
        assert!(auth_service.verify_password(password, &hash).unwrap());
        assert!(!auth_service.verify_password("wrong", &hash).unwrap());
    }

    #[tokio::test]
    async fn test_user_authentication() {
        let auth_service = create_test_auth_service();

        // Valid authentication
        let result = auth_service
            .authenticate_user("testuser", "test123")
            .await
            .unwrap();
        assert!(matches!(result, AuthResult::Authenticated(_)));

        // Invalid password
        let result = auth_service
            .authenticate_user("testuser", "wrong")
            .await
            .unwrap();
        assert!(matches!(result, AuthResult::Unauthenticated));

        // Non-existent user
        let result = auth_service
            .authenticate_user("nonexistent", "password")
            .await
            .unwrap();
        assert!(matches!(result, AuthResult::Unauthenticated));
    }

    #[tokio::test]
    async fn test_session_management() {
        let auth_service = create_test_auth_service();

        let user = User {
            username: "testuser".to_string(),
            roles: vec!["user".to_string()],
            email: Some("test@example.com".to_string()),
            full_name: Some("Test User".to_string()),
            last_login: None,
            permissions: vec![Permission::GlobalRead, Permission::SparqlQuery],
        };

        // Create session
        let session_id = auth_service.create_session(user.clone()).await.unwrap();

        // Validate session
        let result = auth_service.validate_session(&session_id).await.unwrap();
        assert!(matches!(result, AuthResult::Authenticated(_)));

        // Logout
        auth_service.logout(&session_id).await.unwrap();

        // Session should be invalid after logout
        let result = auth_service.validate_session(&session_id).await.unwrap();
        assert!(matches!(result, AuthResult::Unauthenticated));
    }

    #[test]
    fn test_password_strength() {
        use passwords::*;

        assert_eq!(check_password_strength("123"), PasswordStrength::Weak);
        assert_eq!(check_password_strength("password"), PasswordStrength::Weak);
        assert_eq!(
            check_password_strength("Password1"),
            PasswordStrength::Medium
        );
        assert_eq!(
            check_password_strength("Password1!"),
            PasswordStrength::Strong
        );
        assert_eq!(
            check_password_strength("VerySecurePassword123!"),
            PasswordStrength::Strong
        );
    }

    #[test]
    fn test_permission_system() {
        let user = User {
            username: "testuser".to_string(),
            roles: vec!["user".to_string()],
            email: None,
            full_name: None,
            last_login: None,
            permissions: vec![Permission::GlobalRead, Permission::SparqlQuery],
        };

        let auth_service = create_test_auth_service();

        assert!(auth_service.has_permission(&user, &Permission::GlobalRead));
        assert!(auth_service.has_permission(&user, &Permission::SparqlQuery));
        assert!(!auth_service.has_permission(&user, &Permission::SparqlUpdate));
        assert!(!auth_service.has_permission(&user, &Permission::SystemConfig));
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_jwt_operations() {
        let mut config = SecurityConfig::default();
        config.jwt = Some(JwtConfig {
            secret: "a".repeat(32),
            expiration_secs: 3600,
            issuer: "oxirs-fuseki".to_string(),
            audience: "test".to_string(),
        });

        let auth_service = AuthService::new(config);

        let user = User {
            username: "testuser".to_string(),
            roles: vec!["user".to_string()],
            email: None,
            full_name: None,
            last_login: None,
            permissions: vec![Permission::GlobalRead],
        };

        // Generate token
        let token = auth_service.generate_jwt_token(&user).unwrap();
        assert!(!token.is_empty());

        // Validate token
        let validation = auth_service.validate_jwt_token(&token).unwrap();
        assert_eq!(validation.user.username, user.username);
        assert_eq!(validation.user.roles, user.roles);
    }
}
