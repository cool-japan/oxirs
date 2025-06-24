//! Comprehensive authentication and authorization system

use crate::config::{SecurityConfig, UserConfig, JwtConfig, OAuthConfig, LdapConfig};
use crate::error::{FusekiError, FusekiResult};
use axum::{
    extract::{FromRequestParts, State},
    http::{request::Parts, StatusCode, HeaderMap},
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use axum_extra::headers::{authorization::Bearer, Authorization, HeaderMapExt};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[cfg(feature = "auth")]
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

/// Authentication result
#[derive(Debug, Clone)]
pub enum AuthResult {
    Authenticated(User),
    Unauthenticated,
    Forbidden,
    Expired,
    Invalid,
    Locked,
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// JWT claims structure
#[cfg(feature = "auth")]
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,       // Subject (username)
    pub exp: usize,        // Expiration time
    pub iat: usize,        // Issued at
    pub iss: String,       // Issuer
    pub aud: String,       // Audience
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
}

/// Authentication service manager
#[derive(Clone)]
pub struct AuthService {
    config: Arc<SecurityConfig>,
    users: Arc<RwLock<HashMap<String, UserConfig>>>,
    sessions: Arc<RwLock<HashMap<String, UserSession>>>,
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
        
        Self {
            config: Arc::new(config),
            users: Arc::new(RwLock::new(users)),
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Authenticate user with username/password
    pub async fn authenticate_user(&self, username: &str, password: &str) -> FusekiResult<AuthResult> {
        let users = self.users.read().await;
        
        if let Some(user_config) = users.get(username) {
            // Check if user is enabled
            if !user_config.enabled {
                warn!("Login attempt for disabled user: {}", username);
                return Ok(AuthResult::Forbidden);
            }
            
            // Check if user is locked
            if let Some(locked_until) = user_config.locked_until {
                if locked_until > Utc::now() {
                    warn!("Login attempt for locked user: {} (locked until {})", username, locked_until);
                    return Ok(AuthResult::Locked);
                }
            }
            
            // Verify password
            if self.verify_password(password, &user_config.password_hash)? {
                debug!("Successful authentication for user: {}", username);
                
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
                
                Ok(AuthResult::Authenticated(user))
            } else {
                warn!("Failed authentication attempt for user: {}", username);
                
                // Increment failed login attempts
                self.increment_failed_attempts(username).await?;
                
                Ok(AuthResult::Unauthenticated)
            }
        } else {
            warn!("Authentication attempt for non-existent user: {}", username);
            Ok(AuthResult::Unauthenticated)
        }
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
            
            encode(&header, &claims, &encoding_key)
                .map_err(|e| FusekiError::authentication(format!("Failed to generate JWT token: {}", e)))
        } else {
            Err(FusekiError::configuration("JWT configuration not available"))
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
            Err(FusekiError::configuration("JWT configuration not available"))
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
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);
        
        info!("Created session for user: {}", session.user.username);
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
        Ok(argon2.verify_password(password.as_bytes(), &parsed_hash).is_ok())
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
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::SparqlQuery,
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
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::SparqlQuery,
                    ]);
                }
                _ => {
                    // Custom role - could be dataset-specific
                    if role.starts_with("dataset:") {
                        let dataset_name = role.strip_prefix("dataset:").unwrap_or("");
                        if role.ends_with(":admin") {
                            let dataset_name = dataset_name.strip_suffix(":admin").unwrap_or(dataset_name);
                            permissions.push(Permission::DatasetAdmin(dataset_name.to_string()));
                        } else if role.ends_with(":write") {
                            let dataset_name = dataset_name.strip_suffix(":write").unwrap_or(dataset_name);
                            permissions.push(Permission::DatasetWrite(dataset_name.to_string()));
                        } else if role.ends_with(":read") {
                            let dataset_name = dataset_name.strip_suffix(":read").unwrap_or(dataset_name);
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
                ("writer", Permission::GlobalRead | Permission::GlobalWrite | 
                            Permission::SparqlQuery | Permission::SparqlUpdate | 
                            Permission::GraphStore) => return true,
                ("reader", Permission::GlobalRead | Permission::SparqlQuery) => return true,
                ("user", Permission::GlobalRead | Permission::SparqlQuery) => return true,
                _ => continue,
            }
        }
        false
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
        let auth_service = AuthService::from_request_parts(parts, state).await
            .map_err(|_| AuthError::ServiceUnavailable)?;

        // Try JWT authentication first
        #[cfg(feature = "auth")]
        if let Some(auth_header) = parts.headers.typed_get::<Authorization<Bearer>>() {
            let token = auth_header.token();
            
            match auth_service.validate_jwt_token(token) {
                Ok(validation) => {
                    return Ok(AuthUser(validation.user));
                }
                Err(_) => {
                    return Err(AuthError::InvalidToken);
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
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authentication token"),
            AuthError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid authentication token"),
            AuthError::TokenExpired => (StatusCode::UNAUTHORIZED, "Authentication token expired"),
            AuthError::ServiceUnavailable => (StatusCode::SERVICE_UNAVAILABLE, "Authentication service unavailable"),
            AuthError::InsufficientPermissions => (StatusCode::FORBIDDEN, "Insufficient permissions"),
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
        
        let score = (length >= 8) as u8 +
                   (length >= 12) as u8 +
                   has_uppercase as u8 +
                   has_lowercase as u8 +
                   has_digit as u8 +
                   has_special as u8;
        
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SecurityConfig, CorsConfig, SessionConfig, SameSitePolicy};
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
        let result = auth_service.authenticate_user("testuser", "test123").await.unwrap();
        assert!(matches!(result, AuthResult::Authenticated(_)));
        
        // Invalid password
        let result = auth_service.authenticate_user("testuser", "wrong").await.unwrap();
        assert!(matches!(result, AuthResult::Unauthenticated));
        
        // Non-existent user
        let result = auth_service.authenticate_user("nonexistent", "password").await.unwrap();
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
        assert_eq!(check_password_strength("Password1"), PasswordStrength::Medium);
        assert_eq!(check_password_strength("Password1!"), PasswordStrength::Strong);
        assert_eq!(check_password_strength("VerySecurePassword123!"), PasswordStrength::Strong);
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