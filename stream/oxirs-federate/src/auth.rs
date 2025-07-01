//! # Multi-Service Authentication and Identity Propagation
//!
//! This module provides comprehensive authentication and authorization for federated services,
//! including identity propagation, token management, and security policy enforcement.

use anyhow::{anyhow, Result};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Authentication and authorization manager for federated services
#[derive(Debug)]
pub struct AuthManager {
    config: AuthConfig,
    token_store: Arc<RwLock<TokenStore>>,
    identity_propagator: IdentityPropagator,
    policy_engine: PolicyEngine,
    session_manager: SessionManager,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// JWT signing key for tokens
    pub jwt_secret: String,
    /// Token expiration duration
    pub token_expiry: Duration,
    /// Enable identity propagation across services
    pub enable_identity_propagation: bool,
    /// Supported authentication methods
    pub supported_auth_methods: HashSet<AuthMethod>,
    /// Enable service-to-service authentication
    pub enable_service_auth: bool,
    /// Service authentication key
    pub service_auth_key: String,
    /// Enable role-based access control
    pub enable_rbac: bool,
    /// Default permissions for authenticated users
    pub default_permissions: HashSet<Permission>,
    /// Token refresh threshold (percentage of expiry time)
    pub refresh_threshold: f64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: "oxirs-federation-secret".to_string(),
            token_expiry: Duration::from_secs(3600), // 1 hour
            enable_identity_propagation: true,
            supported_auth_methods: [AuthMethod::Bearer, AuthMethod::ApiKey, AuthMethod::Basic]
                .into_iter()
                .collect(),
            enable_service_auth: true,
            service_auth_key: "oxirs-service-key".to_string(),
            enable_rbac: true,
            default_permissions: [Permission::QueryRead, Permission::SchemaRead]
                .into_iter()
                .collect(),
            refresh_threshold: 0.8, // Refresh when 80% of token life is used
        }
    }
}

/// Authentication methods supported
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthMethod {
    /// Bearer token authentication
    Bearer,
    /// API key authentication
    ApiKey,
    /// Basic HTTP authentication
    Basic,
    /// OAuth2 authentication
    OAuth2,
    /// Service-to-service authentication
    ServiceToService,
    /// SAML authentication
    Saml,
}

/// System permissions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read query results
    QueryRead,
    /// Execute queries
    QueryExecute,
    /// Read schema information
    SchemaRead,
    /// Modify schema
    SchemaWrite,
    /// Administration access
    Admin,
    /// Service configuration
    ServiceConfig,
    /// Federation management
    FederationManage,
    /// Monitoring access
    MonitoringRead,
    /// Custom permission
    Custom(String),
}

/// User identity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// Unique user identifier
    pub user_id: String,
    /// Username
    pub username: String,
    /// User email
    pub email: Option<String>,
    /// User roles
    pub roles: HashSet<String>,
    /// User permissions
    pub permissions: HashSet<Permission>,
    /// Authentication method used
    pub auth_method: AuthMethod,
    /// Authentication timestamp
    pub authenticated_at: u64,
    /// Token expiration timestamp
    pub expires_at: u64,
    /// Additional claims
    pub claims: HashMap<String, serde_json::Value>,
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Token identifier
    pub token_id: String,
    /// JWT token string
    pub token: String,
    /// Token type
    pub token_type: TokenType,
    /// Associated identity
    pub identity: Identity,
    /// Token creation time
    pub created_at: SystemTime,
    /// Token expiration time
    pub expires_at: SystemTime,
    /// Whether token is active
    pub is_active: bool,
}

/// Token types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    /// Access token for API access
    Access,
    /// Refresh token for token renewal
    Refresh,
    /// Service-to-service token
    Service,
    /// Temporary session token
    Session,
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// Expiration time
    pub exp: u64,
    /// Issued at
    pub iat: u64,
    /// Not before
    pub nbf: u64,
    /// JWT ID
    pub jti: String,
    /// Custom claims
    pub user_id: String,
    pub username: String,
    pub roles: HashSet<String>,
    pub permissions: HashSet<Permission>,
    pub auth_method: AuthMethod,
}

/// Token storage and management
#[derive(Debug)]
struct TokenStore {
    active_tokens: HashMap<String, AuthToken>,
    user_tokens: HashMap<String, HashSet<String>>, // user_id -> token_ids
    revoked_tokens: HashSet<String>,
}

impl TokenStore {
    fn new() -> Self {
        Self {
            active_tokens: HashMap::new(),
            user_tokens: HashMap::new(),
            revoked_tokens: HashSet::new(),
        }
    }
}

/// Identity propagation manager
#[derive(Debug)]
pub struct IdentityPropagator {
    propagation_header: String,
    service_registry: Arc<RwLock<HashMap<String, ServiceAuthConfig>>>,
}

/// Service authentication configuration
#[derive(Debug, Clone)]
pub struct ServiceAuthConfig {
    pub service_id: String,
    pub auth_method: AuthMethod,
    pub api_key: Option<String>,
    pub certificate: Option<String>,
    pub trust_level: TrustLevel,
    pub required_permissions: HashSet<Permission>,
}

/// Trust levels for services
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Untrusted = 0,
    Basic = 1,
    Trusted = 2,
    HighlyTrusted = 3,
}

/// Policy engine for authorization
#[derive(Debug)]
pub struct PolicyEngine {
    policies: Arc<RwLock<HashMap<String, AuthPolicy>>>,
    role_permissions: Arc<RwLock<HashMap<String, HashSet<Permission>>>>,
}

/// Authorization policy
#[derive(Debug, Clone)]
pub struct AuthPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub rules: Vec<PolicyRule>,
    pub is_active: bool,
}

/// Policy rule
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub rule_id: String,
    pub condition: PolicyCondition,
    pub action: PolicyAction,
    pub effect: PolicyEffect,
}

/// Policy conditions
#[derive(Debug, Clone)]
pub enum PolicyCondition {
    HasRole(String),
    HasPermission(Permission),
    IsUser(String),
    ServiceId(String),
    TimeRange { start: u64, end: u64 },
    IpAddress(String),
    And(Vec<PolicyCondition>),
    Or(Vec<PolicyCondition>),
    Not(Box<PolicyCondition>),
}

/// Policy actions
#[derive(Debug, Clone)]
pub enum PolicyAction {
    Allow,
    Deny,
    RequireAdditionalAuth,
    Log,
    RateLimit { max_requests: u32, window: Duration },
}

/// Policy effects
#[derive(Debug, Clone)]
pub enum PolicyEffect {
    Allow,
    Deny,
}

/// Session management
#[derive(Debug)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    config: SessionConfig,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub session_timeout: Duration,
    pub max_concurrent_sessions: usize,
    pub enable_session_refresh: bool,
}

/// User session information
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub identity: Identity,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub is_active: bool,
}

/// Authentication result
#[derive(Debug, Clone)]
pub struct AuthResult {
    pub success: bool,
    pub identity: Option<Identity>,
    pub token: Option<AuthToken>,
    pub error: Option<String>,
    pub required_actions: Vec<AuthAction>,
}

/// Required authentication actions
#[derive(Debug, Clone)]
pub enum AuthAction {
    ProvideCredentials,
    RefreshToken,
    MultiFactorAuth,
    AgreeToTerms,
    UpdatePassword,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new() -> Self {
        Self::with_config(AuthConfig::default())
    }

    /// Create a new authentication manager with configuration
    pub fn with_config(config: AuthConfig) -> Self {
        let token_store = Arc::new(RwLock::new(TokenStore::new()));
        let identity_propagator = IdentityPropagator::new();
        let policy_engine = PolicyEngine::new();
        let session_manager = SessionManager::new();

        Self {
            config,
            token_store,
            identity_propagator,
            policy_engine,
            session_manager,
        }
    }

    /// Authenticate a user with credentials
    pub async fn authenticate(
        &self,
        auth_method: AuthMethod,
        credentials: AuthCredentials,
    ) -> Result<AuthResult> {
        debug!("Authenticating user with method: {:?}", auth_method);

        if !self.config.supported_auth_methods.contains(&auth_method) {
            return Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some(format!(
                    "Authentication method {:?} not supported",
                    auth_method
                )),
                required_actions: vec![],
            });
        }

        match auth_method {
            AuthMethod::Bearer => self.authenticate_bearer(&credentials).await,
            AuthMethod::ApiKey => self.authenticate_api_key(&credentials).await,
            AuthMethod::Basic => self.authenticate_basic(&credentials).await,
            AuthMethod::ServiceToService => self.authenticate_service(&credentials).await,
            _ => Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Authentication method not implemented".to_string()),
                required_actions: vec![],
            }),
        }
    }

    /// Validate an authentication token
    pub async fn validate_token(&self, token: &str) -> Result<AuthResult> {
        let token_store = self.token_store.read().await;

        // Check if token is revoked
        if token_store.revoked_tokens.contains(token) {
            return Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Token has been revoked".to_string()),
                required_actions: vec![AuthAction::ProvideCredentials],
            });
        }

        // Validate JWT token
        let claims = self.decode_jwt(token)?;

        // Check expiration
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        if claims.exp < current_time {
            return Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Token has expired".to_string()),
                required_actions: vec![AuthAction::RefreshToken],
            });
        }

        // Reconstruct identity from claims
        let identity = Identity {
            user_id: claims.user_id,
            username: claims.username,
            email: None, // Would be stored in custom claims
            roles: claims.roles,
            permissions: claims.permissions,
            auth_method: claims.auth_method,
            authenticated_at: claims.iat,
            expires_at: claims.exp,
            claims: HashMap::new(),
        };

        Ok(AuthResult {
            success: true,
            identity: Some(identity),
            token: None,
            error: None,
            required_actions: vec![],
        })
    }

    /// Create a propagated identity for service-to-service calls
    pub async fn create_propagated_identity(
        &self,
        original_identity: &Identity,
        target_service: &str,
    ) -> Result<String> {
        if !self.config.enable_identity_propagation {
            return Err(anyhow!("Identity propagation is disabled"));
        }

        self.identity_propagator
            .create_propagated_token(original_identity, target_service, &self.config)
            .await
    }

    /// Authorize an action for an identity
    pub async fn authorize(
        &self,
        identity: &Identity,
        action: &str,
        resource: &str,
    ) -> Result<bool> {
        if !self.config.enable_rbac {
            return Ok(true); // RBAC disabled, allow all
        }

        self.policy_engine
            .evaluate_authorization(identity, action, resource)
            .await
    }

    /// Create a new access token
    pub async fn create_access_token(&self, identity: &Identity) -> Result<AuthToken> {
        let token_id = Uuid::new_v4().to_string();
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let expires_at = current_time + self.config.token_expiry.as_secs();

        let claims = JwtClaims {
            sub: identity.user_id.clone(),
            iss: "oxirs-federation".to_string(),
            aud: "oxirs-services".to_string(),
            exp: expires_at,
            iat: current_time,
            nbf: current_time,
            jti: token_id.clone(),
            user_id: identity.user_id.clone(),
            username: identity.username.clone(),
            roles: identity.roles.clone(),
            permissions: identity.permissions.clone(),
            auth_method: identity.auth_method.clone(),
        };

        let jwt_token = self.encode_jwt(&claims)?;

        let auth_token = AuthToken {
            token_id: token_id.clone(),
            token: jwt_token,
            token_type: TokenType::Access,
            identity: identity.clone(),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.token_expiry,
            is_active: true,
        };

        // Store token
        let mut token_store = self.token_store.write().await;
        token_store
            .active_tokens
            .insert(token_id.clone(), auth_token.clone());
        token_store
            .user_tokens
            .entry(identity.user_id.clone())
            .or_default()
            .insert(token_id);

        Ok(auth_token)
    }

    /// Revoke a token
    pub async fn revoke_token(&self, token_id: &str) -> Result<()> {
        let mut token_store = self.token_store.write().await;

        if let Some(token) = token_store.active_tokens.remove(token_id) {
            token_store.revoked_tokens.insert(token.token);

            // Remove from user tokens
            if let Some(user_tokens) = token_store.user_tokens.get_mut(&token.identity.user_id) {
                user_tokens.remove(token_id);
            }
        }

        Ok(())
    }

    /// Get authentication statistics
    pub async fn get_auth_statistics(&self) -> AuthStatistics {
        let token_store = self.token_store.read().await;
        let sessions = self.session_manager.sessions.read().await;

        let active_tokens = token_store.active_tokens.len();
        let revoked_tokens = token_store.revoked_tokens.len();
        let active_sessions = sessions.values().filter(|s| s.is_active).count();
        let unique_users = token_store.user_tokens.len();

        AuthStatistics {
            active_tokens,
            revoked_tokens,
            active_sessions,
            unique_users,
            total_authentications: active_tokens + revoked_tokens,
        }
    }

    // Private helper methods

    async fn authenticate_bearer(&self, credentials: &AuthCredentials) -> Result<AuthResult> {
        if let AuthCredentials::Bearer { token } = credentials {
            self.validate_token(token).await
        } else {
            Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Invalid credentials for bearer authentication".to_string()),
                required_actions: vec![AuthAction::ProvideCredentials],
            })
        }
    }

    async fn authenticate_api_key(&self, credentials: &AuthCredentials) -> Result<AuthResult> {
        if let AuthCredentials::ApiKey { key } = credentials {
            // In a real implementation, you'd validate the API key against a database
            // For now, we'll create a simple validation
            if key == "valid-api-key" {
                let identity = Identity {
                    user_id: "api-user".to_string(),
                    username: "API User".to_string(),
                    email: None,
                    roles: ["api_user"].into_iter().map(String::from).collect(),
                    permissions: self.config.default_permissions.clone(),
                    auth_method: AuthMethod::ApiKey,
                    authenticated_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                    expires_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                        + self.config.token_expiry.as_secs(),
                    claims: HashMap::new(),
                };

                let token = self.create_access_token(&identity).await?;

                Ok(AuthResult {
                    success: true,
                    identity: Some(identity),
                    token: Some(token),
                    error: None,
                    required_actions: vec![],
                })
            } else {
                Ok(AuthResult {
                    success: false,
                    identity: None,
                    token: None,
                    error: Some("Invalid API key".to_string()),
                    required_actions: vec![AuthAction::ProvideCredentials],
                })
            }
        } else {
            Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Invalid credentials for API key authentication".to_string()),
                required_actions: vec![AuthAction::ProvideCredentials],
            })
        }
    }

    async fn authenticate_basic(&self, credentials: &AuthCredentials) -> Result<AuthResult> {
        if let AuthCredentials::Basic { username, password } = credentials {
            // In a real implementation, you'd validate against a user store
            if username == "admin" && password == "password" {
                let identity = Identity {
                    user_id: "admin".to_string(),
                    username: username.clone(),
                    email: Some("admin@example.com".to_string()),
                    roles: ["admin", "user"].into_iter().map(String::from).collect(),
                    permissions: [
                        Permission::QueryRead,
                        Permission::QueryExecute,
                        Permission::SchemaRead,
                        Permission::SchemaWrite,
                        Permission::Admin,
                    ]
                    .into_iter()
                    .collect(),
                    auth_method: AuthMethod::Basic,
                    authenticated_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                    expires_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                        + self.config.token_expiry.as_secs(),
                    claims: HashMap::new(),
                };

                let token = self.create_access_token(&identity).await?;

                Ok(AuthResult {
                    success: true,
                    identity: Some(identity),
                    token: Some(token),
                    error: None,
                    required_actions: vec![],
                })
            } else {
                Ok(AuthResult {
                    success: false,
                    identity: None,
                    token: None,
                    error: Some("Invalid username or password".to_string()),
                    required_actions: vec![AuthAction::ProvideCredentials],
                })
            }
        } else {
            Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Invalid credentials for basic authentication".to_string()),
                required_actions: vec![AuthAction::ProvideCredentials],
            })
        }
    }

    async fn authenticate_service(&self, credentials: &AuthCredentials) -> Result<AuthResult> {
        if let AuthCredentials::Service { service_id, secret } = credentials {
            if secret == &self.config.service_auth_key {
                let identity = Identity {
                    user_id: format!("service:{}", service_id),
                    username: format!("Service {}", service_id),
                    email: None,
                    roles: ["service"].into_iter().map(String::from).collect(),
                    permissions: [
                        Permission::QueryRead,
                        Permission::QueryExecute,
                        Permission::SchemaRead,
                    ]
                    .into_iter()
                    .collect(),
                    auth_method: AuthMethod::ServiceToService,
                    authenticated_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                    expires_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                        + self.config.token_expiry.as_secs(),
                    claims: [(
                        "service_id".to_string(),
                        serde_json::Value::String(service_id.clone()),
                    )]
                    .into_iter()
                    .collect(),
                };

                let token = self.create_access_token(&identity).await?;

                Ok(AuthResult {
                    success: true,
                    identity: Some(identity),
                    token: Some(token),
                    error: None,
                    required_actions: vec![],
                })
            } else {
                Ok(AuthResult {
                    success: false,
                    identity: None,
                    token: None,
                    error: Some("Invalid service credentials".to_string()),
                    required_actions: vec![AuthAction::ProvideCredentials],
                })
            }
        } else {
            Ok(AuthResult {
                success: false,
                identity: None,
                token: None,
                error: Some("Invalid credentials for service authentication".to_string()),
                required_actions: vec![AuthAction::ProvideCredentials],
            })
        }
    }

    fn encode_jwt(&self, claims: &JwtClaims) -> Result<String> {
        let header = Header::new(Algorithm::HS256);
        let encoding_key = EncodingKey::from_secret(self.config.jwt_secret.as_bytes());

        encode(&header, claims, &encoding_key).map_err(|e| anyhow!("Failed to encode JWT: {}", e))
    }

    fn decode_jwt(&self, token: &str) -> Result<JwtClaims> {
        let decoding_key = DecodingKey::from_secret(self.config.jwt_secret.as_bytes());
        let validation = Validation::new(Algorithm::HS256);

        decode::<JwtClaims>(token, &decoding_key, &validation)
            .map(|data| data.claims)
            .map_err(|e| anyhow!("Failed to decode JWT: {}", e))
    }
}

impl Default for AuthManager {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityPropagator {
    fn new() -> Self {
        Self {
            propagation_header: "X-Oxirs-Identity".to_string(),
            service_registry: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn create_propagated_token(
        &self,
        identity: &Identity,
        target_service: &str,
        config: &AuthConfig,
    ) -> Result<String> {
        // Create a reduced identity for propagation
        let propagated_identity = Identity {
            user_id: identity.user_id.clone(),
            username: identity.username.clone(),
            email: identity.email.clone(),
            roles: identity.roles.clone(),
            permissions: identity.permissions.clone(),
            auth_method: AuthMethod::ServiceToService,
            authenticated_at: identity.authenticated_at,
            expires_at: identity.expires_at,
            claims: [(
                "original_auth_method".to_string(),
                serde_json::Value::String(format!("{:?}", identity.auth_method)),
            )]
            .into_iter()
            .collect(),
        };

        // Encode as JWT
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let claims = JwtClaims {
            sub: propagated_identity.user_id.clone(),
            iss: "oxirs-federation".to_string(),
            aud: target_service.to_string(),
            exp: propagated_identity.expires_at,
            iat: current_time,
            nbf: current_time,
            jti: Uuid::new_v4().to_string(),
            user_id: propagated_identity.user_id,
            username: propagated_identity.username,
            roles: propagated_identity.roles,
            permissions: propagated_identity.permissions,
            auth_method: propagated_identity.auth_method,
        };

        let header = Header::new(Algorithm::HS256);
        let encoding_key = EncodingKey::from_secret(config.jwt_secret.as_bytes());

        encode(&header, &claims, &encoding_key)
            .map_err(|e| anyhow!("Failed to create propagated token: {}", e))
    }
}

impl PolicyEngine {
    fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            role_permissions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn evaluate_authorization(
        &self,
        identity: &Identity,
        action: &str,
        resource: &str,
    ) -> Result<bool> {
        // Simple permission-based authorization
        let required_permission = match action {
            "query" => Permission::QueryExecute,
            "read_schema" => Permission::SchemaRead,
            "write_schema" => Permission::SchemaWrite,
            "admin" => Permission::Admin,
            _ => return Ok(false),
        };

        Ok(identity.permissions.contains(&required_permission))
    }
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config: SessionConfig {
                session_timeout: Duration::from_secs(1800), // 30 minutes
                max_concurrent_sessions: 10,
                enable_session_refresh: true,
            },
        }
    }
}

/// Authentication credentials
#[derive(Debug, Clone)]
pub enum AuthCredentials {
    Bearer {
        token: String,
    },
    ApiKey {
        key: String,
    },
    Basic {
        username: String,
        password: String,
    },
    Service {
        service_id: String,
        secret: String,
    },
    OAuth2 {
        access_token: String,
        refresh_token: Option<String>,
    },
}

/// Authentication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatistics {
    pub active_tokens: usize,
    pub revoked_tokens: usize,
    pub active_sessions: usize,
    pub unique_users: usize,
    pub total_authentications: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_manager_creation() {
        let auth_manager = AuthManager::new();
        assert!(auth_manager.config.enable_identity_propagation);
    }

    #[tokio::test]
    async fn test_basic_authentication() {
        let auth_manager = AuthManager::new();

        let credentials = AuthCredentials::Basic {
            username: "admin".to_string(),
            password: "password".to_string(),
        };

        let result = auth_manager
            .authenticate(AuthMethod::Basic, credentials)
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.identity.is_some());
        assert!(result.token.is_some());
    }

    #[tokio::test]
    async fn test_api_key_authentication() {
        let auth_manager = AuthManager::new();

        let credentials = AuthCredentials::ApiKey {
            key: "valid-api-key".to_string(),
        };

        let result = auth_manager
            .authenticate(AuthMethod::ApiKey, credentials)
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.identity.is_some());
    }

    #[tokio::test]
    async fn test_service_authentication() {
        let auth_manager = AuthManager::new();

        let credentials = AuthCredentials::Service {
            service_id: "test-service".to_string(),
            secret: "oxirs-service-key".to_string(),
        };

        let result = auth_manager
            .authenticate(AuthMethod::ServiceToService, credentials)
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.identity.is_some());
    }

    #[tokio::test]
    async fn test_token_validation() {
        let auth_manager = AuthManager::new();

        // First authenticate to get a token
        let credentials = AuthCredentials::Basic {
            username: "admin".to_string(),
            password: "password".to_string(),
        };

        let auth_result = auth_manager
            .authenticate(AuthMethod::Basic, credentials)
            .await
            .unwrap();
        assert!(auth_result.success);

        let token = auth_result.token.unwrap();

        // Now validate the token
        let validation_result = auth_manager.validate_token(&token.token).await.unwrap();
        assert!(validation_result.success);
    }

    #[tokio::test]
    async fn test_identity_propagation() {
        let auth_manager = AuthManager::new();

        let identity = Identity {
            user_id: "test-user".to_string(),
            username: "Test User".to_string(),
            email: Some("test@example.com".to_string()),
            roles: ["user"].into_iter().map(String::from).collect(),
            permissions: [Permission::QueryRead].into_iter().collect(),
            auth_method: AuthMethod::Basic,
            authenticated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            claims: HashMap::new(),
        };

        let propagated_token = auth_manager
            .create_propagated_identity(&identity, "target-service")
            .await;

        assert!(propagated_token.is_ok());
    }

    #[tokio::test]
    async fn test_authorization() {
        let auth_manager = AuthManager::new();

        let identity = Identity {
            user_id: "test-user".to_string(),
            username: "Test User".to_string(),
            email: None,
            roles: ["user"].into_iter().map(String::from).collect(),
            permissions: [Permission::QueryExecute, Permission::SchemaRead]
                .into_iter()
                .collect(),
            auth_method: AuthMethod::Basic,
            authenticated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            claims: HashMap::new(),
        };

        let can_query = auth_manager
            .authorize(&identity, "query", "sparql")
            .await
            .unwrap();
        assert!(can_query);

        let can_admin = auth_manager
            .authorize(&identity, "admin", "system")
            .await
            .unwrap();
        assert!(!can_admin); // User doesn't have admin permission
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let auth_manager = AuthManager::new();

        let credentials = AuthCredentials::Basic {
            username: "admin".to_string(),
            password: "password".to_string(),
        };

        let auth_result = auth_manager
            .authenticate(AuthMethod::Basic, credentials)
            .await
            .unwrap();
        let token = auth_result.token.unwrap();

        // Revoke the token
        auth_manager.revoke_token(&token.token_id).await.unwrap();

        // Token validation should fail
        let validation_result = auth_manager.validate_token(&token.token).await.unwrap();
        assert!(!validation_result.success);
    }
}
