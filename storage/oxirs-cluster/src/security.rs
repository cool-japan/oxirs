//! Security and Authentication Module
//!
//! Provides comprehensive security features including authentication, authorization,
//! encryption, and audit logging for the distributed cluster.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::raft::OxirsNodeId;

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enable_tls: bool,
    pub require_client_auth: bool,
    pub jwt_secret: String,
    pub token_expiry_hours: u64,
    pub max_failed_attempts: u32,
    pub lockout_duration_minutes: u64,
    pub enable_audit_logging: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enable_tls: true,
            require_client_auth: true,
            jwt_secret: "default-secret-change-in-production".to_string(),
            token_expiry_hours: 24,
            max_failed_attempts: 5,
            lockout_duration_minutes: 30,
            enable_audit_logging: true,
        }
    }
}

/// User authentication information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    pub password_hash: String,
    pub roles: HashSet<String>,
    pub permissions: HashSet<Permission>,
    pub created_at: SystemTime,
    pub last_login: Option<SystemTime>,
    pub is_active: bool,
}

/// System permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Permission {
    // Cluster management
    ClusterRead,
    ClusterWrite,
    ClusterAdmin,

    // Node management
    NodeAdd,
    NodeRemove,
    NodeStatus,

    // Query permissions
    QueryRead,
    QueryWrite,
    QueryAdmin,

    // Data permissions
    DataRead,
    DataWrite,
    DataDelete,
    DataAdmin,

    // System permissions
    SystemMonitor,
    SystemConfig,
    SystemAdmin,

    // Audit permissions
    AuditRead,
    AuditWrite,
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token_id: String,
    pub username: String,
    pub roles: HashSet<String>,
    pub permissions: HashSet<Permission>,
    pub issued_at: SystemTime,
    pub expires_at: SystemTime,
    pub node_id: Option<OxirsNodeId>,
}

impl AuthToken {
    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }

    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(role)
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: SystemTime,
    pub event_type: AuditEventType,
    pub username: Option<String>,
    pub node_id: OxirsNodeId,
    pub source_ip: Option<String>,
    pub operation: String,
    pub resource: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub additional_data: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemOperation,
    SecurityEvent,
    ClusterOperation,
}

/// Failed login attempt tracking
#[derive(Debug, Clone)]
struct FailedAttempt {
    username: String,
    attempts: u32,
    last_attempt: SystemTime,
    lockout_until: Option<SystemTime>,
}

/// Security manager for the cluster
pub struct SecurityManager {
    config: AuthConfig,
    users: Arc<RwLock<HashMap<String, User>>>,
    active_tokens: Arc<RwLock<HashMap<String, AuthToken>>>,
    failed_attempts: Arc<RwLock<HashMap<String, FailedAttempt>>>,
    audit_log: Arc<RwLock<Vec<AuditEntry>>>,
    node_id: OxirsNodeId,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new(config: AuthConfig, node_id: OxirsNodeId) -> Self {
        Self {
            config,
            users: Arc::new(RwLock::new(HashMap::new())),
            active_tokens: Arc::new(RwLock::new(HashMap::new())),
            failed_attempts: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            node_id,
        }
    }

    /// Initialize with default admin user
    pub async fn initialize(&self) -> Result<()> {
        let admin_user = User {
            username: "admin".to_string(),
            password_hash: self.hash_password("admin123")?,
            roles: vec!["admin".to_string()].into_iter().collect(),
            permissions: vec![
                Permission::SystemAdmin,
                Permission::ClusterAdmin,
                Permission::DataAdmin,
                Permission::QueryAdmin,
                Permission::AuditRead,
                Permission::AuditWrite,
            ]
            .into_iter()
            .collect(),
            created_at: SystemTime::now(),
            last_login: None,
            is_active: true,
        };

        let mut users = self.users.write().await;
        users.insert("admin".to_string(), admin_user);

        info!("Security manager initialized with default admin user");
        self.audit_system_event("Security manager initialized")
            .await;

        Ok(())
    }

    /// Create a new user
    pub async fn create_user(
        &self,
        username: String,
        password: String,
        roles: HashSet<String>,
        permissions: HashSet<Permission>,
    ) -> Result<()> {
        let mut users = self.users.write().await;

        if users.contains_key(&username) {
            return Err(anyhow::anyhow!("User already exists: {}", username));
        }

        let user = User {
            username: username.clone(),
            password_hash: self.hash_password(&password)?,
            roles,
            permissions,
            created_at: SystemTime::now(),
            last_login: None,
            is_active: true,
        };

        users.insert(username.clone(), user);
        info!("Created user: {}", username);
        self.audit_system_event(&format!("User created: {}", username))
            .await;

        Ok(())
    }

    /// Authenticate user and return token
    pub async fn authenticate(
        &self,
        username: &str,
        password: &str,
        source_ip: Option<String>,
    ) -> Result<AuthToken> {
        // Check if user is locked out
        if self.is_user_locked_out(username).await {
            self.audit_authentication_event(
                Some(username.to_string()),
                source_ip.clone(),
                "Login attempt - user locked out",
                false,
                Some("User account is locked due to too many failed attempts".to_string()),
            )
            .await;
            return Err(anyhow::anyhow!(
                "Account is locked due to too many failed attempts"
            ));
        }

        let users = self.users.read().await;
        let user = users
            .get(username)
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        if !user.is_active {
            self.audit_authentication_event(
                Some(username.to_string()),
                source_ip,
                "Login attempt - inactive user",
                false,
                Some("User account is inactive".to_string()),
            )
            .await;
            return Err(anyhow::anyhow!("User account is inactive"));
        }

        if !self.verify_password(password, &user.password_hash)? {
            self.record_failed_attempt(username).await;
            self.audit_authentication_event(
                Some(username.to_string()),
                source_ip,
                "Login attempt - invalid password",
                false,
                Some("Invalid password".to_string()),
            )
            .await;
            return Err(anyhow::anyhow!("Invalid credentials"));
        }

        // Clear failed attempts on successful login
        self.clear_failed_attempts(username).await;

        // Create token
        let token = self.create_token(user).await?;

        // Update last login
        drop(users);
        let mut users = self.users.write().await;
        if let Some(user) = users.get_mut(username) {
            user.last_login = Some(SystemTime::now());
        }

        self.audit_authentication_event(
            Some(username.to_string()),
            source_ip,
            "Successful login",
            true,
            None,
        )
        .await;

        info!("User {} authenticated successfully", username);
        Ok(token)
    }

    /// Validate an authentication token
    pub async fn validate_token(&self, token_id: &str) -> Result<AuthToken> {
        let tokens = self.active_tokens.read().await;
        let token = tokens
            .get(token_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid token"))?;

        if token.is_expired() {
            drop(tokens);
            self.revoke_token(token_id).await?;
            return Err(anyhow::anyhow!("Token expired"));
        }

        Ok(token.clone())
    }

    /// Check if user has permission
    pub async fn check_permission(&self, token_id: &str, permission: Permission) -> Result<bool> {
        let token = self.validate_token(token_id).await?;
        Ok(token.has_permission(&permission))
    }

    /// Revoke an authentication token
    pub async fn revoke_token(&self, token_id: &str) -> Result<()> {
        let mut tokens = self.active_tokens.write().await;
        if tokens.remove(token_id).is_some() {
            info!("Token revoked: {}", token_id);
            self.audit_authentication_event(None, None, "Token revoked", true, None)
                .await;
        }
        Ok(())
    }

    /// Get user information
    pub async fn get_user(&self, username: &str) -> Option<User> {
        let users = self.users.read().await;
        users.get(username).cloned()
    }

    /// Update user permissions
    pub async fn update_user_permissions(
        &self,
        username: &str,
        permissions: HashSet<Permission>,
    ) -> Result<()> {
        let mut users = self.users.write().await;
        let user = users
            .get_mut(username)
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        user.permissions = permissions;
        info!("Updated permissions for user: {}", username);
        self.audit_system_event(&format!("Updated permissions for user: {}", username))
            .await;

        Ok(())
    }

    /// Deactivate user
    pub async fn deactivate_user(&self, username: &str) -> Result<()> {
        let mut users = self.users.write().await;
        let user = users
            .get_mut(username)
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        user.is_active = false;
        info!("Deactivated user: {}", username);
        self.audit_system_event(&format!("User deactivated: {}", username))
            .await;

        Ok(())
    }

    /// Get audit log entries
    pub async fn get_audit_log(&self, limit: Option<usize>) -> Vec<AuditEntry> {
        let audit_log = self.audit_log.read().await;
        if let Some(limit) = limit {
            audit_log.iter().rev().take(limit).cloned().collect()
        } else {
            audit_log.clone()
        }
    }

    /// Clean up expired tokens
    pub async fn cleanup_expired_tokens(&self) {
        let mut tokens = self.active_tokens.write().await;
        let before_count = tokens.len();

        tokens.retain(|_, token| !token.is_expired());

        let after_count = tokens.len();
        if before_count != after_count {
            info!("Cleaned up {} expired tokens", before_count - after_count);
        }
    }

    /// Hash password using SHA-256 with salt
    fn hash_password(&self, password: &str) -> Result<String> {
        let salt = "oxirs_cluster_salt"; // In production, use random salt per user
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(salt.as_bytes());
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Verify password hash
    fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        let computed_hash = self.hash_password(password)?;
        Ok(computed_hash == hash)
    }

    /// Create authentication token
    async fn create_token(&self, user: &User) -> Result<AuthToken> {
        let token_id = uuid::Uuid::new_v4().to_string();
        let now = SystemTime::now();
        let expires_at = now + Duration::from_secs(self.config.token_expiry_hours * 3600);

        let token = AuthToken {
            token_id: token_id.clone(),
            username: user.username.clone(),
            roles: user.roles.clone(),
            permissions: user.permissions.clone(),
            issued_at: now,
            expires_at,
            node_id: Some(self.node_id),
        };

        let mut tokens = self.active_tokens.write().await;
        tokens.insert(token_id, token.clone());

        Ok(token)
    }

    /// Check if user is locked out
    async fn is_user_locked_out(&self, username: &str) -> bool {
        let failed_attempts = self.failed_attempts.read().await;
        if let Some(attempt) = failed_attempts.get(username) {
            if let Some(lockout_until) = attempt.lockout_until {
                return SystemTime::now() < lockout_until;
            }
        }
        false
    }

    /// Record failed login attempt
    async fn record_failed_attempt(&self, username: &str) {
        let mut failed_attempts = self.failed_attempts.write().await;
        let now = SystemTime::now();

        let attempt = failed_attempts
            .entry(username.to_string())
            .or_insert(FailedAttempt {
                username: username.to_string(),
                attempts: 0,
                last_attempt: now,
                lockout_until: None,
            });

        attempt.attempts += 1;
        attempt.last_attempt = now;

        if attempt.attempts >= self.config.max_failed_attempts {
            let lockout_duration = Duration::from_secs(self.config.lockout_duration_minutes * 60);
            attempt.lockout_until = Some(now + lockout_duration);
            warn!(
                "User {} locked out due to {} failed attempts",
                username, attempt.attempts
            );
        }
    }

    /// Clear failed attempts for user
    async fn clear_failed_attempts(&self, username: &str) {
        let mut failed_attempts = self.failed_attempts.write().await;
        failed_attempts.remove(username);
    }

    /// Audit authentication event
    async fn audit_authentication_event(
        &self,
        username: Option<String>,
        source_ip: Option<String>,
        operation: &str,
        success: bool,
        error_message: Option<String>,
    ) {
        if !self.config.enable_audit_logging {
            return;
        }

        let entry = AuditEntry {
            timestamp: SystemTime::now(),
            event_type: AuditEventType::Authentication,
            username,
            node_id: self.node_id,
            source_ip,
            operation: operation.to_string(),
            resource: None,
            success,
            error_message,
            additional_data: HashMap::new(),
        };

        let mut audit_log = self.audit_log.write().await;
        audit_log.push(entry);

        // Limit audit log size
        if audit_log.len() > 10000 {
            audit_log.drain(0..1000);
        }
    }

    /// Audit system event
    async fn audit_system_event(&self, operation: &str) {
        if !self.config.enable_audit_logging {
            return;
        }

        let entry = AuditEntry {
            timestamp: SystemTime::now(),
            event_type: AuditEventType::SystemOperation,
            username: None,
            node_id: self.node_id,
            source_ip: None,
            operation: operation.to_string(),
            resource: None,
            success: true,
            error_message: None,
            additional_data: HashMap::new(),
        };

        let mut audit_log = self.audit_log.write().await;
        audit_log.push(entry);

        // Limit audit log size
        if audit_log.len() > 10000 {
            audit_log.drain(0..1000);
        }
    }

    /// Audit data access event
    pub async fn audit_data_access(
        &self,
        username: Option<String>,
        operation: &str,
        resource: Option<String>,
        success: bool,
        error_message: Option<String>,
    ) {
        if !self.config.enable_audit_logging {
            return;
        }

        let entry = AuditEntry {
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataAccess,
            username,
            node_id: self.node_id,
            source_ip: None,
            operation: operation.to_string(),
            resource,
            success,
            error_message,
            additional_data: HashMap::new(),
        };

        let mut audit_log = self.audit_log.write().await;
        audit_log.push(entry);

        // Limit audit log size
        if audit_log.len() > 10000 {
            audit_log.drain(0..1000);
        }
    }

    /// Start background cleanup tasks
    pub async fn start_background_tasks(&self) {
        let security_manager = SecurityManager {
            config: self.config.clone(),
            users: Arc::clone(&self.users),
            active_tokens: Arc::clone(&self.active_tokens),
            failed_attempts: Arc::clone(&self.failed_attempts),
            audit_log: Arc::clone(&self.audit_log),
            node_id: self.node_id,
        };

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            loop {
                interval.tick().await;
                security_manager.cleanup_expired_tokens().await;
            }
        });
    }
}

/// Role-based permission management
pub struct RoleManager {
    roles: HashMap<String, HashSet<Permission>>,
}

impl RoleManager {
    pub fn new() -> Self {
        let mut roles = HashMap::new();

        // Default roles
        roles.insert(
            "admin".to_string(),
            vec![
                Permission::SystemAdmin,
                Permission::ClusterAdmin,
                Permission::DataAdmin,
                Permission::QueryAdmin,
                Permission::AuditRead,
                Permission::AuditWrite,
            ]
            .into_iter()
            .collect(),
        );

        roles.insert(
            "operator".to_string(),
            vec![
                Permission::ClusterRead,
                Permission::ClusterWrite,
                Permission::NodeAdd,
                Permission::NodeRemove,
                Permission::NodeStatus,
                Permission::SystemMonitor,
            ]
            .into_iter()
            .collect(),
        );

        roles.insert(
            "developer".to_string(),
            vec![
                Permission::QueryRead,
                Permission::QueryWrite,
                Permission::DataRead,
                Permission::DataWrite,
                Permission::ClusterRead,
            ]
            .into_iter()
            .collect(),
        );

        roles.insert(
            "readonly".to_string(),
            vec![
                Permission::QueryRead,
                Permission::DataRead,
                Permission::ClusterRead,
                Permission::NodeStatus,
            ]
            .into_iter()
            .collect(),
        );

        Self { roles }
    }

    pub fn get_role_permissions(&self, role: &str) -> Option<&HashSet<Permission>> {
        self.roles.get(role)
    }

    pub fn create_role(&mut self, role: String, permissions: HashSet<Permission>) {
        self.roles.insert(role, permissions);
    }

    pub fn get_all_roles(&self) -> Vec<String> {
        self.roles.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_manager_creation() {
        let config = AuthConfig::default();
        let security_manager = SecurityManager::new(config, 1);
        security_manager.initialize().await.unwrap();

        let admin_user = security_manager.get_user("admin").await;
        assert!(admin_user.is_some());
        assert!(admin_user.unwrap().has_permission(&Permission::SystemAdmin));
    }

    #[tokio::test]
    async fn test_user_authentication() {
        let config = AuthConfig::default();
        let security_manager = SecurityManager::new(config, 1);
        security_manager.initialize().await.unwrap();

        // Test successful authentication
        let token = security_manager
            .authenticate("admin", "admin123", None)
            .await
            .unwrap();
        assert!(!token.is_expired());
        assert!(token.has_permission(&Permission::SystemAdmin));

        // Test failed authentication
        let result = security_manager
            .authenticate("admin", "wrong_password", None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_token_validation() {
        let config = AuthConfig::default();
        let security_manager = SecurityManager::new(config, 1);
        security_manager.initialize().await.unwrap();

        let token = security_manager
            .authenticate("admin", "admin123", None)
            .await
            .unwrap();
        let token_id = token.token_id.clone();

        // Test valid token
        let validated = security_manager.validate_token(&token_id).await.unwrap();
        assert_eq!(validated.username, "admin");

        // Test token revocation
        security_manager.revoke_token(&token_id).await.unwrap();
        let result = security_manager.validate_token(&token_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_permission_checking() {
        let config = AuthConfig::default();
        let security_manager = SecurityManager::new(config, 1);
        security_manager.initialize().await.unwrap();

        let token = security_manager
            .authenticate("admin", "admin123", None)
            .await
            .unwrap();
        let token_id = token.token_id.clone();

        // Test permission check
        let has_admin = security_manager
            .check_permission(&token_id, Permission::SystemAdmin)
            .await
            .unwrap();
        assert!(has_admin);

        let has_invalid = security_manager
            .check_permission(&token_id, Permission::DataRead)
            .await
            .unwrap();
        assert!(has_invalid); // Admin should have all permissions
    }

    #[tokio::test]
    async fn test_failed_attempt_lockout() {
        let mut config = AuthConfig::default();
        config.max_failed_attempts = 2;
        config.lockout_duration_minutes = 1;

        let security_manager = SecurityManager::new(config, 1);
        security_manager.initialize().await.unwrap();

        // First failed attempt
        let _ = security_manager.authenticate("admin", "wrong", None).await;

        // Second failed attempt
        let _ = security_manager.authenticate("admin", "wrong", None).await;

        // Should be locked out now
        let result = security_manager
            .authenticate("admin", "admin123", None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("locked"));
    }

    #[test]
    fn test_role_manager() {
        let role_manager = RoleManager::new();

        let admin_permissions = role_manager.get_role_permissions("admin").unwrap();
        assert!(admin_permissions.contains(&Permission::SystemAdmin));

        let readonly_permissions = role_manager.get_role_permissions("readonly").unwrap();
        assert!(readonly_permissions.contains(&Permission::DataRead));
        assert!(!readonly_permissions.contains(&Permission::DataWrite));
    }
}
