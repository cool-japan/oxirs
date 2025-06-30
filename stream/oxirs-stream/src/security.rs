//! # Advanced Security Framework
//!
//! Comprehensive security implementation for OxiRS Stream, providing authentication,
//! authorization, encryption, audit logging, and threat detection capabilities.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub authentication: AuthConfig,
    /// Authorization configuration
    pub authorization: AuthzConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Audit logging configuration
    pub audit: AuditConfig,
    /// Threat detection configuration
    pub threat_detection: ThreatDetectionConfig,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// Session management configuration
    pub session: SessionConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication: AuthConfig::default(),
            authorization: AuthzConfig::default(),
            encryption: EncryptionConfig::default(),
            audit: AuditConfig::default(),
            threat_detection: ThreatDetectionConfig::default(),
            rate_limiting: RateLimitConfig::default(),
            session: SessionConfig::default(),
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enabled authentication methods
    pub methods: Vec<AuthMethod>,
    /// Multi-factor authentication settings
    pub mfa: MfaConfig,
    /// Token settings
    pub token: TokenConfig,
    /// Password policy
    pub password_policy: PasswordPolicy,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            methods: vec![AuthMethod::ApiKey, AuthMethod::JWT],
            mfa: MfaConfig::default(),
            token: TokenConfig::default(),
            password_policy: PasswordPolicy::default(),
        }
    }
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthMethod {
    ApiKey,
    JWT,
    OAuth2,
    SAML,
    Certificate,
    Basic,
}

/// Multi-factor authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaConfig {
    pub enabled: bool,
    pub required_for_admin: bool,
    pub methods: Vec<MfaMethod>,
    pub backup_codes: bool,
}

impl Default for MfaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            required_for_admin: true,
            methods: vec![MfaMethod::TOTP],
            backup_codes: true,
        }
    }
}

/// Multi-factor authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MfaMethod {
    TOTP,    // Time-based One-Time Password
    SMS,     // SMS-based codes
    Email,   // Email-based codes
    Hardware, // Hardware keys (FIDO2/WebAuthn)
}

/// Token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub jwt_secret: String,
    pub access_token_ttl: ChronoDuration,
    pub refresh_token_ttl: ChronoDuration,
    pub issuer: String,
    pub audience: String,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            jwt_secret: "change-this-secret".to_string(),
            access_token_ttl: ChronoDuration::hours(1),
            refresh_token_ttl: ChronoDuration::days(30),
            issuer: "oxirs-stream".to_string(),
            audience: "oxirs-api".to_string(),
        }
    }
}

/// Password policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicy {
    pub min_length: usize,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_numbers: bool,
    pub require_symbols: bool,
    pub max_age_days: Option<u32>,
    pub history_count: usize,
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 12,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_symbols: true,
            max_age_days: Some(90),
            history_count: 5,
        }
    }
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthzConfig {
    /// Authorization model
    pub model: AuthzModel,
    /// Default permissions
    pub default_permissions: Vec<Permission>,
    /// Role-based access control
    pub rbac: RbacConfig,
    /// Attribute-based access control
    pub abac: AbacConfig,
}

impl Default for AuthzConfig {
    fn default() -> Self {
        Self {
            model: AuthzModel::RBAC,
            default_permissions: vec![Permission::Read],
            rbac: RbacConfig::default(),
            abac: AbacConfig::default(),
        }
    }
}

/// Authorization models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthzModel {
    RBAC, // Role-Based Access Control
    ABAC, // Attribute-Based Access Control
    MAC,  // Mandatory Access Control
    DAC,  // Discretionary Access Control
}

/// Permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Permission {
    Read,
    Write,
    Delete,
    Admin,
    Execute,
    Stream,
    Query,
    Configure,
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Permission::Read => write!(f, "read"),
            Permission::Write => write!(f, "write"),
            Permission::Delete => write!(f, "delete"),
            Permission::Admin => write!(f, "admin"),
            Permission::Execute => write!(f, "execute"),
            Permission::Stream => write!(f, "stream"),
            Permission::Query => write!(f, "query"),
            Permission::Configure => write!(f, "configure"),
        }
    }
}

/// Role-based access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    pub enabled: bool,
    pub default_role: String,
    pub role_hierarchy: HashMap<String, Vec<String>>,
    pub role_permissions: HashMap<String, Vec<Permission>>,
}

impl Default for RbacConfig {
    fn default() -> Self {
        let mut role_permissions = HashMap::new();
        role_permissions.insert("viewer".to_string(), vec![Permission::Read, Permission::Query]);
        role_permissions.insert("user".to_string(), vec![Permission::Read, Permission::Write, Permission::Stream, Permission::Query]);
        role_permissions.insert("admin".to_string(), vec![Permission::Read, Permission::Write, Permission::Delete, Permission::Admin, Permission::Execute, Permission::Stream, Permission::Query, Permission::Configure]);
        
        let mut role_hierarchy = HashMap::new();
        role_hierarchy.insert("admin".to_string(), vec!["user".to_string(), "viewer".to_string()]);
        role_hierarchy.insert("user".to_string(), vec!["viewer".to_string()]);
        
        Self {
            enabled: true,
            default_role: "viewer".to_string(),
            role_hierarchy,
            role_permissions,
        }
    }
}

/// Attribute-based access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacConfig {
    pub enabled: bool,
    pub policy_engine: PolicyEngine,
    pub attributes: Vec<AttributeDefinition>,
}

impl Default for AbacConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy_engine: PolicyEngine::default(),
            attributes: vec![
                AttributeDefinition {
                    name: "department".to_string(),
                    attribute_type: AttributeType::String,
                    required: false,
                },
                AttributeDefinition {
                    name: "security_level".to_string(),
                    attribute_type: AttributeType::Integer,
                    required: true,
                },
            ],
        }
    }
}

/// Policy engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEngine {
    pub language: PolicyLanguage,
    pub cache_policies: bool,
    pub cache_ttl: ChronoDuration,
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self {
            language: PolicyLanguage::OPA,
            cache_policies: true,
            cache_ttl: ChronoDuration::minutes(30),
        }
    }
}

/// Policy languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyLanguage {
    OPA,    // Open Policy Agent
    Cedar,  // Amazon Cedar
    Custom, // Custom policy language
}

/// Attribute definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeDefinition {
    pub name: String,
    pub attribute_type: AttributeType,
    pub required: bool,
}

/// Attribute types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeType {
    String,
    Integer,
    Boolean,
    DateTime,
    Array,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Data at rest encryption
    pub at_rest: EncryptionAtRest,
    /// Data in transit encryption
    pub in_transit: EncryptionInTransit,
    /// Field-level encryption
    pub field_level: FieldLevelEncryption,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            at_rest: EncryptionAtRest::default(),
            in_transit: EncryptionInTransit::default(),
            field_level: FieldLevelEncryption::default(),
        }
    }
}

/// Data at rest encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAtRest {
    pub enabled: bool,
    pub algorithm: EncryptionAlgorithm,
    pub key_management: KeyManagement,
}

impl Default for EncryptionAtRest {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: EncryptionAlgorithm::AES256GCM,
            key_management: KeyManagement::default(),
        }
    }
}

/// Data in transit encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInTransit {
    pub enabled: bool,
    pub tls_version: TlsVersion,
    pub cipher_suites: Vec<String>,
    pub certificate_validation: bool,
}

impl Default for EncryptionInTransit {
    fn default() -> Self {
        Self {
            enabled: true,
            tls_version: TlsVersion::V1_3,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            certificate_validation: true,
        }
    }
}

/// Field-level encryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLevelEncryption {
    pub enabled: bool,
    pub fields: Vec<String>,
    pub algorithm: EncryptionAlgorithm,
}

impl Default for FieldLevelEncryption {
    fn default() -> Self {
        Self {
            enabled: false,
            fields: vec!["password".to_string(), "ssn".to_string()],
            algorithm: EncryptionAlgorithm::AES256GCM,
        }
    }
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256GCM,
    AES256CBC,
    ChaCha20Poly1305,
}

/// TLS versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TlsVersion {
    V1_2,
    V1_3,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagement {
    pub provider: KeyProvider,
    pub rotation_interval: ChronoDuration,
    pub key_derivation: KeyDerivation,
}

impl Default for KeyManagement {
    fn default() -> Self {
        Self {
            provider: KeyProvider::Local,
            rotation_interval: ChronoDuration::days(90),
            key_derivation: KeyDerivation::PBKDF2,
        }
    }
}

/// Key providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyProvider {
    Local,
    HSM,
    AWS_KMS,
    Azure_KeyVault,
    HashiCorp_Vault,
}

/// Key derivation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivation {
    PBKDF2,
    Scrypt,
    Argon2,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub events: Vec<AuditEvent>,
    pub retention_days: u32,
    pub log_format: AuditLogFormat,
    pub output: AuditOutput,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            events: vec![
                AuditEvent::Authentication,
                AuditEvent::Authorization,
                AuditEvent::DataAccess,
                AuditEvent::ConfigChange,
            ],
            retention_days: 365,
            log_format: AuditLogFormat::JSON,
            output: AuditOutput::File("/var/log/oxirs/audit.log".to_string()),
        }
    }
}

/// Audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    Authentication,
    Authorization,
    DataAccess,
    ConfigChange,
    UserManagement,
    SecurityAlert,
}

/// Audit log formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLogFormat {
    JSON,
    CEF,   // Common Event Format
    LEEF,  // Log Event Extended Format
    Syslog,
}

/// Audit output destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutput {
    File(String),
    Syslog,
    Database,
    SIEM(String), // SIEM endpoint
}

/// Threat detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionConfig {
    pub enabled: bool,
    pub rules: Vec<ThreatRule>,
    pub anomaly_detection: AnomalyDetectionConfig,
    pub response: ThreatResponseConfig,
}

impl Default for ThreatDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![
                ThreatRule {
                    name: "Multiple Failed Logins".to_string(),
                    description: "Detect multiple failed login attempts".to_string(),
                    condition: "failed_logins > 5 in 5m".to_string(),
                    severity: ThreatSeverity::High,
                    enabled: true,
                },
                ThreatRule {
                    name: "Unusual Access Pattern".to_string(),
                    description: "Detect unusual access patterns".to_string(),
                    condition: "requests > 1000 in 1m".to_string(),
                    severity: ThreatSeverity::Medium,
                    enabled: true,
                },
            ],
            anomaly_detection: AnomalyDetectionConfig::default(),
            response: ThreatResponseConfig::default(),
        }
    }
}

/// Threat detection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatRule {
    pub name: String,
    pub description: String,
    pub condition: String,
    pub severity: ThreatSeverity,
    pub enabled: bool,
}

/// Threat severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    pub enabled: bool,
    pub algorithms: Vec<AnomalyAlgorithm>,
    pub sensitivity: f64,
    pub learning_period_days: u32,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![
                AnomalyAlgorithm::IsolationForest,
                AnomalyAlgorithm::StatisticalOutlier,
            ],
            sensitivity: 0.8,
            learning_period_days: 30,
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    IsolationForest,
    StatisticalOutlier,
    LSTM,
    KMeans,
}

/// Threat response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatResponseConfig {
    pub auto_response: bool,
    pub actions: Vec<ResponseAction>,
    pub escalation: EscalationConfig,
}

impl Default for ThreatResponseConfig {
    fn default() -> Self {
        Self {
            auto_response: true,
            actions: vec![
                ResponseAction::Block,
                ResponseAction::RateLimit,
                ResponseAction::Alert,
            ],
            escalation: EscalationConfig::default(),
        }
    }
}

/// Response actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseAction {
    Block,
    RateLimit,
    Alert,
    Quarantine,
    Notify,
}

/// Escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    pub enabled: bool,
    pub thresholds: HashMap<ThreatSeverity, u32>,
    pub notifications: Vec<String>, // email addresses or webhook URLs
}

impl Default for EscalationConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(ThreatSeverity::Low, 10);
        thresholds.insert(ThreatSeverity::Medium, 5);
        thresholds.insert(ThreatSeverity::High, 1);
        thresholds.insert(ThreatSeverity::Critical, 1);
        
        Self {
            enabled: true,
            thresholds,
            notifications: vec!["security@example.com".to_string()],
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub global_limit: RateLimit,
    pub per_user_limit: RateLimit,
    pub per_ip_limit: RateLimit,
    pub burst_limit: RateLimit,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            global_limit: RateLimit {
                requests: 10000,
                window: ChronoDuration::minutes(1),
            },
            per_user_limit: RateLimit {
                requests: 100,
                window: ChronoDuration::minutes(1),
            },
            per_ip_limit: RateLimit {
                requests: 1000,
                window: ChronoDuration::minutes(1),
            },
            burst_limit: RateLimit {
                requests: 50,
                window: ChronoDuration::seconds(10),
            },
        }
    }
}

/// Rate limit definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests: u32,
    pub window: ChronoDuration,
}

/// Session management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub timeout: ChronoDuration,
    pub max_concurrent: u32,
    pub secure_cookies: bool,
    pub same_site: SameSite,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: ChronoDuration::hours(8),
            max_concurrent: 5,
            secure_cookies: true,
            same_site: SameSite::Strict,
        }
    }
}

/// SameSite cookie attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SameSite {
    Strict,
    Lax,
    None,
}

/// Security context for requests
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub roles: HashSet<String>,
    pub permissions: HashSet<Permission>,
    pub attributes: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub authentication_method: Option<AuthMethod>,
    pub authenticated_at: Option<DateTime<Utc>>,
}

impl SecurityContext {
    /// Check if user has permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }
    
    /// Check if user has role
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(role)
    }
    
    /// Check if user has any of the specified roles
    pub fn has_any_role(&self, roles: &[String]) -> bool {
        roles.iter().any(|role| self.roles.contains(role))
    }
    
    /// Get attribute value
    pub fn get_attribute(&self, name: &str) -> Option<&String> {
        self.attributes.get(name)
    }
}

/// Security manager
pub struct SecurityManager {
    config: SecurityConfig,
    auth_provider: Arc<dyn AuthenticationProvider>,
    authz_provider: Arc<dyn AuthorizationProvider>,
    audit_logger: Arc<dyn AuditLogger>,
    threat_detector: Arc<dyn ThreatDetector>,
    rate_limiter: Arc<dyn RateLimiter>,
    metrics: Arc<RwLock<SecurityMetrics>>,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new(
        config: SecurityConfig,
        auth_provider: Arc<dyn AuthenticationProvider>,
        authz_provider: Arc<dyn AuthorizationProvider>,
        audit_logger: Arc<dyn AuditLogger>,
        threat_detector: Arc<dyn ThreatDetector>,
        rate_limiter: Arc<dyn RateLimiter>,
    ) -> Self {
        Self {
            config,
            auth_provider,
            authz_provider,
            audit_logger,
            threat_detector,
            rate_limiter,
            metrics: Arc::new(RwLock::new(SecurityMetrics::default())),
        }
    }
    
    /// Authenticate a request
    pub async fn authenticate(&self, credentials: &Credentials) -> Result<SecurityContext> {
        let start_time = Instant::now();
        
        // Check rate limits first
        if self.config.rate_limiting.enabled {
            self.rate_limiter.check_limit(&credentials.identifier()).await?;
        }
        
        // Perform authentication
        let result = self.auth_provider.authenticate(credentials).await;
        
        // Log audit event
        let audit_event = AuditLogEntry {
            event_type: AuditEvent::Authentication,
            timestamp: Utc::now(),
            user_id: credentials.user_id().map(|s| s.to_string()),
            ip_address: credentials.ip_address().map(|s| s.to_string()),
            success: result.is_ok(),
            details: format!("Authentication attempt for {}", credentials.identifier()),
        };
        
        let _ = self.audit_logger.log(audit_event).await;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.authentication_attempts += 1;
            if result.is_ok() {
                metrics.authentication_successes += 1;
            } else {
                metrics.authentication_failures += 1;
            }
            metrics.authentication_latency_ms = start_time.elapsed().as_millis() as f64;
        }
        
        // Check for threats
        if result.is_err() {
            let _ = self.threat_detector.detect_threat(&ThreatContext {
                event_type: "authentication_failure".to_string(),
                user_id: credentials.user_id().map(|s| s.to_string()),
                ip_address: credentials.ip_address().map(|s| s.to_string()),
                timestamp: Utc::now(),
                details: HashMap::new(),
            }).await;
        }
        
        result
    }
    
    /// Authorize a request
    pub async fn authorize(&self, context: &SecurityContext, resource: &str, action: &Permission) -> Result<bool> {
        let start_time = Instant::now();
        
        let result = self.authz_provider.authorize(context, resource, action).await;
        let success = result.as_ref().map(|&b| b).unwrap_or(false);
        
        // Log audit event
        let audit_event = AuditLogEntry {
            event_type: AuditEvent::Authorization,
            timestamp: Utc::now(),
            user_id: context.user_id.clone(),
            ip_address: context.ip_address.clone(),
            success,
            details: format!("Authorization check for {} on {}", action, resource),
        };
        
        let _ = self.audit_logger.log(audit_event).await;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.authorization_checks += 1;
            if success {
                metrics.authorization_successes += 1;
            } else {
                metrics.authorization_failures += 1;
            }
            metrics.authorization_latency_ms = start_time.elapsed().as_millis() as f64;
        }
        
        result
    }
    
    /// Log data access
    pub async fn log_data_access(&self, context: &SecurityContext, resource: &str, operation: &str) -> Result<()> {
        let audit_event = AuditLogEntry {
            event_type: AuditEvent::DataAccess,
            timestamp: Utc::now(),
            user_id: context.user_id.clone(),
            ip_address: context.ip_address.clone(),
            success: true,
            details: format!("Data access: {} on {}", operation, resource),
        };
        
        self.audit_logger.log(audit_event).await
    }
    
    /// Get security metrics
    pub async fn get_metrics(&self) -> SecurityMetrics {
        self.metrics.read().await.clone()
    }
}

/// Authentication provider trait
#[async_trait::async_trait]
pub trait AuthenticationProvider: Send + Sync {
    async fn authenticate(&self, credentials: &Credentials) -> Result<SecurityContext>;
    async fn validate_token(&self, token: &str) -> Result<SecurityContext>;
    async fn refresh_token(&self, refresh_token: &str) -> Result<(String, String)>; // (access_token, refresh_token)
}

/// Authorization provider trait
#[async_trait::async_trait]
pub trait AuthorizationProvider: Send + Sync {
    async fn authorize(&self, context: &SecurityContext, resource: &str, action: &Permission) -> Result<bool>;
    async fn get_user_permissions(&self, user_id: &str) -> Result<HashSet<Permission>>;
    async fn get_user_roles(&self, user_id: &str) -> Result<HashSet<String>>;
}

/// Audit logger trait
#[async_trait::async_trait]
pub trait AuditLogger: Send + Sync {
    async fn log(&self, entry: AuditLogEntry) -> Result<()>;
    async fn query(&self, filter: AuditFilter) -> Result<Vec<AuditLogEntry>>;
}

/// Threat detector trait
#[async_trait::async_trait]
pub trait ThreatDetector: Send + Sync {
    async fn detect_threat(&self, context: &ThreatContext) -> Result<Option<ThreatAlert>>;
    async fn learn_pattern(&self, context: &ThreatContext) -> Result<()>;
}

/// Rate limiter trait
#[async_trait::async_trait]
pub trait RateLimiter: Send + Sync {
    async fn check_limit(&self, identifier: &str) -> Result<()>;
    async fn reset_limit(&self, identifier: &str) -> Result<()>;
}

/// Credentials for authentication
#[derive(Debug, Clone)]
pub enum Credentials {
    ApiKey { key: String, ip_address: Option<String> },
    JWT { token: String, ip_address: Option<String> },
    UserPassword { username: String, password: String, ip_address: Option<String> },
    Certificate { cert: Vec<u8>, ip_address: Option<String> },
}

impl Credentials {
    pub fn identifier(&self) -> String {
        match self {
            Credentials::ApiKey { key, .. } => format!("api_key:{}", key),
            Credentials::JWT { token, .. } => format!("jwt:{}", token),
            Credentials::UserPassword { username, .. } => format!("user:{}", username),
            Credentials::Certificate { .. } => "certificate".to_string(),
        }
    }
    
    pub fn user_id(&self) -> Option<&str> {
        match self {
            Credentials::UserPassword { username, .. } => Some(username),
            _ => None,
        }
    }
    
    pub fn ip_address(&self) -> Option<&str> {
        match self {
            Credentials::ApiKey { ip_address, .. } |
            Credentials::JWT { ip_address, .. } |
            Credentials::UserPassword { ip_address, .. } |
            Credentials::Certificate { ip_address, .. } => ip_address.as_deref(),
        }
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub event_type: AuditEvent,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub success: bool,
    pub details: String,
}

/// Audit filter for queries
#[derive(Debug, Clone)]
pub struct AuditFilter {
    pub event_types: Option<Vec<AuditEvent>>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub success: Option<bool>,
}

/// Threat context
#[derive(Debug, Clone)]
pub struct ThreatContext {
    pub event_type: String,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub details: HashMap<String, String>,
}

/// Threat alert
#[derive(Debug, Clone)]
pub struct ThreatAlert {
    pub id: Uuid,
    pub rule_name: String,
    pub severity: ThreatSeverity,
    pub description: String,
    pub context: ThreatContext,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

/// Security metrics
#[derive(Debug, Clone, Default)]
pub struct SecurityMetrics {
    pub authentication_attempts: u64,
    pub authentication_successes: u64,
    pub authentication_failures: u64,
    pub authorization_checks: u64,
    pub authorization_successes: u64,
    pub authorization_failures: u64,
    pub threat_alerts: u64,
    pub rate_limit_violations: u64,
    pub authentication_latency_ms: f64,
    pub authorization_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_config_defaults() {
        let config = SecurityConfig::default();
        assert!(config.authentication.methods.contains(&AuthMethod::ApiKey));
        assert!(config.authorization.rbac.enabled);
        assert!(config.encryption.at_rest.enabled);
        assert!(config.audit.enabled);
    }
    
    #[test]
    fn test_security_context_permissions() {
        let mut context = SecurityContext {
            user_id: Some("test_user".to_string()),
            session_id: None,
            roles: HashSet::new(),
            permissions: HashSet::new(),
            attributes: HashMap::new(),
            ip_address: None,
            user_agent: None,
            authentication_method: None,
            authenticated_at: None,
        };
        
        context.permissions.insert(Permission::Read);
        assert!(context.has_permission(&Permission::Read));
        assert!(!context.has_permission(&Permission::Write));
    }
    
    #[test]
    fn test_credentials_identifier() {
        let creds = Credentials::ApiKey {
            key: "test_key".to_string(),
            ip_address: None,
        };
        
        assert_eq!(creds.identifier(), "api_key:test_key");
    }
    
    #[test]
    fn test_rbac_config_default_roles() {
        let rbac = RbacConfig::default();
        assert!(rbac.role_permissions.contains_key("admin"));
        assert!(rbac.role_permissions.contains_key("user"));
        assert!(rbac.role_permissions.contains_key("viewer"));
        
        let admin_perms = rbac.role_permissions.get("admin").unwrap();
        assert!(admin_perms.contains(&Permission::Admin));
        assert!(admin_perms.contains(&Permission::Read));
        assert!(admin_perms.contains(&Permission::Write));
    }
}