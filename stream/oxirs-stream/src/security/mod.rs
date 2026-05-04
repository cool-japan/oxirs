//! # Advanced Security Framework
//!
//! Comprehensive security implementation for OxiRS Stream, providing authentication,
//! authorization, encryption, audit logging, and threat detection capabilities.

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

pub mod post_quantum;
pub mod traits;

pub use post_quantum::{PQCryptoMetrics, PostQuantumCryptoEngine, PostQuantumKeyPair};
pub use traits::{
    AuditFilter, AuditLogEntry, AuditLogger, AuthenticationProvider, AuthorizationProvider,
    Credentials, RateLimiter, SecurityContext, SecurityMetrics, ThreatAlert, ThreatContext,
    ThreatDetector,
};

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityConfig {
    /// Authentication configuration
    pub authentication: AuthConfig,
    /// Authorization configuration
    pub authorization: AuthzConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Post-quantum cryptography configuration
    pub post_quantum: PostQuantumConfig,
    /// Quantum-resistant certificates configuration
    pub quantum_resistant_certs: QuantumResistantCerts,
    /// Audit logging configuration
    pub audit: AuditConfig,
    /// Threat detection configuration
    pub threat_detection: ThreatDetectionConfig,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// Session management configuration
    pub session: SessionConfig,
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
    TOTP,     // Time-based One-Time Password
    SMS,      // SMS-based codes
    Email,    // Email-based codes
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
        role_permissions.insert(
            "viewer".to_string(),
            vec![Permission::Read, Permission::Query],
        );
        role_permissions.insert(
            "user".to_string(),
            vec![
                Permission::Read,
                Permission::Write,
                Permission::Stream,
                Permission::Query,
            ],
        );
        role_permissions.insert(
            "admin".to_string(),
            vec![
                Permission::Read,
                Permission::Write,
                Permission::Delete,
                Permission::Admin,
                Permission::Execute,
                Permission::Stream,
                Permission::Query,
                Permission::Configure,
            ],
        );

        let mut role_hierarchy = HashMap::new();
        role_hierarchy.insert(
            "admin".to_string(),
            vec!["user".to_string(), "viewer".to_string()],
        );
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
    pub policy_files: Vec<String>,
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self {
            language: PolicyLanguage::OPA,
            cache_policies: true,
            policy_files: vec!["/etc/oxirs/policies/".to_string()],
        }
    }
}

/// Policy languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyLanguage {
    OPA,    // Open Policy Agent (Rego)
    XACML,  // eXtensible Access Control Markup Language
    Cedar,  // Amazon Cedar
    Custom, // Custom policy language
}

/// Attribute definition for ABAC
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
    Float,
    List,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EncryptionConfig {
    /// Data at rest encryption
    pub at_rest: EncryptionAtRest,
    /// Data in transit encryption
    pub in_transit: EncryptionInTransit,
    /// Field-level encryption
    pub field_level: FieldLevelEncryption,
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
    // Classical symmetric encryption
    AES256GCM,
    AES256CBC,
    ChaCha20Poly1305,

    // Post-quantum key encapsulation mechanisms (KEMs)
    Kyber512,
    Kyber768,
    Kyber1024,

    // Lattice-based encryption
    NewHope1024,
    FrodoKEM640,
    FrodoKEM976,
    FrodoKEM1344,

    // Multivariate-based encryption
    Rainbow1,
    Rainbow3,
    Rainbow5,

    // Hash-based signatures with encryption
    SphincsPlus128s,
    SphincsPlus256s,

    // Isogeny-based encryption
    SikeP434,
    SikeP503,
    SikeP751,

    // Code-based encryption
    McEliece348864,
    McEliece460896,
    McEliece6688128,

    // Hybrid classical-quantum
    HybridAesKyber768,
    HybridChaCha20NewHope,
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
    AwsKms,
    AzureKeyVault,
    HashiCorpVault,
}

/// Key derivation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivation {
    // Classical key derivation
    PBKDF2,
    Scrypt,
    Argon2,

    // Post-quantum key derivation
    LatticeBasedKDF,
    HashBasedKDF,
    CodeBasedKDF,

    // Quantum-resistant hybrid approaches
    HybridArgon2Lattice,
    HybridScryptHash,
}

/// Post-quantum signature algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostQuantumSignature {
    // Lattice-based signatures
    Dilithium2,
    Dilithium3,
    Dilithium5,

    // Hash-based signatures
    SphincsPlusSha2128s,
    SphincsPlusSha2128f,
    SphincsPlusSha2192s,
    SphincsPlusSha2192f,
    SphincsPlusSha2256s,
    SphincsPlusSha2256f,
    SphincsPlusShake128s,
    SphincsPlusShake128f,
    SphincsPlusShake192s,
    SphincsPlusShake192f,
    SphincsPlusShake256s,
    SphincsPlusShake256f,

    // Multivariate signatures
    RainbowIClassic,
    RainbowICircumzenithal,
    RainbowICompressed,
    RainbowIiiClassic,
    RainbowIiiCircumzenithal,
    RainbowIiiCompressed,
    RainbowVClassic,
    RainbowVCircumzenithal,
    RainbowVCompressed,

    // Falcon signatures
    Falcon512,
    Falcon1024,

    // PICNIC signatures
    PicnicL1Fs,
    PicnicL1Ur,
    PicnicL3Fs,
    PicnicL3Ur,
    PicnicL5Fs,
    PicnicL5Ur,
}

/// Post-quantum cryptography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostQuantumConfig {
    pub enabled: bool,
    pub primary_kem: Option<EncryptionAlgorithm>,
    pub signature_algorithm: Option<PostQuantumSignature>,
    pub hybrid_mode: bool,
    pub classical_fallback: bool,
    pub quantum_security_level: QuantumSecurityLevel,
    pub key_size_preferences: KeySizePreferences,
}

impl Default for PostQuantumConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            primary_kem: Some(EncryptionAlgorithm::Kyber768),
            signature_algorithm: Some(PostQuantumSignature::Dilithium3),
            hybrid_mode: true,
            classical_fallback: true,
            quantum_security_level: QuantumSecurityLevel::Level3,
            key_size_preferences: KeySizePreferences::default(),
        }
    }
}

/// Quantum security levels (NIST standardization levels)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumSecurityLevel {
    Level1, // Equivalent to AES-128
    Level2, // Equivalent to SHA-256
    Level3, // Equivalent to AES-192
    Level4, // Equivalent to SHA-384
    Level5, // Equivalent to AES-256
}

/// Key size preferences for post-quantum algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySizePreferences {
    pub prefer_smaller_keys: bool,
    pub prefer_faster_signing: bool,
    pub prefer_faster_verification: bool,
    pub max_signature_size_kb: Option<u32>,
    pub max_public_key_size_kb: Option<u32>,
}

impl Default for KeySizePreferences {
    fn default() -> Self {
        Self {
            prefer_smaller_keys: true,
            prefer_faster_signing: false,
            prefer_faster_verification: true,
            max_signature_size_kb: Some(50),
            max_public_key_size_kb: Some(10),
        }
    }
}

/// Quantum-resistant certificate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResistantCerts {
    pub enabled: bool,
    pub use_hybrid_certificates: bool,
    pub pq_signature_algorithm: PostQuantumSignature,
    pub classical_signature_fallback: bool,
    pub certificate_chain_validation: PQCertValidation,
}

impl Default for QuantumResistantCerts {
    fn default() -> Self {
        Self {
            enabled: false,
            use_hybrid_certificates: true,
            pq_signature_algorithm: PostQuantumSignature::Dilithium3,
            classical_signature_fallback: true,
            certificate_chain_validation: PQCertValidation::default(),
        }
    }
}

/// Post-quantum certificate validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCertValidation {
    pub require_pq_signatures: bool,
    pub allow_mixed_chain: bool,
    pub minimum_security_level: QuantumSecurityLevel,
    pub validate_quantum_resistance: bool,
}

impl Default for PQCertValidation {
    fn default() -> Self {
        Self {
            require_pq_signatures: false,
            allow_mixed_chain: true,
            minimum_security_level: QuantumSecurityLevel::Level3,
            validate_quantum_resistance: true,
        }
    }
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
    CEF,  // Common Event Format
    LEEF, // Log Event Extended Format
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
            self.rate_limiter
                .check_limit(&credentials.identifier())
                .await?;
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
            let _ = self
                .threat_detector
                .detect_threat(&ThreatContext {
                    event_type: "authentication_failure".to_string(),
                    user_id: credentials.user_id().map(|s| s.to_string()),
                    ip_address: credentials.ip_address().map(|s| s.to_string()),
                    timestamp: Utc::now(),
                    details: HashMap::new(),
                })
                .await;
        }

        result
    }

    /// Authorize a request
    pub async fn authorize(
        &self,
        context: &SecurityContext,
        resource: &str,
        action: &Permission,
    ) -> Result<bool> {
        let start_time = Instant::now();

        let result = self
            .authz_provider
            .authorize(context, resource, action)
            .await;
        let success = result.as_ref().map(|&b| b).unwrap_or(false);

        // Log audit event
        let audit_event = AuditLogEntry {
            event_type: AuditEvent::Authorization,
            timestamp: Utc::now(),
            user_id: context.user_id.clone(),
            ip_address: context.ip_address.clone(),
            success,
            details: format!("Authorization check for {action} on {resource}"),
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
    pub async fn log_data_access(
        &self,
        context: &SecurityContext,
        resource: &str,
        operation: &str,
    ) -> Result<()> {
        let audit_event = AuditLogEntry {
            event_type: AuditEvent::DataAccess,
            timestamp: Utc::now(),
            user_id: context.user_id.clone(),
            ip_address: context.ip_address.clone(),
            success: true,
            details: format!("Data access: {operation} on {resource}"),
        };

        self.audit_logger.log(audit_event).await
    }

    /// Get security metrics
    pub async fn get_metrics(&self) -> SecurityMetrics {
        self.metrics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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

    // ------------------------------------------------------------------
    // Post-quantum cryptography tests
    // ------------------------------------------------------------------

    /// Without the `post-quantum` feature the engine must return a descriptive
    /// error rather than fake zero bytes.
    #[cfg(not(feature = "post-quantum"))]
    #[tokio::test]
    async fn test_pq_dilithium_requires_feature_flag() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);
        let result = engine
            .generate_keypair(&PostQuantumSignature::Dilithium3)
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("post-quantum"),
            "error should mention the feature flag, got: {msg}"
        );
    }

    #[cfg(not(feature = "post-quantum"))]
    #[tokio::test]
    async fn test_pq_kyber_encapsulate_requires_feature_flag() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);
        let dummy_pk = vec![0u8; 32];
        let result = engine
            .encapsulate(&dummy_pk, &EncryptionAlgorithm::Kyber768)
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("post-quantum"),
            "error should mention the feature flag, got: {msg}"
        );
    }

    /// With the `post-quantum` feature a Dilithium3 round-trip must succeed.
    #[cfg(feature = "post-quantum")]
    #[tokio::test]
    async fn test_pq_dilithium3_round_trip() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);

        let keypair = engine
            .generate_keypair(&PostQuantumSignature::Dilithium3)
            .await
            .expect("Dilithium3 key generation failed");

        let data = b"OxiRS post-quantum signing test vector";
        let sig_bytes = engine
            .sign(&keypair.id, data)
            .await
            .expect("Dilithium3 signing failed");

        let valid = engine
            .verify(
                &keypair.public_key,
                data,
                &sig_bytes,
                &PostQuantumSignature::Dilithium3,
            )
            .await
            .expect("Dilithium3 verification failed");
        assert!(valid, "valid Dilithium3 signature must verify as true");

        let tampered = b"OxiRS post-quantum signing test vector TAMPERED";
        let invalid = engine
            .verify(
                &keypair.public_key,
                tampered,
                &sig_bytes,
                &PostQuantumSignature::Dilithium3,
            )
            .await
            .expect("Dilithium3 tampered-data verification call failed");
        assert!(
            !invalid,
            "Dilithium3 signature over tampered data must not verify"
        );
    }

    /// With the `post-quantum` feature a Kyber768 KEM round-trip must succeed.
    #[cfg(feature = "post-quantum")]
    #[tokio::test]
    async fn test_pq_kyber768_kem_round_trip() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);

        // Generate a fresh Kyber768 key pair for KEM
        use pqcrypto_traits::kem::{PublicKey as _, SecretKey as _};
        let (pk, sk) = pqcrypto_kyber::kyber768::keypair();
        let pk_bytes = pk.as_bytes().to_vec();
        let sk_bytes = sk.as_bytes().to_vec();

        // Encapsulate
        let (ct, ss_enc) = engine
            .encapsulate(&pk_bytes, &EncryptionAlgorithm::Kyber768)
            .await
            .expect("Kyber768 encapsulation failed");

        // Decapsulate
        let ss_dec = engine
            .decapsulate(&sk_bytes, &ct, &EncryptionAlgorithm::Kyber768)
            .await
            .expect("Kyber768 decapsulation failed");

        assert_eq!(
            ss_enc, ss_dec,
            "encapsulated and decapsulated shared secrets must match"
        );
        assert_eq!(ss_enc.len(), 32, "Kyber768 shared secret must be 32 bytes");
    }

    /// With the `post-quantum` feature, SPHINCS+ SHA2-256s round-trip.
    #[cfg(feature = "post-quantum")]
    #[tokio::test]
    async fn test_pq_sphincsplus_sha2_256s_round_trip() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);

        let keypair = engine
            .generate_keypair(&PostQuantumSignature::SphincsPlusSha2256s)
            .await
            .expect("SPHINCS+ SHA2-256s key generation failed");

        let data = b"SPHINCS+ signing test vector";
        let sig_bytes = engine
            .sign(&keypair.id, data)
            .await
            .expect("SPHINCS+ signing failed");

        let valid = engine
            .verify(
                &keypair.public_key,
                data,
                &sig_bytes,
                &PostQuantumSignature::SphincsPlusSha2256s,
            )
            .await
            .expect("SPHINCS+ verification failed");
        assert!(valid, "valid SPHINCS+ signature must verify");
    }

    /// Without the `post-quantum` feature Falcon must return a descriptive error.
    #[cfg(not(feature = "post-quantum"))]
    #[tokio::test]
    async fn test_pq_falcon_requires_feature_flag() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);
        let result = engine
            .generate_keypair(&PostQuantumSignature::Falcon1024)
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("post-quantum"),
            "error should mention the feature flag, got: {msg}"
        );
    }

    /// With the `post-quantum` feature a Falcon512 round-trip must succeed.
    #[cfg(feature = "post-quantum")]
    #[tokio::test]
    async fn test_pq_falcon512_round_trip() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);

        let keypair = engine
            .generate_keypair(&PostQuantumSignature::Falcon512)
            .await
            .expect("Falcon512 key generation failed");

        let data = b"OxiRS Falcon512 signing test vector";

        let sig_bytes = engine
            .sign(&keypair.id, data)
            .await
            .expect("Falcon512 signing failed");
        assert!(
            !sig_bytes.is_empty(),
            "Falcon512 signature must not be empty"
        );

        // Correct data → valid
        let valid = engine
            .verify(
                &keypair.public_key,
                data,
                &sig_bytes,
                &PostQuantumSignature::Falcon512,
            )
            .await
            .expect("Falcon512 verification failed");
        assert!(valid, "valid Falcon512 signature must verify as true");

        // Tampered data → invalid
        let tampered = b"OxiRS Falcon512 signing test vector TAMPERED";
        let invalid = engine
            .verify(
                &keypair.public_key,
                tampered,
                &sig_bytes,
                &PostQuantumSignature::Falcon512,
            )
            .await
            .expect("Falcon512 tampered-data verification call failed");
        assert!(
            !invalid,
            "Falcon512 signature over tampered data must not verify"
        );
    }

    /// With the `post-quantum` feature a Falcon1024 round-trip must succeed.
    #[cfg(feature = "post-quantum")]
    #[tokio::test]
    async fn test_pq_falcon1024_round_trip() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);

        let keypair = engine
            .generate_keypair(&PostQuantumSignature::Falcon1024)
            .await
            .expect("Falcon1024 key generation failed");

        let data = b"OxiRS Falcon1024 signing test vector";

        let sig_bytes = engine
            .sign(&keypair.id, data)
            .await
            .expect("Falcon1024 signing failed");
        assert!(
            !sig_bytes.is_empty(),
            "Falcon1024 signature must not be empty"
        );

        // Correct data → valid
        let valid = engine
            .verify(
                &keypair.public_key,
                data,
                &sig_bytes,
                &PostQuantumSignature::Falcon1024,
            )
            .await
            .expect("Falcon1024 verification failed");
        assert!(valid, "valid Falcon1024 signature must verify as true");

        // Tampered data → invalid
        let tampered = b"OxiRS Falcon1024 signing test vector TAMPERED";
        let invalid = engine
            .verify(
                &keypair.public_key,
                tampered,
                &sig_bytes,
                &PostQuantumSignature::Falcon1024,
            )
            .await
            .expect("Falcon1024 tampered-data verification call failed");
        assert!(
            !invalid,
            "Falcon1024 signature over tampered data must not verify"
        );
    }

    /// NewHope and FrodoKEM are always unavailable (no crate exists).
    #[tokio::test]
    async fn test_pq_newhope_not_yet_supported() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);
        let dummy_pk = vec![0u8; 32];
        let result = engine
            .encapsulate(&dummy_pk, &EncryptionAlgorithm::NewHope1024)
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("NewHope"),
            "error should mention NewHope, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_pq_frodokem_not_yet_supported() {
        let config = PostQuantumConfig::default();
        let engine = PostQuantumCryptoEngine::new(config);
        let dummy_pk = vec![0u8; 32];
        let result = engine
            .encapsulate(&dummy_pk, &EncryptionAlgorithm::FrodoKEM976)
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("FrodoKEM"),
            "error should mention FrodoKEM, got: {msg}"
        );
    }
}
