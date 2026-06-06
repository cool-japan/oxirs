//! Security-domain configuration types: auth, identity, SAML, OAuth, JWT, LDAP, MFA, API keys,
//! certificates, ReBAC.
use crate::config::config_runtime::{CorsConfig, RateLimitConfig, SameSitePolicy, SessionConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use validator::{Validate, ValidationError};

/// Security configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct SecurityConfig {
    pub auth_required: bool,

    #[validate(nested)]
    pub users: HashMap<String, UserConfig>,

    #[validate(nested)]
    pub jwt: Option<JwtConfig>,

    #[validate(nested)]
    pub oauth: Option<OAuthConfig>,

    #[validate(nested)]
    pub ldap: Option<LdapConfig>,

    #[validate(nested)]
    pub rate_limiting: Option<RateLimitConfig>,

    pub cors: CorsConfig,

    pub session: SessionConfig,

    #[validate(nested)]
    pub authentication: AuthenticationConfig,

    #[validate(nested)]
    pub api_keys: Option<ApiKeyConfig>,

    #[validate(nested)]
    pub certificate: Option<CertificateConfig>,

    #[validate(nested)]
    pub saml: Option<SamlConfig>,

    #[validate(nested)]
    pub rebac: Option<RebacConfig>,

    #[validate(nested)]
    pub mfa: Option<MfaConfig>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct AuthenticationConfig {
    pub enabled: bool,
}

/// API Key configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyConfig {
    /// Enable API key authentication
    pub enabled: bool,

    /// Default key expiration in days
    #[validate(range(min = 1, max = 3650))] // 1 day to 10 years
    pub default_expiration_days: u32,

    /// Maximum number of keys per user
    #[validate(range(min = 1, max = 100))]
    pub max_keys_per_user: u32,

    /// Default rate limiting for API keys
    pub default_rate_limit: Option<ApiKeyRateLimit>,

    /// Enable usage analytics
    pub usage_analytics: bool,

    /// Storage backend configuration
    pub storage: ApiKeyStorageConfig,
}

/// API key rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyRateLimit {
    #[validate(range(min = 1))]
    pub requests_per_minute: u32,

    #[validate(range(min = 1))]
    pub requests_per_hour: u32,

    #[validate(range(min = 1))]
    pub requests_per_day: u32,

    #[validate(range(min = 1))]
    pub burst_limit: u32,
}

/// API key storage configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ApiKeyStorageConfig {
    /// Storage backend type
    pub backend: ApiKeyStorageBackend,

    /// Connection string or file path
    #[validate(length(min = 1))]
    pub connection: String,

    /// Encryption key for sensitive data
    #[validate(length(min = 32))]
    pub encryption_key: Option<String>,
}

/// API key storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyStorageBackend {
    Memory,
    File,
    Sqlite,
    Postgres,
    Redis,
}

/// Certificate authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CertificateConfig {
    /// Enable certificate authentication
    pub enabled: bool,

    /// Require client certificates for all connections
    pub require_client_cert: bool,

    /// Trust store configuration - paths to trusted CA certificates
    pub trust_store: Vec<PathBuf>,

    /// Certificate Revocation List (CRL) URLs or file paths
    pub crl_sources: Vec<String>,

    /// Enable CRL checking
    pub check_crl: bool,

    /// Enable OCSP checking
    pub check_ocsp: bool,

    /// Allow self-signed certificates (for development only)
    pub allow_self_signed: bool,

    /// Certificate to user mapping rules
    pub user_mapping: CertificateUserMapping,

    /// Maximum certificate chain length
    #[validate(range(min = 1, max = 10))]
    pub max_chain_length: u8,

    /// Certificate validation strictness
    pub validation_level: CertificateValidationLevel,

    /// Trusted issuer DN patterns for certificate validation
    /// Certificates from these issuers will be trusted without requiring CA certificates in trust store
    pub trusted_issuers: Option<Vec<String>>,
}

/// Certificate user mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CertificateUserMapping {
    /// How to extract username from certificate
    pub username_source: CertificateUsernameSource,

    /// Subject DN to username mapping rules
    pub dn_mapping_rules: Vec<DnMappingRule>,

    /// Default roles for certificate users
    pub default_roles: Vec<String>,

    /// OU to role mapping
    pub ou_role_mapping: HashMap<String, Vec<String>>,
}

/// Source for extracting username from certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateUsernameSource {
    /// Use Common Name (CN) from subject DN
    CommonName,
    /// Use entire subject DN
    SubjectDn,
    /// Use email from Subject Alternative Name
    EmailSan,
    /// Use custom regex pattern
    CustomPattern(String),
}

/// Subject DN to username mapping rule
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DnMappingRule {
    /// Regex pattern to match against subject DN
    #[validate(length(min = 1))]
    pub pattern: String,

    /// Replacement string (supports capture groups)
    #[validate(length(min = 1))]
    pub replacement: String,

    /// Roles to assign to users matching this pattern
    pub roles: Vec<String>,
}

/// Certificate validation strictness levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateValidationLevel {
    /// Strict validation - all checks must pass
    Strict,
    /// Moderate validation - allow some minor issues
    Moderate,
    /// Permissive validation - for development/testing
    Permissive,
}

/// SAML authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlConfig {
    /// Enable SAML authentication
    pub enabled: bool,

    /// Service Provider (SP) entity ID
    #[validate(length(min = 1))]
    pub sp_entity_id: String,

    /// SP X.509 certificate for signing
    pub sp_cert_path: Option<PathBuf>,

    /// SP private key for signing
    pub sp_key_path: Option<PathBuf>,

    /// Identity Provider (IdP) configuration
    pub idp: SamlIdpConfig,

    /// Assertion consumer service URL
    #[validate(length(min = 1))]
    pub acs_url: String,

    /// Single logout service URL
    pub slo_url: Option<String>,

    /// SAML attribute mappings
    pub attribute_mappings: SamlAttributeMappings,

    /// Session timeout in seconds
    #[validate(range(min = 300, max = 86400))]
    pub session_timeout_secs: u64,
}

/// SAML Identity Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlIdpConfig {
    /// IdP entity ID
    #[validate(length(min = 1))]
    pub entity_id: String,

    /// IdP SSO URL
    #[validate(length(min = 1))]
    pub sso_url: String,

    /// IdP SLO URL
    pub slo_url: Option<String>,

    /// IdP X.509 certificate for verification
    #[validate(custom(function = "validate_path"))]
    pub cert_path: PathBuf,

    /// IdP metadata URL (alternative to manual configuration)
    pub metadata_url: Option<String>,
}

/// SAML attribute mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SamlAttributeMappings {
    /// SAML attribute name for username
    #[validate(length(min = 1))]
    pub username_attribute: String,

    /// SAML attribute name for email
    pub email_attribute: Option<String>,

    /// SAML attribute name for full name
    pub name_attribute: Option<String>,

    /// SAML attribute name for groups/roles
    pub groups_attribute: Option<String>,

    /// Group to role mapping
    pub group_role_mapping: HashMap<String, Vec<String>>,
}

/// ReBAC (Relationship-Based Access Control) configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RebacConfig {
    /// Enable ReBAC authorization
    pub enabled: bool,

    /// Policy evaluation mode
    pub policy_mode: RebacPolicyMode,

    /// Storage backend for relationships
    pub storage: RebacStorageBackend,

    /// OpenFGA server configuration (if using OpenFGA backend)
    pub openfga: Option<OpenFgaConfig>,

    /// Initial relationship tuples to load on startup
    pub initial_relationships: Vec<RelationshipTupleConfig>,

    /// Enable audit logging for authorization decisions
    pub audit_enabled: bool,

    /// Cache authorization decisions for this many seconds
    #[validate(range(min = 0, max = 3600))]
    pub cache_ttl_secs: u64,
}

/// ReBAC policy evaluation modes
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RebacPolicyMode {
    /// Use only traditional RBAC
    RbacOnly,

    /// Use only ReBAC (relationship-based)
    RebacOnly,

    /// Try RBAC first, fall back to ReBAC (default)
    #[default]
    Combined,

    /// Require both RBAC and ReBAC to allow
    Both,
}

/// ReBAC storage backends
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RebacStorageBackend {
    /// In-memory storage (fast, not persistent)
    #[default]
    Memory,

    /// OpenFGA server (scalable, persistent)
    OpenFga,

    /// RDF-based storage using named graphs (future)
    Rdf,

    /// Database storage (future)
    Database,
}

/// OpenFGA server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OpenFgaConfig {
    /// OpenFGA API URL
    #[validate(length(min = 1))]
    pub api_url: String,

    /// Store ID
    #[validate(length(min = 1))]
    pub store_id: String,

    /// Authorization model ID (optional, uses latest if not specified)
    pub model_id: Option<String>,

    /// API token for authentication
    pub api_token: Option<String>,

    /// Enable TLS for OpenFGA connection
    pub tls_enabled: bool,

    /// Connection timeout in seconds
    #[validate(range(min = 1, max = 300))]
    pub timeout_secs: u64,
}

/// Relationship tuple configuration for initial loading
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RelationshipTupleConfig {
    /// Subject (e.g., "user:alice", "organization:engineering")
    #[validate(length(min = 1))]
    pub subject: String,

    /// Relation (e.g., "owner", "can_read", "member")
    #[validate(length(min = 1))]
    pub relation: String,

    /// Object/resource (e.g., "dataset:public", "graph:`http://example.org/g1`")
    #[validate(length(min = 1))]
    pub object: String,

    /// Optional condition
    pub condition: Option<RelationshipConditionConfig>,
}

/// Relationship condition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RelationshipConditionConfig {
    /// Time-based condition
    TimeWindow {
        not_before: Option<String>, // ISO 8601 datetime
        not_after: Option<String>,  // ISO 8601 datetime
    },

    /// IP address condition
    IpAddress { allowed_ips: Vec<String> },

    /// Custom attribute-based condition
    Attribute { key: String, value: String },
}

/// Multi-Factor Authentication (MFA) configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MfaConfig {
    /// Enable MFA globally
    pub enabled: bool,

    /// Require MFA for all users (if false, users can opt-in)
    pub required: bool,

    /// Available MFA methods
    pub methods: MfaMethods,

    /// TOTP configuration
    #[validate(nested)]
    pub totp: Option<TotpConfig>,

    /// SMS configuration
    #[validate(nested)]
    pub sms: Option<SmsConfig>,

    /// Email configuration
    #[validate(nested)]
    pub email: Option<EmailConfig>,

    /// WebAuthn configuration
    #[validate(nested)]
    pub webauthn: Option<WebAuthnConfig>,

    /// Backup codes configuration
    pub backup_codes: BackupCodesConfig,

    /// MFA session duration in seconds (how long after MFA verification before re-prompt)
    #[validate(range(min = 300, max = 86400))] // 5 min to 24 hours
    pub session_duration_secs: u64,

    /// Storage path for MFA secrets
    pub storage_path: PathBuf,
}

/// Available MFA methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaMethods {
    pub totp: bool,
    pub sms: bool,
    pub email: bool,
    pub webauthn: bool,
}

/// TOTP (Time-based One-Time Password) configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TotpConfig {
    /// Issuer name shown in authenticator apps
    #[validate(length(min = 1))]
    pub issuer: String,

    /// Number of digits in TOTP code
    #[validate(range(min = 6, max = 8))]
    pub digits: u32,

    /// Time step in seconds
    #[validate(range(min = 15, max = 60))]
    pub time_step_secs: u32,

    /// Number of time steps to look back/forward for validation
    #[validate(range(min = 0, max = 5))]
    pub skew: u32,
}

/// SMS-based MFA configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SmsConfig {
    /// SMS provider
    pub provider: SmsProvider,

    /// API key or credentials
    pub api_key: String,

    /// Sender phone number or ID
    pub sender: String,

    /// Code expiration in seconds
    #[validate(range(min = 60, max = 600))]
    pub code_expiration_secs: u64,
}

/// SMS providers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmsProvider {
    Twilio,
    Aws,
    Custom,
}

/// Email-based MFA configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EmailConfig {
    /// SMTP server
    #[validate(length(min = 1))]
    pub smtp_server: String,

    /// SMTP port
    #[validate(range(min = 1, max = 65535))]
    pub smtp_port: u16,

    /// Sender email address
    #[validate(email)]
    pub from_address: String,

    /// SMTP username
    pub smtp_username: Option<String>,

    /// SMTP password
    pub smtp_password: Option<String>,

    /// Use TLS
    pub use_tls: bool,

    /// Code expiration in seconds
    #[validate(range(min = 60, max = 600))]
    pub code_expiration_secs: u64,
}

/// WebAuthn configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct WebAuthnConfig {
    /// Relying Party ID (usually the domain)
    #[validate(length(min = 1))]
    pub rp_id: String,

    /// Relying Party name
    #[validate(length(min = 1))]
    pub rp_name: String,

    /// Origin URL
    #[validate(url)]
    pub origin: String,

    /// Require resident keys
    pub require_resident_key: bool,

    /// User verification requirement
    pub user_verification: UserVerificationRequirement,
}

/// User verification requirements for WebAuthn
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserVerificationRequirement {
    Required,
    Preferred,
    Discouraged,
}

/// Backup codes configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BackupCodesConfig {
    /// Enable backup codes
    pub enabled: bool,

    /// Number of backup codes to generate
    #[validate(range(min = 5, max = 20))]
    pub count: u32,

    /// Length of each backup code
    #[validate(range(min = 8, max = 16))]
    pub length: u32,
}

/// JWT configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct JwtConfig {
    #[validate(length(min = 32))]
    pub secret: String,

    #[validate(range(min = 300, max = 86400))] // 5 min to 24 hours
    pub expiration_secs: u64,

    #[validate(length(min = 1))]
    pub issuer: String,

    #[validate(length(min = 1))]
    pub audience: String,
}

/// OAuth configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OAuthConfig {
    #[validate(length(min = 1))]
    pub provider: String,

    #[validate(length(min = 1))]
    pub client_id: String,

    #[validate(length(min = 1))]
    pub client_secret: String,

    #[validate(url)]
    pub auth_url: String,

    #[validate(url)]
    pub token_url: String,

    #[validate(url)]
    pub user_info_url: String,

    pub scopes: Vec<String>,
}

/// LDAP configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LdapConfig {
    #[validate(url)]
    pub server: String,

    #[validate(length(min = 1))]
    pub bind_dn: String,

    #[validate(length(min = 1))]
    pub bind_password: String,

    #[validate(length(min = 1))]
    pub user_base_dn: String,

    #[validate(length(min = 1))]
    pub user_filter: String,

    #[validate(length(min = 1))]
    pub group_base_dn: String,

    #[validate(length(min = 1))]
    pub group_filter: String,

    pub use_tls: bool,
}

/// User configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct UserConfig {
    #[validate(length(min = 1))]
    pub password_hash: String,

    pub roles: Vec<String>,

    pub permissions: Vec<crate::auth::types::Permission>,

    pub enabled: bool,

    pub email: Option<String>,

    pub full_name: Option<String>,

    pub last_login: Option<chrono::DateTime<chrono::Utc>>,

    pub failed_login_attempts: u32,

    pub locked_until: Option<chrono::DateTime<chrono::Utc>>,
}

/// Custom validation function for PathBuf
fn validate_path(path: &std::path::Path) -> Result<(), ValidationError> {
    if path.as_os_str().is_empty() {
        return Err(ValidationError::new("path_empty"));
    }
    Ok(())
}
