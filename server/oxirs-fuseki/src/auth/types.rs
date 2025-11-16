//! Authentication types and data structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// MFA method configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MfaMethod {
    Totp,
    Sms,
    Email,
    Hardware,
    Backup,
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

/// Permission types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Read,
    Write,
    Admin,
    GlobalAdmin,
    GlobalRead,
    GlobalWrite,
    DatasetCreate,
    DatasetDelete,
    DatasetManage,
    DatasetRead(String),
    DatasetWrite(String),
    DatasetAdmin(String),
    UserManage,
    UserManagement,
    SystemConfig,
    SystemMetrics,
    QueryExecute,
    UpdateExecute,
    SparqlQuery,
    SparqlUpdate,
    GraphStore,
    Upload,
    Download,
    Backup,
    Restore,
    Monitor,
    Audit,
    ServiceManage,
    ClusterManage,
    FederationManage,
}

/// MFA status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaStatus {
    pub enabled: bool,
    pub enrolled_methods: Vec<MfaMethodInfo>,
    pub backup_codes_remaining: u8,
    pub last_used: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub message: String,
}

/// MFA method information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaMethodInfo {
    pub method_type: MfaType,
    pub identifier: String, // Phone number, email, or device name
    pub enrolled_at: DateTime<Utc>,
    pub last_used: Option<DateTime<Utc>>,
    pub enabled: bool,
}

/// JWT token claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
    pub exp: i64,
    pub iat: i64,
    pub nbf: i64,
    pub iss: String,
    pub aud: String,
}

/// MFA secrets storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfaSecrets {
    pub totp_secret: Option<String>,
    pub backup_codes: Vec<String>,
    pub recovery_codes: Vec<String>,
    pub hardware_tokens: Vec<String>,
    pub sms_phone: Option<String>,
    pub mfa_email: Option<String>,
    pub webauthn_challenges: HashMap<String, String>,
}

/// User session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    pub user: User,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

/// Login request
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
    pub mfa_token: Option<String>,
}

/// Login response
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: User,
    pub mfa_required: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub message: String,
}

/// Token validation result
#[derive(Debug)]
pub struct TokenValidation {
    pub user: User,
    pub expires_at: DateTime<Utc>,
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token expired")]
    TokenExpired,
    #[error("Invalid token")]
    InvalidToken,
    #[error("MFA required")]
    MfaRequired,
    #[error("Invalid MFA token")]
    InvalidMfaToken,
    #[error("User not found")]
    UserNotFound,
    #[error("User disabled")]
    UserDisabled,
    #[error("User locked")]
    UserLocked,
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Certificate required")]
    CertificateRequired,
    #[error("Invalid certificate")]
    InvalidCertificate,
    #[error("SAML error: {0}")]
    SamlError(String),
    #[error("LDAP error: {0}")]
    LdapError(String),
    #[error("OAuth error: {0}")]
    OAuthError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Password strength levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PasswordStrength {
    VeryWeak,
    Weak,
    Medium,
    Strong,
    VeryStrong,
}
