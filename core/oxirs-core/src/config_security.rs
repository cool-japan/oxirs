//! Security configuration: authentication, authorization, encryption, audit.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub authentication: AuthenticationConfig,
    pub authorization: AuthorizationConfig,
    pub encryption: EncryptionConfig,
    pub audit: AuditConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub providers: Vec<AuthProvider>,
    pub session: SessionConfig,
    pub token: TokenConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthProvider {
    None,
    Basic {
        realm: String,
    },
    Bearer {
        issuer: String,
        audience: String,
    },
    OAuth2 {
        client_id: String,
        client_secret: String,
    },
    LDAP {
        server: String,
        base_dn: String,
    },
    Custom {
        provider_type: String,
        config: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub timeout: Duration,
    pub storage: SessionStorage,
    pub encryption: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStorage {
    Memory,
    File,
    Database,
    Redis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub token_type: TokenType,
    pub expiry: Duration,
    pub refresh_enabled: bool,
    pub signing_key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    JWT,
    Opaque,
    PASETO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    pub model: AuthorizationModel,
    pub permissions: Vec<Permission>,
    pub roles: Vec<Role>,
    pub policies: Vec<Policy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorizationModel {
    None,
    RBAC,
    ABAC,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub name: String,
    pub description: String,
    pub resource: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub description: String,
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub name: String,
    pub expression: String,
    pub effect: PolicyEffect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyEffect {
    Allow,
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub at_rest: EncryptionAtRest,
    pub in_transit: EncryptionInTransit,
    pub key_management: KeyManagementConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAtRest {
    pub enabled: bool,
    pub algorithm: EncryptionAlgorithm,
    pub key_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInTransit {
    pub enabled: bool,
    pub tls_version: TlsVersion,
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES128,
    AES256,
    ChaCha20,
    XChaCha20,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsVersion {
    TLSv1_2,
    TLSv1_3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    pub provider: KeyProvider,
    pub rotation_interval: Duration,
    pub kdf: KeyDerivationFunction,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyProvider {
    File,
    Environment,
    HSM,
    Vault,
    AWS_KMS,
    Azure_KeyVault,
    GCP_KMS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    PBKDF2,
    Scrypt,
    Argon2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_path: PathBuf,
    pub events: Vec<AuditEvent>,
    pub retention: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEvent {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    AdminActions,
    SecurityEvents,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication: AuthenticationConfig::default(),
            authorization: AuthorizationConfig::default(),
            encryption: EncryptionConfig::default(),
            audit: AuditConfig::default(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            providers: vec![AuthProvider::None],
            session: SessionConfig::default(),
            token: TokenConfig::default(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600),
            storage: SessionStorage::Memory,
            encryption: true,
        }
    }
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            token_type: TokenType::JWT,
            expiry: Duration::from_secs(3600),
            refresh_enabled: true,
            signing_key: "default_key".to_string(),
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        Self {
            model: AuthorizationModel::RBAC,
            permissions: Vec::new(),
            roles: Vec::new(),
            policies: Vec::new(),
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            at_rest: EncryptionAtRest::default(),
            in_transit: EncryptionInTransit::default(),
            key_management: KeyManagementConfig::default(),
        }
    }
}

impl Default for EncryptionAtRest {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: EncryptionAlgorithm::AES256,
            key_size: 256,
        }
    }
}

impl Default for EncryptionInTransit {
    fn default() -> Self {
        Self {
            enabled: true,
            tls_version: TlsVersion::TLSv1_3,
            cert_path: PathBuf::from("cert.pem"),
            key_path: PathBuf::from("key.pem"),
        }
    }
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            provider: KeyProvider::File,
            rotation_interval: Duration::from_secs(86400 * 30),
            kdf: KeyDerivationFunction::Argon2,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_path: PathBuf::from("./audit.log"),
            events: vec![AuditEvent::Authentication, AuditEvent::Authorization],
            retention: Duration::from_secs(86400 * 365),
        }
    }
}
