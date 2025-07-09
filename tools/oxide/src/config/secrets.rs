//! Secret management for sensitive configuration data
//!
//! Provides secure storage and retrieval of sensitive information like
//! API keys, passwords, and authentication tokens.

use crate::cli::error::{CliError, CliResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

/// Secret manager for handling sensitive data
pub struct SecretManager {
    /// Secrets directory
    secrets_dir: PathBuf,
    /// In-memory cache of decrypted secrets
    cache: HashMap<String, SecretValue>,
    /// Encryption key (derived from master password or system keyring)
    key: Option<Vec<u8>>,
}

/// A secret value with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretValue {
    /// The actual secret value (encrypted when stored)
    pub value: String,
    /// Description of the secret
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Expiration time (if any)
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Secret storage backend
#[derive(Debug, Clone, Copy)]
pub enum SecretBackend {
    /// File-based storage (encrypted)
    File,
    /// System keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service)
    SystemKeyring,
    /// Environment variables
    Environment,
    /// HashiCorp Vault
    Vault,
}

impl SecretManager {
    /// Create a new secret manager
    pub fn new(_backend: SecretBackend) -> CliResult<Self> {
        let secrets_dir = Self::get_secrets_dir()?;

        // Ensure secrets directory exists with restricted permissions
        if !secrets_dir.exists() {
            fs::create_dir_all(&secrets_dir).map_err(|e| {
                CliError::config_error(format!("Cannot create secrets directory: {e}"))
            })?;

            // Set restrictive permissions (700 - owner only)
            #[cfg(unix)]
            {
                let metadata = fs::metadata(&secrets_dir)?;
                let mut permissions = metadata.permissions();
                permissions.set_mode(0o700);
                fs::set_permissions(&secrets_dir, permissions)?;
            }
        }

        Ok(Self {
            secrets_dir,
            cache: HashMap::new(),
            key: None,
        })
    }

    /// Get the secrets directory
    fn get_secrets_dir() -> CliResult<PathBuf> {
        // Check environment variable first
        if let Ok(dir) = std::env::var("OXIDE_SECRETS_DIR") {
            return Ok(PathBuf::from(dir));
        }

        // Use platform-specific secure directory
        #[cfg(target_os = "macos")]
        let base_dir = dirs::home_dir().map(|h| h.join("Library/Application Support"));

        #[cfg(target_os = "linux")]
        let base_dir = dirs::config_dir();

        #[cfg(target_os = "windows")]
        let base_dir = dirs::data_local_dir();

        base_dir
            .map(|p| p.join("oxide/secrets"))
            .ok_or_else(|| CliError::config_error("Cannot determine secrets directory"))
    }

    /// Initialize with master password
    pub fn unlock(&mut self, password: &str) -> CliResult<()> {
        // Derive encryption key from password using a key derivation function
        self.key = Some(self.derive_key(password)?);

        // Try to decrypt a test secret to verify the password
        if self.secrets_dir.join(".test").exists() {
            self.get_secret(".test")?;
        }

        Ok(())
    }

    /// Check if the secret manager is unlocked
    pub fn is_unlocked(&self) -> bool {
        self.key.is_some()
    }

    /// Store a secret
    pub fn set_secret(
        &mut self,
        name: &str,
        value: &str,
        description: Option<String>,
    ) -> CliResult<()> {
        if !self.is_unlocked() {
            return Err(CliError::config_error("Secret manager is locked"));
        }

        let secret = SecretValue {
            value: value.to_string(),
            description,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            expires_at: None,
        };

        // Encrypt and store
        self.store_encrypted_secret(name, &secret)?;

        // Update cache
        self.cache.insert(name.to_string(), secret);

        Ok(())
    }

    /// Retrieve a secret
    pub fn get_secret(&mut self, name: &str) -> CliResult<String> {
        // Check cache first
        if let Some(secret) = self.cache.get(name) {
            // Check expiration
            if let Some(expires) = secret.expires_at {
                if expires < chrono::Utc::now() {
                    self.cache.remove(name);
                    return Err(CliError::config_error("Secret has expired"));
                }
            }
            return Ok(secret.value.clone());
        }

        // Try environment variable
        let env_name = format!("OXIDE_SECRET_{}", name.to_uppercase().replace('-', "_"));
        if let Ok(value) = std::env::var(&env_name) {
            return Ok(value);
        }

        // Load from encrypted storage
        if !self.is_unlocked() {
            return Err(CliError::config_error("Secret manager is locked"));
        }

        let secret = self.load_encrypted_secret(name)?;

        // Check expiration
        if let Some(expires) = secret.expires_at {
            if expires < chrono::Utc::now() {
                return Err(CliError::config_error("Secret has expired"));
            }
        }

        let value = secret.value.clone();
        self.cache.insert(name.to_string(), secret);

        Ok(value)
    }

    /// Delete a secret
    pub fn delete_secret(&mut self, name: &str) -> CliResult<()> {
        self.cache.remove(name);

        let path = self.secrets_dir.join(format!("{name}.secret"));
        if path.exists() {
            fs::remove_file(path)
                .map_err(|e| CliError::config_error(format!("Cannot delete secret: {e}")))?;
        }

        Ok(())
    }

    /// List all secrets (names only, not values)
    pub fn list_secrets(&self) -> CliResult<Vec<SecretInfo>> {
        let mut secrets = Vec::new();

        if self.secrets_dir.exists() {
            for entry in fs::read_dir(&self.secrets_dir)? {
                let entry = entry?;
                let path = entry.path();

                if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                    if path.extension().and_then(|e| e.to_str()) == Some("secret")
                        && name != ".test"
                    {
                        // Try to load metadata without decrypting value
                        if let Ok(metadata) = self.load_secret_metadata(name) {
                            secrets.push(metadata);
                        }
                    }
                }
            }
        }

        // Also list environment variable secrets
        for (key, _) in std::env::vars() {
            if key.starts_with("OXIDE_SECRET_") {
                let name = key
                    .strip_prefix("OXIDE_SECRET_")
                    .unwrap()
                    .to_lowercase()
                    .replace('_', "-");

                secrets.push(SecretInfo {
                    name,
                    description: Some("Environment variable".to_string()),
                    created_at: None,
                    expires_at: None,
                    source: SecretSource::Environment,
                });
            }
        }

        secrets.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(secrets)
    }

    /// Derive encryption key from password
    fn derive_key(&self, password: &str) -> CliResult<Vec<u8>> {
        use ring::pbkdf2;

        let salt = b"oxide-secret-salt"; // In production, use a random salt stored separately
        let mut key = vec![0u8; 32];

        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(100_000).unwrap(),
            salt,
            password.as_bytes(),
            &mut key,
        );

        Ok(key)
    }

    /// Encrypt and store a secret
    fn store_encrypted_secret(&self, name: &str, secret: &SecretValue) -> CliResult<()> {
        use ring::aead;

        let key = self
            .key
            .as_ref()
            .ok_or_else(|| CliError::config_error("No encryption key available"))?;

        // Serialize secret
        let plaintext = serde_json::to_vec(secret)
            .map_err(|e| CliError::config_error(format!("Cannot serialize secret: {e}")))?;

        // Encrypt using AES-GCM
        let key = aead::UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| CliError::config_error("Invalid encryption key"))?;
        let key = aead::LessSafeKey::new(key);

        let nonce = aead::Nonce::assume_unique_for_key([0u8; 12]); // In production, use random nonce
        let mut ciphertext = plaintext.clone();

        key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut ciphertext)
            .map_err(|_| CliError::config_error("Encryption failed"))?;

        // Write to file
        let path = self.secrets_dir.join(format!("{name}.secret"));
        let mut file = File::create(&path)
            .map_err(|e| CliError::config_error(format!("Cannot create secret file: {e}")))?;

        // Set restrictive permissions
        #[cfg(unix)]
        {
            let metadata = file.metadata()?;
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o600);
            file.set_permissions(permissions)?;
        }

        file.write_all(&ciphertext)
            .map_err(|e| CliError::config_error(format!("Cannot write secret: {e}")))?;

        Ok(())
    }

    /// Load and decrypt a secret
    fn load_encrypted_secret(&self, name: &str) -> CliResult<SecretValue> {
        use ring::aead;

        let key = self
            .key
            .as_ref()
            .ok_or_else(|| CliError::config_error("No decryption key available"))?;

        // Read encrypted data
        let path = self.secrets_dir.join(format!("{name}.secret"));
        let mut file = File::open(&path)
            .map_err(|_| CliError::config_error(format!("Secret '{name}' not found")))?;

        let mut ciphertext = Vec::new();
        file.read_to_end(&mut ciphertext)
            .map_err(|e| CliError::config_error(format!("Cannot read secret: {e}")))?;

        // Decrypt using AES-GCM
        let key = aead::UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| CliError::config_error("Invalid decryption key"))?;
        let key = aead::LessSafeKey::new(key);

        let nonce = aead::Nonce::assume_unique_for_key([0u8; 12]);

        let plaintext = key
            .open_in_place(nonce, aead::Aad::empty(), &mut ciphertext)
            .map_err(|_| CliError::config_error("Decryption failed - wrong password?"))?;

        // Deserialize secret
        serde_json::from_slice(plaintext)
            .map_err(|e| CliError::config_error(format!("Cannot deserialize secret: {e}")))
    }

    /// Load secret metadata without decrypting the value
    fn load_secret_metadata(&self, name: &str) -> CliResult<SecretInfo> {
        // For now, return basic info
        // In a real implementation, we'd store metadata separately
        Ok(SecretInfo {
            name: name.to_string(),
            description: None,
            created_at: None,
            expires_at: None,
            source: SecretSource::File,
        })
    }
}

/// Information about a secret (without the actual value)
#[derive(Debug, Clone)]
pub struct SecretInfo {
    pub name: String,
    pub description: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source: SecretSource,
}

/// Source of a secret
#[derive(Debug, Clone, Copy)]
pub enum SecretSource {
    File,
    Environment,
    SystemKeyring,
}

/// Integration with system keyring
pub mod keyring {
    use super::*;

    /// Store a secret in the system keyring
    pub fn store_in_keyring(_service: &str, _name: &str, _value: &str) -> CliResult<()> {
        // This would use platform-specific APIs:
        // - macOS: Security framework
        // - Windows: Windows Credential Manager
        // - Linux: Secret Service API

        // For now, return not implemented
        Err(CliError::config_error("System keyring not yet implemented"))
    }

    /// Retrieve a secret from the system keyring
    pub fn get_from_keyring(_service: &str, _name: &str) -> CliResult<String> {
        Err(CliError::config_error("System keyring not yet implemented"))
    }
}

/// Secure credential helpers
pub mod credentials {
    use super::*;

    /// SPARQL endpoint credentials
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EndpointCredentials {
        pub url: String,
        pub username: Option<String>,
        pub password: Option<String>,
        pub auth_type: AuthType,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum AuthType {
        None,
        Basic,
        Bearer,
        OAuth2,
    }

    /// Get credentials for a SPARQL endpoint
    pub fn get_endpoint_credentials(
        manager: &mut SecretManager,
        url: &str,
    ) -> CliResult<EndpointCredentials> {
        let secret_name = format!("endpoint_{}", url.replace(['/', ':'], "_"));

        if let Ok(creds_json) = manager.get_secret(&secret_name) {
            serde_json::from_str(&creds_json)
                .map_err(|e| CliError::config_error(format!("Invalid credentials format: {e}")))
        } else {
            Ok(EndpointCredentials {
                url: url.to_string(),
                username: None,
                password: None,
                auth_type: AuthType::None,
            })
        }
    }

    /// Store credentials for a SPARQL endpoint
    pub fn store_endpoint_credentials(
        manager: &mut SecretManager,
        creds: &EndpointCredentials,
    ) -> CliResult<()> {
        let secret_name = format!("endpoint_{}", creds.url.replace(['/', ':'], "_"));
        let creds_json = serde_json::to_string(creds)
            .map_err(|e| CliError::config_error(format!("Cannot serialize credentials: {e}")))?;

        manager.set_secret(
            &secret_name,
            &creds_json,
            Some(format!("Credentials for {}", creds.url)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_secret_manager_creation() {
        let dir = tempdir().unwrap();
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("OXIDE_SECRETS_DIR", dir.path()) };

        let manager = SecretManager::new(SecretBackend::File).unwrap();
        assert!(!manager.is_unlocked());

        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("OXIDE_SECRETS_DIR") };
    }

    #[test]
    fn test_secret_storage_and_retrieval() {
        let dir = tempdir().unwrap();
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("OXIDE_SECRETS_DIR", dir.path()) };

        let mut manager = SecretManager::new(SecretBackend::File).unwrap();
        manager.unlock("test-password").unwrap();

        // Store a secret
        manager
            .set_secret("test-key", "test-value", Some("Test secret".to_string()))
            .unwrap();

        // Retrieve it
        let value = manager.get_secret("test-key").unwrap();
        assert_eq!(value, "test-value");

        // List secrets
        let secrets = manager.list_secrets().unwrap();
        assert!(secrets.iter().any(|s| s.name == "test-key"));

        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("OXIDE_SECRETS_DIR") };
    }

    #[test]
    fn test_environment_secret() {
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::set_var("OXIDE_SECRET_API_KEY", "secret-api-key") };

        let mut manager = SecretManager::new(SecretBackend::File).unwrap();
        let value = manager.get_secret("api-key").unwrap();
        assert_eq!(value, "secret-api-key");

        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { std::env::remove_var("OXIDE_SECRET_API_KEY") };
    }
}
