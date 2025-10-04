//! Secure Credential Management
//!
//! Provides encrypted storage and retrieval of API keys and sensitive credentials.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Encrypted credential
#[derive(Debug, Clone)]
pub struct SecureCredential {
    provider: String,
    encrypted_value: Vec<u8>,
    created_at: chrono::DateTime<chrono::Utc>,
    last_accessed: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
}

impl SecureCredential {
    /// Get the decrypted credential value
    pub fn decrypt(&self, encryptor: &super::encryption::Encryptor) -> Result<String> {
        let decrypted = encryptor.decrypt(&self.encrypted_value)?;

        // Update last accessed time
        if let Ok(mut last_accessed) = self.last_accessed.write() {
            *last_accessed = chrono::Utc::now();
        }

        Ok(String::from_utf8(decrypted)?)
    }

    /// Get credential metadata
    pub fn metadata(&self) -> CredentialMetadata {
        let last_accessed = self.last_accessed.read()
            .map(|t| *t)
            .unwrap_or(self.created_at);

        CredentialMetadata {
            provider: self.provider.clone(),
            created_at: self.created_at,
            last_accessed,
        }
    }
}

/// Credential metadata (non-sensitive)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialMetadata {
    pub provider: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Credential manager with encryption
pub struct CredentialManager {
    credentials: Arc<RwLock<HashMap<String, SecureCredential>>>,
    encryptor: Option<super::encryption::Encryptor>,
    encrypt_enabled: bool,
}

impl CredentialManager {
    pub fn new(encrypt_enabled: bool) -> Result<Self> {
        let encryptor = if encrypt_enabled {
            Some(super::encryption::Encryptor::new()?)
        } else {
            None
        };

        Ok(Self {
            credentials: Arc::new(RwLock::new(HashMap::new())),
            encryptor,
            encrypt_enabled,
        })
    }

    /// Store a credential securely
    pub fn store(&mut self, provider: &str, api_key: &str) -> Result<()> {
        // Validate input
        if provider.is_empty() {
            return Err(anyhow!("Provider name cannot be empty"));
        }
        if api_key.is_empty() {
            return Err(anyhow!("API key cannot be empty"));
        }

        // Encrypt if enabled
        let encrypted_value = if self.encrypt_enabled {
            self.encryptor
                .as_ref()
                .ok_or_else(|| anyhow!("Encryptor not initialized"))?
                .encrypt(api_key.as_bytes())?
        } else {
            // Store as plain bytes if encryption disabled
            api_key.as_bytes().to_vec()
        };

        let credential = SecureCredential {
            provider: provider.to_string(),
            encrypted_value,
            created_at: chrono::Utc::now(),
            last_accessed: Arc::new(RwLock::new(chrono::Utc::now())),
        };

        let mut credentials = self.credentials.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        credentials.insert(provider.to_string(), credential);

        Ok(())
    }

    /// Retrieve a credential securely
    pub fn retrieve(&self, provider: &str) -> Result<Option<SecureCredential>> {
        let credentials = self.credentials.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(credentials.get(provider).cloned())
    }

    /// Get decrypted credential value
    pub fn get_api_key(&self, provider: &str) -> Result<Option<String>> {
        let credential = self.retrieve(provider)?;

        match credential {
            Some(cred) => {
                if self.encrypt_enabled {
                    let encryptor = self.encryptor
                        .as_ref()
                        .ok_or_else(|| anyhow!("Encryptor not initialized"))?;
                    Ok(Some(cred.decrypt(encryptor)?))
                } else {
                    // Decrypt without encryptor (plain text)
                    Ok(Some(String::from_utf8(cred.encrypted_value)?))
                }
            }
            None => Ok(None),
        }
    }

    /// Remove a credential
    pub fn remove(&mut self, provider: &str) -> Result<bool> {
        let mut credentials = self.credentials.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        Ok(credentials.remove(provider).is_some())
    }

    /// List all credential metadata
    pub fn list_metadata(&self) -> Result<Vec<CredentialMetadata>> {
        let credentials = self.credentials.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(credentials.values()
            .map(|cred| cred.metadata())
            .collect())
    }

    /// Rotate encryption key (re-encrypt all credentials with new key)
    pub fn rotate_encryption_key(&mut self) -> Result<()> {
        if !self.encrypt_enabled {
            return Ok(());
        }

        let old_encryptor = self.encryptor
            .as_ref()
            .ok_or_else(|| anyhow!("Encryptor not initialized"))?;

        // Create new encryptor
        let new_encryptor = super::encryption::Encryptor::new()?;

        // Re-encrypt all credentials
        let mut credentials = self.credentials.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        for credential in credentials.values_mut() {
            // Decrypt with old key
            let decrypted = old_encryptor.decrypt(&credential.encrypted_value)?;

            // Encrypt with new key
            credential.encrypted_value = new_encryptor.encrypt(&decrypted)?;
        }

        self.encryptor = Some(new_encryptor);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_storage_and_retrieval() {
        let mut manager = CredentialManager::new(false).unwrap();

        // Store credential
        manager.store("openai", "sk-test-key-123").unwrap();

        // Retrieve credential
        let api_key = manager.get_api_key("openai").unwrap();
        assert_eq!(api_key, Some("sk-test-key-123".to_string()));
    }

    #[test]
    fn test_encrypted_credential_storage() {
        let mut manager = CredentialManager::new(true).unwrap();

        // Store encrypted credential
        manager.store("anthropic", "sk-ant-key-456").unwrap();

        // Retrieve and decrypt
        let api_key = manager.get_api_key("anthropic").unwrap();
        assert_eq!(api_key, Some("sk-ant-key-456".to_string()));
    }

    #[test]
    fn test_credential_removal() {
        let mut manager = CredentialManager::new(false).unwrap();

        manager.store("test_provider", "test_key").unwrap();
        assert!(manager.get_api_key("test_provider").unwrap().is_some());

        manager.remove("test_provider").unwrap();
        assert!(manager.get_api_key("test_provider").unwrap().is_none());
    }

    #[test]
    fn test_list_metadata() {
        let mut manager = CredentialManager::new(false).unwrap();

        manager.store("provider1", "key1").unwrap();
        manager.store("provider2", "key2").unwrap();

        let metadata = manager.list_metadata().unwrap();
        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_invalid_input() {
        let mut manager = CredentialManager::new(false).unwrap();

        // Empty provider should fail
        assert!(manager.store("", "key").is_err());

        // Empty API key should fail
        assert!(manager.store("provider", "").is_err());
    }
}
