//! MFA Storage Implementation
//!
//! Provides persistent storage for MFA-related data including:
//! - TOTP secrets
//! - Backup codes
//! - Email addresses
//! - SMS phone numbers
//! - WebAuthn credentials

use crate::error::{FusekiError, FusekiResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// MFA storage backend
pub struct MfaStorage {
    /// In-memory cache of MFA data
    cache: Arc<DashMap<String, UserMfaData>>,
    /// Optional persistent storage path
    storage_path: Option<String>,
}

/// Complete MFA data for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMfaData {
    pub username: String,
    pub totp_secret: Option<String>,
    pub backup_codes: Vec<String>,
    pub email: Option<String>,
    pub sms_phone: Option<String>,
    pub webauthn_credentials: Vec<WebAuthnCredential>,
    pub enrolled_methods: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// WebAuthn credential
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebAuthnCredential {
    pub credential_id: String,
    pub public_key: String,
    pub counter: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

impl MfaStorage {
    /// Create a new MFA storage instance
    pub fn new(storage_path: Option<String>) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            storage_path,
        }
    }

    /// Initialize storage from persistent backend
    pub async fn initialize(&self) -> FusekiResult<()> {
        if let Some(path) = &self.storage_path {
            info!("Initializing MFA storage from: {}", path);
            self.load_from_disk(path).await?;
        }
        Ok(())
    }

    /// Load MFA data from disk
    async fn load_from_disk(&self, path: &str) -> FusekiResult<()> {
        let path = Path::new(path);
        if !path.exists() {
            debug!("MFA storage file does not exist, starting with empty storage");
            return Ok(());
        }

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to read MFA storage: {}", e)))?;

        let data: HashMap<String, UserMfaData> = serde_json::from_str(&content)
            .map_err(|e| FusekiError::internal(format!("Failed to parse MFA storage: {}", e)))?;

        for (username, user_data) in data {
            self.cache.insert(username, user_data);
        }

        info!("Loaded MFA data for {} users", self.cache.len());
        Ok(())
    }

    /// Save MFA data to disk
    async fn save_to_disk(&self) -> FusekiResult<()> {
        if let Some(path) = &self.storage_path {
            let data: HashMap<String, UserMfaData> = self
                .cache
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect();

            let content = serde_json::to_string_pretty(&data)
                .map_err(|e| FusekiError::internal(format!("Failed to serialize MFA data: {}", e)))?;

            fs::write(path, content)
                .await
                .map_err(|e| FusekiError::internal(format!("Failed to write MFA storage: {}", e)))?;

            debug!("Saved MFA data to disk");
        }
        Ok(())
    }

    /// Store TOTP secret for user
    pub async fn store_totp_secret(&self, username: &str, secret: &str) -> FusekiResult<()> {
        let now = chrono::Utc::now();

        self.cache
            .entry(username.to_string())
            .and_modify(|data| {
                data.totp_secret = Some(secret.to_string());
                data.updated_at = now;
                if !data.enrolled_methods.contains(&"totp".to_string()) {
                    data.enrolled_methods.push("totp".to_string());
                }
            })
            .or_insert_with(|| UserMfaData {
                username: username.to_string(),
                totp_secret: Some(secret.to_string()),
                backup_codes: Vec::new(),
                email: None,
                sms_phone: None,
                webauthn_credentials: Vec::new(),
                enrolled_methods: vec!["totp".to_string()],
                created_at: now,
                updated_at: now,
            });

        self.save_to_disk().await?;
        info!("Stored TOTP secret for user: {}", username);
        Ok(())
    }

    /// Get TOTP secret for user
    pub async fn get_totp_secret(&self, username: &str) -> FusekiResult<Option<String>> {
        Ok(self.cache.get(username).and_then(|data| data.totp_secret.clone()))
    }

    /// Store backup codes for user
    pub async fn store_backup_codes(&self, username: &str, codes: Vec<String>) -> FusekiResult<()> {
        let now = chrono::Utc::now();

        self.cache
            .entry(username.to_string())
            .and_modify(|data| {
                data.backup_codes = codes.clone();
                data.updated_at = now;
            })
            .or_insert_with(|| UserMfaData {
                username: username.to_string(),
                totp_secret: None,
                backup_codes: codes,
                email: None,
                sms_phone: None,
                webauthn_credentials: Vec::new(),
                enrolled_methods: Vec::new(),
                created_at: now,
                updated_at: now,
            });

        self.save_to_disk().await?;
        info!("Stored {} backup codes for user: {}", codes.len(), username);
        Ok(())
    }

    /// Get backup codes for user
    pub async fn get_backup_codes(&self, username: &str) -> FusekiResult<Vec<String>> {
        Ok(self
            .cache
            .get(username)
            .map(|data| data.backup_codes.clone())
            .unwrap_or_default())
    }

    /// Verify and consume a backup code
    pub async fn verify_backup_code(&self, username: &str, code: &str) -> FusekiResult<bool> {
        let mut consumed = false;

        self.cache.entry(username.to_string()).and_modify(|data| {
            if let Some(index) = data.backup_codes.iter().position(|c| c == code) {
                data.backup_codes.remove(index);
                data.updated_at = chrono::Utc::now();
                consumed = true;
            }
        });

        if consumed {
            self.save_to_disk().await?;
            info!("Consumed backup code for user: {}", username);
        }

        Ok(consumed)
    }

    /// Store email for MFA
    pub async fn store_email(&self, username: &str, email: &str) -> FusekiResult<()> {
        let now = chrono::Utc::now();

        self.cache
            .entry(username.to_string())
            .and_modify(|data| {
                data.email = Some(email.to_string());
                data.updated_at = now;
            })
            .or_insert_with(|| UserMfaData {
                username: username.to_string(),
                totp_secret: None,
                backup_codes: Vec::new(),
                email: Some(email.to_string()),
                sms_phone: None,
                webauthn_credentials: Vec::new(),
                enrolled_methods: Vec::new(),
                created_at: now,
                updated_at: now,
            });

        self.save_to_disk().await?;
        debug!("Stored email for user: {}", username);
        Ok(())
    }

    /// Get email for user
    pub async fn get_email(&self, username: &str) -> FusekiResult<Option<String>> {
        Ok(self.cache.get(username).and_then(|data| data.email.clone()))
    }

    /// Store SMS phone number
    pub async fn store_sms_phone(&self, username: &str, phone: &str) -> FusekiResult<()> {
        let now = chrono::Utc::now();

        self.cache
            .entry(username.to_string())
            .and_modify(|data| {
                data.sms_phone = Some(phone.to_string());
                data.updated_at = now;
                if !data.enrolled_methods.contains(&"sms".to_string()) {
                    data.enrolled_methods.push("sms".to_string());
                }
            })
            .or_insert_with(|| UserMfaData {
                username: username.to_string(),
                totp_secret: None,
                backup_codes: Vec::new(),
                email: None,
                sms_phone: Some(phone.to_string()),
                webauthn_credentials: Vec::new(),
                enrolled_methods: vec!["sms".to_string()],
                created_at: now,
                updated_at: now,
            });

        self.save_to_disk().await?;
        info!("Stored SMS phone for user: {}", username);
        Ok(())
    }

    /// Get SMS phone number for user
    pub async fn get_sms_phone(&self, username: &str) -> FusekiResult<Option<String>> {
        Ok(self.cache.get(username).and_then(|data| data.sms_phone.clone()))
    }

    /// Store WebAuthn credential
    pub async fn store_webauthn_credential(
        &self,
        username: &str,
        credential: WebAuthnCredential,
    ) -> FusekiResult<()> {
        let now = chrono::Utc::now();

        self.cache
            .entry(username.to_string())
            .and_modify(|data| {
                data.webauthn_credentials.push(credential.clone());
                data.updated_at = now;
                if !data.enrolled_methods.contains(&"webauthn".to_string()) {
                    data.enrolled_methods.push("webauthn".to_string());
                }
            })
            .or_insert_with(|| UserMfaData {
                username: username.to_string(),
                totp_secret: None,
                backup_codes: Vec::new(),
                email: None,
                sms_phone: None,
                webauthn_credentials: vec![credential],
                enrolled_methods: vec!["webauthn".to_string()],
                created_at: now,
                updated_at: now,
            });

        self.save_to_disk().await?;
        info!("Stored WebAuthn credential for user: {}", username);
        Ok(())
    }

    /// Get WebAuthn credentials for user
    pub async fn get_webauthn_credentials(
        &self,
        username: &str,
    ) -> FusekiResult<Vec<WebAuthnCredential>> {
        Ok(self
            .cache
            .get(username)
            .map(|data| data.webauthn_credentials.clone())
            .unwrap_or_default())
    }

    /// Get enrolled MFA methods for user
    pub async fn get_enrolled_methods(&self, username: &str) -> FusekiResult<Vec<String>> {
        Ok(self
            .cache
            .get(username)
            .map(|data| data.enrolled_methods.clone())
            .unwrap_or_default())
    }

    /// Disable MFA method for user
    pub async fn disable_method(&self, username: &str, method: &str) -> FusekiResult<()> {
        self.cache.entry(username.to_string()).and_modify(|data| {
            data.enrolled_methods.retain(|m| m != method);

            match method {
                "totp" => data.totp_secret = None,
                "sms" => data.sms_phone = None,
                "webauthn" => data.webauthn_credentials.clear(),
                _ => warn!("Unknown MFA method: {}", method),
            }

            data.updated_at = chrono::Utc::now();
        });

        self.save_to_disk().await?;
        info!("Disabled MFA method '{}' for user: {}", method, username);
        Ok(())
    }

    /// Remove all MFA data for user
    pub async fn remove_user(&self, username: &str) -> FusekiResult<()> {
        self.cache.remove(username);
        self.save_to_disk().await?;
        info!("Removed all MFA data for user: {}", username);
        Ok(())
    }

    /// Get complete MFA data for user
    pub async fn get_user_data(&self, username: &str) -> FusekiResult<Option<UserMfaData>> {
        Ok(self.cache.get(username).map(|data| data.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_totp_secret_storage() {
        let storage = MfaStorage::new(None);
        let username = "testuser";
        let secret = "JBSWY3DPEHPK3PXP";

        storage.store_totp_secret(username, secret).await.unwrap();
        let retrieved = storage.get_totp_secret(username).await.unwrap();
        assert_eq!(retrieved, Some(secret.to_string()));
    }

    #[tokio::test]
    async fn test_backup_codes() {
        let storage = MfaStorage::new(None);
        let username = "testuser";
        let codes = vec!["ABC123".to_string(), "DEF456".to_string()];

        storage.store_backup_codes(username, codes.clone()).await.unwrap();
        let retrieved = storage.get_backup_codes(username).await.unwrap();
        assert_eq!(retrieved, codes);

        // Test code verification and consumption
        let verified = storage.verify_backup_code(username, "ABC123").await.unwrap();
        assert!(verified);

        let remaining = storage.get_backup_codes(username).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], "DEF456");
    }

    #[tokio::test]
    async fn test_email_storage() {
        let storage = MfaStorage::new(None);
        let username = "testuser";
        let email = "test@example.com";

        storage.store_email(username, email).await.unwrap();
        let retrieved = storage.get_email(username).await.unwrap();
        assert_eq!(retrieved, Some(email.to_string()));
    }

    #[tokio::test]
    async fn test_enrolled_methods() {
        let storage = MfaStorage::new(None);
        let username = "testuser";

        storage.store_totp_secret(username, "SECRET").await.unwrap();
        storage.store_sms_phone(username, "+1234567890").await.unwrap();

        let methods = storage.get_enrolled_methods(username).await.unwrap();
        assert_eq!(methods.len(), 2);
        assert!(methods.contains(&"totp".to_string()));
        assert!(methods.contains(&"sms".to_string()));
    }
}
