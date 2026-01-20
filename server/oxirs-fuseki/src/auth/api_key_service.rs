//! API Key Management Service
//!
//! This module provides core API key management functionality including:
//! - Secure key generation using SciRS2-core's cryptographic random
//! - Key rotation with automatic revocation
//! - Key expiration and lifecycle management
//! - Usage tracking and analytics
//! - Key validation and authentication

use crate::auth::types::Permission;
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// API key with full details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Hashed key value (never store plaintext)
    pub key_hash: String,
    /// Owner username
    pub owner: String,
    /// Permissions granted
    pub permissions: Vec<Permission>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,
    /// Last used timestamp
    pub last_used_at: Option<DateTime<Utc>>,
    /// Usage count
    pub usage_count: u64,
    /// Is active (not revoked)
    pub is_active: bool,
    /// Key metadata
    pub metadata: ApiKeyMetadata,
}

/// API key metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyMetadata {
    /// Description
    pub description: Option<String>,
    /// Allowed IP addresses
    pub allowed_ips: Vec<String>,
    /// Allowed CIDR ranges
    pub allowed_cidrs: Vec<String>,
    /// Rate limit (requests per minute)
    pub rate_limit: Option<u32>,
    /// Tags for organization
    pub tags: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// API key creation parameters
#[derive(Debug, Clone)]
pub struct CreateApiKeyParams {
    pub name: String,
    pub owner: String,
    pub permissions: Vec<Permission>,
    pub expires_in: Option<Duration>,
    pub metadata: ApiKeyMetadata,
}

/// API key rotation result
#[derive(Debug, Clone)]
pub struct ApiKeyRotation {
    /// New key ID
    pub new_key_id: String,
    /// New plaintext key (returned only once)
    pub new_key: String,
    /// Old key ID (now revoked)
    pub old_key_id: String,
    /// Rotation timestamp
    pub rotated_at: DateTime<Utc>,
}

/// API key usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyUsageStats {
    pub key_id: String,
    pub owner: String,
    pub total_uses: u64,
    pub last_used: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub age_days: i64,
    pub uses_per_day: f64,
}

/// API key service for managing authentication keys
pub struct ApiKeyService {
    /// Active API keys (key_id -> ApiKey)
    keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    /// Key hash to ID mapping for lookup
    hash_to_id: Arc<RwLock<HashMap<String, String>>>,
    /// Usage tracking (key_id -> usage timestamps)
    usage_history: Arc<RwLock<HashMap<String, Vec<DateTime<Utc>>>>>,
    /// Key prefix for identification
    key_prefix: String,
}

impl ApiKeyService {
    /// Create new API key service
    pub fn new(key_prefix: impl Into<String>) -> Self {
        ApiKeyService {
            keys: Arc::new(RwLock::new(HashMap::new())),
            hash_to_id: Arc::new(RwLock::new(HashMap::new())),
            usage_history: Arc::new(RwLock::new(HashMap::new())),
            key_prefix: key_prefix.into(),
        }
    }

    /// Generate a new API key
    pub async fn create_key(&self, params: CreateApiKeyParams) -> FusekiResult<(String, ApiKey)> {
        // Generate secure random key
        let raw_key = self.generate_secure_key().await;

        // Hash the key for storage
        let key_hash = self.hash_key(&raw_key);

        // Create key ID
        let key_id = uuid::Uuid::new_v4().to_string();

        // Calculate expiration
        let expires_at = params.expires_in.map(|duration| Utc::now() + duration);

        let api_key = ApiKey {
            id: key_id.clone(),
            name: params.name,
            key_hash: key_hash.clone(),
            owner: params.owner,
            permissions: params.permissions,
            created_at: Utc::now(),
            expires_at,
            last_used_at: None,
            usage_count: 0,
            is_active: true,
            metadata: params.metadata,
        };

        // Store the key
        let mut keys = self.keys.write().await;
        let mut hash_map = self.hash_to_id.write().await;

        keys.insert(key_id.clone(), api_key.clone());
        hash_map.insert(key_hash, key_id.clone());

        info!("Created API key: {} for owner: {}", key_id, api_key.owner);

        // Return raw key and API key info (raw key shown only once)
        Ok((raw_key, api_key))
    }

    /// Generate secure random API key using cryptographically secure random
    async fn generate_secure_key(&self) -> String {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

        // Generate 32 bytes of cryptographically secure random data using UUIDs
        let uuid1 = uuid::Uuid::new_v4();
        let uuid2 = uuid::Uuid::new_v4();

        let random_bytes: Vec<u8> = uuid1
            .as_bytes()
            .iter()
            .chain(uuid2.as_bytes().iter())
            .copied() // Copy bytes instead of referencing
            .collect();

        let random_part = URL_SAFE_NO_PAD.encode(&random_bytes[..32]);

        format!("{}{}", self.key_prefix, random_part)
    }

    /// Hash API key using SHA-256
    fn hash_key(&self, key: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Validate and authenticate an API key
    pub async fn validate_key(&self, raw_key: &str) -> FusekiResult<ApiKey> {
        // Hash the provided key
        let key_hash = self.hash_key(raw_key);

        // Look up key ID
        let key_id = {
            let hash_map = self.hash_to_id.read().await;
            hash_map
                .get(&key_hash)
                .cloned()
                .ok_or_else(|| FusekiError::authentication("Invalid API key"))?
        };

        // Get the key
        let mut key = {
            let keys = self.keys.read().await;
            keys.get(&key_id)
                .cloned()
                .ok_or_else(|| FusekiError::authentication("API key not found"))?
        };

        // Check if key is active
        if !key.is_active {
            return Err(FusekiError::authentication("API key has been revoked"));
        }

        // Check expiration
        if let Some(expires_at) = key.expires_at {
            if Utc::now() > expires_at {
                // Auto-revoke expired key
                self.revoke_key(&key_id).await?;
                return Err(FusekiError::authentication("API key has expired"));
            }
        }

        // Update usage statistics
        key.last_used_at = Some(Utc::now());
        key.usage_count += 1;

        // Update in storage
        {
            let mut keys = self.keys.write().await;
            if let Some(stored_key) = keys.get_mut(&key_id) {
                stored_key.last_used_at = key.last_used_at;
                stored_key.usage_count = key.usage_count;
            }
        }

        // Track usage
        {
            let mut history = self.usage_history.write().await;
            history
                .entry(key_id.clone())
                .or_insert_with(Vec::new)
                .push(Utc::now());
        }

        debug!("API key validated: {}", key_id);

        Ok(key)
    }

    /// Rotate an API key (create new, revoke old)
    pub async fn rotate_key(&self, key_id: &str) -> FusekiResult<ApiKeyRotation> {
        // Get existing key
        let old_key = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .cloned()
                .ok_or_else(|| FusekiError::not_found("API key not found"))?
        };

        // Create new key with same parameters
        let (new_raw_key, new_key) = self
            .create_key(CreateApiKeyParams {
                name: format!("{} (rotated)", old_key.name),
                owner: old_key.owner.clone(),
                permissions: old_key.permissions.clone(),
                expires_in: old_key.expires_at.map(|exp| exp - Utc::now()),
                metadata: old_key.metadata.clone(),
            })
            .await?;

        // Revoke old key
        self.revoke_key(key_id).await?;

        info!("Rotated API key: {} -> {}", key_id, new_key.id);

        Ok(ApiKeyRotation {
            new_key_id: new_key.id,
            new_key: new_raw_key,
            old_key_id: key_id.to_string(),
            rotated_at: Utc::now(),
        })
    }

    /// Revoke an API key
    pub async fn revoke_key(&self, key_id: &str) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;

        let key = keys
            .get_mut(key_id)
            .ok_or_else(|| FusekiError::not_found("API key not found"))?;

        key.is_active = false;

        info!("Revoked API key: {}", key_id);

        Ok(())
    }

    /// Delete an API key completely
    pub async fn delete_key(&self, key_id: &str) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;
        let mut hash_map = self.hash_to_id.write().await;

        let key = keys
            .remove(key_id)
            .ok_or_else(|| FusekiError::not_found("API key not found"))?;

        hash_map.remove(&key.key_hash);

        // Remove usage history
        {
            let mut history = self.usage_history.write().await;
            history.remove(key_id);
        }

        info!("Deleted API key: {}", key_id);

        Ok(())
    }

    /// Get all keys for an owner
    pub async fn get_keys_for_owner(&self, owner: &str) -> Vec<ApiKey> {
        let keys = self.keys.read().await;
        keys.values()
            .filter(|key| key.owner == owner)
            .cloned()
            .collect()
    }

    /// Get key by ID
    pub async fn get_key(&self, key_id: &str) -> Option<ApiKey> {
        let keys = self.keys.read().await;
        keys.get(key_id).cloned()
    }

    /// Get usage statistics for a key
    pub async fn get_usage_stats(&self, key_id: &str) -> FusekiResult<ApiKeyUsageStats> {
        let key = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .cloned()
                .ok_or_else(|| FusekiError::not_found("API key not found"))?
        };

        let age_days = (Utc::now() - key.created_at).num_days();
        let uses_per_day = if age_days > 0 {
            key.usage_count as f64 / age_days as f64
        } else {
            key.usage_count as f64
        };

        Ok(ApiKeyUsageStats {
            key_id: key.id,
            owner: key.owner,
            total_uses: key.usage_count,
            last_used: key.last_used_at,
            created_at: key.created_at,
            age_days,
            uses_per_day,
        })
    }

    /// Cleanup expired keys
    pub async fn cleanup_expired_keys(&self) -> usize {
        let mut keys = self.keys.write().await;
        let now = Utc::now();

        let expired_keys: Vec<String> = keys
            .iter()
            .filter_map(|(id, key)| {
                if let Some(expires_at) = key.expires_at {
                    if now > expires_at && key.is_active {
                        Some(id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        let count = expired_keys.len();

        for key_id in expired_keys {
            if let Some(key) = keys.get_mut(&key_id) {
                key.is_active = false;
            }
        }

        if count > 0 {
            info!("Auto-revoked {} expired API keys", count);
        }

        count
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Every hour

            loop {
                interval.tick().await;
                self.cleanup_expired_keys().await;
            }
        });
    }

    /// Get total key count
    pub async fn get_key_count(&self) -> usize {
        let keys = self.keys.read().await;
        keys.len()
    }

    /// Get active key count
    pub async fn get_active_key_count(&self) -> usize {
        let keys = self.keys.read().await;
        keys.values().filter(|k| k.is_active).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> CreateApiKeyParams {
        CreateApiKeyParams {
            name: "Test Key".to_string(),
            owner: "test_user".to_string(),
            permissions: vec![Permission::GlobalRead, Permission::SparqlQuery],
            expires_in: Some(Duration::days(30)),
            metadata: ApiKeyMetadata {
                description: Some("Test API key".to_string()),
                allowed_ips: vec![],
                allowed_cidrs: vec![],
                rate_limit: None,
                tags: vec![],
                attributes: HashMap::new(),
            },
        }
    }

    #[tokio::test]
    async fn test_create_key() {
        let service = ApiKeyService::new("oxirs_");

        let (raw_key, api_key) = service.create_key(create_test_params()).await.unwrap();

        assert!(raw_key.starts_with("oxirs_"));
        assert_eq!(api_key.owner, "test_user");
        assert!(api_key.is_active);
    }

    #[tokio::test]
    async fn test_validate_key() {
        let service = ApiKeyService::new("oxirs_");

        let (raw_key, _) = service.create_key(create_test_params()).await.unwrap();

        // Validate with correct key
        let result = service.validate_key(&raw_key).await;
        assert!(result.is_ok());

        // Validate with incorrect key
        let result = service.validate_key("invalid_key").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_revoke_key() {
        let service = ApiKeyService::new("oxirs_");

        let (raw_key, api_key) = service.create_key(create_test_params()).await.unwrap();

        // Revoke the key
        service.revoke_key(&api_key.id).await.unwrap();

        // Should fail to validate revoked key
        let result = service.validate_key(&raw_key).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rotate_key() {
        let service = ApiKeyService::new("oxirs_");

        let (old_raw_key, old_key) = service.create_key(create_test_params()).await.unwrap();

        // Rotate the key
        let rotation = service.rotate_key(&old_key.id).await.unwrap();

        // Old key should be revoked
        let result = service.validate_key(&old_raw_key).await;
        assert!(result.is_err());

        // New key should work
        let result = service.validate_key(&rotation.new_key).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_usage_tracking() {
        let service = ApiKeyService::new("oxirs_");

        let (raw_key, api_key) = service.create_key(create_test_params()).await.unwrap();

        // Use the key multiple times
        for _ in 0..5 {
            service.validate_key(&raw_key).await.unwrap();
        }

        // Check usage stats
        let stats = service.get_usage_stats(&api_key.id).await.unwrap();
        assert_eq!(stats.total_uses, 5);
    }

    #[tokio::test]
    async fn test_expired_key() {
        let service = ApiKeyService::new("oxirs_");

        let mut params = create_test_params();
        params.expires_in = Some(Duration::seconds(-1)); // Already expired

        let (raw_key, _) = service.create_key(params).await.unwrap();

        // Should fail due to expiration
        let result = service.validate_key(&raw_key).await;
        assert!(result.is_err());
    }
}
