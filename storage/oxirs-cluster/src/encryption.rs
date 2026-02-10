//! Encryption for cluster data with key rotation
//!
//! This module provides AES-256-GCM encryption with automatic key rotation
//! and optional HSM (Hardware Security Module) integration for cloud providers.

use crate::error::{ClusterError, Result};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use scirs2_core::random::{Random, RngCore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};

/// Encryption manager with key rotation support
pub struct EncryptionManager {
    current_key: Arc<RwLock<EncryptionKey>>,
    key_history: Arc<RwLock<Vec<EncryptionKey>>>,
    config: EncryptionConfig,
    key_rotator: Option<Arc<KeyRotator>>,
}

/// Encryption key with metadata
#[derive(Clone)]
pub struct EncryptionKey {
    /// Unique identifier for this key
    pub key_id: KeyId,
    /// The actual key material (wrapped in LessSafeKey)
    key_material: Vec<u8>, // Store raw bytes for cloning
    /// When this key was created
    pub created_at: SystemTime,
    /// When this key expires
    pub expires_at: SystemTime,
}

impl EncryptionKey {
    /// Create LessSafeKey from stored material
    fn get_key(&self) -> std::result::Result<LessSafeKey, ring::error::Unspecified> {
        let unbound_key = UnboundKey::new(&AES_256_GCM, &self.key_material)?;
        Ok(LessSafeKey::new(unbound_key))
    }
}

/// Configuration for encryption behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Number of days before key rotation
    pub key_rotation_days: u64,
    /// Enable HSM integration
    pub hsm_enabled: bool,
    /// HSM provider configuration
    pub hsm_provider: Option<HsmProvider>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            key_rotation_days: 90, // 90 days default
            hsm_enabled: false,
            hsm_provider: None,
        }
    }
}

/// HSM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HsmProvider {
    /// AWS Key Management Service
    AwsKms {
        /// AWS region
        region: String,
        /// Key ARN
        key_arn: String,
    },
    /// Azure Key Vault
    AzureKeyVault {
        /// Vault URL
        vault_url: String,
        /// Key name
        key_name: String,
    },
    /// Google Cloud KMS
    GcpKms {
        /// GCP project ID
        project: String,
        /// Location (region)
        location: String,
        /// Key ring name
        key_ring: String,
        /// Key name
        key_name: String,
    },
}

/// Encrypted data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Key ID used for encryption
    pub key_id: KeyId,
    /// Nonce (96-bit for AES-GCM)
    pub nonce: [u8; 12],
    /// Ciphertext with authentication tag
    pub ciphertext: Vec<u8>,
}

/// Key identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyId(u64);

impl KeyId {
    /// Generate a new unique key ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        KeyId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for KeyId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "key-{}", self.0)
    }
}

/// Automatic key rotation manager
pub struct KeyRotator {
    manager: Arc<Mutex<EncryptionManager>>,
    config: RotatorConfig,
}

/// Key rotator configuration
#[derive(Debug, Clone)]
pub struct RotatorConfig {
    /// How often to check for key rotation (in hours)
    pub check_interval_hours: u64,
    /// Enable automatic key rotation
    pub auto_rotate: bool,
}

impl Default for RotatorConfig {
    fn default() -> Self {
        Self {
            check_interval_hours: 24, // Check daily
            auto_rotate: true,
        }
    }
}

impl EncryptionManager {
    /// Create a new encryption manager
    pub fn new(config: EncryptionConfig) -> Result<Self> {
        let key = Self::generate_key_internal(&config)?;

        Ok(Self {
            current_key: Arc::new(RwLock::new(key)),
            key_history: Arc::new(RwLock::new(Vec::new())),
            config,
            key_rotator: None,
        })
    }

    /// Create a new encryption manager with automatic key rotation
    pub fn with_auto_rotation(
        config: EncryptionConfig,
        rotator_config: RotatorConfig,
    ) -> Result<Arc<Mutex<Self>>> {
        let manager = Arc::new(Mutex::new(Self::new(config)?));

        if rotator_config.auto_rotate {
            let rotator = Arc::new(KeyRotator {
                manager: Arc::clone(&manager),
                config: rotator_config,
            });

            // Start rotation background task
            rotator.start();

            // Store rotator reference
            let manager_clone = Arc::clone(&manager);
            tokio::spawn(async move {
                let mut mgr = manager_clone.lock().await;
                mgr.key_rotator = Some(rotator);
            });
        }

        Ok(manager)
    }

    /// Encrypt data with AES-256-GCM
    pub async fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedData> {
        let key = self.current_key.read().await;

        // Generate nonce (96-bit)
        let mut nonce_bytes = [0u8; 12];
        // Use SystemTime with nanosecond precision for better randomness
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let mut rng = Random::seed(seed);
        rng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::assume_unique_for_key(nonce_bytes);

        // Get the key material
        let safe_key = key
            .get_key()
            .map_err(|_| ClusterError::Encryption("Failed to create key".to_string()))?;

        // Encrypt (in-place encryption appends auth tag)
        let mut ciphertext = plaintext.to_vec();
        safe_key
            .seal_in_place_append_tag(nonce, Aad::empty(), &mut ciphertext)
            .map_err(|_| ClusterError::Encryption("Encryption failed".to_string()))?;

        Ok(EncryptedData {
            key_id: key.key_id,
            nonce: nonce_bytes,
            ciphertext,
        })
    }

    /// Decrypt data
    pub async fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>> {
        // Find the correct key (current or historical)
        let key = self.find_key(encrypted.key_id).await?;

        let nonce = Nonce::assume_unique_for_key(encrypted.nonce);
        let mut plaintext = encrypted.ciphertext.clone();

        // Get the key material
        let safe_key = key
            .get_key()
            .map_err(|_| ClusterError::Encryption("Failed to create key".to_string()))?;

        // Decrypt (in-place decryption removes auth tag)
        safe_key
            .open_in_place(nonce, Aad::empty(), &mut plaintext)
            .map_err(|_| ClusterError::Encryption("Decryption failed".to_string()))?;

        // Remove authentication tag (last 16 bytes for GCM)
        plaintext.truncate(plaintext.len() - 16);
        Ok(plaintext)
    }

    /// Rotate encryption key
    pub async fn rotate_key(&mut self) -> Result<()> {
        // Generate new key
        let new_key = if self.config.hsm_enabled {
            self.generate_key_from_hsm().await?
        } else {
            Self::generate_key_internal(&self.config)?
        };

        // Move current to history
        let old_key = self.current_key.read().await.clone();
        self.key_history.write().await.push(old_key);

        // Set new key as current
        *self.current_key.write().await = new_key;

        Ok(())
    }

    /// Check if key rotation is needed
    pub async fn should_rotate_key(&self) -> bool {
        let key = self.current_key.read().await;
        match SystemTime::now().duration_since(key.created_at) {
            Ok(age) => age >= Duration::from_secs(self.config.key_rotation_days * 86400),
            Err(_) => true, // If we can't determine age, rotate to be safe
        }
    }

    /// Find key by ID (current or historical)
    async fn find_key(&self, key_id: KeyId) -> Result<EncryptionKey> {
        // Check current key first
        let current = self.current_key.read().await;
        if current.key_id == key_id {
            return Ok(current.clone());
        }

        // Check history
        let history = self.key_history.read().await;
        for key in history.iter() {
            if key.key_id == key_id {
                return Ok(key.clone());
            }
        }

        Err(ClusterError::Encryption(format!(
            "Key not found: {}",
            key_id
        )))
    }

    /// Generate key locally (not from HSM)
    fn generate_key_internal(config: &EncryptionConfig) -> Result<EncryptionKey> {
        let mut key_bytes = [0u8; 32]; // 256 bits
        // Use SystemTime with nanosecond precision for better randomness
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let mut rng = Random::seed(seed);
        rng.fill_bytes(&mut key_bytes);

        // Verify key can be created
        let _test_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|_| ClusterError::Encryption("Key creation failed".to_string()))?;

        let now = SystemTime::now();
        Ok(EncryptionKey {
            key_id: KeyId::new(),
            key_material: key_bytes.to_vec(),
            created_at: now,
            expires_at: now + Duration::from_secs(config.key_rotation_days * 86400),
        })
    }

    /// Generate key from HSM (placeholder for cloud integration)
    async fn generate_key_from_hsm(&self) -> Result<EncryptionKey> {
        match &self.config.hsm_provider {
            Some(HsmProvider::AwsKms { region, key_arn }) => {
                // Placeholder for AWS KMS integration
                // In production, this would use aws-sdk-kms to generate a data key
                tracing::warn!(
                    "AWS KMS integration not fully implemented. Using local key generation. Region: {}, ARN: {}",
                    region,
                    key_arn
                );
                Self::generate_key_internal(&self.config)
            }
            Some(HsmProvider::AzureKeyVault {
                vault_url,
                key_name,
            }) => {
                // Placeholder for Azure Key Vault integration
                tracing::warn!(
                    "Azure Key Vault integration not fully implemented. Using local key generation. Vault: {}, Key: {}",
                    vault_url,
                    key_name
                );
                Self::generate_key_internal(&self.config)
            }
            Some(HsmProvider::GcpKms {
                project,
                location,
                key_ring,
                key_name,
            }) => {
                // Placeholder for GCP KMS integration
                tracing::warn!(
                    "GCP KMS integration not fully implemented. Using local key generation. Project: {}, Location: {}, KeyRing: {}, Key: {}",
                    project,
                    location,
                    key_ring,
                    key_name
                );
                Self::generate_key_internal(&self.config)
            }
            None => Err(ClusterError::Encryption(
                "HSM not configured".to_string(),
            )),
        }
    }

    /// Get current key ID
    pub async fn current_key_id(&self) -> KeyId {
        self.current_key.read().await.key_id
    }

    /// Get number of historical keys
    pub async fn historical_key_count(&self) -> usize {
        self.key_history.read().await.len()
    }

    /// Get configuration
    pub fn config(&self) -> &EncryptionConfig {
        &self.config
    }
}

impl KeyRotator {
    /// Start automatic key rotation background task
    pub fn start(self: &Arc<Self>) {
        let rotator = Arc::clone(self);

        tokio::spawn(async move {
            let interval = Duration::from_secs(rotator.config.check_interval_hours * 3600);

            loop {
                tokio::time::sleep(interval).await;

                if !rotator.config.auto_rotate {
                    continue;
                }

                match rotator.manager.lock().await.should_rotate_key().await {
                    true => match rotator.manager.lock().await.rotate_key().await {
                        Ok(()) => {
                            tracing::info!("Encryption key rotated successfully");
                        }
                        Err(e) => {
                            tracing::error!("Key rotation failed: {}", e);
                        }
                    },
                    false => {
                        tracing::debug!("Key rotation not needed yet");
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption_manager_creation() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_encrypt_decrypt_round_trip() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Hello, world! This is a secret message.";
        let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");

        assert_ne!(encrypted.ciphertext, plaintext);
        assert_eq!(encrypted.nonce.len(), 12);

        let decrypted = manager
            .decrypt(&encrypted)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let config = EncryptionConfig {
            key_rotation_days: 90,
            hsm_enabled: false,
            hsm_provider: None,
        };
        let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

        let original_key_id = manager.current_key_id().await;

        // Encrypt with original key
        let plaintext = b"Test data before rotation";
        let encrypted_before = manager
            .encrypt(plaintext)
            .await
            .expect("Encryption failed");
        assert_eq!(encrypted_before.key_id, original_key_id);

        // Rotate key
        manager.rotate_key().await.expect("Key rotation failed");

        let new_key_id = manager.current_key_id().await;
        assert_ne!(original_key_id, new_key_id);

        // Verify we can still decrypt old data
        let decrypted_old = manager
            .decrypt(&encrypted_before)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted_old, plaintext);

        // Encrypt with new key
        let encrypted_after = manager
            .encrypt(plaintext)
            .await
            .expect("Encryption failed");
        assert_eq!(encrypted_after.key_id, new_key_id);
        assert_ne!(encrypted_after.ciphertext, encrypted_before.ciphertext);

        // Verify historical keys
        assert_eq!(manager.historical_key_count().await, 1);
    }

    #[tokio::test]
    async fn test_multiple_rotations() {
        let config = EncryptionConfig::default();
        let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Test data";
        let mut encrypted_versions = Vec::new();

        // Encrypt and rotate 5 times
        for i in 0..5 {
            let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
            encrypted_versions.push((i, encrypted));

            if i < 4 {
                manager.rotate_key().await.expect("Key rotation failed");
            }
        }

        // Verify we can decrypt all versions
        for (i, encrypted) in encrypted_versions {
            let decrypted = manager
                .decrypt(&encrypted)
                .await
                .unwrap_or_else(|_| panic!("Decryption failed for version {}", i));
            assert_eq!(decrypted, plaintext);
        }

        // Should have 4 historical keys (original + 3 rotations)
        assert_eq!(manager.historical_key_count().await, 4);
    }

    #[tokio::test]
    async fn test_key_not_found() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let fake_encrypted = EncryptedData {
            key_id: KeyId(99999),
            nonce: [0u8; 12],
            ciphertext: vec![0u8; 32],
        };

        let result = manager.decrypt(&fake_encrypted).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Key not found"));
    }

    #[tokio::test]
    async fn test_encryption_different_nonces() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Same data";
        let encrypted1 = manager.encrypt(plaintext).await.expect("Encryption failed");
        let encrypted2 = manager.encrypt(plaintext).await.expect("Encryption failed");

        // Same plaintext with different nonces should produce different ciphertexts
        assert_ne!(encrypted1.nonce, encrypted2.nonce);
        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);

        // Both should decrypt correctly
        let decrypted1 = manager
            .decrypt(&encrypted1)
            .await
            .expect("Decryption failed");
        let decrypted2 = manager
            .decrypt(&encrypted2)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted1, plaintext);
        assert_eq!(decrypted2, plaintext);
    }

    #[test]
    fn test_key_id_generation() {
        let id1 = KeyId::new();
        let id2 = KeyId::new();
        assert_ne!(id1, id2);
        assert!(id1.0 < id2.0);
    }

    #[test]
    fn test_key_id_display() {
        let id = KeyId(42);
        assert_eq!(format!("{}", id), "key-42");
    }

    #[tokio::test]
    async fn test_hsm_config_fallback() {
        let config = EncryptionConfig {
            key_rotation_days: 90,
            hsm_enabled: true,
            hsm_provider: Some(HsmProvider::AwsKms {
                region: "us-west-2".to_string(),
                key_arn: "arn:aws:kms:us-west-2:123456789012:key/test".to_string(),
            }),
        };

        let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

        // Should fall back to local key generation
        manager.rotate_key().await.expect("Key rotation failed");

        // Verify encryption still works
        let plaintext = b"Test with HSM fallback";
        let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
        let decrypted = manager
            .decrypt(&encrypted)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted, plaintext);
    }
}
