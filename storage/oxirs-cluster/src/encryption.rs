//! Encryption for cluster data with key rotation
//!
//! This module provides AES-256-GCM encryption with automatic key rotation
//! and optional HSM (Hardware Security Module) integration for cloud providers.

use crate::error::{ClusterError, Result};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use scirs2_core::metrics::{Counter, Histogram};
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
    metrics: EncryptionMetrics,
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
    /// Enable encryption validation
    pub enable_validation: bool,
    /// Require encryption for all data
    pub require_encryption: bool,
    /// Allowed encryption algorithms
    pub allowed_algorithms: Vec<EncryptionAlgorithm>,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            key_rotation_days: 90, // 90 days default
            hsm_enabled: false,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        }
    }
}

/// Encryption validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Encryption operation metrics
#[derive(Clone)]
struct EncryptionMetrics {
    encryption_operations: Arc<Counter>,
    decryption_operations: Arc<Counter>,
    encryption_time: Arc<Histogram>,
    decryption_time: Arc<Histogram>,
    encryption_size: Arc<Histogram>,
    key_rotation_count: Arc<Counter>,
    validation_failures: Arc<Counter>,
}

impl EncryptionMetrics {
    fn new() -> Self {
        Self {
            encryption_operations: Arc::new(Counter::new(
                "encryption_operations_total".to_string(),
            )),
            decryption_operations: Arc::new(Counter::new(
                "decryption_operations_total".to_string(),
            )),
            encryption_time: Arc::new(Histogram::new(
                "encryption_duration_microseconds".to_string(),
            )),
            decryption_time: Arc::new(Histogram::new(
                "decryption_duration_microseconds".to_string(),
            )),
            encryption_size: Arc::new(Histogram::new("encryption_data_size_bytes".to_string())),
            key_rotation_count: Arc::new(Counter::new("key_rotations_total".to_string())),
            validation_failures: Arc::new(Counter::new(
                "encryption_validation_failures".to_string(),
            )),
        }
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct EncryptionMetricsSnapshot {
    pub encryption_operations: u64,
    pub decryption_operations: u64,
    pub avg_encryption_time_us: f64,
    pub avg_decryption_time_us: f64,
    pub avg_data_size_bytes: f64,
    pub key_rotation_count: u64,
    pub validation_failures: u64,
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
            metrics: EncryptionMetrics::new(),
        })
    }

    /// Validate encryption configuration at startup
    pub fn validate_config(&self) -> ValidationResult {
        let mut result = ValidationResult::success();

        // Validate key rotation period
        if self.config.key_rotation_days == 0 {
            result.add_error("Key rotation days cannot be zero".to_string());
        } else if self.config.key_rotation_days > 365 {
            result.add_warning(format!(
                "Key rotation period ({} days) exceeds recommended maximum (365 days)",
                self.config.key_rotation_days
            ));
        }

        // Validate HSM configuration
        if self.config.hsm_enabled && self.config.hsm_provider.is_none() {
            result.add_error("HSM enabled but no provider configured".to_string());
        }

        // Validate encryption algorithms
        if self.config.allowed_algorithms.is_empty() {
            result.add_error("No encryption algorithms configured".to_string());
        }

        // Validate current algorithm is in allowed list
        if self.config.enable_validation {
            let current_algo = EncryptionAlgorithm::Aes256Gcm; // Current implementation
            if !self.config.allowed_algorithms.contains(&current_algo) {
                result.add_error(format!(
                    "Current algorithm {:?} not in allowed algorithms list",
                    current_algo
                ));
            }
        }

        if !result.is_valid {
            self.metrics.validation_failures.inc();
        }

        result
    }

    /// Validate that data can be encrypted/decrypted
    pub async fn validate_encryption_roundtrip(&self) -> Result<ValidationResult> {
        let test_data = b"encryption_validation_test_data";

        match self.encrypt(test_data).await {
            Ok(encrypted) => match self.decrypt(&encrypted).await {
                Ok(decrypted) => {
                    if decrypted == test_data {
                        Ok(ValidationResult::success())
                    } else {
                        Ok(ValidationResult::failure(
                            "Decrypted data does not match original".to_string(),
                        ))
                    }
                }
                Err(e) => Ok(ValidationResult::failure(format!(
                    "Decryption failed: {}",
                    e
                ))),
            },
            Err(e) => Ok(ValidationResult::failure(format!(
                "Encryption failed: {}",
                e
            ))),
        }
    }

    /// Validate key is not expired
    pub async fn validate_current_key(&self) -> ValidationResult {
        let mut result = ValidationResult::success();
        let key = self.current_key.read().await;

        if SystemTime::now() > key.expires_at {
            result.add_error("Current encryption key has expired".to_string());
        } else {
            let remaining = key
                .expires_at
                .duration_since(SystemTime::now())
                .unwrap_or(Duration::from_secs(0));

            if remaining < Duration::from_secs(7 * 86400) {
                // 7 days
                result.add_warning(format!(
                    "Current encryption key expires in {} days",
                    remaining.as_secs() / 86400
                ));
            }
        }

        result
    }

    /// Get encryption metrics
    pub fn get_metrics(&self) -> EncryptionMetricsSnapshot {
        EncryptionMetricsSnapshot {
            encryption_operations: self.metrics.encryption_operations.get(),
            decryption_operations: self.metrics.decryption_operations.get(),
            avg_encryption_time_us: self.metrics.encryption_time.get_stats().mean,
            avg_decryption_time_us: self.metrics.decryption_time.get_stats().mean,
            avg_data_size_bytes: self.metrics.encryption_size.get_stats().mean,
            key_rotation_count: self.metrics.key_rotation_count.get(),
            validation_failures: self.metrics.validation_failures.get(),
        }
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
        let start = std::time::Instant::now();

        // Record encryption operation
        self.metrics.encryption_operations.inc();

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

        // Record metrics
        let elapsed = start.elapsed();
        self.metrics
            .encryption_time
            .observe(elapsed.as_micros() as f64);
        self.metrics.encryption_size.observe(plaintext.len() as f64);

        Ok(EncryptedData {
            key_id: key.key_id,
            nonce: nonce_bytes,
            ciphertext,
        })
    }

    /// Decrypt data
    pub async fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        // Record decryption operation
        self.metrics.decryption_operations.inc();

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

        // Record metrics
        let elapsed = start.elapsed();
        self.metrics
            .decryption_time
            .observe(elapsed.as_micros() as f64);

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

        // Record key rotation metric
        self.metrics.key_rotation_count.inc();

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
            None => Err(ClusterError::Encryption("HSM not configured".to_string())),
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
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        };
        let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

        let original_key_id = manager.current_key_id().await;

        // Encrypt with original key
        let plaintext = b"Test data before rotation";
        let encrypted_before = manager.encrypt(plaintext).await.expect("Encryption failed");
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
        let encrypted_after = manager.encrypt(plaintext).await.expect("Encryption failed");
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
        assert!(result.unwrap_err().to_string().contains("Key not found"));
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
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
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

    // ======================================================================
    // COMPREHENSIVE VALIDATION TESTS
    // ======================================================================

    #[tokio::test]
    async fn test_validation_config_zero_rotation_days() {
        let config = EncryptionConfig {
            key_rotation_days: 0,
            hsm_enabled: false,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        };

        let manager = EncryptionManager::new(config).expect("Failed to create manager");
        let result = manager.validate_config();

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("rotation days cannot be zero")));
    }

    #[tokio::test]
    async fn test_validation_config_excessive_rotation_period() {
        let config = EncryptionConfig {
            key_rotation_days: 400,
            hsm_enabled: false,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        };

        let manager = EncryptionManager::new(config).expect("Failed to create manager");
        let result = manager.validate_config();

        assert!(result.is_valid); // Valid but with warning
        assert!(!result.warnings.is_empty());
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("exceeds recommended maximum")));
    }

    #[tokio::test]
    async fn test_validation_config_hsm_enabled_no_provider() {
        let config = EncryptionConfig {
            key_rotation_days: 90,
            hsm_enabled: true,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        };

        let manager = EncryptionManager::new(config).expect("Failed to create manager");
        let result = manager.validate_config();

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("HSM enabled but no provider")));
    }

    #[tokio::test]
    async fn test_validation_config_no_algorithms() {
        let config = EncryptionConfig {
            key_rotation_days: 90,
            hsm_enabled: false,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![],
        };

        let manager = EncryptionManager::new(config).expect("Failed to create manager");
        let result = manager.validate_config();

        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("No encryption algorithms")));
    }

    #[tokio::test]
    async fn test_validation_encryption_roundtrip() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let result = manager
            .validate_encryption_roundtrip()
            .await
            .expect("Validation failed");

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_validation_current_key_not_expired() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let result = manager.validate_current_key().await;

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_validation_result_add_error() {
        let mut result = ValidationResult::success();
        assert!(result.is_valid);

        result.add_error("Test error".to_string());

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0], "Test error");
    }

    #[tokio::test]
    async fn test_validation_result_add_warning() {
        let mut result = ValidationResult::success();
        assert!(result.is_valid);

        result.add_warning("Test warning".to_string());

        assert!(result.is_valid); // Warnings don't affect validity
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.warnings[0], "Test warning");
    }

    #[tokio::test]
    async fn test_validation_result_failure() {
        let result = ValidationResult::failure("Critical error".to_string());

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0], "Critical error");
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Test data for metrics";

        // Perform multiple operations
        for _ in 0..5 {
            let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
            manager
                .decrypt(&encrypted)
                .await
                .expect("Decryption failed");
        }

        let metrics = manager.get_metrics();

        assert_eq!(metrics.encryption_operations, 5);
        assert_eq!(metrics.decryption_operations, 5);
        assert!(metrics.avg_encryption_time_us >= 0.0);
        assert!(metrics.avg_decryption_time_us >= 0.0);
        assert!(metrics.avg_data_size_bytes > 0.0);
    }

    #[tokio::test]
    async fn test_metrics_key_rotation_count() {
        let config = EncryptionConfig::default();
        let mut manager = EncryptionManager::new(config).expect("Failed to create manager");

        let initial_metrics = manager.get_metrics();
        assert_eq!(initial_metrics.key_rotation_count, 0);

        // Rotate key twice
        manager.rotate_key().await.expect("Key rotation failed");
        manager.rotate_key().await.expect("Key rotation failed");

        let final_metrics = manager.get_metrics();
        assert_eq!(final_metrics.key_rotation_count, 2);
    }

    #[tokio::test]
    async fn test_nonce_uniqueness() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Test data";
        let mut nonces = std::collections::HashSet::new();

        // Generate 100 encrypted messages and ensure all nonces are unique
        for _ in 0..100 {
            let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");
            assert!(nonces.insert(encrypted.nonce), "Duplicate nonce detected!");
        }

        assert_eq!(nonces.len(), 100);
    }

    #[tokio::test]
    async fn test_empty_plaintext_encryption() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"";
        let encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");

        // Even empty plaintext should produce ciphertext with auth tag
        assert!(!encrypted.ciphertext.is_empty());

        let decrypted = manager
            .decrypt(&encrypted)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted, plaintext);
    }

    #[tokio::test]
    async fn test_large_plaintext_encryption() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        // Create 1MB of data
        let plaintext = vec![0xAB; 1024 * 1024];
        let encrypted = manager
            .encrypt(&plaintext)
            .await
            .expect("Encryption failed");

        let decrypted = manager
            .decrypt(&encrypted)
            .await
            .expect("Decryption failed");
        assert_eq!(decrypted, plaintext);

        // Verify metrics recorded the size
        let metrics = manager.get_metrics();
        assert!(metrics.avg_data_size_bytes > 1000000.0);
    }

    #[tokio::test]
    async fn test_tampered_ciphertext_detection() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Secret data";
        let mut encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");

        // Tamper with the ciphertext
        if let Some(byte) = encrypted.ciphertext.first_mut() {
            *byte ^= 0xFF;
        }

        // Decryption should fail due to authentication failure
        let result = manager.decrypt(&encrypted).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));
    }

    #[tokio::test]
    async fn test_tampered_nonce_detection() {
        let config = EncryptionConfig::default();
        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        let plaintext = b"Secret data";
        let mut encrypted = manager.encrypt(plaintext).await.expect("Encryption failed");

        // Tamper with the nonce
        encrypted.nonce[0] ^= 0xFF;

        // Decryption should fail due to authentication failure
        let result = manager.decrypt(&encrypted).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_key_expiration_warning() {
        let config = EncryptionConfig {
            key_rotation_days: 1, // 1 day for fast expiration testing
            hsm_enabled: false,
            hsm_provider: None,
            enable_validation: true,
            require_encryption: true,
            allowed_algorithms: vec![EncryptionAlgorithm::Aes256Gcm],
        };

        let manager = EncryptionManager::new(config).expect("Failed to create manager");

        // Key should not be expired yet
        let result = manager.validate_current_key().await;
        assert!(result.is_valid);
    }

    #[test]
    fn test_encryption_config_default() {
        let config = EncryptionConfig::default();

        assert_eq!(config.key_rotation_days, 90);
        assert!(!config.hsm_enabled);
        assert!(config.hsm_provider.is_none());
        assert!(config.enable_validation);
        assert!(config.require_encryption);
        assert_eq!(config.allowed_algorithms.len(), 1);
        assert_eq!(config.allowed_algorithms[0], EncryptionAlgorithm::Aes256Gcm);
    }

    #[test]
    fn test_key_id_uniqueness() {
        let mut ids = std::collections::HashSet::new();

        for _ in 0..1000 {
            let id = KeyId::new();
            assert!(ids.insert(id), "Duplicate KeyId generated!");
        }

        assert_eq!(ids.len(), 1000);
    }

    #[tokio::test]
    async fn test_concurrent_encryption() {
        let config = EncryptionConfig::default();
        let manager =
            std::sync::Arc::new(EncryptionManager::new(config).expect("Failed to create manager"));

        let mut handles = vec![];

        // Spawn 10 concurrent encryption tasks
        for i in 0..10 {
            let manager_clone = std::sync::Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                let plaintext = format!("Test data {}", i);
                let encrypted = manager_clone
                    .encrypt(plaintext.as_bytes())
                    .await
                    .expect("Encryption failed");
                let decrypted = manager_clone
                    .decrypt(&encrypted)
                    .await
                    .expect("Decryption failed");
                assert_eq!(decrypted, plaintext.as_bytes());
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("Task panicked");
        }

        let metrics = manager.get_metrics();
        assert_eq!(metrics.encryption_operations, 10);
        assert_eq!(metrics.decryption_operations, 10);
    }
}
