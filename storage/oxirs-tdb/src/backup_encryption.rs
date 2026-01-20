//! Backup Encryption for Data at Rest
//!
//! Production Security Feature: AES-256-GCM Encrypted Backups
//!
//! This module provides encryption capabilities for database backups:
//! - AES-256-GCM authenticated encryption
//! - PBKDF2 key derivation from passwords
//! - Secure random nonce generation
//! - Encrypted backup metadata
//! - Key rotation support
//! - Compliance with security best practices
//!
//! ## Security Properties
//!
//! - **Encryption**: AES-256-GCM provides confidentiality and authenticity
//! - **Key Derivation**: PBKDF2-SHA256 with 600,000 iterations (OWASP 2023)
//! - **Authentication**: GCM tag prevents tampering
//! - **Nonces**: Unique 96-bit nonces per encryption
//! - **Salt**: Random 32-byte salt for each backup
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_tdb::backup_encryption::{BackupEncryption, EncryptionConfig};
//!
//! // Create encryption manager
//! let config = EncryptionConfig::new("strong-password");
//! let encryption = BackupEncryption::new(config)?;
//!
//! // Encrypt backup data
//! let ciphertext = encryption.encrypt(b"sensitive backup data")?;
//!
//! // Decrypt backup data
//! let plaintext = encryption.decrypt(&ciphertext)?;
//! ```

use crate::error::{Result, TdbError};
use aes_gcm::{
    aead::{rand_core::RngCore, Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use std::time::SystemTime;

/// PBKDF2 iteration count (OWASP 2023 recommendation for PBKDF2-SHA256)
const PBKDF2_ITERATIONS: u32 = 600_000;

/// Salt size in bytes
const SALT_SIZE: usize = 32;

/// Nonce size for AES-GCM (96 bits)
const NONCE_SIZE: usize = 12;

/// AES-256 key size
const KEY_SIZE: usize = 32;

/// Encryption configuration
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Password for key derivation
    password: String,
    /// PBKDF2 iteration count
    iterations: u32,
    /// Enable compression before encryption
    compress_before_encrypt: bool,
}

impl EncryptionConfig {
    /// Create new encryption config with password
    pub fn new(password: impl Into<String>) -> Self {
        Self {
            password: password.into(),
            iterations: PBKDF2_ITERATIONS,
            compress_before_encrypt: true,
        }
    }

    /// Set PBKDF2 iteration count
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Enable/disable compression before encryption
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.compress_before_encrypt = enable;
        self
    }
}

/// Encrypted data container
#[derive(Debug, Clone)]
pub struct EncryptedData {
    /// Salt used for key derivation
    pub salt: Vec<u8>,
    /// Nonce used for encryption
    pub nonce: Vec<u8>,
    /// Ciphertext + authentication tag
    pub ciphertext: Vec<u8>,
    /// Encryption timestamp
    pub encrypted_at: SystemTime,
    /// Whether data was compressed before encryption
    pub compressed: bool,
}

impl EncryptedData {
    /// Serialize encrypted data to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Format: [salt_len(4)][salt][nonce_len(4)][nonce][compressed(1)][ciphertext_len(4)][ciphertext]
        bytes.extend_from_slice(&(self.salt.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.salt);

        bytes.extend_from_slice(&(self.nonce.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.nonce);

        bytes.push(if self.compressed { 1 } else { 0 });

        bytes.extend_from_slice(&(self.ciphertext.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.ciphertext);

        bytes
    }

    /// Deserialize encrypted data from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Read salt
        if bytes.len() < offset + 4 {
            return Err(TdbError::Other(
                "Invalid encrypted data: too short".to_string(),
            ));
        }
        let salt_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if bytes.len() < offset + salt_len {
            return Err(TdbError::Other(
                "Invalid encrypted data: salt truncated".to_string(),
            ));
        }
        let salt = bytes[offset..offset + salt_len].to_vec();
        offset += salt_len;

        // Read nonce
        if bytes.len() < offset + 4 {
            return Err(TdbError::Other(
                "Invalid encrypted data: nonce missing".to_string(),
            ));
        }
        let nonce_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if bytes.len() < offset + nonce_len {
            return Err(TdbError::Other(
                "Invalid encrypted data: nonce truncated".to_string(),
            ));
        }
        let nonce = bytes[offset..offset + nonce_len].to_vec();
        offset += nonce_len;

        // Read compressed flag
        if bytes.len() < offset + 1 {
            return Err(TdbError::Other(
                "Invalid encrypted data: compressed flag missing".to_string(),
            ));
        }
        let compressed = bytes[offset] != 0;
        offset += 1;

        // Read ciphertext
        if bytes.len() < offset + 4 {
            return Err(TdbError::Other(
                "Invalid encrypted data: ciphertext length missing".to_string(),
            ));
        }
        let ciphertext_len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if bytes.len() < offset + ciphertext_len {
            return Err(TdbError::Other(
                "Invalid encrypted data: ciphertext truncated".to_string(),
            ));
        }
        let ciphertext = bytes[offset..offset + ciphertext_len].to_vec();

        Ok(Self {
            salt,
            nonce,
            ciphertext,
            encrypted_at: SystemTime::now(),
            compressed,
        })
    }
}

/// Backup encryption manager
pub struct BackupEncryption {
    /// Configuration
    config: EncryptionConfig,
}

impl BackupEncryption {
    /// Create new backup encryption manager
    pub fn new(config: EncryptionConfig) -> Result<Self> {
        if config.password.is_empty() {
            return Err(TdbError::Other("Password cannot be empty".to_string()));
        }

        if config.password.len() < 8 {
            return Err(TdbError::Other(
                "Password must be at least 8 characters".to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Encrypt data
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<EncryptedData> {
        // Generate random salt
        let mut salt = vec![0u8; SALT_SIZE];
        OsRng.fill_bytes(&mut salt);

        // Derive encryption key using PBKDF2
        let key = self.derive_key(&salt)?;

        // Create cipher
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| TdbError::Other(format!("Failed to create cipher: {}", e)))?;

        // Generate random nonce
        let mut nonce_bytes = vec![0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Optionally compress data before encryption
        let data_to_encrypt = if self.config.compress_before_encrypt {
            self.compress(plaintext)?
        } else {
            plaintext.to_vec()
        };

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, data_to_encrypt.as_ref())
            .map_err(|e| TdbError::Other(format!("Encryption failed: {}", e)))?;

        Ok(EncryptedData {
            salt,
            nonce: nonce_bytes,
            ciphertext,
            encrypted_at: SystemTime::now(),
            compressed: self.config.compress_before_encrypt,
        })
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>> {
        // Derive decryption key using the same salt
        let key = self.derive_key(&encrypted.salt)?;

        // Create cipher
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| TdbError::Other(format!("Failed to create cipher: {}", e)))?;

        // Create nonce
        let nonce = Nonce::from_slice(&encrypted.nonce);

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|e| {
                TdbError::Other(format!(
                    "Decryption failed (wrong password or corrupted data): {}",
                    e
                ))
            })?;

        // Decompress if needed
        if encrypted.compressed {
            self.decompress(&plaintext)
        } else {
            Ok(plaintext)
        }
    }

    /// Derive encryption key from password and salt using PBKDF2
    fn derive_key(&self, salt: &[u8]) -> Result<Vec<u8>> {
        let mut key = vec![0u8; KEY_SIZE];
        pbkdf2_hmac::<Sha256>(
            self.config.password.as_bytes(),
            salt,
            self.config.iterations,
            &mut key,
        );
        Ok(key)
    }

    /// Compress data using LZ4
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    /// Decompress data using LZ4
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| TdbError::Other(format!("Decompression failed: {}", e)))
    }

    /// Encrypt a file
    pub fn encrypt_file(
        &mut self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<()> {
        let plaintext = std::fs::read(input_path).map_err(TdbError::Io)?;
        let encrypted = self.encrypt(&plaintext)?;
        let bytes = encrypted.to_bytes();
        std::fs::write(output_path, bytes).map_err(TdbError::Io)?;
        Ok(())
    }

    /// Decrypt a file
    pub fn decrypt_file(
        &self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<()> {
        let bytes = std::fs::read(input_path).map_err(TdbError::Io)?;
        let encrypted = EncryptedData::from_bytes(&bytes)?;
        let plaintext = self.decrypt(&encrypted)?;
        std::fs::write(output_path, plaintext).map_err(TdbError::Io)?;
        Ok(())
    }

    /// Change password (re-encrypt with new password)
    pub fn change_password(
        &mut self,
        encrypted: &EncryptedData,
        new_password: impl Into<String>,
    ) -> Result<EncryptedData> {
        // Decrypt with old password
        let plaintext = self.decrypt(encrypted)?;

        // Update password
        self.config.password = new_password.into();

        // Encrypt with new password
        self.encrypt(&plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Helper to create test config with reduced iterations for faster tests
    fn test_config(password: impl Into<String>) -> EncryptionConfig {
        EncryptionConfig::new(password).with_iterations(1000) // Reduced from 600,000 for tests
    }

    #[test]
    fn test_encryption_roundtrip() {
        let config = test_config("test-password-123");
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = b"Sensitive backup data that needs encryption";
        let encrypted = encryption.encrypt(plaintext).unwrap();
        let decrypted = encryption.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext.as_ref(), decrypted.as_slice());
    }

    #[test]
    fn test_wrong_password_fails() {
        let config1 = test_config("password1");
        let mut encryption1 = BackupEncryption::new(config1).unwrap();

        let config2 = test_config("password2");
        let encryption2 = BackupEncryption::new(config2).unwrap();

        let plaintext = b"Secret data";
        let encrypted = encryption1.encrypt(plaintext).unwrap();

        // Decryption with wrong password should fail
        let result = encryption2.decrypt(&encrypted);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = test_config("serialize-test");
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = b"Data to serialize";
        let encrypted = encryption.encrypt(plaintext).unwrap();

        // Serialize and deserialize
        let bytes = encrypted.to_bytes();
        let deserialized = EncryptedData::from_bytes(&bytes).unwrap();

        // Decrypt deserialized data
        let decrypted = encryption.decrypt(&deserialized).unwrap();

        assert_eq!(plaintext.as_ref(), decrypted.as_slice());
    }

    #[test]
    fn test_file_encryption() {
        let temp_dir = env::temp_dir().join("oxirs_tdb_encryption_test");
        std::fs::remove_dir_all(&temp_dir).ok();
        std::fs::create_dir_all(&temp_dir).unwrap();

        let plain_file = temp_dir.join("plaintext.dat");
        let encrypted_file = temp_dir.join("encrypted.dat");
        let decrypted_file = temp_dir.join("decrypted.dat");

        // Write plaintext
        let plaintext = b"Confidential backup data for encryption test";
        std::fs::write(&plain_file, plaintext).unwrap();

        let config = test_config("file-encryption-key");
        let mut encryption = BackupEncryption::new(config).unwrap();

        // Encrypt file
        encryption
            .encrypt_file(&plain_file, &encrypted_file)
            .unwrap();

        // Verify encrypted file is different
        let encrypted_content = std::fs::read(&encrypted_file).unwrap();
        assert_ne!(plaintext.as_ref(), encrypted_content.as_slice());

        // Decrypt file
        encryption
            .decrypt_file(&encrypted_file, &decrypted_file)
            .unwrap();

        // Verify decrypted matches original
        let decrypted_content = std::fs::read(&decrypted_file).unwrap();
        assert_eq!(plaintext.as_ref(), decrypted_content.as_slice());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_compression_before_encryption() {
        let config = test_config("compress-test").with_compression(true);
        let mut encryption = BackupEncryption::new(config).unwrap();

        // Highly compressible data
        let plaintext = vec![b'A'; 1000];
        let encrypted = encryption.encrypt(&plaintext).unwrap();

        assert!(encrypted.compressed);

        // Encrypted + compressed should be smaller than plaintext
        assert!(encrypted.ciphertext.len() < plaintext.len());

        // Decryption should restore original
        let decrypted = encryption.decrypt(&encrypted).unwrap();
        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_password_change() {
        let config = test_config("old-password");
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = b"Data encrypted with old password";
        let encrypted_old = encryption.encrypt(plaintext).unwrap();

        // Change password
        let encrypted_new = encryption
            .change_password(&encrypted_old, "new-password")
            .unwrap();

        // Verify old password can't decrypt new encryption
        let config_old = test_config("old-password");
        let encryption_old = BackupEncryption::new(config_old).unwrap();
        let result = encryption_old.decrypt(&encrypted_new);
        assert!(result.is_err());

        // Verify new password can decrypt
        let decrypted = encryption.decrypt(&encrypted_new).unwrap();
        assert_eq!(plaintext.as_ref(), decrypted.as_slice());
    }

    #[test]
    fn test_weak_password_rejected() {
        let config = EncryptionConfig::new("short");
        let result = BackupEncryption::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_password_rejected() {
        let config = EncryptionConfig::new("");
        let result = BackupEncryption::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_encryptions_unique() {
        let config = test_config("uniqueness-test");
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = b"Same plaintext";
        let encrypted1 = encryption.encrypt(plaintext).unwrap();
        let encrypted2 = encryption.encrypt(plaintext).unwrap();

        // Different salts and nonces should produce different ciphertexts
        assert_ne!(encrypted1.salt, encrypted2.salt);
        assert_ne!(encrypted1.nonce, encrypted2.nonce);
        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);

        // Both should decrypt to same plaintext
        let decrypted1 = encryption.decrypt(&encrypted1).unwrap();
        let decrypted2 = encryption.decrypt(&encrypted2).unwrap();
        assert_eq!(decrypted1, decrypted2);
        assert_eq!(plaintext.as_ref(), decrypted1.as_slice());
    }

    #[test]
    fn test_large_data_encryption() {
        let config = test_config("large-data-test");
        let mut encryption = BackupEncryption::new(config).unwrap();

        // 1MB of data
        let plaintext = vec![0xAB; 1024 * 1024];
        let encrypted = encryption.encrypt(&plaintext).unwrap();
        let decrypted = encryption.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_custom_iterations() {
        let config = test_config("iterations-test").with_iterations(10_000); // Reduced from 100k for faster tests
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = b"Test with custom iterations";
        let encrypted = encryption.encrypt(plaintext).unwrap();
        let decrypted = encryption.decrypt(&encrypted).unwrap();

        assert_eq!(plaintext.as_ref(), decrypted.as_slice());
    }

    #[test]
    fn test_encryption_without_compression() {
        let config = test_config("no-compression").with_compression(false);
        let mut encryption = BackupEncryption::new(config).unwrap();

        let plaintext = vec![b'A'; 1000];
        let encrypted = encryption.encrypt(&plaintext).unwrap();

        assert!(!encrypted.compressed);

        let decrypted = encryption.decrypt(&encrypted).unwrap();
        assert_eq!(plaintext, decrypted);
    }
}
