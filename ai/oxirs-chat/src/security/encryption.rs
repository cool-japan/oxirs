//! Encryption utilities for secure credential storage
//!
//! Uses AES-256-GCM for authenticated encryption, which is already
//! available in the workspace via the `aes-gcm` crate.

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: EncryptionAlgorithm,
    pub key_derivation: KeyDerivation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES256GCM,
    ChaCha20Poly1305,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivation {
    PBKDF2,
    Argon2,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::AES256GCM,
            key_derivation: KeyDerivation::Argon2,
        }
    }
}

/// Encryptor for credential protection using AES-256-GCM
pub struct Encryptor {
    key: Vec<u8>,
    config: EncryptionConfig,
}

impl Encryptor {
    /// Create new encryptor with random key
    pub fn new() -> Result<Self> {
        use scirs2_core::random::Random;

        let mut rng = Random::default();
        let key: Vec<u8> = (0..32).map(|_| rng.gen_range(0u8..=255u8)).collect();

        Ok(Self {
            key,
            config: EncryptionConfig::default(),
        })
    }

    /// Create encryptor with specific key (must be 32 bytes)
    pub fn with_key(key: Vec<u8>) -> Result<Self> {
        if key.len() != 32 {
            return Err(anyhow!("Key must be 32 bytes for AES-256-GCM"));
        }

        Ok(Self {
            key,
            config: EncryptionConfig::default(),
        })
    }

    /// Encrypt data using AES-256-GCM
    ///
    /// Returns the ciphertext with the 12-byte nonce prepended.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let key = Key::<Aes256Gcm>::from_slice(&self.key);
        let cipher = Aes256Gcm::new(key);

        // Generate random 12-byte nonce
        use scirs2_core::random::Random;
        let mut rng = Random::default();
        let nonce_bytes: Vec<u8> = (0..12).map(|_| rng.gen_range(0u8..=255u8)).collect();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow!("AES-256-GCM encryption failed: {}", e))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes;
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data using AES-256-GCM
    ///
    /// Expects the nonce (12 bytes) prepended to the ciphertext.
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            return Err(anyhow!(
                "Invalid encrypted data: too short (need at least 12 bytes for nonce)"
            ));
        }

        let key = Key::<Aes256Gcm>::from_slice(&self.key);
        let cipher = Aes256Gcm::new(key);

        // Extract 12-byte nonce
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("AES-256-GCM decryption failed: {}", e))?;

        Ok(plaintext)
    }

    /// Securely zero memory on key material
    pub fn zeroize_key(&mut self) {
        self.key.iter_mut().for_each(|b| *b = 0);
    }

    /// Get the encryption algorithm in use
    pub fn algorithm(&self) -> &EncryptionAlgorithm {
        &self.config.algorithm
    }
}

impl Default for Encryptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default Encryptor")
    }
}

impl Drop for Encryptor {
    fn drop(&mut self) {
        self.zeroize_key();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_decryption() {
        let encryptor = Encryptor::new().expect("create encryptor");
        let plaintext = b"sensitive API key: sk-test-123";

        let encrypted = encryptor.encrypt(plaintext).expect("encrypt");
        // Ciphertext (after nonce) should be different from plaintext
        assert_ne!(&encrypted[12..], plaintext);

        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let encryptor1 = Encryptor::new().expect("create encryptor 1");
        let encryptor2 = Encryptor::new().expect("create encryptor 2");

        let plaintext = b"test data";
        let encrypted1 = encryptor1.encrypt(plaintext).expect("encrypt 1");
        let encrypted2 = encryptor2.encrypt(plaintext).expect("encrypt 2");

        // Different keys should produce different ciphertexts
        assert_ne!(encrypted1, encrypted2);
    }

    #[test]
    fn test_invalid_key_length() {
        let result = Encryptor::with_key(vec![0; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_invalid_data_too_short() {
        let encryptor = Encryptor::new().expect("create encryptor");
        let result = encryptor.decrypt(&[0; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_tampered_data() {
        let encryptor = Encryptor::new().expect("create encryptor");
        let plaintext = b"hello world";
        let mut encrypted = encryptor.encrypt(plaintext).expect("encrypt");
        // Tamper with ciphertext (after nonce)
        if let Some(byte) = encrypted.get_mut(15) {
            *byte ^= 0xFF;
        }
        let result = encryptor.decrypt(&encrypted);
        assert!(
            result.is_err(),
            "Tampered ciphertext should fail decryption"
        );
    }

    #[test]
    fn test_encrypt_empty_data() {
        let encryptor = Encryptor::new().expect("create encryptor");
        let plaintext: &[u8] = b"";

        let encrypted = encryptor.encrypt(plaintext).expect("encrypt empty");
        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt empty");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_32_byte_key() {
        let key = vec![0x42u8; 32];
        let encryptor = Encryptor::with_key(key).expect("create with key");
        let plaintext = b"test with fixed key";
        let encrypted = encryptor.encrypt(plaintext).expect("encrypt");
        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }
}
