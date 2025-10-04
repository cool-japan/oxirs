//! Encryption utilities for secure credential storage

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
            algorithm: EncryptionAlgorithm::ChaCha20Poly1305,
            key_derivation: KeyDerivation::Argon2,
        }
    }
}

/// Encryptor for credential protection
pub struct Encryptor {
    key: Vec<u8>,
    config: EncryptionConfig,
}

impl Encryptor {
    /// Create new encryptor with random key
    pub fn new() -> Result<Self> {
        use scirs2_core::random::{Random, Rng};

        let mut rng = Random::default();
        let key: Vec<u8> = (0..32).map(|_| rng.random::<u8>()).collect();

        Ok(Self {
            key,
            config: EncryptionConfig::default(),
        })
    }

    /// Create encryptor with specific key
    pub fn with_key(key: Vec<u8>) -> Result<Self> {
        if key.len() != 32 {
            return Err(anyhow!("Key must be 32 bytes"));
        }

        Ok(Self {
            key,
            config: EncryptionConfig::default(),
        })
    }

    /// Encrypt data
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };

        let cipher = ChaCha20Poly1305::new_from_slice(&self.key)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

        // Generate random nonce
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::default();
        let nonce_bytes: Vec<u8> = (0..12).map(|_| rng.random::<u8>()).collect();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes;
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };

        if encrypted_data.len() < 12 {
            return Err(anyhow!("Invalid encrypted data: too short"));
        }

        let cipher = ChaCha20Poly1305::new_from_slice(&self.key)
            .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

        // Extract nonce
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;

        Ok(plaintext)
    }

    /// Securely zero memory
    pub fn zeroize_key(&mut self) {
        self.key.iter_mut().for_each(|b| *b = 0);
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
        let encryptor = Encryptor::new().unwrap();
        let plaintext = b"sensitive API key: sk-test-123";

        let encrypted = encryptor.encrypt(plaintext).unwrap();
        assert_ne!(&encrypted[12..], plaintext); // Ensure it's actually encrypted

        let decrypted = encryptor.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let encryptor1 = Encryptor::new().unwrap();
        let encryptor2 = Encryptor::new().unwrap();

        let plaintext = b"test data";
        let encrypted1 = encryptor1.encrypt(plaintext).unwrap();
        let encrypted2 = encryptor2.encrypt(plaintext).unwrap();

        // Different keys should produce different ciphertexts
        assert_ne!(encrypted1, encrypted2);
    }

    #[test]
    fn test_invalid_key_length() {
        let result = Encryptor::with_key(vec![0; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_invalid_data() {
        let encryptor = Encryptor::new().unwrap();
        let result = encryptor.decrypt(&[0; 5]);
        assert!(result.is_err());
    }
}
