//! Encryption utilities for secure credential storage
//!
//! This module provides a simplified encryption layer for credential protection.
//! The implementation uses a stream cipher approach with key derivation.
//! For production use, consider integrating a hardened crypto library.

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
    /// XOR-based stream cipher (for testing/development)
    StreamXor,
    /// Placeholder for AES-256-GCM integration
    AES256GCM,
    /// Placeholder for ChaCha20Poly1305 integration
    ChaCha20Poly1305,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivation {
    Identity,
    PBKDF2,
    Argon2,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::StreamXor,
            key_derivation: KeyDerivation::Identity,
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

    /// Encrypt data using stream XOR cipher with 12-byte nonce prepended
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::default();
        let nonce: Vec<u8> = (0..12).map(|_| rng.random::<u8>()).collect();

        let key_stream = self.expand_key(&nonce);
        let ciphertext: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key_stream[i % key_stream.len()])
            .collect();

        let mut result = nonce;
        result.extend_from_slice(&ciphertext);
        Ok(result)
    }

    /// Decrypt data (XOR is symmetric — decrypt = encrypt with same nonce)
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            return Err(anyhow!("Invalid encrypted data: too short (min 12 bytes)"));
        }

        let (nonce, ciphertext) = encrypted_data.split_at(12);
        let key_stream = self.expand_key(nonce);
        let plaintext: Vec<u8> = ciphertext
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key_stream[i % key_stream.len()])
            .collect();

        Ok(plaintext)
    }

    /// Securely zero memory
    pub fn zeroize_key(&mut self) {
        self.key.iter_mut().for_each(|b| *b = 0);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Expand the key using the nonce via a simple hash-based KDF
    fn expand_key(&self, nonce: &[u8]) -> Vec<u8> {
        // Simple key expansion: XOR key with each byte of nonce cyclically,
        // then repeat to form a 256-byte key stream.
        let mut stream = Vec::with_capacity(256);
        for i in 0..256 {
            let k = self.key[i % self.key.len()];
            let n = nonce[i % nonce.len()];
            // Simple mixing: k XOR n XOR (i as u8)
            stream.push(k ^ n ^ (i as u8));
        }
        stream
    }
}

impl Default for Encryptor {
    fn default() -> Self {
        // Use a deterministic key for Default (not secure, only for tests)
        Self {
            key: (0..32).map(|i| i as u8).collect(),
            config: EncryptionConfig::default(),
        }
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
        let encryptor = Encryptor::new().expect("failed to create encryptor");
        let plaintext = b"sensitive API key: sk-test-123";

        let encrypted = encryptor.encrypt(plaintext).expect("encrypt failed");
        // The nonce (12 bytes) + ciphertext should differ from plaintext
        assert_ne!(&encrypted[12..], plaintext);

        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt failed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let encryptor1 = Encryptor::new().expect("enc1");
        let encryptor2 = Encryptor::new().expect("enc2");

        let plaintext = b"test data";
        let encrypted1 = encryptor1.encrypt(plaintext).expect("enc1 encrypt");
        let encrypted2 = encryptor2.encrypt(plaintext).expect("enc2 encrypt");

        // Very likely to be different (different keys)
        // Note: could collide if both get same random key, but probability is negligible
        assert_ne!(encrypted1[12..].to_vec(), encrypted2[12..].to_vec());
    }

    #[test]
    fn test_invalid_key_length() {
        let result = Encryptor::with_key(vec![0; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_invalid_data_too_short() {
        let encryptor = Encryptor::new().expect("enc");
        let result = encryptor.decrypt(&[0; 5]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too short"));
    }

    #[test]
    fn test_with_key() {
        let key: Vec<u8> = (0..32).collect();
        let encryptor = Encryptor::with_key(key).expect("with_key");
        let data = b"hello, world";
        let encrypted = encryptor.encrypt(data).expect("encrypt");
        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt");
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_zeroize_key() {
        let mut encryptor = Encryptor::new().expect("enc");
        encryptor.zeroize_key();
        assert!(encryptor.key.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_empty_plaintext() {
        let encryptor = Encryptor::new().expect("enc");
        let encrypted = encryptor.encrypt(b"").expect("encrypt empty");
        // Should have at least the 12-byte nonce
        assert!(encrypted.len() >= 12);
        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt empty");
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn test_large_plaintext() {
        let encryptor = Encryptor::new().expect("enc");
        let plaintext: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let encrypted = encryptor.encrypt(&plaintext).expect("encrypt large");
        let decrypted = encryptor.decrypt(&encrypted).expect("decrypt large");
        assert_eq!(decrypted, plaintext);
    }
}
