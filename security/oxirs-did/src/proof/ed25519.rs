//! Ed25519 signature implementation

use crate::{DidError, DidResult};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// Ed25519 signer
pub struct Ed25519Signer {
    signing_key: SigningKey,
}

impl Ed25519Signer {
    /// Create from secret key bytes (32 bytes)
    pub fn from_bytes(secret_key: &[u8]) -> DidResult<Self> {
        if secret_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 secret key must be 32 bytes".to_string(),
            ));
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(secret_key);

        let signing_key = SigningKey::from_bytes(&bytes);
        Ok(Self { signing_key })
    }

    /// Generate a new random keypair
    pub fn generate() -> Self {
        use scirs2_core::random::Random;
        use scirs2_core::Rng;
        // Use system time as seed for cryptographic key generation
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut rng = Random::seed(seed);
        let bytes: [u8; 32] = rng.random();
        let signing_key = SigningKey::from_bytes(&bytes);
        Self { signing_key }
    }

    /// Get the secret key bytes
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Get the verifying key
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(message);
        signature.to_bytes().to_vec()
    }
}

/// Ed25519 verifier
pub struct Ed25519Verifier {
    verifying_key: VerifyingKey,
}

impl Ed25519Verifier {
    /// Create from public key bytes (32 bytes)
    pub fn from_bytes(public_key: &[u8]) -> DidResult<Self> {
        if public_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 public key must be 32 bytes".to_string(),
            ));
        }

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(public_key);

        let verifying_key =
            VerifyingKey::from_bytes(&bytes).map_err(|e| DidError::InvalidKey(e.to_string()))?;

        Ok(Self { verifying_key })
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> DidResult<bool> {
        if signature.len() != 64 {
            return Err(DidError::InvalidProof(
                "Ed25519 signature must be 64 bytes".to_string(),
            ));
        }

        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature);

        let signature = Signature::from_bytes(&sig_bytes);

        match self.verifying_key.verify(message, &signature) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Sign data with Ed25519
pub fn sign_ed25519(secret_key: &[u8], message: &[u8]) -> DidResult<Vec<u8>> {
    let signer = Ed25519Signer::from_bytes(secret_key)?;
    Ok(signer.sign(message))
}

/// Verify Ed25519 signature
pub fn verify_ed25519(public_key: &[u8], message: &[u8], signature: &[u8]) -> DidResult<bool> {
    let verifier = Ed25519Verifier::from_bytes(public_key)?;
    verifier.verify(message, signature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify() {
        let signer = Ed25519Signer::generate();
        let message = b"Hello, World!";

        let signature = signer.sign(message);
        assert_eq!(signature.len(), 64);

        let verifier = Ed25519Verifier::from_bytes(&signer.public_key_bytes()).unwrap();
        let valid = verifier.verify(message, &signature).unwrap();
        assert!(valid);

        // Verify with wrong message
        let invalid = verifier.verify(b"Wrong message", &signature).unwrap();
        assert!(!invalid);
    }

    #[test]
    fn test_from_bytes() {
        let secret = [42u8; 32];
        let signer = Ed25519Signer::from_bytes(&secret).unwrap();

        let public_key = signer.public_key_bytes();
        assert_eq!(public_key.len(), 32);
    }

    #[test]
    fn test_invalid_key_length() {
        let short_key = [0u8; 16];
        assert!(Ed25519Signer::from_bytes(&short_key).is_err());
        assert!(Ed25519Verifier::from_bytes(&short_key).is_err());
    }
}
