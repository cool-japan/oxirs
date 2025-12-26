//! did:key Method implementation
//!
//! did:key is a self-certifying DID method that encodes the public key
//! directly in the DID identifier. No network resolution is required.

use super::DidMethod;
use crate::did::{Did, DidDocument};
use crate::{DidError, DidResult};
use async_trait::async_trait;

/// did:key method resolver
pub struct DidKeyMethod;

impl Default for DidKeyMethod {
    fn default() -> Self {
        Self::new()
    }
}

impl DidKeyMethod {
    pub fn new() -> Self {
        Self
    }

    /// Extract public key from did:key identifier
    pub fn extract_public_key(&self, did: &Did) -> DidResult<(KeyType, Vec<u8>)> {
        let method_specific_id = did.method_specific_id();

        // Must start with 'z' (base58btc multibase prefix)
        if !method_specific_id.starts_with('z') {
            return Err(DidError::InvalidFormat(
                "did:key identifier must start with 'z'".to_string(),
            ));
        }

        // Decode base58btc (skip the 'z' prefix)
        let decoded = bs58::decode(&method_specific_id[1..])
            .into_vec()
            .map_err(|e| DidError::InvalidFormat(format!("Base58 decode failed: {}", e)))?;

        if decoded.len() < 2 {
            return Err(DidError::InvalidFormat("Decoded key too short".to_string()));
        }

        // Parse multicodec prefix
        let (key_type, key_bytes) = self.parse_multicodec(&decoded)?;

        Ok((key_type, key_bytes))
    }

    /// Parse multicodec prefix to determine key type
    fn parse_multicodec(&self, data: &[u8]) -> DidResult<(KeyType, Vec<u8>)> {
        if data.len() < 2 {
            return Err(DidError::InvalidFormat(
                "Data too short for multicodec".to_string(),
            ));
        }

        // Check for Ed25519 public key (0xed, 0x01)
        if data[0] == 0xed && data[1] == 0x01 {
            if data.len() != 34 {
                return Err(DidError::InvalidFormat(format!(
                    "Ed25519 public key must be 34 bytes (2 prefix + 32 key), got {}",
                    data.len()
                )));
            }
            return Ok((KeyType::Ed25519, data[2..].to_vec()));
        }

        // Check for X25519 public key (0xec, 0x01)
        if data[0] == 0xec && data[1] == 0x01 {
            if data.len() != 34 {
                return Err(DidError::InvalidFormat(
                    "X25519 public key must be 34 bytes".to_string(),
                ));
            }
            return Ok((KeyType::X25519, data[2..].to_vec()));
        }

        // Check for secp256k1 public key (0xe7, 0x01)
        if data[0] == 0xe7 && data[1] == 0x01 {
            return Ok((KeyType::Secp256k1, data[2..].to_vec()));
        }

        // Check for P-256 public key (0x80, 0x24)
        if data[0] == 0x80 && data[1] == 0x24 {
            return Ok((KeyType::P256, data[2..].to_vec()));
        }

        Err(DidError::InvalidFormat(format!(
            "Unknown multicodec prefix: 0x{:02x}{:02x}",
            data[0], data[1]
        )))
    }
}

/// Supported key types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Ed25519,
    X25519,
    Secp256k1,
    P256,
}

impl KeyType {
    /// Get the verification method type string
    pub fn verification_method_type(&self) -> &'static str {
        match self {
            KeyType::Ed25519 => "Ed25519VerificationKey2020",
            KeyType::X25519 => "X25519KeyAgreementKey2020",
            KeyType::Secp256k1 => "EcdsaSecp256k1VerificationKey2019",
            KeyType::P256 => "JsonWebKey2020",
        }
    }
}

#[async_trait]
impl DidMethod for DidKeyMethod {
    fn method_name(&self) -> &str {
        "key"
    }

    async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        if !self.supports(did) {
            return Err(DidError::UnsupportedMethod(did.method().to_string()));
        }

        let (key_type, public_key) = self.extract_public_key(did)?;

        match key_type {
            KeyType::Ed25519 => DidDocument::from_key_ed25519(&public_key),
            _ => Err(DidError::UnsupportedMethod(format!(
                "Key type {:?} not yet fully supported",
                key_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_did_key() {
        let method = DidKeyMethod::new();

        // Create a did:key from known public key
        let public_key = [0u8; 32];
        let did = Did::new_key_ed25519(&public_key).unwrap();

        // Resolve synchronously for testing
        let rt = tokio::runtime::Runtime::new().unwrap();
        let doc = rt.block_on(method.resolve(&did)).unwrap();

        assert_eq!(doc.id, did);
        assert_eq!(doc.verification_method.len(), 1);
    }

    #[test]
    fn test_extract_public_key() {
        let method = DidKeyMethod::new();

        let public_key = [1u8; 32];
        let did = Did::new_key_ed25519(&public_key).unwrap();

        let (key_type, extracted) = method.extract_public_key(&did).unwrap();
        assert_eq!(key_type, KeyType::Ed25519);
        assert_eq!(extracted, public_key.to_vec());
    }

    #[test]
    fn test_invalid_did_key() {
        let method = DidKeyMethod::new();

        let did = Did::new("did:key:invalid").unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(method.resolve(&did));

        assert!(result.is_err());
    }
}
