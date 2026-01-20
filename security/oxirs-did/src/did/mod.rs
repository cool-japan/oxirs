//! DID (Decentralized Identifiers) module

pub mod document;
pub mod methods;
pub mod resolver;

pub use document::DidDocument;
pub use methods::DidKeyMethod;
#[cfg(feature = "did-web")]
pub use methods::DidWebMethod;
pub use resolver::DidResolver;

use crate::{DidError, DidResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// DID identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Did(String);

impl Did {
    /// Create a new DID from string
    pub fn new(did: &str) -> DidResult<Self> {
        Self::validate(did)?;
        Ok(Self(did.to_string()))
    }

    /// Create did:key from Ed25519 public key
    pub fn new_key_ed25519(public_key: &[u8]) -> DidResult<Self> {
        if public_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 public key must be 32 bytes".to_string(),
            ));
        }

        // Multicodec prefix for Ed25519 public key: 0xed01
        let mut prefixed = vec![0xed, 0x01];
        prefixed.extend_from_slice(public_key);

        // Base58btc encode with 'z' prefix
        let encoded = bs58::encode(&prefixed).into_string();
        Ok(Self(format!("did:key:z{}", encoded)))
    }

    /// Create did:web from domain and optional path
    pub fn new_web(domain: &str, path: Option<&str>) -> DidResult<Self> {
        let encoded_domain = domain.replace(':', "%3A").replace('/', "%2F");
        let did = if let Some(p) = path {
            let encoded_path = p.replace('/', ":");
            format!("did:web:{}:{}", encoded_domain, encoded_path)
        } else {
            format!("did:web:{}", encoded_domain)
        };
        Ok(Self(did))
    }

    /// Validate DID format
    fn validate(did: &str) -> DidResult<()> {
        if !did.starts_with("did:") {
            return Err(DidError::InvalidFormat(
                "DID must start with 'did:'".to_string(),
            ));
        }

        let parts: Vec<&str> = did.splitn(3, ':').collect();
        if parts.len() < 3 {
            return Err(DidError::InvalidFormat(
                "DID must have at least method and method-specific-id".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the DID method (e.g., "key", "web")
    pub fn method(&self) -> &str {
        self.0.split(':').nth(1).unwrap_or("unknown")
    }

    /// Get the method-specific identifier
    pub fn method_specific_id(&self) -> &str {
        let parts: Vec<&str> = self.0.splitn(3, ':').collect();
        parts.get(2).unwrap_or(&"")
    }

    /// Get the full DID string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Create a key reference ID from this DID
    pub fn key_id(&self, fragment: &str) -> String {
        format!("{}#{}", self.0, fragment)
    }
}

impl fmt::Display for Did {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Did> for String {
    fn from(did: Did) -> String {
        did.0
    }
}

impl AsRef<str> for Did {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_did_new() {
        let did = Did::new("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK").unwrap();
        assert_eq!(did.method(), "key");
        assert!(did.method_specific_id().starts_with("z6Mk"));
    }

    #[test]
    fn test_did_key_ed25519() {
        let public_key = [0u8; 32];
        let did = Did::new_key_ed25519(&public_key).unwrap();

        assert_eq!(did.method(), "key");
        assert!(did.as_str().starts_with("did:key:z"));
    }

    #[test]
    fn test_did_web() {
        let did = Did::new_web("example.com", None).unwrap();
        assert_eq!(did.as_str(), "did:web:example.com");

        let did_with_path = Did::new_web("example.com", Some("users/alice")).unwrap();
        assert_eq!(did_with_path.as_str(), "did:web:example.com:users:alice");
    }

    #[test]
    fn test_invalid_did() {
        assert!(Did::new("not-a-did").is_err());
        assert!(Did::new("did:").is_err());
        assert!(Did::new("did:key").is_err());
    }

    #[test]
    fn test_key_id() {
        let did = Did::new("did:key:z6Mk...").unwrap();
        assert_eq!(did.key_id("key-1"), "did:key:z6Mk...#key-1");
    }
}
