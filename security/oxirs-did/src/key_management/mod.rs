//! Key management module

use crate::proof::ed25519::Ed25519Signer;
use crate::{Did, DidError, DidResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Key store for managing signing keys
pub struct Keystore {
    /// In-memory key storage (encrypted in production)
    keys: Arc<RwLock<HashMap<String, StoredKey>>>,
}

/// Stored key data
struct StoredKey {
    /// Secret key bytes (encrypted in production)
    secret_key: Vec<u8>,
    /// Key type
    key_type: KeyType,
    /// Associated DID
    did: Did,
}

/// Supported key types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Ed25519,
}

impl Default for Keystore {
    fn default() -> Self {
        Self::new()
    }
}

impl Keystore {
    /// Create a new empty keystore
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a new Ed25519 keypair and store it
    pub async fn generate_ed25519(&self) -> DidResult<Did> {
        let signer = Ed25519Signer::generate();
        let public_key = signer.public_key_bytes();
        let secret_key = signer.secret_key_bytes();
        let did = Did::new_key_ed25519(&public_key)?;

        // Store the secret key
        // In production, this would be encrypted
        let stored = StoredKey {
            secret_key: secret_key.to_vec(),
            key_type: KeyType::Ed25519,
            did: did.clone(),
        };

        self.keys
            .write()
            .await
            .insert(did.as_str().to_string(), stored);

        Ok(did)
    }

    /// Import an Ed25519 secret key
    pub async fn import_ed25519(&self, secret_key: &[u8]) -> DidResult<Did> {
        if secret_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 secret key must be 32 bytes".to_string(),
            ));
        }

        let signer = Ed25519Signer::from_bytes(secret_key)?;
        let public_key = signer.public_key_bytes();
        let did = Did::new_key_ed25519(&public_key)?;

        let stored = StoredKey {
            secret_key: secret_key.to_vec(),
            key_type: KeyType::Ed25519,
            did: did.clone(),
        };

        self.keys
            .write()
            .await
            .insert(did.as_str().to_string(), stored);

        Ok(did)
    }

    /// Get a signer for the given DID
    pub async fn get_signer(&self, did: &Did) -> DidResult<Ed25519Signer> {
        let keys = self.keys.read().await;
        let stored = keys
            .get(did.as_str())
            .ok_or_else(|| DidError::KeyNotFound(did.as_str().to_string()))?;

        match stored.key_type {
            KeyType::Ed25519 => Ed25519Signer::from_bytes(&stored.secret_key),
        }
    }

    /// Check if a DID has a stored key
    pub async fn has_key(&self, did: &Did) -> bool {
        self.keys.read().await.contains_key(did.as_str())
    }

    /// Remove a key from the store
    pub async fn remove(&self, did: &Did) -> bool {
        self.keys.write().await.remove(did.as_str()).is_some()
    }

    /// List all DIDs with stored keys
    pub async fn list_dids(&self) -> Vec<Did> {
        self.keys
            .read()
            .await
            .values()
            .map(|k| k.did.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_and_store() {
        let keystore = Keystore::new();
        let did = keystore.generate_ed25519().await.unwrap();

        assert!(keystore.has_key(&did).await);
    }

    #[tokio::test]
    async fn test_import_key() {
        let keystore = Keystore::new();
        let secret = [42u8; 32];

        let did = keystore.import_ed25519(&secret).await.unwrap();
        assert!(keystore.has_key(&did).await);
    }

    #[tokio::test]
    async fn test_remove_key() {
        let keystore = Keystore::new();
        let did = keystore.generate_ed25519().await.unwrap();

        assert!(keystore.remove(&did).await);
        assert!(!keystore.has_key(&did).await);
    }

    #[tokio::test]
    async fn test_list_dids() {
        let keystore = Keystore::new();
        keystore.generate_ed25519().await.unwrap();
        keystore.generate_ed25519().await.unwrap();

        let dids = keystore.list_dids().await;
        assert_eq!(dids.len(), 2);
    }
}
