//! # DID Key Manager
//!
//! Manages cryptographic key lifecycles within DID documents. Supports
//! key generation (simulated Ed25519, X25519, P-256), rotation, status
//! tracking (Active / Revoked / Expired), multi-key documents, purpose
//! assignment, metadata, and fingerprint computation.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::key_manager::{
//!     KeyManager, KeyAlgorithm, KeyPurpose, KeyStatus,
//! };
//!
//! let mut mgr = KeyManager::new();
//! let key_id = mgr
//!     .generate_key("did:example:alice", KeyAlgorithm::Ed25519, 1_000_000)
//!     .expect("generate failed");
//! mgr.assign_purpose(&key_id, KeyPurpose::Authentication)
//!     .expect("assign failed");
//! assert_eq!(mgr.status(&key_id), Some(KeyStatus::Active));
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// Supported key algorithms (simulation — no real private-key material).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyAlgorithm {
    /// Ed25519 signing key (32-byte public key).
    Ed25519,
    /// X25519 Diffie-Hellman key (32-byte public key).
    X25519,
    /// NIST P-256 / secp256r1 signing key (33-byte compressed public key).
    P256,
}

impl KeyAlgorithm {
    /// Simulated public-key byte length.
    pub fn public_key_len(self) -> usize {
        match self {
            KeyAlgorithm::Ed25519 | KeyAlgorithm::X25519 => 32,
            KeyAlgorithm::P256 => 33,
        }
    }

    /// Human-readable name used in DID Documents.
    pub fn method_type(self) -> &'static str {
        match self {
            KeyAlgorithm::Ed25519 => "Ed25519VerificationKey2020",
            KeyAlgorithm::X25519 => "X25519KeyAgreementKey2020",
            KeyAlgorithm::P256 => "EcdsaSecp256r1VerificationKey2019",
        }
    }
}

/// Purpose for which a key can be used in a DID document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyPurpose {
    /// authentication relationship
    Authentication,
    /// assertionMethod relationship
    AssertionMethod,
    /// keyAgreement relationship
    KeyAgreement,
    /// capabilityInvocation relationship
    CapabilityInvocation,
    /// capabilityDelegation relationship
    CapabilityDelegation,
}

impl KeyPurpose {
    /// W3C DID Core property name.
    pub fn property_name(self) -> &'static str {
        match self {
            KeyPurpose::Authentication => "authentication",
            KeyPurpose::AssertionMethod => "assertionMethod",
            KeyPurpose::KeyAgreement => "keyAgreement",
            KeyPurpose::CapabilityInvocation => "capabilityInvocation",
            KeyPurpose::CapabilityDelegation => "capabilityDelegation",
        }
    }
}

/// Lifecycle status of a managed key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyStatus {
    /// Key is active and may be used.
    Active,
    /// Key has been explicitly revoked.
    Revoked,
    /// Key has passed its expiry timestamp.
    Expired,
    /// Key has been retired after rotation.
    Retired,
}

/// Lifecycle event recorded for a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyEvent {
    /// Descriptive event kind.
    pub kind: KeyEventKind,
    /// Timestamp of the event (Unix epoch seconds).
    pub timestamp: u64,
    /// Optional human-readable detail.
    pub detail: Option<String>,
}

/// Types of key lifecycle events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEventKind {
    Created,
    Activated,
    PurposeAssigned,
    Revoked,
    Expired,
    Retired,
    Rotated,
}

/// Metadata for a single managed key.
#[derive(Debug, Clone)]
pub struct ManagedKey {
    /// Unique key identifier (e.g. `did:example:alice#key-1`).
    pub id: String,
    /// The DID that owns this key.
    pub controller: String,
    /// Algorithm used.
    pub algorithm: KeyAlgorithm,
    /// Simulated public key bytes.
    pub public_key: Vec<u8>,
    /// Current lifecycle status.
    pub status: KeyStatus,
    /// Assigned purposes.
    pub purposes: Vec<KeyPurpose>,
    /// Unix epoch seconds when the key was created.
    pub created_at: u64,
    /// Optional Unix epoch seconds when the key expires.
    pub expires_at: Option<u64>,
    /// Lifecycle events log.
    pub events: Vec<KeyEvent>,
    /// SHA-256-based fingerprint (hex string).
    pub fingerprint: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`KeyManager`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyManagerError {
    /// A key with the given ID was not found.
    KeyNotFound(String),
    /// A key with the same ID already exists.
    DuplicateKey(String),
    /// The key cannot be used because of its current status.
    InvalidStatus(String),
    /// The DID string is malformed.
    InvalidDid(String),
    /// Purpose already assigned to the key.
    DuplicatePurpose(String),
}

impl std::fmt::Display for KeyManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyManagerError::KeyNotFound(id) => write!(f, "Key not found: {id}"),
            KeyManagerError::DuplicateKey(id) => write!(f, "Duplicate key: {id}"),
            KeyManagerError::InvalidStatus(msg) => write!(f, "Invalid key status: {msg}"),
            KeyManagerError::InvalidDid(d) => write!(f, "Invalid DID: {d}"),
            KeyManagerError::DuplicatePurpose(msg) => write!(f, "Duplicate purpose: {msg}"),
        }
    }
}

impl std::error::Error for KeyManagerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: deterministic simulated public key
// ─────────────────────────────────────────────────────────────────────────────

/// Produce a deterministic simulated public key from the DID, algorithm, and
/// a counter.  This avoids pulling in `rand` directly.
fn simulated_public_key(did: &str, algo: KeyAlgorithm, counter: u64) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(did.as_bytes());
    hasher.update((algo as u8).to_be_bytes());
    hasher.update(counter.to_be_bytes());
    let hash = hasher.finalize();
    let len = algo.public_key_len();
    // SHA-256 is 32 bytes; for P-256 (33 bytes) prepend a compression prefix.
    if len <= hash.len() {
        hash[..len].to_vec()
    } else {
        let mut key = vec![0x02u8]; // compressed point prefix
        key.extend_from_slice(&hash[..len - 1]);
        key
    }
}

/// Compute a hex-encoded SHA-256 fingerprint of the public-key bytes.
fn compute_fingerprint(public_key: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(public_key);
    hex::encode(hash)
}

// ─────────────────────────────────────────────────────────────────────────────
// KeyManager
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory DID key manager.
///
/// Tracks keys across multiple DIDs and provides lookup by id, purpose,
/// status, and controller.
#[derive(Debug, Default)]
pub struct KeyManager {
    /// Primary store: key-id → ManagedKey.
    keys: HashMap<String, ManagedKey>,
    /// Index: controller DID → list of key ids.
    keys_by_controller: HashMap<String, Vec<String>>,
    /// Monotonic counter used for deterministic key generation.
    counter: u64,
}

impl KeyManager {
    /// Create an empty key manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of managed keys.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    // ── Generation ───────────────────────────────────────────────────────

    /// Generate a new key for the given DID.
    ///
    /// Returns the key ID.
    pub fn generate_key(
        &mut self,
        did: &str,
        algorithm: KeyAlgorithm,
        timestamp: u64,
    ) -> Result<String, KeyManagerError> {
        if !did.starts_with("did:") {
            return Err(KeyManagerError::InvalidDid(did.to_string()));
        }

        self.counter += 1;
        let key_index = self.keys_by_controller.get(did).map_or(1, |v| v.len() + 1);
        let key_id = format!("{did}#key-{key_index}");

        if self.keys.contains_key(&key_id) {
            return Err(KeyManagerError::DuplicateKey(key_id));
        }

        let public_key = simulated_public_key(did, algorithm, self.counter);
        let fingerprint = compute_fingerprint(&public_key);

        let key = ManagedKey {
            id: key_id.clone(),
            controller: did.to_string(),
            algorithm,
            public_key,
            status: KeyStatus::Active,
            purposes: Vec::new(),
            created_at: timestamp,
            expires_at: None,
            events: vec![
                KeyEvent {
                    kind: KeyEventKind::Created,
                    timestamp,
                    detail: Some(format!("Algorithm: {:?}", algorithm)),
                },
                KeyEvent {
                    kind: KeyEventKind::Activated,
                    timestamp,
                    detail: None,
                },
            ],
            fingerprint,
        };

        self.keys.insert(key_id.clone(), key);
        self.keys_by_controller
            .entry(did.to_string())
            .or_default()
            .push(key_id.clone());

        Ok(key_id)
    }

    /// Generate a key with an explicit expiry timestamp.
    pub fn generate_key_with_expiry(
        &mut self,
        did: &str,
        algorithm: KeyAlgorithm,
        timestamp: u64,
        expires_at: u64,
    ) -> Result<String, KeyManagerError> {
        let key_id = self.generate_key(did, algorithm, timestamp)?;
        if let Some(key) = self.keys.get_mut(&key_id) {
            key.expires_at = Some(expires_at);
        }
        Ok(key_id)
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    /// Get a key by its ID.
    pub fn get_key(&self, key_id: &str) -> Option<&ManagedKey> {
        self.keys.get(key_id)
    }

    /// Current status of a key.
    pub fn status(&self, key_id: &str) -> Option<KeyStatus> {
        self.keys.get(key_id).map(|k| k.status)
    }

    /// List all key IDs for a given DID.
    pub fn keys_for_did(&self, did: &str) -> Vec<String> {
        self.keys_by_controller
            .get(did)
            .cloned()
            .unwrap_or_default()
    }

    /// Find keys by purpose across all DIDs.
    pub fn keys_by_purpose(&self, purpose: KeyPurpose) -> Vec<&ManagedKey> {
        self.keys
            .values()
            .filter(|k| k.purposes.contains(&purpose))
            .collect()
    }

    /// Find keys by status across all DIDs.
    pub fn keys_by_status(&self, status: KeyStatus) -> Vec<&ManagedKey> {
        self.keys.values().filter(|k| k.status == status).collect()
    }

    /// Find a key by its fingerprint.
    pub fn key_by_fingerprint(&self, fingerprint: &str) -> Option<&ManagedKey> {
        self.keys.values().find(|k| k.fingerprint == fingerprint)
    }

    /// All managed keys (unordered).
    pub fn all_keys(&self) -> Vec<&ManagedKey> {
        self.keys.values().collect()
    }

    /// List all controller DIDs.
    pub fn controllers(&self) -> Vec<String> {
        self.keys_by_controller.keys().cloned().collect()
    }

    // ── Purpose assignment ───────────────────────────────────────────────

    /// Assign a purpose to a key.
    pub fn assign_purpose(
        &mut self,
        key_id: &str,
        purpose: KeyPurpose,
    ) -> Result<(), KeyManagerError> {
        let key = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| KeyManagerError::KeyNotFound(key_id.to_string()))?;

        if key.status != KeyStatus::Active {
            return Err(KeyManagerError::InvalidStatus(format!(
                "Cannot assign purpose to {:?} key",
                key.status
            )));
        }

        if key.purposes.contains(&purpose) {
            return Err(KeyManagerError::DuplicatePurpose(format!(
                "{:?} already assigned to {}",
                purpose, key_id
            )));
        }

        key.purposes.push(purpose);
        key.events.push(KeyEvent {
            kind: KeyEventKind::PurposeAssigned,
            timestamp: key.created_at, // use creation time as fallback
            detail: Some(format!("{:?}", purpose)),
        });

        Ok(())
    }

    /// Remove a purpose from a key.
    pub fn remove_purpose(
        &mut self,
        key_id: &str,
        purpose: KeyPurpose,
    ) -> Result<(), KeyManagerError> {
        let key = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| KeyManagerError::KeyNotFound(key_id.to_string()))?;

        key.purposes.retain(|p| *p != purpose);
        Ok(())
    }

    // ── Status transitions ───────────────────────────────────────────────

    /// Revoke a key.
    pub fn revoke_key(&mut self, key_id: &str, timestamp: u64) -> Result<(), KeyManagerError> {
        let key = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| KeyManagerError::KeyNotFound(key_id.to_string()))?;

        if key.status == KeyStatus::Revoked {
            return Err(KeyManagerError::InvalidStatus(
                "Key is already revoked".to_string(),
            ));
        }

        key.status = KeyStatus::Revoked;
        key.events.push(KeyEvent {
            kind: KeyEventKind::Revoked,
            timestamp,
            detail: None,
        });
        Ok(())
    }

    /// Mark a key as expired.
    pub fn expire_key(&mut self, key_id: &str, timestamp: u64) -> Result<(), KeyManagerError> {
        let key = self
            .keys
            .get_mut(key_id)
            .ok_or_else(|| KeyManagerError::KeyNotFound(key_id.to_string()))?;

        if key.status == KeyStatus::Revoked {
            return Err(KeyManagerError::InvalidStatus(
                "Cannot expire a revoked key".to_string(),
            ));
        }

        key.status = KeyStatus::Expired;
        key.events.push(KeyEvent {
            kind: KeyEventKind::Expired,
            timestamp,
            detail: None,
        });
        Ok(())
    }

    /// Check all keys for automatic expiry at the given timestamp.
    /// Returns the IDs of keys that were transitioned to Expired.
    pub fn check_expiry(&mut self, current_time: u64) -> Vec<String> {
        let mut expired_ids = Vec::new();

        let ids: Vec<String> = self
            .keys
            .values()
            .filter(|k| {
                k.status == KeyStatus::Active && k.expires_at.is_some_and(|exp| exp <= current_time)
            })
            .map(|k| k.id.clone())
            .collect();

        for id in ids {
            if let Some(key) = self.keys.get_mut(&id) {
                key.status = KeyStatus::Expired;
                key.events.push(KeyEvent {
                    kind: KeyEventKind::Expired,
                    timestamp: current_time,
                    detail: Some("Automatic expiry".to_string()),
                });
                expired_ids.push(id);
            }
        }

        expired_ids
    }

    // ── Key rotation ─────────────────────────────────────────────────────

    /// Rotate a key: generate a new key with the same algorithm and
    /// purposes, retire the old one.  Returns the new key ID.
    pub fn rotate_key(
        &mut self,
        old_key_id: &str,
        timestamp: u64,
    ) -> Result<String, KeyManagerError> {
        let (controller, algorithm, purposes) = {
            let old = self
                .keys
                .get(old_key_id)
                .ok_or_else(|| KeyManagerError::KeyNotFound(old_key_id.to_string()))?;

            if old.status != KeyStatus::Active {
                return Err(KeyManagerError::InvalidStatus(format!(
                    "Cannot rotate {:?} key",
                    old.status
                )));
            }

            (old.controller.clone(), old.algorithm, old.purposes.clone())
        };

        // Retire old key
        if let Some(old) = self.keys.get_mut(old_key_id) {
            old.status = KeyStatus::Retired;
            old.events.push(KeyEvent {
                kind: KeyEventKind::Retired,
                timestamp,
                detail: Some("Rotated to new key".to_string()),
            });
            old.events.push(KeyEvent {
                kind: KeyEventKind::Rotated,
                timestamp,
                detail: None,
            });
        }

        // Generate new key
        let new_id = self.generate_key(&controller, algorithm, timestamp)?;

        // Transfer purposes
        for purpose in &purposes {
            self.assign_purpose(&new_id, *purpose)?;
        }

        Ok(new_id)
    }

    // ── Metadata helpers ─────────────────────────────────────────────────

    /// Get the events log for a key.
    pub fn events(&self, key_id: &str) -> Option<&[KeyEvent]> {
        self.keys.get(key_id).map(|k| k.events.as_slice())
    }

    /// Get the fingerprint for a key.
    pub fn fingerprint(&self, key_id: &str) -> Option<&str> {
        self.keys.get(key_id).map(|k| k.fingerprint.as_str())
    }

    /// Summary statistics.
    pub fn stats(&self) -> KeyManagerStats {
        let mut active = 0usize;
        let mut revoked = 0usize;
        let mut expired = 0usize;
        let mut retired = 0usize;

        for key in self.keys.values() {
            match key.status {
                KeyStatus::Active => active += 1,
                KeyStatus::Revoked => revoked += 1,
                KeyStatus::Expired => expired += 1,
                KeyStatus::Retired => retired += 1,
            }
        }

        KeyManagerStats {
            total: self.keys.len(),
            active,
            revoked,
            expired,
            retired,
            controllers: self.keys_by_controller.len(),
        }
    }

    /// Remove a key entirely (for cleanup / tests).
    pub fn remove_key(&mut self, key_id: &str) -> Result<ManagedKey, KeyManagerError> {
        let key = self
            .keys
            .remove(key_id)
            .ok_or_else(|| KeyManagerError::KeyNotFound(key_id.to_string()))?;

        if let Some(ids) = self.keys_by_controller.get_mut(&key.controller) {
            ids.retain(|id| id != key_id);
            if ids.is_empty() {
                self.keys_by_controller.remove(&key.controller);
            }
        }

        Ok(key)
    }
}

/// Aggregate statistics for the key manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyManagerStats {
    /// Total number of managed keys.
    pub total: usize,
    /// Active keys.
    pub active: usize,
    /// Revoked keys.
    pub revoked: usize,
    /// Expired keys.
    pub expired: usize,
    /// Retired keys.
    pub retired: usize,
    /// Number of unique controller DIDs.
    pub controllers: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn alice_did() -> &'static str {
        "did:example:alice"
    }

    fn bob_did() -> &'static str {
        "did:example:bob"
    }

    // ── Generation ───────────────────────────────────────────────────────

    #[test]
    fn test_generate_ed25519_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        assert!(id.starts_with(alice_did()));
        let key = mgr.get_key(&id).expect("lookup");
        assert_eq!(key.algorithm, KeyAlgorithm::Ed25519);
        assert_eq!(key.public_key.len(), 32);
        assert_eq!(key.status, KeyStatus::Active);
    }

    #[test]
    fn test_generate_x25519_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::X25519, 1000)
            .expect("gen");
        let key = mgr.get_key(&id).expect("lookup");
        assert_eq!(key.algorithm, KeyAlgorithm::X25519);
        assert_eq!(key.public_key.len(), 32);
    }

    #[test]
    fn test_generate_p256_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::P256, 1000)
            .expect("gen");
        let key = mgr.get_key(&id).expect("lookup");
        assert_eq!(key.algorithm, KeyAlgorithm::P256);
        assert_eq!(key.public_key.len(), 33);
    }

    #[test]
    fn test_generate_invalid_did() {
        let mut mgr = KeyManager::new();
        let result = mgr.generate_key("not-a-did", KeyAlgorithm::Ed25519, 1000);
        assert!(matches!(result, Err(KeyManagerError::InvalidDid(_))));
    }

    #[test]
    fn test_generate_multiple_keys_for_same_did() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let id2 = mgr
            .generate_key(alice_did(), KeyAlgorithm::X25519, 2000)
            .expect("gen2");
        assert_ne!(id1, id2);
        assert_eq!(mgr.keys_for_did(alice_did()).len(), 2);
    }

    #[test]
    fn test_generate_key_with_expiry() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key_with_expiry(alice_did(), KeyAlgorithm::Ed25519, 1000, 5000)
            .expect("gen");
        let key = mgr.get_key(&id).expect("lookup");
        assert_eq!(key.expires_at, Some(5000));
    }

    // ── Lookup ───────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_nonexistent_key() {
        let mgr = KeyManager::new();
        assert!(mgr.get_key("did:example:x#key-99").is_none());
        assert!(mgr.status("did:example:x#key-99").is_none());
    }

    #[test]
    fn test_keys_for_did_empty() {
        let mgr = KeyManager::new();
        assert!(mgr.keys_for_did("did:example:x").is_empty());
    }

    #[test]
    fn test_all_keys() {
        let mut mgr = KeyManager::new();
        mgr.generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.generate_key(bob_did(), KeyAlgorithm::P256, 2000)
            .expect("gen");
        assert_eq!(mgr.all_keys().len(), 2);
    }

    #[test]
    fn test_controllers() {
        let mut mgr = KeyManager::new();
        mgr.generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.generate_key(bob_did(), KeyAlgorithm::P256, 2000)
            .expect("gen");
        let controllers = mgr.controllers();
        assert_eq!(controllers.len(), 2);
        assert!(controllers.contains(&alice_did().to_string()));
        assert!(controllers.contains(&bob_did().to_string()));
    }

    // ── Purpose assignment ───────────────────────────────────────────────

    #[test]
    fn test_assign_authentication_purpose() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&id, KeyPurpose::Authentication)
            .expect("assign");
        let key = mgr.get_key(&id).expect("lookup");
        assert!(key.purposes.contains(&KeyPurpose::Authentication));
    }

    #[test]
    fn test_assign_multiple_purposes() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&id, KeyPurpose::Authentication)
            .expect("auth");
        mgr.assign_purpose(&id, KeyPurpose::AssertionMethod)
            .expect("assert");
        let key = mgr.get_key(&id).expect("lookup");
        assert_eq!(key.purposes.len(), 2);
    }

    #[test]
    fn test_assign_duplicate_purpose_error() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&id, KeyPurpose::Authentication)
            .expect("first");
        let result = mgr.assign_purpose(&id, KeyPurpose::Authentication);
        assert!(matches!(result, Err(KeyManagerError::DuplicatePurpose(_))));
    }

    #[test]
    fn test_assign_purpose_to_revoked_key_error() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        let result = mgr.assign_purpose(&id, KeyPurpose::Authentication);
        assert!(matches!(result, Err(KeyManagerError::InvalidStatus(_))));
    }

    #[test]
    fn test_assign_purpose_to_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.assign_purpose("did:example:x#key-99", KeyPurpose::Authentication);
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }

    #[test]
    fn test_remove_purpose() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&id, KeyPurpose::Authentication)
            .expect("assign");
        mgr.remove_purpose(&id, KeyPurpose::Authentication)
            .expect("remove");
        let key = mgr.get_key(&id).expect("lookup");
        assert!(key.purposes.is_empty());
    }

    #[test]
    fn test_keys_by_purpose() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let id2 = mgr
            .generate_key(bob_did(), KeyAlgorithm::Ed25519, 2000)
            .expect("gen2");
        mgr.assign_purpose(&id1, KeyPurpose::Authentication)
            .expect("a1");
        mgr.assign_purpose(&id2, KeyPurpose::KeyAgreement)
            .expect("a2");

        let auth_keys = mgr.keys_by_purpose(KeyPurpose::Authentication);
        assert_eq!(auth_keys.len(), 1);
        assert_eq!(auth_keys[0].id, id1);

        let agreement_keys = mgr.keys_by_purpose(KeyPurpose::KeyAgreement);
        assert_eq!(agreement_keys.len(), 1);
        assert_eq!(agreement_keys[0].id, id2);
    }

    #[test]
    fn test_key_purpose_property_names() {
        assert_eq!(KeyPurpose::Authentication.property_name(), "authentication");
        assert_eq!(
            KeyPurpose::AssertionMethod.property_name(),
            "assertionMethod"
        );
        assert_eq!(KeyPurpose::KeyAgreement.property_name(), "keyAgreement");
        assert_eq!(
            KeyPurpose::CapabilityInvocation.property_name(),
            "capabilityInvocation"
        );
        assert_eq!(
            KeyPurpose::CapabilityDelegation.property_name(),
            "capabilityDelegation"
        );
    }

    // ── Status transitions ───────────────────────────────────────────────

    #[test]
    fn test_revoke_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        assert_eq!(mgr.status(&id), Some(KeyStatus::Revoked));
    }

    #[test]
    fn test_revoke_already_revoked_error() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("first");
        let result = mgr.revoke_key(&id, 3000);
        assert!(matches!(result, Err(KeyManagerError::InvalidStatus(_))));
    }

    #[test]
    fn test_expire_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.expire_key(&id, 5000).expect("expire");
        assert_eq!(mgr.status(&id), Some(KeyStatus::Expired));
    }

    #[test]
    fn test_expire_revoked_key_error() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        let result = mgr.expire_key(&id, 3000);
        assert!(matches!(result, Err(KeyManagerError::InvalidStatus(_))));
    }

    #[test]
    fn test_keys_by_status() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let _id2 = mgr
            .generate_key(bob_did(), KeyAlgorithm::Ed25519, 2000)
            .expect("gen2");
        mgr.revoke_key(&id1, 3000).expect("revoke");

        let active = mgr.keys_by_status(KeyStatus::Active);
        assert_eq!(active.len(), 1);
        let revoked = mgr.keys_by_status(KeyStatus::Revoked);
        assert_eq!(revoked.len(), 1);
    }

    #[test]
    fn test_check_expiry_auto() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key_with_expiry(alice_did(), KeyAlgorithm::Ed25519, 1000, 5000)
            .expect("gen");

        // Before expiry — nothing happens
        let expired = mgr.check_expiry(4999);
        assert!(expired.is_empty());
        assert_eq!(mgr.status(&id), Some(KeyStatus::Active));

        // At expiry
        let expired = mgr.check_expiry(5000);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], id);
        assert_eq!(mgr.status(&id), Some(KeyStatus::Expired));
    }

    #[test]
    fn test_check_expiry_does_not_affect_revoked() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key_with_expiry(alice_did(), KeyAlgorithm::Ed25519, 1000, 5000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        let expired = mgr.check_expiry(6000);
        assert!(expired.is_empty()); // revoked keys are not auto-expired
    }

    // ── Key rotation ─────────────────────────────────────────────────────

    #[test]
    fn test_rotate_key() {
        let mut mgr = KeyManager::new();
        let old_id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&old_id, KeyPurpose::Authentication)
            .expect("assign");

        let new_id = mgr.rotate_key(&old_id, 2000).expect("rotate");

        // Old key is retired
        assert_eq!(mgr.status(&old_id), Some(KeyStatus::Retired));
        // New key is active
        assert_eq!(mgr.status(&new_id), Some(KeyStatus::Active));
        // Purposes transferred
        let new_key = mgr.get_key(&new_id).expect("new");
        assert!(new_key.purposes.contains(&KeyPurpose::Authentication));
    }

    #[test]
    fn test_rotate_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.rotate_key("did:example:x#key-99", 1000);
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }

    #[test]
    fn test_rotate_revoked_key_error() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        let result = mgr.rotate_key(&id, 3000);
        assert!(matches!(result, Err(KeyManagerError::InvalidStatus(_))));
    }

    #[test]
    fn test_rotate_preserves_algorithm() {
        let mut mgr = KeyManager::new();
        let old_id = mgr
            .generate_key(alice_did(), KeyAlgorithm::P256, 1000)
            .expect("gen");
        let new_id = mgr.rotate_key(&old_id, 2000).expect("rotate");
        let new_key = mgr.get_key(&new_id).expect("new");
        assert_eq!(new_key.algorithm, KeyAlgorithm::P256);
    }

    #[test]
    fn test_rotate_transfers_multiple_purposes() {
        let mut mgr = KeyManager::new();
        let old_id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.assign_purpose(&old_id, KeyPurpose::Authentication)
            .expect("a");
        mgr.assign_purpose(&old_id, KeyPurpose::AssertionMethod)
            .expect("b");

        let new_id = mgr.rotate_key(&old_id, 2000).expect("rotate");
        let new_key = mgr.get_key(&new_id).expect("new");
        assert_eq!(new_key.purposes.len(), 2);
    }

    // ── Events ───────────────────────────────────────────────────────────

    #[test]
    fn test_events_on_creation() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        let events = mgr.events(&id).expect("events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, KeyEventKind::Created);
        assert_eq!(events[1].kind, KeyEventKind::Activated);
    }

    #[test]
    fn test_events_after_revocation() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.revoke_key(&id, 2000).expect("revoke");
        let events = mgr.events(&id).expect("events");
        assert!(events.iter().any(|e| e.kind == KeyEventKind::Revoked));
    }

    #[test]
    fn test_events_after_rotation() {
        let mut mgr = KeyManager::new();
        let old_id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.rotate_key(&old_id, 2000).expect("rotate");
        let events = mgr.events(&old_id).expect("events");
        assert!(events.iter().any(|e| e.kind == KeyEventKind::Retired));
        assert!(events.iter().any(|e| e.kind == KeyEventKind::Rotated));
    }

    #[test]
    fn test_events_nonexistent_key() {
        let mgr = KeyManager::new();
        assert!(mgr.events("did:example:x#key-99").is_none());
    }

    // ── Fingerprint ──────────────────────────────────────────────────────

    #[test]
    fn test_fingerprint_not_empty() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        let fp = mgr.fingerprint(&id).expect("fp");
        assert!(!fp.is_empty());
        // SHA-256 hex is 64 chars
        assert_eq!(fp.len(), 64);
    }

    #[test]
    fn test_fingerprint_unique_per_key() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let id2 = mgr
            .generate_key(alice_did(), KeyAlgorithm::X25519, 2000)
            .expect("gen2");
        let fp1 = mgr.fingerprint(&id1).expect("fp1");
        let fp2 = mgr.fingerprint(&id2).expect("fp2");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_lookup_by_fingerprint() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        let fp = mgr.fingerprint(&id).expect("fp").to_string();
        let found = mgr.key_by_fingerprint(&fp).expect("found");
        assert_eq!(found.id, id);
    }

    #[test]
    fn test_lookup_by_fingerprint_not_found() {
        let mgr = KeyManager::new();
        assert!(mgr.key_by_fingerprint("abcdef1234567890").is_none());
    }

    // ── Stats ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let mgr = KeyManager::new();
        let stats = mgr.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.active, 0);
        assert_eq!(stats.controllers, 0);
    }

    #[test]
    fn test_stats_mixed() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let _id2 = mgr
            .generate_key(bob_did(), KeyAlgorithm::P256, 2000)
            .expect("gen2");
        mgr.revoke_key(&id1, 3000).expect("revoke");

        let stats = mgr.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.revoked, 1);
        assert_eq!(stats.controllers, 2);
    }

    // ── Remove key ───────────────────────────────────────────────────────

    #[test]
    fn test_remove_key() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        let removed = mgr.remove_key(&id).expect("remove");
        assert_eq!(removed.id, id);
        assert_eq!(mgr.key_count(), 0);
        assert!(mgr.keys_for_did(alice_did()).is_empty());
    }

    #[test]
    fn test_remove_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.remove_key("did:example:x#key-99");
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }

    // ── Algorithm metadata ───────────────────────────────────────────────

    #[test]
    fn test_algorithm_method_types() {
        assert_eq!(
            KeyAlgorithm::Ed25519.method_type(),
            "Ed25519VerificationKey2020"
        );
        assert_eq!(
            KeyAlgorithm::X25519.method_type(),
            "X25519KeyAgreementKey2020"
        );
        assert_eq!(
            KeyAlgorithm::P256.method_type(),
            "EcdsaSecp256r1VerificationKey2019"
        );
    }

    #[test]
    fn test_algorithm_public_key_lengths() {
        assert_eq!(KeyAlgorithm::Ed25519.public_key_len(), 32);
        assert_eq!(KeyAlgorithm::X25519.public_key_len(), 32);
        assert_eq!(KeyAlgorithm::P256.public_key_len(), 33);
    }

    // ── Key count ────────────────────────────────────────────────────────

    #[test]
    fn test_key_count() {
        let mut mgr = KeyManager::new();
        assert_eq!(mgr.key_count(), 0);
        mgr.generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        assert_eq!(mgr.key_count(), 1);
        mgr.generate_key(alice_did(), KeyAlgorithm::X25519, 2000)
            .expect("gen");
        assert_eq!(mgr.key_count(), 2);
    }

    // ── Multi-key DID documents ──────────────────────────────────────────

    #[test]
    fn test_multi_key_did_different_algorithms() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let id2 = mgr
            .generate_key(alice_did(), KeyAlgorithm::X25519, 2000)
            .expect("gen2");
        let id3 = mgr
            .generate_key(alice_did(), KeyAlgorithm::P256, 3000)
            .expect("gen3");

        mgr.assign_purpose(&id1, KeyPurpose::Authentication)
            .expect("a1");
        mgr.assign_purpose(&id2, KeyPurpose::KeyAgreement)
            .expect("a2");
        mgr.assign_purpose(&id3, KeyPurpose::CapabilityInvocation)
            .expect("a3");

        let keys = mgr.keys_for_did(alice_did());
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn test_multi_key_rotate_one() {
        let mut mgr = KeyManager::new();
        let id1 = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen1");
        let id2 = mgr
            .generate_key(alice_did(), KeyAlgorithm::X25519, 2000)
            .expect("gen2");
        mgr.assign_purpose(&id1, KeyPurpose::Authentication)
            .expect("a1");
        mgr.assign_purpose(&id2, KeyPurpose::KeyAgreement)
            .expect("a2");

        // Rotate only the authentication key
        let new_id = mgr.rotate_key(&id1, 3000).expect("rotate");
        assert_eq!(mgr.status(&id1), Some(KeyStatus::Retired));
        assert_eq!(mgr.status(&id2), Some(KeyStatus::Active));
        assert_eq!(mgr.status(&new_id), Some(KeyStatus::Active));
        // alice now has 3 keys total (one retired + two active)
        assert_eq!(mgr.keys_for_did(alice_did()).len(), 3);
    }

    // ── Error display ────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = KeyManagerError::KeyNotFound("did:x#key-1".to_string());
        assert!(err.to_string().contains("Key not found"));

        let err = KeyManagerError::DuplicateKey("did:x#key-1".to_string());
        assert!(err.to_string().contains("Duplicate key"));

        let err = KeyManagerError::InvalidStatus("bad".to_string());
        assert!(err.to_string().contains("Invalid key status"));

        let err = KeyManagerError::InvalidDid("foo".to_string());
        assert!(err.to_string().contains("Invalid DID"));

        let err = KeyManagerError::DuplicatePurpose("auth".to_string());
        assert!(err.to_string().contains("Duplicate purpose"));
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_remove_last_key_cleans_controller_index() {
        let mut mgr = KeyManager::new();
        let id = mgr
            .generate_key(alice_did(), KeyAlgorithm::Ed25519, 1000)
            .expect("gen");
        mgr.remove_key(&id).expect("remove");
        assert!(mgr.controllers().is_empty());
    }

    #[test]
    fn test_keys_by_purpose_empty() {
        let mgr = KeyManager::new();
        assert!(mgr.keys_by_purpose(KeyPurpose::Authentication).is_empty());
    }

    #[test]
    fn test_keys_by_status_empty() {
        let mgr = KeyManager::new();
        assert!(mgr.keys_by_status(KeyStatus::Active).is_empty());
    }

    #[test]
    fn test_expire_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.expire_key("did:example:x#key-99", 1000);
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }

    #[test]
    fn test_revoke_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.revoke_key("did:example:x#key-99", 1000);
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }

    #[test]
    fn test_remove_purpose_nonexistent_key_error() {
        let mut mgr = KeyManager::new();
        let result = mgr.remove_purpose("did:example:x#key-99", KeyPurpose::Authentication);
        assert!(matches!(result, Err(KeyManagerError::KeyNotFound(_))));
    }
}
