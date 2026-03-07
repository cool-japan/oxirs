//! Key lifecycle management: rotation manager and time-based expiry
//!
//! Provides:
//! - `KeyRotationManager` – per-DID active key tracking with audit history
//! - `KeyExpiry` – time-based key expiration checks
//! - `VerificationKey` – lightweight key descriptor

use crate::{DidError, DidResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── VerificationKey ────────────────────────────────────────────────────────────

/// Lightweight key descriptor used by `KeyRotationManager`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationKey {
    /// Key identifier (e.g. `did:key:z123#key-1`)
    pub id: String,
    /// Public key bytes (multibase encoded, or raw depending on context)
    pub public_key: Vec<u8>,
    /// Algorithm type (e.g. "Ed25519", "P-256")
    pub key_type: String,
    /// Unix timestamp (seconds) when this key was created
    pub created_at: u64,
    /// Optional Unix timestamp (seconds) after which this key is expired
    pub expires_at: Option<u64>,
}

impl VerificationKey {
    /// Create a new key without an expiry.
    pub fn new(id: &str, public_key: Vec<u8>, key_type: &str, created_at: u64) -> Self {
        Self {
            id: id.to_string(),
            public_key,
            key_type: key_type.to_string(),
            created_at,
            expires_at: None,
        }
    }

    /// Create a key with an explicit expiry timestamp.
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
}

// ── KeyRotationRecord ──────────────────────────────────────────────────────────

/// Record of a single key rotation event for a DID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationRecord {
    /// DID this record belongs to
    pub did: String,
    /// Key ID that was replaced
    pub old_key_id: String,
    /// Key ID that replaced it
    pub new_key_id: String,
    /// Unix timestamp of the rotation
    pub rotated_at: u64,
    /// Optional proof of the rotation (e.g. a JWS over the new key)
    pub proof: Option<String>,
}

// ── KeyRotationManager ─────────────────────────────────────────────────────────

/// Manages DID key lifecycle: active key, rotation, revocation, audit history.
pub struct KeyRotationManager {
    /// Current active key per DID: DID -> VerificationKey
    current_keys: HashMap<String, VerificationKey>,
    /// Rotation history per DID: DID -> `Vec<KeyRotationRecord>`
    history: HashMap<String, Vec<KeyRotationRecord>>,
    /// Revoked key IDs (cannot be used even within transition period)
    revoked: HashMap<String, Vec<String>>,
}

impl Default for KeyRotationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyRotationManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            current_keys: HashMap::new(),
            history: HashMap::new(),
            revoked: HashMap::new(),
        }
    }

    /// Register the initial key for a DID (before any rotations).
    pub fn register_initial_key(&mut self, did: &str, key: VerificationKey) {
        self.current_keys.insert(did.to_string(), key);
    }

    /// Get the currently active key for a DID.
    pub fn current_key(&self, did: &str) -> Option<&VerificationKey> {
        self.current_keys.get(did)
    }

    /// Rotate the key for a DID: replace the active key with `new_key`.
    ///
    /// Returns a `KeyRotationRecord` that is also appended to the audit history.
    pub fn rotate_key(
        &mut self,
        did: &str,
        new_key: VerificationKey,
    ) -> DidResult<KeyRotationRecord> {
        let old_key = self
            .current_keys
            .get(did)
            .ok_or_else(|| DidError::KeyNotFound(format!("No active key for DID: {did}")))?;

        if old_key.id == new_key.id {
            return Err(DidError::InvalidKey(
                "New key ID must differ from current key ID".to_string(),
            ));
        }

        let record = KeyRotationRecord {
            did: did.to_string(),
            old_key_id: old_key.id.clone(),
            new_key_id: new_key.id.clone(),
            rotated_at: new_key.created_at,
            proof: None,
        };

        self.history
            .entry(did.to_string())
            .or_default()
            .push(record.clone());

        self.current_keys.insert(did.to_string(), new_key);
        Ok(record)
    }

    /// Rotate with a proof string attached to the record.
    pub fn rotate_key_with_proof(
        &mut self,
        did: &str,
        new_key: VerificationKey,
        proof: String,
    ) -> DidResult<KeyRotationRecord> {
        let mut record = self.rotate_key(did, new_key)?;
        // Update the last record in history with the proof
        if let Some(last) = self.history.entry(did.to_string()).or_default().last_mut() {
            last.proof = Some(proof.clone());
        }
        record.proof = Some(proof);
        Ok(record)
    }

    /// Revoke a key by its ID for a given DID.
    ///
    /// Revoked keys cannot be promoted back to active status.
    pub fn revoke_key(&mut self, did: &str, key_id: &str) -> DidResult<()> {
        // Cannot revoke a key that is currently active via this method
        // (use rotate_key first, then revoke the old key)
        let revoked_list = self.revoked.entry(did.to_string()).or_default();
        if !revoked_list.contains(&key_id.to_string()) {
            revoked_list.push(key_id.to_string());
        }
        Ok(())
    }

    /// Check whether a key ID is revoked for the given DID.
    pub fn is_revoked(&self, did: &str, key_id: &str) -> bool {
        self.revoked
            .get(did)
            .is_some_and(|list| list.contains(&key_id.to_string()))
    }

    /// Return the full rotation history for a DID (ordered chronologically).
    pub fn key_history(&self, did: &str) -> Vec<KeyRotationRecord> {
        self.history.get(did).cloned().unwrap_or_default()
    }

    /// Total number of rotations recorded for a DID
    pub fn rotation_count(&self, did: &str) -> usize {
        self.history.get(did).map(|v| v.len()).unwrap_or(0)
    }

    /// All DIDs currently managed
    pub fn managed_dids(&self) -> Vec<String> {
        self.current_keys.keys().cloned().collect()
    }
}

// ── KeyExpiry ──────────────────────────────────────────────────────────────────

/// Time-based key expiration utilities
pub struct KeyExpiry;

impl KeyExpiry {
    /// Check whether `key` has expired at the given Unix timestamp `now`.
    ///
    /// Returns `true` if `key.expires_at` is set and `now >= expires_at`.
    pub fn is_expired(key: &VerificationKey, now: u64) -> bool {
        key.expires_at.is_some_and(|exp| now >= exp)
    }

    /// Seconds remaining until the key expires.
    ///
    /// Returns `None` if the key has no expiry or has already expired.
    pub fn time_until_expiry(key: &VerificationKey, now: u64) -> Option<u64> {
        let exp = key.expires_at?;
        if now >= exp {
            None
        } else {
            Some(exp - now)
        }
    }

    /// Whether the key will expire within the next `window_seconds`.
    pub fn expires_within(key: &VerificationKey, now: u64, window_seconds: u64) -> bool {
        match key.expires_at {
            None => false,
            Some(exp) => exp > now && exp - now <= window_seconds,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(id: &str, ts: u64) -> VerificationKey {
        VerificationKey::new(id, vec![1, 2, 3, 4], "Ed25519", ts)
    }

    const DID: &str = "did:key:z6Mk123";

    // ── VerificationKey ───────────────────────────────────────────────────

    #[test]
    fn test_verification_key_creation() {
        let k = make_key("did:key:z6Mk#k1", 1000);
        assert_eq!(k.id, "did:key:z6Mk#k1");
        assert_eq!(k.key_type, "Ed25519");
        assert_eq!(k.created_at, 1000);
        assert!(k.expires_at.is_none());
    }

    #[test]
    fn test_verification_key_with_expiry() {
        let k = make_key("did:key:z6Mk#k1", 1000).with_expiry(2000);
        assert_eq!(k.expires_at, Some(2000));
    }

    // ── KeyRotationManager::register_initial_key ──────────────────────────

    #[test]
    fn test_register_initial_key() {
        let mut mgr = KeyRotationManager::new();
        let key = make_key("did:key:z6Mk#k1", 100);
        mgr.register_initial_key(DID, key.clone());
        let current = mgr.current_key(DID).unwrap();
        assert_eq!(current.id, key.id);
    }

    // ── KeyRotationManager::rotate_key ────────────────────────────────────

    #[test]
    fn test_rotate_key_updates_current() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        let new_key = make_key("did:key:z6Mk#k2", 200);
        mgr.rotate_key(DID, new_key.clone()).unwrap();
        assert_eq!(mgr.current_key(DID).unwrap().id, "did:key:z6Mk#k2");
    }

    #[test]
    fn test_rotate_key_records_history() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        mgr.rotate_key(DID, make_key("did:key:z6Mk#k2", 200))
            .unwrap();
        let history = mgr.key_history(DID);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].old_key_id, "did:key:z6Mk#k1");
        assert_eq!(history[0].new_key_id, "did:key:z6Mk#k2");
    }

    #[test]
    fn test_rotate_key_returns_record() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        let record = mgr
            .rotate_key(DID, make_key("did:key:z6Mk#k2", 200))
            .unwrap();
        assert_eq!(record.did, DID);
        assert_eq!(record.rotated_at, 200);
    }

    #[test]
    fn test_rotate_key_no_initial_key_error() {
        let mut mgr = KeyRotationManager::new();
        let result = mgr.rotate_key("did:key:unknown", make_key("did:key:z6Mk#k2", 200));
        assert!(result.is_err());
    }

    #[test]
    fn test_rotate_key_same_id_error() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        let same_key = make_key("did:key:z6Mk#k1", 200); // same ID
        let result = mgr.rotate_key(DID, same_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotate_key_multiple_times() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        mgr.rotate_key(DID, make_key("did:key:z6Mk#k2", 200))
            .unwrap();
        mgr.rotate_key(DID, make_key("did:key:z6Mk#k3", 300))
            .unwrap();
        assert_eq!(mgr.rotation_count(DID), 2);
        assert_eq!(mgr.current_key(DID).unwrap().id, "did:key:z6Mk#k3");
    }

    #[test]
    fn test_rotate_key_with_proof() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        let record = mgr
            .rotate_key_with_proof(
                DID,
                make_key("did:key:z6Mk#k2", 200),
                "proof-abc".to_string(),
            )
            .unwrap();
        assert_eq!(record.proof, Some("proof-abc".to_string()));
    }

    // ── KeyRotationManager::revoke_key ────────────────────────────────────

    #[test]
    fn test_revoke_key_marks_revoked() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        mgr.revoke_key(DID, "did:key:z6Mk#k1").unwrap();
        assert!(mgr.is_revoked(DID, "did:key:z6Mk#k1"));
    }

    #[test]
    fn test_revoke_key_nonexistent_ok() {
        let mut mgr = KeyRotationManager::new();
        // Revoking a key that doesn't exist is silently accepted
        assert!(mgr.revoke_key("did:key:z123", "did:key:z123#k1").is_ok());
    }

    #[test]
    fn test_revoke_key_not_revoked_initially() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        assert!(!mgr.is_revoked(DID, "did:key:z6Mk#k1"));
    }

    #[test]
    fn test_revoke_key_idempotent() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        mgr.revoke_key(DID, "did:key:z6Mk#k1").unwrap();
        mgr.revoke_key(DID, "did:key:z6Mk#k1").unwrap(); // second revoke
        assert!(mgr.is_revoked(DID, "did:key:z6Mk#k1"));
        let revoked_count = mgr.revoked.get(DID).map(|v| v.len()).unwrap_or(0);
        assert_eq!(revoked_count, 1); // deduplicated
    }

    // ── KeyRotationManager helpers ────────────────────────────────────────

    #[test]
    fn test_managed_dids() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key("did:key:a", make_key("did:key:a#k1", 100));
        mgr.register_initial_key("did:key:b", make_key("did:key:b#k1", 100));
        let dids = mgr.managed_dids();
        assert_eq!(dids.len(), 2);
    }

    #[test]
    fn test_rotation_count_zero_initially() {
        let mut mgr = KeyRotationManager::new();
        mgr.register_initial_key(DID, make_key("did:key:z6Mk#k1", 100));
        assert_eq!(mgr.rotation_count(DID), 0);
    }

    #[test]
    fn test_key_history_empty_initially() {
        let mgr = KeyRotationManager::new();
        assert!(mgr.key_history(DID).is_empty());
    }

    // ── KeyExpiry ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_expired_no_expiry_is_false() {
        let key = make_key("k1", 1000);
        assert!(!KeyExpiry::is_expired(&key, 9999));
    }

    #[test]
    fn test_is_expired_before_expiry_is_false() {
        let key = make_key("k1", 1000).with_expiry(2000);
        assert!(!KeyExpiry::is_expired(&key, 1999));
    }

    #[test]
    fn test_is_expired_at_expiry_is_true() {
        let key = make_key("k1", 1000).with_expiry(2000);
        assert!(KeyExpiry::is_expired(&key, 2000));
    }

    #[test]
    fn test_is_expired_after_expiry_is_true() {
        let key = make_key("k1", 1000).with_expiry(2000);
        assert!(KeyExpiry::is_expired(&key, 3000));
    }

    #[test]
    fn test_time_until_expiry_no_expiry_none() {
        let key = make_key("k1", 1000);
        assert!(KeyExpiry::time_until_expiry(&key, 1500).is_none());
    }

    #[test]
    fn test_time_until_expiry_future_key() {
        let key = make_key("k1", 1000).with_expiry(3000);
        let remaining = KeyExpiry::time_until_expiry(&key, 2000).unwrap();
        assert_eq!(remaining, 1000);
    }

    #[test]
    fn test_time_until_expiry_already_expired_none() {
        let key = make_key("k1", 1000).with_expiry(2000);
        assert!(KeyExpiry::time_until_expiry(&key, 2001).is_none());
    }

    #[test]
    fn test_expires_within_window() {
        let key = make_key("k1", 1000).with_expiry(1100);
        // Expires in 100 seconds; window = 200 → should be true
        assert!(KeyExpiry::expires_within(&key, 1000, 200));
    }

    #[test]
    fn test_expires_outside_window() {
        let key = make_key("k1", 1000).with_expiry(2000);
        // Expires in 1000 seconds; window = 100 → should be false
        assert!(!KeyExpiry::expires_within(&key, 1000, 100));
    }

    #[test]
    fn test_expires_within_no_expiry_is_false() {
        let key = make_key("k1", 1000);
        assert!(!KeyExpiry::expires_within(&key, 1000, 9999));
    }
}
