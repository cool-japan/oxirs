//! Key rotation mechanism for DID documents
//!
//! Key rotation allows DID controllers to replace compromised or expired keys
//! while maintaining DID continuity. Old keys remain valid during a configurable
//! transition period to allow verifiers to update.
//!
//! # Rotation Process
//!
//! 1. Generate a new key pair
//! 2. Add new key to DID Document
//! 3. Record rotation event with timestamps
//! 4. Old key remains valid until `revocation_date` passes
//! 5. Remove old key from DID Document after transition period

use crate::did::document::{DidDocument, VerificationRelationship};
use crate::proof::ed25519::Ed25519Signer;
use crate::{Did, DidError, DidResult, VerificationMethod};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A record of a key rotation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationRecord {
    /// The key ID that was replaced (old key)
    pub old_key_id: String,
    /// The key ID that replaced it (new key)
    pub new_key_id: String,
    /// When the rotation occurred
    pub rotation_date: DateTime<Utc>,
    /// When the old key will be fully revoked (end of transition period)
    pub revocation_date: Option<DateTime<Utc>>,
    /// Reason for rotation (optional metadata)
    pub reason: Option<KeyRotationReason>,
}

/// Reason for key rotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyRotationReason {
    /// Scheduled/periodic rotation
    Scheduled,
    /// Key was potentially compromised
    Compromise,
    /// Cryptographic algorithm upgrade
    AlgorithmUpgrade,
    /// Organizational change
    OrganizationalChange,
    /// Key expiration
    Expiration,
    /// Other reason
    Other,
}

impl KeyRotationReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Compromise => "compromise",
            Self::AlgorithmUpgrade => "algorithm_upgrade",
            Self::OrganizationalChange => "organizational_change",
            Self::Expiration => "expiration",
            Self::Other => "other",
        }
    }
}

/// Parameters for a key rotation operation
#[derive(Debug, Clone)]
pub struct KeyRotation {
    /// The new key to rotate to
    pub new_key: VerificationMethod,
    /// The new secret key bytes (for signing with new key)
    pub new_secret_key: Vec<u8>,
    /// When the rotation takes effect
    pub rotation_date: DateTime<Utc>,
    /// How many days to keep the old key valid (transition period)
    pub transition_period_days: u32,
    /// Reason for this rotation
    pub reason: Option<KeyRotationReason>,
}

impl KeyRotation {
    /// Create a new key rotation with default 30-day transition period
    pub fn new(new_key: VerificationMethod, new_secret_key: Vec<u8>) -> Self {
        Self {
            new_key,
            new_secret_key,
            rotation_date: Utc::now(),
            transition_period_days: 30,
            reason: None,
        }
    }

    /// Set the transition period (days the old key remains valid after rotation)
    pub fn with_transition_period(mut self, days: u32) -> Self {
        self.transition_period_days = days;
        self
    }

    /// Set the rotation reason
    pub fn with_reason(mut self, reason: KeyRotationReason) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Set an explicit rotation date
    pub fn with_rotation_date(mut self, date: DateTime<Utc>) -> Self {
        self.rotation_date = date;
        self
    }

    /// Calculate the revocation date (rotation_date + transition_period_days)
    pub fn revocation_date(&self) -> DateTime<Utc> {
        self.rotation_date + Duration::days(self.transition_period_days as i64)
    }
}

/// Registry tracking all key rotations for one or more DIDs
pub struct KeyRotationRegistry {
    /// Rotation records indexed by old key ID
    rotations: Vec<KeyRotationRecord>,
    /// Current active keys: DID -> key_id mapping
    current_keys: HashMap<String, String>,
    /// All known keys: key_id -> VerificationMethod
    key_map: HashMap<String, VerificationMethod>,
}

impl Default for KeyRotationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyRotationRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            rotations: Vec::new(),
            current_keys: HashMap::new(),
            key_map: HashMap::new(),
        }
    }

    /// Register the initial key for a DID (before any rotations)
    pub fn register_initial_key(&mut self, did: &Did, key: VerificationMethod) {
        let key_id = key.id.clone();
        self.current_keys
            .insert(did.as_str().to_string(), key_id.clone());
        self.key_map.insert(key_id, key);
    }

    /// Perform a key rotation on a DID Document
    ///
    /// This:
    /// 1. Adds the new key to the DID Document
    /// 2. Updates verification relationships to point to the new key
    /// 3. Records the rotation event
    /// 4. Optionally keeps the old key during a transition period
    pub fn rotate_key(
        &mut self,
        did_doc: &mut DidDocument,
        rotation: KeyRotation,
    ) -> DidResult<KeyRotationRecord> {
        let did_str = did_doc.id.as_str().to_string();

        // Find the current key for this DID
        let old_key_id = self
            .current_keys
            .get(&did_str)
            .cloned()
            .or_else(|| {
                // Fall back to first authentication key in document
                did_doc.authentication.first().map(|rel| match rel {
                    VerificationRelationship::Reference(id) => id.clone(),
                    VerificationRelationship::Embedded(vm) => vm.id.clone(),
                })
            })
            .ok_or_else(|| {
                DidError::KeyNotFound(format!("No current key found for DID: {}", did_str))
            })?;

        let new_key_id = rotation.new_key.id.clone();

        // Ensure new key isn't the same as old key
        if old_key_id == new_key_id {
            return Err(DidError::InvalidKey(
                "New key ID must differ from old key ID".to_string(),
            ));
        }

        // Ensure new key isn't already in the document
        if did_doc.get_verification_method(&new_key_id).is_some() {
            return Err(DidError::InvalidKey(format!(
                "Key {} already exists in DID Document",
                new_key_id
            )));
        }

        // Calculate revocation date for old key
        let revocation_date = if rotation.transition_period_days > 0 {
            Some(rotation.revocation_date())
        } else {
            None
        };

        // Add new key to verification methods
        let new_key_vm = rotation.new_key.clone();
        did_doc.verification_method.push(new_key_vm.clone());

        // Update all verification relationships to use the new key
        update_verification_relationships(&mut did_doc.authentication, &old_key_id, &new_key_id);
        update_verification_relationships(&mut did_doc.assertion_method, &old_key_id, &new_key_id);
        update_verification_relationships(
            &mut did_doc.capability_invocation,
            &old_key_id,
            &new_key_id,
        );
        update_verification_relationships(
            &mut did_doc.capability_delegation,
            &old_key_id,
            &new_key_id,
        );
        update_verification_relationships(&mut did_doc.key_agreement, &old_key_id, &new_key_id);

        // If no transition period, remove old key immediately
        if rotation.transition_period_days == 0 {
            did_doc.verification_method.retain(|vm| vm.id != old_key_id);
        }

        // Create rotation record
        let record = KeyRotationRecord {
            old_key_id: old_key_id.clone(),
            new_key_id: new_key_id.clone(),
            rotation_date: rotation.rotation_date,
            revocation_date,
            reason: rotation.reason,
        };

        // Update registry state
        self.rotations.push(record.clone());
        self.current_keys.insert(did_str, new_key_id.clone());
        self.key_map.insert(new_key_id, new_key_vm);

        Ok(record)
    }

    /// Check if a key is currently valid (not revoked, within transition period if applicable)
    ///
    /// # Arguments
    /// * `key_id` - The key ID to check
    /// * `at_time` - The time to check validity at (None = now)
    pub fn is_key_valid(&self, key_id: &str, at_time: Option<DateTime<Utc>>) -> bool {
        let check_time = at_time.unwrap_or_else(Utc::now);

        // Check if this key was ever rotated away from
        for record in &self.rotations {
            if record.old_key_id == key_id {
                // This key was rotated away from
                match record.revocation_date {
                    Some(revoke_date) => {
                        // Key is valid until revocation date
                        if check_time > revoke_date {
                            return false;
                        }
                    }
                    None => {
                        // No transition period - immediately invalid after rotation
                        if check_time >= record.rotation_date {
                            return false;
                        }
                    }
                }
            }
        }

        // Key is either current or within its transition period
        self.key_map.contains_key(key_id)
    }

    /// Get the current active key ID for a DID
    pub fn get_current_key_id(&self, did: &Did) -> Option<&String> {
        self.current_keys.get(did.as_str())
    }

    /// Get the current active VerificationMethod for a DID
    pub fn get_current_key(&self, did: &Did) -> Option<&VerificationMethod> {
        self.current_keys
            .get(did.as_str())
            .and_then(|id| self.key_map.get(id))
    }

    /// Get all rotation records (ordered chronologically)
    pub fn get_rotation_history(&self) -> &[KeyRotationRecord] {
        &self.rotations
    }

    /// Get rotation history for a specific key
    pub fn get_key_rotation_chain(&self, initial_key_id: &str) -> Vec<&KeyRotationRecord> {
        let mut chain = Vec::new();
        let mut current = initial_key_id;

        loop {
            let record = self.rotations.iter().find(|r| r.old_key_id == current);
            match record {
                Some(r) => {
                    chain.push(r);
                    current = &r.new_key_id;
                }
                None => break,
            }
        }

        chain
    }

    /// Apply pending revocations for keys past their revocation date
    ///
    /// Removes old keys from the DID Document that have passed their transition period.
    pub fn apply_pending_revocations(
        &self,
        did_doc: &mut DidDocument,
        at_time: Option<DateTime<Utc>>,
    ) {
        let check_time = at_time.unwrap_or_else(Utc::now);

        let keys_to_remove: Vec<String> = self
            .rotations
            .iter()
            .filter(|r| {
                if let Some(revoke_date) = r.revocation_date {
                    check_time > revoke_date
                } else {
                    false
                }
            })
            .map(|r| r.old_key_id.clone())
            .collect();

        did_doc
            .verification_method
            .retain(|vm| !keys_to_remove.contains(&vm.id));
    }

    /// Count total rotations in registry
    pub fn rotation_count(&self) -> usize {
        self.rotations.len()
    }
}

/// Generate a new Ed25519 verification method for key rotation
///
/// Returns (VerificationMethod, secret_key_bytes)
pub fn generate_rotation_key(
    did: &Did,
    key_fragment: &str,
) -> DidResult<(VerificationMethod, Vec<u8>)> {
    let signer = Ed25519Signer::generate();
    let secret_key = signer.secret_key_bytes().to_vec();
    let public_key = signer.public_key_bytes();

    let key_id = format!("{}#{}", did.as_str(), key_fragment);
    let vm = VerificationMethod::ed25519(&key_id, did.as_str(), &public_key);

    Ok((vm, secret_key))
}

/// Update verification relationships to point from old_key_id to new_key_id
fn update_verification_relationships(
    relationships: &mut [VerificationRelationship],
    old_key_id: &str,
    new_key_id: &str,
) {
    for rel in relationships.iter_mut() {
        match rel {
            VerificationRelationship::Reference(id) if id == old_key_id => {
                *id = new_key_id.to_string();
            }
            VerificationRelationship::Embedded(vm) if vm.id == old_key_id => {
                // Embedded keys: update the ID (content still references old key)
                vm.id = new_key_id.to_string();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::did::DidDocument;

    fn create_test_did_doc() -> (DidDocument, Vec<u8>) {
        let signer = Ed25519Signer::generate();
        let secret_key = signer.secret_key_bytes().to_vec();
        let public_key = signer.public_key_bytes();

        let doc = DidDocument::from_key_ed25519(&public_key).unwrap();
        (doc, secret_key)
    }

    #[test]
    fn test_key_rotation_basic() {
        let (mut doc, _old_secret) = create_test_did_doc();
        let did = doc.id.clone();
        let old_key_id = doc.verification_method[0].id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        // Generate new key
        let (new_vm, new_secret) = generate_rotation_key(&did, "key-2").unwrap();
        let rotation = KeyRotation::new(new_vm, new_secret)
            .with_transition_period(30)
            .with_reason(KeyRotationReason::Scheduled);

        let record = registry.rotate_key(&mut doc, rotation).unwrap();

        assert_eq!(record.old_key_id, old_key_id);
        assert!(record.new_key_id.contains("key-2"));
        assert!(record.revocation_date.is_some());
        assert_eq!(record.reason, Some(KeyRotationReason::Scheduled));

        // New key should be in the document
        assert!(doc.get_verification_method(&record.new_key_id).is_some());

        // Verification relationships should now reference new key
        let auth_refs: Vec<&str> = doc
            .authentication
            .iter()
            .filter_map(|rel| match rel {
                VerificationRelationship::Reference(id) => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert!(auth_refs.contains(&record.new_key_id.as_str()));
    }

    #[test]
    fn test_key_rotation_no_transition_period() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();
        let old_key_id = doc.verification_method[0].id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        let (new_vm, new_secret) = generate_rotation_key(&did, "key-2").unwrap();
        let rotation = KeyRotation::new(new_vm, new_secret).with_transition_period(0);

        let _record = registry.rotate_key(&mut doc, rotation).unwrap();

        // Old key should be immediately removed
        assert!(doc.get_verification_method(&old_key_id).is_none());
    }

    #[test]
    fn test_is_key_valid_during_transition() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();
        let old_key_id = doc.verification_method[0].id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        let (new_vm, new_secret) = generate_rotation_key(&did, "key-2").unwrap();
        let rotation = KeyRotation::new(new_vm, new_secret).with_transition_period(30);

        let _record = registry.rotate_key(&mut doc, rotation).unwrap();

        // Old key should still be valid (within transition period)
        assert!(registry.is_key_valid(&old_key_id, None));

        // Old key should be invalid after transition period
        let past_transition = Utc::now() + Duration::days(31);
        assert!(!registry.is_key_valid(&old_key_id, Some(past_transition)));
    }

    #[test]
    fn test_rotation_history() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        // Rotate twice
        let (vm1, secret1) = generate_rotation_key(&did, "key-2").unwrap();
        registry
            .rotate_key(&mut doc, KeyRotation::new(vm1, secret1))
            .unwrap();

        // Need to update registry's current key before second rotation
        let (vm2, secret2) = generate_rotation_key(&did, "key-3").unwrap();
        registry
            .rotate_key(&mut doc, KeyRotation::new(vm2, secret2))
            .unwrap();

        assert_eq!(registry.rotation_count(), 2);
        assert_eq!(registry.get_rotation_history().len(), 2);
    }

    #[test]
    fn test_get_current_key() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        let (new_vm, new_secret) = generate_rotation_key(&did, "key-2").unwrap();
        let new_key_id = new_vm.id.clone();
        registry
            .rotate_key(&mut doc, KeyRotation::new(new_vm, new_secret))
            .unwrap();

        let current = registry.get_current_key(&did).unwrap();
        assert_eq!(current.id, new_key_id);
    }

    #[test]
    fn test_apply_pending_revocations() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();
        let old_key_id = doc.verification_method[0].id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        let (new_vm, new_secret) = generate_rotation_key(&did, "key-2").unwrap();
        let rotation = KeyRotation::new(new_vm, new_secret).with_transition_period(30);
        registry.rotate_key(&mut doc, rotation).unwrap();

        // Old key should still be in doc (within transition period)
        assert!(doc.get_verification_method(&old_key_id).is_some());

        // Apply revocations as if 31 days have passed
        let future_time = Utc::now() + Duration::days(31);
        registry.apply_pending_revocations(&mut doc, Some(future_time));

        // Old key should now be removed from doc
        assert!(doc.get_verification_method(&old_key_id).is_none());
    }

    #[test]
    fn test_rotation_chain() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();
        let initial_key_id = doc.verification_method[0].id.clone();

        let mut registry = KeyRotationRegistry::new();
        registry.register_initial_key(&did, doc.verification_method[0].clone());

        let (vm1, s1) = generate_rotation_key(&did, "key-2").unwrap();
        registry
            .rotate_key(&mut doc, KeyRotation::new(vm1, s1))
            .unwrap();

        let (vm2, s2) = generate_rotation_key(&did, "key-3").unwrap();
        registry
            .rotate_key(&mut doc, KeyRotation::new(vm2, s2))
            .unwrap();

        let chain = registry.get_key_rotation_chain(&initial_key_id);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_generate_rotation_key() {
        let did = Did::new("did:key:z6Mk123").unwrap();
        let (vm, secret) = generate_rotation_key(&did, "key-2").unwrap();

        assert!(vm.id.contains("key-2"));
        assert_eq!(vm.controller, did.as_str());
        assert_eq!(secret.len(), 32);
        assert!(vm.public_key_multibase.is_some());
    }

    #[test]
    fn test_duplicate_key_rotation_error() {
        let (mut doc, _) = create_test_did_doc();
        let did = doc.id.clone();

        let mut registry = KeyRotationRegistry::new();
        let original_vm = doc.verification_method[0].clone();
        registry.register_initial_key(&did, original_vm.clone());

        // Try to rotate to the same key ID
        let rotation = KeyRotation::new(original_vm, vec![0u8; 32]);
        let result = registry.rotate_key(&mut doc, rotation);
        assert!(result.is_err());
    }
}
