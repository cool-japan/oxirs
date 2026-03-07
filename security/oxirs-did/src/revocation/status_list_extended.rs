//! Extended tests and `StatusListCredential` wrapper for StatusList2021
//!
//! Provides:
//! - `StatusListCredential` – wraps a `StatusList2021` as a named credential
//! - Additional test coverage for bitstring edge cases and batch operations

use crate::revocation::status_list::{CredentialStatus, StatusList2021, StatusPurpose};
use crate::{DidError, DidResult};
use serde::{Deserialize, Serialize};

// ── StatusListCredential ───────────────────────────────────────────────────────

/// A Verifiable Credential wrapping a `StatusList2021` bitstring.
///
/// This type provides a structured view over the JSON-LD credential produced
/// by `StatusList2021::to_credential()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusListCredential {
    /// Unique credential identifier (URL where the status list is hosted)
    pub id: String,
    /// Issuer DID or URL
    pub issuer: String,
    /// The underlying status list
    pub status_list: StatusList2021Inner,
}

/// Inner representation of the status list's bitstring and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusList2021Inner {
    /// Size (number of entries) in the bitstring
    pub size: usize,
    /// Status purpose
    pub purpose: StatusPurpose,
    /// Encoded bitstring (base64url of packed bytes)
    pub encoded_list: String,
}

impl StatusListCredential {
    /// Create a new `StatusListCredential` from a `StatusList2021`.
    pub fn from_status_list(sl: &StatusList2021) -> DidResult<Self> {
        let encoded_list = sl.encode_bitstring()?;
        Ok(Self {
            id: sl.id.clone(),
            issuer: sl.issuer.clone(),
            status_list: StatusList2021Inner {
                size: sl.size(),
                purpose: sl.purpose,
                encoded_list,
            },
        })
    }

    /// Serialize to a W3C-compatible JSON-LD document.
    pub fn to_json_ld(&self) -> DidResult<serde_json::Value> {
        Ok(serde_json::json!({
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://w3id.org/vc/status-list/2021/v1"
            ],
            "id": self.id,
            "type": ["VerifiableCredential", "StatusList2021Credential"],
            "issuer": self.issuer,
            "credentialSubject": {
                "id": format!("{}#list", self.id),
                "type": "StatusList2021",
                "statusPurpose": self.status_list.purpose.as_str(),
                "encodedList": self.status_list.encoded_list
            }
        }))
    }

    /// Return the number of entries in this credential's status list.
    pub fn size(&self) -> usize {
        self.status_list.size
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::revocation::status_list::{RevocationRegistry, MIN_LIST_SIZE};

    fn small_list(size: usize) -> StatusList2021 {
        StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", size).unwrap()
    }

    // ── StatusListCredential ─────────────────────────────────────────────

    #[test]
    fn test_status_list_credential_from_status_list() {
        let sl = small_list(1000);
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        assert_eq!(cred.id, "https://example.com/status/1");
        assert_eq!(cred.issuer, "did:key:z6Mk");
        assert_eq!(cred.size(), 1000);
    }

    #[test]
    fn test_status_list_credential_to_json_ld_type() {
        let sl = small_list(500);
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        let doc = cred.to_json_ld().unwrap();
        let types = doc["type"].as_array().unwrap();
        assert!(types
            .iter()
            .any(|t| t.as_str() == Some("StatusList2021Credential")));
    }

    #[test]
    fn test_status_list_credential_json_ld_id() {
        let sl = small_list(100);
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        let doc = cred.to_json_ld().unwrap();
        assert_eq!(doc["id"].as_str().unwrap(), "https://example.com/status/1");
    }

    #[test]
    fn test_status_list_credential_json_ld_encoded_list_is_string() {
        let mut sl = small_list(1024);
        sl.set_status(5, true).unwrap();
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        let doc = cred.to_json_ld().unwrap();
        assert!(doc["credentialSubject"]["encodedList"].is_string());
    }

    #[test]
    fn test_status_list_credential_purpose_revocation() {
        let sl = StatusList2021::new_with_purpose(
            "https://example.com/status/1",
            "did:key:z6Mk",
            1000,
            StatusPurpose::Revocation,
        )
        .unwrap();
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        let doc = cred.to_json_ld().unwrap();
        assert_eq!(
            doc["credentialSubject"]["statusPurpose"].as_str().unwrap(),
            "revocation"
        );
    }

    #[test]
    fn test_status_list_credential_purpose_suspension() {
        let sl = StatusList2021::new_with_purpose(
            "https://example.com/status/1",
            "did:key:z6Mk",
            1000,
            StatusPurpose::Suspension,
        )
        .unwrap();
        let cred = StatusListCredential::from_status_list(&sl).unwrap();
        assert_eq!(cred.status_list.purpose, StatusPurpose::Suspension);
    }

    // ── StatusList2021 edge-case coverage ────────────────────────────────

    #[test]
    fn test_status_list_first_index() {
        let mut list = small_list(100);
        list.set_status(0, true).unwrap();
        assert!(list.is_revoked(0).unwrap());
    }

    #[test]
    fn test_status_list_last_index() {
        let mut list = small_list(100);
        list.set_status(99, true).unwrap();
        assert!(list.is_revoked(99).unwrap());
    }

    #[test]
    fn test_status_list_all_set() {
        let mut list = small_list(16);
        for i in 0..16 {
            list.set_status(i, true).unwrap();
        }
        assert_eq!(list.revoked_count(), 16);
    }

    #[test]
    fn test_status_list_none_set() {
        let list = small_list(1000);
        assert_eq!(list.revoked_count(), 0);
    }

    #[test]
    fn test_status_list_toggle_revoke_reinstate() {
        let mut list = small_list(100);
        list.set_status(50, true).unwrap();
        assert!(list.is_revoked(50).unwrap());
        list.set_status(50, false).unwrap();
        assert!(!list.is_revoked(50).unwrap());
        list.set_status(50, true).unwrap();
        assert!(list.is_revoked(50).unwrap());
    }

    #[test]
    fn test_status_list_bitstring_encode_all_zeros() {
        let list = small_list(8);
        let encoded = list.encode_bitstring().unwrap();
        // Decoded should be a byte of 0x00 → all false
        let decoded = StatusList2021::decode_bitstring(&encoded, 8).unwrap();
        assert!(decoded.iter().all(|&b| !b));
    }

    #[test]
    fn test_status_list_bitstring_encode_all_ones() {
        let mut list = small_list(8);
        for i in 0..8 {
            list.set_status(i, true).unwrap();
        }
        let encoded = list.encode_bitstring().unwrap();
        let decoded = StatusList2021::decode_bitstring(&encoded, 8).unwrap();
        assert!(decoded.iter().all(|&b| b));
    }

    #[test]
    fn test_status_list_decode_bitstring_invalid_base64() {
        let result = StatusList2021::decode_bitstring("!!!invalid!!!", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_status_list_batch_set_and_verify() {
        let mut list = small_list(200);
        let batch: Vec<(usize, bool)> = (0..20).map(|i| (i * 5, true)).collect();
        list.set_batch(&batch).unwrap();
        for i in 0..20 {
            assert!(
                list.is_revoked(i * 5).unwrap(),
                "Index {} should be revoked",
                i * 5
            );
        }
    }

    #[test]
    fn test_status_list_batch_out_of_bounds() {
        let mut list = small_list(10);
        let result = list.set_batch(&[(5, true), (99, true)]); // 99 is out of bounds
        assert!(result.is_err());
    }

    #[test]
    fn test_status_list_revoked_indices_order() {
        let mut list = small_list(100);
        list.set_status(10, true).unwrap();
        list.set_status(1, true).unwrap();
        list.set_status(50, true).unwrap();
        let indices = list.revoked_indices();
        // Should be in ascending order
        assert_eq!(indices, vec![1, 10, 50]);
    }

    // ── RevocationRegistry edge cases ─────────────────────────────────────

    #[test]
    fn test_registry_capacity_minimum_size() {
        // Capacity below MIN_LIST_SIZE should be bumped to MIN_LIST_SIZE
        let registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            100, // below MIN_LIST_SIZE
        )
        .unwrap();
        // Registry should still work (the implementation uses capacity.max(MIN_LIST_SIZE))
        assert_eq!(registry.registered_count(), 0);
    }

    #[test]
    fn test_registry_register_many_credentials() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();
        for i in 0..50 {
            registry
                .register_credential(&format!("urn:uuid:cred-{i}"))
                .unwrap();
        }
        assert_eq!(registry.registered_count(), 50);
    }

    #[test]
    fn test_registry_revoke_all_registered() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();
        for i in 0..5 {
            registry
                .register_credential(&format!("urn:uuid:cred-{i}"))
                .unwrap();
        }
        for i in 0..5 {
            registry.revoke(&format!("urn:uuid:cred-{i}")).unwrap();
        }
        assert_eq!(registry.revoked_count(), 5);
    }

    #[test]
    fn test_registry_is_revoked_at_index() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();
        let status = registry.register_credential("urn:uuid:cred-1").unwrap();
        let idx = status.index().unwrap();
        assert!(!registry.is_revoked_at_index(idx).unwrap());
        registry.revoke("urn:uuid:cred-1").unwrap();
        assert!(registry.is_revoked_at_index(idx).unwrap());
    }

    #[test]
    fn test_registry_reinstate_after_revoke() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();
        registry.register_credential("urn:uuid:cred-1").unwrap();
        registry.revoke("urn:uuid:cred-1").unwrap();
        assert!(registry.is_revoked("urn:uuid:cred-1").unwrap());
        registry.reinstate("urn:uuid:cred-1").unwrap();
        assert!(!registry.is_revoked("urn:uuid:cred-1").unwrap());
        assert_eq!(registry.revoked_count(), 0);
    }

    #[test]
    fn test_registry_credential_not_registered_revoke_fails() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();
        assert!(registry.revoke("urn:uuid:unknown").is_err());
    }

    #[test]
    fn test_credential_status_entry_fields() {
        let status = CredentialStatus::new_status_list_2021(
            "https://example.com/status/1",
            77,
            StatusPurpose::Revocation,
        );
        assert_eq!(status.status_list_index, "77");
        assert_eq!(status.index().unwrap(), 77);
        assert_eq!(status.status_purpose, "revocation");
        assert_eq!(status.id, "https://example.com/status/1#77");
    }

    #[test]
    fn test_credential_status_invalid_index_parse() {
        let status = CredentialStatus {
            id: "id".to_string(),
            status_type: "StatusList2021Entry".to_string(),
            status_list_index: "not-a-number".to_string(),
            status_list_credential: "url".to_string(),
            status_purpose: "revocation".to_string(),
        };
        assert!(status.index().is_err());
    }
}
