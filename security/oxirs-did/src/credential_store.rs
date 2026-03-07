//! # Credential Store
//!
//! An in-memory W3C Verifiable Credential store providing CRUD operations,
//! revocation tracking, status resolution, and flexible filtering.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::credential_store::{
//!     CredentialStore, CredentialSubject, VerifiableCredential,
//! };
//! use std::collections::HashMap;
//!
//! let mut store = CredentialStore::new();
//!
//! let subject = CredentialSubject {
//!     id: "did:example:alice".to_string(),
//!     claims: HashMap::from([("name".to_string(), "Alice".to_string())]),
//! };
//!
//! let vc = VerifiableCredential {
//!     id: "urn:vc:001".to_string(),
//!     types: vec!["VerifiableCredential".to_string()],
//!     issuer: "did:example:issuer".to_string(),
//!     issuance_date: "2024-01-01".to_string(),
//!     expiration_date: None,
//!     subject,
//!     proof: None,
//! };
//!
//! store.store(vc).expect("store failed");
//! assert_eq!(store.count(), 1);
//! ```

use std::collections::{HashMap, HashSet};

// ─── Domain types ─────────────────────────────────────────────────────────────

/// Credential subject: the entity the credential is about
#[derive(Debug, Clone, PartialEq)]
pub struct CredentialSubject {
    /// DID or identifier of the subject
    pub id: String,
    /// Arbitrary string-valued claims about the subject
    pub claims: HashMap<String, String>,
}

/// A W3C Verifiable Credential
#[derive(Debug, Clone)]
pub struct VerifiableCredential {
    /// Unique credential identifier (URI)
    pub id: String,
    /// Credential types (must include `"VerifiableCredential"`)
    pub types: Vec<String>,
    /// Issuer DID
    pub issuer: String,
    /// ISO 8601 issuance date string (e.g. `"2024-01-15"`)
    pub issuance_date: String,
    /// Optional ISO 8601 expiration date string
    pub expiration_date: Option<String>,
    /// The credential subject
    pub subject: CredentialSubject,
    /// Optional compact proof / JWS string
    pub proof: Option<String>,
}

/// The resolved status of a credential at a given point in time
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialStatus {
    /// Credential is valid and not expired
    Valid,
    /// Credential's expiration date is in the past (compared to `current_date`)
    Expired,
    /// Credential has been explicitly revoked
    Revoked,
    /// Credential is not yet active / issuance date is in the future
    Pending,
}

/// Filter criteria for searching credentials
///
/// Each `Some(value)` field acts as an AND-filter; `None` fields are ignored.
#[derive(Debug, Clone, Default)]
pub struct CredentialFilter {
    /// Match credentials issued by this DID (exact)
    pub issuer: Option<String>,
    /// Match credentials whose subject id equals this value
    pub subject_id: Option<String>,
    /// Match credentials that include this type string
    pub credential_type: Option<String>,
    /// Match credentials with this resolved status
    pub status: Option<CredentialStatus>,
}

/// Errors returned by [`CredentialStore`] operations
#[derive(Debug)]
pub enum StoreError {
    /// A credential with the same `id` already exists in the store
    DuplicateId(String),
    /// The credential failed a basic validity check
    InvalidCredential(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::DuplicateId(id) => write!(f, "duplicate credential id: {}", id),
            StoreError::InvalidCredential(msg) => write!(f, "invalid credential: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

// ─── Credential store ─────────────────────────────────────────────────────────

/// In-memory store for Verifiable Credentials
pub struct CredentialStore {
    credentials: HashMap<String, VerifiableCredential>,
    revoked: HashSet<String>,
}

impl Default for CredentialStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialStore {
    /// Create an empty credential store.
    pub fn new() -> Self {
        Self {
            credentials: HashMap::new(),
            revoked: HashSet::new(),
        }
    }

    // ── Write operations ─────────────────────────────────────────────────

    /// Store a new credential.
    ///
    /// Returns [`StoreError::DuplicateId`] if a credential with the same `id`
    /// already exists, or [`StoreError::InvalidCredential`] if the `id` is empty.
    pub fn store(&mut self, vc: VerifiableCredential) -> Result<(), StoreError> {
        if vc.id.is_empty() {
            return Err(StoreError::InvalidCredential(
                "credential id must not be empty".to_string(),
            ));
        }
        if self.credentials.contains_key(&vc.id) {
            return Err(StoreError::DuplicateId(vc.id.clone()));
        }
        self.credentials.insert(vc.id.clone(), vc);
        Ok(())
    }

    /// Revoke the credential with the given `id`.
    ///
    /// Returns `true` if the credential exists (and was newly revoked or was already
    /// revoked), `false` if the credential is not found.
    pub fn revoke(&mut self, id: &str) -> bool {
        if self.credentials.contains_key(id) {
            self.revoked.insert(id.to_string());
            true
        } else {
            false
        }
    }

    // ── Read operations ──────────────────────────────────────────────────

    /// Retrieve a credential by id.
    pub fn get(&self, id: &str) -> Option<&VerifiableCredential> {
        self.credentials.get(id)
    }

    /// Returns `true` if the credential with `id` has been revoked.
    pub fn is_revoked(&self, id: &str) -> bool {
        self.revoked.contains(id)
    }

    /// Resolve the [`CredentialStatus`] of a credential.
    ///
    /// # Arguments
    /// * `id`           – credential identifier
    /// * `current_date` – ISO 8601 date string representing "now" (e.g. `"2025-06-01"`)
    ///
    /// Returns `None` if no credential with that id exists.
    pub fn status(&self, id: &str, current_date: &str) -> Option<CredentialStatus> {
        let vc = self.credentials.get(id)?;

        if self.revoked.contains(id) {
            return Some(CredentialStatus::Revoked);
        }

        // Pending: issuance date is strictly after current_date (lexicographic ISO 8601 compare)
        if vc.issuance_date.as_str() > current_date {
            return Some(CredentialStatus::Pending);
        }

        // Expired: expiration date is strictly before current_date
        if let Some(ref exp) = vc.expiration_date {
            if exp.as_str() < current_date {
                return Some(CredentialStatus::Expired);
            }
        }

        Some(CredentialStatus::Valid)
    }

    // ── Search ───────────────────────────────────────────────────────────

    /// Search credentials matching *all* non-None fields of `filter`.
    ///
    /// `current_date` is used to resolve the status field of the filter.
    pub fn search<'a>(
        &'a self,
        filter: &CredentialFilter,
        current_date: &str,
    ) -> Vec<&'a VerifiableCredential> {
        self.credentials
            .values()
            .filter(|vc| {
                if let Some(ref issuer) = filter.issuer {
                    if &vc.issuer != issuer {
                        return false;
                    }
                }
                if let Some(ref subject_id) = filter.subject_id {
                    if &vc.subject.id != subject_id {
                        return false;
                    }
                }
                if let Some(ref cred_type) = filter.credential_type {
                    if !vc.types.contains(cred_type) {
                        return false;
                    }
                }
                if let Some(ref required_status) = filter.status {
                    let resolved = self.status(&vc.id, current_date);
                    if resolved.as_ref() != Some(required_status) {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Return all credentials issued by `issuer`.
    pub fn credentials_by_issuer(&self, issuer: &str) -> Vec<&VerifiableCredential> {
        self.credentials
            .values()
            .filter(|vc| vc.issuer == issuer)
            .collect()
    }

    /// Return all credentials whose subject id equals `subject_id`.
    pub fn credentials_for_subject(&self, subject_id: &str) -> Vec<&VerifiableCredential> {
        self.credentials
            .values()
            .filter(|vc| vc.subject.id == subject_id)
            .collect()
    }

    // ── Counters ─────────────────────────────────────────────────────────

    /// Total number of stored credentials (including revoked).
    pub fn count(&self) -> usize {
        self.credentials.len()
    }

    /// Number of revoked credential ids tracked by the store.
    pub fn revoked_count(&self) -> usize {
        self.revoked.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vc(id: &str, issuer: &str, subject_id: &str) -> VerifiableCredential {
        VerifiableCredential {
            id: id.to_string(),
            types: vec!["VerifiableCredential".to_string()],
            issuer: issuer.to_string(),
            issuance_date: "2024-01-01".to_string(),
            expiration_date: None,
            subject: CredentialSubject {
                id: subject_id.to_string(),
                claims: HashMap::new(),
            },
            proof: None,
        }
    }

    fn make_vc_with_expiry(id: &str, issuer: &str, subject_id: &str, expiry: &str) -> VerifiableCredential {
        let mut vc = make_vc(id, issuer, subject_id);
        vc.expiration_date = Some(expiry.to_string());
        vc
    }

    fn today() -> &'static str {
        "2025-06-01"
    }

    // ── store / get ────────────────────────────────────────────────────────

    #[test]
    fn test_store_and_get() {
        let mut store = CredentialStore::new();
        let vc = make_vc("urn:vc:1", "did:issuer:1", "did:alice");
        store.store(vc).expect("store failed");
        let fetched = store.get("urn:vc:1");
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().issuer, "did:issuer:1");
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let store = CredentialStore::new();
        assert!(store.get("urn:vc:nope").is_none());
    }

    #[test]
    fn test_store_multiple() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i", "s")).unwrap();
        store.store(make_vc("2", "i", "s")).unwrap();
        assert_eq!(store.count(), 2);
    }

    // ── Duplicate id error ─────────────────────────────────────────────────

    #[test]
    fn test_duplicate_id_returns_error() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:x", "i", "s")).unwrap();
        let result = store.store(make_vc("urn:vc:x", "i2", "s2"));
        assert!(result.is_err());
        match result.unwrap_err() {
            StoreError::DuplicateId(id) => assert_eq!(id, "urn:vc:x"),
            _ => panic!("expected DuplicateId"),
        }
    }

    #[test]
    fn test_empty_id_returns_invalid_credential_error() {
        let mut store = CredentialStore::new();
        let result = store.store(make_vc("", "i", "s"));
        assert!(result.is_err());
    }

    // ── revoke / is_revoked ────────────────────────────────────────────────

    #[test]
    fn test_revoke_existing() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:r", "i", "s")).unwrap();
        let ok = store.revoke("urn:vc:r");
        assert!(ok);
        assert!(store.is_revoked("urn:vc:r"));
    }

    #[test]
    fn test_revoke_nonexistent_returns_false() {
        let mut store = CredentialStore::new();
        assert!(!store.revoke("urn:vc:ghost"));
    }

    #[test]
    fn test_is_revoked_not_revoked() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:ok", "i", "s")).unwrap();
        assert!(!store.is_revoked("urn:vc:ok"));
    }

    #[test]
    fn test_revoke_twice_still_revoked() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:2x", "i", "s")).unwrap();
        store.revoke("urn:vc:2x");
        store.revoke("urn:vc:2x");
        assert!(store.is_revoked("urn:vc:2x"));
    }

    // ── status ─────────────────────────────────────────────────────────────

    #[test]
    fn test_status_valid() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:v", "i", "s")).unwrap();
        let status = store.status("urn:vc:v", today());
        assert_eq!(status, Some(CredentialStatus::Valid));
    }

    #[test]
    fn test_status_revoked() {
        let mut store = CredentialStore::new();
        store.store(make_vc("urn:vc:rv", "i", "s")).unwrap();
        store.revoke("urn:vc:rv");
        let status = store.status("urn:vc:rv", today());
        assert_eq!(status, Some(CredentialStatus::Revoked));
    }

    #[test]
    fn test_status_expired() {
        let mut store = CredentialStore::new();
        store
            .store(make_vc_with_expiry("urn:vc:exp", "i", "s", "2020-01-01"))
            .unwrap();
        let status = store.status("urn:vc:exp", today());
        assert_eq!(status, Some(CredentialStatus::Expired));
    }

    #[test]
    fn test_status_not_yet_expired() {
        let mut store = CredentialStore::new();
        store
            .store(make_vc_with_expiry("urn:vc:future", "i", "s", "2099-12-31"))
            .unwrap();
        let status = store.status("urn:vc:future", today());
        assert_eq!(status, Some(CredentialStatus::Valid));
    }

    #[test]
    fn test_status_nonexistent_returns_none() {
        let store = CredentialStore::new();
        assert!(store.status("urn:vc:nope", today()).is_none());
    }

    #[test]
    fn test_status_pending_future_issuance() {
        let mut store = CredentialStore::new();
        let mut vc = make_vc("urn:vc:pend", "i", "s");
        vc.issuance_date = "2099-01-01".to_string();
        store.store(vc).unwrap();
        let status = store.status("urn:vc:pend", today());
        assert_eq!(status, Some(CredentialStatus::Pending));
    }

    #[test]
    fn test_status_revoked_overrides_expired() {
        let mut store = CredentialStore::new();
        store
            .store(make_vc_with_expiry("urn:vc:re", "i", "s", "2020-01-01"))
            .unwrap();
        store.revoke("urn:vc:re");
        // Revoked should take priority
        let status = store.status("urn:vc:re", today());
        assert_eq!(status, Some(CredentialStatus::Revoked));
    }

    // ── search ─────────────────────────────────────────────────────────────

    #[test]
    fn test_search_by_issuer() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "issuer-A", "alice")).unwrap();
        store.store(make_vc("2", "issuer-B", "bob")).unwrap();
        let filter = CredentialFilter {
            issuer: Some("issuer-A".to_string()),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].issuer, "issuer-A");
    }

    #[test]
    fn test_search_by_subject_id() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i", "did:alice")).unwrap();
        store.store(make_vc("2", "i", "did:bob")).unwrap();
        let filter = CredentialFilter {
            subject_id: Some("did:alice".to_string()),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject.id, "did:alice");
    }

    #[test]
    fn test_search_by_credential_type() {
        let mut store = CredentialStore::new();
        let mut vc = make_vc("1", "i", "s");
        vc.types.push("DriversLicense".to_string());
        store.store(vc).unwrap();
        store.store(make_vc("2", "i", "s2")).unwrap();

        let filter = CredentialFilter {
            credential_type: Some("DriversLicense".to_string()),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_status_valid() {
        let mut store = CredentialStore::new();
        store.store(make_vc("v", "i", "s")).unwrap();
        store.store(make_vc("r", "i", "s2")).unwrap();
        store.revoke("r");
        let filter = CredentialFilter {
            status: Some(CredentialStatus::Valid),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert!(results.iter().all(|vc| vc.id == "v"));
    }

    #[test]
    fn test_search_none_filter_matches_all() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i1", "s1")).unwrap();
        store.store(make_vc("2", "i2", "s2")).unwrap();
        let filter = CredentialFilter::default();
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_combined_filters() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "issuer-A", "did:alice")).unwrap();
        store.store(make_vc("2", "issuer-A", "did:bob")).unwrap();
        store.store(make_vc("3", "issuer-B", "did:alice")).unwrap();
        let filter = CredentialFilter {
            issuer: Some("issuer-A".to_string()),
            subject_id: Some("did:alice".to_string()),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    // ── credentials_by_issuer / credentials_for_subject ──────────────────

    #[test]
    fn test_credentials_by_issuer() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "iss-A", "s")).unwrap();
        store.store(make_vc("2", "iss-A", "s2")).unwrap();
        store.store(make_vc("3", "iss-B", "s3")).unwrap();
        let by_a = store.credentials_by_issuer("iss-A");
        assert_eq!(by_a.len(), 2);
        assert!(by_a.iter().all(|vc| vc.issuer == "iss-A"));
    }

    #[test]
    fn test_credentials_by_issuer_not_found() {
        let store = CredentialStore::new();
        assert!(store.credentials_by_issuer("nobody").is_empty());
    }

    #[test]
    fn test_credentials_for_subject() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i1", "did:alice")).unwrap();
        store.store(make_vc("2", "i2", "did:alice")).unwrap();
        store.store(make_vc("3", "i3", "did:bob")).unwrap();
        let for_alice = store.credentials_for_subject("did:alice");
        assert_eq!(for_alice.len(), 2);
    }

    #[test]
    fn test_credentials_for_subject_not_found() {
        let store = CredentialStore::new();
        assert!(store.credentials_for_subject("did:nobody").is_empty());
    }

    // ── count / revoked_count ──────────────────────────────────────────────

    #[test]
    fn test_count_empty_store() {
        let store = CredentialStore::new();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_count_after_stores() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i", "s")).unwrap();
        store.store(make_vc("2", "i", "s")).unwrap();
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_revoked_count_initial_zero() {
        let store = CredentialStore::new();
        assert_eq!(store.revoked_count(), 0);
    }

    #[test]
    fn test_revoked_count_after_revocations() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i", "s")).unwrap();
        store.store(make_vc("2", "i", "s")).unwrap();
        store.revoke("1");
        store.revoke("2");
        assert_eq!(store.revoked_count(), 2);
    }

    #[test]
    fn test_revoked_count_does_not_double_count() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "i", "s")).unwrap();
        store.revoke("1");
        store.revoke("1"); // second revoke
        assert_eq!(store.revoked_count(), 1);
    }

    // ── expiration date comparison ─────────────────────────────────────────

    #[test]
    fn test_expiration_boundary_same_day_valid() {
        let mut store = CredentialStore::new();
        // Expiry on the same day as current_date → expiry >= current_date → Valid
        store
            .store(make_vc_with_expiry("urn:vc:bd", "i", "s", today()))
            .unwrap();
        let status = store.status("urn:vc:bd", today());
        assert_eq!(status, Some(CredentialStatus::Valid));
    }

    #[test]
    fn test_expiration_one_day_past() {
        let mut store = CredentialStore::new();
        store
            .store(make_vc_with_expiry("urn:vc:past", "i", "s", "2025-05-31"))
            .unwrap();
        let status = store.status("urn:vc:past", today());
        assert_eq!(status, Some(CredentialStatus::Expired));
    }

    // ── CredentialSubject with claims ──────────────────────────────────────

    #[test]
    fn test_credential_subject_claims() {
        let mut claims = HashMap::new();
        claims.insert("degree".to_string(), "PhD".to_string());
        let subject = CredentialSubject {
            id: "did:example:alice".to_string(),
            claims,
        };
        assert_eq!(subject.claims.get("degree"), Some(&"PhD".to_string()));
    }

    // ── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn test_credential_store_default_is_empty() {
        let store = CredentialStore::default();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_vc_has_verifiable_credential_type() {
        let vc = make_vc("id1", "iss", "sub");
        assert!(vc.types.contains(&"VerifiableCredential".to_string()));
    }

    #[test]
    fn test_search_empty_store() {
        let store = CredentialStore::new();
        let filter = CredentialFilter::default();
        assert!(store.search(&filter, today()).is_empty());
    }

    #[test]
    fn test_search_revoked_status() {
        let mut store = CredentialStore::new();
        store.store(make_vc("a", "iss", "sub")).unwrap();
        store.revoke("a");
        let filter = CredentialFilter {
            status: Some(CredentialStatus::Revoked),
            ..Default::default()
        };
        let results = store.search(&filter, today());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_get_returns_correct_issuer() {
        let mut store = CredentialStore::new();
        store.store(make_vc("id-x", "iss-x", "sub-x")).unwrap();
        let vc = store.get("id-x").expect("should exist");
        assert_eq!(vc.issuer, "iss-x");
        assert_eq!(vc.subject.id, "sub-x");
    }

    #[test]
    fn test_verifiable_credential_with_proof() {
        let mut vc = make_vc("proof-vc", "iss", "sub");
        vc.proof = Some("jws.payload.sig".to_string());
        let mut store = CredentialStore::new();
        store.store(vc).unwrap();
        let fetched = store.get("proof-vc").expect("should exist");
        assert!(fetched.proof.is_some());
    }

    #[test]
    fn test_verifiable_credential_without_proof() {
        let vc = make_vc("no-proof-vc", "iss", "sub");
        let mut store = CredentialStore::new();
        store.store(vc).unwrap();
        let fetched = store.get("no-proof-vc").expect("should exist");
        assert!(fetched.proof.is_none());
    }

    #[test]
    fn test_multiple_types_on_credential() {
        let mut vc = make_vc("multi-type", "iss", "sub");
        vc.types.push("UniversityDegreeCredential".to_string());
        let mut store = CredentialStore::new();
        store.store(vc).unwrap();
        let fetched = store.get("multi-type").expect("should exist");
        assert_eq!(fetched.types.len(), 2);
    }

    #[test]
    fn test_store_error_display_duplicate() {
        let err = StoreError::DuplicateId("test-id".to_string());
        let msg = err.to_string();
        assert!(msg.contains("test-id"));
    }

    #[test]
    fn test_store_error_display_invalid() {
        let err = StoreError::InvalidCredential("missing field".to_string());
        let msg = err.to_string();
        assert!(msg.contains("missing field"));
    }

    #[test]
    fn test_credentials_for_subject_multiple_issuers() {
        let mut store = CredentialStore::new();
        store.store(make_vc("1", "iss1", "did:alice")).unwrap();
        store.store(make_vc("2", "iss2", "did:alice")).unwrap();
        store.store(make_vc("3", "iss3", "did:bob")).unwrap();
        let for_alice = store.credentials_for_subject("did:alice");
        assert_eq!(for_alice.len(), 2);
        assert!(for_alice.iter().all(|vc| vc.subject.id == "did:alice"));
    }
}
