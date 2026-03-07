//! # Verifiable Credential Presenter
//!
//! VP (Verifiable Presentation) construction, credential selection, selective
//! disclosure, validation, and JSON serialisation following the W3C VC Data
//! Model 1.1 specification.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::vc_presenter::{
//!     VcPresenter, VerifiableCredential, CredentialSubject, VpRequest,
//! };
//! use std::collections::HashMap;
//!
//! let subject = CredentialSubject {
//!     id: "did:example:alice".to_string(),
//!     claims: HashMap::from([("name".to_string(), "Alice".to_string())]),
//! };
//! let vc = VerifiableCredential {
//!     id: "urn:vc:001".to_string(),
//!     types: vec!["VerifiableCredential".to_string(), "AlumniCredential".to_string()],
//!     issuer: "did:example:university".to_string(),
//!     issuance_date: "2024-01-01".to_string(),
//!     expiration_date: None,
//!     subject,
//!     proof: None,
//! };
//!
//! let mut presenter = VcPresenter::new("did:example:alice");
//! presenter.add_credential(vc);
//! let request = VpRequest::new("did:example:alice");
//! let vp = presenter.build_presentation(&request).expect("build ok");
//! assert!(vp.is_valid_structure());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the VP presenter
#[derive(Debug, Clone, PartialEq)]
pub enum PresenterError {
    /// No credentials matched the selection criteria
    NoCredentialsSelected,
    /// The holder DID does not match any credential subject
    HolderMismatch { holder: String, subject: String },
    /// The presentation structure is malformed
    InvalidStructure(String),
    /// A credential has expired
    CredentialExpired(String),
    /// JSON serialisation failed
    SerializationError(String),
}

impl fmt::Display for PresenterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PresenterError::NoCredentialsSelected => {
                write!(f, "No credentials matched the selection criteria")
            }
            PresenterError::HolderMismatch { holder, subject } => {
                write!(
                    f,
                    "Holder {holder} does not match credential subject {subject}"
                )
            }
            PresenterError::InvalidStructure(msg) => {
                write!(f, "Invalid presentation structure: {msg}")
            }
            PresenterError::CredentialExpired(id) => {
                write!(f, "Credential {id} has expired")
            }
            PresenterError::SerializationError(msg) => {
                write!(f, "Serialization error: {msg}")
            }
        }
    }
}

impl std::error::Error for PresenterError {}

// ─── Domain types ─────────────────────────────────────────────────────────────

/// Credential subject (the entity the credential is about)
#[derive(Debug, Clone, PartialEq)]
pub struct CredentialSubject {
    /// DID or identifier of the subject
    pub id: String,
    /// Arbitrary string-valued claims
    pub claims: HashMap<String, String>,
}

/// A W3C Verifiable Credential stored inside the presenter
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

impl VerifiableCredential {
    /// Returns `true` if the credential has all mandatory W3C fields.
    pub fn has_valid_structure(&self) -> bool {
        !self.id.is_empty()
            && self.types.contains(&"VerifiableCredential".to_string())
            && !self.issuer.is_empty()
            && !self.issuance_date.is_empty()
            && !self.subject.id.is_empty()
    }

    /// Returns `true` if the credential is expired relative to `current_date`
    /// (compared as string prefix, so "2024-01-01" < "2025-01-01").
    pub fn is_expired(&self, current_date: &str) -> bool {
        match &self.expiration_date {
            Some(exp) => exp.as_str() < current_date,
            None => false,
        }
    }

    /// Return a copy of this credential with only the listed claim keys
    /// retained in the subject.  Structural fields (id, types, issuer, dates)
    /// are always preserved.
    pub fn selective_disclose(&self, keys: &[&str]) -> VerifiableCredential {
        let allowed: HashSet<&str> = keys.iter().copied().collect();
        let filtered_claims: HashMap<String, String> = self
            .subject
            .claims
            .iter()
            .filter(|(k, _)| allowed.contains(k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        VerifiableCredential {
            id: self.id.clone(),
            types: self.types.clone(),
            issuer: self.issuer.clone(),
            issuance_date: self.issuance_date.clone(),
            expiration_date: self.expiration_date.clone(),
            subject: CredentialSubject {
                id: self.subject.id.clone(),
                claims: filtered_claims,
            },
            proof: self.proof.clone(),
        }
    }

    /// Serialize to a JSON object string (no external crate needed)
    pub fn to_json(&self) -> String {
        let types_json: String = self
            .types
            .iter()
            .map(|t| format!(r#""{t}""#))
            .collect::<Vec<_>>()
            .join(", ");

        let claims_json: String = self
            .subject
            .claims
            .iter()
            .map(|(k, v)| {
                let ek = escape_json_string(k);
                let ev = escape_json_string(v);
                format!(r#""{ek}": "{ev}""#)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let proof_json = match &self.proof {
            Some(p) => format!(r#", "proof": "{}""#, escape_json_string(p)),
            None => String::new(),
        };

        let exp_json = match &self.expiration_date {
            Some(d) => format!(r#", "expirationDate": "{d}""#),
            None => String::new(),
        };

        format!(
            r#"{{"id": "{id}", "type": [{types}], "issuer": "{issuer}", "issuanceDate": "{idate}"{exp}, "credentialSubject": {{"id": "{sid}", {claims}}}{proof}}}"#,
            id = escape_json_string(&self.id),
            types = types_json,
            issuer = escape_json_string(&self.issuer),
            idate = self.issuance_date,
            exp = exp_json,
            sid = escape_json_string(&self.subject.id),
            claims = claims_json,
            proof = proof_json,
        )
    }
}

// ─── Presentation Proof ───────────────────────────────────────────────────────

/// Proof stub attached to a Verifiable Presentation
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationProof {
    /// Proof type (e.g. `"Ed25519Signature2020"`)
    pub proof_type: String,
    /// ISO 8601 creation timestamp
    pub created: String,
    /// Verification method reference DID URL
    pub verification_method: String,
    /// Placeholder proof value (base58 / base64url encoded signature)
    pub proof_value: String,
    /// Proof purpose (e.g. `"authentication"`)
    pub proof_purpose: String,
}

impl PresentationProof {
    /// Create a stub proof for a holder DID
    pub fn stub(holder_did: &str) -> Self {
        Self {
            proof_type: "Ed25519Signature2020".to_string(),
            created: "2024-01-01T00:00:00Z".to_string(),
            verification_method: format!("{holder_did}#key-1"),
            proof_value: "zStubProofValue000000000000000000000000000000000".to_string(),
            proof_purpose: "authentication".to_string(),
        }
    }

    /// Serialize to JSON object string
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"type": "{pt}", "created": "{created}", "verificationMethod": "{vm}", "proofPurpose": "{pp}", "proofValue": "{pv}"}}"#,
            pt = self.proof_type,
            created = self.created,
            vm = self.verification_method,
            pp = self.proof_purpose,
            pv = self.proof_value,
        )
    }
}

// ─── Verifiable Presentation ──────────────────────────────────────────────────

/// A W3C Verifiable Presentation
#[derive(Debug, Clone)]
pub struct VerifiablePresentation {
    /// W3C context URIs
    pub context: Vec<String>,
    /// Presentation types (must include `"VerifiablePresentation"`)
    pub types: Vec<String>,
    /// Holder DID
    pub holder: String,
    /// Included Verifiable Credentials
    pub verifiable_credential: Vec<VerifiableCredential>,
    /// Optional presentation proof
    pub proof: Option<PresentationProof>,
}

impl VerifiablePresentation {
    /// Returns `true` if the presentation has valid W3C structure.
    pub fn is_valid_structure(&self) -> bool {
        self.types.contains(&"VerifiablePresentation".to_string())
            && !self.holder.is_empty()
            && !self.verifiable_credential.is_empty()
            && self
                .verifiable_credential
                .iter()
                .all(|vc| vc.has_valid_structure())
    }

    /// Validate structure and optionally check expiry.
    /// Returns a list of error messages; empty list means valid.
    pub fn validate(&self, current_date: Option<&str>) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();

        if !self.types.contains(&"VerifiablePresentation".to_string()) {
            errors.push("Missing type 'VerifiablePresentation'".to_string());
        }

        if self.holder.is_empty() {
            errors.push("Missing holder DID".to_string());
        }

        if self.verifiable_credential.is_empty() {
            errors.push("No verifiable credentials present".to_string());
        }

        for vc in &self.verifiable_credential {
            if !vc.has_valid_structure() {
                errors.push(format!("Credential '{}' has invalid structure", vc.id));
            }
            if let Some(date) = current_date {
                if vc.is_expired(date) {
                    errors.push(format!("Credential '{}' has expired", vc.id));
                }
            }
        }

        errors
    }

    /// Serialize the presentation to a JSON string.
    pub fn to_json(&self) -> String {
        let context_json: String = self
            .context
            .iter()
            .map(|c| format!(r#""{c}""#))
            .collect::<Vec<_>>()
            .join(", ");

        let types_json: String = self
            .types
            .iter()
            .map(|t| format!(r#""{t}""#))
            .collect::<Vec<_>>()
            .join(", ");

        let credentials_json: String = self
            .verifiable_credential
            .iter()
            .map(|vc| vc.to_json())
            .collect::<Vec<_>>()
            .join(", ");

        let proof_json = match &self.proof {
            Some(p) => format!(r#", "proof": {}"#, p.to_json()),
            None => String::new(),
        };

        format!(
            r#"{{"@context": [{ctx}], "type": [{types}], "holder": "{holder}", "verifiableCredential": [{creds}]{proof}}}"#,
            ctx = context_json,
            types = types_json,
            holder = escape_json_string(&self.holder),
            creds = credentials_json,
            proof = proof_json,
        )
    }

    /// Statistics: number of credentials and which VC types are present.
    pub fn statistics(&self) -> VpStatistics {
        let credential_count = self.verifiable_credential.len();
        let mut type_set: HashSet<String> = HashSet::new();
        for vc in &self.verifiable_credential {
            for t in &vc.types {
                if t != "VerifiableCredential" {
                    type_set.insert(t.clone());
                }
            }
        }
        let mut types_present: Vec<String> = type_set.into_iter().collect();
        types_present.sort();
        VpStatistics {
            credential_count,
            types_present,
            has_proof: self.proof.is_some(),
        }
    }
}

/// Statistics about a Verifiable Presentation
#[derive(Debug, Clone, PartialEq)]
pub struct VpStatistics {
    /// Number of VCs included
    pub credential_count: usize,
    /// Distinct non-base VC types present
    pub types_present: Vec<String>,
    /// Whether the VP has an attached proof
    pub has_proof: bool,
}

// ─── Request / filter ────────────────────────────────────────────────────────

/// Criteria for building a VP (which credentials to include)
#[derive(Debug, Clone, Default)]
pub struct VpRequest {
    /// The holder's DID
    pub holder: String,
    /// Filter by VC type (any match is included)
    pub required_types: Vec<String>,
    /// Filter by issuer DID (exact match)
    pub required_issuer: Option<String>,
    /// Filter by subject DID (exact match)
    pub required_subject: Option<String>,
    /// Only include non-expired credentials (requires a current date)
    pub exclude_expired: bool,
    /// Current date string for expiry check (e.g. `"2025-03-01"`)
    pub current_date: Option<String>,
    /// Keys to selectively disclose from each credential's claims
    pub disclose_only: Option<Vec<String>>,
    /// Attach a stub proof to the VP
    pub include_proof: bool,
}

impl VpRequest {
    /// Create a minimal request for the given holder
    pub fn new(holder: impl Into<String>) -> Self {
        Self {
            holder: holder.into(),
            ..Default::default()
        }
    }

    /// Require a specific credential type
    pub fn with_type(mut self, vc_type: impl Into<String>) -> Self {
        self.required_types.push(vc_type.into());
        self
    }

    /// Require a specific issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.required_issuer = Some(issuer.into());
        self
    }

    /// Require a specific credential subject
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.required_subject = Some(subject.into());
        self
    }

    /// Exclude expired credentials from the presentation
    pub fn exclude_expired(mut self, current_date: impl Into<String>) -> Self {
        self.exclude_expired = true;
        self.current_date = Some(current_date.into());
        self
    }

    /// Selectively disclose only the given claim keys
    pub fn disclose_keys(mut self, keys: Vec<String>) -> Self {
        self.disclose_only = Some(keys);
        self
    }

    /// Attach a stub proof
    pub fn with_proof(mut self) -> Self {
        self.include_proof = true;
        self
    }
}

// ─── VcPresenter ──────────────────────────────────────────────────────────────

/// Builder / manager for Verifiable Presentations
///
/// Holds a pool of Verifiable Credentials and can construct VPs on demand
/// based on flexible selection criteria.
pub struct VcPresenter {
    /// The holder DID of this presenter
    holder_did: String,
    /// Pool of credentials available for presentation
    credentials: Vec<VerifiableCredential>,
}

impl VcPresenter {
    /// Create a new presenter for the given holder DID
    pub fn new(holder_did: impl Into<String>) -> Self {
        Self {
            holder_did: holder_did.into(),
            credentials: Vec::new(),
        }
    }

    /// Add a credential to the pool
    pub fn add_credential(&mut self, vc: VerifiableCredential) {
        self.credentials.push(vc);
    }

    /// Add multiple credentials to the pool
    pub fn add_credentials(&mut self, vcs: impl IntoIterator<Item = VerifiableCredential>) {
        for vc in vcs {
            self.credentials.push(vc);
        }
    }

    /// Return the number of credentials in the pool
    pub fn credential_count(&self) -> usize {
        self.credentials.len()
    }

    /// Select credentials matching the given filter criteria.
    ///
    /// - `types`: if non-empty, at least one type must match
    /// - `issuer`: if Some, issuer must match exactly
    /// - `subject`: if Some, subject.id must match exactly
    pub fn select_credentials(
        &self,
        types: &[String],
        issuer: Option<&str>,
        subject: Option<&str>,
    ) -> Vec<&VerifiableCredential> {
        self.credentials
            .iter()
            .filter(|vc| {
                // Type filter
                if !types.is_empty() && !types.iter().any(|t| vc.types.contains(t)) {
                    return false;
                }
                // Issuer filter
                if let Some(iss) = issuer {
                    if vc.issuer != iss {
                        return false;
                    }
                }
                // Subject filter
                if let Some(sub) = subject {
                    if vc.subject.id != sub {
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Verify that the holder DID matches the subject of each credential.
    /// Returns `Err` with the first mismatch found.
    pub fn verify_holder_binding(&self, vc: &VerifiableCredential) -> Result<(), PresenterError> {
        if vc.subject.id != self.holder_did {
            return Err(PresenterError::HolderMismatch {
                holder: self.holder_did.clone(),
                subject: vc.subject.id.clone(),
            });
        }
        Ok(())
    }

    /// Build a Verifiable Presentation according to the given request.
    pub fn build_presentation(
        &self,
        request: &VpRequest,
    ) -> Result<VerifiablePresentation, PresenterError> {
        // Select matching credentials
        let selected = self.select_credentials(
            &request.required_types,
            request.required_issuer.as_deref(),
            request.required_subject.as_deref(),
        );

        if selected.is_empty() {
            return Err(PresenterError::NoCredentialsSelected);
        }

        // Apply expiry filter and selective disclosure
        let mut included: Vec<VerifiableCredential> = Vec::new();

        for vc in selected {
            // Expiry check
            if request.exclude_expired {
                if let Some(date) = &request.current_date {
                    if vc.is_expired(date) {
                        continue; // skip expired
                    }
                }
            }

            // Selective disclosure
            let processed_vc = match &request.disclose_only {
                Some(keys) => {
                    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
                    vc.selective_disclose(&key_refs)
                }
                None => vc.clone(),
            };

            included.push(processed_vc);
        }

        if included.is_empty() {
            return Err(PresenterError::NoCredentialsSelected);
        }

        // Build proof
        let proof = if request.include_proof {
            Some(PresentationProof::stub(&self.holder_did))
        } else {
            None
        };

        Ok(VerifiablePresentation {
            context: vec![
                "https://www.w3.org/2018/credentials/v1".to_string(),
                "https://w3id.org/security/suites/ed25519-2020/v1".to_string(),
            ],
            types: vec!["VerifiablePresentation".to_string()],
            holder: self.holder_did.clone(),
            verifiable_credential: included,
            proof,
        })
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Escape a string for safe embedding inside a JSON string value.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn alice_subject() -> CredentialSubject {
        let mut claims = HashMap::new();
        claims.insert("name".to_string(), "Alice".to_string());
        claims.insert("degree".to_string(), "Bachelor of Science".to_string());
        CredentialSubject {
            id: "did:example:alice".to_string(),
            claims,
        }
    }

    fn alumni_vc() -> VerifiableCredential {
        VerifiableCredential {
            id: "urn:vc:alumni:001".to_string(),
            types: vec![
                "VerifiableCredential".to_string(),
                "AlumniCredential".to_string(),
            ],
            issuer: "did:example:university".to_string(),
            issuance_date: "2024-01-01".to_string(),
            expiration_date: None,
            subject: alice_subject(),
            proof: None,
        }
    }

    fn id_vc() -> VerifiableCredential {
        let mut claims = HashMap::new();
        claims.insert("dob".to_string(), "2000-01-01".to_string());
        VerifiableCredential {
            id: "urn:vc:id:002".to_string(),
            types: vec![
                "VerifiableCredential".to_string(),
                "IdentityCredential".to_string(),
            ],
            issuer: "did:example:gov".to_string(),
            issuance_date: "2023-06-01".to_string(),
            expiration_date: Some("2025-06-01".to_string()),
            subject: CredentialSubject {
                id: "did:example:alice".to_string(),
                claims,
            },
            proof: None,
        }
    }

    // ── VcPresenter construction ──────────────────────────────────────────────

    #[test]
    fn test_new_presenter_starts_empty() {
        let p = VcPresenter::new("did:example:alice");
        assert_eq!(p.credential_count(), 0);
    }

    #[test]
    fn test_add_credential_increments_count() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        assert_eq!(p.credential_count(), 1);
    }

    #[test]
    fn test_add_credentials_batch() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credentials(vec![alumni_vc(), id_vc()]);
        assert_eq!(p.credential_count(), 2);
    }

    // ── build_presentation ────────────────────────────────────────────────────

    #[test]
    fn test_build_basic_presentation() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.is_valid_structure());
        assert_eq!(vp.holder, "did:example:alice");
        assert_eq!(vp.verifiable_credential.len(), 1);
    }

    #[test]
    fn test_build_fails_with_no_credentials() {
        let p = VcPresenter::new("did:example:alice");
        let req = VpRequest::new("did:example:alice");
        let err = p.build_presentation(&req).unwrap_err();
        assert_eq!(err, PresenterError::NoCredentialsSelected);
    }

    #[test]
    fn test_build_with_type_filter() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let req = VpRequest::new("did:example:alice").with_type("AlumniCredential");
        let vp = p.build_presentation(&req).expect("build ok");
        assert_eq!(vp.verifiable_credential.len(), 1);
        assert!(vp.verifiable_credential[0]
            .types
            .contains(&"AlumniCredential".to_string()));
    }

    #[test]
    fn test_build_with_issuer_filter() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let req = VpRequest::new("did:example:alice").with_issuer("did:example:gov");
        let vp = p.build_presentation(&req).expect("build ok");
        assert_eq!(vp.verifiable_credential.len(), 1);
        assert_eq!(vp.verifiable_credential[0].issuer, "did:example:gov");
    }

    #[test]
    fn test_type_filter_no_match_returns_error() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice").with_type("NonExistentType");
        let err = p.build_presentation(&req).unwrap_err();
        assert_eq!(err, PresenterError::NoCredentialsSelected);
    }

    // ── Expiry handling ───────────────────────────────────────────────────────

    #[test]
    fn test_credential_not_expired_when_no_exp() {
        assert!(!alumni_vc().is_expired("2099-01-01"));
    }

    #[test]
    fn test_credential_expired() {
        // id_vc expires 2025-06-01; current date is after that
        assert!(id_vc().is_expired("2026-01-01"));
    }

    #[test]
    fn test_credential_not_expired() {
        // id_vc expires 2025-06-01; current date before expiry
        assert!(!id_vc().is_expired("2024-01-01"));
    }

    #[test]
    fn test_build_excludes_expired_credentials() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc()); // no expiry
        p.add_credential(id_vc()); // expires 2025-06-01
        let req = VpRequest::new("did:example:alice").exclude_expired("2026-01-01");
        let vp = p.build_presentation(&req).expect("build ok");
        // Only alumni_vc should be included
        assert_eq!(vp.verifiable_credential.len(), 1);
        assert_eq!(vp.verifiable_credential[0].id, "urn:vc:alumni:001");
    }

    #[test]
    fn test_build_all_expired_returns_error() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(id_vc()); // expires 2025-06-01
        let req = VpRequest::new("did:example:alice").exclude_expired("2030-01-01");
        let err = p.build_presentation(&req).unwrap_err();
        assert_eq!(err, PresenterError::NoCredentialsSelected);
    }

    // ── Selective disclosure ──────────────────────────────────────────────────

    #[test]
    fn test_selective_disclose_keeps_specified_keys() {
        let vc = alumni_vc();
        let disclosed = vc.selective_disclose(&["name"]);
        assert!(disclosed.subject.claims.contains_key("name"));
        assert!(!disclosed.subject.claims.contains_key("degree"));
    }

    #[test]
    fn test_selective_disclose_keeps_structural_fields() {
        let vc = alumni_vc();
        let disclosed = vc.selective_disclose(&["name"]);
        assert_eq!(disclosed.id, vc.id);
        assert_eq!(disclosed.issuer, vc.issuer);
        assert_eq!(disclosed.types, vc.types);
    }

    #[test]
    fn test_selective_disclose_empty_keys() {
        let vc = alumni_vc();
        let disclosed = vc.selective_disclose(&[]);
        assert!(disclosed.subject.claims.is_empty());
    }

    #[test]
    fn test_build_with_selective_disclosure() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice").disclose_keys(vec!["name".to_string()]);
        let vp = p.build_presentation(&req).expect("build ok");
        let vc = &vp.verifiable_credential[0];
        assert!(vc.subject.claims.contains_key("name"));
        assert!(!vc.subject.claims.contains_key("degree"));
    }

    // ── Presentation proof ────────────────────────────────────────────────────

    #[test]
    fn test_stub_proof_fields() {
        let proof = PresentationProof::stub("did:example:alice");
        assert_eq!(proof.proof_type, "Ed25519Signature2020");
        assert_eq!(proof.verification_method, "did:example:alice#key-1");
        assert!(!proof.proof_value.is_empty());
        assert_eq!(proof.proof_purpose, "authentication");
    }

    #[test]
    fn test_build_presentation_with_proof() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice").with_proof();
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.proof.is_some());
    }

    #[test]
    fn test_build_presentation_without_proof() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.proof.is_none());
    }

    // ── VP validation ─────────────────────────────────────────────────────────

    #[test]
    fn test_vp_is_valid_structure() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.is_valid_structure());
    }

    #[test]
    fn test_vp_validate_no_errors_for_valid() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let errors = vp.validate(None);
        assert!(errors.is_empty(), "errors = {:?}", errors);
    }

    #[test]
    fn test_vp_validate_detects_expired() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(id_vc()); // expires 2025-06-01
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let errors = vp.validate(Some("2030-01-01"));
        assert!(errors.iter().any(|e| e.contains("expired")));
    }

    // ── VP JSON serialisation ─────────────────────────────────────────────────

    #[test]
    fn test_vp_to_json_contains_holder() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let json = vp.to_json();
        assert!(json.contains("did:example:alice"), "json = {json}");
    }

    #[test]
    fn test_vp_to_json_contains_type() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let json = vp.to_json();
        assert!(json.contains("VerifiablePresentation"), "json = {json}");
    }

    #[test]
    fn test_vp_to_json_contains_credential() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let json = vp.to_json();
        assert!(json.contains("AlumniCredential"), "json = {json}");
    }

    #[test]
    fn test_vc_to_json_format() {
        let vc = alumni_vc();
        let json = vc.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("VerifiableCredential"));
        assert!(json.contains("did:example:university"));
    }

    #[test]
    fn test_proof_to_json() {
        let proof = PresentationProof::stub("did:example:alice");
        let json = proof.to_json();
        assert!(json.contains("Ed25519Signature2020"));
        assert!(json.contains("authentication"));
    }

    // ── Holder binding ────────────────────────────────────────────────────────

    #[test]
    fn test_holder_binding_match() {
        let p = VcPresenter::new("did:example:alice");
        let vc = alumni_vc(); // subject.id = did:example:alice
        assert!(p.verify_holder_binding(&vc).is_ok());
    }

    #[test]
    fn test_holder_binding_mismatch() {
        let p = VcPresenter::new("did:example:bob");
        let vc = alumni_vc(); // subject.id = did:example:alice
        let err = p.verify_holder_binding(&vc).unwrap_err();
        assert!(matches!(err, PresenterError::HolderMismatch { .. }));
    }

    // ── VP statistics ─────────────────────────────────────────────────────────

    #[test]
    fn test_statistics_credential_count() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let stats = vp.statistics();
        assert_eq!(stats.credential_count, 2);
    }

    #[test]
    fn test_statistics_types_present() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        let stats = vp.statistics();
        assert!(stats
            .types_present
            .contains(&"AlumniCredential".to_string()));
    }

    #[test]
    fn test_statistics_has_proof_false() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(!vp.statistics().has_proof);
    }

    #[test]
    fn test_statistics_has_proof_true() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice").with_proof();
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.statistics().has_proof);
    }

    // ── Credential selection ──────────────────────────────────────────────────

    #[test]
    fn test_select_all_with_empty_criteria() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let selected = p.select_credentials(&[], None, None);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_by_type() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let selected = p.select_credentials(&["IdentityCredential".to_string()], None, None);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "urn:vc:id:002");
    }

    #[test]
    fn test_select_by_issuer_and_type() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        p.add_credential(id_vc());
        let selected = p.select_credentials(
            &["VerifiableCredential".to_string()],
            Some("did:example:university"),
            None,
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].issuer, "did:example:university");
    }

    // ── PresenterError display ────────────────────────────────────────────────

    #[test]
    fn test_error_display_no_credentials() {
        let err = PresenterError::NoCredentialsSelected;
        assert!(err.to_string().contains("No credentials"));
    }

    #[test]
    fn test_error_display_holder_mismatch() {
        let err = PresenterError::HolderMismatch {
            holder: "did:a".to_string(),
            subject: "did:b".to_string(),
        };
        assert!(err.to_string().contains("did:a"));
        assert!(err.to_string().contains("did:b"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_vc_has_valid_structure_true() {
        assert!(alumni_vc().has_valid_structure());
    }

    #[test]
    fn test_vc_has_valid_structure_missing_type() {
        let mut vc = alumni_vc();
        vc.types = vec!["AlumniCredential".to_string()]; // missing base type
        assert!(!vc.has_valid_structure());
    }

    #[test]
    fn test_vc_has_valid_structure_empty_issuer() {
        let mut vc = alumni_vc();
        vc.issuer = String::new();
        assert!(!vc.has_valid_structure());
    }

    #[test]
    fn test_escape_json_string_quotes() {
        let result = escape_json_string(r#"he said "hello""#);
        assert!(result.contains("\\\""));
    }

    #[test]
    fn test_escape_json_string_backslash() {
        let result = escape_json_string("a\\b");
        assert!(result.contains("\\\\"));
    }

    #[test]
    fn test_vp_request_builder_chain() {
        let req = VpRequest::new("did:example:alice")
            .with_type("AlumniCredential")
            .with_issuer("did:example:university")
            .with_proof();
        assert_eq!(req.required_types, vec!["AlumniCredential"]);
        assert_eq!(
            req.required_issuer,
            Some("did:example:university".to_string())
        );
        assert!(req.include_proof);
    }

    #[test]
    fn test_vp_to_json_contains_proof_when_attached() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice").with_proof();
        let vp = p.build_presentation(&req).expect("build ok");
        let json = vp.to_json();
        assert!(json.contains("Ed25519Signature2020"), "json = {json}");
    }

    #[test]
    fn test_vp_context_includes_w3c() {
        let mut p = VcPresenter::new("did:example:alice");
        p.add_credential(alumni_vc());
        let req = VpRequest::new("did:example:alice");
        let vp = p.build_presentation(&req).expect("build ok");
        assert!(vp.context.iter().any(|c| c.contains("w3.org")));
    }
}
