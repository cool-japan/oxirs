//! # Presentation Request
//!
//! Verifiable Presentation request and response handling.
//!
//! This module implements the W3C Verifiable Presentation request/response
//! lifecycle: a verifier issues a [`PresentationRequest`] containing a
//! challenge, domain, required credential types and (optionally) a list of
//! trusted issuers.  A holder constructs a [`VerifiablePresentation`] and
//! submits it back; [`PresentationValidator`] checks all constraints and
//! returns a [`ValidationOutcome`].
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::presentation_request::{
//!     PresentationRequest, VerifiablePresentation, PresentedCredential,
//!     PresentationValidator, ValidationOutcome,
//! };
//! use std::collections::HashMap;
//!
//! let request = PresentationRequest::new("req-1", "nonce-abc", "example.com")
//!     .require_type("UniversityDegreeCredential")
//!     .trust_issuer("did:example:issuer");
//!
//! let credential = PresentedCredential {
//!     credential_type: "UniversityDegreeCredential".to_string(),
//!     issuer: "did:example:issuer".to_string(),
//!     subject: "did:example:alice".to_string(),
//!     claims: HashMap::new(),
//! };
//!
//! let presentation = VerifiablePresentation {
//!     holder: "did:example:alice".to_string(),
//!     credentials: vec![credential],
//!     challenge: "nonce-abc".to_string(),
//!     domain: "example.com".to_string(),
//! };
//!
//! let validator = PresentationValidator::new();
//! let outcome = validator.validate(&presentation, &request, 0);
//! assert_eq!(outcome, ValidationOutcome::Valid);
//! ```

use std::collections::HashMap;

// ─── Presentation Request ─────────────────────────────────────────────────────

/// A request for a verifiable presentation issued by a verifier to a holder.
#[derive(Debug, Clone)]
pub struct PresentationRequest {
    /// Unique identifier for this request.
    pub request_id: String,
    /// Cryptographic challenge (nonce) that the holder must echo back.
    pub challenge: String,
    /// Domain the holder is expected to present to.
    pub domain: String,
    /// Required credential types that must be present in the presentation.
    pub required_types: Vec<String>,
    /// Optional allowlist of issuer DIDs; empty means any issuer is accepted.
    pub trusted_issuers: Vec<String>,
    /// Optional expiry as Unix timestamp in milliseconds.
    pub expires_at_ms: Option<u64>,
}

impl PresentationRequest {
    /// Creates a new request with the given `request_id`, `challenge` and
    /// `domain`.  No required types or trusted issuers are set initially.
    pub fn new(request_id: &str, challenge: &str, domain: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            challenge: challenge.to_string(),
            domain: domain.to_string(),
            required_types: Vec::new(),
            trusted_issuers: Vec::new(),
            expires_at_ms: None,
        }
    }

    /// Adds a required credential type to this request (builder style).
    pub fn require_type(mut self, credential_type: &str) -> Self {
        self.required_types.push(credential_type.to_string());
        self
    }

    /// Adds a trusted issuer DID to this request (builder style).
    pub fn trust_issuer(mut self, issuer_did: &str) -> Self {
        self.trusted_issuers.push(issuer_did.to_string());
        self
    }

    /// Sets the expiry timestamp in milliseconds (builder style).
    pub fn with_expiry(mut self, expires_at_ms: u64) -> Self {
        self.expires_at_ms = Some(expires_at_ms);
        self
    }

    /// Returns `true` when the request has expired relative to `now_ms`.
    ///
    /// A request without an expiry never expires.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.is_some_and(|exp| now_ms > exp)
    }
}

// ─── Presented Credential ─────────────────────────────────────────────────────

/// A credential included in a [`VerifiablePresentation`].
#[derive(Debug, Clone)]
pub struct PresentedCredential {
    /// The credential type (e.g. `"UniversityDegreeCredential"`).
    pub credential_type: String,
    /// DID of the credential issuer.
    pub issuer: String,
    /// DID of the credential subject.
    pub subject: String,
    /// Arbitrary string-typed claims carried by this credential.
    pub claims: HashMap<String, String>,
}

// ─── Verifiable Presentation ──────────────────────────────────────────────────

/// A verifiable presentation submitted by a holder in response to a
/// [`PresentationRequest`].
#[derive(Debug, Clone)]
pub struct VerifiablePresentation {
    /// DID of the holder constructing this presentation.
    pub holder: String,
    /// Credentials included in the presentation.
    pub credentials: Vec<PresentedCredential>,
    /// The challenge echoed from the originating request.
    pub challenge: String,
    /// The domain echoed from the originating request.
    pub domain: String,
}

// ─── Validation Outcome ───────────────────────────────────────────────────────

/// The result of validating a [`VerifiablePresentation`] against a
/// [`PresentationRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationOutcome {
    /// All constraints are satisfied.
    Valid,
    /// The request had expired before the presentation was validated.
    Expired,
    /// The presentation's challenge does not match the request's challenge.
    ChallengeMismatch,
    /// The presentation's domain does not match the request's domain.
    DomainMismatch,
    /// A required credential type is absent from the presentation.
    MissingCredentialType(String),
    /// A credential was issued by an issuer not in the trusted list.
    UntrustedIssuer(String),
}

// ─── Validator ────────────────────────────────────────────────────────────────

/// Validates a [`VerifiablePresentation`] against a [`PresentationRequest`].
pub struct PresentationValidator;

impl PresentationValidator {
    /// Creates a new [`PresentationValidator`].
    pub fn new() -> Self {
        Self
    }

    /// Validates `presentation` against `request` at wall-clock time `now_ms`.
    ///
    /// Checks are performed in order:
    /// 1. Request not expired
    /// 2. Challenge matches
    /// 3. Domain matches
    /// 4. All required credential types present
    /// 5. All credential issuers trusted (only when the trusted-issuers list is
    ///    non-empty)
    pub fn validate(
        &self,
        presentation: &VerifiablePresentation,
        request: &PresentationRequest,
        now_ms: u64,
    ) -> ValidationOutcome {
        if request.is_expired(now_ms) {
            return ValidationOutcome::Expired;
        }

        if !self.challenge_matches(presentation, request) {
            return ValidationOutcome::ChallengeMismatch;
        }

        if !self.domain_matches(presentation, request) {
            return ValidationOutcome::DomainMismatch;
        }

        for required in &request.required_types {
            let found = presentation
                .credentials
                .iter()
                .any(|c| &c.credential_type == required);
            if !found {
                return ValidationOutcome::MissingCredentialType(required.clone());
            }
        }

        if !self.all_issuers_trusted(presentation, request) {
            // Identify the first untrusted issuer for the error payload.
            let untrusted = presentation
                .credentials
                .iter()
                .find(|c| !request.trusted_issuers.contains(&c.issuer))
                .map(|c| c.issuer.clone())
                .unwrap_or_default();
            return ValidationOutcome::UntrustedIssuer(untrusted);
        }

        ValidationOutcome::Valid
    }

    /// Returns `true` when the presentation's challenge equals the request's
    /// challenge (case-sensitive).
    pub fn challenge_matches(
        &self,
        presentation: &VerifiablePresentation,
        request: &PresentationRequest,
    ) -> bool {
        presentation.challenge == request.challenge
    }

    /// Returns `true` when the presentation's domain equals the request's
    /// domain (case-sensitive).
    pub fn domain_matches(
        &self,
        presentation: &VerifiablePresentation,
        request: &PresentationRequest,
    ) -> bool {
        presentation.domain == request.domain
    }

    /// Returns `true` when every required type in `request` is satisfied by at
    /// least one credential in `presentation`.
    pub fn all_types_satisfied(
        &self,
        presentation: &VerifiablePresentation,
        request: &PresentationRequest,
    ) -> bool {
        request.required_types.iter().all(|required| {
            presentation
                .credentials
                .iter()
                .any(|c| &c.credential_type == required)
        })
    }

    /// Returns `true` when every credential in `presentation` was issued by a
    /// DID in `request.trusted_issuers`, or when `trusted_issuers` is empty
    /// (meaning all issuers are accepted).
    pub fn all_issuers_trusted(
        &self,
        presentation: &VerifiablePresentation,
        request: &PresentationRequest,
    ) -> bool {
        if request.trusted_issuers.is_empty() {
            return true;
        }
        presentation
            .credentials
            .iter()
            .all(|c| request.trusted_issuers.contains(&c.issuer))
    }
}

impl Default for PresentationValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_request() -> PresentationRequest {
        PresentationRequest::new("req-1", "nonce-abc", "example.com")
    }

    fn make_credential(cred_type: &str, issuer: &str) -> PresentedCredential {
        PresentedCredential {
            credential_type: cred_type.to_string(),
            issuer: issuer.to_string(),
            subject: "did:example:alice".to_string(),
            claims: HashMap::new(),
        }
    }

    fn make_presentation(
        challenge: &str,
        domain: &str,
        creds: Vec<PresentedCredential>,
    ) -> VerifiablePresentation {
        VerifiablePresentation {
            holder: "did:example:alice".to_string(),
            credentials: creds,
            challenge: challenge.to_string(),
            domain: domain.to_string(),
        }
    }

    fn valid_presentation() -> VerifiablePresentation {
        make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential(
                "UniversityDegreeCredential",
                "did:example:issuer",
            )],
        )
    }

    // ── PresentationRequest builder ───────────────────────────────────────────

    #[test]
    fn test_new_sets_fields() {
        let req = PresentationRequest::new("r1", "ch1", "dom1");
        assert_eq!(req.request_id, "r1");
        assert_eq!(req.challenge, "ch1");
        assert_eq!(req.domain, "dom1");
        assert!(req.required_types.is_empty());
        assert!(req.trusted_issuers.is_empty());
        assert!(req.expires_at_ms.is_none());
    }

    #[test]
    fn test_require_type_appends() {
        let req = make_request().require_type("TypeA").require_type("TypeB");
        assert_eq!(req.required_types, vec!["TypeA", "TypeB"]);
    }

    #[test]
    fn test_require_type_single() {
        let req = make_request().require_type("UniversityDegreeCredential");
        assert_eq!(req.required_types.len(), 1);
        assert_eq!(req.required_types[0], "UniversityDegreeCredential");
    }

    #[test]
    fn test_trust_issuer_appends() {
        let req = make_request()
            .trust_issuer("did:example:issuer1")
            .trust_issuer("did:example:issuer2");
        assert_eq!(req.trusted_issuers.len(), 2);
        assert!(req
            .trusted_issuers
            .contains(&"did:example:issuer1".to_string()));
        assert!(req
            .trusted_issuers
            .contains(&"did:example:issuer2".to_string()));
    }

    #[test]
    fn test_trust_issuer_single() {
        let req = make_request().trust_issuer("did:example:only");
        assert_eq!(req.trusted_issuers, vec!["did:example:only"]);
    }

    #[test]
    fn test_with_expiry_sets_value() {
        let req = make_request().with_expiry(9_999_999);
        assert_eq!(req.expires_at_ms, Some(9_999_999));
    }

    #[test]
    fn test_with_expiry_zero() {
        let req = make_request().with_expiry(0);
        assert_eq!(req.expires_at_ms, Some(0));
    }

    // ── is_expired ────────────────────────────────────────────────────────────

    #[test]
    fn test_is_expired_no_expiry_never_expires() {
        let req = make_request();
        assert!(!req.is_expired(u64::MAX));
    }

    #[test]
    fn test_is_expired_before_expiry() {
        let req = make_request().with_expiry(1_000_000);
        assert!(!req.is_expired(999_999));
    }

    #[test]
    fn test_is_expired_at_expiry_boundary() {
        let req = make_request().with_expiry(1_000_000);
        // exactly at expiry → not expired
        assert!(!req.is_expired(1_000_000));
    }

    #[test]
    fn test_is_expired_after_expiry() {
        let req = make_request().with_expiry(1_000_000);
        assert!(req.is_expired(1_000_001));
    }

    #[test]
    fn test_is_expired_far_in_future() {
        let req = make_request().with_expiry(u64::MAX - 1);
        assert!(!req.is_expired(0));
    }

    // ── validate: Valid ───────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_no_constraints() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 0), ValidationOutcome::Valid);
    }

    #[test]
    fn test_validate_valid_with_required_type() {
        let req = make_request().require_type("UniversityDegreeCredential");
        let pres = valid_presentation();
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 0), ValidationOutcome::Valid);
    }

    #[test]
    fn test_validate_valid_with_trusted_issuer() {
        let req = make_request()
            .require_type("UniversityDegreeCredential")
            .trust_issuer("did:example:issuer");
        let pres = valid_presentation();
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 0), ValidationOutcome::Valid);
    }

    #[test]
    fn test_validate_valid_multiple_credentials() {
        let req = make_request()
            .require_type("TypeA")
            .require_type("TypeB")
            .trust_issuer("did:example:issuer");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![
                make_credential("TypeA", "did:example:issuer"),
                make_credential("TypeB", "did:example:issuer"),
            ],
        );
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 0), ValidationOutcome::Valid);
    }

    // ── validate: Expired ─────────────────────────────────────────────────────

    #[test]
    fn test_validate_expired_request() {
        let req = make_request().with_expiry(500);
        let pres = valid_presentation();
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 1_000), ValidationOutcome::Expired);
    }

    #[test]
    fn test_validate_expired_takes_priority_over_challenge_mismatch() {
        let req = make_request().with_expiry(500);
        let pres = make_presentation("wrong-nonce", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 1_000), ValidationOutcome::Expired);
    }

    // ── validate: ChallengeMismatch ───────────────────────────────────────────

    #[test]
    fn test_validate_challenge_mismatch() {
        let req = make_request();
        let pres = make_presentation("wrong-nonce", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::ChallengeMismatch
        );
    }

    #[test]
    fn test_validate_challenge_case_sensitive() {
        let req = make_request(); // challenge = "nonce-abc"
        let pres = make_presentation("NONCE-ABC", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::ChallengeMismatch
        );
    }

    // ── validate: DomainMismatch ──────────────────────────────────────────────

    #[test]
    fn test_validate_domain_mismatch() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "other.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::DomainMismatch
        );
    }

    #[test]
    fn test_validate_domain_case_sensitive() {
        let req = make_request(); // domain = "example.com"
        let pres = make_presentation("nonce-abc", "EXAMPLE.COM", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::DomainMismatch
        );
    }

    // ── validate: MissingCredentialType ───────────────────────────────────────

    #[test]
    fn test_validate_missing_required_type() {
        let req = make_request().require_type("DriverLicense");
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::MissingCredentialType("DriverLicense".to_string())
        );
    }

    #[test]
    fn test_validate_missing_one_of_two_required_types() {
        let req = make_request().require_type("TypeA").require_type("TypeB");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential("TypeA", "did:example:issuer")],
        );
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::MissingCredentialType("TypeB".to_string())
        );
    }

    // ── validate: UntrustedIssuer ─────────────────────────────────────────────

    #[test]
    fn test_validate_untrusted_issuer() {
        let req = make_request()
            .require_type("UniversityDegreeCredential")
            .trust_issuer("did:example:trusted");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential(
                "UniversityDegreeCredential",
                "did:example:untrusted",
            )],
        );
        let v = PresentationValidator::new();
        assert_eq!(
            v.validate(&pres, &req, 0),
            ValidationOutcome::UntrustedIssuer("did:example:untrusted".to_string())
        );
    }

    #[test]
    fn test_validate_no_trusted_issuers_list_accepts_any() {
        // trusted_issuers is empty → any issuer is accepted
        let req = make_request().require_type("TypeA");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential("TypeA", "did:example:anyone")],
        );
        let v = PresentationValidator::new();
        assert_eq!(v.validate(&pres, &req, 0), ValidationOutcome::Valid);
    }

    // ── challenge_matches ─────────────────────────────────────────────────────

    #[test]
    fn test_challenge_matches_true() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert!(v.challenge_matches(&pres, &req));
    }

    #[test]
    fn test_challenge_matches_false() {
        let req = make_request();
        let pres = make_presentation("other-nonce", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert!(!v.challenge_matches(&pres, &req));
    }

    // ── domain_matches ────────────────────────────────────────────────────────

    #[test]
    fn test_domain_matches_true() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert!(v.domain_matches(&pres, &req));
    }

    #[test]
    fn test_domain_matches_false() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "evil.com", vec![]);
        let v = PresentationValidator::new();
        assert!(!v.domain_matches(&pres, &req));
    }

    // ── all_types_satisfied ───────────────────────────────────────────────────

    #[test]
    fn test_all_types_satisfied_empty_required() {
        let req = make_request();
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        assert!(v.all_types_satisfied(&pres, &req));
    }

    #[test]
    fn test_all_types_satisfied_true() {
        let req = make_request().require_type("TypeX");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential("TypeX", "did:example:issuer")],
        );
        let v = PresentationValidator::new();
        assert!(v.all_types_satisfied(&pres, &req));
    }

    #[test]
    fn test_all_types_satisfied_false() {
        let req = make_request().require_type("TypeX").require_type("TypeY");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential("TypeX", "did:example:issuer")],
        );
        let v = PresentationValidator::new();
        assert!(!v.all_types_satisfied(&pres, &req));
    }

    // ── all_issuers_trusted ───────────────────────────────────────────────────

    #[test]
    fn test_all_issuers_trusted_empty_allowlist() {
        let req = make_request(); // trusted_issuers empty
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![make_credential("TypeA", "did:example:anyone")],
        );
        let v = PresentationValidator::new();
        assert!(v.all_issuers_trusted(&pres, &req));
    }

    #[test]
    fn test_all_issuers_trusted_all_in_list() {
        let req = make_request()
            .trust_issuer("did:example:issuerA")
            .trust_issuer("did:example:issuerB");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![
                make_credential("TypeA", "did:example:issuerA"),
                make_credential("TypeB", "did:example:issuerB"),
            ],
        );
        let v = PresentationValidator::new();
        assert!(v.all_issuers_trusted(&pres, &req));
    }

    #[test]
    fn test_all_issuers_trusted_one_untrusted() {
        let req = make_request().trust_issuer("did:example:trusted");
        let pres = make_presentation(
            "nonce-abc",
            "example.com",
            vec![
                make_credential("TypeA", "did:example:trusted"),
                make_credential("TypeB", "did:example:rogue"),
            ],
        );
        let v = PresentationValidator::new();
        assert!(!v.all_issuers_trusted(&pres, &req));
    }

    #[test]
    fn test_all_issuers_trusted_empty_credentials() {
        let req = make_request().trust_issuer("did:example:issuer");
        let pres = make_presentation("nonce-abc", "example.com", vec![]);
        let v = PresentationValidator::new();
        // No credentials → trivially all trusted
        assert!(v.all_issuers_trusted(&pres, &req));
    }

    // ── Default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_validator_default() {
        let _v: PresentationValidator = PresentationValidator;
    }

    // ── PresentedCredential fields ────────────────────────────────────────────

    #[test]
    fn test_presented_credential_fields() {
        let mut claims = HashMap::new();
        claims.insert("degree".to_string(), "BSc".to_string());
        let cred = PresentedCredential {
            credential_type: "DegreeCredential".to_string(),
            issuer: "did:example:uni".to_string(),
            subject: "did:example:student".to_string(),
            claims: claims.clone(),
        };
        assert_eq!(cred.credential_type, "DegreeCredential");
        assert_eq!(cred.issuer, "did:example:uni");
        assert_eq!(cred.subject, "did:example:student");
        assert_eq!(cred.claims.get("degree").map(String::as_str), Some("BSc"));
    }

    // ── ValidationOutcome PartialEq ───────────────────────────────────────────

    #[test]
    fn test_validation_outcome_equality() {
        assert_eq!(ValidationOutcome::Valid, ValidationOutcome::Valid);
        assert_eq!(ValidationOutcome::Expired, ValidationOutcome::Expired);
        assert_eq!(
            ValidationOutcome::ChallengeMismatch,
            ValidationOutcome::ChallengeMismatch
        );
        assert_eq!(
            ValidationOutcome::DomainMismatch,
            ValidationOutcome::DomainMismatch
        );
        assert_ne!(ValidationOutcome::Valid, ValidationOutcome::Expired);
    }

    #[test]
    fn test_validation_outcome_missing_type_inner_value() {
        let o = ValidationOutcome::MissingCredentialType("Foo".to_string());
        assert_eq!(
            o,
            ValidationOutcome::MissingCredentialType("Foo".to_string())
        );
        assert_ne!(
            o,
            ValidationOutcome::MissingCredentialType("Bar".to_string())
        );
    }

    #[test]
    fn test_validation_outcome_untrusted_issuer_inner_value() {
        let o = ValidationOutcome::UntrustedIssuer("did:x:y".to_string());
        assert_eq!(o, ValidationOutcome::UntrustedIssuer("did:x:y".to_string()));
        assert_ne!(o, ValidationOutcome::UntrustedIssuer("did:x:z".to_string()));
    }

    // ── Builder chaining ──────────────────────────────────────────────────────

    #[test]
    fn test_builder_chain_all_options() {
        let req = PresentationRequest::new("r99", "c99", "d99")
            .require_type("T1")
            .require_type("T2")
            .trust_issuer("did:example:i1")
            .trust_issuer("did:example:i2")
            .with_expiry(99_999);
        assert_eq!(req.request_id, "r99");
        assert_eq!(req.challenge, "c99");
        assert_eq!(req.domain, "d99");
        assert_eq!(req.required_types.len(), 2);
        assert_eq!(req.trusted_issuers.len(), 2);
        assert_eq!(req.expires_at_ms, Some(99_999));
    }
}
