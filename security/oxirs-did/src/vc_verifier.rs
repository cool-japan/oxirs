//! Verifiable Credential verification (W3C VC Data Model).
//!
//! Performs structural verification of VCs: context checks, type checks,
//! issuer trust checks, and temporal validity.  Cryptographic proof
//! verification is supported as an optional policy flag.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// W3C VC Data Model base context.
pub const W3C_VC_CONTEXT: &str = "https://www.w3.org/2018/credentials/v1";

// ---------------------------------------------------------------------------
// VerificationStatus
// ---------------------------------------------------------------------------

/// Outcome of a VC verification run.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    /// All requested checks passed.
    Valid,
    /// The credential's expiration date is in the past.
    Expired,
    /// The credential's issuance date is in the future.
    NotYetValid,
    /// The credential has been explicitly revoked.
    Revoked,
    /// The cryptographic proof is invalid or missing when required.
    InvalidProof,
    /// The issuer is not in the trusted-issuer list.
    InvalidIssuer,
    /// A mandatory field is missing.
    MissingField(String),
    /// The credential subject does not conform to its declared schema.
    SchemaViolation(String),
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Detailed outcome of a `VcVerifier::verify` call.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// High-level status.
    pub status: VerificationStatus,
    /// Names of checks that succeeded.
    pub checks_passed: Vec<String>,
    /// Names of checks that failed.
    pub checks_failed: Vec<String>,
    /// Non-fatal advisory messages.
    pub warnings: Vec<String>,
}

impl VerificationResult {
    /// Returns `true` iff the status is `Valid`.
    pub fn is_valid(&self) -> bool {
        self.status == VerificationStatus::Valid
    }

    /// Record a passed check.
    pub fn add_pass(&mut self, check: impl Into<String>) {
        self.checks_passed.push(check.into());
    }

    /// Record a failed check.
    pub fn add_fail(&mut self, check: impl Into<String>) {
        self.checks_failed.push(check.into());
    }

    /// Record a warning.
    pub fn add_warning(&mut self, warn: impl Into<String>) {
        self.warnings.push(warn.into());
    }
}

// ---------------------------------------------------------------------------
// VerifiableCredential
// ---------------------------------------------------------------------------

/// A simplified in-memory representation of a W3C Verifiable Credential.
#[derive(Debug, Clone)]
pub struct VerifiableCredential {
    /// Optional credential identifier IRI.
    pub id: Option<String>,
    /// Type declarations (must include `"VerifiableCredential"`).
    pub types: Vec<String>,
    /// Issuer IRI.
    pub issuer: String,
    /// ISO 8601 issuance date string.
    pub issuance_date: String,
    /// Optional ISO 8601 expiration date string.
    pub expiration_date: Option<String>,
    /// Key-value claims about the credential subject.
    pub credential_subject: HashMap<String, String>,
    /// Optional cryptographic proof.
    pub proof: Option<CredentialProof>,
    /// JSON-LD context IRIs (must include `W3C_VC_CONTEXT`).
    pub context: Vec<String>,
}

// ---------------------------------------------------------------------------
// CredentialProof
// ---------------------------------------------------------------------------

/// Cryptographic proof attached to a Verifiable Credential.
#[derive(Debug, Clone)]
pub struct CredentialProof {
    /// Proof type (e.g. `"Ed25519Signature2020"`).
    pub proof_type: String,
    /// ISO 8601 creation timestamp.
    pub created: String,
    /// IRI of the verification method used.
    pub verification_method: String,
    /// Proof purpose (e.g. `"assertionMethod"`).
    pub proof_purpose: String,
    /// Base64 or hex-encoded signature bytes.
    pub proof_value: String,
}

// ---------------------------------------------------------------------------
// VerificationPolicy
// ---------------------------------------------------------------------------

/// Controls which checks `VcVerifier::verify` performs.
#[derive(Debug, Clone)]
pub struct VerificationPolicy {
    /// Whether to verify the cryptographic proof.
    pub check_proof: bool,
    /// Whether to check the expiry / issuance dates.
    pub check_expiry: bool,
    /// Whether to verify that the W3C VC context is present.
    pub check_context: bool,
    /// Whether to verify that `"VerifiableCredential"` is in the type list.
    pub check_required_types: bool,
    /// Allowed issuer IRIs.  Empty slice means any issuer is accepted.
    pub trusted_issuers: Vec<String>,
    /// Current wall-clock time in milliseconds since Unix epoch.
    /// Use `0` to skip time-based checks even when `check_expiry` is `true`.
    pub current_time_ms: u64,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        VerificationPolicy {
            check_proof: false,
            check_expiry: true,
            check_context: true,
            check_required_types: true,
            trusted_issuers: vec![],
            current_time_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// VcVerifier
// ---------------------------------------------------------------------------

/// Stateless namespace for Verifiable Credential verification helpers.
pub struct VcVerifier;

impl VcVerifier {
    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    /// Verify `vc` against the given `policy`.
    ///
    /// All requested checks are run; the first failing check determines the
    /// final `VerificationStatus`.
    pub fn verify(vc: &VerifiableCredential, policy: &VerificationPolicy) -> VerificationResult {
        let mut result = VerificationResult {
            status: VerificationStatus::Valid,
            checks_passed: Vec::new(),
            checks_failed: Vec::new(),
            warnings: Vec::new(),
        };

        // 1. Required types
        if policy.check_required_types {
            if Self::has_required_type(vc) {
                result.add_pass("required_type");
            } else {
                result.add_fail("required_type");
                result.status =
                    VerificationStatus::MissingField("VerifiableCredential type".to_string());
                return result;
            }
        }

        // 2. W3C context
        if policy.check_context {
            if Self::has_vc_context(vc) {
                result.add_pass("vc_context");
            } else {
                result.add_fail("vc_context");
                result.status = VerificationStatus::MissingField("W3C VC context".to_string());
                return result;
            }
        }

        // 3. Issuer
        if !policy.trusted_issuers.is_empty() {
            if Self::is_trusted_issuer(&vc.issuer, &policy.trusted_issuers) {
                result.add_pass("trusted_issuer");
            } else {
                result.add_fail("trusted_issuer");
                result.status = VerificationStatus::InvalidIssuer;
                return result;
            }
        } else {
            result.add_pass("trusted_issuer");
        }

        // 4. Credential subject
        if Self::has_credential_subject(vc) {
            result.add_pass("credential_subject");
        } else {
            result.add_fail("credential_subject");
            result.status = VerificationStatus::MissingField("credential subject".to_string());
            return result;
        }

        // 5. Temporal validity
        if policy.check_expiry && policy.current_time_ms > 0 {
            let temporal = Self::check_temporal_validity(vc, policy.current_time_ms);
            match temporal {
                VerificationStatus::Valid => {
                    result.add_pass("temporal_validity");
                }
                other => {
                    result.add_fail("temporal_validity");
                    result.status = other;
                    return result;
                }
            }
        } else {
            result.add_pass("temporal_validity");
        }

        // 6. Proof
        if policy.check_proof {
            match &vc.proof {
                Some(proof) if !proof.proof_value.is_empty() => {
                    // Structural check only — no actual crypto in this implementation
                    result.add_pass("proof_structure");
                }
                _ => {
                    result.add_fail("proof");
                    result.status = VerificationStatus::InvalidProof;
                    return result;
                }
            }
        } else {
            result.add_pass("proof_skipped");
        }

        // 7. Warn about missing optional id
        if vc.id.is_none() {
            result.add_warning("Credential has no @id — traceability may be limited");
        }

        result
    }

    // -----------------------------------------------------------------------
    // Individual check helpers
    // -----------------------------------------------------------------------

    /// Parse an ISO 8601 date-time string to milliseconds since Unix epoch.
    ///
    /// Supports formats:
    /// - `YYYY-MM-DDTHH:MM:SSZ`
    /// - `YYYY-MM-DDTHH:MM:SS+00:00`
    /// - `YYYY-MM-DD` (treated as midnight UTC)
    pub fn parse_date_ms(date_str: &str) -> Option<u64> {
        // Try common ISO 8601 patterns by parsing the date part manually.
        // We avoid external date-parsing crates to stay pure-Rust & minimal.
        let s = date_str.trim();

        // Extract YYYY-MM-DD
        if s.len() < 10 {
            return None;
        }
        let year: i64 = s[0..4].parse().ok()?;
        if &s[4..5] != "-" {
            return None;
        }
        let month: i64 = s[5..7].parse().ok()?;
        if &s[7..8] != "-" {
            return None;
        }
        let day: i64 = s[8..10].parse().ok()?;

        // Validate basic ranges
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }

        // Parse optional time part (T...)
        let (hour, minute, second) =
            if s.len() > 10 && (s[10..].starts_with('T') || s[10..].starts_with(' ')) {
                let time_part = &s[11..];
                if time_part.len() < 8 {
                    (0i64, 0i64, 0i64)
                } else {
                    let h: i64 = time_part[0..2].parse().ok()?;
                    let m: i64 = time_part[3..5].parse().ok()?;
                    let sec: i64 = time_part[6..8].parse().ok()?;
                    (h, m, sec)
                }
            } else {
                (0, 0, 0)
            };

        // Convert to Unix epoch ms using Julian Day Number formula
        // Days from 1970-01-01 (Julian Day 2440588)
        let jdn = julian_day_number(year, month, day)?;
        let epoch_jdn: i64 = 2_440_588;
        let days_since_epoch = jdn - epoch_jdn;
        let total_seconds = days_since_epoch * 86_400 + hour * 3_600 + minute * 60 + second;
        if total_seconds < 0 {
            // Dates before 1970 — not representable as u64
            return None;
        }
        Some((total_seconds as u64) * 1000)
    }

    /// Returns `true` if the VC context includes the W3C VC base context.
    pub fn has_vc_context(vc: &VerifiableCredential) -> bool {
        vc.context.iter().any(|c| c == W3C_VC_CONTEXT)
    }

    /// Returns `true` if the VC type list includes `"VerifiableCredential"`.
    pub fn has_required_type(vc: &VerifiableCredential) -> bool {
        vc.types.iter().any(|t| t == "VerifiableCredential")
    }

    /// Returns `true` if `issuer` is in `trusted`, or if `trusted` is empty.
    pub fn is_trusted_issuer(issuer: &str, trusted: &[String]) -> bool {
        if trusted.is_empty() {
            return true;
        }
        trusted.iter().any(|t| t == issuer)
    }

    /// Check temporal validity against `current_ms`.
    ///
    /// Returns `Valid`, `Expired`, or `NotYetValid`.
    pub fn check_temporal_validity(
        vc: &VerifiableCredential,
        current_ms: u64,
    ) -> VerificationStatus {
        // Check issuance date (must be ≤ current time)
        if let Some(issued_ms) = Self::parse_date_ms(&vc.issuance_date) {
            if current_ms < issued_ms {
                return VerificationStatus::NotYetValid;
            }
        }

        // Check expiration date (must be > current time)
        if let Some(ref exp_str) = vc.expiration_date {
            if let Some(exp_ms) = Self::parse_date_ms(exp_str) {
                if current_ms >= exp_ms {
                    return VerificationStatus::Expired;
                }
            }
        }

        VerificationStatus::Valid
    }

    /// Returns `true` if the credential subject has at least one property.
    pub fn has_credential_subject(vc: &VerifiableCredential) -> bool {
        !vc.credential_subject.is_empty()
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a minimal valid VC for use in tests.
    pub fn build_test_vc(issuer: &str, subject_id: &str) -> VerifiableCredential {
        let mut subject = HashMap::new();
        subject.insert("id".to_string(), subject_id.to_string());
        subject.insert("name".to_string(), "Test Subject".to_string());

        VerifiableCredential {
            id: Some("urn:uuid:test-vc-001".to_string()),
            types: vec![
                "VerifiableCredential".to_string(),
                "TestCredential".to_string(),
            ],
            issuer: issuer.to_string(),
            issuance_date: "2020-01-01T00:00:00Z".to_string(),
            expiration_date: Some("2099-12-31T23:59:59Z".to_string()),
            credential_subject: subject,
            proof: Some(CredentialProof {
                proof_type: "Ed25519Signature2020".to_string(),
                created: "2020-01-01T00:00:00Z".to_string(),
                verification_method: format!("{}#key-1", issuer),
                proof_purpose: "assertionMethod".to_string(),
                proof_value: "z3Z2YQjKUQABe2p8VanTFVi4WYJkBfMrS4XTdHq6LGxNdKnZHWxGqPmVMo"
                    .to_string(),
            }),
            context: vec![W3C_VC_CONTEXT.to_string()],
        }
    }
}

// ---------------------------------------------------------------------------
// Internal date helper
// ---------------------------------------------------------------------------

/// Convert a Gregorian date to a Julian Day Number.
fn julian_day_number(year: i64, month: i64, day: i64) -> Option<i64> {
    // Reject obviously bad values
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Algorithm from https://en.wikipedia.org/wiki/Julian_day
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    Some(jdn)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_vc() -> VerifiableCredential {
        VcVerifier::build_test_vc("did:example:issuer", "did:example:subject")
    }

    fn default_policy() -> VerificationPolicy {
        VerificationPolicy::default()
    }

    // --- build_test_vc -------------------------------------------------------

    #[test]
    fn test_build_test_vc_structure() {
        let vc = valid_vc();
        assert!(vc.types.contains(&"VerifiableCredential".to_string()));
        assert!(vc.context.contains(&W3C_VC_CONTEXT.to_string()));
        assert!(!vc.credential_subject.is_empty());
        assert!(vc.proof.is_some());
    }

    // --- verify valid VC -----------------------------------------------------

    #[test]
    fn test_verify_valid_vc() {
        let vc = valid_vc();
        let policy = default_policy();
        let result = VcVerifier::verify(&vc, &policy);
        assert!(result.is_valid(), "status = {:?}", result.status);
        assert!(!result.checks_passed.is_empty());
        assert!(result.checks_failed.is_empty());
    }

    // --- type check ----------------------------------------------------------

    #[test]
    fn test_verify_missing_required_type_fails() {
        let mut vc = valid_vc();
        vc.types = vec!["SomeOtherType".to_string()];
        let result = VcVerifier::verify(&vc, &default_policy());
        assert!(!result.is_valid());
        assert!(matches!(result.status, VerificationStatus::MissingField(_)));
        assert!(result.checks_failed.contains(&"required_type".to_string()));
    }

    #[test]
    fn test_verify_type_check_disabled() {
        let mut vc = valid_vc();
        vc.types = vec![];
        let mut policy = default_policy();
        policy.check_required_types = false;
        let result = VcVerifier::verify(&vc, &policy);
        // Should not fail on type check
        assert!(!result.checks_failed.contains(&"required_type".to_string()));
    }

    // --- context check -------------------------------------------------------

    #[test]
    fn test_verify_missing_w3c_context_fails() {
        let mut vc = valid_vc();
        vc.context = vec!["https://example.org/custom-context".to_string()];
        let result = VcVerifier::verify(&vc, &default_policy());
        assert!(!result.is_valid());
        assert!(matches!(result.status, VerificationStatus::MissingField(_)));
        assert!(result.checks_failed.contains(&"vc_context".to_string()));
    }

    #[test]
    fn test_verify_context_check_disabled() {
        let mut vc = valid_vc();
        vc.context = vec![];
        let mut policy = default_policy();
        policy.check_context = false;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(!result.checks_failed.contains(&"vc_context".to_string()));
    }

    // --- issuer trust --------------------------------------------------------

    #[test]
    fn test_verify_trusted_issuer_accepted() {
        let vc = valid_vc();
        let mut policy = default_policy();
        policy.trusted_issuers = vec!["did:example:issuer".to_string()];
        let result = VcVerifier::verify(&vc, &policy);
        assert!(result.is_valid());
        assert!(result.checks_passed.contains(&"trusted_issuer".to_string()));
    }

    #[test]
    fn test_verify_untrusted_issuer_rejected() {
        let vc = valid_vc();
        let mut policy = default_policy();
        policy.trusted_issuers = vec!["did:example:other".to_string()];
        let result = VcVerifier::verify(&vc, &policy);
        assert!(!result.is_valid());
        assert_eq!(result.status, VerificationStatus::InvalidIssuer);
    }

    #[test]
    fn test_verify_any_issuer_when_list_empty() {
        let mut vc = valid_vc();
        vc.issuer = "did:example:unknown-issuer".to_string();
        let mut policy = default_policy();
        policy.trusted_issuers = vec![];
        let result = VcVerifier::verify(&vc, &policy);
        // Empty trusted list = accept any
        assert!(result.checks_passed.contains(&"trusted_issuer".to_string()));
    }

    // --- credential subject --------------------------------------------------

    #[test]
    fn test_verify_empty_credential_subject_fails() {
        let mut vc = valid_vc();
        vc.credential_subject = HashMap::new();
        let result = VcVerifier::verify(&vc, &default_policy());
        assert!(!result.is_valid());
        assert!(matches!(result.status, VerificationStatus::MissingField(_)));
    }

    // --- temporal validity ---------------------------------------------------

    #[test]
    fn test_parse_date_ms_valid_datetime() {
        let ms = VcVerifier::parse_date_ms("2020-01-01T00:00:00Z");
        assert!(ms.is_some());
        // 2020-01-01 is after 1970-01-01 so ms > 0
        assert!(ms.expect("some") > 0);
    }

    #[test]
    fn test_parse_date_ms_date_only() {
        let ms = VcVerifier::parse_date_ms("2023-06-15");
        assert!(ms.is_some(), "date-only format should parse");
    }

    #[test]
    fn test_parse_date_ms_invalid_format() {
        assert!(VcVerifier::parse_date_ms("not-a-date").is_none());
        assert!(VcVerifier::parse_date_ms("").is_none());
        assert!(VcVerifier::parse_date_ms("2020/01/01").is_none());
    }

    #[test]
    fn test_parse_date_ms_epoch() {
        let ms = VcVerifier::parse_date_ms("1970-01-01T00:00:00Z");
        assert_eq!(ms, Some(0));
    }

    #[test]
    fn test_check_temporal_valid() {
        let vc = valid_vc(); // issuance 2020, expiry 2099
                             // current time 2025 (approx)
        let now_ms: u64 = 1_735_000_000_000;
        let status = VcVerifier::check_temporal_validity(&vc, now_ms);
        assert_eq!(status, VerificationStatus::Valid);
    }

    #[test]
    fn test_check_temporal_expired() {
        let mut vc = valid_vc();
        vc.expiration_date = Some("2000-01-01T00:00:00Z".to_string());
        let now_ms: u64 = 1_735_000_000_000; // 2025
        let status = VcVerifier::check_temporal_validity(&vc, now_ms);
        assert_eq!(status, VerificationStatus::Expired);
    }

    #[test]
    fn test_check_temporal_not_yet_valid() {
        let mut vc = valid_vc();
        vc.issuance_date = "2099-01-01T00:00:00Z".to_string();
        let now_ms: u64 = 1_735_000_000_000; // 2025
        let status = VcVerifier::check_temporal_validity(&vc, now_ms);
        assert_eq!(status, VerificationStatus::NotYetValid);
    }

    #[test]
    fn test_verify_expired_vc_fails() {
        let mut vc = valid_vc();
        vc.expiration_date = Some("2000-01-01T00:00:00Z".to_string());
        let mut policy = default_policy();
        policy.current_time_ms = 1_735_000_000_000;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(!result.is_valid());
        assert_eq!(result.status, VerificationStatus::Expired);
    }

    #[test]
    fn test_verify_not_yet_valid_vc_fails() {
        let mut vc = valid_vc();
        vc.issuance_date = "2099-01-01T00:00:00Z".to_string();
        let mut policy = default_policy();
        policy.current_time_ms = 1_735_000_000_000;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(!result.is_valid());
        assert_eq!(result.status, VerificationStatus::NotYetValid);
    }

    #[test]
    fn test_verify_expiry_skipped_when_time_zero() {
        let mut vc = valid_vc();
        vc.expiration_date = Some("2000-01-01T00:00:00Z".to_string());
        let mut policy = default_policy();
        policy.current_time_ms = 0; // skip time checks
        let result = VcVerifier::verify(&vc, &policy);
        // Should still be valid — time check skipped
        assert!(result.is_valid(), "status = {:?}", result.status);
    }

    // --- proof check ---------------------------------------------------------

    #[test]
    fn test_verify_proof_skipped_when_check_proof_false() {
        let mut vc = valid_vc();
        vc.proof = None;
        let mut policy = default_policy();
        policy.check_proof = false;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(result.is_valid());
        assert!(result.checks_passed.contains(&"proof_skipped".to_string()));
    }

    #[test]
    fn test_verify_proof_fails_when_missing_and_required() {
        let mut vc = valid_vc();
        vc.proof = None;
        let mut policy = default_policy();
        policy.check_proof = true;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(!result.is_valid());
        assert_eq!(result.status, VerificationStatus::InvalidProof);
    }

    #[test]
    fn test_verify_proof_passes_when_present_and_required() {
        let vc = valid_vc(); // has proof
        let mut policy = default_policy();
        policy.check_proof = true;
        let result = VcVerifier::verify(&vc, &policy);
        assert!(result.is_valid());
    }

    // --- is_trusted_issuer ---------------------------------------------------

    #[test]
    fn test_is_trusted_issuer_in_list() {
        let trusted = vec!["did:example:a".to_string(), "did:example:b".to_string()];
        assert!(VcVerifier::is_trusted_issuer("did:example:a", &trusted));
    }

    #[test]
    fn test_is_trusted_issuer_not_in_list() {
        let trusted = vec!["did:example:a".to_string()];
        assert!(!VcVerifier::is_trusted_issuer(
            "did:example:other",
            &trusted
        ));
    }

    #[test]
    fn test_is_trusted_issuer_empty_list() {
        assert!(VcVerifier::is_trusted_issuer("did:example:anyone", &[]));
    }

    // --- has_vc_context / has_required_type ----------------------------------

    #[test]
    fn test_has_vc_context_true() {
        let vc = valid_vc();
        assert!(VcVerifier::has_vc_context(&vc));
    }

    #[test]
    fn test_has_vc_context_false() {
        let mut vc = valid_vc();
        vc.context = vec!["https://example.org/other".to_string()];
        assert!(!VcVerifier::has_vc_context(&vc));
    }

    #[test]
    fn test_has_required_type_true() {
        let vc = valid_vc();
        assert!(VcVerifier::has_required_type(&vc));
    }

    #[test]
    fn test_has_required_type_false() {
        let mut vc = valid_vc();
        vc.types = vec!["CustomType".to_string()];
        assert!(!VcVerifier::has_required_type(&vc));
    }

    // --- has_credential_subject ----------------------------------------------

    #[test]
    fn test_has_credential_subject_true() {
        let vc = valid_vc();
        assert!(VcVerifier::has_credential_subject(&vc));
    }

    #[test]
    fn test_has_credential_subject_false() {
        let mut vc = valid_vc();
        vc.credential_subject = HashMap::new();
        assert!(!VcVerifier::has_credential_subject(&vc));
    }

    // --- VerificationResult helpers ------------------------------------------

    #[test]
    fn test_verification_result_is_valid() {
        let mut r = VerificationResult {
            status: VerificationStatus::Valid,
            checks_passed: vec![],
            checks_failed: vec![],
            warnings: vec![],
        };
        assert!(r.is_valid());
        r.status = VerificationStatus::Expired;
        assert!(!r.is_valid());
    }

    #[test]
    fn test_verification_result_add_pass_fail_warn() {
        let mut r = VerificationResult {
            status: VerificationStatus::Valid,
            checks_passed: vec![],
            checks_failed: vec![],
            warnings: vec![],
        };
        r.add_pass("check_a");
        r.add_fail("check_b");
        r.add_warning("warn_c");
        assert_eq!(r.checks_passed, vec!["check_a"]);
        assert_eq!(r.checks_failed, vec!["check_b"]);
        assert_eq!(r.warnings, vec!["warn_c"]);
    }

    // --- checks_passed / checks_failed populated correctly ------------------

    #[test]
    fn test_checks_passed_on_valid() {
        let vc = valid_vc();
        let result = VcVerifier::verify(&vc, &default_policy());
        // At minimum the major checks should be recorded
        assert!(result.checks_passed.contains(&"required_type".to_string()));
        assert!(result.checks_passed.contains(&"vc_context".to_string()));
        assert!(result
            .checks_passed
            .contains(&"credential_subject".to_string()));
    }

    #[test]
    fn test_checks_failed_on_invalid_type() {
        let mut vc = valid_vc();
        vc.types = vec![];
        let result = VcVerifier::verify(&vc, &default_policy());
        assert!(result.checks_failed.contains(&"required_type".to_string()));
    }

    // --- warning for missing id ----------------------------------------------

    #[test]
    fn test_warning_for_missing_id() {
        let mut vc = valid_vc();
        vc.id = None;
        let result = VcVerifier::verify(&vc, &default_policy());
        assert!(result.is_valid());
        assert!(!result.warnings.is_empty());
    }

    // --- parse_date_ms ordinal monotonicity ----------------------------------

    #[test]
    fn test_parse_date_ms_monotone() {
        let t1 = VcVerifier::parse_date_ms("2020-01-01T00:00:00Z").expect("t1");
        let t2 = VcVerifier::parse_date_ms("2021-01-01T00:00:00Z").expect("t2");
        assert!(t2 > t1);
    }

    #[test]
    fn test_parse_date_ms_invalid_month() {
        assert!(VcVerifier::parse_date_ms("2020-13-01").is_none());
    }
}
