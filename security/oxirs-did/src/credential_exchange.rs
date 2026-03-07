//! Verifiable Credential exchange protocols (W3C VC Data Model inspired).
//!
//! Implements VP creation, verification, credential selection, and
//! a base64-JSON encoding (not real JWT — no cryptographic signing).

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A Verifiable Credential.
#[derive(Debug, Clone)]
pub struct VerifiableCredential {
    /// Credential identifier (IRI or UUID).
    pub id: String,
    /// Credential type identifiers (e.g. `["VerifiableCredential", "AgeCredential"]`).
    pub types: Vec<String>,
    /// DID or IRI of the issuer.
    pub issuer: String,
    /// DID or IRI of the credential subject.
    pub subject: String,
    /// Arbitrary claim key-value pairs.
    pub claims: HashMap<String, String>,
    /// Unix timestamp (seconds) when the credential was issued.
    pub issued_at: u64,
    /// Unix timestamp (seconds) when the credential expires. `None` = no expiry.
    pub expires_at: Option<u64>,
}

impl VerifiableCredential {
    /// Create a new, non-expiring credential.
    pub fn new(
        id: impl Into<String>,
        types: Vec<String>,
        issuer: impl Into<String>,
        subject: impl Into<String>,
        claims: HashMap<String, String>,
        issued_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            types,
            issuer: issuer.into(),
            subject: subject.into(),
            claims,
            issued_at,
            expires_at: None,
        }
    }

    /// Builder: set an expiry timestamp.
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Add a claim.
    pub fn with_claim(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.claims.insert(key.into(), value.into());
        self
    }
}

/// A Verifiable Presentation.
#[derive(Debug, Clone)]
pub struct VerifiablePresentation {
    /// Presentation identifier.
    pub id: String,
    /// DID or IRI of the holder.
    pub holder: String,
    /// Credentials included in this presentation.
    pub credentials: Vec<VerifiableCredential>,
    /// Optional base64-encoded proof string.
    pub proof: Option<String>,
}

/// A presentation request from a verifier.
#[derive(Debug, Clone)]
pub struct PresentationRequest {
    /// Request identifier.
    pub id: String,
    /// DID or IRI of the verifier.
    pub verifier: String,
    /// Credential types the verifier requires (e.g. `["AgeCredential"]`).
    pub required_types: Vec<String>,
    /// Claim keys that must be present in at least one credential.
    pub required_claims: Vec<String>,
    /// Random challenge for replay protection.
    pub challenge: String,
}

/// Result of VP verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the presentation satisfies the request.
    pub valid: bool,
    /// Errors that caused validation failure.
    pub errors: Vec<String>,
    /// Non-fatal warnings.
    pub warnings: Vec<String>,
}

impl VerificationResult {
    fn new() -> Self {
        Self {
            valid: false,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CredentialExchange
// ──────────────────────────────────────────────────────────────────────────────

/// Manages credential exchange between holders, issuers, and verifiers.
pub struct CredentialExchange;

impl CredentialExchange {
    /// Create a new `CredentialExchange`.
    pub fn new() -> Self {
        Self
    }

    /// Create a `VerifiablePresentation` from a holder's credentials in
    /// response to a `PresentationRequest`.
    ///
    /// Only credentials that match the request's required types are included.
    /// A simple proof string encodes the holder + challenge.
    pub fn create_presentation(
        &self,
        holder: impl Into<String>,
        credentials: Vec<VerifiableCredential>,
        request: &PresentationRequest,
    ) -> VerifiablePresentation {
        let holder = holder.into();
        let selected = self.select_credentials(&credentials, request);
        let proof = Some(simple_proof(&holder, &request.challenge));
        VerifiablePresentation {
            id: format!("urn:vp:{}", pseudorandom_id(&holder, &request.id)),
            holder,
            credentials: selected,
            proof,
        }
    }

    /// Verify that a `VerifiablePresentation` satisfies a `PresentationRequest`.
    pub fn verify_presentation(
        &self,
        vp: &VerifiablePresentation,
        request: &PresentationRequest,
    ) -> VerificationResult {
        let now = current_unix_secs();
        let mut result = VerificationResult::new();

        // 1. Check required credential types
        for required_type in &request.required_types {
            let satisfied = vp
                .credentials
                .iter()
                .any(|vc| vc.types.contains(required_type));
            if !satisfied {
                result
                    .errors
                    .push(format!("missing required credential type: {required_type}"));
            }
        }

        // 2. Check required claims
        for required_claim in &request.required_claims {
            let satisfied = vp
                .credentials
                .iter()
                .any(|vc| vc.claims.contains_key(required_claim));
            if !satisfied {
                result
                    .errors
                    .push(format!("missing required claim: {required_claim}"));
            }
        }

        // 3. Check expiry on each credential
        for vc in &vp.credentials {
            if is_expired(vc, now) {
                result
                    .errors
                    .push(format!("credential '{}' has expired", vc.id));
            }
        }

        // 4. Warn when no proof is attached
        if vp.proof.is_none() {
            result
                .warnings
                .push("no cryptographic proof attached".to_string());
        }

        // 5. Warn when presentation has no credentials
        if vp.credentials.is_empty() {
            result
                .warnings
                .push("presentation contains no credentials".to_string());
        }

        result.valid = result.errors.is_empty();
        result
    }

    /// Select credentials that match the required types in `request`.
    pub fn select_credentials(
        &self,
        available: &[VerifiableCredential],
        request: &PresentationRequest,
    ) -> Vec<VerifiableCredential> {
        if request.required_types.is_empty() {
            return available.to_vec();
        }
        available
            .iter()
            .filter(|vc| {
                request
                    .required_types
                    .iter()
                    .any(|rt| vc.types.contains(rt))
            })
            .cloned()
            .collect()
    }

    /// Encode a presentation as a base64 JSON string (not a real JWT).
    pub fn encode_jwt_like(&self, vp: &VerifiablePresentation) -> String {
        // Build a simple JSON-like string, then base64-encode it.
        let cred_count = vp.credentials.len();
        let proof = vp.proof.as_deref().unwrap_or("none");
        let json = format!(
            r#"{{"id":"{}","holder":"{}","credential_count":{},"proof":"{}"}}"#,
            vp.id, vp.holder, cred_count, proof
        );
        base64_encode(json.as_bytes())
    }
}

impl Default for CredentialExchange {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Standalone utility functions
// ──────────────────────────────────────────────────────────────────────────────

/// Return `true` if `vc` has expired before `now_secs`.
pub fn is_expired(vc: &VerifiableCredential, now_secs: u64) -> bool {
    vc.expires_at.map(|exp| now_secs >= exp).unwrap_or(false)
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

fn current_unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn simple_proof(holder: &str, challenge: &str) -> String {
    format!("holder={holder};challenge={challenge}")
}

/// Deterministic pseudo-ID from two strings (not cryptographic).
fn pseudorandom_id(a: &str, b: &str) -> String {
    let hash: u64 = a.bytes().chain(b.bytes()).fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    });
    format!("{hash:016x}")
}

/// Minimal base64 encoding (no external dependency).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        output.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        output.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        output.push(if chunk.len() > 1 {
            ALPHABET[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    output
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn age_vc(subject: &str, age: &str, issued: u64) -> VerifiableCredential {
        let mut claims = HashMap::new();
        claims.insert("age".to_string(), age.to_string());
        VerifiableCredential::new(
            format!("urn:vc:age:{subject}"),
            vec!["VerifiableCredential".into(), "AgeCredential".into()],
            "did:example:issuer",
            subject,
            claims,
            issued,
        )
    }

    fn email_vc(subject: &str, email: &str, issued: u64) -> VerifiableCredential {
        let mut claims = HashMap::new();
        claims.insert("email".to_string(), email.to_string());
        VerifiableCredential::new(
            format!("urn:vc:email:{subject}"),
            vec!["VerifiableCredential".into(), "EmailCredential".into()],
            "did:example:issuer",
            subject,
            claims,
            issued,
        )
    }

    fn age_request() -> PresentationRequest {
        PresentationRequest {
            id: "req-001".to_string(),
            verifier: "did:example:verifier".to_string(),
            required_types: vec!["AgeCredential".to_string()],
            required_claims: vec!["age".to_string()],
            challenge: "nonce-xyz".to_string(),
        }
    }

    // ── VerifiableCredential ──────────────────────────────────────────────────

    #[test]
    fn test_vc_creation() {
        let vc = age_vc("alice", "30", 1_700_000_000);
        assert_eq!(vc.subject, "alice");
        assert!(vc.types.contains(&"AgeCredential".to_string()));
        assert_eq!(vc.claims["age"], "30");
    }

    #[test]
    fn test_vc_no_expiry_by_default() {
        let vc = age_vc("alice", "30", 1_700_000_000);
        assert!(vc.expires_at.is_none());
    }

    #[test]
    fn test_vc_with_expiry() {
        let vc = age_vc("alice", "30", 1_700_000_000).with_expiry(2_000_000_000);
        assert_eq!(vc.expires_at, Some(2_000_000_000));
    }

    #[test]
    fn test_vc_with_claim_builder() {
        let vc = age_vc("alice", "30", 0).with_claim("country", "EE");
        assert_eq!(vc.claims["country"], "EE");
    }

    // ── is_expired ────────────────────────────────────────────────────────────

    #[test]
    fn test_not_expired_no_expiry() {
        let vc = age_vc("alice", "30", 0);
        assert!(!is_expired(&vc, 9_999_999_999));
    }

    #[test]
    fn test_not_expired_before_expiry() {
        let vc = age_vc("alice", "30", 0).with_expiry(2_000_000_000);
        assert!(!is_expired(&vc, 1_000_000_000));
    }

    #[test]
    fn test_expired_at_boundary() {
        let vc = age_vc("alice", "30", 0).with_expiry(1_000);
        assert!(is_expired(&vc, 1_000)); // now_secs >= expires_at → expired
    }

    #[test]
    fn test_expired_after_expiry() {
        let vc = age_vc("alice", "30", 0).with_expiry(500);
        assert!(is_expired(&vc, 1_000));
    }

    // ── create_presentation ───────────────────────────────────────────────────

    #[test]
    fn test_create_presentation_holder() {
        let exchange = CredentialExchange::new();
        let vp = exchange.create_presentation(
            "did:example:alice",
            vec![age_vc("alice", "30", 0)],
            &age_request(),
        );
        assert_eq!(vp.holder, "did:example:alice");
    }

    #[test]
    fn test_create_presentation_has_proof() {
        let exchange = CredentialExchange::new();
        let vp = exchange.create_presentation(
            "did:example:alice",
            vec![age_vc("alice", "30", 0)],
            &age_request(),
        );
        assert!(vp.proof.is_some());
        let proof = vp.proof.unwrap();
        assert!(proof.contains("alice"));
        assert!(proof.contains("nonce-xyz"));
    }

    #[test]
    fn test_create_presentation_filters_credentials() {
        let exchange = CredentialExchange::new();
        let creds = vec![
            age_vc("alice", "30", 0),
            email_vc("alice", "alice@example.com", 0),
        ];
        let vp = exchange.create_presentation("did:example:alice", creds, &age_request());
        // Only AgeCredential should be included
        assert_eq!(vp.credentials.len(), 1);
        assert!(vp.credentials[0]
            .types
            .contains(&"AgeCredential".to_string()));
    }

    #[test]
    fn test_create_presentation_id_not_empty() {
        let exchange = CredentialExchange::new();
        let vp = exchange.create_presentation("did:example:alice", vec![], &age_request());
        assert!(!vp.id.is_empty());
    }

    // ── select_credentials ────────────────────────────────────────────────────

    #[test]
    fn test_select_credentials_by_type() {
        let exchange = CredentialExchange::new();
        let creds = vec![
            age_vc("alice", "30", 0),
            email_vc("alice", "alice@example.com", 0),
        ];
        let selected = exchange.select_credentials(&creds, &age_request());
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_select_credentials_empty_request_returns_all() {
        let exchange = CredentialExchange::new();
        let creds = vec![age_vc("a", "30", 0), email_vc("a", "a@b.com", 0)];
        let request = PresentationRequest {
            id: "r".into(),
            verifier: "v".into(),
            required_types: vec![],
            required_claims: vec![],
            challenge: "c".into(),
        };
        let selected = exchange.select_credentials(&creds, &request);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_credentials_no_match() {
        let exchange = CredentialExchange::new();
        let creds = vec![age_vc("a", "30", 0)];
        let request = PresentationRequest {
            id: "r".into(),
            verifier: "v".into(),
            required_types: vec!["DegreeCredential".into()],
            required_claims: vec![],
            challenge: "c".into(),
        };
        let selected = exchange.select_credentials(&creds, &request);
        assert!(selected.is_empty());
    }

    // ── verify_presentation ───────────────────────────────────────────────────

    #[test]
    fn test_verify_presentation_valid() {
        let exchange = CredentialExchange::new();
        let vp = exchange.create_presentation(
            "did:example:alice",
            vec![age_vc("alice", "30", 0)],
            &age_request(),
        );
        let result = exchange.verify_presentation(&vp, &age_request());
        assert!(result.valid, "Errors: {:?}", result.errors);
    }

    #[test]
    fn test_verify_presentation_missing_type() {
        let exchange = CredentialExchange::new();
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![email_vc("alice", "a@b.com", 0)],
            proof: Some("p".into()),
        };
        let result = exchange.verify_presentation(&vp, &age_request());
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("AgeCredential")));
    }

    #[test]
    fn test_verify_presentation_missing_claim() {
        let exchange = CredentialExchange::new();
        // AgeCredential type but no "age" claim
        let mut claims = HashMap::new();
        claims.insert("name".to_string(), "Alice".to_string());
        let vc = VerifiableCredential::new(
            "urn:vc:1",
            vec!["VerifiableCredential".into(), "AgeCredential".into()],
            "issuer",
            "alice",
            claims,
            0,
        );
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![vc],
            proof: Some("p".into()),
        };
        let result = exchange.verify_presentation(&vp, &age_request());
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("age")));
    }

    #[test]
    fn test_verify_presentation_expired_credential() {
        let exchange = CredentialExchange::new();
        let expired_vc = age_vc("alice", "30", 0).with_expiry(1); // expired long ago
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![expired_vc],
            proof: Some("p".into()),
        };
        let result = exchange.verify_presentation(&vp, &age_request());
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("expired")));
    }

    #[test]
    fn test_verify_presentation_no_proof_warning() {
        let exchange = CredentialExchange::new();
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![age_vc("alice", "30", 0)],
            proof: None,
        };
        let result = exchange.verify_presentation(&vp, &age_request());
        assert!(result.warnings.iter().any(|w| w.contains("proof")));
    }

    #[test]
    fn test_verify_presentation_empty_credentials_warning() {
        let exchange = CredentialExchange::new();
        let request = PresentationRequest {
            id: "r".into(),
            verifier: "v".into(),
            required_types: vec![],
            required_claims: vec![],
            challenge: "c".into(),
        };
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![],
            proof: None,
        };
        let result = exchange.verify_presentation(&vp, &request);
        assert!(result.warnings.iter().any(|w| w.contains("no credentials")));
    }

    // ── encode_jwt_like ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_jwt_like_non_empty() {
        let exchange = CredentialExchange::new();
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "did:example:alice".into(),
            credentials: vec![],
            proof: Some("proof-value".into()),
        };
        let encoded = exchange.encode_jwt_like(&vp);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_jwt_like_base64_chars() {
        let exchange = CredentialExchange::new();
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "holder".into(),
            credentials: vec![],
            proof: None,
        };
        let encoded = exchange.encode_jwt_like(&vp);
        assert!(
            encoded
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='),
            "encoded contains non-base64 chars"
        );
    }

    #[test]
    fn test_encode_jwt_like_different_vps_differ() {
        let exchange = CredentialExchange::new();
        let vp1 = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "alice".into(),
            credentials: vec![],
            proof: None,
        };
        let vp2 = VerifiablePresentation {
            id: "vp-2".into(),
            holder: "bob".into(),
            credentials: vec![],
            proof: None,
        };
        assert_ne!(
            exchange.encode_jwt_like(&vp1),
            exchange.encode_jwt_like(&vp2)
        );
    }

    // ── base64_encode ─────────────────────────────────────────────────────────

    #[test]
    fn test_base64_encode_hello() {
        // "Hello" → "SGVsbG8="
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_base64_encode_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn test_base64_encode_man() {
        // "Man" → "TWFu"
        assert_eq!(base64_encode(b"Man"), "TWFu");
    }

    // ── CredentialExchange default ────────────────────────────────────────────

    #[test]
    fn test_credential_exchange_default() {
        let _ = CredentialExchange::new();
    }

    // ── Multiple claims and types ─────────────────────────────────────────────

    #[test]
    fn test_vc_multiple_claims() {
        let mut claims = HashMap::new();
        claims.insert("age".to_string(), "25".to_string());
        claims.insert("name".to_string(), "Bob".to_string());
        claims.insert("email".to_string(), "bob@example.com".to_string());
        let vc = VerifiableCredential::new(
            "urn:vc:1",
            vec!["VerifiableCredential".into(), "IdentityCredential".into()],
            "issuer",
            "bob",
            claims,
            0,
        );
        assert_eq!(vc.claims.len(), 3);
    }

    #[test]
    fn test_select_multiple_matching_types() {
        let exchange = CredentialExchange::new();
        let creds = vec![
            age_vc("alice", "30", 0),
            age_vc("alice", "30", 1), // second AgeCredential
        ];
        let selected = exchange.select_credentials(&creds, &age_request());
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_verify_presentation_multiple_errors() {
        let exchange = CredentialExchange::new();
        let request = PresentationRequest {
            id: "r".into(),
            verifier: "v".into(),
            required_types: vec!["TypeA".into(), "TypeB".into()],
            required_claims: vec!["claimA".into(), "claimB".into()],
            challenge: "c".into(),
        };
        let vp = VerifiablePresentation {
            id: "vp-1".into(),
            holder: "h".into(),
            credentials: vec![],
            proof: Some("p".into()),
        };
        let result = exchange.verify_presentation(&vp, &request);
        assert!(!result.valid);
        // Should have errors for both types and both claims
        assert!(result.errors.len() >= 2);
    }
}
