//! # Verifiable Presentation Builder
//!
//! W3C Verifiable Presentation builder following the VC Data Model 1.1 spec.
//!
//! # Example
//!
//! ```rust
//! use oxirs_did::presentation_builder::{
//!     PresentationBuilder, VerifiableCredential, CredentialSubject, ProofBlock,
//! };
//! use std::collections::HashMap;
//!
//! let subject = CredentialSubject {
//!     id: Some("did:example:holder".to_string()),
//!     claims: HashMap::from([("name".to_string(), "Alice".to_string())]),
//! };
//! let vc = VerifiableCredential {
//!     context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
//!     id: None,
//!     types: vec!["VerifiableCredential".to_string()],
//!     issuer: "did:example:issuer".to_string(),
//!     issuance_date: "2024-01-01T00:00:00Z".to_string(),
//!     expiration_date: None,
//!     subject,
//!     proof: None,
//! };
//! let vp = PresentationBuilder::new()
//!     .holder("did:example:holder")
//!     .add_credential(vc)
//!     .build()
//!     .expect("build ok");
//! assert!(vp.is_valid_structure());
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the presentation builder / parser
#[derive(Debug, Clone, PartialEq)]
pub enum PresentationError {
    /// No credentials were added to the presentation
    MissingCredentials,
    /// The built or parsed structure is invalid
    InvalidStructure(String),
    /// JSON parsing failed
    JsonParseError(String),
}

impl fmt::Display for PresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PresentationError::MissingCredentials => {
                write!(f, "Presentation must include at least one credential")
            }
            PresentationError::InvalidStructure(msg) => {
                write!(f, "Invalid presentation structure: {msg}")
            }
            PresentationError::JsonParseError(msg) => {
                write!(f, "JSON parse error: {msg}")
            }
        }
    }
}

impl std::error::Error for PresentationError {}

// ---------------------------------------------------------------------------
// Credential subject
// ---------------------------------------------------------------------------

/// The subject of a Verifiable Credential
#[derive(Debug, Clone)]
pub struct CredentialSubject {
    /// Optional DID of the subject
    pub id: Option<String>,
    /// Additional claims (key → value)
    pub claims: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Proof block
// ---------------------------------------------------------------------------

/// A cryptographic proof attached to a credential or presentation
#[derive(Debug, Clone)]
pub struct ProofBlock {
    /// Proof type (e.g. `Ed25519Signature2020`)
    pub proof_type: String,
    /// ISO-8601 creation timestamp
    pub created: String,
    /// Verification method reference
    pub verification_method: String,
    /// Purpose (e.g. `assertionMethod`, `authentication`)
    pub proof_purpose: String,
    /// Encoded proof value (multibase / base58)
    pub proof_value: String,
}

// ---------------------------------------------------------------------------
// Verifiable Credential
// ---------------------------------------------------------------------------

/// A W3C Verifiable Credential
#[derive(Debug, Clone)]
pub struct VerifiableCredential {
    /// JSON-LD context URLs
    pub context: Vec<String>,
    /// Optional credential identifier
    pub id: Option<String>,
    /// Credential types (must include `VerifiableCredential`)
    pub types: Vec<String>,
    /// Issuer DID
    pub issuer: String,
    /// Issuance date (ISO-8601)
    pub issuance_date: String,
    /// Optional expiration date (ISO-8601)
    pub expiration_date: Option<String>,
    /// Credential subject
    pub subject: CredentialSubject,
    /// Optional attached proof
    pub proof: Option<ProofBlock>,
}

// ---------------------------------------------------------------------------
// Verifiable Presentation
// ---------------------------------------------------------------------------

/// A W3C Verifiable Presentation
#[derive(Debug, Clone)]
pub struct VerifiablePresentation {
    /// JSON-LD context URLs
    pub context: Vec<String>,
    /// Optional presentation identifier
    pub id: Option<String>,
    /// Presentation types (must include `VerifiablePresentation`)
    pub types: Vec<String>,
    /// Holder DID
    pub holder: Option<String>,
    /// Included Verifiable Credentials
    pub credentials: Vec<VerifiableCredential>,
    /// Challenge for replay-protection
    pub challenge: Option<String>,
    /// Domain binding
    pub domain: Option<String>,
    /// Optional proof on the presentation itself
    proof: Option<ProofBlock>,
}

impl VerifiablePresentation {
    /// Attach a proof to the presentation
    pub fn add_proof(&mut self, proof: ProofBlock) {
        self.proof = Some(proof);
    }

    /// Number of credentials contained in this presentation
    pub fn credential_count(&self) -> usize {
        self.credentials.len()
    }

    /// Structural validity check:
    /// - must have type `VerifiablePresentation`
    /// - must contain at least one credential
    pub fn is_valid_structure(&self) -> bool {
        let has_vp_type = self.types.iter().any(|t| t == "VerifiablePresentation");
        has_vp_type && !self.credentials.is_empty()
    }

    /// References to all credential subjects
    pub fn credential_subjects(&self) -> Vec<&CredentialSubject> {
        self.credentials.iter().map(|vc| &vc.subject).collect()
    }

    // -----------------------------------------------------------------------
    // JSON serialisation (hand-rolled, no serde dependency added)
    // -----------------------------------------------------------------------

    /// Serialise to a JSON string
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");

        // @context
        out.push_str("  \"@context\": [");
        let ctx_items: Vec<String> = self
            .context
            .iter()
            .map(|s| format!("\"{}\"", escape_json(s)))
            .collect();
        out.push_str(&ctx_items.join(", "));
        out.push_str("],\n");

        // id (optional)
        if let Some(ref id) = self.id {
            out.push_str(&format!("  \"id\": \"{}\",\n", escape_json(id)));
        }

        // type
        out.push_str("  \"type\": [");
        let type_items: Vec<String> = self
            .types
            .iter()
            .map(|t| format!("\"{}\"", escape_json(t)))
            .collect();
        out.push_str(&type_items.join(", "));
        out.push_str("],\n");

        // holder
        if let Some(ref h) = self.holder {
            out.push_str(&format!("  \"holder\": \"{}\",\n", escape_json(h)));
        }

        // challenge
        if let Some(ref c) = self.challenge {
            out.push_str(&format!("  \"challenge\": \"{}\",\n", escape_json(c)));
        }

        // domain
        if let Some(ref d) = self.domain {
            out.push_str(&format!("  \"domain\": \"{}\",\n", escape_json(d)));
        }

        // verifiableCredential
        out.push_str("  \"verifiableCredential\": [\n");
        let vc_strs: Vec<String> = self.credentials.iter().map(vc_to_json).collect();
        out.push_str(&vc_strs.join(",\n"));
        out.push_str("\n  ]");

        // proof
        if let Some(ref p) = self.proof {
            out.push_str(",\n  \"proof\": ");
            out.push_str(&proof_to_json(p));
        }

        out.push_str("\n}");
        out
    }

    /// Parse a `VerifiablePresentation` from a JSON string.
    ///
    /// This is a simplified parser that extracts the top-level scalar fields
    /// and a minimal credential list.  It does **not** require serde.
    pub fn from_json(s: &str) -> Result<Self, PresentationError> {
        // Extract types array
        let types = extract_string_array(s, "\"type\"")
            .or_else(|| extract_string_array(s, "\"@type\""))
            .unwrap_or_default();

        if types.is_empty() {
            return Err(PresentationError::JsonParseError(
                "Missing 'type' field".to_string(),
            ));
        }

        let context = extract_string_array(s, "\"@context\"")
            .unwrap_or_else(|| vec!["https://www.w3.org/2018/credentials/v1".to_string()]);

        let id = extract_string_value(s, "\"id\"");
        let holder = extract_string_value(s, "\"holder\"");
        let challenge = extract_string_value(s, "\"challenge\"");
        let domain = extract_string_value(s, "\"domain\"");

        // Parse credentials (simplified: extract issuer and issuanceDate per vc block)
        let credentials = parse_credentials_from_json(s);

        Ok(VerifiablePresentation {
            context,
            id,
            types,
            holder,
            credentials,
            challenge,
            domain,
            proof: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for `VerifiablePresentation`
pub struct PresentationBuilder {
    id: Option<String>,
    holder: Option<String>,
    credentials: Vec<VerifiableCredential>,
    context: Vec<String>,
    types: Vec<String>,
    challenge: Option<String>,
    domain: Option<String>,
}

impl PresentationBuilder {
    /// Create a new builder with sensible defaults
    pub fn new() -> Self {
        Self {
            id: None,
            holder: None,
            credentials: Vec::new(),
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            types: vec!["VerifiablePresentation".to_string()],
            challenge: None,
            domain: None,
        }
    }

    /// Set the presentation identifier
    pub fn id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Set the holder DID
    pub fn holder(mut self, did: &str) -> Self {
        self.holder = Some(did.to_string());
        self
    }

    /// Add a Verifiable Credential
    pub fn add_credential(mut self, vc: VerifiableCredential) -> Self {
        self.credentials.push(vc);
        self
    }

    /// Add an additional JSON-LD context URL
    pub fn add_context(mut self, ctx: &str) -> Self {
        let ctx_str = ctx.to_string();
        if !self.context.contains(&ctx_str) {
            self.context.push(ctx_str);
        }
        self
    }

    /// Add an additional type
    pub fn add_type(mut self, t: &str) -> Self {
        let t_str = t.to_string();
        if !self.types.contains(&t_str) {
            self.types.push(t_str);
        }
        self
    }

    /// Set a replay-protection challenge string
    pub fn challenge(mut self, c: &str) -> Self {
        self.challenge = Some(c.to_string());
        self
    }

    /// Set the domain binding
    pub fn domain(mut self, d: &str) -> Self {
        self.domain = Some(d.to_string());
        self
    }

    /// Build the `VerifiablePresentation`, returning an error if validation fails
    pub fn build(self) -> Result<VerifiablePresentation, PresentationError> {
        if self.credentials.is_empty() {
            return Err(PresentationError::MissingCredentials);
        }
        Ok(VerifiablePresentation {
            context: self.context,
            id: self.id,
            types: self.types,
            holder: self.holder,
            credentials: self.credentials,
            challenge: self.challenge,
            domain: self.domain,
            proof: None,
        })
    }
}

impl Default for PresentationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Escape a string for inclusion in a JSON string literal
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Serialise a `ProofBlock` to a JSON object string
fn proof_to_json(p: &ProofBlock) -> String {
    format!(
        "{{\n    \"type\": \"{}\",\n    \"created\": \"{}\",\n    \"verificationMethod\": \"{}\",\n    \"proofPurpose\": \"{}\",\n    \"proofValue\": \"{}\"\n  }}",
        escape_json(&p.proof_type),
        escape_json(&p.created),
        escape_json(&p.verification_method),
        escape_json(&p.proof_purpose),
        escape_json(&p.proof_value),
    )
}

/// Serialise a `VerifiableCredential` to a JSON object string
fn vc_to_json(vc: &VerifiableCredential) -> String {
    let mut out = String::from("    {\n");

    // @context
    out.push_str("      \"@context\": [");
    let ctx_items: Vec<String> = vc
        .context
        .iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect();
    out.push_str(&ctx_items.join(", "));
    out.push_str("],\n");

    // id
    if let Some(ref id) = vc.id {
        out.push_str(&format!("      \"id\": \"{}\",\n", escape_json(id)));
    }

    // type
    out.push_str("      \"type\": [");
    let type_items: Vec<String> = vc
        .types
        .iter()
        .map(|t| format!("\"{}\"", escape_json(t)))
        .collect();
    out.push_str(&type_items.join(", "));
    out.push_str("],\n");

    // issuer
    out.push_str(&format!(
        "      \"issuer\": \"{}\",\n",
        escape_json(&vc.issuer)
    ));

    // issuanceDate
    out.push_str(&format!(
        "      \"issuanceDate\": \"{}\",\n",
        escape_json(&vc.issuance_date)
    ));

    // expirationDate
    if let Some(ref exp) = vc.expiration_date {
        out.push_str(&format!(
            "      \"expirationDate\": \"{}\",\n",
            escape_json(exp)
        ));
    }

    // credentialSubject
    out.push_str("      \"credentialSubject\": {\n");
    if let Some(ref id) = vc.subject.id {
        out.push_str(&format!("        \"id\": \"{}\"", escape_json(id)));
        if !vc.subject.claims.is_empty() {
            out.push(',');
        }
        out.push('\n');
    }
    let claim_items: Vec<String> = vc
        .subject
        .claims
        .iter()
        .map(|(k, v)| format!("        \"{}\": \"{}\"", escape_json(k), escape_json(v)))
        .collect();
    out.push_str(&claim_items.join(",\n"));
    if !claim_items.is_empty() {
        out.push('\n');
    }
    out.push_str("      }");

    // proof
    if let Some(ref p) = vc.proof {
        out.push_str(",\n      \"proof\": ");
        out.push_str(
            &proof_to_json(p)
                .replace("  {", "      {")
                .replace("  }", "      }"),
        );
    }

    out.push_str("\n    }");
    out
}

/// Extract a scalar string value for a JSON key.
/// Only handles simple `"key": "value"` patterns.
fn extract_string_value(json: &str, key: &str) -> Option<String> {
    let key_pos = json.find(key)?;
    let after_key = &json[key_pos + key.len()..];
    // Skip to the colon, then to the opening quote
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..];
    let open_quote = after_colon.find('"')?;
    let rest = &after_colon[open_quote + 1..];
    let close_quote = rest.find('"')?;
    Some(unescape_json(&rest[..close_quote]))
}

/// Extract an array of strings from a JSON key.
/// Handles `"key": ["v1", "v2"]` patterns.
fn extract_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let key_pos = json.find(key)?;
    let after_key = &json[key_pos + key.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = &after_key[colon_pos + 1..];
    let open_bracket = after_colon.find('[')?;
    let after_open = &after_colon[open_bracket + 1..];
    let close_bracket = after_open.find(']')?;
    let array_content = &after_open[..close_bracket];

    let items: Vec<String> = array_content
        .split(',')
        .filter_map(|item| {
            let trimmed = item.trim();
            if trimmed.starts_with('"') && trimmed.ends_with('"') {
                Some(unescape_json(&trimmed[1..trimmed.len() - 1]))
            } else {
                None
            }
        })
        .collect();

    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

/// Unescape basic JSON escape sequences
fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

/// Very simplified credential parser — extracts issuer + issuanceDate pairs
fn parse_credentials_from_json(json: &str) -> Vec<VerifiableCredential> {
    // Find "verifiableCredential" array
    let Some(vc_pos) = json.find("\"verifiableCredential\"") else {
        return Vec::new();
    };
    let after = &json[vc_pos..];
    let Some(open_bracket) = after.find('[') else {
        return Vec::new();
    };
    let array_start = vc_pos + open_bracket + 1;

    // Walk through the array collecting top-level {} blocks
    let mut depth = 1i32;
    let mut pos = array_start;
    let bytes = json.as_bytes();
    let mut blocks: Vec<&str> = Vec::new();
    let mut block_start: Option<usize> = None;

    while pos < bytes.len() && depth > 0 {
        match bytes[pos] {
            b'[' if block_start.is_none() => depth += 1,
            b']' if block_start.is_none() => {
                depth -= 1;
            }
            b'{' => {
                if depth == 1 {
                    block_start = Some(pos);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 1 {
                    if let Some(start) = block_start.take() {
                        blocks.push(&json[start..=pos]);
                    }
                }
            }
            _ => {}
        }
        pos += 1;
    }

    blocks
        .into_iter()
        .map(|block| {
            let issuer =
                extract_string_value(block, "\"issuer\"").unwrap_or_else(|| "unknown".to_string());
            let issuance_date = extract_string_value(block, "\"issuanceDate\"")
                .unwrap_or_else(|| "unknown".to_string());
            let id = extract_string_value(block, "\"id\"");
            let types = extract_string_array(block, "\"type\"")
                .unwrap_or_else(|| vec!["VerifiableCredential".to_string()]);
            let context = extract_string_array(block, "\"@context\"")
                .unwrap_or_else(|| vec!["https://www.w3.org/2018/credentials/v1".to_string()]);
            let subject_id = extract_string_value(block, "\"id\"");
            VerifiableCredential {
                context,
                id,
                types,
                issuer,
                issuance_date,
                expiration_date: extract_string_value(block, "\"expirationDate\""),
                subject: CredentialSubject {
                    id: subject_id,
                    claims: HashMap::new(),
                },
                proof: None,
            }
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vc() -> VerifiableCredential {
        VerifiableCredential {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            id: Some("urn:uuid:vc-1".to_string()),
            types: vec![
                "VerifiableCredential".to_string(),
                "UniversityDegree".to_string(),
            ],
            issuer: "did:example:issuer".to_string(),
            issuance_date: "2024-01-15T10:00:00Z".to_string(),
            expiration_date: Some("2025-01-15T10:00:00Z".to_string()),
            subject: CredentialSubject {
                id: Some("did:example:holder".to_string()),
                claims: HashMap::from([
                    ("degree".to_string(), "Bachelor of Science".to_string()),
                    ("gpa".to_string(), "3.8".to_string()),
                ]),
            },
            proof: None,
        }
    }

    fn sample_proof() -> ProofBlock {
        ProofBlock {
            proof_type: "Ed25519Signature2020".to_string(),
            created: "2024-01-15T10:01:00Z".to_string(),
            verification_method: "did:example:issuer#key-1".to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "z3FXVHm5X5wZYZ".to_string(),
        }
    }

    // --- Builder tests ---

    #[test]
    fn test_builder_basic() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .expect("should build");
        assert!(vp.is_valid_structure());
    }

    #[test]
    fn test_builder_missing_credentials() {
        let err = PresentationBuilder::new().build().unwrap_err();
        assert_eq!(err, PresentationError::MissingCredentials);
    }

    #[test]
    fn test_builder_with_id() {
        let vp = PresentationBuilder::new()
            .id("urn:uuid:pres-123")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert_eq!(vp.id.as_deref(), Some("urn:uuid:pres-123"));
    }

    #[test]
    fn test_builder_with_holder() {
        let vp = PresentationBuilder::new()
            .holder("did:example:holder")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert_eq!(vp.holder.as_deref(), Some("did:example:holder"));
    }

    #[test]
    fn test_builder_with_challenge() {
        let vp = PresentationBuilder::new()
            .challenge("random-nonce-42")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert_eq!(vp.challenge.as_deref(), Some("random-nonce-42"));
    }

    #[test]
    fn test_builder_with_domain() {
        let vp = PresentationBuilder::new()
            .domain("https://verifier.example")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert_eq!(vp.domain.as_deref(), Some("https://verifier.example"));
    }

    #[test]
    fn test_builder_add_context_dedup() {
        let vp = PresentationBuilder::new()
            .add_context("https://www.w3.org/2018/credentials/v1") // already present
            .add_context("https://schema.org")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        // Must not duplicate the default context
        let count = vp
            .context
            .iter()
            .filter(|c| c.as_str() == "https://www.w3.org/2018/credentials/v1")
            .count();
        assert_eq!(count, 1);
        assert!(vp.context.contains(&"https://schema.org".to_string()));
    }

    #[test]
    fn test_builder_add_type() {
        let vp = PresentationBuilder::new()
            .add_type("CredentialManagerPresentation")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert!(vp
            .types
            .contains(&"CredentialManagerPresentation".to_string()));
        assert!(vp.types.contains(&"VerifiablePresentation".to_string()));
    }

    #[test]
    fn test_builder_multiple_credentials() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert_eq!(vp.credential_count(), 2);
    }

    // --- Validity ---

    #[test]
    fn test_is_valid_structure_true() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert!(vp.is_valid_structure());
    }

    #[test]
    fn test_is_valid_structure_no_vp_type() {
        let mut vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        vp.types = vec!["SomethingElse".to_string()];
        assert!(!vp.is_valid_structure());
    }

    // --- Proof ---

    #[test]
    fn test_add_proof() {
        let mut vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        assert!(vp.proof.is_none());
        vp.add_proof(sample_proof());
        assert!(vp.proof.is_some());
    }

    #[test]
    fn test_proof_clone() {
        let p = sample_proof();
        let p2 = p.clone();
        assert_eq!(p.proof_type, p2.proof_type);
    }

    // --- Credential subjects ---

    #[test]
    fn test_credential_subjects() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let subjects = vp.credential_subjects();
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0].id.as_deref(), Some("did:example:holder"));
    }

    // --- JSON serialisation ---

    #[test]
    fn test_to_json_contains_type() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("VerifiablePresentation"));
    }

    #[test]
    fn test_to_json_contains_context() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("https://www.w3.org/2018/credentials/v1"));
    }

    #[test]
    fn test_to_json_contains_holder() {
        let vp = PresentationBuilder::new()
            .holder("did:example:alice")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("did:example:alice"));
    }

    #[test]
    fn test_to_json_contains_issuer() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("did:example:issuer"));
    }

    #[test]
    fn test_to_json_contains_challenge() {
        let vp = PresentationBuilder::new()
            .challenge("abc-123")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("abc-123"));
    }

    #[test]
    fn test_to_json_contains_domain() {
        let vp = PresentationBuilder::new()
            .domain("https://verifier.example")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        assert!(json.contains("https://verifier.example"));
    }

    #[test]
    fn test_to_json_with_proof() {
        let mut vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        vp.add_proof(sample_proof());
        let json = vp.to_json();
        assert!(json.contains("Ed25519Signature2020"));
        assert!(json.contains("assertionMethod"));
    }

    // --- JSON round-trip ---

    #[test]
    fn test_from_json_round_trip_type() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        let parsed = VerifiablePresentation::from_json(&json).unwrap();
        assert!(parsed.types.contains(&"VerifiablePresentation".to_string()));
    }

    #[test]
    fn test_from_json_round_trip_holder() {
        let vp = PresentationBuilder::new()
            .holder("did:example:holder")
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        let parsed = VerifiablePresentation::from_json(&json).unwrap();
        assert_eq!(parsed.holder.as_deref(), Some("did:example:holder"));
    }

    #[test]
    fn test_from_json_missing_type() {
        let err = VerifiablePresentation::from_json("{}").unwrap_err();
        matches!(err, PresentationError::JsonParseError(_));
    }

    #[test]
    fn test_from_json_credential_count() {
        let vp = PresentationBuilder::new()
            .add_credential(sample_vc())
            .build()
            .unwrap();
        let json = vp.to_json();
        let parsed = VerifiablePresentation::from_json(&json).unwrap();
        assert_eq!(parsed.credential_count(), 1);
    }

    // --- Error display ---

    #[test]
    fn test_error_display_missing_credentials() {
        let e = PresentationError::MissingCredentials;
        assert!(e.to_string().contains("credential"));
    }

    #[test]
    fn test_error_display_invalid_structure() {
        let e = PresentationError::InvalidStructure("oops".to_string());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn test_error_display_json_parse() {
        let e = PresentationError::JsonParseError("bad json".to_string());
        assert!(e.to_string().contains("bad json"));
    }

    // --- escape_json ---

    #[test]
    fn test_escape_json_quotes() {
        let escaped = escape_json("say \"hello\"");
        assert_eq!(escaped, r#"say \"hello\""#);
    }

    #[test]
    fn test_escape_json_backslash() {
        let escaped = escape_json(r"c:\path");
        assert_eq!(escaped, r"c:\\path");
    }
}
