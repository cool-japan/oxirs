//! # Linked Data Proof Purpose Validation
//!
//! Implements the W3C Linked Data Proofs proof purpose framework for DID-based
//! cryptographic verification.  Each proof purpose encodes the *intent* of a
//! proof (authentication, making claims, establishing shared secrets, exercising
//! or delegating capabilities) and is validated against the verification
//! relationships declared in the controller's DID document.
//!
//! ## Features
//!
//! - Five standard W3C proof purposes: Authentication, AssertionMethod,
//!   KeyAgreement, CapabilityInvocation, CapabilityDelegation
//! - Custom proof purpose registration with user-defined validators
//! - Multi-step proof chain verification (ordered chain of proofs)
//! - Purpose validation against a simulated DID document relationship set
//! - Challenge / domain / expiry guards for authentication proofs
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::proof_purpose::{
//!     ProofPurposeValidator, ProofPurposeKind, ProofEntry,
//!     DidRelationships, ValidationContext,
//! };
//!
//! let mut rels = DidRelationships::new("did:example:alice");
//! rels.add_authentication("did:example:alice#key-1");
//! rels.add_assertion_method("did:example:alice#key-2");
//!
//! let ctx = ValidationContext {
//!     challenge: Some("nonce-abc".to_string()),
//!     domain: Some("example.com".to_string()),
//!     current_time_ms: 1_700_000_000_000,
//! };
//!
//! let entry = ProofEntry {
//!     purpose: ProofPurposeKind::Authentication,
//!     verification_method: "did:example:alice#key-1".to_string(),
//!     created_ms: 1_699_999_000_000,
//!     expires_ms: Some(1_700_100_000_000),
//!     challenge: Some("nonce-abc".to_string()),
//!     domain: Some("example.com".to_string()),
//!     nonce: None,
//! };
//!
//! let validator = ProofPurposeValidator::new();
//! let result = validator.validate(&entry, &rels, &ctx);
//! assert!(result.is_ok());
//! ```

use std::collections::{HashMap, HashSet};

// ─── Purpose kinds ───────────────────────────────────────────────────────────

/// Standard W3C proof purpose kinds plus a custom extension point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProofPurposeKind {
    /// Prove the identity of the DID subject (challenge-response).
    Authentication,
    /// Make verifiable claims / issue credentials.
    AssertionMethod,
    /// Establish a shared secret (e.g. ECDH key agreement).
    KeyAgreement,
    /// Exercise an authorization capability.
    CapabilityInvocation,
    /// Delegate an authorization capability to another party.
    CapabilityDelegation,
    /// User-defined proof purpose identified by a URI.
    Custom(String),
}

impl ProofPurposeKind {
    /// The canonical string identifier for this purpose.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Authentication => "authentication",
            Self::AssertionMethod => "assertionMethod",
            Self::KeyAgreement => "keyAgreement",
            Self::CapabilityInvocation => "capabilityInvocation",
            Self::CapabilityDelegation => "capabilityDelegation",
            Self::Custom(uri) => uri.as_str(),
        }
    }

    /// Parse from a string identifier.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "authentication" => Self::Authentication,
            "assertionMethod" => Self::AssertionMethod,
            "keyAgreement" => Self::KeyAgreement,
            "capabilityInvocation" => Self::CapabilityInvocation,
            "capabilityDelegation" => Self::CapabilityDelegation,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Whether this is one of the five standard purposes.
    pub fn is_standard(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

impl std::fmt::Display for ProofPurposeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── DID document relationship sets ──────────────────────────────────────────

/// A simulated DID-document relationship set.
///
/// Records which verification-method IDs are authorised for each proof purpose
/// for a given controller DID.
#[derive(Debug, Clone)]
pub struct DidRelationships {
    /// The controller DID this set belongs to.
    pub controller: String,
    /// Mapping from proof purpose kind to the set of authorised verification method IDs.
    relationships: HashMap<String, HashSet<String>>,
}

impl DidRelationships {
    /// Create a new empty relationship set for `controller`.
    pub fn new(controller: impl Into<String>) -> Self {
        Self {
            controller: controller.into(),
            relationships: HashMap::new(),
        }
    }

    // ── Convenience adders for standard purposes ────────────────────────

    /// Authorise `vm_id` for the `authentication` relationship.
    pub fn add_authentication(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::Authentication, vm_id);
    }

    /// Authorise `vm_id` for the `assertionMethod` relationship.
    pub fn add_assertion_method(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::AssertionMethod, vm_id);
    }

    /// Authorise `vm_id` for the `keyAgreement` relationship.
    pub fn add_key_agreement(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::KeyAgreement, vm_id);
    }

    /// Authorise `vm_id` for the `capabilityInvocation` relationship.
    pub fn add_capability_invocation(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::CapabilityInvocation, vm_id);
    }

    /// Authorise `vm_id` for the `capabilityDelegation` relationship.
    pub fn add_capability_delegation(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::CapabilityDelegation, vm_id);
    }

    /// Authorise `vm_id` for a custom proof purpose.
    pub fn add_custom(&mut self, purpose_uri: impl Into<String>, vm_id: impl Into<String>) {
        let purpose = ProofPurposeKind::Custom(purpose_uri.into());
        self.add(purpose, vm_id);
    }

    /// Generic adder.
    pub fn add(&mut self, purpose: ProofPurposeKind, vm_id: impl Into<String>) {
        self.relationships
            .entry(purpose.as_str().to_string())
            .or_default()
            .insert(vm_id.into());
    }

    /// Check whether `vm_id` is authorised for `purpose`.
    pub fn is_authorised(&self, purpose: &ProofPurposeKind, vm_id: &str) -> bool {
        self.relationships
            .get(purpose.as_str())
            .is_some_and(|set| set.contains(vm_id))
    }

    /// Return all verification method IDs authorised for `purpose`.
    pub fn methods_for(&self, purpose: &ProofPurposeKind) -> Vec<String> {
        self.relationships
            .get(purpose.as_str())
            .map_or_else(Vec::new, |set| {
                let mut v: Vec<String> = set.iter().cloned().collect();
                v.sort();
                v
            })
    }

    /// Return the set of all registered purpose kinds.
    pub fn purposes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.relationships.keys().cloned().collect();
        v.sort();
        v
    }

    /// Total number of (purpose, vm) pairs.
    pub fn total_entries(&self) -> usize {
        self.relationships.values().map(|s| s.len()).sum()
    }

    /// Remove a verification method ID from a given purpose.
    pub fn remove(&mut self, purpose: &ProofPurposeKind, vm_id: &str) -> bool {
        if let Some(set) = self.relationships.get_mut(purpose.as_str()) {
            let removed = set.remove(vm_id);
            if set.is_empty() {
                self.relationships.remove(purpose.as_str());
            }
            removed
        } else {
            false
        }
    }
}

// ─── Proof entry ─────────────────────────────────────────────────────────────

/// A single proof entry to be validated against a DID document.
#[derive(Debug, Clone)]
pub struct ProofEntry {
    /// The declared proof purpose.
    pub purpose: ProofPurposeKind,
    /// The verification method ID used to create this proof.
    pub verification_method: String,
    /// Creation timestamp in milliseconds since Unix epoch.
    pub created_ms: u64,
    /// Optional expiration timestamp in milliseconds since Unix epoch.
    pub expires_ms: Option<u64>,
    /// Challenge string (required for authentication proofs).
    pub challenge: Option<String>,
    /// Domain restriction.
    pub domain: Option<String>,
    /// Optional nonce for replay protection.
    pub nonce: Option<String>,
}

// ─── Validation context ──────────────────────────────────────────────────────

/// Contextual information supplied by the verifier at validation time.
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    /// Expected challenge (for authentication proofs).
    pub challenge: Option<String>,
    /// Expected domain.
    pub domain: Option<String>,
    /// Current wall-clock time in milliseconds since Unix epoch.
    pub current_time_ms: u64,
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by the proof purpose validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PurposeError {
    /// The verification method is not authorised for the declared purpose.
    Unauthorised {
        purpose: String,
        verification_method: String,
    },
    /// The proof has expired.
    Expired { expires_ms: u64, current_ms: u64 },
    /// The proof was created in the future relative to the verifier's clock.
    FutureProof { created_ms: u64, current_ms: u64 },
    /// Challenge mismatch (authentication).
    ChallengeMismatch { expected: String, actual: String },
    /// Challenge required but missing.
    ChallengeMissing,
    /// Domain mismatch.
    DomainMismatch { expected: String, actual: String },
    /// An unknown custom purpose was encountered and no validator is registered.
    UnknownCustomPurpose(String),
    /// Proof chain validation failed at a specific index.
    ChainError { index: usize, reason: String },
    /// Proof chain is empty.
    EmptyChain,
}

impl std::fmt::Display for PurposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unauthorised {
                purpose,
                verification_method,
            } => write!(
                f,
                "Verification method '{verification_method}' is not authorised for purpose '{purpose}'"
            ),
            Self::Expired {
                expires_ms,
                current_ms,
            } => write!(
                f,
                "Proof expired at {expires_ms} ms, current time is {current_ms} ms"
            ),
            Self::FutureProof {
                created_ms,
                current_ms,
            } => write!(
                f,
                "Proof created at {created_ms} ms is in the future (current: {current_ms} ms)"
            ),
            Self::ChallengeMismatch { expected, actual } => {
                write!(
                    f,
                    "Challenge mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::ChallengeMissing => write!(f, "Authentication proof requires a challenge"),
            Self::DomainMismatch { expected, actual } => {
                write!(f, "Domain mismatch: expected '{expected}', got '{actual}'")
            }
            Self::UnknownCustomPurpose(uri) => {
                write!(f, "Unknown custom proof purpose: '{uri}'")
            }
            Self::ChainError { index, reason } => {
                write!(f, "Proof chain error at index {index}: {reason}")
            }
            Self::EmptyChain => write!(f, "Proof chain must contain at least one entry"),
        }
    }
}

impl std::error::Error for PurposeError {}

// ─── Custom purpose validator trait ──────────────────────────────────────────

/// Trait for user-supplied custom proof purpose validators.
///
/// Implement this to teach the validator how to handle non-standard proof
/// purpose URIs.
pub trait CustomPurposeValidator: Send + Sync {
    /// Validate `entry` for the given `relationships` and `context`.
    ///
    /// Return `Ok(())` if valid, or `Err(reason_string)` if invalid.
    fn validate(
        &self,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<(), String>;
}

// ─── Validation result ───────────────────────────────────────────────────────

/// Detailed result of a single-entry validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation succeeded.
    pub valid: bool,
    /// The purpose that was validated.
    pub purpose: String,
    /// The verification method ID that was checked.
    pub verification_method: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Detailed result of a proof-chain validation.
#[derive(Debug, Clone)]
pub struct ChainValidationResult {
    /// Overall success.
    pub valid: bool,
    /// Per-entry results.
    pub entries: Vec<ValidationResult>,
    /// Total number of proofs in the chain.
    pub chain_length: usize,
    /// Error message (if any).
    pub error: Option<String>,
}

// ─── Proof purpose validator ─────────────────────────────────────────────────

/// The main proof purpose validator.
///
/// Validates individual proof entries and ordered proof chains against DID
/// document relationships.  Supports registration of custom purpose validators.
pub struct ProofPurposeValidator {
    /// Registered custom purpose validators keyed by purpose URI.
    custom_validators: HashMap<String, Box<dyn CustomPurposeValidator>>,
    /// Maximum allowed clock skew in milliseconds.
    max_clock_skew_ms: u64,
}

impl Default for ProofPurposeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofPurposeValidator {
    /// Create a new validator with default settings.
    pub fn new() -> Self {
        Self {
            custom_validators: HashMap::new(),
            max_clock_skew_ms: 300_000, // 5 minutes
        }
    }

    /// Set the maximum allowed clock skew in milliseconds.
    pub fn with_max_clock_skew_ms(mut self, ms: u64) -> Self {
        self.max_clock_skew_ms = ms;
        self
    }

    /// Register a custom proof purpose validator.
    pub fn register_custom(
        &mut self,
        purpose_uri: impl Into<String>,
        validator: Box<dyn CustomPurposeValidator>,
    ) {
        self.custom_validators.insert(purpose_uri.into(), validator);
    }

    /// Return the set of registered custom purpose URIs.
    pub fn registered_custom_purposes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.custom_validators.keys().cloned().collect();
        v.sort();
        v
    }

    // ── Single-entry validation ─────────────────────────────────────────

    /// Validate a single proof entry against the DID document relationships.
    pub fn validate(
        &self,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<ValidationResult, PurposeError> {
        // 1. Check authorisation
        if !relationships.is_authorised(&entry.purpose, &entry.verification_method) {
            return Err(PurposeError::Unauthorised {
                purpose: entry.purpose.as_str().to_string(),
                verification_method: entry.verification_method.clone(),
            });
        }

        // 2. Check expiration
        if let Some(expires_ms) = entry.expires_ms {
            if context.current_time_ms > expires_ms {
                return Err(PurposeError::Expired {
                    expires_ms,
                    current_ms: context.current_time_ms,
                });
            }
        }

        // 3. Check future proof (with clock skew tolerance)
        if entry.created_ms > context.current_time_ms + self.max_clock_skew_ms {
            return Err(PurposeError::FutureProof {
                created_ms: entry.created_ms,
                current_ms: context.current_time_ms,
            });
        }

        // 4. Purpose-specific checks
        match &entry.purpose {
            ProofPurposeKind::Authentication => {
                self.validate_authentication(entry, context)?;
            }
            ProofPurposeKind::AssertionMethod => {
                self.validate_assertion_method(entry, context)?;
            }
            ProofPurposeKind::KeyAgreement => {
                self.validate_key_agreement(entry, context)?;
            }
            ProofPurposeKind::CapabilityInvocation => {
                self.validate_capability_invocation(entry, context)?;
            }
            ProofPurposeKind::CapabilityDelegation => {
                self.validate_capability_delegation(entry, context)?;
            }
            ProofPurposeKind::Custom(uri) => {
                self.validate_custom(uri, entry, relationships, context)?;
            }
        }

        Ok(ValidationResult {
            valid: true,
            purpose: entry.purpose.as_str().to_string(),
            verification_method: entry.verification_method.clone(),
            message: format!(
                "Proof purpose '{}' validated successfully for method '{}'",
                entry.purpose.as_str(),
                entry.verification_method
            ),
        })
    }

    // ── Purpose-specific validators ─────────────────────────────────────

    fn validate_authentication(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        // Authentication requires a matching challenge
        if let Some(ref expected) = context.challenge {
            match &entry.challenge {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(PurposeError::ChallengeMismatch {
                        expected: expected.clone(),
                        actual: actual.clone(),
                    });
                }
                None => {
                    return Err(PurposeError::ChallengeMissing);
                }
            }
        }

        // Domain check
        self.check_domain(entry, context)?;

        Ok(())
    }

    fn validate_assertion_method(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        // Assertion method: domain check is optional but enforced if present
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_key_agreement(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        // Key agreement: domain check if present
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_capability_invocation(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        // Capability invocation: domain check if present
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_capability_delegation(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        // Capability delegation: domain check if present
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_custom(
        &self,
        uri: &str,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        if let Some(validator) = self.custom_validators.get(uri) {
            validator
                .validate(entry, relationships, context)
                .map_err(|reason| PurposeError::ChainError { index: 0, reason })
        } else {
            Err(PurposeError::UnknownCustomPurpose(uri.to_string()))
        }
    }

    fn check_domain(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        if let Some(ref expected) = context.domain {
            if let Some(ref actual) = entry.domain {
                if actual != expected {
                    return Err(PurposeError::DomainMismatch {
                        expected: expected.clone(),
                        actual: actual.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    // ── Proof chain validation ──────────────────────────────────────────

    /// Validate an ordered proof chain.
    ///
    /// All entries in the chain must validate individually, and they must be
    /// ordered by creation time (non-decreasing).
    pub fn validate_chain(
        &self,
        chain: &[ProofEntry],
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<ChainValidationResult, PurposeError> {
        if chain.is_empty() {
            return Err(PurposeError::EmptyChain);
        }

        let mut results = Vec::with_capacity(chain.len());
        let mut prev_created_ms: u64 = 0;

        for (i, entry) in chain.iter().enumerate() {
            // Check chain ordering (non-decreasing creation time)
            if entry.created_ms < prev_created_ms {
                return Err(PurposeError::ChainError {
                    index: i,
                    reason: format!(
                        "Proof at index {} has created_ms={} which is before previous entry's created_ms={}",
                        i, entry.created_ms, prev_created_ms
                    ),
                });
            }
            prev_created_ms = entry.created_ms;

            // Validate individual entry
            match self.validate(entry, relationships, context) {
                Ok(result) => results.push(result),
                Err(e) => {
                    return Err(PurposeError::ChainError {
                        index: i,
                        reason: e.to_string(),
                    });
                }
            }
        }

        Ok(ChainValidationResult {
            valid: true,
            chain_length: chain.len(),
            entries: results,
            error: None,
        })
    }

    // ── Utility methods ─────────────────────────────────────────────────

    /// Check whether a purpose kind requires a challenge.
    pub fn requires_challenge(purpose: &ProofPurposeKind) -> bool {
        matches!(purpose, ProofPurposeKind::Authentication)
    }

    /// Return a human-readable description of the purpose.
    pub fn describe_purpose(purpose: &ProofPurposeKind) -> &'static str {
        match purpose {
            ProofPurposeKind::Authentication => {
                "Prove the identity of the DID subject via challenge-response"
            }
            ProofPurposeKind::AssertionMethod => {
                "Make verifiable claims about a subject (e.g. issue credentials)"
            }
            ProofPurposeKind::KeyAgreement => {
                "Establish a shared secret for encrypted communication"
            }
            ProofPurposeKind::CapabilityInvocation => {
                "Exercise an authorization capability granted by a controller"
            }
            ProofPurposeKind::CapabilityDelegation => {
                "Delegate an authorization capability to another party"
            }
            ProofPurposeKind::Custom(_) => "User-defined proof purpose",
        }
    }

    /// Return the DID document relationship name corresponding to a purpose.
    pub fn relationship_name(purpose: &ProofPurposeKind) -> &str {
        purpose.as_str()
    }
}

// ─── Proof purpose registry ──────────────────────────────────────────────────

/// Registry for tracking known proof purposes and their metadata.
///
/// Useful for applications that need to enumerate and describe all available
/// proof purposes (both standard and custom).
#[derive(Debug, Clone)]
pub struct ProofPurposeRegistry {
    /// Registered purposes with human-readable descriptions.
    purposes: HashMap<String, PurposeMetadata>,
}

/// Metadata about a registered proof purpose.
#[derive(Debug, Clone)]
pub struct PurposeMetadata {
    /// The purpose URI / identifier.
    pub purpose_id: String,
    /// Human-readable label.
    pub label: String,
    /// Human-readable description.
    pub description: String,
    /// Whether a challenge is required.
    pub requires_challenge: bool,
    /// Whether a domain restriction is recommended.
    pub recommends_domain: bool,
}

impl Default for ProofPurposeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofPurposeRegistry {
    /// Create a registry pre-populated with the five standard W3C purposes.
    pub fn new() -> Self {
        let mut registry = Self {
            purposes: HashMap::new(),
        };

        registry.register(PurposeMetadata {
            purpose_id: "authentication".to_string(),
            label: "Authentication".to_string(),
            description: "Prove the identity of the DID subject".to_string(),
            requires_challenge: true,
            recommends_domain: true,
        });

        registry.register(PurposeMetadata {
            purpose_id: "assertionMethod".to_string(),
            label: "Assertion Method".to_string(),
            description: "Make verifiable claims about a subject".to_string(),
            requires_challenge: false,
            recommends_domain: false,
        });

        registry.register(PurposeMetadata {
            purpose_id: "keyAgreement".to_string(),
            label: "Key Agreement".to_string(),
            description: "Establish a shared secret for encrypted communication".to_string(),
            requires_challenge: false,
            recommends_domain: false,
        });

        registry.register(PurposeMetadata {
            purpose_id: "capabilityInvocation".to_string(),
            label: "Capability Invocation".to_string(),
            description: "Exercise an authorization capability".to_string(),
            requires_challenge: false,
            recommends_domain: true,
        });

        registry.register(PurposeMetadata {
            purpose_id: "capabilityDelegation".to_string(),
            label: "Capability Delegation".to_string(),
            description: "Delegate an authorization capability".to_string(),
            requires_challenge: false,
            recommends_domain: true,
        });

        registry
    }

    /// Register a new proof purpose.
    pub fn register(&mut self, metadata: PurposeMetadata) {
        self.purposes.insert(metadata.purpose_id.clone(), metadata);
    }

    /// Look up a purpose by ID.
    pub fn get(&self, purpose_id: &str) -> Option<&PurposeMetadata> {
        self.purposes.get(purpose_id)
    }

    /// Check whether a purpose ID is registered.
    pub fn is_registered(&self, purpose_id: &str) -> bool {
        self.purposes.contains_key(purpose_id)
    }

    /// Return all registered purpose IDs (sorted).
    pub fn all_purpose_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.purposes.keys().cloned().collect();
        v.sort();
        v
    }

    /// Total number of registered purposes.
    pub fn count(&self) -> usize {
        self.purposes.len()
    }

    /// Remove a custom purpose by ID. Returns `true` if it existed.
    pub fn unregister(&mut self, purpose_id: &str) -> bool {
        self.purposes.remove(purpose_id).is_some()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ─────────────────────────────────────────────────

    fn make_relationships() -> DidRelationships {
        let mut rels = DidRelationships::new("did:example:alice");
        rels.add_authentication("did:example:alice#key-1");
        rels.add_assertion_method("did:example:alice#key-2");
        rels.add_key_agreement("did:example:alice#key-3");
        rels.add_capability_invocation("did:example:alice#key-4");
        rels.add_capability_delegation("did:example:alice#key-5");
        rels
    }

    fn make_context() -> ValidationContext {
        ValidationContext {
            challenge: Some("challenge-123".to_string()),
            domain: Some("example.com".to_string()),
            current_time_ms: 1_700_000_000_000,
        }
    }

    fn make_entry(purpose: ProofPurposeKind, vm: &str) -> ProofEntry {
        ProofEntry {
            purpose,
            verification_method: vm.to_string(),
            created_ms: 1_699_999_000_000,
            expires_ms: Some(1_700_100_000_000),
            challenge: Some("challenge-123".to_string()),
            domain: Some("example.com".to_string()),
            nonce: None,
        }
    }

    // ── ProofPurposeKind tests ──────────────────────────────────────────

    #[test]
    fn test_purpose_kind_as_str() {
        assert_eq!(ProofPurposeKind::Authentication.as_str(), "authentication");
        assert_eq!(
            ProofPurposeKind::AssertionMethod.as_str(),
            "assertionMethod"
        );
        assert_eq!(ProofPurposeKind::KeyAgreement.as_str(), "keyAgreement");
        assert_eq!(
            ProofPurposeKind::CapabilityInvocation.as_str(),
            "capabilityInvocation"
        );
        assert_eq!(
            ProofPurposeKind::CapabilityDelegation.as_str(),
            "capabilityDelegation"
        );
    }

    #[test]
    fn test_purpose_kind_custom() {
        let custom = ProofPurposeKind::Custom("https://example.com/myPurpose".to_string());
        assert_eq!(custom.as_str(), "https://example.com/myPurpose");
        assert!(!custom.is_standard());
    }

    #[test]
    fn test_purpose_kind_is_standard() {
        assert!(ProofPurposeKind::Authentication.is_standard());
        assert!(ProofPurposeKind::AssertionMethod.is_standard());
        assert!(ProofPurposeKind::KeyAgreement.is_standard());
        assert!(ProofPurposeKind::CapabilityInvocation.is_standard());
        assert!(ProofPurposeKind::CapabilityDelegation.is_standard());
    }

    #[test]
    fn test_purpose_kind_from_str_lossy() {
        assert_eq!(
            ProofPurposeKind::from_str_lossy("authentication"),
            ProofPurposeKind::Authentication
        );
        assert_eq!(
            ProofPurposeKind::from_str_lossy("assertionMethod"),
            ProofPurposeKind::AssertionMethod
        );
        assert_eq!(
            ProofPurposeKind::from_str_lossy("keyAgreement"),
            ProofPurposeKind::KeyAgreement
        );
        assert_eq!(
            ProofPurposeKind::from_str_lossy("capabilityInvocation"),
            ProofPurposeKind::CapabilityInvocation
        );
        assert_eq!(
            ProofPurposeKind::from_str_lossy("capabilityDelegation"),
            ProofPurposeKind::CapabilityDelegation
        );
        let c = ProofPurposeKind::from_str_lossy("custom:xyz");
        assert_eq!(c, ProofPurposeKind::Custom("custom:xyz".to_string()));
    }

    #[test]
    fn test_purpose_kind_display() {
        assert_eq!(
            format!("{}", ProofPurposeKind::Authentication),
            "authentication"
        );
        assert_eq!(
            format!("{}", ProofPurposeKind::Custom("urn:x".to_string())),
            "urn:x"
        );
    }

    // ── DidRelationships tests ──────────────────────────────────────────

    #[test]
    fn test_did_relationships_new() {
        let rels = DidRelationships::new("did:example:bob");
        assert_eq!(rels.controller, "did:example:bob");
        assert_eq!(rels.total_entries(), 0);
    }

    #[test]
    fn test_did_relationships_add_and_check() {
        let rels = make_relationships();
        assert!(rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
        assert!(rels.is_authorised(
            &ProofPurposeKind::AssertionMethod,
            "did:example:alice#key-2"
        ));
        assert!(rels.is_authorised(&ProofPurposeKind::KeyAgreement, "did:example:alice#key-3"));
        assert!(rels.is_authorised(
            &ProofPurposeKind::CapabilityInvocation,
            "did:example:alice#key-4"
        ));
        assert!(rels.is_authorised(
            &ProofPurposeKind::CapabilityDelegation,
            "did:example:alice#key-5"
        ));
    }

    #[test]
    fn test_did_relationships_unauthorised() {
        let rels = make_relationships();
        assert!(!rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-2"));
        assert!(!rels.is_authorised(
            &ProofPurposeKind::AssertionMethod,
            "did:example:alice#key-1"
        ));
    }

    #[test]
    fn test_did_relationships_methods_for() {
        let rels = make_relationships();
        let auth_methods = rels.methods_for(&ProofPurposeKind::Authentication);
        assert_eq!(auth_methods, vec!["did:example:alice#key-1"]);
    }

    #[test]
    fn test_did_relationships_purposes() {
        let rels = make_relationships();
        let purposes = rels.purposes();
        assert_eq!(purposes.len(), 5);
        assert!(purposes.contains(&"authentication".to_string()));
        assert!(purposes.contains(&"assertionMethod".to_string()));
    }

    #[test]
    fn test_did_relationships_total_entries() {
        let rels = make_relationships();
        assert_eq!(rels.total_entries(), 5);
    }

    #[test]
    fn test_did_relationships_remove() {
        let mut rels = make_relationships();
        assert!(rels.remove(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
        assert!(!rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
        assert_eq!(rels.total_entries(), 4);
    }

    #[test]
    fn test_did_relationships_remove_nonexistent() {
        let mut rels = make_relationships();
        assert!(!rels.remove(
            &ProofPurposeKind::Authentication,
            "did:example:alice#key-99"
        ));
    }

    #[test]
    fn test_did_relationships_custom_purpose() {
        let mut rels = DidRelationships::new("did:example:carol");
        rels.add_custom("https://custom.example/sign", "did:example:carol#key-c");
        assert!(rels.is_authorised(
            &ProofPurposeKind::Custom("https://custom.example/sign".to_string()),
            "did:example:carol#key-c"
        ));
    }

    #[test]
    fn test_did_relationships_multiple_methods_per_purpose() {
        let mut rels = DidRelationships::new("did:example:dave");
        rels.add_authentication("did:example:dave#key-a");
        rels.add_authentication("did:example:dave#key-b");
        let methods = rels.methods_for(&ProofPurposeKind::Authentication);
        assert_eq!(methods.len(), 2);
    }

    // ── Validator: authentication purpose ───────────────────────────────

    #[test]
    fn test_validate_authentication_success() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.valid);
        assert_eq!(r.purpose, "authentication");
    }

    #[test]
    fn test_validate_authentication_challenge_mismatch() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        entry.challenge = Some("wrong-challenge".to_string());
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::ChallengeMismatch { .. }));
    }

    #[test]
    fn test_validate_authentication_challenge_missing() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        entry.challenge = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::ChallengeMissing));
    }

    #[test]
    fn test_validate_authentication_domain_mismatch() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        entry.domain = Some("evil.com".to_string());
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::DomainMismatch { .. }));
    }

    // ── Validator: assertion method purpose ──────────────────────────────

    #[test]
    fn test_validate_assertion_method_success() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_assertion_method_unauthorised() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-1");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::Unauthorised { .. }));
    }

    // ── Validator: key agreement purpose ─────────────────────────────────

    #[test]
    fn test_validate_key_agreement_success() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::KeyAgreement, "did:example:alice#key-3");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_key_agreement_unauthorised() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::KeyAgreement, "did:example:alice#key-99");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
    }

    // ── Validator: capability invocation purpose ────────────────────────

    #[test]
    fn test_validate_capability_invocation_success() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(
            ProofPurposeKind::CapabilityInvocation,
            "did:example:alice#key-4",
        );
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_capability_invocation_unauthorised() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(
            ProofPurposeKind::CapabilityInvocation,
            "did:example:alice#key-1",
        );
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
    }

    // ── Validator: capability delegation purpose ────────────────────────

    #[test]
    fn test_validate_capability_delegation_success() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(
            ProofPurposeKind::CapabilityDelegation,
            "did:example:alice#key-5",
        );
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_capability_delegation_unauthorised() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(
            ProofPurposeKind::CapabilityDelegation,
            "did:example:alice#key-1",
        );
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
    }

    // ── Validator: expiration ───────────────────────────────────────────

    #[test]
    fn test_validate_expired_proof() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext {
            current_time_ms: 1_800_000_000_000, // far in the future
            ..Default::default()
        };
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        entry.expires_ms = Some(1_700_000_000_000);
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::Expired { .. }));
    }

    #[test]
    fn test_validate_future_proof() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext {
            current_time_ms: 1_000_000_000_000,
            ..Default::default()
        };
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        entry.created_ms = 1_500_000_000_000; // way in the future
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::FutureProof { .. }));
    }

    #[test]
    fn test_validate_within_clock_skew() {
        let validator = ProofPurposeValidator::new().with_max_clock_skew_ms(600_000);
        let rels = make_relationships();
        let ctx = ValidationContext {
            current_time_ms: 1_700_000_000_000,
            ..Default::default()
        };
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        // created 5 minutes in the future — within 10 min skew
        entry.created_ms = 1_700_000_300_000;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    // ── Custom purpose validators ───────────────────────────────────────

    struct AlwaysAcceptValidator;
    impl CustomPurposeValidator for AlwaysAcceptValidator {
        fn validate(
            &self,
            _entry: &ProofEntry,
            _relationships: &DidRelationships,
            _context: &ValidationContext,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    struct AlwaysRejectValidator;
    impl CustomPurposeValidator for AlwaysRejectValidator {
        fn validate(
            &self,
            _entry: &ProofEntry,
            _relationships: &DidRelationships,
            _context: &ValidationContext,
        ) -> Result<(), String> {
            Err("custom rejection".to_string())
        }
    }

    #[test]
    fn test_custom_purpose_accepted() {
        let mut validator = ProofPurposeValidator::new();
        validator.register_custom(
            "https://example.com/custom",
            Box::new(AlwaysAcceptValidator),
        );
        let mut rels = DidRelationships::new("did:example:custom");
        rels.add_custom("https://example.com/custom", "did:example:custom#key-c");

        let entry = ProofEntry {
            purpose: ProofPurposeKind::Custom("https://example.com/custom".to_string()),
            verification_method: "did:example:custom#key-c".to_string(),
            created_ms: 0,
            expires_ms: None,
            challenge: None,
            domain: None,
            nonce: None,
        };
        let ctx = ValidationContext::default();
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_purpose_rejected() {
        let mut validator = ProofPurposeValidator::new();
        validator.register_custom(
            "https://example.com/custom",
            Box::new(AlwaysRejectValidator),
        );
        let mut rels = DidRelationships::new("did:example:custom");
        rels.add_custom("https://example.com/custom", "did:example:custom#key-c");

        let entry = ProofEntry {
            purpose: ProofPurposeKind::Custom("https://example.com/custom".to_string()),
            verification_method: "did:example:custom#key-c".to_string(),
            created_ms: 0,
            expires_ms: None,
            challenge: None,
            domain: None,
            nonce: None,
        };
        let ctx = ValidationContext::default();
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_custom_purpose() {
        let validator = ProofPurposeValidator::new();
        let mut rels = DidRelationships::new("did:example:unknown");
        rels.add_custom("https://unknown.example/purpose", "did:example:unknown#key");

        let entry = ProofEntry {
            purpose: ProofPurposeKind::Custom("https://unknown.example/purpose".to_string()),
            verification_method: "did:example:unknown#key".to_string(),
            created_ms: 0,
            expires_ms: None,
            challenge: None,
            domain: None,
            nonce: None,
        };
        let ctx = ValidationContext::default();
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::UnknownCustomPurpose(_)));
    }

    #[test]
    fn test_registered_custom_purposes() {
        let mut validator = ProofPurposeValidator::new();
        validator.register_custom("urn:alpha", Box::new(AlwaysAcceptValidator));
        validator.register_custom("urn:beta", Box::new(AlwaysAcceptValidator));
        let purposes = validator.registered_custom_purposes();
        assert_eq!(purposes.len(), 2);
        assert!(purposes.contains(&"urn:alpha".to_string()));
        assert!(purposes.contains(&"urn:beta".to_string()));
    }

    // ── Proof chain validation ──────────────────────────────────────────

    #[test]
    fn test_chain_empty() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let result = validator.validate_chain(&[], &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::EmptyChain));
    }

    #[test]
    fn test_chain_single_entry() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        let result = validator.validate_chain(&[entry], &rels, &ctx);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.valid);
        assert_eq!(r.chain_length, 1);
        assert_eq!(r.entries.len(), 1);
    }

    #[test]
    fn test_chain_multi_step() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();

        let entry1 = ProofEntry {
            purpose: ProofPurposeKind::Authentication,
            verification_method: "did:example:alice#key-1".to_string(),
            created_ms: 1_699_999_000_000,
            expires_ms: Some(1_700_100_000_000),
            challenge: Some("challenge-123".to_string()),
            domain: Some("example.com".to_string()),
            nonce: None,
        };
        let entry2 = ProofEntry {
            purpose: ProofPurposeKind::AssertionMethod,
            verification_method: "did:example:alice#key-2".to_string(),
            created_ms: 1_699_999_500_000,
            expires_ms: None,
            challenge: None,
            domain: Some("example.com".to_string()),
            nonce: None,
        };
        let result = validator.validate_chain(&[entry1, entry2], &rels, &ctx);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert_eq!(r.chain_length, 2);
    }

    #[test]
    fn test_chain_out_of_order() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();

        let entry1 = ProofEntry {
            purpose: ProofPurposeKind::Authentication,
            verification_method: "did:example:alice#key-1".to_string(),
            created_ms: 1_700_000_000_000, // later
            expires_ms: Some(1_700_100_000_000),
            challenge: Some("challenge-123".to_string()),
            domain: Some("example.com".to_string()),
            nonce: None,
        };
        let entry2 = ProofEntry {
            purpose: ProofPurposeKind::AssertionMethod,
            verification_method: "did:example:alice#key-2".to_string(),
            created_ms: 1_699_998_000_000, // earlier (violates non-decreasing)
            expires_ms: None,
            challenge: None,
            domain: Some("example.com".to_string()),
            nonce: None,
        };
        let result = validator.validate_chain(&[entry1, entry2], &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::ChainError { index: 1, .. }));
    }

    #[test]
    fn test_chain_with_invalid_entry() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();

        let good = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        let bad = ProofEntry {
            purpose: ProofPurposeKind::AssertionMethod,
            verification_method: "did:example:alice#key-WRONG".to_string(),
            created_ms: 1_700_000_000_000,
            expires_ms: None,
            challenge: None,
            domain: None,
            nonce: None,
        };
        let result = validator.validate_chain(&[good, bad], &rels, &ctx);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PurposeError::ChainError { index: 1, .. }));
    }

    // ── Proof purpose registry tests ────────────────────────────────────

    #[test]
    fn test_registry_default_has_five_purposes() {
        let reg = ProofPurposeRegistry::new();
        assert_eq!(reg.count(), 5);
    }

    #[test]
    fn test_registry_all_purpose_ids() {
        let reg = ProofPurposeRegistry::new();
        let ids = reg.all_purpose_ids();
        assert!(ids.contains(&"authentication".to_string()));
        assert!(ids.contains(&"assertionMethod".to_string()));
        assert!(ids.contains(&"keyAgreement".to_string()));
        assert!(ids.contains(&"capabilityInvocation".to_string()));
        assert!(ids.contains(&"capabilityDelegation".to_string()));
    }

    #[test]
    fn test_registry_get() {
        let reg = ProofPurposeRegistry::new();
        let auth = reg.get("authentication");
        assert!(auth.is_some());
        let meta = auth.expect("should exist");
        assert_eq!(meta.label, "Authentication");
        assert!(meta.requires_challenge);
        assert!(meta.recommends_domain);
    }

    #[test]
    fn test_registry_get_assertion() {
        let reg = ProofPurposeRegistry::new();
        let am = reg.get("assertionMethod").expect("should exist");
        assert!(!am.requires_challenge);
        assert!(!am.recommends_domain);
    }

    #[test]
    fn test_registry_is_registered() {
        let reg = ProofPurposeRegistry::new();
        assert!(reg.is_registered("keyAgreement"));
        assert!(!reg.is_registered("nonExistentPurpose"));
    }

    #[test]
    fn test_registry_register_custom() {
        let mut reg = ProofPurposeRegistry::new();
        reg.register(PurposeMetadata {
            purpose_id: "https://custom.example/purpose".to_string(),
            label: "Custom Purpose".to_string(),
            description: "A custom proof purpose".to_string(),
            requires_challenge: true,
            recommends_domain: false,
        });
        assert_eq!(reg.count(), 6);
        assert!(reg.is_registered("https://custom.example/purpose"));
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = ProofPurposeRegistry::new();
        assert!(reg.unregister("authentication"));
        assert_eq!(reg.count(), 4);
        assert!(!reg.is_registered("authentication"));
    }

    #[test]
    fn test_registry_unregister_nonexistent() {
        let mut reg = ProofPurposeRegistry::new();
        assert!(!reg.unregister("nonexistent"));
        assert_eq!(reg.count(), 5);
    }

    // ── Utility method tests ────────────────────────────────────────────

    #[test]
    fn test_requires_challenge() {
        assert!(ProofPurposeValidator::requires_challenge(
            &ProofPurposeKind::Authentication
        ));
        assert!(!ProofPurposeValidator::requires_challenge(
            &ProofPurposeKind::AssertionMethod
        ));
        assert!(!ProofPurposeValidator::requires_challenge(
            &ProofPurposeKind::KeyAgreement
        ));
    }

    #[test]
    fn test_describe_purpose() {
        let desc = ProofPurposeValidator::describe_purpose(&ProofPurposeKind::Authentication);
        assert!(desc.contains("identity"));
        let desc2 = ProofPurposeValidator::describe_purpose(&ProofPurposeKind::AssertionMethod);
        assert!(desc2.contains("claims"));
    }

    #[test]
    fn test_relationship_name() {
        assert_eq!(
            ProofPurposeValidator::relationship_name(&ProofPurposeKind::Authentication),
            "authentication"
        );
        assert_eq!(
            ProofPurposeValidator::relationship_name(&ProofPurposeKind::CapabilityDelegation),
            "capabilityDelegation"
        );
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn test_error_display_unauthorised() {
        let err = PurposeError::Unauthorised {
            purpose: "authentication".to_string(),
            verification_method: "did:x#k".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("not authorised"));
    }

    #[test]
    fn test_error_display_expired() {
        let err = PurposeError::Expired {
            expires_ms: 100,
            current_ms: 200,
        };
        let msg = format!("{err}");
        assert!(msg.contains("expired"));
    }

    #[test]
    fn test_error_display_future() {
        let err = PurposeError::FutureProof {
            created_ms: 300,
            current_ms: 100,
        };
        let msg = format!("{err}");
        assert!(msg.contains("future"));
    }

    #[test]
    fn test_error_display_challenge_mismatch() {
        let err = PurposeError::ChallengeMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("mismatch"));
    }

    #[test]
    fn test_error_display_chain_error() {
        let err = PurposeError::ChainError {
            index: 2,
            reason: "bad".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("index 2"));
    }

    #[test]
    fn test_error_display_empty_chain() {
        let err = PurposeError::EmptyChain;
        let msg = format!("{err}");
        assert!(msg.contains("at least one"));
    }

    // ── No-expiration proof passes ──────────────────────────────────────

    #[test]
    fn test_validate_no_expiration() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        assert!(result.is_ok());
    }

    // ── Authentication without expected challenge in context ─────────────

    #[test]
    fn test_validate_authentication_no_expected_challenge() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext {
            challenge: None,
            domain: None,
            current_time_ms: 1_700_000_000_000,
        };
        let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        entry.challenge = Some("any-challenge".to_string());
        entry.domain = None;
        let result = validator.validate(&entry, &rels, &ctx);
        // When context has no expected challenge, any challenge is accepted
        assert!(result.is_ok());
    }

    // ── Validator with_max_clock_skew_ms builder ────────────────────────

    #[test]
    fn test_validator_default_construction() {
        let v = ProofPurposeValidator::default();
        // Should work the same as new()
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        assert!(v.validate(&entry, &rels, &ctx).is_ok());
    }

    #[test]
    fn test_validation_result_fields() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = ValidationContext::default();
        let mut entry = make_entry(
            ProofPurposeKind::CapabilityInvocation,
            "did:example:alice#key-4",
        );
        entry.created_ms = 0;
        entry.expires_ms = None;
        entry.challenge = None;
        entry.domain = None;
        let r = validator
            .validate(&entry, &rels, &ctx)
            .expect("should succeed");
        assert!(r.valid);
        assert_eq!(r.purpose, "capabilityInvocation");
        assert_eq!(r.verification_method, "did:example:alice#key-4");
        assert!(r.message.contains("validated successfully"));
    }

    #[test]
    fn test_chain_validation_result_fields() {
        let validator = ProofPurposeValidator::new();
        let rels = make_relationships();
        let ctx = make_context();
        let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
        let r = validator
            .validate_chain(&[entry], &rels, &ctx)
            .expect("should succeed");
        assert!(r.valid);
        assert_eq!(r.chain_length, 1);
        assert!(r.error.is_none());
        assert_eq!(r.entries.len(), 1);
    }
}
