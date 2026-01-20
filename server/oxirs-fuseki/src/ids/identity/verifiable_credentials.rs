//! W3C Verifiable Credentials Support
//!
//! Implementation of W3C Verifiable Credentials Data Model v2.0
//! <https://www.w3.org/TR/vc-data-model-2.0/>

use crate::ids::types::{IdsError, IdsResult, IdsUri};
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use ring::digest::{Context as DigestContext, SHA256};
use ring::rand::SystemRandom;
use ring::signature::{self, Ed25519KeyPair, KeyPair};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// W3C Verifiable Credential
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifiableCredential {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Credential ID
    pub id: IdsUri,

    /// Credential types
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,

    /// Issuer
    pub issuer: IdsUri,

    /// Issuance date
    pub issuance_date: DateTime<Utc>,

    /// Expiration date (optional)
    pub expiration_date: Option<DateTime<Utc>>,

    /// Credential subject
    pub credential_subject: CredentialSubject,

    /// Cryptographic proof
    pub proof: Option<Proof>,
}

impl VerifiableCredential {
    /// Check if credential is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expiration_date {
            Utc::now() > exp
        } else {
            false
        }
    }

    /// Check if credential has a valid proof
    pub fn has_proof(&self) -> bool {
        self.proof.is_some()
    }
}

/// Credential Subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// Subject ID
    pub id: IdsUri,

    /// Subject claims
    #[serde(flatten)]
    pub claims: HashMap<String, serde_json::Value>,
}

/// Cryptographic Proof
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proof {
    /// Proof type (e.g., "Ed25519Signature2020")
    #[serde(rename = "type")]
    pub proof_type: String,

    /// Creation timestamp
    pub created: DateTime<Utc>,

    /// Verification method (e.g., DID URL)
    pub verification_method: String,

    /// Proof purpose (e.g., "assertionMethod")
    pub proof_purpose: String,

    /// Proof value (base64 encoded signature)
    pub proof_value: String,
}

/// Proof purpose types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofPurpose {
    /// For making assertions
    AssertionMethod,
    /// For authentication
    Authentication,
    /// For key agreement
    KeyAgreement,
    /// For capability invocation
    CapabilityInvocation,
    /// For capability delegation
    CapabilityDelegation,
}

impl ProofPurpose {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ProofPurpose::AssertionMethod => "assertionMethod",
            ProofPurpose::Authentication => "authentication",
            ProofPurpose::KeyAgreement => "keyAgreement",
            ProofPurpose::CapabilityInvocation => "capabilityInvocation",
            ProofPurpose::CapabilityDelegation => "capabilityDelegation",
        }
    }
}

/// Builder for creating Verifiable Credentials
pub struct VerifiableCredentialBuilder {
    context: Vec<String>,
    credential_type: Vec<String>,
    issuer: Option<IdsUri>,
    subject_id: Option<IdsUri>,
    claims: HashMap<String, serde_json::Value>,
    expiration_days: Option<i64>,
}

impl Default for VerifiableCredentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifiableCredentialBuilder {
    /// Create a new builder with default W3C VC context
    pub fn new() -> Self {
        Self {
            context: vec![
                "https://www.w3.org/ns/credentials/v2".to_string(),
                "https://w3id.org/security/suites/ed25519-2020/v1".to_string(),
            ],
            credential_type: vec!["VerifiableCredential".to_string()],
            issuer: None,
            subject_id: None,
            claims: HashMap::new(),
            expiration_days: None,
        }
    }

    /// Add additional context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// Add credential type
    pub fn with_type(mut self, credential_type: impl Into<String>) -> Self {
        self.credential_type.push(credential_type.into());
        self
    }

    /// Set issuer
    pub fn issuer(mut self, issuer: IdsUri) -> Self {
        self.issuer = Some(issuer);
        self
    }

    /// Set subject ID
    pub fn subject(mut self, subject_id: IdsUri) -> Self {
        self.subject_id = Some(subject_id);
        self
    }

    /// Add a claim
    pub fn claim(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.claims.insert(key.into(), value.into());
        self
    }

    /// Set expiration in days from now
    pub fn expires_in_days(mut self, days: i64) -> Self {
        self.expiration_days = Some(days);
        self
    }

    /// Build the credential (without proof)
    pub fn build(self) -> IdsResult<VerifiableCredential> {
        let issuer = self
            .issuer
            .ok_or_else(|| IdsError::InternalError("Issuer is required".to_string()))?;

        let subject_id = self
            .subject_id
            .ok_or_else(|| IdsError::InternalError("Subject ID is required".to_string()))?;

        let now = Utc::now();
        let expiration_date = self.expiration_days.map(|days| now + Duration::days(days));

        let credential_id = IdsUri::new(format!("urn:uuid:{}", Uuid::new_v4())).map_err(|e| {
            IdsError::InternalError(format!("Failed to create credential ID: {}", e))
        })?;

        Ok(VerifiableCredential {
            context: self.context,
            id: credential_id,
            credential_type: self.credential_type,
            issuer,
            issuance_date: now,
            expiration_date,
            credential_subject: CredentialSubject {
                id: subject_id,
                claims: self.claims,
            },
            proof: None,
        })
    }
}

/// Credential Issuer for creating and signing credentials
pub struct CredentialIssuer {
    /// Ed25519 key pair
    key_pair: Ed25519KeyPair,
    /// Issuer DID/URI
    issuer_id: IdsUri,
    /// Verification method ID
    verification_method: String,
}

impl CredentialIssuer {
    /// Create a new credential issuer with generated keys
    pub fn new(issuer_id: IdsUri) -> IdsResult<Self> {
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| IdsError::InternalError(format!("Failed to generate key pair: {}", e)))?;

        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        let verification_method = format!("{}#key-1", issuer_id.as_str());

        Ok(Self {
            key_pair,
            issuer_id,
            verification_method,
        })
    }

    /// Create from existing PKCS#8 key
    pub fn from_pkcs8(issuer_id: IdsUri, pkcs8_bytes: &[u8]) -> IdsResult<Self> {
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes)
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        let verification_method = format!("{}#key-1", issuer_id.as_str());

        Ok(Self {
            key_pair,
            issuer_id,
            verification_method,
        })
    }

    /// Get issuer ID
    pub fn issuer_id(&self) -> &IdsUri {
        &self.issuer_id
    }

    /// Get public key bytes
    pub fn public_key(&self) -> &[u8] {
        self.key_pair.public_key().as_ref()
    }

    /// Get public key as base64
    pub fn public_key_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.public_key())
    }

    /// Compute hash of credential for signing (excluding proof)
    fn compute_credential_hash(credential: &VerifiableCredential) -> IdsResult<Vec<u8>> {
        // Create a copy without proof for hashing
        let mut cred_for_hash = credential.clone();
        cred_for_hash.proof = None;

        let json = serde_json::to_string(&cred_for_hash).map_err(|e| {
            IdsError::SerializationError(format!("Failed to serialize credential: {}", e))
        })?;

        let mut context = DigestContext::new(&SHA256);
        context.update(json.as_bytes());
        Ok(context.finish().as_ref().to_vec())
    }

    /// Issue a credential (creates proof and returns signed credential)
    pub fn issue(&self, mut credential: VerifiableCredential) -> IdsResult<VerifiableCredential> {
        let hash = Self::compute_credential_hash(&credential)?;
        let sig = self.key_pair.sign(&hash);

        let proof = Proof {
            proof_type: "Ed25519Signature2020".to_string(),
            created: Utc::now(),
            verification_method: self.verification_method.clone(),
            proof_purpose: ProofPurpose::AssertionMethod.as_str().to_string(),
            proof_value: base64::engine::general_purpose::STANDARD.encode(sig.as_ref()),
        };

        credential.proof = Some(proof);
        Ok(credential)
    }
}

/// Credential Verifier for validating credentials
pub struct CredentialVerifier;

impl CredentialVerifier {
    /// Verify a credential's proof
    pub fn verify(
        credential: &VerifiableCredential,
        public_key: &[u8],
    ) -> IdsResult<VerificationResult> {
        // Check if credential has proof
        let proof = credential.proof.as_ref().ok_or_else(|| {
            IdsError::TrustVerificationFailed("Credential has no proof".to_string())
        })?;

        // Check proof type
        if proof.proof_type != "Ed25519Signature2020" {
            return Ok(VerificationResult {
                valid: false,
                error: Some(format!("Unsupported proof type: {}", proof.proof_type)),
                checks: VerificationChecks::default(),
            });
        }

        // Decode signature
        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(&proof.proof_value)
            .map_err(|e| IdsError::InternalError(format!("Invalid proof encoding: {}", e)))?;

        // Compute hash
        let hash = CredentialIssuer::compute_credential_hash(credential)?;

        // Verify signature
        let public_key = signature::UnparsedPublicKey::new(&signature::ED25519, public_key);
        let signature_valid = public_key.verify(&hash, &sig_bytes).is_ok();

        // Check expiration
        let not_expired = !credential.is_expired();

        // Check issuance date
        let issuance_valid = credential.issuance_date <= Utc::now();

        let checks = VerificationChecks {
            signature_valid,
            not_expired,
            issuance_valid,
            proof_purpose_valid: proof.proof_purpose == ProofPurpose::AssertionMethod.as_str(),
        };

        let valid = checks.all_valid();

        Ok(VerificationResult {
            valid,
            error: if valid {
                None
            } else {
                Some("Verification failed".to_string())
            },
            checks,
        })
    }

    /// Verify credential and check specific claims
    pub fn verify_with_claims(
        credential: &VerifiableCredential,
        public_key: &[u8],
        expected_claims: &HashMap<String, serde_json::Value>,
    ) -> IdsResult<VerificationResult> {
        let mut result = Self::verify(credential, public_key)?;

        if result.valid {
            // Check that all expected claims are present and match
            for (key, expected_value) in expected_claims {
                if let Some(actual_value) = credential.credential_subject.claims.get(key) {
                    if actual_value != expected_value {
                        result.valid = false;
                        result.error =
                            Some(format!("Claim '{}' does not match expected value", key));
                        break;
                    }
                } else {
                    result.valid = false;
                    result.error = Some(format!("Missing required claim: {}", key));
                    break;
                }
            }
        }

        Ok(result)
    }
}

/// Result of credential verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Overall validity
    pub valid: bool,
    /// Error message if invalid
    pub error: Option<String>,
    /// Individual check results
    pub checks: VerificationChecks,
}

/// Individual verification checks
#[derive(Debug, Clone, Default)]
pub struct VerificationChecks {
    /// Signature is valid
    pub signature_valid: bool,
    /// Credential is not expired
    pub not_expired: bool,
    /// Issuance date is valid (not in the future)
    pub issuance_valid: bool,
    /// Proof purpose is valid
    pub proof_purpose_valid: bool,
}

impl VerificationChecks {
    /// Check if all verification checks passed
    pub fn all_valid(&self) -> bool {
        self.signature_valid && self.not_expired && self.issuance_valid && self.proof_purpose_valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_builder() {
        let issuer = IdsUri::new("https://issuer.example.org").expect("valid URI");
        let subject = IdsUri::new("https://subject.example.org").expect("valid URI");

        let credential = VerifiableCredentialBuilder::new()
            .with_type("IdsConnectorCredential")
            .issuer(issuer.clone())
            .subject(subject.clone())
            .claim(
                "connectorId",
                serde_json::json!("urn:ids:connector:example"),
            )
            .claim(
                "securityProfile",
                serde_json::json!("TRUST_SECURITY_PROFILE"),
            )
            .expires_in_days(365)
            .build()
            .expect("build credential");

        assert_eq!(credential.issuer, issuer);
        assert_eq!(credential.credential_subject.id, subject);
        assert!(credential
            .credential_type
            .contains(&"VerifiableCredential".to_string()));
        assert!(credential
            .credential_type
            .contains(&"IdsConnectorCredential".to_string()));
        assert!(credential.expiration_date.is_some());
        assert!(credential.proof.is_none()); // No proof until signed
    }

    #[test]
    fn test_credential_issuance_and_verification() {
        let issuer_id = IdsUri::new("https://issuer.example.org").expect("valid URI");
        let subject_id = IdsUri::new("https://subject.example.org").expect("valid URI");

        // Create issuer
        let issuer = CredentialIssuer::new(issuer_id.clone()).expect("create issuer");

        // Build credential
        let credential = VerifiableCredentialBuilder::new()
            .with_type("IdsConnectorCredential")
            .issuer(issuer_id)
            .subject(subject_id)
            .claim(
                "connectorId",
                serde_json::json!("urn:ids:connector:example"),
            )
            .expires_in_days(365)
            .build()
            .expect("build credential");

        // Issue (sign) credential
        let signed_credential = issuer.issue(credential).expect("issue credential");

        assert!(signed_credential.proof.is_some());
        assert_eq!(
            signed_credential
                .proof
                .as_ref()
                .map(|p| p.proof_type.as_str()),
            Some("Ed25519Signature2020")
        );

        // Verify credential
        let result = CredentialVerifier::verify(&signed_credential, issuer.public_key())
            .expect("verify credential");

        assert!(
            result.valid,
            "Credential should be valid: {:?}",
            result.error
        );
        assert!(result.checks.signature_valid);
        assert!(result.checks.not_expired);
        assert!(result.checks.issuance_valid);
    }

    #[test]
    fn test_credential_tampering_detection() {
        let issuer_id = IdsUri::new("https://issuer.example.org").expect("valid URI");
        let subject_id = IdsUri::new("https://subject.example.org").expect("valid URI");

        let issuer = CredentialIssuer::new(issuer_id.clone()).expect("create issuer");

        let credential = VerifiableCredentialBuilder::new()
            .issuer(issuer_id)
            .subject(subject_id)
            .claim("role", serde_json::json!("user"))
            .build()
            .expect("build credential");

        let mut signed_credential = issuer.issue(credential).expect("issue credential");

        // Tamper with the credential
        signed_credential
            .credential_subject
            .claims
            .insert("role".to_string(), serde_json::json!("admin"));

        // Verification should fail
        let result = CredentialVerifier::verify(&signed_credential, issuer.public_key())
            .expect("verify credential");

        assert!(
            !result.valid,
            "Tampered credential should fail verification"
        );
        assert!(!result.checks.signature_valid);
    }

    #[test]
    fn test_expired_credential() {
        let issuer_id = IdsUri::new("https://issuer.example.org").expect("valid URI");
        let subject_id = IdsUri::new("https://subject.example.org").expect("valid URI");

        // Create credential that's already expired
        let mut credential = VerifiableCredentialBuilder::new()
            .issuer(issuer_id.clone())
            .subject(subject_id)
            .build()
            .expect("build credential");

        // Set expiration to the past
        credential.expiration_date = Some(Utc::now() - Duration::days(1));

        let issuer = CredentialIssuer::new(issuer_id).expect("create issuer");
        let signed_credential = issuer.issue(credential).expect("issue credential");

        assert!(signed_credential.is_expired());

        let result = CredentialVerifier::verify(&signed_credential, issuer.public_key())
            .expect("verify credential");

        assert!(!result.valid);
        assert!(!result.checks.not_expired);
    }

    #[test]
    fn test_verify_with_claims() {
        let issuer_id = IdsUri::new("https://issuer.example.org").expect("valid URI");
        let subject_id = IdsUri::new("https://subject.example.org").expect("valid URI");

        let issuer = CredentialIssuer::new(issuer_id.clone()).expect("create issuer");

        let credential = VerifiableCredentialBuilder::new()
            .issuer(issuer_id)
            .subject(subject_id)
            .claim("role", serde_json::json!("connector"))
            .claim("securityLevel", serde_json::json!(2))
            .build()
            .expect("build credential");

        let signed_credential = issuer.issue(credential).expect("issue credential");

        // Verify with correct claims
        let mut expected_claims = HashMap::new();
        expected_claims.insert("role".to_string(), serde_json::json!("connector"));

        let result = CredentialVerifier::verify_with_claims(
            &signed_credential,
            issuer.public_key(),
            &expected_claims,
        )
        .expect("verify with claims");

        assert!(result.valid);

        // Verify with incorrect claims
        let mut wrong_claims = HashMap::new();
        wrong_claims.insert("role".to_string(), serde_json::json!("admin"));

        let result = CredentialVerifier::verify_with_claims(
            &signed_credential,
            issuer.public_key(),
            &wrong_claims,
        )
        .expect("verify with claims");

        assert!(!result.valid);
    }
}
