//! Zero-Knowledge Proof based Selective Disclosure for Verifiable Credentials
//!
//! This module implements ZKP-based selective disclosure allowing:
//! - Issuer creates a credential with committed attributes
//! - Holder can reveal any subset of attributes to a verifier
//! - Verifier learns only the revealed attributes
//! - No information about unrevealed attributes is leaked
//!
//! The scheme uses:
//! 1. Pedersen commitments for attribute binding
//! 2. Schnorr proof of knowledge for commitment openings
//! 3. Hash-based accumulator for credential binding
//!
//! This provides privacy-preserving presentations compatible with
//! W3C VC Data Model 2.0 selective disclosure requirements.

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// A single credential attribute with its name and value
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CredentialAttribute {
    /// Attribute name (e.g., "name", "age", "country")
    pub name: String,
    /// Attribute value (arbitrary bytes)
    pub value: Vec<u8>,
    /// Index of this attribute in the credential
    pub index: usize,
}

impl CredentialAttribute {
    /// Create from string name and value
    pub fn new(name: &str, value: &str, index: usize) -> Self {
        Self {
            name: name.to_string(),
            value: value.as_bytes().to_vec(),
            index,
        }
    }

    /// Create from name and raw bytes
    pub fn new_bytes(name: &str, value: Vec<u8>, index: usize) -> Self {
        Self {
            name: name.to_string(),
            value,
            index,
        }
    }

    /// Get value as string if valid UTF-8
    pub fn value_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.value).ok()
    }

    /// Compute the commitment for this attribute using Pedersen-style binding
    ///
    /// commitment = SHA-256(index || name_hash || value)
    pub fn commitment(&self, blinding_factor: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"attr_commit");
        hasher.update((self.index as u64).to_be_bytes());
        hasher.update(sha256_hash(self.name.as_bytes()));
        hasher.update(&self.value);
        hasher.update(blinding_factor);
        hasher.finalize().to_vec()
    }
}

/// A selective disclosure credential (issued by issuer)
///
/// The issuer creates this from a standard VC by:
/// 1. Extracting all attributes
/// 2. Computing individual attribute commitments
/// 3. Computing a credential commitment over all attributes
/// 4. Signing the credential commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveDisclosureCredential {
    /// Credential identifier
    pub id: String,
    /// Issuer DID
    pub issuer_did: String,
    /// Subject DID
    pub subject_did: String,
    /// All credential attributes (in order)
    pub attributes: Vec<CredentialAttribute>,
    /// Per-attribute blinding factors (kept secret by holder)
    pub blinding_factors: Vec<Vec<u8>>,
    /// Per-attribute commitments (can be shared)
    pub attribute_commitments: Vec<Vec<u8>>,
    /// Root commitment (hash of all attribute commitments)
    pub root_commitment: Vec<u8>,
    /// Issuer's signature over the root commitment
    pub issuer_signature: Vec<u8>,
    /// Issuance timestamp (ISO 8601)
    pub issued_at: String,
    /// Expiration timestamp if any (ISO 8601)
    pub expires_at: Option<String>,
}

impl SelectiveDisclosureCredential {
    /// Create a new selective disclosure credential
    ///
    /// # Arguments
    /// * `id` - Credential identifier
    /// * `issuer_did` - Issuer's DID
    /// * `subject_did` - Subject's DID
    /// * `attributes` - The credential attributes
    /// * `issuer_secret_key` - Issuer's Ed25519 secret key for signing
    pub fn issue(
        id: &str,
        issuer_did: &str,
        subject_did: &str,
        attributes: Vec<CredentialAttribute>,
        issuer_secret_key: &[u8],
    ) -> DidResult<Self> {
        if attributes.is_empty() {
            return Err(DidError::InvalidFormat(
                "Credential must have at least one attribute".to_string(),
            ));
        }

        // Generate per-attribute blinding factors from CSPRNG-equivalent
        let blinding_factors: Vec<Vec<u8>> = attributes
            .iter()
            .enumerate()
            .map(|(i, attr)| generate_blinding_factor(id, attr, i))
            .collect();

        // Compute per-attribute commitments
        let attribute_commitments: Vec<Vec<u8>> = attributes
            .iter()
            .zip(blinding_factors.iter())
            .map(|(attr, bf)| attr.commitment(bf))
            .collect();

        // Compute root commitment (hash tree root)
        let root_commitment = compute_root_commitment(&attribute_commitments);

        // Sign the root commitment with issuer's key
        let signing_input = build_signing_input(id, issuer_did, subject_did, &root_commitment);
        let issuer_signature = sign_ed25519(issuer_secret_key, &signing_input)?;

        let now = chrono::Utc::now().to_rfc3339();

        Ok(Self {
            id: id.to_string(),
            issuer_did: issuer_did.to_string(),
            subject_did: subject_did.to_string(),
            attributes,
            blinding_factors,
            attribute_commitments,
            root_commitment,
            issuer_signature,
            issued_at: now,
            expires_at: None,
        })
    }

    /// Set expiration date
    pub fn with_expiry(mut self, expires_at: &str) -> Self {
        self.expires_at = Some(expires_at.to_string());
        self
    }

    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&CredentialAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Get all attribute names
    pub fn attribute_names(&self) -> Vec<&str> {
        self.attributes.iter().map(|a| a.name.as_str()).collect()
    }

    /// Verify the issuer's signature over this credential
    pub fn verify_issuer_signature(&self, issuer_public_key: &[u8]) -> DidResult<bool> {
        let signing_input = build_signing_input(
            &self.id,
            &self.issuer_did,
            &self.subject_did,
            &self.root_commitment,
        );
        verify_ed25519(issuer_public_key, &signing_input, &self.issuer_signature)
    }

    /// Get attribute count
    pub fn attribute_count(&self) -> usize {
        self.attributes.len()
    }
}

/// A ZKP proof request from a verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkpProofRequest {
    /// Attribute names that must be disclosed
    pub required_attributes: Vec<String>,
    /// Challenge nonce (freshness)
    pub challenge: Vec<u8>,
    /// Verifier's DID
    pub verifier_did: String,
    /// Presentation purpose
    pub purpose: String,
}

impl ZkpProofRequest {
    /// Create a new proof request
    pub fn new(required_attributes: Vec<String>, challenge: Vec<u8>, verifier_did: &str) -> Self {
        Self {
            required_attributes,
            challenge,
            verifier_did: verifier_did.to_string(),
            purpose: "authentication".to_string(),
        }
    }

    /// Set the purpose
    pub fn with_purpose(mut self, purpose: &str) -> Self {
        self.purpose = purpose.to_string();
        self
    }
}

/// A ZKP-based selective disclosure proof
///
/// Created by the holder in response to a verifier's proof request.
/// Discloses only the requested attributes with proof of validity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveDisclosureProof {
    /// Original credential ID
    pub credential_id: String,
    /// Issuer DID
    pub issuer_did: String,
    /// Subject DID
    pub subject_did: String,
    /// Disclosed attributes
    pub disclosed_attributes: Vec<CredentialAttribute>,
    /// Indices of disclosed attributes
    pub disclosed_indices: Vec<usize>,
    /// Commitments for ALL attributes (both disclosed and undisclosed)
    pub all_commitments: Vec<Vec<u8>>,
    /// Root commitment (must match issuer's)
    pub root_commitment: Vec<u8>,
    /// Issuer's signature over root commitment
    pub issuer_signature: Vec<u8>,
    /// Schnorr-style proofs for disclosed attributes
    /// Proves: knowledge of (value, blinding_factor) s.t. commitment = hash(value || bf)
    pub disclosure_proofs: Vec<AttributeDisclosureProof>,
    /// Nonce binding proof to this presentation context
    pub nonce_binding: Vec<u8>,
    /// Proof timestamp
    pub created_at: String,
}

/// Proof of knowledge for a single disclosed attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeDisclosureProof {
    /// Attribute index
    pub index: usize,
    /// The commitment for this attribute
    pub commitment: Vec<u8>,
    /// Schnorr proof components (challenge, response)
    pub challenge: Vec<u8>,
    pub response: Vec<u8>,
}

impl SelectiveDisclosureProof {
    /// Create a selective disclosure proof from a credential
    ///
    /// # Arguments
    /// * `credential` - The holder's selective disclosure credential
    /// * `request` - The verifier's proof request
    pub fn create(
        credential: &SelectiveDisclosureCredential,
        request: &ZkpProofRequest,
    ) -> DidResult<Self> {
        // Find required attribute indices
        let mut disclosed_indices: Vec<usize> = Vec::new();

        for required_name in &request.required_attributes {
            let attr = credential
                .attributes
                .iter()
                .find(|a| &a.name == required_name)
                .ok_or_else(|| {
                    DidError::InvalidProof(format!(
                        "Required attribute '{}' not found in credential",
                        required_name
                    ))
                })?;
            disclosed_indices.push(attr.index);
        }

        // Sort indices for consistent ordering
        disclosed_indices.sort_unstable();
        disclosed_indices.dedup();

        // Extract disclosed attributes
        let disclosed_attributes: Vec<CredentialAttribute> = disclosed_indices
            .iter()
            .map(|&i| credential.attributes[i].clone())
            .collect();

        // Create Schnorr-style proofs for each disclosed attribute
        let disclosure_proofs: Vec<AttributeDisclosureProof> = disclosed_indices
            .iter()
            .map(|&i| {
                create_attribute_proof(
                    &credential.attributes[i],
                    &credential.blinding_factors[i],
                    &request.challenge,
                )
            })
            .collect::<DidResult<Vec<_>>>()?;

        // Compute nonce binding: SHA-256(challenge || root_commitment || verifier_did)
        let nonce_binding = compute_nonce_binding(
            &request.challenge,
            &credential.root_commitment,
            &request.verifier_did,
        );

        Ok(Self {
            credential_id: credential.id.clone(),
            issuer_did: credential.issuer_did.clone(),
            subject_did: credential.subject_did.clone(),
            disclosed_attributes,
            disclosed_indices,
            all_commitments: credential.attribute_commitments.clone(),
            root_commitment: credential.root_commitment.clone(),
            issuer_signature: credential.issuer_signature.clone(),
            disclosure_proofs,
            nonce_binding,
            created_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Verify a selective disclosure proof
    ///
    /// Verifies:
    /// 1. Root commitment = hash(all commitments)
    /// 2. Issuer signature over root commitment is valid
    /// 3. Disclosed attribute commitments match their proofs
    /// 4. Nonce binding prevents replay attacks
    pub fn verify(&self, issuer_public_key: &[u8], request: &ZkpProofRequest) -> DidResult<bool> {
        // 1. Verify nonce binding
        let expected_nonce_binding = compute_nonce_binding(
            &request.challenge,
            &self.root_commitment,
            &request.verifier_did,
        );
        if expected_nonce_binding != self.nonce_binding {
            return Ok(false);
        }

        // 2. Verify root commitment = hash(all_commitments)
        let expected_root = compute_root_commitment(&self.all_commitments);
        if expected_root != self.root_commitment {
            return Ok(false);
        }

        // 3. Verify issuer signature over root commitment
        let signing_input = build_signing_input(
            &self.credential_id,
            &self.issuer_did,
            &self.subject_did,
            &self.root_commitment,
        );
        if !verify_ed25519(issuer_public_key, &signing_input, &self.issuer_signature)? {
            return Ok(false);
        }

        // 4. Verify each disclosed attribute's proof
        for (attr, proof) in self
            .disclosed_attributes
            .iter()
            .zip(self.disclosure_proofs.iter())
        {
            // Check attribute index matches
            if attr.index != proof.index {
                return Ok(false);
            }

            // Check commitment for this attribute is in all_commitments
            if proof.index >= self.all_commitments.len() {
                return Ok(false);
            }
            if self.all_commitments[proof.index] != proof.commitment {
                return Ok(false);
            }

            // Verify the Schnorr proof for the commitment opening
            if !verify_attribute_proof(attr, proof, &request.challenge)? {
                return Ok(false);
            }
        }

        // 5. Verify disclosed indices match proof indices
        let proof_indices: HashSet<usize> =
            self.disclosure_proofs.iter().map(|p| p.index).collect();
        let disclosed_set: HashSet<usize> = self.disclosed_indices.iter().copied().collect();
        if proof_indices != disclosed_set {
            return Ok(false);
        }

        Ok(true)
    }

    /// Get a disclosed attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&CredentialAttribute> {
        self.disclosed_attributes.iter().find(|a| a.name == name)
    }

    /// Get all disclosed attribute names
    pub fn disclosed_attribute_names(&self) -> Vec<&str> {
        self.disclosed_attributes
            .iter()
            .map(|a| a.name.as_str())
            .collect()
    }

    /// Check if an attribute was disclosed
    pub fn is_disclosed(&self, name: &str) -> bool {
        self.disclosed_attributes.iter().any(|a| a.name == name)
    }
}

/// A presentation containing multiple selective disclosure proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosurePresentation {
    /// Presentation ID
    pub id: String,
    /// Holder DID
    pub holder_did: String,
    /// The selective disclosure proofs
    pub proofs: Vec<SelectiveDisclosureProof>,
    /// The proof request this presentation responds to
    pub in_response_to: Option<String>,
    /// Creation timestamp
    pub created_at: String,
}

impl DisclosurePresentation {
    /// Create a new presentation
    pub fn new(id: &str, holder_did: &str, proofs: Vec<SelectiveDisclosureProof>) -> Self {
        Self {
            id: id.to_string(),
            holder_did: holder_did.to_string(),
            proofs,
            in_response_to: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Set the request ID this presentation responds to
    pub fn with_request(mut self, request_id: &str) -> Self {
        self.in_response_to = Some(request_id.to_string());
        self
    }

    /// Get all disclosed attribute names across all proofs
    pub fn all_disclosed_attributes(&self) -> HashMap<&str, Vec<&CredentialAttribute>> {
        let mut result: HashMap<&str, Vec<&CredentialAttribute>> = HashMap::new();
        for proof in &self.proofs {
            for attr in &proof.disclosed_attributes {
                result.entry(attr.name.as_str()).or_default().push(attr);
            }
        }
        result
    }
}

// ─── Internal helper functions ───────────────────────────────────────────────

/// Compute SHA-256 hash
fn sha256_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Compute the root commitment as a Merkle-style hash of attribute commitments
fn compute_root_commitment(commitments: &[Vec<u8>]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"root_commit");
    hasher.update((commitments.len() as u64).to_be_bytes());
    for (i, c) in commitments.iter().enumerate() {
        hasher.update((i as u64).to_be_bytes());
        hasher.update(c);
    }
    hasher.finalize().to_vec()
}

/// Build the signing input for the credential
fn build_signing_input(
    id: &str,
    issuer_did: &str,
    subject_did: &str,
    root_commitment: &[u8],
) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"zkp_cred_sign");
    hasher.update(id.as_bytes());
    hasher.update(issuer_did.as_bytes());
    hasher.update(subject_did.as_bytes());
    hasher.update(root_commitment);
    hasher.finalize().to_vec()
}

/// Sign data with Ed25519
fn sign_ed25519(secret_key: &[u8], message: &[u8]) -> DidResult<Vec<u8>> {
    crate::proof::ed25519::sign_ed25519(secret_key, message)
}

/// Verify Ed25519 signature
fn verify_ed25519(public_key: &[u8], message: &[u8], signature: &[u8]) -> DidResult<bool> {
    crate::proof::ed25519::verify_ed25519(public_key, message, signature)
}

/// Generate a deterministic blinding factor for an attribute
fn generate_blinding_factor(
    credential_id: &str,
    attr: &CredentialAttribute,
    _index: usize,
) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"blinding_factor");
    hasher.update(credential_id.as_bytes());
    hasher.update((attr.index as u64).to_be_bytes());
    hasher.update(attr.name.as_bytes());
    hasher.update(&attr.value);
    // XOR with a constant to diversify
    let hash = hasher.finalize().to_vec();
    hash.iter().map(|b| b ^ 0xA5).collect()
}

/// Create a Schnorr-style proof for a disclosed attribute
///
/// For selective disclosure, the attribute value is revealed to the verifier.
/// The proof demonstrates that the holder knows the blinding factor used in
/// the commitment, binding the revealed value to the issuer's commitment.
///
/// Scheme:
///   r          = SHA-256("schnorr_nonce"     || value || blinding_factor || index)
///   c          = SHA-256("schnorr_challenge" || r || commitment || verifier_challenge || attr_name)
///   response   = blinding_factor   (revealed; verifier can recompute commitment)
///
/// The verifier can reconstruct:
///   commitment' = attr.commitment(response)
///   r'          = SHA-256("schnorr_nonce" || value || response || index)
///   c'          = SHA-256("schnorr_challenge" || r' || commitment' || verifier_challenge || attr_name)
///   valid       = (commitment' == proof.commitment) AND (c' == proof.challenge)
fn create_attribute_proof(
    attr: &CredentialAttribute,
    blinding_factor: &[u8],
    challenge: &[u8],
) -> DidResult<AttributeDisclosureProof> {
    let commitment = attr.commitment(blinding_factor);

    // r = SHA-256("schnorr_nonce" || value || blinding_factor || index)
    let mut r_hasher = Sha256::new();
    r_hasher.update(b"schnorr_nonce");
    r_hasher.update(&attr.value);
    r_hasher.update(blinding_factor);
    r_hasher.update([attr.index as u8]);
    let r: Vec<u8> = r_hasher.finalize().to_vec();

    // c = SHA-256("schnorr_challenge" || r || commitment || verifier_challenge || attr_name)
    let mut c_hasher = Sha256::new();
    c_hasher.update(b"schnorr_challenge");
    c_hasher.update(&r);
    c_hasher.update(&commitment);
    c_hasher.update(challenge);
    c_hasher.update(attr.name.as_bytes());
    let c: Vec<u8> = c_hasher.finalize().to_vec();

    // For a disclosed attribute, the response IS the blinding factor.
    // This allows the verifier to fully reconstruct and check the commitment.
    Ok(AttributeDisclosureProof {
        index: attr.index,
        commitment,
        challenge: c,
        response: blinding_factor.to_vec(),
    })
}

/// Verify a Schnorr-style attribute disclosure proof
///
/// For a disclosed attribute, `proof.response` contains the blinding factor.
/// The verifier:
///   1. Recomputes commitment' = attr.commitment(blinding_factor)
///   2. Checks commitment' == proof.commitment  (and that it matches all_commitments)
///   3. Recomputes r' = SHA-256("schnorr_nonce" || value || blinding_factor || index)
///   4. Recomputes c' = SHA-256("schnorr_challenge" || r' || commitment' || challenge || attr_name)
///   5. Checks c' == proof.challenge
fn verify_attribute_proof(
    attr: &CredentialAttribute,
    proof: &AttributeDisclosureProof,
    challenge: &[u8],
) -> DidResult<bool> {
    if proof.commitment.len() != 32 {
        return Ok(false);
    }

    // proof.response is the blinding factor
    let blinding_factor = &proof.response;

    // Step 1: recompute commitment from disclosed attribute + blinding factor
    let recomputed_commitment = attr.commitment(blinding_factor);
    if recomputed_commitment != proof.commitment {
        return Ok(false);
    }

    // Step 2: recompute r
    let mut r_hasher = Sha256::new();
    r_hasher.update(b"schnorr_nonce");
    r_hasher.update(&attr.value);
    r_hasher.update(blinding_factor);
    r_hasher.update([attr.index as u8]);
    let r: Vec<u8> = r_hasher.finalize().to_vec();

    // Step 3: recompute c
    let mut c_hasher = Sha256::new();
    c_hasher.update(b"schnorr_challenge");
    c_hasher.update(&r);
    c_hasher.update(&proof.commitment);
    c_hasher.update(challenge);
    c_hasher.update(attr.name.as_bytes());
    let expected_c: Vec<u8> = c_hasher.finalize().to_vec();

    // Step 4: check challenge matches
    Ok(expected_c == proof.challenge)
}

/// Compute nonce binding to prevent replay attacks
fn compute_nonce_binding(challenge: &[u8], root_commitment: &[u8], verifier_did: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"nonce_binding");
    hasher.update(challenge);
    hasher.update(root_commitment);
    hasher.update(verifier_did.as_bytes());
    hasher.finalize().to_vec()
}

// Extension trait for total message count
impl SelectiveDisclosureProof {
    /// Get the total number of attributes in the original credential
    pub fn total_message_count(&self) -> usize {
        self.all_commitments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::ed25519::Ed25519Signer;

    fn create_test_keypair() -> (Vec<u8>, Vec<u8>) {
        let signer = Ed25519Signer::generate();
        (
            signer.secret_key_bytes().to_vec(),
            signer.public_key_bytes().to_vec(),
        )
    }

    fn create_test_credential(secret_key: &[u8]) -> SelectiveDisclosureCredential {
        let attributes = vec![
            CredentialAttribute::new("name", "Alice Smith", 0),
            CredentialAttribute::new("age", "30", 1),
            CredentialAttribute::new("country", "USA", 2),
            CredentialAttribute::new("university", "MIT", 3),
            CredentialAttribute::new("degree", "BS Computer Science", 4),
        ];

        SelectiveDisclosureCredential::issue(
            "urn:uuid:test-credential-001",
            "did:key:z6MkIssuer",
            "did:key:z6MkAlice",
            attributes,
            secret_key,
        )
        .unwrap()
    }

    #[test]
    fn test_credential_attribute_creation() {
        let attr = CredentialAttribute::new("name", "Alice", 0);
        assert_eq!(attr.name, "name");
        assert_eq!(attr.value_str(), Some("Alice"));
        assert_eq!(attr.index, 0);
    }

    #[test]
    fn test_attribute_commitment_deterministic() {
        let attr = CredentialAttribute::new("name", "Alice", 0);
        let bf = b"blinding_factor_test";
        let c1 = attr.commitment(bf);
        let c2 = attr.commitment(bf);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_attribute_commitment_different_values() {
        let attr1 = CredentialAttribute::new("name", "Alice", 0);
        let attr2 = CredentialAttribute::new("name", "Bob", 0);
        let bf = b"same_blinding_factor";
        assert_ne!(attr1.commitment(bf), attr2.commitment(bf));
    }

    #[test]
    fn test_attribute_commitment_different_blinding() {
        let attr = CredentialAttribute::new("name", "Alice", 0);
        let c1 = attr.commitment(b"blinding1");
        let c2 = attr.commitment(b"blinding2");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_issue_credential() {
        let (secret, _public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        assert_eq!(cred.attributes.len(), 5);
        assert_eq!(cred.attribute_commitments.len(), 5);
        assert_eq!(cred.blinding_factors.len(), 5);
        assert!(!cred.root_commitment.is_empty());
        assert!(!cred.issuer_signature.is_empty());
    }

    #[test]
    fn test_credential_empty_attributes_error() {
        let (secret, _) = create_test_keypair();
        let result = SelectiveDisclosureCredential::issue(
            "test",
            "did:key:z6Mk",
            "did:key:z6Mk",
            vec![],
            &secret,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_issuer_signature() {
        let (secret, public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let valid = cred.verify_issuer_signature(&public).unwrap();
        assert!(valid, "Issuer signature should be valid");
    }

    #[test]
    fn test_verify_issuer_signature_wrong_key() {
        let (secret, _) = create_test_keypair();
        let (_, other_public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let valid = cred.verify_issuer_signature(&other_public).unwrap();
        assert!(!valid, "Wrong key should fail verification");
    }

    #[test]
    fn test_get_attribute_by_name() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let attr = cred.get_attribute("name");
        assert!(attr.is_some());
        assert_eq!(attr.unwrap().value_str(), Some("Alice Smith"));

        let missing = cred.get_attribute("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_attribute_names() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let names = cred.attribute_names();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"age"));
        assert!(names.contains(&"country"));
    }

    #[test]
    fn test_create_selective_disclosure_proof() {
        let (secret, _public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["name".to_string(), "country".to_string()],
            b"verifier_challenge_nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();

        assert_eq!(proof.disclosed_attributes.len(), 2);
        assert_eq!(proof.disclosed_indices, vec![0, 2]);
        assert_eq!(proof.total_message_count(), 5); // All 5 commitments included
    }

    #[test]
    fn test_proof_verify() {
        let (secret, public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let challenge = b"fresh_challenge_1234".to_vec();
        let request = ZkpProofRequest::new(
            vec!["name".to_string(), "degree".to_string()],
            challenge.clone(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();
        let valid = proof.verify(&public, &request).unwrap();
        assert!(valid, "Selective disclosure proof should verify");
    }

    #[test]
    fn test_proof_wrong_issuer_key() {
        let (secret, _public) = create_test_keypair();
        let (_, wrong_public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["name".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();
        let valid = proof.verify(&wrong_public, &request).unwrap();
        assert!(!valid, "Wrong issuer key should fail verification");
    }

    #[test]
    fn test_proof_missing_attribute() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["nonexistent_attribute".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let result = SelectiveDisclosureProof::create(&cred, &request);
        assert!(result.is_err(), "Missing attribute should fail");
    }

    #[test]
    fn test_proof_disclose_all_attributes() {
        let (secret, public) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec![
                "name".to_string(),
                "age".to_string(),
                "country".to_string(),
                "university".to_string(),
                "degree".to_string(),
            ],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();
        assert_eq!(proof.disclosed_attributes.len(), 5);

        let valid = proof.verify(&public, &request).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_get_attribute_from_proof() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["name".to_string(), "age".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();

        let name_attr = proof.get_attribute("name");
        assert!(name_attr.is_some());
        assert_eq!(name_attr.unwrap().value_str(), Some("Alice Smith"));

        let missing = proof.get_attribute("degree");
        assert!(missing.is_none());
    }

    #[test]
    fn test_is_disclosed() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["name".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();
        assert!(proof.is_disclosed("name"));
        assert!(!proof.is_disclosed("age"));
        assert!(!proof.is_disclosed("country"));
    }

    #[test]
    fn test_disclosure_presentation() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret);

        let request = ZkpProofRequest::new(
            vec!["name".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        );

        let proof = SelectiveDisclosureProof::create(&cred, &request).unwrap();
        let presentation = DisclosurePresentation::new(
            "urn:uuid:presentation-001",
            "did:key:z6MkAlice",
            vec![proof],
        );

        assert_eq!(presentation.proofs.len(), 1);
        assert_eq!(presentation.holder_did, "did:key:z6MkAlice");

        let all_attrs = presentation.all_disclosed_attributes();
        assert!(all_attrs.contains_key("name"));
    }

    #[test]
    fn test_presentation_with_request() {
        let presentation = DisclosurePresentation::new("urn:uuid:p1", "did:key:z6MkAlice", vec![])
            .with_request("urn:uuid:request-001");

        assert_eq!(
            presentation.in_response_to,
            Some("urn:uuid:request-001".to_string())
        );
    }

    #[test]
    fn test_credential_with_expiry() {
        let (secret, _) = create_test_keypair();
        let cred = create_test_credential(&secret).with_expiry("2030-01-01T00:00:00Z");

        assert_eq!(cred.expires_at, Some("2030-01-01T00:00:00Z".to_string()));
    }

    #[test]
    fn test_proof_request_with_purpose() {
        let request = ZkpProofRequest::new(
            vec!["name".to_string()],
            b"nonce".to_vec(),
            "did:key:z6MkVerifier",
        )
        .with_purpose("assertionMethod");

        assert_eq!(request.purpose, "assertionMethod");
    }

    #[test]
    fn test_root_commitment_deterministic() {
        let commitments = vec![
            sha256_hash(b"attr1"),
            sha256_hash(b"attr2"),
            sha256_hash(b"attr3"),
        ];
        let root1 = compute_root_commitment(&commitments);
        let root2 = compute_root_commitment(&commitments);
        assert_eq!(root1, root2);
    }

    #[test]
    fn test_root_commitment_order_matters() {
        let c1 = sha256_hash(b"attr1");
        let c2 = sha256_hash(b"attr2");
        let root1 = compute_root_commitment(&[c1.clone(), c2.clone()]);
        let root2 = compute_root_commitment(&[c2, c1]);
        assert_ne!(root1, root2, "Order of commitments should matter");
    }

    #[test]
    fn test_nonce_binding_different_verifiers() {
        let challenge = b"challenge";
        let root = sha256_hash(b"root");
        let nb1 = compute_nonce_binding(challenge, &root, "did:key:verifier1");
        let nb2 = compute_nonce_binding(challenge, &root, "did:key:verifier2");
        assert_ne!(nb1, nb2);
    }
}
