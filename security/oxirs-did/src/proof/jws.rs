//! JSON Web Signature (JWS) implementation
//!
//! RFC 7515 compliant JWS with support for:
//! - EdDSA (Ed25519) - fully implemented
//! - ES256K (secp256k1) - signature format defined, key ops stubbed
//! - RS256 (RSA) - signature format defined, key ops stubbed
//!
//! Supports JsonWebSignature2020 proof type per:
//! <https://w3c-ccg.github.io/lds-jws2020/>

use crate::did::DidDocument;
use crate::proof::ed25519::{Ed25519Signer, Ed25519Verifier};
use crate::proof::ProofPurpose;
use crate::{DidError, DidResult, VerificationMethod};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// JWS Algorithm identifiers per RFC 7518
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JwsAlgorithm {
    /// Edwards-curve Digital Signature Algorithm with Ed25519
    #[serde(rename = "EdDSA")]
    EdDsa,
    /// ECDSA using P-256 and SHA-256
    #[serde(rename = "ES256")]
    Es256,
    /// ECDSA using secp256k1 and SHA-256
    #[serde(rename = "ES256K")]
    Es256K,
    /// RSASSA-PKCS1-v1_5 using SHA-256
    #[serde(rename = "RS256")]
    Rs256,
}

impl JwsAlgorithm {
    /// Get the string identifier
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EdDsa => "EdDSA",
            Self::Es256 => "ES256",
            Self::Es256K => "ES256K",
            Self::Rs256 => "RS256",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> DidResult<Self> {
        match s {
            "EdDSA" => Ok(Self::EdDsa),
            "ES256" => Ok(Self::Es256),
            "ES256K" => Ok(Self::Es256K),
            "RS256" => Ok(Self::Rs256),
            other => Err(DidError::InvalidKey(format!(
                "Unknown JWS algorithm: {}",
                other
            ))),
        }
    }

    /// Get expected signature length in bytes
    pub fn signature_length(&self) -> Option<usize> {
        match self {
            Self::EdDsa => Some(64),  // Ed25519: 64 bytes
            Self::Es256 => Some(64),  // P-256: r||s each 32 bytes
            Self::Es256K => Some(64), // secp256k1: r||s each 32 bytes
            Self::Rs256 => None,      // RSA: key-size dependent
        }
    }
}

/// JWS JOSE Header (RFC 7515 Section 4)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwsHeader {
    /// Algorithm
    pub alg: JwsAlgorithm,
    /// Key ID (verification method URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
    /// Base64url-encode payload flag (RFC 7797)
    /// When false, payload is not base64url-encoded (detached payload)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64: Option<bool>,
    /// Critical header parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crit: Option<Vec<String>>,
    /// Content type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cty: Option<String>,
}

impl JwsHeader {
    /// Create a standard header for EdDSA
    pub fn ed_dsa(kid: Option<&str>) -> Self {
        Self {
            alg: JwsAlgorithm::EdDsa,
            kid: kid.map(String::from),
            b64: None,
            crit: None,
            cty: None,
        }
    }

    /// Create a detached payload header (RFC 7797)
    /// Used for JsonWebSignature2020 where payload is the document hash
    pub fn detached(alg: JwsAlgorithm, kid: Option<&str>) -> Self {
        Self {
            alg,
            kid: kid.map(String::from),
            b64: Some(false),
            crit: Some(vec!["b64".to_string()]),
            cty: None,
        }
    }

    /// Encode header as base64url JSON
    pub fn encode(&self) -> DidResult<String> {
        let json = serde_json::to_string(self)
            .map_err(|e| DidError::SerializationError(format!("Header serialize: {}", e)))?;
        Ok(URL_SAFE_NO_PAD.encode(json.as_bytes()))
    }

    /// Decode header from base64url JSON
    pub fn decode(encoded: &str) -> DidResult<Self> {
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|e| DidError::InvalidProof(format!("Header base64 decode: {}", e)))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DidError::SerializationError(format!("Header deserialize: {}", e)))
    }
}

/// Compact JWS token (header.payload.signature)
#[derive(Debug, Clone)]
pub struct CompactJws {
    /// Base64url-encoded header
    pub header_b64: String,
    /// Base64url-encoded payload (empty for detached)
    pub payload_b64: String,
    /// Base64url-encoded signature
    pub signature_b64: String,
}

impl CompactJws {
    /// Parse a compact JWS string
    pub fn parse(jws: &str) -> DidResult<Self> {
        let parts: Vec<&str> = jws.split('.').collect();
        if parts.len() != 3 {
            return Err(DidError::InvalidProof(format!(
                "JWS must have 3 parts separated by '.', got {}",
                parts.len()
            )));
        }

        Ok(Self {
            header_b64: parts[0].to_string(),
            payload_b64: parts[1].to_string(),
            signature_b64: parts[2].to_string(),
        })
    }

    /// Create a detached compact JWS (payload omitted per RFC 7515 Section 7.2.4)
    pub fn to_detached_string(&self) -> String {
        format!("{}..{}", self.header_b64, self.signature_b64)
    }

    /// Get the signing input (header_b64 + "." + payload_b64)
    pub fn signing_input(&self) -> Vec<u8> {
        format!("{}.{}", self.header_b64, self.payload_b64).into_bytes()
    }

    /// Get decoded signature bytes
    pub fn signature_bytes(&self) -> DidResult<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.signature_b64)
            .map_err(|e| DidError::InvalidProof(format!("Signature base64 decode: {}", e)))
    }

    /// Get decoded header
    pub fn header(&self) -> DidResult<JwsHeader> {
        JwsHeader::decode(&self.header_b64)
    }
}

impl std::fmt::Display for CompactJws {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}",
            self.header_b64, self.payload_b64, self.signature_b64
        )
    }
}

/// JsonWebSignature2020 proof structure
///
/// Per <https://w3c-ccg.github.io/lds-jws2020/>
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonWebSignature2020 {
    /// Proof type identifier
    #[serde(rename = "type")]
    pub proof_type: String,
    /// ISO 8601 creation timestamp
    pub created: String,
    /// Verification method URL
    pub verification_method: String,
    /// Proof purpose
    pub proof_purpose: String,
    /// Compact JWS token (detached signature)
    pub jws: String,
    /// Optional challenge for authentication proofs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    /// Optional domain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    /// Optional nonce
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
}

impl JsonWebSignature2020 {
    /// The proof type identifier string
    pub const PROOF_TYPE: &'static str = "JsonWebSignature2020";

    /// Sign a JSON document using JsonWebSignature2020
    ///
    /// The signing process:
    /// 1. Canonicalize the document (or use raw JSON bytes)
    /// 2. Create JWS header with algorithm and key ID
    /// 3. Create signing input: base64url(header) + "." + base64url(payload_hash)
    /// 4. Sign with the private key
    /// 5. Create detached JWS: base64url(header) + ".." + base64url(signature)
    pub fn sign(
        document: &serde_json::Value,
        secret_key_bytes: &[u8],
        vm_id: &str,
        purpose: ProofPurpose,
    ) -> DidResult<Self> {
        // Serialize document to canonical JSON bytes
        let doc_bytes = serialize_document(document)?;

        // Hash the document for signing input
        let doc_hash = sha256_hash(&doc_bytes);

        // Create JWS header (detached, no b64 encoding of payload)
        let header = JwsHeader::detached(JwsAlgorithm::EdDsa, Some(vm_id));
        let header_b64 = header.encode()?;

        // Payload is base64url of the document hash
        let payload_b64 = URL_SAFE_NO_PAD.encode(&doc_hash);

        // Signing input: ASCII(header_b64 || "." || payload_b64)
        let signing_input = format!("{}.{}", header_b64, payload_b64).into_bytes();

        // Sign with Ed25519
        let signer = Ed25519Signer::from_bytes(secret_key_bytes)?;
        let signature = signer.sign(&signing_input);
        let signature_b64 = URL_SAFE_NO_PAD.encode(&signature);

        // Detached JWS: header..signature (payload omitted)
        let jws = format!("{}..{}", header_b64, signature_b64);

        Ok(Self {
            proof_type: Self::PROOF_TYPE.to_string(),
            created: chrono::Utc::now().to_rfc3339(),
            verification_method: vm_id.to_string(),
            proof_purpose: purpose.as_str().to_string(),
            jws,
            challenge: None,
            domain: None,
            nonce: None,
        })
    }

    /// Verify a JsonWebSignature2020 proof against a document
    ///
    /// The verification process:
    /// 1. Parse the detached JWS
    /// 2. Reconstruct the signing input from header and document hash
    /// 3. Verify the signature with the public key
    pub fn verify(
        document: &serde_json::Value,
        proof: &Self,
        public_key_bytes: &[u8],
    ) -> DidResult<bool> {
        // Parse the detached JWS
        let jws_str = &proof.jws;

        // Handle both detached ("header..sig") and full ("header.payload.sig") formats
        let (header_b64, signature_b64) = parse_jws_parts(jws_str)?;

        // Decode the header to get the algorithm
        let header = JwsHeader::decode(&header_b64)?;

        // Serialize the document to get the payload
        let doc_bytes = serialize_document(document)?;
        let doc_hash = sha256_hash(&doc_bytes);
        let payload_b64 = URL_SAFE_NO_PAD.encode(&doc_hash);

        // Reconstruct signing input
        let signing_input = format!("{}.{}", header_b64, payload_b64).into_bytes();

        // Decode signature
        let signature = URL_SAFE_NO_PAD
            .decode(&signature_b64)
            .map_err(|e| DidError::InvalidProof(format!("Signature decode error: {}", e)))?;

        // Verify based on algorithm
        match header.alg {
            JwsAlgorithm::EdDsa => {
                let verifier = Ed25519Verifier::from_bytes(public_key_bytes)?;
                verifier.verify(&signing_input, &signature)
            }
            other => Err(DidError::InvalidProof(format!(
                "Algorithm {:?} verification not yet implemented",
                other
            ))),
        }
    }

    /// Verify using a VerificationMethod from a DID Document
    pub fn verify_with_method(
        document: &serde_json::Value,
        proof: &Self,
        vm: &VerificationMethod,
    ) -> DidResult<bool> {
        let public_key = vm.get_public_key_bytes()?;
        Self::verify(document, proof, &public_key)
    }

    /// Set challenge for authentication proofs
    pub fn with_challenge(mut self, challenge: &str) -> Self {
        self.challenge = Some(challenge.to_string());
        self
    }

    /// Set domain
    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = Some(domain.to_string());
        self
    }

    /// Set nonce
    pub fn with_nonce(mut self, nonce: &str) -> Self {
        self.nonce = Some(nonce.to_string());
        self
    }

    /// Convert to a crate::proof::Proof for embedding in documents
    pub fn to_proof(&self) -> crate::proof::Proof {
        crate::proof::Proof {
            proof_type: Self::PROOF_TYPE.to_string(),
            created: chrono::DateTime::parse_from_rfc3339(&self.created)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now()),
            verification_method: self.verification_method.clone(),
            proof_purpose: self.proof_purpose.clone(),
            proof_value: None,
            jws: Some(self.jws.clone()),
            challenge: self.challenge.clone(),
            domain: self.domain.clone(),
            nonce: self.nonce.clone(),
            cryptosuite: None,
        }
    }
}

/// JWS Signer for creating compact JWS tokens
pub struct JwsSigner {
    algorithm: JwsAlgorithm,
    secret_key: Vec<u8>,
    key_id: Option<String>,
}

impl JwsSigner {
    /// Create an EdDSA (Ed25519) signer
    pub fn ed_dsa(secret_key: &[u8], key_id: Option<&str>) -> DidResult<Self> {
        if secret_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 secret key must be 32 bytes".to_string(),
            ));
        }
        Ok(Self {
            algorithm: JwsAlgorithm::EdDsa,
            secret_key: secret_key.to_vec(),
            key_id: key_id.map(String::from),
        })
    }

    /// Sign a payload and return a compact JWS
    pub fn sign(&self, payload: &[u8]) -> DidResult<CompactJws> {
        let header = JwsHeader {
            alg: self.algorithm,
            kid: self.key_id.clone(),
            b64: None,
            crit: None,
            cty: None,
        };

        let header_b64 = header.encode()?;
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload);
        let signing_input = format!("{}.{}", header_b64, payload_b64).into_bytes();

        let signature_b64 = match self.algorithm {
            JwsAlgorithm::EdDsa => {
                let signer = Ed25519Signer::from_bytes(&self.secret_key)?;
                let sig = signer.sign(&signing_input);
                URL_SAFE_NO_PAD.encode(&sig)
            }
            other => {
                return Err(DidError::SigningFailed(format!(
                    "Algorithm {:?} signing not yet implemented",
                    other
                )));
            }
        };

        Ok(CompactJws {
            header_b64,
            payload_b64,
            signature_b64,
        })
    }
}

/// JWS Verifier for verifying compact JWS tokens
pub struct JwsVerifier {
    public_key: Vec<u8>,
}

impl JwsVerifier {
    /// Create an Ed25519 verifier
    pub fn ed25519(public_key: &[u8]) -> DidResult<Self> {
        if public_key.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 public key must be 32 bytes".to_string(),
            ));
        }
        Ok(Self {
            public_key: public_key.to_vec(),
        })
    }

    /// Verify a compact JWS token
    pub fn verify_compact(&self, jws: &CompactJws) -> DidResult<bool> {
        let header = jws.header()?;
        let signing_input = jws.signing_input();
        let signature = jws.signature_bytes()?;

        match header.alg {
            JwsAlgorithm::EdDsa => {
                let verifier = Ed25519Verifier::from_bytes(&self.public_key)?;
                verifier.verify(&signing_input, &signature)
            }
            other => Err(DidError::VerificationFailed(format!(
                "Algorithm {:?} verification not yet implemented",
                other
            ))),
        }
    }

    /// Verify a JWS string (compact serialization)
    pub fn verify_string(&self, jws_str: &str) -> DidResult<bool> {
        let jws = CompactJws::parse(jws_str)?;
        self.verify_compact(&jws)
    }
}

/// Parse a JWS string into header and signature parts
/// Handles both full ("header.payload.sig") and detached ("header..sig") formats
fn parse_jws_parts(jws_str: &str) -> DidResult<(String, String)> {
    let parts: Vec<&str> = jws_str.split('.').collect();

    match parts.len() {
        3 => {
            // Standard or detached format
            Ok((parts[0].to_string(), parts[2].to_string()))
        }
        _ => Err(DidError::InvalidProof(format!(
            "Invalid JWS format: expected 3 dot-separated parts, got {}",
            parts.len()
        ))),
    }
}

/// Serialize a JSON document to canonical bytes for signing
fn serialize_document(doc: &serde_json::Value) -> DidResult<Vec<u8>> {
    // Use a deterministic serialization
    // In a full implementation, this would use JSON-LD canonicalization (RDFC-1.0)
    // For now, we use a sorted-key JSON serialization as an approximation
    let canonical = canonicalize_json(doc)?;
    Ok(canonical.into_bytes())
}

/// Produce a deterministic JSON string by sorting object keys recursively
fn canonicalize_json(value: &serde_json::Value) -> DidResult<String> {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());

            let mut parts = Vec::with_capacity(sorted.len());
            for (k, v) in sorted {
                let key = serde_json::to_string(k)
                    .map_err(|e| DidError::SerializationError(e.to_string()))?;
                let val = canonicalize_json(v)?;
                parts.push(format!("{}:{}", key, val));
            }
            Ok(format!("{{{}}}", parts.join(",")))
        }
        serde_json::Value::Array(arr) => {
            let parts: DidResult<Vec<String>> = arr.iter().map(canonicalize_json).collect();
            Ok(format!("[{}]", parts?.join(",")))
        }
        other => {
            serde_json::to_string(other).map_err(|e| DidError::SerializationError(e.to_string()))
        }
    }
}

/// Compute SHA-256 hash
fn sha256_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Attach a JsonWebSignature2020 proof to a document
pub fn attach_jws_proof(
    document: &mut serde_json::Value,
    proof: &JsonWebSignature2020,
) -> DidResult<()> {
    let proof_value =
        serde_json::to_value(proof).map_err(|e| DidError::SerializationError(e.to_string()))?;

    if let Some(obj) = document.as_object_mut() {
        obj.insert("proof".to_string(), proof_value);
        Ok(())
    } else {
        Err(DidError::InvalidFormat(
            "Document must be a JSON object".to_string(),
        ))
    }
}

/// Extract and remove proof from a document (for verification)
pub fn extract_jws_proof(document: &mut serde_json::Value) -> DidResult<JsonWebSignature2020> {
    if let Some(obj) = document.as_object_mut() {
        let proof_value = obj
            .remove("proof")
            .ok_or_else(|| DidError::InvalidProof("Document has no 'proof' field".to_string()))?;

        serde_json::from_value(proof_value)
            .map_err(|e| DidError::SerializationError(format!("Proof deserialize: {}", e)))
    } else {
        Err(DidError::InvalidFormat(
            "Document must be a JSON object".to_string(),
        ))
    }
}

/// Sign a document end-to-end: add proof to document
pub fn sign_document(
    document: &mut serde_json::Value,
    secret_key_bytes: &[u8],
    vm_id: &str,
    purpose: ProofPurpose,
) -> DidResult<()> {
    // Sign the document without proof field
    let proof = JsonWebSignature2020::sign(document, secret_key_bytes, vm_id, purpose)?;
    attach_jws_proof(document, &proof)
}

/// Verify a document end-to-end: extract proof and verify
pub fn verify_document(
    document: &mut serde_json::Value,
    public_key_bytes: &[u8],
) -> DidResult<bool> {
    // Extract proof (this modifies the document by removing the proof)
    let proof = extract_jws_proof(document)?;

    // Verify without the proof field
    JsonWebSignature2020::verify(document, &proof, public_key_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::ed25519::Ed25519Signer;

    fn generate_test_keypair() -> (Vec<u8>, Vec<u8>) {
        let signer = Ed25519Signer::generate();
        (
            signer.secret_key_bytes().to_vec(),
            signer.public_key_bytes().to_vec(),
        )
    }

    #[test]
    fn test_jws_header_encode_decode() {
        let header = JwsHeader::ed_dsa(Some("did:key:z123#key-1"));
        let encoded = header.encode().unwrap();
        let decoded = JwsHeader::decode(&encoded).unwrap();
        assert_eq!(decoded.alg, JwsAlgorithm::EdDsa);
        assert_eq!(decoded.kid, Some("did:key:z123#key-1".to_string()));
    }

    #[test]
    fn test_detached_jws_header() {
        let header = JwsHeader::detached(JwsAlgorithm::EdDsa, Some("kid"));
        assert_eq!(header.b64, Some(false));
        assert!(header.crit.is_some());
        assert!(header.crit.as_ref().unwrap().contains(&"b64".to_string()));
    }

    #[test]
    fn test_compact_jws_sign_verify() {
        let (secret, public) = generate_test_keypair();
        let payload = b"Hello, World!";

        let signer = JwsSigner::ed_dsa(&secret, Some("test-key")).unwrap();
        let compact = signer.sign(payload).unwrap();

        let jws_string = compact.to_string();
        assert_eq!(jws_string.split('.').count(), 3);

        let verifier = JwsVerifier::ed25519(&public).unwrap();
        let valid = verifier.verify_string(&jws_string).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_compact_jws_invalid_signature() {
        let (secret, _) = generate_test_keypair();
        let (_, other_public) = generate_test_keypair();
        let payload = b"Hello, World!";

        let signer = JwsSigner::ed_dsa(&secret, None).unwrap();
        let compact = signer.sign(payload).unwrap();

        let verifier = JwsVerifier::ed25519(&other_public).unwrap();
        let valid = verifier.verify_compact(&compact).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_json_web_signature_2020_sign_verify() {
        let (secret, public) = generate_test_keypair();
        let document = serde_json::json!({
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential"],
            "issuer": "did:key:z6Mk",
            "credentialSubject": {
                "id": "did:example:alice",
                "name": "Alice"
            }
        });

        let vm_id = "did:key:z6Mk#key-1";
        let proof =
            JsonWebSignature2020::sign(&document, &secret, vm_id, ProofPurpose::AssertionMethod)
                .unwrap();

        assert_eq!(proof.proof_type, JsonWebSignature2020::PROOF_TYPE);
        assert_eq!(proof.verification_method, vm_id);
        assert!(proof.jws.contains(".."));

        let valid = JsonWebSignature2020::verify(&document, &proof, &public).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_jws_tampered_document() {
        let (secret, public) = generate_test_keypair();
        let document = serde_json::json!({
            "name": "Alice",
            "age": 30
        });

        let proof = JsonWebSignature2020::sign(
            &document,
            &secret,
            "did:key:z123#key-1",
            ProofPurpose::AssertionMethod,
        )
        .unwrap();

        // Tamper with the document
        let tampered = serde_json::json!({
            "name": "Bob",  // Changed!
            "age": 30
        });

        let valid = JsonWebSignature2020::verify(&tampered, &proof, &public).unwrap();
        assert!(!valid, "Tampered document should not verify");
    }

    #[test]
    fn test_sign_verify_document_end_to_end() {
        let (secret, public) = generate_test_keypair();
        let mut document = serde_json::json!({
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": "TestDocument",
            "subject": "test"
        });

        sign_document(
            &mut document,
            &secret,
            "did:key:z6Mk#key-1",
            ProofPurpose::AssertionMethod,
        )
        .unwrap();

        assert!(document.get("proof").is_some());

        let valid = verify_document(&mut document, &public).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_canonicalize_json_deterministic() {
        // Same data in different order should produce same canonical form
        let v1 = serde_json::json!({"b": 2, "a": 1});
        let v2 = serde_json::json!({"a": 1, "b": 2});

        let c1 = canonicalize_json(&v1).unwrap();
        let c2 = canonicalize_json(&v2).unwrap();

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_jws_algorithm_roundtrip() {
        for alg in [
            JwsAlgorithm::EdDsa,
            JwsAlgorithm::Es256,
            JwsAlgorithm::Es256K,
            JwsAlgorithm::Rs256,
        ] {
            let s = alg.as_str();
            let parsed = JwsAlgorithm::parse(s).unwrap();
            assert_eq!(parsed, alg);
        }
    }

    #[test]
    fn test_proof_to_crate_proof() {
        let (secret, _) = generate_test_keypair();
        let document = serde_json::json!({"test": true});

        let jws_proof = JsonWebSignature2020::sign(
            &document,
            &secret,
            "did:key:z#key-1",
            ProofPurpose::AssertionMethod,
        )
        .unwrap();

        let proof = jws_proof.to_proof();
        assert_eq!(proof.proof_type, JsonWebSignature2020::PROOF_TYPE);
        assert!(proof.jws.is_some());
    }
}
