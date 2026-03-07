//! JWS algorithm enum, extended header, and mock signer for testing
//!
//! Provides:
//! - `JwsAlgorithm` – enum of supported JWS signing algorithms
//! - `JwsHeader` – compact JOSE header using `JwsAlgorithm`
//! - `MockJwsSigner` – deterministic HMAC-SHA256 signer for unit tests

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── JwsAlgorithm ─────────────────────────────────────────────────────────────

/// JWS signing algorithm identifiers (RFC 7518)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JwsAlgorithm {
    /// Ed25519 (Edwards-curve Digital Signature Algorithm)
    EdDSA,
    /// ECDSA using P-256 and SHA-256
    Es256,
    /// ECDSA using P-384 and SHA-384
    Es384,
    /// RSASSA-PKCS1-v1_5 using SHA-256
    Rs256,
    /// RSASSA-PSS using SHA-256
    Ps256,
}

impl JwsAlgorithm {
    /// Return the standard algorithm identifier string used in JOSE headers.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EdDSA => "EdDSA",
            Self::Es256 => "ES256",
            Self::Es384 => "ES384",
            Self::Rs256 => "RS256",
            Self::Ps256 => "PS256",
        }
    }

    /// Parse from the JOSE algorithm string, returning `None` for unknown values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "EdDSA" => Some(Self::EdDSA),
            "ES256" => Some(Self::Es256),
            "ES384" => Some(Self::Es384),
            "RS256" => Some(Self::Rs256),
            "PS256" => Some(Self::Ps256),
            _ => None,
        }
    }

    /// Whether this algorithm produces fixed-length signatures
    pub fn is_fixed_length(&self) -> bool {
        matches!(self, Self::EdDSA | Self::Es256 | Self::Es384)
    }
}

impl std::fmt::Display for JwsAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── JwsHeader ─────────────────────────────────────────────────────────────────

/// Compact JOSE header using the typed `JwsAlgorithm` enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JwsHeader {
    /// Signing algorithm
    pub alg: JwsAlgorithm,
    /// Key ID (verification method URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
    /// Token type (typically "JWT")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typ: Option<String>,
    /// Whether the payload is base64url-encoded (RFC 7797)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64: Option<bool>,
}

impl JwsHeader {
    /// Create an EdDSA header
    pub fn ed_dsa(kid: Option<&str>) -> Self {
        Self {
            alg: JwsAlgorithm::EdDSA,
            kid: kid.map(String::from),
            typ: Some("JWT".to_string()),
            b64: None,
        }
    }

    /// Create an ES256 header
    pub fn es256(kid: Option<&str>) -> Self {
        Self {
            alg: JwsAlgorithm::Es256,
            kid: kid.map(String::from),
            typ: Some("JWT".to_string()),
            b64: None,
        }
    }

    /// Create an RS256 header
    pub fn rs256(kid: Option<&str>) -> Self {
        Self {
            alg: JwsAlgorithm::Rs256,
            kid: kid.map(String::from),
            typ: Some("JWT".to_string()),
            b64: None,
        }
    }

    /// Encode the header as base64url(JSON)
    pub fn encode(&self) -> DidResult<String> {
        let json =
            serde_json::to_string(self).map_err(|e| DidError::SerializationError(e.to_string()))?;
        Ok(URL_SAFE_NO_PAD.encode(json.as_bytes()))
    }

    /// Decode from base64url(JSON)
    pub fn decode(encoded: &str) -> DidResult<Self> {
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|e| DidError::InvalidProof(format!("JwsHeader base64url decode: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DidError::InvalidProof(format!("JwsHeader JSON parse: {e}")))
    }
}

// ── JwsSigner / JwsVerifier traits ───────────────────────────────────────────

/// Trait for signing raw byte payloads (returns raw signature bytes)
pub trait JwsSigner: Send + Sync {
    /// Sign the given payload bytes, returning raw signature bytes.
    fn sign(&self, payload: &[u8]) -> DidResult<Vec<u8>>;

    /// The algorithm this signer implements.
    fn algorithm(&self) -> JwsAlgorithm;
}

/// Trait for verifying raw byte payloads against a signature
pub trait JwsVerifier: Send + Sync {
    /// Verify the signature against the payload. Returns `true` on success.
    fn verify(&self, payload: &[u8], signature: &[u8]) -> DidResult<bool>;

    /// The algorithm this verifier implements.
    fn algorithm(&self) -> JwsAlgorithm;
}

// ── MockJwsSigner ─────────────────────────────────────────────────────────────

/// Deterministic mock signer using HMAC-SHA256 with a fixed key.
///
/// Produces a 32-byte HMAC-SHA256 of `key || payload` for repeatable tests.
/// NOT suitable for production use.
pub struct MockJwsSigner {
    /// Fixed secret key for HMAC computation
    key: Vec<u8>,
    /// Key identifier
    pub kid: Option<String>,
}

impl MockJwsSigner {
    /// Create a new mock signer with the given key material.
    pub fn new(key: impl Into<Vec<u8>>, kid: Option<&str>) -> Self {
        Self {
            key: key.into(),
            kid: kid.map(String::from),
        }
    }

    /// Create with a well-known test key (`[0u8; 32]`).
    pub fn test_key() -> Self {
        Self::new(vec![0u8; 32], Some("mock-key-1"))
    }

    /// Compute HMAC-SHA256(key, payload) deterministically.
    pub fn hmac_sha256(&self, payload: &[u8]) -> Vec<u8> {
        // Simple deterministic: SHA256(key || SHA256(payload)) — no constant-time requirement for tests
        let mut hasher = Sha256::new();
        hasher.update(&self.key);
        hasher.update(payload);
        hasher.finalize().to_vec()
    }
}

impl JwsSigner for MockJwsSigner {
    fn sign(&self, payload: &[u8]) -> DidResult<Vec<u8>> {
        Ok(self.hmac_sha256(payload))
    }

    fn algorithm(&self) -> JwsAlgorithm {
        JwsAlgorithm::EdDSA // reported as EdDSA in tests
    }
}

/// Mock verifier matching `MockJwsSigner`.
pub struct MockJwsVerifier {
    key: Vec<u8>,
}

impl MockJwsVerifier {
    /// Create a verifier matching the given signer.
    pub fn from_signer(signer: &MockJwsSigner) -> Self {
        Self {
            key: signer.key.clone(),
        }
    }

    /// Create with the well-known test key.
    pub fn test_key() -> Self {
        Self { key: vec![0u8; 32] }
    }
}

impl JwsVerifier for MockJwsVerifier {
    fn verify(&self, payload: &[u8], signature: &[u8]) -> DidResult<bool> {
        let mut hasher = Sha256::new();
        hasher.update(&self.key);
        hasher.update(payload);
        let expected = hasher.finalize();
        Ok(expected.as_slice() == signature)
    }

    fn algorithm(&self) -> JwsAlgorithm {
        JwsAlgorithm::EdDSA
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JwsAlgorithm ─────────────────────────────────────────────────────

    #[test]
    fn test_algorithm_as_str() {
        assert_eq!(JwsAlgorithm::EdDSA.as_str(), "EdDSA");
        assert_eq!(JwsAlgorithm::Es256.as_str(), "ES256");
        assert_eq!(JwsAlgorithm::Es384.as_str(), "ES384");
        assert_eq!(JwsAlgorithm::Rs256.as_str(), "RS256");
        assert_eq!(JwsAlgorithm::Ps256.as_str(), "PS256");
    }

    #[test]
    fn test_algorithm_parse_valid() {
        assert_eq!(JwsAlgorithm::parse("EdDSA"), Some(JwsAlgorithm::EdDSA));
        assert_eq!(JwsAlgorithm::parse("ES256"), Some(JwsAlgorithm::Es256));
        assert_eq!(JwsAlgorithm::parse("ES384"), Some(JwsAlgorithm::Es384));
        assert_eq!(JwsAlgorithm::parse("RS256"), Some(JwsAlgorithm::Rs256));
        assert_eq!(JwsAlgorithm::parse("PS256"), Some(JwsAlgorithm::Ps256));
    }

    #[test]
    fn test_algorithm_parse_invalid() {
        assert!(JwsAlgorithm::parse("HS256").is_none());
        assert!(JwsAlgorithm::parse("").is_none());
        assert!(JwsAlgorithm::parse("edDSA").is_none()); // case-sensitive
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", JwsAlgorithm::Es256), "ES256");
    }

    #[test]
    fn test_algorithm_is_fixed_length() {
        assert!(JwsAlgorithm::EdDSA.is_fixed_length());
        assert!(JwsAlgorithm::Es256.is_fixed_length());
        assert!(JwsAlgorithm::Es384.is_fixed_length());
        assert!(!JwsAlgorithm::Rs256.is_fixed_length());
        assert!(!JwsAlgorithm::Ps256.is_fixed_length());
    }

    #[test]
    fn test_algorithm_roundtrip_via_str() {
        let algs = [
            JwsAlgorithm::EdDSA,
            JwsAlgorithm::Es256,
            JwsAlgorithm::Es384,
            JwsAlgorithm::Rs256,
            JwsAlgorithm::Ps256,
        ];
        for alg in &algs {
            let s = alg.as_str();
            let parsed = JwsAlgorithm::parse(s).unwrap();
            assert_eq!(parsed, *alg);
        }
    }

    // ── JwsHeader ────────────────────────────────────────────────────────

    #[test]
    fn test_header_ed_dsa() {
        let h = JwsHeader::ed_dsa(Some("key-1"));
        assert_eq!(h.alg, JwsAlgorithm::EdDSA);
        assert_eq!(h.kid, Some("key-1".to_string()));
        assert_eq!(h.typ, Some("JWT".to_string()));
        assert!(h.b64.is_none());
    }

    #[test]
    fn test_header_es256() {
        let h = JwsHeader::es256(None);
        assert_eq!(h.alg, JwsAlgorithm::Es256);
        assert!(h.kid.is_none());
    }

    #[test]
    fn test_header_rs256() {
        let h = JwsHeader::rs256(Some("rsa-key"));
        assert_eq!(h.alg, JwsAlgorithm::Rs256);
        assert_eq!(h.kid, Some("rsa-key".to_string()));
    }

    #[test]
    fn test_header_encode_decode_roundtrip() {
        let h = JwsHeader::ed_dsa(Some("did:key:z123#key-1"));
        let encoded = h.encode().unwrap();
        let decoded = JwsHeader::decode(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn test_header_decode_invalid_base64() {
        assert!(JwsHeader::decode("!!!invalid!!!").is_err());
    }

    #[test]
    fn test_header_encode_contains_alg() {
        let h = JwsHeader::es256(None);
        let encoded = h.encode().unwrap();
        // Decoded JSON should contain "ES256"
        let decoded = JwsHeader::decode(&encoded).unwrap();
        assert_eq!(decoded.alg.as_str(), "ES256");
    }

    #[test]
    fn test_header_b64_field_serialised() {
        let h = JwsHeader {
            alg: JwsAlgorithm::EdDSA,
            kid: None,
            typ: None,
            b64: Some(false),
        };
        let encoded = h.encode().unwrap();
        let decoded = JwsHeader::decode(&encoded).unwrap();
        assert_eq!(decoded.b64, Some(false));
    }

    // ── MockJwsSigner ─────────────────────────────────────────────────────

    #[test]
    fn test_mock_signer_deterministic() {
        let signer = MockJwsSigner::test_key();
        let payload = b"hello world";
        let sig1 = signer.sign(payload).unwrap();
        let sig2 = signer.sign(payload).unwrap();
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_mock_signer_different_payload_different_sig() {
        let signer = MockJwsSigner::test_key();
        let sig1 = signer.sign(b"payload one").unwrap();
        let sig2 = signer.sign(b"payload two").unwrap();
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_mock_signer_sign_verify_roundtrip() {
        let signer = MockJwsSigner::test_key();
        let verifier = MockJwsVerifier::test_key();
        let payload = b"test payload for verification";
        let sig = signer.sign(payload).unwrap();
        assert!(verifier.verify(payload, &sig).unwrap());
    }

    #[test]
    fn test_mock_verifier_wrong_payload_fails() {
        let signer = MockJwsSigner::test_key();
        let verifier = MockJwsVerifier::test_key();
        let sig = signer.sign(b"original").unwrap();
        assert!(!verifier.verify(b"tampered", &sig).unwrap());
    }

    #[test]
    fn test_mock_verifier_wrong_key_fails() {
        let signer = MockJwsSigner::new([1u8; 32].to_vec(), None);
        let sig = signer.sign(b"data").unwrap();
        let wrong_verifier = MockJwsVerifier { key: vec![2u8; 32] };
        assert!(!wrong_verifier.verify(b"data", &sig).unwrap());
    }

    #[test]
    fn test_mock_signer_algorithm() {
        let signer = MockJwsSigner::test_key();
        assert_eq!(signer.algorithm(), JwsAlgorithm::EdDSA);
    }

    #[test]
    fn test_mock_verifier_from_signer() {
        let signer = MockJwsSigner::new(vec![42u8; 32], Some("test-kid"));
        let verifier = MockJwsVerifier::from_signer(&signer);
        let sig = signer.sign(b"hello").unwrap();
        assert!(verifier.verify(b"hello", &sig).unwrap());
    }

    #[test]
    fn test_mock_signer_kid_field() {
        let signer = MockJwsSigner::new(vec![0u8; 32], Some("did:key:z123#k1"));
        assert_eq!(signer.kid, Some("did:key:z123#k1".to_string()));
    }

    #[test]
    fn test_mock_signer_test_key_has_kid() {
        let signer = MockJwsSigner::test_key();
        assert!(signer.kid.is_some());
    }

    #[test]
    fn test_mock_signer_signature_length_is_32() {
        let signer = MockJwsSigner::test_key();
        let sig = signer.sign(b"data").unwrap();
        assert_eq!(sig.len(), 32); // SHA256 output
    }
}
