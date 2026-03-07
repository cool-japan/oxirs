//! Ed25519 JWS signer/verifier for use in the signatures module
//!
//! Provides a structured `JwsSignature` type and `Ed25519JwsSigner` /
//! `Ed25519JwsVerifier` that operate independently of the proof module.
//! Compact serialization: `<base64url(header)>.<base64url(payload)>.<base64url(sig)>`.

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── JwsSignatureHeader ──────────────────────────────────────────────────────

/// JOSE header for a structured JWS
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JwsSignatureHeader {
    /// Algorithm identifier ("EdDSA", "ES256", …)
    pub alg: String,
    /// Key-ID (verification method URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
    /// Token type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typ: Option<String>,
}

impl JwsSignatureHeader {
    /// Create an EdDSA header
    pub fn ed_dsa(kid: Option<&str>) -> Self {
        Self {
            alg: "EdDSA".to_string(),
            kid: kid.map(String::from),
            typ: Some("JWT".to_string()),
        }
    }

    /// Create an ES256 header
    pub fn es256(kid: Option<&str>) -> Self {
        Self {
            alg: "ES256".to_string(),
            kid: kid.map(String::from),
            typ: Some("JWT".to_string()),
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
            .map_err(|e| DidError::InvalidProof(format!("header base64url decode: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DidError::InvalidProof(format!("header JSON parse: {e}")))
    }
}

// ── JwsPayload ──────────────────────────────────────────────────────────────

/// Base64url-encoded payload wrapper
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JwsPayload {
    /// Raw bytes of the payload
    raw: Vec<u8>,
}

impl JwsPayload {
    /// Create from raw bytes
    pub fn from_bytes(raw: &[u8]) -> Self {
        Self { raw: raw.to_vec() }
    }

    /// Create from JSON-serialisable value
    pub fn from_json<T: Serialize>(value: &T) -> DidResult<Self> {
        let json =
            serde_json::to_vec(value).map_err(|e| DidError::SerializationError(e.to_string()))?;
        Ok(Self { raw: json })
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// Encode as base64url
    pub fn encode(&self) -> String {
        URL_SAFE_NO_PAD.encode(&self.raw)
    }

    /// Decode from base64url
    pub fn decode(encoded: &str) -> DidResult<Self> {
        let raw = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|e| DidError::InvalidProof(format!("payload base64url decode: {e}")))?;
        Ok(Self { raw })
    }

    /// Deserialise the raw bytes as JSON
    pub fn as_json<T: for<'de> Deserialize<'de>>(&self) -> DidResult<T> {
        serde_json::from_slice(&self.raw)
            .map_err(|e| DidError::InvalidProof(format!("payload JSON parse: {e}")))
    }

    /// SHA-256 digest of the raw payload
    pub fn digest(&self) -> Vec<u8> {
        Sha256::digest(&self.raw).to_vec()
    }
}

// ── JwsSignature ────────────────────────────────────────────────────────────

/// Structured JWS with header, payload and raw signature bytes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwsSignature {
    /// JOSE header
    pub header: JwsSignatureHeader,
    /// Payload
    pub payload: JwsPayload,
    /// Raw signature bytes
    pub signature: Vec<u8>,
}

impl JwsSignature {
    /// Serialize to compact JWS: `<header>.<payload>.<sig>`
    pub fn to_compact(&self) -> DidResult<String> {
        let h = self.header.encode()?;
        let p = self.payload.encode();
        let s = URL_SAFE_NO_PAD.encode(&self.signature);
        Ok(format!("{h}.{p}.{s}"))
    }

    /// Parse from compact serialization
    pub fn from_compact(compact: &str) -> DidResult<Self> {
        let parts: Vec<&str> = compact.splitn(3, '.').collect();
        if parts.len() != 3 {
            return Err(DidError::InvalidProof(
                "JWS compact must have 3 parts".to_string(),
            ));
        }
        let header = JwsSignatureHeader::decode(parts[0])?;
        let payload = JwsPayload::decode(parts[1])?;
        let signature = URL_SAFE_NO_PAD
            .decode(parts[2])
            .map_err(|e| DidError::InvalidProof(format!("sig base64url decode: {e}")))?;
        Ok(Self {
            header,
            payload,
            signature,
        })
    }

    /// Return the signing input: `ASCII(base64url(header) || '.' || base64url(payload))`
    pub fn signing_input(&self) -> DidResult<Vec<u8>> {
        let h = self.header.encode()?;
        let p = self.payload.encode();
        Ok(format!("{h}.{p}").into_bytes())
    }
}

// ── Ed25519JwsSigner trait  ─────────────────────────────────────────────────

/// Trait for JWS signing
pub trait JwsSignerTrait {
    /// Sign a raw payload and return a structured `JwsSignature`
    fn sign_payload(&self, payload: &[u8]) -> DidResult<JwsSignature>;
}

/// Trait for JWS verification
pub trait JwsVerifierTrait {
    /// Verify a structured `JwsSignature`
    fn verify_jws(&self, jws: &JwsSignature) -> DidResult<bool>;
}

// ── Ed25519JwsSigner ────────────────────────────────────────────────────────

/// Ed25519-based JWS signer
///
/// Uses ed25519-dalek for signing.  The algorithm identifier is `"EdDSA"`.
pub struct Ed25519JwsSigner {
    signing_key: SigningKey,
    kid: Option<String>,
}

impl Ed25519JwsSigner {
    /// Create from a 32-byte Ed25519 secret seed
    pub fn from_secret_bytes(bytes: &[u8], kid: Option<&str>) -> DidResult<Self> {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| DidError::InvalidKey("Ed25519 secret must be 32 bytes".to_string()))?;
        let signing_key = SigningKey::from_bytes(&arr);
        Ok(Self {
            signing_key,
            kid: kid.map(String::from),
        })
    }

    /// Generate a new random key pair using OS entropy
    pub fn generate(kid: Option<&str>) -> DidResult<Self> {
        use ed25519_dalek::SigningKey as DalekSigningKey;
        // Use OS RNG via p256's rand_core integration
        let mut seed = [0u8; 32];
        use p256::elliptic_curve::rand_core::RngCore;
        p256::elliptic_curve::rand_core::OsRng.fill_bytes(&mut seed);
        let signing_key = DalekSigningKey::from_bytes(&seed);
        Ok(Self {
            signing_key,
            kid: kid.map(String::from),
        })
    }

    /// Return the 32-byte compressed public key
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Return the corresponding verifier
    pub fn verifier(&self) -> Ed25519JwsVerifier {
        Ed25519JwsVerifier {
            verifying_key: self.signing_key.verifying_key(),
        }
    }
}

impl JwsSignerTrait for Ed25519JwsSigner {
    fn sign_payload(&self, payload: &[u8]) -> DidResult<JwsSignature> {
        let header = JwsSignatureHeader::ed_dsa(self.kid.as_deref());
        let jws_payload = JwsPayload::from_bytes(payload);

        // signing input = base64url(header) + '.' + base64url(payload)
        let h_enc = header.encode()?;
        let p_enc = jws_payload.encode();
        let input = format!("{h_enc}.{p_enc}");

        let sig: DalekSignature = self.signing_key.sign(input.as_bytes());
        Ok(JwsSignature {
            header,
            payload: jws_payload,
            signature: sig.to_bytes().to_vec(),
        })
    }
}

// ── Ed25519JwsVerifier ──────────────────────────────────────────────────────

/// Ed25519-based JWS verifier
pub struct Ed25519JwsVerifier {
    verifying_key: VerifyingKey,
}

impl Ed25519JwsVerifier {
    /// Create from a 32-byte compressed public key
    pub fn from_public_bytes(bytes: &[u8]) -> DidResult<Self> {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| DidError::InvalidKey("Ed25519 public key must be 32 bytes".to_string()))?;
        let verifying_key = VerifyingKey::from_bytes(&arr)
            .map_err(|e| DidError::InvalidKey(format!("Invalid Ed25519 public key: {e}")))?;
        Ok(Self { verifying_key })
    }
}

impl JwsVerifierTrait for Ed25519JwsVerifier {
    fn verify_jws(&self, jws: &JwsSignature) -> DidResult<bool> {
        let h_enc = jws.header.encode()?;
        let p_enc = jws.payload.encode();
        let input = format!("{h_enc}.{p_enc}");

        let sig_bytes: [u8; 64] = jws
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| DidError::InvalidProof("Signature must be 64 bytes".to_string()))?;
        let sig = DalekSignature::from_bytes(&sig_bytes);

        Ok(self.verifying_key.verify(input.as_bytes(), &sig).is_ok())
    }
}

// ── EcdsaJwsSigner (P-256 / ES256) ─────────────────────────────────────────

/// ES256 (ECDSA-P256) JWS signer
pub struct EcdsaJwsSigner {
    signing_key: p256::ecdsa::SigningKey,
    kid: Option<String>,
}

impl EcdsaJwsSigner {
    /// Create from a 32-byte P-256 secret scalar
    pub fn from_secret_bytes(bytes: &[u8], kid: Option<&str>) -> DidResult<Self> {
        if bytes.len() != 32 {
            return Err(DidError::InvalidKey(format!(
                "P-256 secret key must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        let signing_key = p256::ecdsa::SigningKey::from_bytes(bytes.into())
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 key: {e}")))?;
        Ok(Self {
            signing_key,
            kid: kid.map(String::from),
        })
    }

    /// Generate a random P-256 key pair
    pub fn generate(kid: Option<&str>) -> Self {
        let signing_key =
            p256::ecdsa::SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng);
        Self {
            signing_key,
            kid: kid.map(String::from),
        }
    }

    /// 33-byte compressed public key
    pub fn public_key_compressed(&self) -> Vec<u8> {
        use p256::elliptic_curve::sec1::ToEncodedPoint;
        self.signing_key
            .verifying_key()
            .to_encoded_point(true)
            .as_bytes()
            .to_vec()
    }

    /// Return the corresponding verifier
    pub fn verifier(&self) -> EcdsaJwsVerifier {
        EcdsaJwsVerifier {
            verifying_key: *self.signing_key.verifying_key(),
        }
    }
}

impl JwsSignerTrait for EcdsaJwsSigner {
    fn sign_payload(&self, payload: &[u8]) -> DidResult<JwsSignature> {
        use p256::ecdsa::signature::Signer as P256Sign;

        let header = JwsSignatureHeader::es256(self.kid.as_deref());
        let jws_payload = JwsPayload::from_bytes(payload);

        let h_enc = header.encode()?;
        let p_enc = jws_payload.encode();
        let input = format!("{h_enc}.{p_enc}");

        let sig: p256::ecdsa::Signature = self.signing_key.sign(input.as_bytes());
        let sig_bytes = sig.to_bytes().to_vec();

        Ok(JwsSignature {
            header,
            payload: jws_payload,
            signature: sig_bytes,
        })
    }
}

// ── EcdsaJwsVerifier ────────────────────────────────────────────────────────

/// ES256 (ECDSA-P256) JWS verifier
pub struct EcdsaJwsVerifier {
    verifying_key: p256::ecdsa::VerifyingKey,
}

impl EcdsaJwsVerifier {
    /// Create from a 33-byte compressed public key
    pub fn from_compressed(bytes: &[u8]) -> DidResult<Self> {
        let verifying_key = p256::ecdsa::VerifyingKey::from_sec1_bytes(bytes)
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 public key: {e}")))?;
        Ok(Self { verifying_key })
    }
}

impl JwsVerifierTrait for EcdsaJwsVerifier {
    fn verify_jws(&self, jws: &JwsSignature) -> DidResult<bool> {
        use p256::ecdsa::signature::Verifier as P256Verify;

        let h_enc = jws.header.encode()?;
        let p_enc = jws.payload.encode();
        let input = format!("{h_enc}.{p_enc}");

        let sig = p256::ecdsa::Signature::from_slice(&jws.signature)
            .map_err(|e| DidError::InvalidProof(format!("Invalid ECDSA signature: {e}")))?;

        Ok(self.verifying_key.verify(input.as_bytes(), &sig).is_ok())
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ed25519_signer() -> (Ed25519JwsSigner, [u8; 32]) {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i + 1) as u8;
        }
        let signer = Ed25519JwsSigner::from_secret_bytes(&seed, Some("key-1")).unwrap();
        (signer, seed)
    }

    // ── JwsSignatureHeader tests ─────────────────────────────────────────────

    #[test]
    fn test_header_ed_dsa_fields() {
        let h = JwsSignatureHeader::ed_dsa(Some("key-1"));
        assert_eq!(h.alg, "EdDSA");
        assert_eq!(h.kid, Some("key-1".to_string()));
        assert_eq!(h.typ, Some("JWT".to_string()));
    }

    #[test]
    fn test_header_es256_fields() {
        let h = JwsSignatureHeader::es256(None);
        assert_eq!(h.alg, "ES256");
        assert!(h.kid.is_none());
    }

    #[test]
    fn test_header_encode_decode_roundtrip() {
        let h = JwsSignatureHeader::ed_dsa(Some("did:key:z#key-1"));
        let encoded = h.encode().unwrap();
        let decoded = JwsSignatureHeader::decode(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn test_header_decode_invalid_base64() {
        assert!(JwsSignatureHeader::decode("!!!").is_err());
    }

    #[test]
    fn test_header_decode_invalid_json() {
        let b64 = URL_SAFE_NO_PAD.encode(b"not-json");
        assert!(JwsSignatureHeader::decode(&b64).is_err());
    }

    // ── JwsPayload tests ─────────────────────────────────────────────────────

    #[test]
    fn test_payload_from_bytes_roundtrip() {
        let data = b"hello world";
        let p = JwsPayload::from_bytes(data);
        let enc = p.encode();
        let decoded = JwsPayload::decode(&enc).unwrap();
        assert_eq!(decoded.as_bytes(), data);
    }

    #[test]
    fn test_payload_from_json() {
        let val = serde_json::json!({"iss": "did:key:z"});
        let p = JwsPayload::from_json(&val).unwrap();
        let decoded: serde_json::Value = p.as_json().unwrap();
        assert_eq!(decoded["iss"], "did:key:z");
    }

    #[test]
    fn test_payload_digest_deterministic() {
        let p = JwsPayload::from_bytes(b"abc");
        assert_eq!(p.digest(), p.digest());
    }

    #[test]
    fn test_payload_digest_differs_for_different_data() {
        let p1 = JwsPayload::from_bytes(b"abc");
        let p2 = JwsPayload::from_bytes(b"def");
        assert_ne!(p1.digest(), p2.digest());
    }

    // ── JwsSignature compact format tests ───────────────────────────────────

    #[test]
    fn test_compact_three_parts() {
        let (signer, _) = make_ed25519_signer();
        let jws = signer.sign_payload(b"test payload").unwrap();
        let compact = jws.to_compact().unwrap();
        assert_eq!(compact.split('.').count(), 3);
    }

    #[test]
    fn test_compact_roundtrip() {
        let (signer, _) = make_ed25519_signer();
        let jws = signer.sign_payload(b"round-trip").unwrap();
        let compact = jws.to_compact().unwrap();
        let parsed = JwsSignature::from_compact(&compact).unwrap();
        assert_eq!(parsed.payload.as_bytes(), b"round-trip");
        assert_eq!(parsed.header.alg, "EdDSA");
    }

    #[test]
    fn test_compact_from_invalid_format() {
        assert!(JwsSignature::from_compact("only.two").is_err());
        assert!(JwsSignature::from_compact("a.b").is_err());
    }

    #[test]
    fn test_compact_invalid_base64_signature() {
        assert!(JwsSignature::from_compact("dGVzdA.dGVzdA.!!!").is_err());
    }

    // ── Ed25519JwsSigner tests ───────────────────────────────────────────────

    #[test]
    fn test_ed25519_sign_verify_roundtrip() {
        let (signer, _) = make_ed25519_signer();
        let verifier = signer.verifier();
        let jws = signer.sign_payload(b"hello ed25519").unwrap();
        assert!(verifier.verify_jws(&jws).unwrap());
    }

    #[test]
    fn test_ed25519_invalid_signature_detected() {
        let (signer, _) = make_ed25519_signer();
        let verifier = signer.verifier();
        let mut jws = signer.sign_payload(b"tamper me").unwrap();
        // Flip a byte in the signature
        if let Some(b) = jws.signature.first_mut() {
            *b ^= 0xFF;
        }
        assert!(!verifier.verify_jws(&jws).unwrap());
    }

    #[test]
    fn test_ed25519_wrong_key_fails() {
        let (signer, _) = make_ed25519_signer();
        let jws = signer.sign_payload(b"signed payload").unwrap();

        // Build verifier with a different key
        let other_signer = Ed25519JwsSigner::generate(None).unwrap();
        let other_verifier = other_signer.verifier();
        assert!(!other_verifier.verify_jws(&jws).unwrap());
    }

    #[test]
    fn test_ed25519_bad_secret_length() {
        assert!(Ed25519JwsSigner::from_secret_bytes(&[0u8; 31], None).is_err());
        assert!(Ed25519JwsSigner::from_secret_bytes(&[0u8; 33], None).is_err());
    }

    #[test]
    fn test_ed25519_bad_public_length() {
        assert!(Ed25519JwsVerifier::from_public_bytes(&[0u8; 31]).is_err());
        assert!(Ed25519JwsVerifier::from_public_bytes(&[0u8; 33]).is_err());
    }

    #[test]
    fn test_ed25519_header_contains_kid() {
        let mut seed = [0u8; 32];
        seed[0] = 9;
        let signer = Ed25519JwsSigner::from_secret_bytes(&seed, Some("did:key:z#key-0")).unwrap();
        let jws = signer.sign_payload(b"kid check").unwrap();
        assert_eq!(jws.header.kid, Some("did:key:z#key-0".to_string()));
    }

    #[test]
    fn test_ed25519_public_key_bytes_roundtrip() {
        let (signer, seed) = make_ed25519_signer();
        let pk = signer.public_key_bytes();
        let signer2 = Ed25519JwsSigner::from_secret_bytes(&seed, None).unwrap();
        assert_eq!(signer2.public_key_bytes(), pk);
    }

    // ── EcdsaJwsSigner (ES256) tests ─────────────────────────────────────────

    #[test]
    fn test_ecdsa_sign_verify_roundtrip() {
        let signer = EcdsaJwsSigner::generate(Some("p256-key-1"));
        let verifier = signer.verifier();
        let jws = signer.sign_payload(b"hello ecdsa p256").unwrap();
        assert!(verifier.verify_jws(&jws).unwrap());
    }

    #[test]
    fn test_ecdsa_invalid_signature_detected() {
        let signer = EcdsaJwsSigner::generate(None);
        let verifier = signer.verifier();
        let mut jws = signer.sign_payload(b"tamper ecdsa").unwrap();
        if let Some(b) = jws.signature.first_mut() {
            *b ^= 0xFF;
        }
        assert!(verifier.verify_jws(&jws).is_err() || !verifier.verify_jws(&jws).unwrap_or(true));
    }

    #[test]
    fn test_ecdsa_wrong_key_fails() {
        let signer = EcdsaJwsSigner::generate(None);
        let jws = signer.sign_payload(b"ecdsa cross-key").unwrap();
        let other_signer = EcdsaJwsSigner::generate(None);
        let other_verifier = other_signer.verifier();
        assert!(!other_verifier.verify_jws(&jws).unwrap_or(true));
    }

    #[test]
    fn test_ecdsa_bad_secret_length() {
        assert!(EcdsaJwsSigner::from_secret_bytes(&[0u8; 31], None).is_err());
        assert!(EcdsaJwsSigner::from_secret_bytes(&[0u8; 33], None).is_err());
    }

    #[test]
    fn test_ecdsa_header_alg_es256() {
        let signer = EcdsaJwsSigner::generate(Some("es256-kid"));
        let jws = signer.sign_payload(b"algorithm check").unwrap();
        assert_eq!(jws.header.alg, "ES256");
        assert_eq!(jws.header.kid, Some("es256-kid".to_string()));
    }

    #[test]
    fn test_ecdsa_compact_roundtrip() {
        let signer = EcdsaJwsSigner::generate(None);
        let verifier = signer.verifier();
        let jws = signer.sign_payload(b"compact ecdsa").unwrap();
        let compact = jws.to_compact().unwrap();
        let parsed = JwsSignature::from_compact(&compact).unwrap();
        assert!(verifier.verify_jws(&parsed).unwrap());
    }

    #[test]
    fn test_ecdsa_compressed_pubkey_length() {
        let signer = EcdsaJwsSigner::generate(None);
        let pk = signer.public_key_compressed();
        assert_eq!(pk.len(), 33);
        // Must start with 0x02 or 0x03 for compressed point
        assert!(pk[0] == 0x02 || pk[0] == 0x03);
    }

    #[test]
    fn test_ecdsa_verifier_from_compressed() {
        let signer = EcdsaJwsSigner::generate(None);
        let pk = signer.public_key_compressed();
        let verifier = EcdsaJwsVerifier::from_compressed(&pk).unwrap();
        let jws = signer.sign_payload(b"verifier from compressed").unwrap();
        assert!(verifier.verify_jws(&jws).unwrap());
    }

    #[test]
    fn test_ecdsa_verifier_from_invalid_key() {
        assert!(EcdsaJwsVerifier::from_compressed(&[0u8; 33]).is_err());
    }

    // ── Signing-input determinism ────────────────────────────────────────────

    #[test]
    fn test_signing_input_deterministic() {
        let (signer, _) = make_ed25519_signer();
        let jws = signer.sign_payload(b"determinism").unwrap();
        let input1 = jws.signing_input().unwrap();
        let input2 = jws.signing_input().unwrap();
        assert_eq!(input1, input2);
    }

    #[test]
    fn test_signing_input_differs_for_different_payload() {
        let (signer, _) = make_ed25519_signer();
        let jws1 = signer.sign_payload(b"payload one").unwrap();
        let jws2 = signer.sign_payload(b"payload two").unwrap();
        assert_ne!(jws1.signing_input().unwrap(), jws2.signing_input().unwrap());
    }

    // ── JSON-payload convenience ─────────────────────────────────────────────

    #[test]
    fn test_json_payload_sign_verify() {
        let (signer, _) = make_ed25519_signer();
        let verifier = signer.verifier();
        let claims = serde_json::json!({
            "sub": "did:key:zAlice",
            "iss": "did:key:zIssuer",
            "exp": 9999999999u64,
        });
        let payload_bytes = serde_json::to_vec(&claims).unwrap();
        let jws = signer.sign_payload(&payload_bytes).unwrap();
        assert!(verifier.verify_jws(&jws).unwrap());

        let decoded: serde_json::Value = jws.payload.as_json().unwrap();
        assert_eq!(decoded["sub"], "did:key:zAlice");
    }
}
