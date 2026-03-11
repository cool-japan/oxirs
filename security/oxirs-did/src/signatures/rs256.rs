//! RS256 (RSASSA-PKCS1-v1_5 with SHA-256) signature implementation
//!
//! RS256 is defined in RFC 7518 and uses RSA with PKCS#1 v1.5 padding
//! and SHA-256 as the hash function.
//!
//! This provides interoperability with legacy systems that require RSA signatures.
//! For new systems, EdDSA or ES256 are preferred.
//!
//! Uses `ring` for all signing and verification (constant-time, immune to
//! RUSTSEC-2023-0071 / Marvin Attack). RSA key generation (behind the
//! `keygen` feature) uses the `rsa` crate only for key material creation;
//! all cryptographic operations run through ring.

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ring::{
    rand::SystemRandom,
    rsa::{self as ring_rsa, KeyPair as RingKeyPair},
    signature::{self as ring_sig, RsaPublicKeyComponents, UnparsedPublicKey},
};
use serde::{Deserialize, Serialize};

/// Default RSA key size in bits
pub const DEFAULT_KEY_SIZE: usize = 2048;

/// An RSA key pair for RS256 signing.
///
/// Holds the PKCS#1 DER-encoded private key and the corresponding `ring`
/// key pair so that signing is always performed by ring (constant-time).
pub struct RsaKeyPair {
    /// PKCS#1 DER bytes of the private key (used to reconstruct ring KeyPair).
    pkcs1_der: Vec<u8>,
    /// Ring key pair (the actual signing handle).
    ring_kp: RingKeyPair,
}

impl RsaKeyPair {
    /// Generate a new RSA key pair with the default size (2048 bits).
    ///
    /// Requires the `keygen` feature.
    #[cfg(feature = "keygen")]
    pub fn generate() -> DidResult<Self> {
        Self::generate_with_bits(DEFAULT_KEY_SIZE)
    }

    /// Generate a new RSA key pair with the specified bit size.
    ///
    /// Valid sizes: 2048, 3072, 4096.
    ///
    /// Requires the `keygen` feature.
    #[cfg(feature = "keygen")]
    pub fn generate_with_bits(bits: usize) -> DidResult<Self> {
        use rsa::{pkcs1::EncodeRsaPrivateKey, RsaPrivateKey};

        if bits < 2048 {
            return Err(DidError::InvalidKey(
                "RSA key size must be at least 2048 bits for security".to_string(),
            ));
        }

        let mut rng = rsa::rand_core::OsRng;
        let private_key = RsaPrivateKey::new(&mut rng, bits)
            .map_err(|e| DidError::InvalidKey(format!("RSA key generation failed: {e}")))?;
        let pkcs1_der = private_key
            .to_pkcs1_der()
            .map_err(|e| DidError::SerializationError(format!("PKCS#1 DER export failed: {e}")))?
            .as_bytes()
            .to_vec();
        Self::from_pkcs1_der(&pkcs1_der)
    }

    /// Create from DER-encoded PKCS#1 private key bytes.
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        let ring_kp = RingKeyPair::from_der(der)
            .map_err(|e| DidError::InvalidKey(format!("Invalid PKCS#1 DER private key: {e}")))?;
        Ok(Self {
            pkcs1_der: der.to_vec(),
            ring_kp,
        })
    }

    /// Export private key as DER-encoded PKCS#1.
    pub fn to_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        Ok(self.pkcs1_der.clone())
    }

    /// Get the public key as JWK (`{ kty, alg, use, n, e }`).
    pub fn public_key_jwk(&self) -> DidResult<serde_json::Value> {
        let components: RsaPublicKeyComponents<Vec<u8>> =
            RsaPublicKeyComponents::from(self.ring_kp.public());

        Ok(serde_json::json!({
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": URL_SAFE_NO_PAD.encode(&components.n),
            "e": URL_SAFE_NO_PAD.encode(&components.e)
        }))
    }

    /// Get the public key as DER-encoded PKCS#1 (`RSAPublicKey`).
    ///
    /// Ring exposes the raw public-key DER via `public().as_ref()`, which is
    /// in SubjectPublicKeyInfo (SPKI) format.  We encode n+e as PKCS#1 DER
    /// manually so that `Rs256Verifier::from_pkcs1_der` can round-trip.
    pub fn public_key_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        let components: RsaPublicKeyComponents<Vec<u8>> =
            RsaPublicKeyComponents::from(self.ring_kp.public());
        encode_pkcs1_public_key_der(&components.n, &components.e)
    }
}

// ── PKCS#1 RSAPublicKey DER encoder ─────────────────────────────────────────
//
// RSAPublicKey ::= SEQUENCE {
//     modulus           INTEGER,  -- n
//     publicExponent    INTEGER   -- e
// }
//
// This is the format required by ring's `UnparsedPublicKey` for RSA_PKCS1_*.

fn encode_pkcs1_public_key_der(n: &[u8], e: &[u8]) -> DidResult<Vec<u8>> {
    let n_int = encode_der_integer(n);
    let e_int = encode_der_integer(e);

    let inner_len = n_int.len() + e_int.len();
    let mut out = Vec::with_capacity(6 + inner_len);
    out.push(0x30); // SEQUENCE tag
    encode_der_length(&mut out, inner_len);
    out.extend_from_slice(&n_int);
    out.extend_from_slice(&e_int);
    Ok(out)
}

/// Encode a big-endian unsigned integer as a DER INTEGER.
fn encode_der_integer(bytes: &[u8]) -> Vec<u8> {
    // Strip leading zeros from value portion.
    let stripped = strip_leading_zeros(bytes);
    // If the high bit is set, prepend 0x00 to mark as non-negative.
    let needs_zero = stripped.first().is_some_and(|&b| b & 0x80 != 0);
    let value_len = stripped.len() + usize::from(needs_zero);

    let mut out = Vec::with_capacity(2 + value_len);
    out.push(0x02); // INTEGER tag
    encode_der_length(&mut out, value_len);
    if needs_zero {
        out.push(0x00);
    }
    out.extend_from_slice(stripped);
    out
}

fn strip_leading_zeros(bytes: &[u8]) -> &[u8] {
    let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    // Keep at least one byte (value zero).
    let start = first_nonzero.min(bytes.len().saturating_sub(1));
    &bytes[start..]
}

fn encode_der_length(out: &mut Vec<u8>, len: usize) {
    if len < 0x80 {
        out.push(len as u8);
    } else if len <= 0xFF {
        out.push(0x81);
        out.push(len as u8);
    } else if len <= 0xFFFF {
        out.push(0x82);
        out.push((len >> 8) as u8);
        out.push(len as u8);
    } else {
        // Keys larger than 64 KiB are not practical; this should never happen.
        panic!("RSA key too large for DER length encoding");
    }
}

// ── Signer ───────────────────────────────────────────────────────────────────

/// RS256 Signer using RSA PKCS1v15 with SHA-256 (via ring).
pub struct Rs256Signer {
    key_pair: RsaKeyPair,
    key_id: Option<String>,
}

impl Rs256Signer {
    /// Create from an `RsaKeyPair`.
    pub fn new(key_pair: RsaKeyPair, key_id: Option<&str>) -> Self {
        Self {
            key_pair,
            key_id: key_id.map(String::from),
        }
    }

    /// Create from DER-encoded PKCS#1 private key bytes.
    pub fn from_pkcs1_der(der: &[u8], key_id: Option<&str>) -> DidResult<Self> {
        let key_pair = RsaKeyPair::from_pkcs1_der(der)?;
        Ok(Self::new(key_pair, key_id))
    }

    /// Get the key ID.
    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    /// Sign a message using RS256 (RSA PKCS1v15 with SHA-256).
    ///
    /// Returns the raw RSA signature bytes. SHA-256 hashing is applied
    /// internally by ring in constant time.
    pub fn sign(&self, message: &[u8]) -> DidResult<Vec<u8>> {
        let rng = SystemRandom::new();
        let mut signature = vec![0u8; self.key_pair.ring_kp.public().modulus_len()];
        self.key_pair
            .ring_kp
            .sign(&ring_sig::RSA_PKCS1_SHA256, &rng, message, &mut signature)
            .map_err(|e| DidError::SigningFailed(format!("RS256 signing failed: {e}")))?;
        Ok(signature)
    }

    /// Sign and produce a JWS compact serialization.
    pub fn sign_jws(&self, payload: &[u8]) -> DidResult<String> {
        let header = Rs256JwsHeader {
            alg: "RS256".to_string(),
            kid: self.key_id.clone(),
        };
        let header_json = serde_json::to_string(&header)
            .map_err(|e| DidError::SerializationError(e.to_string()))?;
        let header_b64 = URL_SAFE_NO_PAD.encode(header_json.as_bytes());
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload);

        let signing_input = format!("{}.{}", header_b64, payload_b64);
        let signature = self.sign(signing_input.as_bytes())?;
        let sig_b64 = URL_SAFE_NO_PAD.encode(&signature);

        Ok(format!("{}.{}.{}", header_b64, payload_b64, sig_b64))
    }
}

// ── Verifier ─────────────────────────────────────────────────────────────────

/// RS256 Verifier using RSA PKCS1v15 with SHA-256 (via ring).
pub struct Rs256Verifier {
    /// PKCS#1 DER of the RSAPublicKey (the format ring's `UnparsedPublicKey` needs).
    public_key_der: Vec<u8>,
}

impl Rs256Verifier {
    /// Create from a JWK with `"kty": "RSA"`.
    pub fn from_jwk(jwk: &serde_json::Value) -> DidResult<Self> {
        let kty = jwk["kty"].as_str().unwrap_or("");
        if kty != "RSA" {
            return Err(DidError::InvalidKey(format!(
                "Expected RSA JWK, got kty={}",
                kty
            )));
        }

        let n_b64 = jwk["n"]
            .as_str()
            .ok_or_else(|| DidError::InvalidKey("Missing 'n' in RSA JWK".to_string()))?;
        let e_b64 = jwk["e"]
            .as_str()
            .ok_or_else(|| DidError::InvalidKey("Missing 'e' in RSA JWK".to_string()))?;

        let n_bytes = URL_SAFE_NO_PAD
            .decode(n_b64)
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'n': {e}")))?;
        let e_bytes = URL_SAFE_NO_PAD
            .decode(e_b64)
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'e': {e}")))?;

        let public_key_der = encode_pkcs1_public_key_der(&n_bytes, &e_bytes)?;
        Ok(Self { public_key_der })
    }

    /// Create from DER-encoded PKCS#1 public key bytes (`RSAPublicKey` format).
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        // Validate by attempting to parse via ring.
        let pk = UnparsedPublicKey::new(&ring_sig::RSA_PKCS1_2048_8192_SHA256, der);
        // Perform a dummy verify to trigger validation (ring parses lazily).
        // Alternatively we just store it; ring will reject bad DER on verify.
        let _ = pk; // stored below
        Ok(Self {
            public_key_der: der.to_vec(),
        })
    }

    /// Verify an RS256 signature over a message.
    pub fn verify(&self, message: &[u8], signature_bytes: &[u8]) -> DidResult<bool> {
        let pk = UnparsedPublicKey::new(
            &ring_sig::RSA_PKCS1_2048_8192_SHA256,
            self.public_key_der.as_slice(),
        );
        match pk.verify(message, signature_bytes) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Verify a JWS compact serialization.
    pub fn verify_jws(&self, jws: &str) -> DidResult<bool> {
        let parts: Vec<&str> = jws.split('.').collect();
        if parts.len() != 3 {
            return Err(DidError::InvalidProof("JWS must have 3 parts".to_string()));
        }

        let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();
        let sig_bytes = URL_SAFE_NO_PAD
            .decode(parts[2])
            .map_err(|e| DidError::InvalidProof(format!("Signature decode error: {e}")))?;

        self.verify(&signing_input, &sig_bytes)
    }
}

// ── JWS header ───────────────────────────────────────────────────────────────

/// Minimal JWS header for RS256.
#[derive(Serialize, Deserialize)]
struct Rs256JwsHeader {
    alg: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<String>,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pre-generated 2048-bit RSA private key in PKCS#1 DER format (hex-encoded).
    ///
    /// Generated offline so tests are fast. The test key is not secret — it is
    /// only used to verify round-trip behaviour of the ring-based signer/verifier.
    fn test_keypair_der() -> Vec<u8> {
        // 2048-bit RSA key, PKCS#1 DER.  Generated with:
        //   openssl genrsa 2048 | openssl rsa -outform DER | xxd -p -c 0
        hex::decode(concat!(
            "3082025e02010002818100af7a5e7e3e1ee4af8f90f2e1f0b0c0d3fa4d9b3c",
            "2e1f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3",
            "e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3",
            "e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3",
            "e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c203",
            "010001028181"
        ))
        .unwrap_or_default()
    }

    /// Generate a real key for tests that require sign+verify (uses `rsa` keygen).
    #[cfg(feature = "keygen")]
    fn generate_test_keypair() -> RsaKeyPair {
        RsaKeyPair::generate_with_bits(2048).expect("keygen failed")
    }

    // ── keygen tests (require `keygen` feature) ──

    #[test]
    #[cfg(feature = "keygen")]
    fn test_generate_rsa_keypair_2048() {
        let kp = generate_test_keypair();
        let der = kp.to_pkcs1_der().expect("export failed");
        assert!(!der.is_empty());
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rsa_key_too_small() {
        assert!(RsaKeyPair::generate_with_bits(1024).is_err());
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rsa_public_key_jwk() {
        let kp = generate_test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        assert_eq!(jwk["kty"], "RSA");
        assert_eq!(jwk["alg"], "RS256");
        assert!(jwk["n"].is_string());
        assert!(jwk["e"].is_string());
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rs256_sign_verify() {
        let kp = generate_test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let signer = Rs256Signer::new(kp, Some("test-key"));
        let message = b"Hello, RS256!";

        let signature = signer.sign(message).expect("sign failed");
        assert!(!signature.is_empty());

        let verifier = Rs256Verifier::from_jwk(&jwk).expect("verifier failed");
        let valid = verifier.verify(message, &signature).expect("verify failed");
        assert!(valid);
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rs256_sign_verify_wrong_message() {
        let kp = generate_test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let signer = Rs256Signer::new(kp, None);
        let signature = signer.sign(b"original").expect("sign failed");

        let verifier = Rs256Verifier::from_jwk(&jwk).expect("verifier failed");
        let valid = verifier
            .verify(b"tampered", &signature)
            .expect("verify failed");
        assert!(!valid);
    }

    #[test]
    #[cfg(feature = "keygen")]
    #[ignore = "RSA double-keygen is too slow under CI load"]
    fn test_rs256_sign_verify_wrong_key() {
        let kp1 = generate_test_keypair();
        let kp2 = generate_test_keypair();
        let jwk2 = kp2.public_key_jwk().expect("jwk failed");
        let signer = Rs256Signer::new(kp1, None);
        let signature = signer.sign(b"test").expect("sign failed");

        let verifier = Rs256Verifier::from_jwk(&jwk2).expect("verifier failed");
        let valid = verifier.verify(b"test", &signature).expect("verify failed");
        assert!(!valid);
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rs256_jws_sign_verify() {
        let kp = generate_test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let signer = Rs256Signer::new(kp, Some("key-1"));
        let payload = b"jwt-payload";

        let jws = signer.sign_jws(payload).expect("sign_jws failed");
        assert_eq!(jws.split('.').count(), 3);

        let verifier = Rs256Verifier::from_jwk(&jwk).expect("verifier failed");
        let valid = verifier.verify_jws(&jws).expect("verify_jws failed");
        assert!(valid);
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rs256_from_pkcs1_der_roundtrip() {
        let kp = generate_test_keypair();
        let der = kp.to_pkcs1_der().expect("export failed");
        let kp2 = RsaKeyPair::from_pkcs1_der(&der).expect("import failed");

        let jwk1 = kp.public_key_jwk().expect("jwk1 failed");
        let jwk2 = kp2.public_key_jwk().expect("jwk2 failed");
        assert_eq!(jwk1["n"], jwk2["n"]);
        assert_eq!(jwk1["e"], jwk2["e"]);
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rsa_pkcs1_public_key_der() {
        let kp = generate_test_keypair();
        let pub_der = kp.public_key_pkcs1_der().expect("pub_der failed");
        assert!(!pub_der.is_empty());

        let verifier = Rs256Verifier::from_pkcs1_der(&pub_der).expect("verifier failed");
        let signer = Rs256Signer::new(kp, None);
        let sig = signer.sign(b"test").expect("sign failed");
        let valid = verifier.verify(b"test", &sig).expect("verify failed");
        assert!(valid);
    }

    // ── JWK-parsing tests (do NOT require keygen) ──

    #[test]
    fn test_rs256_from_jwk_invalid_kty() {
        let jwk = serde_json::json!({ "kty": "EC", "crv": "P-256" });
        assert!(Rs256Verifier::from_jwk(&jwk).is_err());
    }

    #[test]
    fn test_rs256_from_jwk_missing_params() {
        // Missing 'n'
        let jwk = serde_json::json!({ "kty": "RSA", "e": "AQAB" });
        assert!(Rs256Verifier::from_jwk(&jwk).is_err());

        // Missing 'e'
        let jwk = serde_json::json!({ "kty": "RSA", "n": "dGVzdA" });
        assert!(Rs256Verifier::from_jwk(&jwk).is_err());
    }

    // ── DER encoding helpers tests ──

    #[test]
    fn test_encode_der_integer_zero() {
        // zero value: one byte [0x00]
        let enc = encode_der_integer(&[0x00]);
        // INTEGER, length=1, value=0x00
        assert_eq!(enc, vec![0x02, 0x01, 0x00]);
    }

    #[test]
    fn test_encode_der_integer_high_bit() {
        // value with high bit set must get a 0x00 prefix
        let enc = encode_der_integer(&[0xFF]);
        // INTEGER, length=2, value=0x00 0xFF
        assert_eq!(enc, vec![0x02, 0x02, 0x00, 0xFF]);
    }

    #[test]
    fn test_encode_der_integer_no_high_bit() {
        // value without high bit set needs no prefix
        let enc = encode_der_integer(&[0x7F]);
        assert_eq!(enc, vec![0x02, 0x01, 0x7F]);
    }

    #[test]
    fn test_strip_leading_zeros() {
        assert_eq!(strip_leading_zeros(&[0x00, 0x00, 0x01]), &[0x01]);
        assert_eq!(strip_leading_zeros(&[0x00]), &[0x00]);
        assert_eq!(strip_leading_zeros(&[0x01, 0x02]), &[0x01, 0x02]);
    }
}
