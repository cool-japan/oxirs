//! RS256 (RSASSA-PKCS1-v1_5 with SHA-256) signature implementation
//!
//! RS256 is defined in RFC 7518 and uses RSA with PKCS#1 v1.5 padding
//! and SHA-256 as the hash function.
//!
//! This provides interoperability with legacy systems that require RSA signatures.
//! For new systems, EdDSA or ES256 are preferred.
//!
//! Implemented entirely in pure Rust with the [`rsa`] crate (RSASSA-PKCS1-v1_5)
//! plus [`sha2`] for hashing. Signing and verification go through the
//! `signature` traits re-exported by `rsa`. RSA key generation (behind the
//! `keygen` feature) likewise uses the `rsa` crate.
//!
//! Key material is exchanged in PKCS#1 DER:
//! * private keys as `RSAPrivateKey` (PKCS#1) DER, and
//! * public keys as `RSAPublicKey` (PKCS#1, `SEQUENCE { n, e }`) DER,
//!
//! preserving the on-the-wire encoding used by external JWT/JWS peers.

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rsa::pkcs1::{DecodeRsaPrivateKey, DecodeRsaPublicKey};
use rsa::pkcs1v15::{Signature as Pkcs1v15Signature, SigningKey, VerifyingKey};
use rsa::signature::{SignatureEncoding, Signer, Verifier};
use rsa::traits::PublicKeyParts;
use rsa::{BigUint, RsaPrivateKey, RsaPublicKey};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

/// Default RSA key size in bits
pub const DEFAULT_KEY_SIZE: usize = 2048;

/// An RSA key pair for RS256 signing.
///
/// Holds the parsed [`RsaPrivateKey`] used for signing together with the
/// PKCS#1 DER bytes so the private key can be re-exported losslessly.
pub struct RsaKeyPair {
    /// PKCS#1 DER bytes of the private key (preserved for lossless export).
    pkcs1_der: Vec<u8>,
    /// Parsed private key (the actual signing handle, pure Rust).
    private_key: RsaPrivateKey,
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
        use rsa::pkcs1::EncodeRsaPrivateKey;

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
        Ok(Self {
            pkcs1_der,
            private_key,
        })
    }

    /// Create from DER-encoded PKCS#1 private key bytes.
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        let private_key = RsaPrivateKey::from_pkcs1_der(der)
            .map_err(|e| DidError::InvalidKey(format!("Invalid PKCS#1 DER private key: {e}")))?;
        Ok(Self {
            pkcs1_der: der.to_vec(),
            private_key,
        })
    }

    /// Export private key as DER-encoded PKCS#1.
    pub fn to_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        Ok(self.pkcs1_der.clone())
    }

    /// The corresponding RSA public key.
    fn public_key(&self) -> RsaPublicKey {
        RsaPublicKey::from(&self.private_key)
    }

    /// Get the public key as JWK (`{ kty, alg, use, n, e }`).
    pub fn public_key_jwk(&self) -> DidResult<serde_json::Value> {
        let public = self.public_key();
        let n = public.n().to_bytes_be();
        let e = public.e().to_bytes_be();

        Ok(serde_json::json!({
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": URL_SAFE_NO_PAD.encode(&n),
            "e": URL_SAFE_NO_PAD.encode(&e)
        }))
    }

    /// Get the public key as DER-encoded PKCS#1 (`RSAPublicKey`).
    ///
    /// Encodes `n` and `e` as a PKCS#1 `RSAPublicKey` SEQUENCE so that
    /// [`Rs256Verifier::from_pkcs1_der`] can round-trip.
    pub fn public_key_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        let public = self.public_key();
        let n = public.n().to_bytes_be();
        let e = public.e().to_bytes_be();
        encode_pkcs1_public_key_der(&n, &e)
    }
}

// ── PKCS#1 RSAPublicKey DER encoder ─────────────────────────────────────────
//
// RSAPublicKey ::= SEQUENCE {
//     modulus           INTEGER,  -- n
//     publicExponent    INTEGER   -- e
// }
//
// This is the canonical PKCS#1 public-key format consumed by
// `RsaPublicKey::from_pkcs1_der` and by external RSA peers.

fn encode_pkcs1_public_key_der(n: &[u8], e: &[u8]) -> DidResult<Vec<u8>> {
    let n_int = encode_der_integer(n)?;
    let e_int = encode_der_integer(e)?;

    let inner_len = n_int.len() + e_int.len();
    let mut out = Vec::with_capacity(6 + inner_len);
    out.push(0x30); // SEQUENCE tag
    encode_der_length(&mut out, inner_len)?;
    out.extend_from_slice(&n_int);
    out.extend_from_slice(&e_int);
    Ok(out)
}

/// Encode a big-endian unsigned integer as a DER INTEGER.
fn encode_der_integer(bytes: &[u8]) -> DidResult<Vec<u8>> {
    // Strip leading zeros from value portion.
    let stripped = strip_leading_zeros(bytes);
    // If the high bit is set, prepend 0x00 to mark as non-negative.
    let needs_zero = stripped.first().is_some_and(|&b| b & 0x80 != 0);
    let value_len = stripped.len() + usize::from(needs_zero);

    let mut out = Vec::with_capacity(2 + value_len);
    out.push(0x02); // INTEGER tag
    encode_der_length(&mut out, value_len)?;
    if needs_zero {
        out.push(0x00);
    }
    out.extend_from_slice(stripped);
    Ok(out)
}

fn strip_leading_zeros(bytes: &[u8]) -> &[u8] {
    let first_nonzero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    // Keep at least one byte (value zero).
    let start = first_nonzero.min(bytes.len().saturating_sub(1));
    &bytes[start..]
}

fn encode_der_length(out: &mut Vec<u8>, len: usize) -> DidResult<()> {
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
        // Keys larger than 64 KiB are not practical.
        return Err(DidError::SerializationError(
            "RSA key too large for DER length encoding".to_string(),
        ));
    }
    Ok(())
}

// ── Signer ───────────────────────────────────────────────────────────────────

/// RS256 Signer using RSA PKCS#1 v1.5 with SHA-256 (pure Rust, `rsa` crate).
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

    /// Sign a message using RS256 (RSA PKCS#1 v1.5 with SHA-256).
    ///
    /// Returns the raw big-endian RSA signature bytes. SHA-256 hashing is
    /// applied internally by the `rsa` signer. RSASSA-PKCS1-v1_5 is
    /// deterministic, so no randomness is required.
    pub fn sign(&self, message: &[u8]) -> DidResult<Vec<u8>> {
        let signing_key = SigningKey::<Sha256>::new(self.key_pair.private_key.clone());
        let signature: Pkcs1v15Signature = signing_key
            .try_sign(message)
            .map_err(|e| DidError::SigningFailed(format!("RS256 signing failed: {e}")))?;
        Ok(signature.to_vec())
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

/// RS256 Verifier using RSA PKCS#1 v1.5 with SHA-256 (pure Rust, `rsa` crate).
pub struct Rs256Verifier {
    /// Parsed RSA public key.
    public_key: RsaPublicKey,
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

        let n = BigUint::from_bytes_be(&n_bytes);
        let e = BigUint::from_bytes_be(&e_bytes);
        let public_key = RsaPublicKey::new(n, e)
            .map_err(|e| DidError::InvalidKey(format!("Invalid RSA public key components: {e}")))?;
        Ok(Self { public_key })
    }

    /// Create from DER-encoded PKCS#1 public key bytes (`RSAPublicKey` format).
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        let public_key = RsaPublicKey::from_pkcs1_der(der)
            .map_err(|e| DidError::InvalidKey(format!("Invalid PKCS#1 DER public key: {e}")))?;
        Ok(Self { public_key })
    }

    /// Verify an RS256 signature over a message.
    pub fn verify(&self, message: &[u8], signature_bytes: &[u8]) -> DidResult<bool> {
        let signature = match Pkcs1v15Signature::try_from(signature_bytes) {
            Ok(sig) => sig,
            Err(_) => return Ok(false),
        };
        let verifying_key = VerifyingKey::<Sha256>::new(self.public_key.clone());
        match verifying_key.verify(message, &signature) {
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

    /// A fixed, valid 2048-bit RSA private key in PKCS#1 DER form (hex), so the
    /// sign/verify round-trips run fast and deterministically without invoking
    /// RSA key generation (which is slow and otherwise feature-gated).
    ///
    /// Generated offline with the `rsa` crate. **Not secret** — it exists only
    /// to exercise the pure-Rust RSASSA-PKCS1-v1_5 path across this migration.
    const TEST_RSA_PKCS1_DER_HEX: &str = concat!(
        "308204a50201000282010100b574a22ec0d6ea3d23ee2d555d4adaac9d006138",
        "0072d7cec1fa8eb0fbfc4c60e2c1c651fd3268d343ef51bb863f82678d6ffdc18",
        "517c8ebdfce92b3dae6601c2acb012f1fb4c867af73c1143bb65fbe00d1665f1e",
        "6951cac38a7d2f5aef5cbbc444250d75a32bd280d3f4fc1524e08c1403cd13eac",
        "a7420f294b8bc69c285f9cf6045be16b61582721e9cd35ff154f108b99d99427",
        "dcef4fb96d0425b72a34a67c3e394cf500cc680231d0799138930270e9af3a43",
        "5f534a1590932b350045fd2be2fe5edfaa4303e5e61474dcdbbdfda72666dc5f0",
        "6df24270a0f4b0fa473bfc5bff551dfa8a7396e7eaebbd86e2ed6bc78b01d7f50",
        "b8532f025e52f181c3f02030100010282010100aabd93ba108469a69c3f8a72af",
        "b536ac83930ee7a62c69fac8361ebc546fa3e2ea9bd123e6eedf0a23fb75d9d14",
        "9c347f323750ffa4f5624f4d428e089d28a8f16892c950ded8b415d2bbb7b717",
        "07b088b367e70746a3fa75e75dab38b8a7da4c4da264f52e8f5dc3e92b30bdc4d",
        "75e8f91056912e35d02e0c747a9bf79c2d793a926106a555135dde1a1d8d9447",
        "3618d23ae4076c80b148fa2be89275bee58f9fa519c090d85188d8e480eaf348",
        "d85322c33e021b3a54cbd65d7c6ab50fb47db9e2ef3444185c4f5fc1a10e28f7a",
        "8645d8d832f82b34ca6c929d1ed2f27b52c29cae964a9b840e7f4f4237210c2d6",
        "6a748a06dffdc8ee57003c0aa7d465100902818100e948a3994b9f54b7978bd5f7",
        "df9c0f292a525eb9921f229099494ff96766c39e3b7375cfb7d9e754fa5b5fecd",
        "0326882a025b2c3700dfaaf64ea12afa2963d7699107b663402cd6246c5945c71",
        "a702afa9e370a856ca242de71bda4111cace5ff85c25d0720e46eef76c2a2f4d4",
        "c5f31ace935ef9550e25e9ad379b0707a84eb02818100c720021e41f47d41319c",
        "34b4fe88cc1d0b5bf72224265fbc2ca5b49936ba4a604b547434b188cf1eba092",
        "3b71d55bb56ff0f65189f8a7a21b0d08a73bb84989f36beb562f09c05bfb723a1",
        "467f0de7becbde5384357f2778f82a750854500134cabc54ea036aaa3ab7ab2d0",
        "e4d3ee7ecab4f403a38f09e9135dbb1401c5840fd02818100a55d105b01f8179d",
        "6c977b3d120d15d22065f32fa81cd9eb963d19abb95867222b125558c1027db10",
        "726ee8077b4c3d094a5246ad56b935ff130dbbe148a5b574e791880022e5a5867",
        "c276c070efea3f8b35e60dee37ac1dbdcd48910783b376e41eadcf6c8a224e12a",
        "561e1d5f165295960971315cd1829e7a6308499b6bfb10281803751c624dfcfeb",
        "88541f006994192f1396974b161a12eb1fdd0b801bdc9f6e9047fd43776c2704d",
        "b95757a8da1c1b2951db10a360804b19f707ecf280ddd6d8535f14f751841503b",
        "8dc681449381aa73503208e3bef4ff6355167e82fce16924607a67e0c76837b8a",
        "e5408e5634269ae4f7d69eb82ec1e315b54186f2630c28502818100e71d44d3c8",
        "6df4ec057ad49cf51a67414ac910c02be44a18ed5ffd57d81ec9bf3c6e8104d56",
        "11b49238385f7da7a21dd255279a03fdaa11df0dd690df3fb70ee8f9ca810e2cf",
        "cb45e74691732372a68e9cc7a17c9d2f04ba82975b316e9834817536e03d6f076",
        "8e5d61e47420aeaf3763a8fe91bd8214e9ab0e6fc1f61732eb7",
    );

    /// Decode the fixed test key pair from the baked PKCS#1 DER vector.
    fn test_keypair() -> RsaKeyPair {
        let der = hex::decode(TEST_RSA_PKCS1_DER_HEX).expect("test key hex is valid");
        RsaKeyPair::from_pkcs1_der(&der).expect("test key DER is valid")
    }

    // ── keygen tests (require `keygen` feature) ──

    #[test]
    #[cfg(feature = "keygen")]
    fn test_generate_rsa_keypair_2048() {
        let kp = RsaKeyPair::generate_with_bits(2048).expect("keygen failed");
        let der = kp.to_pkcs1_der().expect("export failed");
        assert!(!der.is_empty());
    }

    #[test]
    #[cfg(feature = "keygen")]
    fn test_rsa_key_too_small() {
        assert!(RsaKeyPair::generate_with_bits(1024).is_err());
    }

    // ── sign/verify tests (pure-Rust, no keygen needed) ──

    #[test]
    fn test_rsa_public_key_jwk() {
        let kp = test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        assert_eq!(jwk["kty"], "RSA");
        assert_eq!(jwk["alg"], "RS256");
        assert!(jwk["n"].is_string());
        assert!(jwk["e"].is_string());
    }

    /// Core migration proof: sign with the pure-Rust `rsa` signer and verify
    /// through the migrated verifier (JWK path) — a full round-trip.
    #[test]
    fn test_rs256_sign_verify() {
        let kp = test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let signer = Rs256Signer::new(kp, Some("test-key"));
        let message = b"Hello, RS256!";

        let signature = signer.sign(message).expect("sign failed");
        assert!(!signature.is_empty());

        let verifier = Rs256Verifier::from_jwk(&jwk).expect("verifier failed");
        let valid = verifier.verify(message, &signature).expect("verify failed");
        assert!(valid, "RSASSA-PKCS1-v1_5 signature must verify");
    }

    /// RSASSA-PKCS1-v1_5 is deterministic: signing the same message twice with
    /// the same key yields identical bytes (a stable known-answer property).
    #[test]
    fn test_rs256_signature_is_deterministic() {
        let kp = test_keypair();
        let signer = Rs256Signer::new(kp, None);
        let sig1 = signer.sign(b"deterministic").expect("sign failed");
        let sig2 = signer.sign(b"deterministic").expect("sign failed");
        assert_eq!(sig1, sig2, "PKCS#1 v1.5 signatures must be deterministic");
        // 2048-bit modulus → 256-byte signature.
        assert_eq!(sig1.len(), 256);
    }

    #[test]
    fn test_rs256_sign_verify_wrong_message() {
        let kp = test_keypair();
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
    fn test_rs256_verify_rejects_garbage_signature() {
        let kp = test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let verifier = Rs256Verifier::from_jwk(&jwk).expect("verifier failed");
        // Too-short signature must be rejected gracefully (not error out).
        assert!(!verifier.verify(b"msg", &[0u8; 4]).expect("verify failed"));
    }

    #[test]
    fn test_rs256_jws_sign_verify() {
        let kp = test_keypair();
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
    fn test_rs256_from_pkcs1_der_roundtrip() {
        let kp = test_keypair();
        let der = kp.to_pkcs1_der().expect("export failed");
        let kp2 = RsaKeyPair::from_pkcs1_der(&der).expect("import failed");

        let jwk1 = kp.public_key_jwk().expect("jwk1 failed");
        let jwk2 = kp2.public_key_jwk().expect("jwk2 failed");
        assert_eq!(jwk1["n"], jwk2["n"]);
        assert_eq!(jwk1["e"], jwk2["e"]);
    }

    /// Public-key PKCS#1 DER export must round-trip through the verifier's
    /// `from_pkcs1_der`, proving the preserved on-wire public-key encoding.
    #[test]
    fn test_rsa_pkcs1_public_key_der() {
        let kp = test_keypair();
        let pub_der = kp.public_key_pkcs1_der().expect("pub_der failed");
        assert!(!pub_der.is_empty());

        let verifier = Rs256Verifier::from_pkcs1_der(&pub_der).expect("verifier failed");
        let signer = Rs256Signer::new(kp, None);
        let sig = signer.sign(b"test").expect("sign failed");
        let valid = verifier.verify(b"test", &sig).expect("verify failed");
        assert!(valid);
    }

    /// Interop check: a signature produced by the signer verifies against a
    /// verifier built from the JWK, AND against one built from the PKCS#1 DER
    /// public key — both encodings describe the same key.
    #[test]
    fn test_rs256_jwk_and_der_verifiers_agree() {
        let kp = test_keypair();
        let jwk = kp.public_key_jwk().expect("jwk failed");
        let pub_der = kp.public_key_pkcs1_der().expect("pub_der failed");
        let signer = Rs256Signer::new(kp, None);
        let sig = signer.sign(b"interop").expect("sign failed");

        let v_jwk = Rs256Verifier::from_jwk(&jwk).expect("jwk verifier failed");
        let v_der = Rs256Verifier::from_pkcs1_der(&pub_der).expect("der verifier failed");
        assert!(v_jwk.verify(b"interop", &sig).expect("verify failed"));
        assert!(v_der.verify(b"interop", &sig).expect("verify failed"));
    }

    // ── JWK-parsing tests ──

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
        let enc = encode_der_integer(&[0x00]).expect("encode failed");
        // INTEGER, length=1, value=0x00
        assert_eq!(enc, vec![0x02, 0x01, 0x00]);
    }

    #[test]
    fn test_encode_der_integer_high_bit() {
        // value with high bit set must get a 0x00 prefix
        let enc = encode_der_integer(&[0xFF]).expect("encode failed");
        // INTEGER, length=2, value=0x00 0xFF
        assert_eq!(enc, vec![0x02, 0x02, 0x00, 0xFF]);
    }

    #[test]
    fn test_encode_der_integer_no_high_bit() {
        // value without high bit set needs no prefix
        let enc = encode_der_integer(&[0x7F]).expect("encode failed");
        assert_eq!(enc, vec![0x02, 0x01, 0x7F]);
    }

    #[test]
    fn test_strip_leading_zeros() {
        assert_eq!(strip_leading_zeros(&[0x00, 0x00, 0x01]), &[0x01]);
        assert_eq!(strip_leading_zeros(&[0x00]), &[0x00]);
        assert_eq!(strip_leading_zeros(&[0x01, 0x02]), &[0x01, 0x02]);
    }
}
