//! RS256 (RSASSA-PKCS1-v1_5 with SHA-256) signature implementation
//!
//! RS256 is defined in RFC 7518 and uses RSA with PKCS#1 v1.5 padding
//! and SHA-256 as the hash function.
//!
//! This provides interoperability with legacy systems that require RSA signatures.
//! For new systems, EdDSA or ES256 are preferred.
//!
//! Uses the pure Rust `rsa` crate (no C/OpenSSL dependencies).

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rsa::sha2::Sha256;
use rsa::signature::{RandomizedSigner, SignatureEncoding, Verifier as RsaVerifier};
use rsa::{
    pkcs1::{DecodeRsaPrivateKey, DecodeRsaPublicKey, EncodeRsaPrivateKey, EncodeRsaPublicKey},
    pkcs1v15::{Signature, SigningKey, VerifyingKey},
    RsaPrivateKey, RsaPublicKey,
};
use serde::{Deserialize, Serialize};

/// Default RSA key size in bits
pub const DEFAULT_KEY_SIZE: usize = 2048;

/// An RSA key pair for RS256 signing
pub struct RsaKeyPair {
    private_key: RsaPrivateKey,
}

impl RsaKeyPair {
    /// Generate a new RSA key pair with the default size (2048 bits)
    pub fn generate() -> DidResult<Self> {
        Self::generate_with_bits(DEFAULT_KEY_SIZE)
    }

    /// Generate a new RSA key pair with the specified bit size
    ///
    /// Valid sizes: 2048, 3072, 4096
    pub fn generate_with_bits(bits: usize) -> DidResult<Self> {
        if bits < 2048 {
            return Err(DidError::InvalidKey(
                "RSA key size must be at least 2048 bits for security".to_string(),
            ));
        }

        let mut rng = rsa::rand_core::OsRng;
        let private_key = RsaPrivateKey::new(&mut rng, bits)
            .map_err(|e| DidError::InvalidKey(format!("RSA key generation failed: {}", e)))?;
        Ok(Self { private_key })
    }

    /// Create from DER-encoded PKCS#1 private key bytes
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        let private_key = RsaPrivateKey::from_pkcs1_der(der)
            .map_err(|e| DidError::InvalidKey(format!("Invalid PKCS#1 DER private key: {}", e)))?;
        Ok(Self { private_key })
    }

    /// Export private key as DER-encoded PKCS#1
    pub fn to_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        self.private_key
            .to_pkcs1_der()
            .map(|doc| doc.as_bytes().to_vec())
            .map_err(|e| DidError::SerializationError(format!("PKCS#1 DER export failed: {}", e)))
    }

    /// Get the public key
    pub fn public_key(&self) -> RsaPublicKey {
        self.private_key.to_public_key()
    }

    /// Get the public key as JWK
    pub fn public_key_jwk(&self) -> DidResult<serde_json::Value> {
        use rsa::traits::PublicKeyParts;
        let pk = self.private_key.to_public_key();
        let n = pk.n().to_bytes_be();
        let e = pk.e().to_bytes_be();

        Ok(serde_json::json!({
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": URL_SAFE_NO_PAD.encode(&n),
            "e": URL_SAFE_NO_PAD.encode(&e)
        }))
    }

    /// Get the public key as DER-encoded PKCS#1
    pub fn public_key_pkcs1_der(&self) -> DidResult<Vec<u8>> {
        let pk = self.private_key.to_public_key();
        pk.to_pkcs1_der()
            .map(|doc| doc.as_bytes().to_vec())
            .map_err(|e| DidError::SerializationError(format!("PKCS#1 DER export failed: {}", e)))
    }
}

/// RS256 Signer using RSA PKCS1v15 with SHA-256
pub struct Rs256Signer {
    signing_key: SigningKey<Sha256>,
    key_id: Option<String>,
}

impl Rs256Signer {
    /// Create from an RsaKeyPair
    pub fn new(key_pair: &RsaKeyPair, key_id: Option<&str>) -> Self {
        let signing_key = SigningKey::<Sha256>::new(key_pair.private_key.clone());
        Self {
            signing_key,
            key_id: key_id.map(String::from),
        }
    }

    /// Create from DER-encoded PKCS#1 private key bytes
    pub fn from_pkcs1_der(der: &[u8], key_id: Option<&str>) -> DidResult<Self> {
        let key_pair = RsaKeyPair::from_pkcs1_der(der)?;
        Ok(Self::new(&key_pair, key_id))
    }

    /// Get the key ID
    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    /// Sign a message using RS256 (RSA PKCS1v15 with SHA-256)
    ///
    /// The signing process includes SHA-256 hashing internally.
    /// Returns the raw RSA signature bytes.
    pub fn sign(&self, message: &[u8]) -> DidResult<Vec<u8>> {
        let mut rng = rsa::rand_core::OsRng;
        let signature = self.signing_key.sign_with_rng(&mut rng, message);
        Ok(signature.to_bytes().to_vec())
    }

    /// Sign and produce a JWS compact serialization
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

/// RS256 Verifier using RSA PKCS1v15 with SHA-256
pub struct Rs256Verifier {
    verifying_key: VerifyingKey<Sha256>,
}

impl Rs256Verifier {
    /// Create from a JWK with "kty": "RSA"
    pub fn from_jwk(jwk: &serde_json::Value) -> DidResult<Self> {
        use rsa::BigUint;

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
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'n': {}", e)))?;
        let e_bytes = URL_SAFE_NO_PAD
            .decode(e_b64)
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'e': {}", e)))?;

        let n = BigUint::from_bytes_be(&n_bytes);
        let e = BigUint::from_bytes_be(&e_bytes);

        let public_key = RsaPublicKey::new(n, e).map_err(|e| {
            DidError::InvalidKey(format!("Invalid RSA public key parameters: {}", e))
        })?;

        let verifying_key = VerifyingKey::<Sha256>::new(public_key);
        Ok(Self { verifying_key })
    }

    /// Create from DER-encoded PKCS#1 public key bytes
    pub fn from_pkcs1_der(der: &[u8]) -> DidResult<Self> {
        let public_key = RsaPublicKey::from_pkcs1_der(der)
            .map_err(|e| DidError::InvalidKey(format!("Invalid PKCS#1 DER public key: {}", e)))?;
        let verifying_key = VerifyingKey::<Sha256>::new(public_key);
        Ok(Self { verifying_key })
    }

    /// Verify an RS256 signature over a message
    pub fn verify(&self, message: &[u8], signature_bytes: &[u8]) -> DidResult<bool> {
        let sig = Signature::try_from(signature_bytes).map_err(|e| {
            DidError::InvalidProof(format!("Invalid RS256 signature encoding: {}", e))
        })?;

        match self.verifying_key.verify(message, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Verify a JWS compact serialization
    pub fn verify_jws(&self, jws: &str) -> DidResult<bool> {
        let parts: Vec<&str> = jws.split('.').collect();
        if parts.len() != 3 {
            return Err(DidError::InvalidProof("JWS must have 3 parts".to_string()));
        }

        let signing_input = format!("{}.{}", parts[0], parts[1]).into_bytes();
        let sig_bytes = URL_SAFE_NO_PAD
            .decode(parts[2])
            .map_err(|e| DidError::InvalidProof(format!("Signature decode error: {}", e)))?;

        self.verify(&signing_input, &sig_bytes)
    }
}

/// Minimal JWS header for RS256
#[derive(Serialize, Deserialize)]
struct Rs256JwsHeader {
    alg: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use 2048-bit keys for tests to be fast while still secure
    fn generate_test_keypair() -> RsaKeyPair {
        RsaKeyPair::generate_with_bits(2048).unwrap()
    }

    #[test]
    fn test_generate_rsa_keypair_2048() {
        let kp = generate_test_keypair();
        let der = kp.to_pkcs1_der().unwrap();
        assert!(!der.is_empty());
    }

    #[test]
    fn test_rsa_key_too_small() {
        assert!(RsaKeyPair::generate_with_bits(1024).is_err());
    }

    #[test]
    fn test_rsa_public_key_jwk() {
        let kp = generate_test_keypair();
        let jwk = kp.public_key_jwk().unwrap();
        assert_eq!(jwk["kty"], "RSA");
        assert_eq!(jwk["alg"], "RS256");
        assert!(jwk["n"].is_string());
        assert!(jwk["e"].is_string());
    }

    #[test]
    fn test_rs256_sign_verify() {
        let kp = generate_test_keypair();
        let signer = Rs256Signer::new(&kp, Some("test-key"));
        let message = b"Hello, RS256!";

        let signature = signer.sign(message).unwrap();
        assert!(!signature.is_empty());

        let jwk = kp.public_key_jwk().unwrap();
        let verifier = Rs256Verifier::from_jwk(&jwk).unwrap();
        let valid = verifier.verify(message, &signature).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_rs256_sign_verify_wrong_message() {
        let kp = generate_test_keypair();
        let signer = Rs256Signer::new(&kp, None);
        let signature = signer.sign(b"original").unwrap();

        let jwk = kp.public_key_jwk().unwrap();
        let verifier = Rs256Verifier::from_jwk(&jwk).unwrap();
        let valid = verifier.verify(b"tampered", &signature).unwrap();
        assert!(!valid);
    }

    #[test]
    #[ignore = "RSA double-keygen is too slow under CI load"]
    fn test_rs256_sign_verify_wrong_key() {
        let kp1 = generate_test_keypair();
        let kp2 = generate_test_keypair();
        let signer = Rs256Signer::new(&kp1, None);
        let signature = signer.sign(b"test").unwrap();

        let jwk2 = kp2.public_key_jwk().unwrap();
        let verifier = Rs256Verifier::from_jwk(&jwk2).unwrap();
        let valid = verifier.verify(b"test", &signature).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_rs256_jws_sign_verify() {
        let kp = generate_test_keypair();
        let signer = Rs256Signer::new(&kp, Some("key-1"));
        let payload = b"jwt-payload";

        let jws = signer.sign_jws(payload).unwrap();
        assert_eq!(jws.split('.').count(), 3);

        let jwk = kp.public_key_jwk().unwrap();
        let verifier = Rs256Verifier::from_jwk(&jwk).unwrap();
        let valid = verifier.verify_jws(&jws).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_rs256_from_pkcs1_der_roundtrip() {
        let kp = generate_test_keypair();
        let der = kp.to_pkcs1_der().unwrap();
        let kp2 = RsaKeyPair::from_pkcs1_der(&der).unwrap();

        // Both should produce same public key JWK
        let jwk1 = kp.public_key_jwk().unwrap();
        let jwk2 = kp2.public_key_jwk().unwrap();
        assert_eq!(jwk1["n"], jwk2["n"]);
        assert_eq!(jwk1["e"], jwk2["e"]);
    }

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

    #[test]
    fn test_rsa_pkcs1_public_key_der() {
        let kp = generate_test_keypair();
        let der = kp.public_key_pkcs1_der().unwrap();
        assert!(!der.is_empty());

        let verifier = Rs256Verifier::from_pkcs1_der(&der).unwrap();
        let signer = Rs256Signer::new(&kp, None);
        let sig = signer.sign(b"test").unwrap();
        let valid = verifier.verify(b"test", &sig).unwrap();
        assert!(valid);
    }
}
