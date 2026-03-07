//! ES256 (ECDSA with P-256 and SHA-256) signature implementation
//!
//! ES256 is defined in RFC 7518 and uses NIST P-256 (secp256r1) curve
//! with SHA-256 as the hash function.
//!
//! This is commonly used with JWS (JSON Web Signatures) and is required
//! for interoperability with many W3C VC ecosystem components.
//!
//! Uses the pure Rust `p256` crate (no C/OpenSSL dependencies).

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use p256::ecdsa::signature::{Signer as P256Signer, Verifier as P256Verifier};
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use p256::elliptic_curve::rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// A P-256 (secp256r1) key pair for ES256 signing
#[derive(Clone)]
pub struct P256KeyPair {
    signing_key: SigningKey,
}

impl P256KeyPair {
    /// Generate a new random P-256 key pair using OS entropy
    pub fn generate() -> Self {
        let signing_key = SigningKey::random(&mut OsRng);
        Self { signing_key }
    }

    /// Create from raw 32-byte private key scalar
    pub fn from_secret_bytes(bytes: &[u8]) -> DidResult<Self> {
        if bytes.len() != 32 {
            return Err(DidError::InvalidKey(
                "P-256 private key must be 32 bytes".to_string(),
            ));
        }
        let signing_key = SigningKey::from_bytes(bytes.into())
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 key: {}", e)))?;
        Ok(Self { signing_key })
    }

    /// Get the raw 32-byte private key scalar
    pub fn secret_bytes(&self) -> Vec<u8> {
        self.signing_key.to_bytes().to_vec()
    }

    /// Get the compressed public key (33 bytes)
    pub fn public_key_compressed(&self) -> Vec<u8> {
        use p256::elliptic_curve::sec1::ToEncodedPoint;
        let vk = self.signing_key.verifying_key();
        let encoded = vk.to_encoded_point(true);
        encoded.as_bytes().to_vec()
    }

    /// Get the uncompressed public key (65 bytes: 0x04 || x || y)
    pub fn public_key_uncompressed(&self) -> Vec<u8> {
        use p256::elliptic_curve::sec1::ToEncodedPoint;
        let vk = self.signing_key.verifying_key();
        let encoded = vk.to_encoded_point(false);
        encoded.as_bytes().to_vec()
    }

    /// Get the public key as JWK
    pub fn public_key_jwk(&self) -> serde_json::Value {
        let uncompressed = self.public_key_uncompressed();
        // uncompressed = 0x04 || 32-byte x || 32-byte y
        if uncompressed.len() == 65 {
            let x = URL_SAFE_NO_PAD.encode(&uncompressed[1..33]);
            let y = URL_SAFE_NO_PAD.encode(&uncompressed[33..65]);
            serde_json::json!({
                "kty": "EC",
                "crv": "P-256",
                "x": x,
                "y": y
            })
        } else {
            serde_json::json!({ "kty": "EC", "crv": "P-256" })
        }
    }

    /// Get the verifying key
    pub fn verifying_key(&self) -> &VerifyingKey {
        self.signing_key.verifying_key()
    }
}

/// ES256 Signer using P-256
pub struct Es256Signer {
    key_pair: P256KeyPair,
    key_id: Option<String>,
}

impl Es256Signer {
    /// Create from a P-256 key pair
    pub fn new(key_pair: P256KeyPair, key_id: Option<&str>) -> Self {
        Self {
            key_pair,
            key_id: key_id.map(String::from),
        }
    }

    /// Create from raw 32-byte private key
    pub fn from_secret_bytes(bytes: &[u8], key_id: Option<&str>) -> DidResult<Self> {
        let key_pair = P256KeyPair::from_secret_bytes(bytes)?;
        Ok(Self {
            key_pair,
            key_id: key_id.map(String::from),
        })
    }

    /// Get the key ID
    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    /// Sign a message using ES256 (ECDSA with P-256 + SHA-256)
    ///
    /// Returns DER-encoded signature or raw R||S (64 bytes) depending on format
    pub fn sign(&self, message: &[u8]) -> DidResult<Vec<u8>> {
        let signature: Signature = self.key_pair.signing_key.sign(message);
        // Return raw R||S format (64 bytes) for JWS compatibility (RFC 7518)
        Ok(signature.to_bytes().to_vec())
    }

    /// Sign and produce a JWS compact serialization
    pub fn sign_jws(&self, payload: &[u8]) -> DidResult<String> {
        let header = Es256JwsHeader {
            alg: "ES256".to_string(),
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

    /// Get the key pair
    pub fn key_pair(&self) -> &P256KeyPair {
        &self.key_pair
    }
}

/// ES256 Verifier using P-256
pub struct Es256Verifier {
    verifying_key: VerifyingKey,
}

impl Es256Verifier {
    /// Create from compressed public key bytes (33 bytes)
    pub fn from_compressed(bytes: &[u8]) -> DidResult<Self> {
        use p256::elliptic_curve::sec1::FromEncodedPoint;
        use p256::EncodedPoint;

        let encoded = EncodedPoint::from_bytes(bytes)
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 compressed key: {}", e)))?;
        let vk = VerifyingKey::from_encoded_point(&encoded)
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 verifying key: {}", e)))?;
        Ok(Self { verifying_key: vk })
    }

    /// Create from uncompressed public key bytes (65 bytes)
    pub fn from_uncompressed(bytes: &[u8]) -> DidResult<Self> {
        use p256::elliptic_curve::sec1::FromEncodedPoint;
        use p256::EncodedPoint;

        if bytes.len() != 65 || bytes[0] != 0x04 {
            return Err(DidError::InvalidKey(
                "Uncompressed P-256 key must be 65 bytes starting with 0x04".to_string(),
            ));
        }
        let encoded = EncodedPoint::from_bytes(bytes)
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 point: {}", e)))?;
        let vk = VerifyingKey::from_encoded_point(&encoded)
            .map_err(|e| DidError::InvalidKey(format!("Invalid P-256 verifying key: {}", e)))?;
        Ok(Self { verifying_key: vk })
    }

    /// Create from a JWK with "kty": "EC", "crv": "P-256"
    pub fn from_jwk(jwk: &serde_json::Value) -> DidResult<Self> {
        let kty = jwk["kty"].as_str().unwrap_or("");
        let crv = jwk["crv"].as_str().unwrap_or("");

        if kty != "EC" || crv != "P-256" {
            return Err(DidError::InvalidKey(format!(
                "Expected EC/P-256 JWK, got kty={} crv={}",
                kty, crv
            )));
        }

        let x_b64 = jwk["x"]
            .as_str()
            .ok_or_else(|| DidError::InvalidKey("Missing 'x' in JWK".to_string()))?;
        let y_b64 = jwk["y"]
            .as_str()
            .ok_or_else(|| DidError::InvalidKey("Missing 'y' in JWK".to_string()))?;

        let x = URL_SAFE_NO_PAD
            .decode(x_b64)
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'x': {}", e)))?;
        let y = URL_SAFE_NO_PAD
            .decode(y_b64)
            .map_err(|e| DidError::InvalidKey(format!("Invalid 'y': {}", e)))?;

        // Build uncompressed point
        let mut uncompressed = vec![0x04u8];
        uncompressed.extend_from_slice(&x);
        uncompressed.extend_from_slice(&y);

        Self::from_uncompressed(&uncompressed)
    }

    /// Verify an ES256 signature (raw R||S, 64 bytes) over a message
    pub fn verify(&self, message: &[u8], signature_bytes: &[u8]) -> DidResult<bool> {
        if signature_bytes.len() != 64 {
            return Err(DidError::InvalidProof(format!(
                "ES256 signature must be 64 bytes (R||S), got {}",
                signature_bytes.len()
            )));
        }
        let sig = Signature::from_bytes(signature_bytes.into()).map_err(|e| {
            DidError::InvalidProof(format!("Invalid ES256 signature encoding: {}", e))
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
            .map_err(|e| DidError::InvalidProof(format!("Signature base64 decode error: {}", e)))?;

        self.verify(&signing_input, &sig_bytes)
    }
}

/// Minimal JWS header for ES256
#[derive(Serialize, Deserialize)]
struct Es256JwsHeader {
    alg: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_keypair() -> P256KeyPair {
        P256KeyPair::generate()
    }

    #[test]
    fn test_generate_p256_keypair() {
        let kp = generate_keypair();
        let secret = kp.secret_bytes();
        assert_eq!(secret.len(), 32);
        let compressed = kp.public_key_compressed();
        assert_eq!(compressed.len(), 33);
        let uncompressed = kp.public_key_uncompressed();
        assert_eq!(uncompressed.len(), 65);
        assert_eq!(uncompressed[0], 0x04);
    }

    #[test]
    fn test_p256_from_secret_bytes() {
        let kp = generate_keypair();
        let secret = kp.secret_bytes();

        let kp2 = P256KeyPair::from_secret_bytes(&secret).unwrap();
        // Same secret should produce same public key
        assert_eq!(kp.public_key_compressed(), kp2.public_key_compressed());
    }

    #[test]
    fn test_p256_invalid_secret_bytes() {
        assert!(P256KeyPair::from_secret_bytes(&[0u8; 31]).is_err()); // wrong length
        assert!(P256KeyPair::from_secret_bytes(&[0u8; 33]).is_err()); // wrong length
    }

    #[test]
    fn test_es256_sign_verify() {
        let kp = generate_keypair();
        let signer = Es256Signer::new(kp.clone(), Some("test-key"));
        let message = b"Hello, ES256!";

        let signature = signer.sign(message).unwrap();
        assert_eq!(signature.len(), 64); // R||S format

        let verifier = Es256Verifier::from_compressed(&kp.public_key_compressed()).unwrap();
        let valid = verifier.verify(message, &signature).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_es256_sign_verify_wrong_message() {
        let kp = generate_keypair();
        let signer = Es256Signer::new(kp.clone(), None);
        let message = b"Original message";
        let signature = signer.sign(message).unwrap();

        let verifier = Es256Verifier::from_compressed(&kp.public_key_compressed()).unwrap();
        let valid = verifier.verify(b"Wrong message", &signature).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_es256_sign_verify_wrong_key() {
        let kp1 = generate_keypair();
        let kp2 = generate_keypair();
        let signer = Es256Signer::new(kp1, None);
        let message = b"Test message";
        let signature = signer.sign(message).unwrap();

        let verifier = Es256Verifier::from_compressed(&kp2.public_key_compressed()).unwrap();
        let valid = verifier.verify(message, &signature).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_es256_jws_sign_verify() {
        let kp = generate_keypair();
        let signer = Es256Signer::new(kp.clone(), Some("key-1"));
        let payload = b"jwt-payload-data";

        let jws = signer.sign_jws(payload).unwrap();
        // JWS must have 3 parts
        assert_eq!(jws.split('.').count(), 3);

        let verifier = Es256Verifier::from_compressed(&kp.public_key_compressed()).unwrap();
        let valid = verifier.verify_jws(&jws).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_es256_from_uncompressed() {
        let kp = generate_keypair();
        let uncompressed = kp.public_key_uncompressed();

        let verifier = Es256Verifier::from_uncompressed(&uncompressed).unwrap();
        let signer = Es256Signer::new(kp, None);
        let message = b"test";
        let sig = signer.sign(message).unwrap();
        let valid = verifier.verify(message, &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_es256_from_jwk() {
        let kp = generate_keypair();
        let jwk = kp.public_key_jwk();

        let verifier = Es256Verifier::from_jwk(&jwk).unwrap();
        let signer = Es256Signer::new(kp, None);
        let message = b"JWK test";
        let sig = signer.sign(message).unwrap();
        let valid = verifier.verify(message, &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_es256_from_jwk_invalid() {
        // Wrong curve
        let jwk = serde_json::json!({ "kty": "EC", "crv": "secp256k1" });
        assert!(Es256Verifier::from_jwk(&jwk).is_err());

        // Wrong key type
        let jwk = serde_json::json!({ "kty": "OKP", "crv": "Ed25519" });
        assert!(Es256Verifier::from_jwk(&jwk).is_err());

        // Missing coordinates
        let jwk = serde_json::json!({ "kty": "EC", "crv": "P-256" });
        assert!(Es256Verifier::from_jwk(&jwk).is_err());
    }

    #[test]
    fn test_public_key_jwk_format() {
        let kp = generate_keypair();
        let jwk = kp.public_key_jwk();
        assert_eq!(jwk["kty"], "EC");
        assert_eq!(jwk["crv"], "P-256");
        assert!(jwk["x"].is_string());
        assert!(jwk["y"].is_string());
    }

    #[test]
    fn test_es256_invalid_signature_length() {
        let kp = generate_keypair();
        let verifier = Es256Verifier::from_compressed(&kp.public_key_compressed()).unwrap();
        // Wrong signature length
        assert!(verifier.verify(b"test", &[0u8; 63]).is_err());
        assert!(verifier.verify(b"test", &[0u8; 65]).is_err());
    }
}
