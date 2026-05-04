//! Selective Disclosure JWT (SD-JWT) — IETF draft-ietf-oauth-selective-disclosure-jwt.
//!
//! Implements the core SD-JWT mechanism for selectively disclosable Verifiable
//! Credentials:
//!
//! - Claims are individually salted and hashed; only the hash is put in the JWT
//! - The holder presents a *compact serialization*: `<JWT>~<disclosure1>~…~`
//! - Verifiers reconstruct and verify only the disclosed claims
//!
//! ## Structure
//!
//! ```text
//! SD-JWT = <Issuer-JWT>~<Disclosure>*[~<KB-JWT>]
//! ```
//!
//! Each `<Disclosure>` is a base64url-encoded JSON array: `[salt, claim_name, value]`.
//!
//! ## Security properties
//! - SHA-256 hash binding prevents claim forgery
//! - Each salt is 128-bit random (cryptographically random via `uuid`)
//! - The issuer JWT `_sd` array lists the expected hashes

use crate::{DidError, DidResult};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Disclosure
// ─────────────────────────────────────────────────────────────────────────────

/// A single SD-JWT disclosure: `[salt, claim_name, value]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Disclosure {
    /// Random 128-bit salt (hex string).
    pub salt: String,
    /// Claim name (JSON key).
    pub claim_name: String,
    /// Claim value (arbitrary JSON).
    pub claim_value: Value,
}

impl Disclosure {
    /// Create a new disclosure with the given salt.
    pub fn new(salt: impl Into<String>, claim_name: impl Into<String>, claim_value: Value) -> Self {
        Self {
            salt: salt.into(),
            claim_name: claim_name.into(),
            claim_value,
        }
    }

    /// Generate a disclosure with a fresh random salt.
    pub fn generate(claim_name: impl Into<String>, claim_value: Value) -> Self {
        // Use UUID v4 as source of 128-bit randomness
        let salt = uuid::Uuid::new_v4().to_string().replace('-', "");
        Self::new(salt, claim_name, claim_value)
    }

    /// Serialize to the SD-JWT compact disclosure form (base64url).
    ///
    /// The underlying JSON array is `[salt, claim_name, value]`.
    pub fn to_base64url(&self) -> DidResult<String> {
        let arr = serde_json::json!([self.salt, self.claim_name, self.claim_value]);
        let json = serde_json::to_string(&arr).map_err(|e| {
            DidError::InvalidCredential(format!("Failed to serialize disclosure: {e}"))
        })?;
        Ok(URL_SAFE_NO_PAD.encode(json.as_bytes()))
    }

    /// Deserialize a disclosure from base64url compact form.
    pub fn from_base64url(encoded: &str) -> DidResult<Self> {
        let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|e| {
            DidError::InvalidCredential(format!("Invalid base64url disclosure: {e}"))
        })?;
        let arr: Value = serde_json::from_slice(&bytes)
            .map_err(|e| DidError::InvalidCredential(format!("Invalid disclosure JSON: {e}")))?;
        let arr = arr
            .as_array()
            .ok_or_else(|| DidError::InvalidCredential("Disclosure must be a JSON array".into()))?;
        if arr.len() != 3 {
            return Err(DidError::InvalidCredential(format!(
                "Disclosure array must have 3 elements, got {}",
                arr.len()
            )));
        }
        let salt = arr[0]
            .as_str()
            .ok_or_else(|| DidError::InvalidCredential("Disclosure salt must be a string".into()))?
            .to_string();
        let claim_name = arr[1]
            .as_str()
            .ok_or_else(|| {
                DidError::InvalidCredential("Disclosure claim_name must be a string".into())
            })?
            .to_string();
        let claim_value = arr[2].clone();
        Ok(Self {
            salt,
            claim_name,
            claim_value,
        })
    }

    /// Compute the SHA-256 disclosure hash (base64url-encoded).
    ///
    /// This is the hash placed in the `_sd` array in the issuer JWT.
    pub fn hash(&self) -> DidResult<String> {
        let encoded = self.to_base64url()?;
        let digest = Sha256::digest(encoded.as_bytes());
        Ok(URL_SAFE_NO_PAD.encode(digest))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SdJwtClaims — payload extension for SD-JWT
// ─────────────────────────────────────────────────────────────────────────────

/// SD-JWT claims object: contains `_sd` (list of hashes) and `_sd_alg`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SdJwtClaims {
    /// SHA-256 hashes of individual disclosures.
    #[serde(rename = "_sd", default, skip_serializing_if = "Vec::is_empty")]
    pub sd_hashes: Vec<String>,
    /// Hash algorithm identifier (always `"sha-256"`).
    #[serde(rename = "_sd_alg", default, skip_serializing_if = "String::is_empty")]
    pub sd_alg: String,
}

impl SdJwtClaims {
    /// Create an SD-JWT claims object for SHA-256.
    pub fn new_sha256(hashes: Vec<String>) -> Self {
        Self {
            sd_hashes: hashes,
            sd_alg: "sha-256".into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SdJwtIssuer — build an SD-JWT payload
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for creating SD-JWT payloads from a set of selectively disclosable claims.
pub struct SdJwtIssuer {
    /// Always-visible claims (non-selectively-disclosed).
    pub public_claims: HashMap<String, Value>,
    /// Disclosures (one per selectively disclosable claim).
    pub disclosures: Vec<Disclosure>,
}

impl SdJwtIssuer {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            public_claims: HashMap::new(),
            disclosures: Vec::new(),
        }
    }

    /// Add a public (always visible) claim.
    pub fn public_claim(mut self, key: impl Into<String>, value: Value) -> Self {
        self.public_claims.insert(key.into(), value);
        self
    }

    /// Add a selectively disclosable claim.
    pub fn sd_claim(mut self, key: impl Into<String>, value: Value) -> Self {
        self.disclosures.push(Disclosure::generate(key, value));
        self
    }

    /// Build the `SdJwtClaims` object (hashes for the JWT payload).
    pub fn build_sd_claims(&self) -> DidResult<SdJwtClaims> {
        let hashes = self
            .disclosures
            .iter()
            .map(|d| d.hash())
            .collect::<DidResult<Vec<_>>>()?;
        Ok(SdJwtClaims::new_sha256(hashes))
    }

    /// Build the full SD-JWT compact serialization.
    ///
    /// The `jwt_compact` argument is the issuer JWT (already signed).
    pub fn build_compact(&self, jwt_compact: &str) -> DidResult<String> {
        let mut parts = vec![jwt_compact.to_string()];
        for d in &self.disclosures {
            parts.push(d.to_base64url()?);
        }
        // Trailing ~ required by spec
        Ok(parts.join("~") + "~")
    }
}

impl Default for SdJwtIssuer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SdJwtVerifier — verify and reconstruct disclosed claims
// ─────────────────────────────────────────────────────────────────────────────

/// Verifier for SD-JWT compact serializations.
pub struct SdJwtVerifier;

impl SdJwtVerifier {
    /// Parse an SD-JWT compact serialization.
    ///
    /// Returns `(jwt_compact, disclosures)`.
    pub fn parse(compact: &str) -> DidResult<(String, Vec<Disclosure>)> {
        let parts: Vec<&str> = compact.split('~').collect();
        if parts.is_empty() {
            return Err(DidError::InvalidCredential(
                "Empty SD-JWT compact serialization".into(),
            ));
        }
        let jwt = parts[0].to_string();
        let mut disclosures = Vec::new();
        for part in &parts[1..] {
            if part.is_empty() {
                // Trailing ~ produces empty part — skip it
                continue;
            }
            disclosures.push(Disclosure::from_base64url(part)?);
        }
        Ok((jwt, disclosures))
    }

    /// Verify disclosure hashes against the `_sd` array extracted from the
    /// JWT payload JSON object `sd_claims`.
    ///
    /// Returns the set of verified (claim_name, value) pairs.
    pub fn verify_disclosures(
        disclosures: &[Disclosure],
        sd_claims: &SdJwtClaims,
    ) -> DidResult<HashMap<String, Value>> {
        let mut verified = HashMap::new();
        for d in disclosures {
            let hash = d.hash()?;
            if sd_claims.sd_hashes.contains(&hash) {
                verified.insert(d.claim_name.clone(), d.claim_value.clone());
            } else {
                return Err(DidError::InvalidCredential(format!(
                    "Disclosure hash mismatch for claim {:?}",
                    d.claim_name
                )));
            }
        }
        Ok(verified)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_disclosure_round_trip() {
        let d = Disclosure::new("salt123", "name", json!("Alice"));
        let encoded = d.to_base64url().unwrap();
        let decoded = Disclosure::from_base64url(&encoded).unwrap();
        assert_eq!(decoded, d);
    }

    #[test]
    fn test_disclosure_hash_deterministic() {
        let d = Disclosure::new("fixed_salt", "age", json!(42));
        let h1 = d.hash().unwrap();
        let h2 = d.hash().unwrap();
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn test_disclosure_hash_sha256() {
        // The hash should be SHA-256 of the base64url-encoded JSON array
        let d = Disclosure::new("abc", "sub", json!("user_42"));
        let encoded = d.to_base64url().unwrap();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(encoded.as_bytes()));
        assert_eq!(d.hash().unwrap(), expected);
    }

    #[test]
    fn test_disclosure_wrong_array_size() {
        // JSON array with 2 elements (missing value) should fail
        let json_arr = json!(["salt", "key"]);
        let encoded = URL_SAFE_NO_PAD.encode(json_arr.to_string().as_bytes());
        assert!(Disclosure::from_base64url(&encoded).is_err());
    }

    #[test]
    fn test_disclosure_generate_random_salt() {
        let d1 = Disclosure::generate("field", json!("v"));
        let d2 = Disclosure::generate("field", json!("v"));
        assert_ne!(d1.salt, d2.salt, "salts must be different");
    }

    #[test]
    fn test_sd_jwt_issuer_build_sd_claims() {
        let issuer = SdJwtIssuer::new()
            .public_claim("iss", json!("did:key:z6Mk"))
            .sd_claim("name", json!("Alice"))
            .sd_claim("age", json!(30));

        let claims = issuer.build_sd_claims().unwrap();
        assert_eq!(claims.sd_hashes.len(), 2, "two sd claims → two hashes");
        assert_eq!(claims.sd_alg, "sha-256");
    }

    #[test]
    fn test_sd_jwt_compact_parse_round_trip() {
        let issuer = SdJwtIssuer::new().sd_claim("email", json!("alice@example.com"));

        let jwt_compact = "header.payload.sig";
        let compact = issuer.build_compact(jwt_compact).unwrap();

        assert!(compact.starts_with(jwt_compact), "must start with JWT");
        assert!(compact.ends_with('~'), "must end with trailing ~");

        let (parsed_jwt, disclosures) = SdJwtVerifier::parse(&compact).unwrap();
        assert_eq!(parsed_jwt, jwt_compact);
        assert_eq!(disclosures.len(), 1);
        assert_eq!(disclosures[0].claim_name, "email");
    }

    #[test]
    fn test_sd_jwt_verify_disclosures_success() {
        let issuer = SdJwtIssuer::new().sd_claim("city", json!("Tokyo"));

        let sd_claims = issuer.build_sd_claims().unwrap();
        let verified = SdJwtVerifier::verify_disclosures(&issuer.disclosures, &sd_claims).unwrap();
        assert_eq!(verified.get("city"), Some(&json!("Tokyo")));
    }

    #[test]
    fn test_sd_jwt_verify_disclosures_tampered_value() {
        let issuer = SdJwtIssuer::new().sd_claim("city", json!("Tokyo"));
        let sd_claims = issuer.build_sd_claims().unwrap();

        // Tamper the disclosure value
        let mut tampered = issuer.disclosures.clone();
        tampered[0].claim_value = json!("Osaka");

        let result = SdJwtVerifier::verify_disclosures(&tampered, &sd_claims);
        assert!(result.is_err(), "tampered disclosure must fail hash check");
    }

    #[test]
    fn test_sd_jwt_empty_disclosures() {
        let sd_claims = SdJwtClaims::default();
        let verified = SdJwtVerifier::verify_disclosures(&[], &sd_claims).unwrap();
        assert!(verified.is_empty());
    }

    #[test]
    fn test_sd_jwt_parse_no_disclosures() {
        let (jwt, disclosures) = SdJwtVerifier::parse("h.p.s~").unwrap();
        assert_eq!(jwt, "h.p.s");
        assert!(disclosures.is_empty(), "trailing ~ → empty disclosures");
    }
}
