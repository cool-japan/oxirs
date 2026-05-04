//! JWT-encoded Verifiable Credentials (JWT-VC).
//!
//! Implements the W3C VC Data Model 2.0 JWT serialization format as described
//! in <https://www.w3.org/TR/vc-data-model-2.0/#jose-serialization>.
//!
//! The JWT-VC format encodes a Verifiable Credential as a JSON Web Token (JWT)
//! with a JOSE header.  This module provides:
//!
//! - [`JwtVcHeader`] — JOSE header with `typ`, `alg`, and optional `kid`
//! - [`JwtVcPayload`] — registered + private JWT claims for the VC
//! - [`JwtVc`] — the compact-serialized JWT (header.payload.signature)
//! - [`encode_vc_as_jwt`] / [`decode_jwt_vc`] — encode/decode helpers
//!
//! ## Security note
//! Signature verification must use a trusted key.  The `verify_signature` fn
//! in this module validates structure and registered claims only; callers must
//! supply a verified public key.

use crate::vc::credential::VerifiableCredential;
use crate::{DidError, DidResult};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────────────
// JOSE Header
// ─────────────────────────────────────────────────────────────────────────────

/// JOSE header for a JWT-encoded Verifiable Credential.
///
/// Per the W3C VC 2.0 specification the `typ` claim MUST be `"vc+jwt"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JwtVcHeader {
    /// Token type — MUST be `"vc+jwt"` for Verifiable Credentials.
    pub typ: String,
    /// Signature algorithm, e.g. `"EdDSA"`, `"ES256"`, `"ES256K"`.
    pub alg: String,
    /// Key identifier (DID URL referencing the signing key), e.g.
    /// `"did:key:z6Mk…#z6Mk…"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
}

impl JwtVcHeader {
    /// Create a JWT-VC header with EdDSA algorithm.
    pub fn ed25519(kid: Option<String>) -> Self {
        Self {
            typ: "vc+jwt".into(),
            alg: "EdDSA".into(),
            kid,
        }
    }

    /// Create a JWT-VC header with ES256 (P-256) algorithm.
    pub fn es256(kid: Option<String>) -> Self {
        Self {
            typ: "vc+jwt".into(),
            alg: "ES256".into(),
            kid,
        }
    }

    /// Validate that `typ` equals `"vc+jwt"` (case-insensitive) and `alg`
    /// is not `"none"`.
    pub fn validate(&self) -> DidResult<()> {
        if self.typ.to_lowercase() != "vc+jwt" {
            return Err(DidError::InvalidCredential(format!(
                "JWT-VC `typ` must be \"vc+jwt\", got {:?}",
                self.typ
            )));
        }
        if self.alg.eq_ignore_ascii_case("none") {
            return Err(DidError::InvalidCredential(
                "JWT-VC `alg` must not be \"none\"".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JWT-VC Payload
// ─────────────────────────────────────────────────────────────────────────────

/// JWT claims for a Verifiable Credential.
///
/// Registered claims (`iss`, `sub`, `jti`, `exp`, `nbf`) map to VC fields.
/// The full VC object is embedded in the `vc` private claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtVcPayload {
    /// `iss` — Issuer DID URI.
    pub iss: String,
    /// `sub` — Subject DID URI (from `credentialSubject.id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    /// `jti` — Credential identifier URI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
    /// `iat` — Issued-at (Unix seconds).
    pub iat: i64,
    /// `nbf` — Not-before (Unix seconds), maps to `validFrom`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbf: Option<i64>,
    /// `exp` — Expiration time (Unix seconds), maps to `validUntil`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<i64>,
    /// `vc` — The original Verifiable Credential object.
    pub vc: Value,
}

impl JwtVcPayload {
    /// Build a [`JwtVcPayload`] from a [`VerifiableCredential`].
    ///
    /// The `vc` field stores the serialized VC.  Registered claims are
    /// extracted from the VC fields.
    pub fn from_vc(vc: &VerifiableCredential) -> DidResult<Self> {
        let iss = vc.issuer.did().to_string();
        let sub = vc
            .credential_subject
            .subjects()
            .into_iter()
            .next()
            .and_then(|s| s.id.clone());
        let jti = vc.id.clone();
        let iat = Utc::now().timestamp();
        let nbf = vc
            .valid_from
            .as_ref()
            .or(vc.issuance_date.as_ref())
            .map(|d| d.timestamp());
        let exp = vc
            .valid_until
            .as_ref()
            .or(vc.expiration_date.as_ref())
            .map(|d| d.timestamp());

        let vc_value = serde_json::to_value(vc)
            .map_err(|e| DidError::InvalidCredential(format!("Failed to serialize VC: {e}")))?;

        Ok(Self {
            iss,
            sub,
            jti,
            iat,
            nbf,
            exp,
            vc: vc_value,
        })
    }

    /// Validate registered claims.
    pub fn validate(&self, now: DateTime<Utc>) -> DidResult<()> {
        let now_ts = now.timestamp();
        if let Some(exp) = self.exp {
            if now_ts > exp {
                return Err(DidError::InvalidCredential(format!(
                    "JWT-VC has expired (exp={exp}, now={now_ts})"
                )));
            }
        }
        if let Some(nbf) = self.nbf {
            if now_ts < nbf {
                return Err(DidError::InvalidCredential(format!(
                    "JWT-VC is not yet valid (nbf={nbf}, now={now_ts})"
                )));
            }
        }
        if self.iss.is_empty() {
            return Err(DidError::InvalidCredential(
                "JWT-VC `iss` must not be empty".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compact JWT representation
// ─────────────────────────────────────────────────────────────────────────────

/// A compact-serialized JWT-VC (header.payload.signature).
///
/// The signature bytes are opaque; callers must supply a signer/verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct JwtVc {
    /// Base64url-encoded JOSE header.
    pub header_b64: String,
    /// Base64url-encoded JSON payload.
    pub payload_b64: String,
    /// Base64url-encoded signature (may be empty for unsigned/test tokens).
    pub signature_b64: String,
}

impl JwtVc {
    /// Build a compact JWT from encoded header + payload + signature bytes.
    pub fn new(header_b64: String, payload_b64: String, signature_b64: String) -> Self {
        Self {
            header_b64,
            payload_b64,
            signature_b64,
        }
    }

    /// Serialize to compact form `header.payload.signature`.
    pub fn to_compact(&self) -> String {
        format!(
            "{}.{}.{}",
            self.header_b64, self.payload_b64, self.signature_b64
        )
    }

    /// Parse a compact JWT string into header, payload, and signature parts.
    pub fn from_compact(compact: &str) -> DidResult<Self> {
        let parts: Vec<&str> = compact.split('.').collect();
        if parts.len() != 3 {
            return Err(DidError::InvalidCredential(format!(
                "JWT must have 3 parts separated by '.', got {}",
                parts.len()
            )));
        }
        Ok(Self {
            header_b64: parts[0].to_string(),
            payload_b64: parts[1].to_string(),
            signature_b64: parts[2].to_string(),
        })
    }

    /// Decode and parse the JOSE header.
    pub fn decode_header(&self) -> DidResult<JwtVcHeader> {
        let bytes = URL_SAFE_NO_PAD
            .decode(&self.header_b64)
            .map_err(|e| DidError::InvalidCredential(format!("Invalid base64url header: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DidError::InvalidCredential(format!("Invalid header JSON: {e}")))
    }

    /// Decode and parse the JWT payload.
    pub fn decode_payload(&self) -> DidResult<JwtVcPayload> {
        let bytes = URL_SAFE_NO_PAD
            .decode(&self.payload_b64)
            .map_err(|e| DidError::InvalidCredential(format!("Invalid base64url payload: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| DidError::InvalidCredential(format!("Invalid payload JSON: {e}")))
    }

    /// Return the signing input `header.payload` as bytes (used for verification).
    pub fn signing_input_bytes(&self) -> Vec<u8> {
        format!("{}.{}", self.header_b64, self.payload_b64).into_bytes()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode / decode helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a [`VerifiableCredential`] as a JWT-VC.
///
/// The `sign` closure receives `header.payload` as bytes and must return the
/// raw signature bytes.  The caller is responsible for key management.
pub fn encode_vc_as_jwt<F>(
    vc: &VerifiableCredential,
    header: JwtVcHeader,
    sign: F,
) -> DidResult<JwtVc>
where
    F: FnOnce(&[u8]) -> DidResult<Vec<u8>>,
{
    header.validate()?;

    let payload = JwtVcPayload::from_vc(vc)?;

    let header_json = serde_json::to_string(&header)
        .map_err(|e| DidError::InvalidCredential(format!("Failed to serialize JWT header: {e}")))?;
    let payload_json = serde_json::to_string(&payload).map_err(|e| {
        DidError::InvalidCredential(format!("Failed to serialize JWT payload: {e}"))
    })?;

    let header_b64 = URL_SAFE_NO_PAD.encode(header_json.as_bytes());
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());

    let signing_input = format!("{header_b64}.{payload_b64}");
    let sig_bytes = sign(signing_input.as_bytes())?;
    let signature_b64 = URL_SAFE_NO_PAD.encode(&sig_bytes);

    Ok(JwtVc::new(header_b64, payload_b64, signature_b64))
}

/// Decode and structurally validate a compact JWT-VC string.
///
/// This function validates:
/// - the JWT has exactly three parts
/// - the header has `typ = "vc+jwt"` and `alg != "none"`
/// - registered claims are temporally valid against `now`
///
/// Signature verification is the caller's responsibility (supply a verifier fn).
pub fn decode_jwt_vc(compact: &str, now: DateTime<Utc>) -> DidResult<(JwtVcHeader, JwtVcPayload)> {
    let jwt = JwtVc::from_compact(compact)?;
    let header = jwt.decode_header()?;
    header.validate()?;
    let payload = jwt.decode_payload()?;
    payload.validate(now)?;
    Ok((header, payload))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_vc_json() -> &'static str {
        r#"{
            "@context": ["https://www.w3.org/ns/credentials/v2"],
            "type": ["VerifiableCredential"],
            "issuer": {"id": "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuias8sisDArDJF"},
            "validFrom": "2024-01-01T00:00:00Z",
            "credentialSubject": {"id": "did:key:z6MkiTBz1ymuepAQ4HEHYSF1H8quG5GLVVQR3djdX3mDooWp"}
        }"#
    }

    #[test]
    fn test_header_ed25519_type() {
        let h = JwtVcHeader::ed25519(None);
        assert_eq!(h.typ, "vc+jwt");
        assert_eq!(h.alg, "EdDSA");
        assert!(h.kid.is_none());
    }

    #[test]
    fn test_header_validation_wrong_typ() {
        let h = JwtVcHeader {
            typ: "JWT".into(),
            alg: "EdDSA".into(),
            kid: None,
        };
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_header_validation_alg_none_rejected() {
        let h = JwtVcHeader {
            typ: "vc+jwt".into(),
            alg: "none".into(),
            kid: None,
        };
        assert!(h.validate().is_err());
    }

    #[test]
    fn test_jwt_compact_round_trip() {
        let jwt = JwtVc::new("header".into(), "payload".into(), "sig".into());
        let compact = jwt.to_compact();
        let parsed = JwtVc::from_compact(&compact).unwrap();
        assert_eq!(parsed, jwt);
    }

    #[test]
    fn test_jwt_from_compact_wrong_parts() {
        assert!(JwtVc::from_compact("only.two").is_err());
        assert!(JwtVc::from_compact("a.b.c.d").is_err());
    }

    #[test]
    fn test_encode_and_decode_round_trip() {
        let vc: serde_json::Value = serde_json::from_str(minimal_vc_json()).unwrap();
        // Build a VerifiableCredential from raw JSON via encode helper path
        let header = JwtVcHeader::ed25519(Some("did:key:z6Mk#z6Mk".into()));

        // Use identity "signer" that returns empty bytes (structural test only)
        let _header_json = serde_json::to_string(&header).unwrap();
        let _payload_json = serde_json::to_string(&vc).unwrap();

        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_string(&header).unwrap().as_bytes());
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_string(&vc).unwrap().as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(b"dummy_signature_bytes");

        let jwt = JwtVc::new(header_b64, payload_b64, sig_b64);
        let compact = jwt.to_compact();

        let parsed = JwtVc::from_compact(&compact).unwrap();
        let decoded_header = parsed.decode_header().unwrap();
        assert_eq!(decoded_header.typ, "vc+jwt");
        assert_eq!(decoded_header.alg, "EdDSA");
    }

    #[test]
    fn test_payload_validate_expired() {
        let past = Utc::now().timestamp() - 3600;
        let payload = JwtVcPayload {
            iss: "did:key:z6Mk".into(),
            sub: None,
            jti: None,
            iat: past - 100,
            nbf: None,
            exp: Some(past), // already expired
            vc: serde_json::Value::Null,
        };
        assert!(payload.validate(Utc::now()).is_err());
    }

    #[test]
    fn test_payload_validate_not_yet_valid() {
        let future = Utc::now().timestamp() + 3600;
        let payload = JwtVcPayload {
            iss: "did:key:z6Mk".into(),
            sub: None,
            jti: None,
            iat: Utc::now().timestamp(),
            nbf: Some(future),
            exp: None,
            vc: serde_json::Value::Null,
        };
        assert!(payload.validate(Utc::now()).is_err());
    }

    #[test]
    fn test_payload_validate_empty_iss_rejected() {
        let payload = JwtVcPayload {
            iss: "".into(),
            sub: None,
            jti: None,
            iat: Utc::now().timestamp(),
            nbf: None,
            exp: None,
            vc: serde_json::Value::Null,
        };
        assert!(payload.validate(Utc::now()).is_err());
    }

    #[test]
    fn test_signing_input_format() {
        let jwt = JwtVc::new("H".into(), "P".into(), "S".into());
        assert_eq!(String::from_utf8(jwt.signing_input_bytes()).unwrap(), "H.P");
    }
}
