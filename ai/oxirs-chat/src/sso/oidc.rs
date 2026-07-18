//! OIDC (OpenID Connect) provider abstraction for Enterprise SSO.
//!
//! Provides an OIDC ID-token validator that performs **cryptographic
//! signature verification** against the IdP's JSON Web Key Set (JWKS) before
//! trusting any claim, in addition to validating the structural claims `exp`,
//! `nbf`, `iss`, `aud`, and (optionally) `nonce`.
//!
//! ## Security model
//!
//! [`OidcValidator::validate_id_token`] **fails closed**: unless the validator
//! was constructed with a verification key set via [`OidcValidator::with_jwks`],
//! every token is rejected with [`SsoError::SignatureVerificationUnavailable`].
//! Supported signature algorithms are `RS256` (RSASSA-PKCS1-v1_5 + SHA-256, via
//! the Pure-Rust `rsa` crate) and `ES256` (ECDSA over P-256 + SHA-256, via the
//! Pure-Rust `p256` crate). The `none` algorithm and every unknown/unsupported
//! `alg` are rejected outright. Signature verification happens *before* any
//! claim is trusted.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_chat::sso::oidc::{OidcValidator, SsoConfig, SsoProviderType};
//!
//! let config = SsoConfig {
//!     provider_type: SsoProviderType::Oidc,
//!     issuer_url: "https://accounts.example.com".to_string(),
//!     client_id: "my-app".to_string(),
//!     redirect_uri: "https://app.example.com/callback".to_string(),
//!     scopes: vec!["openid".to_string(), "profile".to_string(), "email".to_string()],
//! };
//! let validator = OidcValidator::new(config);
//! let url = validator.authorization_url("random-state-xyz", "random-nonce-abc");
//! assert!(url.contains("client_id=my-app"));
//! ```

use std::collections::HashMap;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Public error type ──────────────────────────────────────────────────────

/// Errors that can occur in the OIDC / SAML SSO layer.
#[derive(Debug, Error)]
pub enum SsoError {
    /// The token's `exp` claim is in the past.
    #[error("token expired")]
    TokenExpired,

    /// The token's `iss` claim does not match the configured issuer URL.
    #[error("invalid issuer: expected {expected}, got {got}")]
    InvalidIssuer { expected: String, got: String },

    /// The token's `aud` claim does not include the configured client ID.
    #[error("invalid audience")]
    InvalidAudience,

    /// The token / response cannot be parsed (structural or encoding problem).
    #[error("malformed token: {0}")]
    MalformedToken(String),

    /// The configured provider type is not supported by this operation.
    #[error("unsupported provider type")]
    UnsupportedProvider,

    /// A base64 decode step failed.
    #[error("base64 decode error: {0}")]
    Base64Error(String),

    /// The token's `alg` header is `none`, missing, or an algorithm this
    /// validator does not support (only `RS256` and `ES256` are accepted).
    #[error("unsupported or disallowed JWT algorithm: {0}")]
    UnsupportedAlgorithm(String),

    /// No verification key set was configured, so the signature cannot be
    /// checked. The validator fails closed rather than trusting an unverified
    /// token — construct it with [`OidcValidator::with_jwks`].
    #[error("ID-token signature verification is unavailable: no JWKS configured (failing closed)")]
    SignatureVerificationUnavailable,

    /// No key in the configured JWK set matches the token's `kid`/algorithm.
    #[error("no matching verification key found for token")]
    KeyNotFound,

    /// A JWK could not be parsed into a usable public key.
    #[error("invalid verification key: {0}")]
    InvalidKey(String),

    /// The cryptographic signature check failed — the token is forged,
    /// tampered with, or signed by an unknown key.
    #[error("ID-token signature verification failed")]
    SignatureInvalid,

    /// The token's `nonce` claim does not match the expected value.
    #[error("nonce mismatch")]
    NonceMismatch,

    /// A SAML response carried no signature and the helper was not configured to
    /// accept unsigned responses — rejected to avoid an authentication bypass.
    #[error("unsigned SAML response rejected (no ds:Signature; allow_unsigned is off)")]
    UnsignedAssertionRejected,
}

// ── Core config types ──────────────────────────────────────────────────────

/// Which SSO protocol the provider uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SsoProviderType {
    /// OpenID Connect authorization-code flow.
    Oidc,
    /// SAML 2.0 Service Provider.
    Saml,
}

/// Lightweight provider configuration used by [`OidcValidator`] and
/// [`crate::sso::saml_sp::SamlSpHelper`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsoConfig {
    /// Which protocol the provider uses.
    pub provider_type: SsoProviderType,
    /// OIDC: the issuer URL (e.g. `https://accounts.google.com`).
    /// SAML: the IdP entity ID.
    pub issuer_url: String,
    /// OAuth 2.0 / OIDC client ID.
    pub client_id: String,
    /// The redirect / Assertion Consumer Service URL registered with the IdP.
    pub redirect_uri: String,
    /// OIDC scopes to request.  Ignored for SAML providers.
    /// Defaults to `["openid", "profile", "email"]`.
    pub scopes: Vec<String>,
}

impl Default for SsoConfig {
    fn default() -> Self {
        Self {
            provider_type: SsoProviderType::Oidc,
            issuer_url: String::new(),
            client_id: String::new(),
            redirect_uri: String::new(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        }
    }
}

// ── User info ──────────────────────────────────────────────────────────────

/// Normalised user identity extracted from an OIDC ID-token or SAML assertion.
#[derive(Debug, Clone)]
pub struct SsoUserInfo {
    /// The `sub` (subject) claim — stable user identifier from the IdP.
    pub subject: String,
    /// The user's email address (from `email` claim or SAML attribute).
    pub email: Option<String>,
    /// The user's display name (from `name` claim or SAML attribute).
    pub name: Option<String>,
    /// Groups / roles the user belongs to (from `groups` / `roles` claim).
    pub groups: Vec<String>,
    /// All raw claims from the token payload.
    pub raw_claims: HashMap<String, serde_json::Value>,
}

// ── OIDC callback ──────────────────────────────────────────────────────────

/// Parsed parameters from the OIDC authorization-code redirect callback.
#[derive(Debug, Clone)]
pub struct OidcCallback {
    /// The authorization code returned by the IdP.
    pub code: String,
    /// The `state` parameter for CSRF protection (must match what was sent).
    pub state: String,
}

// ── JSON Web Key Set (JWKS) ────────────────────────────────────────────────

/// A JSON Web Key Set — the collection of public keys published by an IdP at
/// its `jwks_uri`, used to verify ID-token signatures.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Jwks {
    /// The individual keys.
    pub keys: Vec<Jwk>,
}

impl Jwks {
    /// Parse a JWKS from its JSON representation (as returned by a `jwks_uri`).
    pub fn from_json(json: &str) -> Result<Self, SsoError> {
        serde_json::from_str(json)
            .map_err(|e| SsoError::InvalidKey(format!("JWKS JSON parse error: {e}")))
    }

    /// Select the key that should verify a token with the given header.
    ///
    /// If the header carries a `kid`, only a key with that exact `kid` is
    /// accepted. Otherwise the first key whose `kty` is compatible with the
    /// header algorithm is used.
    fn select(&self, header: &JwtHeader) -> Result<&Jwk, SsoError> {
        if let Some(kid) = header.kid.as_deref() {
            return self
                .keys
                .iter()
                .find(|k| k.kid.as_deref() == Some(kid))
                .ok_or(SsoError::KeyNotFound);
        }
        let wanted_kty = match header.alg.as_str() {
            "RS256" => "RSA",
            "ES256" => "EC",
            _ => return Err(SsoError::UnsupportedAlgorithm(header.alg.clone())),
        };
        self.keys
            .iter()
            .find(|k| k.kty == wanted_kty)
            .ok_or(SsoError::KeyNotFound)
    }
}

/// A single JSON Web Key (RFC 7517). Only the fields needed to reconstruct an
/// RSA or P-256 public key are modelled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Jwk {
    /// Key type: `RSA` or `EC`.
    pub kty: String,
    /// Key ID, matched against the JWT header `kid`.
    #[serde(default)]
    pub kid: Option<String>,
    /// Intended algorithm (e.g. `RS256`, `ES256`).
    #[serde(default)]
    pub alg: Option<String>,
    /// Public key use (`sig`).
    #[serde(default, rename = "use")]
    pub key_use: Option<String>,
    /// RSA modulus (base64url, big-endian).
    #[serde(default)]
    pub n: Option<String>,
    /// RSA public exponent (base64url, big-endian).
    #[serde(default)]
    pub e: Option<String>,
    /// EC curve name (`P-256`).
    #[serde(default)]
    pub crv: Option<String>,
    /// EC public key X coordinate (base64url, 32 bytes for P-256).
    #[serde(default)]
    pub x: Option<String>,
    /// EC public key Y coordinate (base64url, 32 bytes for P-256).
    #[serde(default)]
    pub y: Option<String>,
}

/// The decoded JOSE header of a JWT.
#[derive(Debug, Clone, Deserialize)]
struct JwtHeader {
    alg: String,
    #[serde(default)]
    kid: Option<String>,
}

// ── OidcValidator ──────────────────────────────────────────────────────────

/// Validates OIDC ID-tokens and builds authorization URLs.
///
/// Cryptographic signature verification is performed against the configured
/// [`Jwks`] (see [`OidcValidator::with_jwks`]). When no key set is configured,
/// [`OidcValidator::validate_id_token`] fails closed — it never trusts an
/// unverified token.
pub struct OidcValidator {
    config: SsoConfig,
    jwks: Option<Jwks>,
}

impl OidcValidator {
    /// Create a new validator with the given provider configuration.
    ///
    /// No verification key set is attached, so [`Self::validate_id_token`] will
    /// fail closed until [`Self::with_jwks`] (or [`Self::set_jwks`]) supplies the
    /// IdP's public keys.
    pub fn new(config: SsoConfig) -> Self {
        Self { config, jwks: None }
    }

    /// Create a validator with the IdP's JWK set attached, enabling signature
    /// verification.
    pub fn with_jwks(config: SsoConfig, jwks: Jwks) -> Self {
        Self {
            config,
            jwks: Some(jwks),
        }
    }

    /// Attach (or replace) the IdP's JWK set — e.g. after fetching it from the
    /// `jwks_uri`.
    pub fn set_jwks(&mut self, jwks: Jwks) {
        self.jwks = Some(jwks);
    }

    /// Validate a JWT `id_token` and extract the normalised [`SsoUserInfo`].
    ///
    /// Validation, in security-critical order:
    /// 1. JWT structure (three dot-separated segments).
    /// 2. Header `alg` is `RS256` or `ES256` — `none` and unknown algorithms
    ///    are rejected.
    /// 3. **Signature is cryptographically verified** against the configured
    ///    [`Jwks`]. Fails closed with [`SsoError::SignatureVerificationUnavailable`]
    ///    when no key set is configured.
    /// 4. `exp` claim is in the future and `nbf` (if present) is not in the future.
    /// 5. `iss` claim matches `config.issuer_url`.
    /// 6. `aud` claim includes `config.client_id`.
    pub fn validate_id_token(&self, id_token: &str) -> Result<SsoUserInfo, SsoError> {
        self.validate_id_token_with_nonce(id_token, None)
    }

    /// Like [`Self::validate_id_token`] but additionally requires the token's
    /// `nonce` claim to equal `expected_nonce` (OIDC replay protection).
    pub fn validate_id_token_with_nonce(
        &self,
        id_token: &str,
        expected_nonce: Option<&str>,
    ) -> Result<SsoUserInfo, SsoError> {
        // 1. Structure + 2. header algorithm allow-listing.
        let (signing_input, signature) = split_signing_input(id_token)?;
        let header = parse_jwt_header(id_token)?;
        let alg = JwtAlg::parse(&header.alg)?;

        // 3. Cryptographic signature verification (fail closed).
        let jwks = self
            .jwks
            .as_ref()
            .ok_or(SsoError::SignatureVerificationUnavailable)?;
        let jwk = jwks.select(&header)?;
        alg.verify(signing_input, &signature, jwk)?;

        // Only now do we parse and trust the claims.
        let claims = parse_jwt_claims(id_token)?;

        // 4. Check expiry
        let exp = claims
            .get("exp")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| SsoError::MalformedToken("missing 'exp' claim".to_string()))?;
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if exp <= now_ts {
            return Err(SsoError::TokenExpired);
        }

        // 4b. Check `nbf` (not-before), allowing a small clock-skew window.
        const CLOCK_SKEW_SECS: i64 = 120;
        if let Some(nbf) = claims.get("nbf").and_then(|v| v.as_i64()) {
            if nbf > now_ts + CLOCK_SKEW_SECS {
                return Err(SsoError::MalformedToken(
                    "token is not yet valid ('nbf' is in the future)".to_string(),
                ));
            }
        }

        // 5. Check issuer
        let iss = claims
            .get("iss")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SsoError::MalformedToken("missing 'iss' claim".to_string()))?
            .to_string();
        if iss != self.config.issuer_url {
            return Err(SsoError::InvalidIssuer {
                expected: self.config.issuer_url.clone(),
                got: iss,
            });
        }

        // 6. Check audience
        let aud_matches = match claims.get("aud") {
            Some(serde_json::Value::String(s)) => s == &self.config.client_id,
            Some(serde_json::Value::Array(arr)) => arr.iter().any(|v| {
                v.as_str()
                    .map(|s| s == self.config.client_id)
                    .unwrap_or(false)
            }),
            _ => false,
        };
        if !aud_matches {
            return Err(SsoError::InvalidAudience);
        }

        // 7. Check nonce (replay protection) when the caller supplied one.
        if let Some(expected) = expected_nonce {
            let token_nonce = claims.get("nonce").and_then(|v| v.as_str());
            if token_nonce != Some(expected) {
                return Err(SsoError::NonceMismatch);
            }
        }

        // Extract user fields
        let subject = claims
            .get("sub")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SsoError::MalformedToken("missing 'sub' claim".to_string()))?
            .to_string();

        let email = claims
            .get("email")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let name = claims
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let groups = extract_string_list(&claims, "groups")
            .into_iter()
            .chain(extract_string_list(&claims, "roles"))
            .collect();

        Ok(SsoUserInfo {
            subject,
            email,
            name,
            groups,
            raw_claims: claims,
        })
    }

    /// Build the OIDC authorization URL for the authorization-code flow.
    ///
    /// The returned URL includes `response_type=code`, `client_id`, `redirect_uri`,
    /// `scope`, `state`, and `nonce` query parameters.
    pub fn authorization_url(&self, state: &str, nonce: &str) -> String {
        let scope = self.config.scopes.join(" ");
        let params = [
            ("response_type", "code"),
            ("client_id", self.config.client_id.as_str()),
            ("redirect_uri", self.config.redirect_uri.as_str()),
            ("scope", scope.as_str()),
            ("state", state),
            ("nonce", nonce),
        ];
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        format!(
            "{}/authorize?{}",
            self.config.issuer_url.trim_end_matches('/'),
            query
        )
    }

    /// Parse the callback query string (e.g. `"code=abc&state=xyz"`) from the
    /// IdP redirect and extract the authorization code and state.
    pub fn parse_callback(&self, query: &str) -> Result<OidcCallback, SsoError> {
        let mut code: Option<String> = None;
        let mut state: Option<String> = None;

        for pair in query.split('&') {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next().unwrap_or("").trim();
            let value = parts.next().unwrap_or("").trim();
            match key {
                "code" => code = Some(value.to_string()),
                "state" => state = Some(value.to_string()),
                _ => {}
            }
        }

        let code =
            code.ok_or_else(|| SsoError::MalformedToken("missing 'code' parameter".to_string()))?;
        let state = state
            .ok_or_else(|| SsoError::MalformedToken("missing 'state' parameter".to_string()))?;

        Ok(OidcCallback { code, state })
    }
}

// ── JWT signature verification ─────────────────────────────────────────────

/// Base64URL-decode (no padding) a JWT segment / JWK field.
fn b64url_decode(s: &str) -> Result<Vec<u8>, SsoError> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(s)
        .map_err(|e| SsoError::Base64Error(e.to_string()))
}

/// Split a compact JWT into its signing input (`header.payload` bytes as they
/// appear in the token) and the decoded signature bytes.
fn split_signing_input(token: &str) -> Result<(&[u8], Vec<u8>), SsoError> {
    let (signing_input, sig_b64) = token.rsplit_once('.').ok_or_else(|| {
        SsoError::MalformedToken("JWT must have three dot-separated segments".to_string())
    })?;
    // Ensure the signing input itself has exactly the header.payload shape.
    if signing_input.split('.').count() != 2 {
        return Err(SsoError::MalformedToken(
            "JWT must have three dot-separated segments".to_string(),
        ));
    }
    let sig = b64url_decode(sig_b64)?;
    Ok((signing_input.as_bytes(), sig))
}

/// Decode and parse the JOSE header (first segment) of a JWT.
fn parse_jwt_header(token: &str) -> Result<JwtHeader, SsoError> {
    let header_b64 = token.split('.').next().ok_or_else(|| {
        SsoError::MalformedToken("JWT must have three dot-separated segments".to_string())
    })?;
    let decoded = b64url_decode(header_b64)?;
    serde_json::from_slice(&decoded)
        .map_err(|e| SsoError::MalformedToken(format!("JWT header JSON parse error: {e}")))
}

/// The signature algorithms this validator accepts. Everything else — including
/// `none` — is rejected before any signature check is attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JwtAlg {
    /// RSASSA-PKCS1-v1_5 using SHA-256.
    Rs256,
    /// ECDSA using P-256 and SHA-256.
    Es256,
}

impl JwtAlg {
    fn parse(alg: &str) -> Result<Self, SsoError> {
        match alg {
            "RS256" => Ok(Self::Rs256),
            "ES256" => Ok(Self::Es256),
            other => Err(SsoError::UnsupportedAlgorithm(other.to_string())),
        }
    }

    /// Verify `signature` over `signing_input` using the public key in `jwk`.
    fn verify(self, signing_input: &[u8], signature: &[u8], jwk: &Jwk) -> Result<(), SsoError> {
        match self {
            Self::Rs256 => verify_rs256(signing_input, signature, jwk),
            Self::Es256 => verify_es256(signing_input, signature, jwk),
        }
    }
}

/// Verify an `RS256` signature using the Pure-Rust `rsa` crate.
fn verify_rs256(signing_input: &[u8], signature: &[u8], jwk: &Jwk) -> Result<(), SsoError> {
    use rsa::pkcs1v15::{Signature, VerifyingKey};
    use rsa::signature::Verifier;
    use rsa::{BigUint, RsaPublicKey};

    if jwk.kty != "RSA" {
        return Err(SsoError::InvalidKey(format!(
            "expected RSA key for RS256, got kty={}",
            jwk.kty
        )));
    }
    let n_b64 = jwk
        .n
        .as_deref()
        .ok_or_else(|| SsoError::InvalidKey("RSA JWK missing 'n'".to_string()))?;
    let e_b64 = jwk
        .e
        .as_deref()
        .ok_or_else(|| SsoError::InvalidKey("RSA JWK missing 'e'".to_string()))?;
    let n = BigUint::from_bytes_be(&b64url_decode(n_b64)?);
    let e = BigUint::from_bytes_be(&b64url_decode(e_b64)?);
    let public_key = RsaPublicKey::new(n, e)
        .map_err(|err| SsoError::InvalidKey(format!("invalid RSA public key: {err}")))?;

    // Use `rsa::sha2::Sha256` (digest 0.10) to match `rsa`'s expected version.
    let verifying_key = VerifyingKey::<rsa::sha2::Sha256>::new(public_key);
    let sig = Signature::try_from(signature).map_err(|_| SsoError::SignatureInvalid)?;
    verifying_key
        .verify(signing_input, &sig)
        .map_err(|_| SsoError::SignatureInvalid)
}

/// Verify an `ES256` signature using the Pure-Rust `p256` crate.
fn verify_es256(signing_input: &[u8], signature: &[u8], jwk: &Jwk) -> Result<(), SsoError> {
    use p256::ecdsa::signature::Verifier;
    use p256::ecdsa::{Signature, VerifyingKey};
    // `EncodedPoint` was removed from the p256 crate root in 0.14; the SEC1 point
    // type is now `p256::Sec1Point` (elliptic-curve 0.14 renamed `EncodedPoint`).
    use p256::Sec1Point;

    if jwk.kty != "EC" {
        return Err(SsoError::InvalidKey(format!(
            "expected EC key for ES256, got kty={}",
            jwk.kty
        )));
    }
    if let Some(crv) = jwk.crv.as_deref() {
        if crv != "P-256" {
            return Err(SsoError::InvalidKey(format!(
                "ES256 requires curve P-256, got {crv}"
            )));
        }
    }
    let x_b64 = jwk
        .x
        .as_deref()
        .ok_or_else(|| SsoError::InvalidKey("EC JWK missing 'x'".to_string()))?;
    let y_b64 = jwk
        .y
        .as_deref()
        .ok_or_else(|| SsoError::InvalidKey("EC JWK missing 'y'".to_string()))?;
    let x = left_pad_32(&b64url_decode(x_b64)?)?;
    let y = left_pad_32(&b64url_decode(y_b64)?)?;

    let point = Sec1Point::from_affine_coordinates(
        p256::FieldBytes::from_slice(&x),
        p256::FieldBytes::from_slice(&y),
        false,
    );
    let verifying_key = VerifyingKey::from_sec1_bytes(point.as_bytes())
        .map_err(|err| SsoError::InvalidKey(format!("invalid EC public key: {err}")))?;
    // JWS ES256 signatures are the fixed 64-byte r‖s concatenation.
    let sig = Signature::from_slice(signature).map_err(|_| SsoError::SignatureInvalid)?;
    verifying_key
        .verify(signing_input, &sig)
        .map_err(|_| SsoError::SignatureInvalid)
}

/// Left-pad an EC coordinate to exactly 32 bytes (P-256 field size), rejecting
/// anything longer.
fn left_pad_32(bytes: &[u8]) -> Result<[u8; 32], SsoError> {
    if bytes.len() > 32 {
        return Err(SsoError::InvalidKey(format!(
            "EC coordinate too long: {} bytes",
            bytes.len()
        )));
    }
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(bytes);
    Ok(out)
}

// ── JWT helpers ────────────────────────────────────────────────────────────

/// Parse the claims (payload) segment of a JWT without verifying the signature.
///
/// A JWT is three Base64URL-encoded segments separated by `.`.
/// This function decodes the *second* segment and parses it as JSON.
pub(crate) fn parse_jwt_claims(
    token: &str,
) -> Result<HashMap<String, serde_json::Value>, SsoError> {
    let segments: Vec<&str> = token.splitn(3, '.').collect();
    if segments.len() != 3 {
        return Err(SsoError::MalformedToken(
            "JWT must have three dot-separated segments".to_string(),
        ));
    }

    let payload_b64 = segments[1];
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|e| SsoError::Base64Error(e.to_string()))?;

    let json_str = std::str::from_utf8(&decoded)
        .map_err(|e| SsoError::MalformedToken(format!("payload is not valid UTF-8: {}", e)))?;

    let claims: HashMap<String, serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| SsoError::MalformedToken(format!("payload JSON parse error: {}", e)))?;

    Ok(claims)
}

/// Extract a list of strings from a claim that may be an array or a single string.
fn extract_string_list(claims: &HashMap<String, serde_json::Value>, key: &str) -> Vec<String> {
    match claims.get(key) {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Minimal percent-encoding for query-string values.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            b => {
                use std::fmt::Write as _;
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test key material ──────────────────────────────────────────────
    //
    // A throwaway RSA-2048 keypair (generated with OpenSSL) used ONLY to sign
    // test tokens. It is not, and must never be, a production key.
    const TEST_RSA_PKCS8_PEM: &str = concat!(
        "-----BEGIN PRIVATE KEY-----\n",
        "MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDq/z2c1qAgjqzJ\n",
        "yG4PCBptlY32sLnYcqOg9H6Q4Ql84xTM9YuP5ZOt1iXOsM8dAYToWOQzLoUIro1g\n",
        "2Rm2DH7QDbIt4PEaYop3AFhPHp1ZFjP6hRzTRINNBDxOSm0uz7H32YRTYokV6Ibn\n",
        "w6SdDPmJqFRgXf9qxBk+d9ljyLiIUQvOzQ+YOPHSos/k4HIPSo3f9U6Pwf+S3p71\n",
        "zV7nqgoRX5whJ8pMQqpX5ZMW9+cAa2zTPd+i8aZxmZHd9gySUqUoumKVm488ysSS\n",
        "nPwjbaA11dgAsDX9zziyZD/cKYFPF0DgLsL9wYcE87Qq52AYh2/zhG/g+FoxOje4\n",
        "QqWw25DHAgMBAAECggEABM/hNRr4AHKrex5NkqU51VCgrZKE27fNPfiDtvfEt/f2\n",
        "bxQAHZw33/FoqMjaFN/5FsDrO1kShFD+uCL58c5jsmL1aRcYGNA3waQSKtyXoEFi\n",
        "Ixkis/jNL4CMs5W2kqTSIh8kJIj6AabXTFunPUgMvBLkV2zVVBxb3/mYTADKNpBY\n",
        "QAEvsu/nToWWg49TgiixpA1k9RIYAQHcI8ZAugjnqiFnicyTthevWQ2cvBwqt4UB\n",
        "lwAASAf2P4qBeogkam+TFvtrnYi6rGskV+4rSRgkrbx5LAZsTVkMDC/eLmbScvyH\n",
        "t1TIpBt5PBw+YJ3hwkpPo+5fYJLWfvEj4rISYu38oQKBgQD11zPoKBlq638QGMxX\n",
        "SWB5o8XBuIw8N+K7GHYZWSUfoWlqy5kXE6A0YWBCdS8x3v84LJKokpZt5un/eACb\n",
        "q7q+o0RLblKLKasurNTCFrvQPtt1ftkidfunOJCbl7Nd5UBTckK80gG2PT5zfZKL\n",
        "B2M+EUV41AUF/tRfVyIQGR9h6QKBgQD0tVI/c0BKiqQfuKbJOduelaU/TamKTIMc\n",
        "ZjtTPJAJhkR3r5L9fWxkRbGcm4H39NZwKsC3oUmeHvQNDMFfn0A9yILyuhrP+BIW\n",
        "Q+t552ohQGRGLfujyfQc4HjFBthogU6XO3dIxFXg4iHkkQynQG+/w1ch+1wK/KxE\n",
        "wy+jhJZ/LwKBgQCbwoz1s6pfDvRDi6K0Tx5cE4Kxea8IXFRAPIBfERcvUkKLUpId\n",
        "h+bCKUwm7z5Gt8Y2ni8RtUawPVTW8v5Xo1e/f4w+yphr6au29/QZQPQgPiMn74W9\n",
        "isk2KuWcX2JaxGycMlHMdrZ085rE67PUeIrNgX3lz1ebc9i0y20ei/xROQKBgQC0\n",
        "FP/bC9CjSpXvdi7fZQG3Gb9K77c1vIq8Covb/HSvXazjO0T74SI0RImpi1NBC2AH\n",
        "mZ7LRBluEK9fLyTbXtGi5f1f7Q8wPwnocsFGq8ORhtaEQvCtn0BTQ+n8bMYzWf1h\n",
        "E/T7iuj8Hs38a7YZGzVhtLpZmqYou7t2uwFC3571JwKBgQDHnEyfkhdrs1GLEnZk\n",
        "LwENaWqvJoGzwWxv4cxA+oeU/olO81WL7zLS7IvvnYa3HXlMEMQZqeyUfBmbnD3a\n",
        "fnj8Y97nHrPh0yreJ7leg7GeY+Vw2QziOUGTeGbXgmqHrE6Amm9I7/Plfgjp0iRn\n",
        "qGQyLF8/TU5I4e0EAxaonr/FSA==\n",
        "-----END PRIVATE KEY-----\n",
    );

    fn b64url(bytes: impl AsRef<[u8]>) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    }

    // ---- RS256 -------------------------------------------------------------

    fn rsa_private_key() -> rsa::RsaPrivateKey {
        use rsa::pkcs8::DecodePrivateKey;
        rsa::RsaPrivateKey::from_pkcs8_pem(TEST_RSA_PKCS8_PEM).expect("parse test RSA key")
    }

    /// The public JWK matching [`TEST_RSA_PKCS8_PEM`], with `kid = "rsa-test"`.
    fn rsa_public_jwk() -> Jwk {
        use rsa::traits::PublicKeyParts;
        let pub_key = rsa_private_key().to_public_key();
        Jwk {
            kty: "RSA".to_string(),
            kid: Some("rsa-test".to_string()),
            alg: Some("RS256".to_string()),
            n: Some(b64url(pub_key.n().to_bytes_be())),
            e: Some(b64url(pub_key.e().to_bytes_be())),
            ..Default::default()
        }
    }

    /// Sign a JWT with the test RSA key. `header` and `payload` are JSON values.
    fn sign_rs256(header: &serde_json::Value, payload: &serde_json::Value) -> String {
        use rsa::pkcs1v15::SigningKey;
        use rsa::signature::{SignatureEncoding, Signer};
        let signing_input = format!(
            "{}.{}",
            b64url(header.to_string()),
            b64url(payload.to_string())
        );
        let key = SigningKey::<rsa::sha2::Sha256>::new(rsa_private_key());
        let sig = key.try_sign(signing_input.as_bytes()).expect("rsa sign");
        format!("{signing_input}.{}", b64url(sig.to_bytes()))
    }

    // ---- ES256 -------------------------------------------------------------

    fn es256_signing_key() -> p256::ecdsa::SigningKey {
        // A fixed non-zero scalar → deterministic test key (no RNG needed).
        let scalar = [7u8; 32];
        p256::ecdsa::SigningKey::from_bytes(&scalar.into()).expect("valid P-256 scalar")
    }

    fn es256_public_jwk() -> Jwk {
        let sk = es256_signing_key();
        let point = sk.verifying_key().to_sec1_point(false);
        Jwk {
            kty: "EC".to_string(),
            kid: Some("ec-test".to_string()),
            alg: Some("ES256".to_string()),
            crv: Some("P-256".to_string()),
            x: Some(b64url(point.x().expect("x coord"))),
            y: Some(b64url(point.y().expect("y coord"))),
            ..Default::default()
        }
    }

    fn sign_es256(header: &serde_json::Value, payload: &serde_json::Value) -> String {
        use p256::ecdsa::signature::Signer;
        let signing_input = format!(
            "{}.{}",
            b64url(header.to_string()),
            b64url(payload.to_string())
        );
        let sig: p256::ecdsa::Signature = es256_signing_key().sign(signing_input.as_bytes());
        format!("{signing_input}.{}", b64url(sig.to_bytes()))
    }

    fn future_exp() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64 + 3600)
            .unwrap_or(9_999_999_999)
    }

    fn make_config() -> SsoConfig {
        SsoConfig {
            provider_type: SsoProviderType::Oidc,
            issuer_url: "https://accounts.example.com".to_string(),
            client_id: "test-client".to_string(),
            redirect_uri: "https://app.example.com/callback".to_string(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        }
    }

    #[test]
    fn test_sso_config_oidc_serialization() {
        let config = make_config();
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: SsoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.issuer_url, config.issuer_url);
        assert_eq!(restored.client_id, config.client_id);
        assert_eq!(restored.redirect_uri, config.redirect_uri);
        assert_eq!(restored.scopes, config.scopes);
        assert_eq!(restored.provider_type, SsoProviderType::Oidc);
    }

    #[test]
    fn test_authorization_url_contains_params() {
        let validator = OidcValidator::new(make_config());
        let url = validator.authorization_url("state-abc", "nonce-xyz");
        assert!(url.contains("client_id=test-client"), "missing client_id");
        assert!(url.contains("redirect_uri="), "missing redirect_uri");
        assert!(url.contains("scope="), "missing scope");
        assert!(url.contains("state=state-abc"), "missing state");
        assert!(url.contains("nonce=nonce-xyz"), "missing nonce");
        assert!(url.contains("response_type=code"), "missing response_type");
    }

    #[test]
    fn test_parse_callback_valid() {
        let validator = OidcValidator::new(make_config());
        let cb = validator
            .parse_callback("code=authcode123&state=mystate456")
            .expect("parse callback");
        assert_eq!(cb.code, "authcode123");
        assert_eq!(cb.state, "mystate456");
    }

    fn rsa_validator() -> OidcValidator {
        OidcValidator::with_jwks(
            make_config(),
            Jwks {
                keys: vec![rsa_public_jwk()],
            },
        )
    }

    fn valid_payload() -> serde_json::Value {
        let exp = future_exp();
        serde_json::json!({
            "sub": "user-42",
            "iss": "https://accounts.example.com",
            "aud": "test-client",
            "exp": exp,
            "iat": exp - 60,
            "email": "alice@example.com",
            "name": "Alice Smith",
            "groups": ["engineers", "rdf-users"]
        })
    }

    /// A correctly-signed RS256 token with valid claims is accepted.
    #[test]
    fn test_validate_id_token_valid_rs256() {
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let token = sign_rs256(&header, &valid_payload());
        let user_info = rsa_validator()
            .validate_id_token(&token)
            .expect("valid RS256 token should be accepted");
        assert_eq!(user_info.subject, "user-42");
        assert_eq!(user_info.email.as_deref(), Some("alice@example.com"));
        assert!(user_info.groups.contains(&"engineers".to_string()));
    }

    /// A correctly-signed ES256 token with valid claims is accepted.
    #[test]
    fn test_validate_id_token_valid_es256() {
        let validator = OidcValidator::with_jwks(
            make_config(),
            Jwks {
                keys: vec![es256_public_jwk()],
            },
        );
        let header = serde_json::json!({"alg": "ES256", "typ": "JWT", "kid": "ec-test"});
        let token = sign_es256(&header, &valid_payload());
        let user_info = validator
            .validate_id_token(&token)
            .expect("valid ES256 token should be accepted");
        assert_eq!(user_info.subject, "user-42");
    }

    /// Adversarial: a token whose payload was tampered with after signing is
    /// rejected (the signature no longer covers the modified payload).
    #[test]
    fn test_validate_id_token_tampered_payload_rejected() {
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let token = sign_rs256(&header, &valid_payload());
        // Replace the payload segment with a forged one (attacker escalates sub),
        // keeping the original signature.
        let mut parts: Vec<&str> = token.split('.').collect();
        let forged = valid_payload();
        let mut forged = forged;
        forged["sub"] = serde_json::json!("attacker");
        let forged_b64 = b64url(forged.to_string());
        parts[1] = &forged_b64;
        let forged_token = parts.join(".");
        let err = rsa_validator()
            .validate_id_token(&forged_token)
            .expect_err("tampered token must be rejected");
        assert!(
            matches!(err, SsoError::SignatureInvalid),
            "expected SignatureInvalid, got: {err}"
        );
    }

    /// Adversarial: `alg: none` (unsigned) tokens are rejected outright.
    #[test]
    fn test_validate_id_token_alg_none_rejected() {
        let header = serde_json::json!({"alg": "none", "typ": "JWT"});
        // Craft an unsigned token: header.payload. (empty signature)
        let token = format!(
            "{}.{}.",
            b64url(header.to_string()),
            b64url(valid_payload().to_string())
        );
        let err = rsa_validator()
            .validate_id_token(&token)
            .expect_err("alg=none must be rejected");
        assert!(
            matches!(err, SsoError::UnsupportedAlgorithm(ref a) if a == "none"),
            "expected UnsupportedAlgorithm(none), got: {err}"
        );
    }

    /// Adversarial: an HS256 token (symmetric alg the validator does not accept)
    /// is rejected as an unsupported algorithm — never HMAC-verified.
    #[test]
    fn test_validate_id_token_hs256_rejected() {
        let header = serde_json::json!({"alg": "HS256", "typ": "JWT"});
        let token = format!(
            "{}.{}.aGVsbG8",
            b64url(header.to_string()),
            b64url(valid_payload().to_string())
        );
        let err = rsa_validator()
            .validate_id_token(&token)
            .expect_err("HS256 must be rejected");
        assert!(matches!(err, SsoError::UnsupportedAlgorithm(_)));
    }

    /// A properly-signed token verified against the WRONG key is rejected.
    #[test]
    fn test_validate_id_token_wrong_key_rejected() {
        // Sign with RSA but verify against an EC-only JWKS → key type mismatch /
        // no matching key.
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT"});
        let token = sign_rs256(&header, &valid_payload());
        let validator = OidcValidator::with_jwks(
            make_config(),
            Jwks {
                keys: vec![es256_public_jwk()],
            },
        );
        let err = validator
            .validate_id_token(&token)
            .expect_err("wrong key must reject");
        assert!(
            matches!(
                err,
                SsoError::KeyNotFound | SsoError::InvalidKey(_) | SsoError::SignatureInvalid
            ),
            "unexpected error: {err}"
        );
    }

    /// Fail-closed: without a configured JWKS, every token is rejected — even a
    /// structurally perfect, correctly-signed one.
    #[test]
    fn test_validate_id_token_fails_closed_without_jwks() {
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let token = sign_rs256(&header, &valid_payload());
        let err = OidcValidator::new(make_config())
            .validate_id_token(&token)
            .expect_err("no JWKS must fail closed");
        assert!(matches!(err, SsoError::SignatureVerificationUnavailable));
    }

    /// Claim checks run on a *signature-valid* token: expiry is enforced.
    #[test]
    fn test_validate_id_token_expired_after_sig() {
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let payload = serde_json::json!({
            "sub": "user-001",
            "iss": "https://accounts.example.com",
            "aud": "test-client",
            "exp": 1_000_000_i64,   // far in the past
            "iat": 900_000_i64,
        });
        let token = sign_rs256(&header, &payload);
        let err = rsa_validator()
            .validate_id_token(&token)
            .expect_err("expired token must be rejected");
        assert!(matches!(err, SsoError::TokenExpired), "got: {err}");
    }

    /// Claim checks run on a signature-valid token: wrong issuer is rejected.
    #[test]
    fn test_validate_id_token_wrong_issuer_after_sig() {
        let exp = future_exp();
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let payload = serde_json::json!({
            "sub": "user-001",
            "iss": "https://evil.example.com",
            "aud": "test-client",
            "exp": exp,
            "iat": exp - 60,
        });
        let token = sign_rs256(&header, &payload);
        let err = rsa_validator()
            .validate_id_token(&token)
            .expect_err("wrong issuer must be rejected");
        assert!(matches!(err, SsoError::InvalidIssuer { .. }), "got: {err}");
    }

    /// Nonce replay protection: a mismatched nonce is rejected.
    #[test]
    fn test_validate_id_token_nonce_mismatch() {
        let exp = future_exp();
        let header = serde_json::json!({"alg": "RS256", "typ": "JWT", "kid": "rsa-test"});
        let payload = serde_json::json!({
            "sub": "u", "iss": "https://accounts.example.com", "aud": "test-client",
            "exp": exp, "iat": exp - 60, "nonce": "server-nonce"
        });
        let token = sign_rs256(&header, &payload);
        let err = rsa_validator()
            .validate_id_token_with_nonce(&token, Some("attacker-nonce"))
            .expect_err("nonce mismatch must reject");
        assert!(matches!(err, SsoError::NonceMismatch));

        // Matching nonce is accepted.
        rsa_validator()
            .validate_id_token_with_nonce(&token, Some("server-nonce"))
            .expect("matching nonce accepted");
    }

    #[test]
    fn test_sso_user_info_fields() {
        let mut raw = HashMap::new();
        raw.insert(
            "custom_claim".to_string(),
            serde_json::Value::String("value".to_string()),
        );
        let info = SsoUserInfo {
            subject: "sub-100".to_string(),
            email: Some("bob@corp.com".to_string()),
            name: Some("Bob Builder".to_string()),
            groups: vec!["builders".to_string()],
            raw_claims: raw,
        };
        assert_eq!(info.subject, "sub-100");
        assert_eq!(info.email.as_deref(), Some("bob@corp.com"));
        assert_eq!(info.name.as_deref(), Some("Bob Builder"));
        assert_eq!(info.groups, vec!["builders"]);
        assert!(info.raw_claims.contains_key("custom_claim"));
    }
}
