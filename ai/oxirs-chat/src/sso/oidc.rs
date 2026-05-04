//! OIDC (OpenID Connect) provider abstraction for Enterprise SSO.
//!
//! Provides a lightweight OIDC validator that parses JWT ID-tokens without
//! performing cryptographic signature verification (signature verification
//! requires a runtime public-key fetch from the JWKS endpoint — that is
//! deferred to a future network-capable layer).  Validates the structural
//! claims: `exp`, `iss`, and `aud`.
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

// ── OidcValidator ──────────────────────────────────────────────────────────

/// Validates OIDC ID-tokens and builds authorization URLs.
///
/// Signature verification is **not** performed here — it requires fetching the
/// IdP's JWKS endpoint at runtime.  All structural / claims validation is done.
pub struct OidcValidator {
    config: SsoConfig,
}

impl OidcValidator {
    /// Create a new validator with the given provider configuration.
    pub fn new(config: SsoConfig) -> Self {
        Self { config }
    }

    /// Validate a JWT `id_token` and extract the normalised [`SsoUserInfo`].
    ///
    /// Validation checks performed:
    /// 1. JWT structure (three dot-separated segments).
    /// 2. Payload is valid base64url-encoded JSON.
    /// 3. `exp` claim is in the future.
    /// 4. `iss` claim matches `config.issuer_url`.
    /// 5. `aud` claim includes `config.client_id`.
    ///
    /// Signature verification is explicitly deferred.
    pub fn validate_id_token(&self, id_token: &str) -> Result<SsoUserInfo, SsoError> {
        let claims = parse_jwt_claims(id_token)?;

        // 1. Check expiry
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

        // 2. Check issuer
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

        // 3. Check audience
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

    /// Build a minimal JWT with the given payload claims.
    fn build_fake_jwt(payload: &serde_json::Value) -> String {
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"alg":"RS256","typ":"JWT"}"#);
        let payload_b64 =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.to_string());
        format!("{}.{}.fakesig", header, payload_b64)
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

    #[test]
    fn test_validate_id_token_expired() {
        let validator = OidcValidator::new(make_config());
        let payload = serde_json::json!({
            "sub": "user-001",
            "iss": "https://accounts.example.com",
            "aud": "test-client",
            "exp": 1_000_000_i64,  // far in the past
            "iat": 900_000_i64,
            "email": "alice@example.com"
        });
        let token = build_fake_jwt(&payload);
        let err = validator
            .validate_id_token(&token)
            .expect_err("should fail with expired token");
        assert!(
            matches!(err, SsoError::TokenExpired),
            "expected TokenExpired, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_id_token_wrong_issuer() {
        let validator = OidcValidator::new(make_config());
        let future_exp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64 + 3600)
            .unwrap_or(9_999_999_999);
        let payload = serde_json::json!({
            "sub": "user-001",
            "iss": "https://evil.example.com",
            "aud": "test-client",
            "exp": future_exp,
            "iat": future_exp - 60,
            "email": "alice@example.com"
        });
        let token = build_fake_jwt(&payload);
        let err = validator
            .validate_id_token(&token)
            .expect_err("should fail with wrong issuer");
        assert!(
            matches!(err, SsoError::InvalidIssuer { .. }),
            "expected InvalidIssuer, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_id_token_valid_claims() {
        let validator = OidcValidator::new(make_config());
        let future_exp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64 + 3600)
            .unwrap_or(9_999_999_999);
        let payload = serde_json::json!({
            "sub": "user-42",
            "iss": "https://accounts.example.com",
            "aud": "test-client",
            "exp": future_exp,
            "iat": future_exp - 60,
            "email": "alice@example.com",
            "name": "Alice Smith",
            "groups": ["engineers", "rdf-users"]
        });
        let token = build_fake_jwt(&payload);
        let user_info = validator
            .validate_id_token(&token)
            .expect("valid token should be accepted");
        assert_eq!(user_info.subject, "user-42");
        assert_eq!(user_info.email.as_deref(), Some("alice@example.com"));
        assert_eq!(user_info.name.as_deref(), Some("Alice Smith"));
        assert!(user_info.groups.contains(&"engineers".to_string()));
        assert!(user_info.groups.contains(&"rdf-users".to_string()));
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
