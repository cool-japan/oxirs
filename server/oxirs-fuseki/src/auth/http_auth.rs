//! HTTP Authentication for SPARQL endpoints
//!
//! Provides a straightforward auth middleware covering:
//! - No authentication (open access)
//! - API key via a request header
//! - HTTP Basic authentication
//! - OAuth2 Bearer token validation
//!
//! This module is intentionally self-contained so it can be unit-tested
//! without a running HTTP server.

use std::collections::HashMap;
use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};

/// Actions a SPARQL client may request, used for authorization decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SparqlAction {
    /// SPARQL SELECT / ASK / CONSTRUCT / DESCRIBE
    Query,
    /// SPARQL INSERT / DELETE / LOAD / CLEAR
    Update,
    /// Graph Store Protocol GET
    GspRead,
    /// Graph Store Protocol PUT / POST / DELETE
    GspWrite,
    /// Administrative operations (dataset create/delete, server config)
    Admin,
}

/// Claims extracted from a validated bearer token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenClaims {
    /// Subject (user identifier, e.g. `"user@example.com"`).
    pub subject: String,
    /// OAuth2 scopes granted to the token.
    pub scopes: Vec<String>,
    /// UNIX timestamp (seconds since epoch) at which the token expires,
    /// or `None` if the token has no expiry.
    pub expires_at: Option<u64>,
}

impl TokenClaims {
    /// Returns `true` when the token has a recorded expiry and it is in
    /// the past relative to the current system time.
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            return now >= exp;
        }
        false
    }

    /// Returns `true` when the claims contain the given scope string.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

/// Authentication configuration for an endpoint.
pub enum AuthConfig {
    /// No authentication required; every request is allowed.
    None,
    /// A static API key that must appear in the specified header
    /// (e.g. `"X-API-Key"`).
    ApiKey {
        /// Name of the HTTP header that must carry the API key.
        header: String,
        /// The expected key value.
        key: String,
    },
    /// OAuth2 Bearer token validated by the supplied [`TokenValidator`].
    Bearer { validator: Arc<dyn TokenValidator> },
    /// HTTP Basic authentication checked against a static credential map.
    /// The map key is the username, value is the plain-text password.
    Basic {
        credentials: HashMap<String, String>,
    },
}

/// Authentication errors.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing authentication credentials")]
    MissingCredentials,

    #[error("Invalid token: {reason}")]
    InvalidToken { reason: String },

    #[error("Token expired")]
    TokenExpired,

    #[error("Insufficient permissions for {action}")]
    InsufficientPermissions { action: String },
}

/// Trait for pluggable bearer-token validators (e.g. JWT, opaque tokens).
pub trait TokenValidator: Send + Sync {
    /// Validate `token` and return the extracted claims, or an [`AuthError`].
    fn validate(&self, token: &str) -> Result<TokenClaims, AuthError>;
}

/// Minimal JWT validator that decodes the header/claims from Base64-URL and
/// verifies expiry. It does **not** validate cryptographic signatures — that
/// should be done in production by replacing this with a proper OIDC library.
pub struct JwtValidator {
    /// Expected `iss` claim (issuer).
    pub issuer: String,
    /// Expected `aud` claim (audience).
    pub audience: String,
}

impl JwtValidator {
    /// Create a new validator for the given issuer and audience.
    pub fn new(issuer: impl Into<String>, audience: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
        }
    }
}

impl TokenValidator for JwtValidator {
    fn validate(&self, token: &str) -> Result<TokenClaims, AuthError> {
        // A JWT has the form header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidToken {
                reason: "JWT must have exactly three dot-separated parts".to_string(),
            });
        }

        // Decode the payload (middle part) using Base64URL (no padding)
        let payload_bytes = base64_url_decode(parts[1]).map_err(|e| AuthError::InvalidToken {
            reason: format!("Base64URL decode error: {e}"),
        })?;

        let payload_str =
            String::from_utf8(payload_bytes).map_err(|_| AuthError::InvalidToken {
                reason: "Payload is not valid UTF-8".to_string(),
            })?;

        let claims: serde_json::Value =
            serde_json::from_str(&payload_str).map_err(|e| AuthError::InvalidToken {
                reason: format!("JSON parse error: {e}"),
            })?;

        // Validate issuer
        if let Some(iss) = claims.get("iss").and_then(|v| v.as_str()) {
            if iss != self.issuer {
                return Err(AuthError::InvalidToken {
                    reason: format!("Issuer mismatch: expected '{}', got '{iss}'", self.issuer),
                });
            }
        }

        // Validate audience
        if let Some(aud) = claims.get("aud") {
            let matches = if let Some(s) = aud.as_str() {
                s == self.audience
            } else if let Some(arr) = aud.as_array() {
                arr.iter()
                    .any(|a| a.as_str().is_some_and(|s| s == self.audience))
            } else {
                false
            };
            if !matches {
                return Err(AuthError::InvalidToken {
                    reason: format!("Audience mismatch: expected '{}'", self.audience),
                });
            }
        }

        // Extract expiry
        let expires_at = claims.get("exp").and_then(|v| v.as_u64());

        // Check expiry
        if let Some(exp) = expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if now >= exp {
                return Err(AuthError::TokenExpired);
            }
        }

        // Extract subject
        let subject = claims
            .get("sub")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Extract scopes
        let scopes = claims
            .get("scope")
            .and_then(|v| v.as_str())
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();

        Ok(TokenClaims {
            subject,
            scopes,
            expires_at,
        })
    }
}

/// Decode a Base64URL-encoded string (with or without padding).
fn base64_url_decode(input: &str) -> Result<Vec<u8>, String> {
    // Convert URL-safe characters and add padding
    let standard = input.replace('-', "+").replace('_', "/");
    let padded = match standard.len() % 4 {
        2 => format!("{standard}=="),
        3 => format!("{standard}="),
        _ => standard,
    };
    STANDARD
        .decode(padded.as_bytes())
        .map_err(|e| format!("{e}"))
}

/// A minimal representation of HTTP headers for use in tests without a real
/// HTTP framework dependency.
#[derive(Default, Debug, Clone)]
pub struct HeaderMap {
    inner: HashMap<String, String>,
}

impl HeaderMap {
    /// Insert a header (header names are lowercased for case-insensitive lookup).
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.inner.insert(name.into().to_lowercase(), value.into());
    }

    /// Retrieve a header value by name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&str> {
        self.inner.get(&name.to_lowercase()).map(String::as_str)
    }
}

/// HTTP authentication middleware.
///
/// Call [`AuthMiddleware::authenticate`] to extract and validate credentials
/// from request headers, then [`AuthMiddleware::authorize`] to check that the
/// resulting claims permit the requested SPARQL action.
pub struct AuthMiddleware {
    config: AuthConfig,
}

impl AuthMiddleware {
    /// Create a new middleware with the given configuration.
    pub fn new(config: AuthConfig) -> Self {
        Self { config }
    }

    /// Authenticate the request from the supplied headers.
    ///
    /// Returns:
    /// - `Ok(None)` — auth is disabled ([`AuthConfig::None`]).
    /// - `Ok(Some(claims))` — the request is authenticated.
    /// - `Err(AuthError)` — the credentials are missing, invalid, or expired.
    pub fn authenticate(&self, headers: &HeaderMap) -> Result<Option<TokenClaims>, AuthError> {
        match &self.config {
            AuthConfig::None => Ok(None),

            AuthConfig::ApiKey { header, key } => {
                let provided = headers.get(header).ok_or(AuthError::MissingCredentials)?;
                if provided != key.as_str() {
                    return Err(AuthError::InvalidToken {
                        reason: "API key mismatch".to_string(),
                    });
                }
                Ok(Some(TokenClaims {
                    subject: "api-key-user".to_string(),
                    scopes: vec!["sparql".to_string()],
                    expires_at: None,
                }))
            }

            AuthConfig::Bearer { validator } => {
                let auth_header = headers
                    .get("authorization")
                    .ok_or(AuthError::MissingCredentials)?;
                let token = auth_header
                    .strip_prefix("Bearer ")
                    .or_else(|| auth_header.strip_prefix("bearer "))
                    .ok_or(AuthError::InvalidToken {
                        reason: "Authorization header must start with 'Bearer '".to_string(),
                    })?;
                let claims = validator.validate(token)?;
                if claims.is_expired() {
                    return Err(AuthError::TokenExpired);
                }
                Ok(Some(claims))
            }

            AuthConfig::Basic { credentials } => {
                let auth_header = headers
                    .get("authorization")
                    .ok_or(AuthError::MissingCredentials)?;
                let encoded = auth_header
                    .strip_prefix("Basic ")
                    .or_else(|| auth_header.strip_prefix("basic "))
                    .ok_or(AuthError::InvalidToken {
                        reason: "Authorization header must start with 'Basic '".to_string(),
                    })?;
                let decoded =
                    STANDARD
                        .decode(encoded.as_bytes())
                        .map_err(|_| AuthError::InvalidToken {
                            reason: "Invalid Base64 encoding".to_string(),
                        })?;
                let credential =
                    String::from_utf8(decoded).map_err(|_| AuthError::InvalidToken {
                        reason: "Credentials are not valid UTF-8".to_string(),
                    })?;
                let (username, password) =
                    credential.split_once(':').ok_or(AuthError::InvalidToken {
                        reason: "Basic auth must be 'username:password'".to_string(),
                    })?;
                match credentials.get(username) {
                    Some(expected) if expected == password => Ok(Some(TokenClaims {
                        subject: username.to_string(),
                        scopes: vec!["sparql".to_string()],
                        expires_at: None,
                    })),
                    Some(_) => Err(AuthError::InvalidToken {
                        reason: "Incorrect password".to_string(),
                    }),
                    None => Err(AuthError::InvalidToken {
                        reason: format!("Unknown user '{username}'"),
                    }),
                }
            }
        }
    }

    /// Authorize the given claims (from `authenticate`) for the requested
    /// [`SparqlAction`].
    ///
    /// `None` claims (open access) permit every action.
    /// Claims with scope `"sparql:admin"` permit every action.
    /// Claims with scope `"sparql:write"` permit Query and Update.
    /// Claims with scope `"sparql"` or `"sparql:read"` permit Query / GSP read only.
    pub fn authorize(
        &self,
        claims: &Option<TokenClaims>,
        action: SparqlAction,
    ) -> Result<(), AuthError> {
        let claims = match claims {
            None => return Ok(()), // AuthConfig::None
            Some(c) => c,
        };

        if claims.has_scope("sparql:admin") {
            return Ok(());
        }

        let permitted = match action {
            SparqlAction::Admin => false, // needs sparql:admin
            SparqlAction::Update | SparqlAction::GspWrite => {
                claims.has_scope("sparql:write") || claims.has_scope("sparql:admin")
            }
            SparqlAction::Query | SparqlAction::GspRead => {
                claims.has_scope("sparql")
                    || claims.has_scope("sparql:read")
                    || claims.has_scope("sparql:write")
            }
        };

        if permitted {
            Ok(())
        } else {
            Err(AuthError::InsufficientPermissions {
                action: format!("{action:?}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── Helper token builders ─────────────────────────────────────────────────

    /// Build a minimal unsigned JWT with the given payload JSON.
    fn make_jwt(payload_json: &str) -> String {
        let header = STANDARD.encode(r#"{"alg":"none","typ":"JWT"}"#);
        let payload = STANDARD.encode(payload_json);
        format!("{header}.{payload}.")
    }

    /// Future timestamp (year ~2100)
    const FAR_FUTURE: u64 = 4_102_444_800;
    /// Past timestamp (year 2000)
    const PAST: u64 = 946_684_800;

    fn make_jwt_valid(issuer: &str, audience: &str) -> String {
        make_jwt(&format!(
            r#"{{"iss":"{issuer}","aud":"{audience}","sub":"alice","scope":"sparql","exp":{FAR_FUTURE}}}"#
        ))
    }

    // ── AuthConfig::None ─────────────────────────────────────────────────────

    #[test]
    fn test_none_auth_always_ok() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        let headers = HeaderMap::default();
        let result = mw.authenticate(&headers);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_none_auth_no_credentials_needed() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        // Even with a garbage header the result is ok
        let mut h = HeaderMap::default();
        h.insert("authorization", "garbage");
        assert!(mw.authenticate(&h).is_ok());
    }

    #[test]
    fn test_none_auth_authorizes_every_action() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        let claims: Option<TokenClaims> = None;
        for action in [
            SparqlAction::Query,
            SparqlAction::Update,
            SparqlAction::GspRead,
            SparqlAction::GspWrite,
            SparqlAction::Admin,
        ] {
            assert!(mw.authorize(&claims, action).is_ok());
        }
    }

    // ── AuthConfig::ApiKey ────────────────────────────────────────────────────

    #[test]
    fn test_api_key_valid() {
        let mw = AuthMiddleware::new(AuthConfig::ApiKey {
            header: "X-API-Key".to_string(),
            key: "secret-key".to_string(),
        });
        let mut h = HeaderMap::default();
        h.insert("X-API-Key", "secret-key");
        assert!(mw.authenticate(&h).is_ok());
    }

    #[test]
    fn test_api_key_invalid() {
        let mw = AuthMiddleware::new(AuthConfig::ApiKey {
            header: "X-API-Key".to_string(),
            key: "secret-key".to_string(),
        });
        let mut h = HeaderMap::default();
        h.insert("X-API-Key", "wrong-key");
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_api_key_missing_header() {
        let mw = AuthMiddleware::new(AuthConfig::ApiKey {
            header: "X-API-Key".to_string(),
            key: "secret-key".to_string(),
        });
        assert!(matches!(
            mw.authenticate(&HeaderMap::default()),
            Err(AuthError::MissingCredentials)
        ));
    }

    #[test]
    fn test_api_key_case_insensitive_header_lookup() {
        let mw = AuthMiddleware::new(AuthConfig::ApiKey {
            header: "x-api-key".to_string(),
            key: "k".to_string(),
        });
        let mut h = HeaderMap::default();
        h.insert("X-API-Key", "k"); // HeaderMap lowercases names
        assert!(mw.authenticate(&h).is_ok());
    }

    #[test]
    fn test_api_key_returns_claims_with_subject() {
        let mw = AuthMiddleware::new(AuthConfig::ApiKey {
            header: "X-API-Key".to_string(),
            key: "k".to_string(),
        });
        let mut h = HeaderMap::default();
        h.insert("X-API-Key", "k");
        let claims = mw.authenticate(&h).unwrap().unwrap();
        assert_eq!(claims.subject, "api-key-user");
    }

    // ── AuthConfig::Basic ─────────────────────────────────────────────────────

    fn basic_auth_header(user: &str, pass: &str) -> String {
        format!("Basic {}", STANDARD.encode(format!("{user}:{pass}")))
    }

    #[test]
    fn test_basic_auth_valid() {
        let mut creds = HashMap::new();
        creds.insert("alice".to_string(), "password123".to_string());
        let mw = AuthMiddleware::new(AuthConfig::Basic { credentials: creds });
        let mut h = HeaderMap::default();
        h.insert("authorization", basic_auth_header("alice", "password123"));
        let claims = mw.authenticate(&h).unwrap().unwrap();
        assert_eq!(claims.subject, "alice");
    }

    #[test]
    fn test_basic_auth_wrong_password() {
        let mut creds = HashMap::new();
        creds.insert("alice".to_string(), "correct".to_string());
        let mw = AuthMiddleware::new(AuthConfig::Basic { credentials: creds });
        let mut h = HeaderMap::default();
        h.insert("authorization", basic_auth_header("alice", "wrong"));
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_basic_auth_unknown_user() {
        let mw = AuthMiddleware::new(AuthConfig::Basic {
            credentials: HashMap::new(),
        });
        let mut h = HeaderMap::default();
        h.insert("authorization", basic_auth_header("nobody", "pass"));
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_basic_auth_missing_header() {
        let mw = AuthMiddleware::new(AuthConfig::Basic {
            credentials: HashMap::new(),
        });
        assert!(matches!(
            mw.authenticate(&HeaderMap::default()),
            Err(AuthError::MissingCredentials)
        ));
    }

    #[test]
    fn test_basic_auth_malformed_base64() {
        let mw = AuthMiddleware::new(AuthConfig::Basic {
            credentials: HashMap::new(),
        });
        let mut h = HeaderMap::default();
        h.insert("authorization", "Basic not-valid-base64!!!");
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_basic_auth_no_colon_separator() {
        let mw = AuthMiddleware::new(AuthConfig::Basic {
            credentials: HashMap::new(),
        });
        let mut h = HeaderMap::default();
        // "usernameonly" with no colon
        h.insert(
            "authorization",
            format!("Basic {}", STANDARD.encode("usernameonly")),
        );
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_basic_auth_multiple_users() {
        let mut creds = HashMap::new();
        creds.insert("alice".to_string(), "apass".to_string());
        creds.insert("bob".to_string(), "bpass".to_string());
        let mw = AuthMiddleware::new(AuthConfig::Basic { credentials: creds });

        let mut h1 = HeaderMap::default();
        h1.insert("authorization", basic_auth_header("alice", "apass"));
        assert!(mw.authenticate(&h1).is_ok());

        let mut h2 = HeaderMap::default();
        h2.insert("authorization", basic_auth_header("bob", "bpass"));
        assert!(mw.authenticate(&h2).is_ok());
    }

    // ── AuthConfig::Bearer / JwtValidator ─────────────────────────────────────

    #[test]
    fn test_bearer_valid_jwt() {
        let validator = Arc::new(JwtValidator::new("https://issuer.example.com", "my-api"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer {
            validator: Arc::clone(&validator) as Arc<dyn TokenValidator>,
        });
        let token = make_jwt_valid("https://issuer.example.com", "my-api");
        let mut h = HeaderMap::default();
        h.insert("authorization", format!("Bearer {token}"));
        let claims = mw.authenticate(&h).unwrap().unwrap();
        assert_eq!(claims.subject, "alice");
    }

    #[test]
    fn test_bearer_missing_header() {
        let validator = Arc::new(JwtValidator::new("iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        assert!(matches!(
            mw.authenticate(&HeaderMap::default()),
            Err(AuthError::MissingCredentials)
        ));
    }

    #[test]
    fn test_bearer_wrong_scheme() {
        let validator = Arc::new(JwtValidator::new("iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let mut h = HeaderMap::default();
        h.insert("authorization", "Basic dXNlcjpwYXNz");
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_bearer_expired_jwt() {
        let validator = Arc::new(JwtValidator::new("iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let token = make_jwt(&format!(
            r#"{{"iss":"iss","aud":"aud","sub":"alice","scope":"sparql","exp":{PAST}}}"#
        ));
        let mut h = HeaderMap::default();
        h.insert("authorization", format!("Bearer {token}"));
        assert!(matches!(mw.authenticate(&h), Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_bearer_issuer_mismatch() {
        let validator = Arc::new(JwtValidator::new("expected-iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let token = make_jwt_valid("wrong-iss", "aud");
        let mut h = HeaderMap::default();
        h.insert("authorization", format!("Bearer {token}"));
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_bearer_audience_mismatch() {
        let validator = Arc::new(JwtValidator::new("iss", "expected-aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let token = make_jwt_valid("iss", "wrong-aud");
        let mut h = HeaderMap::default();
        h.insert("authorization", format!("Bearer {token}"));
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_bearer_not_three_parts() {
        let validator = Arc::new(JwtValidator::new("iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let mut h = HeaderMap::default();
        h.insert("authorization", "Bearer notavalidjwt");
        assert!(matches!(
            mw.authenticate(&h),
            Err(AuthError::InvalidToken { .. })
        ));
    }

    #[test]
    fn test_bearer_scopes_parsed() {
        let validator = Arc::new(JwtValidator::new("iss", "aud"));
        let mw = AuthMiddleware::new(AuthConfig::Bearer { validator });
        let token = make_jwt(&format!(
            r#"{{"iss":"iss","aud":"aud","sub":"bob","scope":"sparql:read sparql:write","exp":{FAR_FUTURE}}}"#
        ));
        let mut h = HeaderMap::default();
        h.insert("authorization", format!("Bearer {token}"));
        let claims = mw.authenticate(&h).unwrap().unwrap();
        assert!(claims.has_scope("sparql:read"));
        assert!(claims.has_scope("sparql:write"));
        assert!(!claims.has_scope("sparql:admin"));
    }

    // ── Authorization ─────────────────────────────────────────────────────────

    fn claims_with_scopes(scopes: &[&str]) -> Option<TokenClaims> {
        Some(TokenClaims {
            subject: "user".to_string(),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            expires_at: None,
        })
    }

    #[test]
    fn test_authorize_sparql_scope_allows_query() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(mw
            .authorize(&claims_with_scopes(&["sparql"]), SparqlAction::Query)
            .is_ok());
    }

    #[test]
    fn test_authorize_sparql_scope_allows_gsp_read() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(mw
            .authorize(&claims_with_scopes(&["sparql"]), SparqlAction::GspRead)
            .is_ok());
    }

    #[test]
    fn test_authorize_sparql_read_scope_allows_query() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(mw
            .authorize(&claims_with_scopes(&["sparql:read"]), SparqlAction::Query)
            .is_ok());
    }

    #[test]
    fn test_authorize_sparql_scope_denies_update() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(matches!(
            mw.authorize(&claims_with_scopes(&["sparql"]), SparqlAction::Update),
            Err(AuthError::InsufficientPermissions { .. })
        ));
    }

    #[test]
    fn test_authorize_write_scope_allows_update() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(mw
            .authorize(&claims_with_scopes(&["sparql:write"]), SparqlAction::Update)
            .is_ok());
    }

    #[test]
    fn test_authorize_write_scope_allows_gsp_write() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(mw
            .authorize(
                &claims_with_scopes(&["sparql:write"]),
                SparqlAction::GspWrite
            )
            .is_ok());
    }

    #[test]
    fn test_authorize_admin_scope_allows_all() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        let claims = claims_with_scopes(&["sparql:admin"]);
        for action in [
            SparqlAction::Query,
            SparqlAction::Update,
            SparqlAction::GspRead,
            SparqlAction::GspWrite,
            SparqlAction::Admin,
        ] {
            assert!(mw.authorize(&claims, action).is_ok());
        }
    }

    #[test]
    fn test_authorize_read_scope_denies_admin() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(matches!(
            mw.authorize(&claims_with_scopes(&["sparql:read"]), SparqlAction::Admin),
            Err(AuthError::InsufficientPermissions { .. })
        ));
    }

    #[test]
    fn test_authorize_write_scope_denies_admin() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        assert!(matches!(
            mw.authorize(&claims_with_scopes(&["sparql:write"]), SparqlAction::Admin),
            Err(AuthError::InsufficientPermissions { .. })
        ));
    }

    #[test]
    fn test_authorize_no_scopes_denies_everything() {
        let mw = AuthMiddleware::new(AuthConfig::None);
        let claims = claims_with_scopes(&[]);
        for action in [
            SparqlAction::Query,
            SparqlAction::Update,
            SparqlAction::GspRead,
            SparqlAction::GspWrite,
            SparqlAction::Admin,
        ] {
            assert!(mw.authorize(&claims, action.clone()).is_err());
        }
    }

    // ── TokenClaims helpers ───────────────────────────────────────────────────

    #[test]
    fn test_claims_is_expired_past_timestamp() {
        let c = TokenClaims {
            subject: "u".to_string(),
            scopes: vec![],
            expires_at: Some(PAST),
        };
        assert!(c.is_expired());
    }

    #[test]
    fn test_claims_is_not_expired_future_timestamp() {
        let c = TokenClaims {
            subject: "u".to_string(),
            scopes: vec![],
            expires_at: Some(FAR_FUTURE),
        };
        assert!(!c.is_expired());
    }

    #[test]
    fn test_claims_is_not_expired_when_no_expiry() {
        let c = TokenClaims {
            subject: "u".to_string(),
            scopes: vec![],
            expires_at: None,
        };
        assert!(!c.is_expired());
    }

    #[test]
    fn test_claims_has_scope_found() {
        let c = TokenClaims {
            subject: "u".to_string(),
            scopes: vec!["sparql".to_string(), "openid".to_string()],
            expires_at: None,
        };
        assert!(c.has_scope("sparql"));
        assert!(c.has_scope("openid"));
    }

    #[test]
    fn test_claims_has_scope_not_found() {
        let c = TokenClaims {
            subject: "u".to_string(),
            scopes: vec!["sparql".to_string()],
            expires_at: None,
        };
        assert!(!c.has_scope("admin"));
    }

    // ── AuthError display ─────────────────────────────────────────────────────

    #[test]
    fn test_auth_error_missing_credentials_display() {
        let msg = format!("{}", AuthError::MissingCredentials);
        assert!(msg.to_lowercase().contains("missing"));
    }

    #[test]
    fn test_auth_error_invalid_token_display() {
        let msg = format!(
            "{}",
            AuthError::InvalidToken {
                reason: "bad sig".to_string()
            }
        );
        assert!(msg.contains("bad sig"));
    }

    #[test]
    fn test_auth_error_token_expired_display() {
        let msg = format!("{}", AuthError::TokenExpired);
        assert!(msg.to_lowercase().contains("expired"));
    }

    #[test]
    fn test_auth_error_insufficient_permissions_display() {
        let msg = format!(
            "{}",
            AuthError::InsufficientPermissions {
                action: "Admin".to_string()
            }
        );
        assert!(msg.contains("Admin"));
    }

    // ── SparqlAction coverage ─────────────────────────────────────────────────

    #[test]
    fn test_sparql_action_debug() {
        assert_eq!(format!("{:?}", SparqlAction::Query), "Query");
        assert_eq!(format!("{:?}", SparqlAction::Update), "Update");
        assert_eq!(format!("{:?}", SparqlAction::GspRead), "GspRead");
        assert_eq!(format!("{:?}", SparqlAction::GspWrite), "GspWrite");
        assert_eq!(format!("{:?}", SparqlAction::Admin), "Admin");
    }

    #[test]
    fn test_sparql_action_equality() {
        assert_eq!(SparqlAction::Query, SparqlAction::Query);
        assert_ne!(SparqlAction::Query, SparqlAction::Update);
    }

    // ── HeaderMap ─────────────────────────────────────────────────────────────

    #[test]
    fn test_header_map_case_insensitive() {
        let mut h = HeaderMap::default();
        h.insert("Content-Type", "application/sparql-query");
        assert_eq!(h.get("content-type"), Some("application/sparql-query"));
        assert_eq!(h.get("CONTENT-TYPE"), Some("application/sparql-query"));
    }

    #[test]
    fn test_header_map_missing_key_returns_none() {
        let h = HeaderMap::default();
        assert!(h.get("x-not-there").is_none());
    }

    #[test]
    fn test_header_map_overwrite() {
        let mut h = HeaderMap::default();
        h.insert("key", "first");
        h.insert("key", "second");
        assert_eq!(h.get("key"), Some("second"));
    }
}
