//! Authentication middleware for SPARQL endpoints.
//!
//! Supports bearer token (JWT-like) validation, API key authentication
//! (header and query-param), session token management, role-based access
//! control (admin/reader/writer), per-user rate limiting, and anonymous
//! access policies.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Roles
// ────────────────────────────────────────────────────────────────────────────

/// Access role for a user or session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full administrative privileges.
    Admin,
    /// Read-only access to datasets.
    Reader,
    /// Read and write access.
    Writer,
    /// Unauthenticated / anonymous visitor.
    Anonymous,
}

impl Role {
    /// Returns `true` if the role permits read operations.
    pub fn can_read(&self) -> bool {
        matches!(self, Role::Admin | Role::Reader | Role::Writer)
    }

    /// Returns `true` if the role permits write (update) operations.
    pub fn can_write(&self) -> bool {
        matches!(self, Role::Admin | Role::Writer)
    }

    /// Returns `true` if the role has administrative privileges.
    pub fn is_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }

    /// Parse a role from a string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Role> {
        match s.trim().to_ascii_lowercase().as_str() {
            "admin" => Some(Role::Admin),
            "reader" | "read" => Some(Role::Reader),
            "writer" | "write" => Some(Role::Writer),
            "anonymous" | "anon" => Some(Role::Anonymous),
            _ => None,
        }
    }

    /// A short label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Role::Admin => "admin",
            Role::Reader => "reader",
            Role::Writer => "writer",
            Role::Anonymous => "anonymous",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Authentication result
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of an authentication attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Successfully authenticated.
    Authenticated(AuthIdentity),
    /// Authentication failed.
    Denied(String),
    /// No credentials were provided; may be allowed for anonymous access.
    NoCredentials,
}

/// Identity of an authenticated principal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthIdentity {
    /// Unique user identifier (e.g., subject claim from JWT or API key id).
    pub user_id: String,
    /// Role assigned to this identity.
    pub role: Role,
    /// Optional display name.
    pub display_name: Option<String>,
    /// Source of the identity (bearer, api_key, session).
    pub auth_method: String,
}

impl AuthIdentity {
    pub fn new(user_id: impl Into<String>, role: Role, method: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            role,
            display_name: None,
            auth_method: method.into(),
        }
    }

    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Bearer token (JWT-like)
// ────────────────────────────────────────────────────────────────────────────

/// Parsed claims from a bearer token (simplified JWT-like structure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenClaims {
    /// Subject (user ID).
    pub sub: String,
    /// Issued-at timestamp (seconds since epoch).
    pub iat: u64,
    /// Expiry timestamp (seconds since epoch).
    pub exp: u64,
    /// Role claim.
    pub role: Role,
    /// Token issuer.
    pub iss: Option<String>,
}

impl TokenClaims {
    /// Returns `true` if the token has expired relative to `now_secs`.
    pub fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.exp
    }

    /// Remaining seconds before expiry. Returns 0 if already expired.
    pub fn ttl(&self, now_secs: u64) -> u64 {
        self.exp.saturating_sub(now_secs)
    }
}

/// A simple bearer-token validator.
///
/// In production this would verify a real JWT signature. Here we parse a
/// simplified `sub:iat:exp:role` format separated by dots for testing purposes.
pub struct BearerValidator {
    /// Accepted issuers (empty = accept any).
    pub accepted_issuers: Vec<String>,
    /// Clock skew tolerance in seconds.
    pub clock_skew_secs: u64,
}

impl Default for BearerValidator {
    fn default() -> Self {
        Self {
            accepted_issuers: Vec::new(),
            clock_skew_secs: 30,
        }
    }
}

impl BearerValidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_issuer(mut self, iss: impl Into<String>) -> Self {
        self.accepted_issuers.push(iss.into());
        self
    }

    /// Parse and validate a bearer token string.
    ///
    /// Expected format: `sub.iat.exp.role[.iss]`
    pub fn validate(&self, token: &str, now_secs: u64) -> Result<TokenClaims, String> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() < 4 {
            return Err("invalid token format: expected at least 4 dot-separated segments".into());
        }

        let sub = parts[0].to_string();
        if sub.is_empty() {
            return Err("token subject is empty".into());
        }

        let iat: u64 = parts[1]
            .parse()
            .map_err(|_| "invalid iat timestamp".to_string())?;
        let exp: u64 = parts[2]
            .parse()
            .map_err(|_| "invalid exp timestamp".to_string())?;

        let role =
            Role::from_str_loose(parts[3]).ok_or_else(|| format!("unknown role: {}", parts[3]))?;

        let iss = parts.get(4).map(|s| s.to_string());

        // Validate expiry
        if now_secs > exp + self.clock_skew_secs {
            return Err(format!(
                "token expired at {}, current time is {}",
                exp, now_secs
            ));
        }

        // Validate iat is not in the future (with skew)
        if iat > now_secs + self.clock_skew_secs {
            return Err(format!(
                "token issued in the future: iat={}, now={}",
                iat, now_secs
            ));
        }

        // Validate issuer if configured
        if !self.accepted_issuers.is_empty() {
            match &iss {
                Some(token_iss) => {
                    if !self.accepted_issuers.contains(token_iss) {
                        return Err(format!("issuer '{}' is not accepted", token_iss));
                    }
                }
                None => {
                    return Err("token has no issuer but issuers are required".into());
                }
            }
        }

        Ok(TokenClaims {
            sub,
            iat,
            exp,
            role,
            iss,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// API key store
// ────────────────────────────────────────────────────────────────────────────

/// A registered API key.
#[derive(Debug, Clone)]
pub struct ApiKeyEntry {
    /// The API key string.
    pub key: String,
    /// Owner / user ID associated with the key.
    pub user_id: String,
    /// Role for requests authenticated with this key.
    pub role: Role,
    /// Optional description.
    pub description: Option<String>,
    /// Whether this key is currently active.
    pub active: bool,
    /// Maximum requests per minute; 0 = unlimited.
    pub rate_limit_rpm: u64,
}

impl ApiKeyEntry {
    pub fn new(key: impl Into<String>, user_id: impl Into<String>, role: Role) -> Self {
        Self {
            key: key.into(),
            user_id: user_id.into(),
            role,
            description: None,
            active: true,
            rate_limit_rpm: 0,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_rate_limit(mut self, rpm: u64) -> Self {
        self.rate_limit_rpm = rpm;
        self
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// In-memory API key store.
pub struct ApiKeyStore {
    /// Keys indexed by the key string.
    keys: HashMap<String, ApiKeyEntry>,
}

impl Default for ApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiKeyStore {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Register a new API key.
    pub fn register(&mut self, entry: ApiKeyEntry) {
        self.keys.insert(entry.key.clone(), entry);
    }

    /// Revoke (deactivate) an API key. Returns `true` if found.
    pub fn revoke(&mut self, key: &str) -> bool {
        if let Some(entry) = self.keys.get_mut(key) {
            entry.deactivate();
            true
        } else {
            false
        }
    }

    /// Remove an API key entirely.
    pub fn remove(&mut self, key: &str) -> bool {
        self.keys.remove(key).is_some()
    }

    /// Validate an API key string. Returns the entry if valid and active.
    pub fn validate(&self, key: &str) -> Result<&ApiKeyEntry, String> {
        match self.keys.get(key) {
            Some(entry) if entry.active => Ok(entry),
            Some(_) => Err("API key has been revoked".into()),
            None => Err("unknown API key".into()),
        }
    }

    /// Count of registered keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Count of active keys.
    pub fn active_count(&self) -> usize {
        self.keys.values().filter(|e| e.active).count()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Session management
// ────────────────────────────────────────────────────────────────────────────

/// A user session token.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session token.
    pub token: String,
    /// User who owns this session.
    pub user_id: String,
    /// Role for this session.
    pub role: Role,
    /// Creation time (seconds since epoch).
    pub created_at: u64,
    /// Expiry time (seconds since epoch).
    pub expires_at: u64,
    /// Last activity time (seconds since epoch).
    pub last_active: u64,
    /// Whether the session has been explicitly revoked.
    pub revoked: bool,
}

impl Session {
    pub fn is_expired(&self, now_secs: u64) -> bool {
        now_secs >= self.expires_at
    }

    pub fn is_valid(&self, now_secs: u64) -> bool {
        !self.revoked && !self.is_expired(now_secs)
    }

    pub fn remaining_secs(&self, now_secs: u64) -> u64 {
        self.expires_at.saturating_sub(now_secs)
    }
}

/// In-memory session store.
pub struct SessionStore {
    sessions: HashMap<String, Session>,
    /// Default session duration in seconds.
    default_ttl_secs: u64,
    /// Counter for generating unique token identifiers.
    next_id: u64,
}

impl SessionStore {
    pub fn new(default_ttl_secs: u64) -> Self {
        Self {
            sessions: HashMap::new(),
            default_ttl_secs,
            next_id: 1,
        }
    }

    /// Create a new session for a user. Returns the session token.
    pub fn create(&mut self, user_id: &str, role: Role, now_secs: u64) -> String {
        let token = format!("sess-{}-{}", self.next_id, now_secs);
        self.next_id += 1;

        let session = Session {
            token: token.clone(),
            user_id: user_id.to_string(),
            role,
            created_at: now_secs,
            expires_at: now_secs + self.default_ttl_secs,
            last_active: now_secs,
            revoked: false,
        };

        self.sessions.insert(token.clone(), session);
        token
    }

    /// Create a session with a custom TTL.
    pub fn create_with_ttl(
        &mut self,
        user_id: &str,
        role: Role,
        now_secs: u64,
        ttl_secs: u64,
    ) -> String {
        let token = format!("sess-{}-{}", self.next_id, now_secs);
        self.next_id += 1;

        let session = Session {
            token: token.clone(),
            user_id: user_id.to_string(),
            role,
            created_at: now_secs,
            expires_at: now_secs + ttl_secs,
            last_active: now_secs,
            revoked: false,
        };

        self.sessions.insert(token.clone(), session);
        token
    }

    /// Validate a session token. Updates `last_active` on success.
    pub fn validate(&mut self, token: &str, now_secs: u64) -> Result<&Session, String> {
        // First check existence
        if !self.sessions.contains_key(token) {
            return Err("unknown session token".into());
        }

        // Update last_active
        if let Some(session) = self.sessions.get_mut(token) {
            if session.revoked {
                return Err("session has been revoked".into());
            }
            if session.is_expired(now_secs) {
                return Err("session has expired".into());
            }
            session.last_active = now_secs;
        }

        // Now return the immutable reference
        self.sessions
            .get(token)
            .ok_or_else(|| "session not found".to_string())
    }

    /// Revoke a session.
    pub fn revoke(&mut self, token: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(token) {
            session.revoked = true;
            true
        } else {
            false
        }
    }

    /// Remove all expired / revoked sessions.
    pub fn cleanup(&mut self, now_secs: u64) -> usize {
        let before = self.sessions.len();
        self.sessions
            .retain(|_, s| !s.revoked && !s.is_expired(now_secs));
        before - self.sessions.len()
    }

    /// Number of active sessions.
    pub fn active_count(&self, now_secs: u64) -> usize {
        self.sessions
            .values()
            .filter(|s| s.is_valid(now_secs))
            .count()
    }

    /// Total sessions (including expired/revoked).
    pub fn total_count(&self) -> usize {
        self.sessions.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-user rate limiting
// ────────────────────────────────────────────────────────────────────────────

/// Per-user token-bucket rate state.
#[derive(Debug, Clone)]
struct UserRateBucket {
    tokens: f64,
    last_refill_ms: u64,
    total_allowed: u64,
    total_denied: u64,
}

/// Per-authenticated-user rate limiter.
pub struct UserRateLimiter {
    /// Requests per second per user.
    requests_per_second: f64,
    /// Maximum burst (initial tokens).
    burst_size: usize,
    /// Per-user buckets.
    buckets: HashMap<String, UserRateBucket>,
}

impl UserRateLimiter {
    pub fn new(requests_per_second: f64, burst_size: usize) -> Self {
        Self {
            requests_per_second,
            burst_size,
            buckets: HashMap::new(),
        }
    }

    /// Check whether a request from `user_id` is allowed at `now_ms`.
    pub fn check(&mut self, user_id: &str, now_ms: u64) -> bool {
        let burst = self.burst_size as f64;
        let rps = self.requests_per_second;

        let bucket = self
            .buckets
            .entry(user_id.to_string())
            .or_insert(UserRateBucket {
                tokens: burst,
                last_refill_ms: now_ms,
                total_allowed: 0,
                total_denied: 0,
            });

        // Refill tokens
        let elapsed_ms = now_ms.saturating_sub(bucket.last_refill_ms);
        let refill = (elapsed_ms as f64 / 1000.0) * rps;
        bucket.tokens = (bucket.tokens + refill).min(burst);
        bucket.last_refill_ms = now_ms;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            bucket.total_allowed += 1;
            true
        } else {
            bucket.total_denied += 1;
            false
        }
    }

    /// Get total allowed count for a user.
    pub fn allowed_count(&self, user_id: &str) -> u64 {
        self.buckets.get(user_id).map_or(0, |b| b.total_allowed)
    }

    /// Get total denied count for a user.
    pub fn denied_count(&self, user_id: &str) -> u64 {
        self.buckets.get(user_id).map_or(0, |b| b.total_denied)
    }

    /// Number of tracked users.
    pub fn user_count(&self) -> usize {
        self.buckets.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Anonymous access policy
// ────────────────────────────────────────────────────────────────────────────

/// Policy governing anonymous (unauthenticated) access.
#[derive(Debug, Clone)]
pub struct AnonymousPolicy {
    /// Whether anonymous read is allowed.
    pub allow_read: bool,
    /// Whether anonymous write is allowed.
    pub allow_write: bool,
    /// Maximum requests per minute for anonymous clients (0 = unlimited).
    pub rate_limit_rpm: u64,
    /// Datasets accessible anonymously (empty = all readable datasets).
    pub allowed_datasets: Vec<String>,
}

impl Default for AnonymousPolicy {
    fn default() -> Self {
        Self {
            allow_read: true,
            allow_write: false,
            rate_limit_rpm: 60,
            allowed_datasets: Vec::new(),
        }
    }
}

impl AnonymousPolicy {
    /// Returns `true` if the operation is allowed under this policy.
    pub fn is_allowed(&self, is_write: bool) -> bool {
        if is_write {
            self.allow_write
        } else {
            self.allow_read
        }
    }

    /// Check if a specific dataset is accessible.
    pub fn dataset_allowed(&self, dataset: &str) -> bool {
        self.allowed_datasets.is_empty() || self.allowed_datasets.iter().any(|d| d == dataset)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Auth middleware
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the authentication middleware.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Name of the HTTP header for API keys.
    pub api_key_header: String,
    /// Query parameter name for API keys.
    pub api_key_param: String,
    /// Whether bearer token auth is enabled.
    pub bearer_enabled: bool,
    /// Whether API key auth is enabled.
    pub api_key_enabled: bool,
    /// Whether session auth is enabled.
    pub session_enabled: bool,
    /// Anonymous access policy.
    pub anonymous_policy: AnonymousPolicy,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_key_header: "X-API-Key".to_string(),
            api_key_param: "api_key".to_string(),
            bearer_enabled: true,
            api_key_enabled: true,
            session_enabled: true,
            anonymous_policy: AnonymousPolicy::default(),
        }
    }
}

/// A simulated HTTP request with headers and query parameters.
#[derive(Debug, Clone)]
pub struct AuthRequest {
    /// Header name-value pairs (lowercased names).
    pub headers: HashMap<String, String>,
    /// Query parameters.
    pub query_params: HashMap<String, String>,
}

impl AuthRequest {
    pub fn new() -> Self {
        Self {
            headers: HashMap::new(),
            query_params: HashMap::new(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .insert(name.into().to_ascii_lowercase(), value.into());
        self
    }

    pub fn with_query_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(name.into(), value.into());
        self
    }

    /// Get a header by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }

    /// Get a query parameter.
    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query_params.get(name).map(|s| s.as_str())
    }
}

impl Default for AuthRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Central authentication middleware.
pub struct AuthMiddleware {
    config: AuthConfig,
    bearer_validator: BearerValidator,
    api_key_store: ApiKeyStore,
    session_store: SessionStore,
    rate_limiter: UserRateLimiter,
}

impl AuthMiddleware {
    /// Create a new middleware with the given configuration.
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            bearer_validator: BearerValidator::new(),
            api_key_store: ApiKeyStore::new(),
            session_store: SessionStore::new(3600),
            rate_limiter: UserRateLimiter::new(10.0, 20),
        }
    }

    /// Access the API key store for registration.
    pub fn api_key_store_mut(&mut self) -> &mut ApiKeyStore {
        &mut self.api_key_store
    }

    /// Access the session store.
    pub fn session_store_mut(&mut self) -> &mut SessionStore {
        &mut self.session_store
    }

    /// Access the bearer validator for configuration.
    pub fn bearer_validator_mut(&mut self) -> &mut BearerValidator {
        &mut self.bearer_validator
    }

    /// Access the rate limiter.
    pub fn rate_limiter_mut(&mut self) -> &mut UserRateLimiter {
        &mut self.rate_limiter
    }

    /// Authenticate an incoming request.
    pub fn authenticate(&mut self, request: &AuthRequest, now_secs: u64) -> AuthResult {
        // 1. Try bearer token
        if self.config.bearer_enabled {
            if let Some(auth_header) = request.header("authorization") {
                if let Some(token) = auth_header.strip_prefix("Bearer ") {
                    return match self.bearer_validator.validate(token.trim(), now_secs) {
                        Ok(claims) => {
                            let identity = AuthIdentity::new(&claims.sub, claims.role, "bearer");
                            AuthResult::Authenticated(identity)
                        }
                        Err(msg) => AuthResult::Denied(format!("bearer auth failed: {}", msg)),
                    };
                }
            }
        }

        // 2. Try API key from header
        if self.config.api_key_enabled {
            let header_name = self.config.api_key_header.to_ascii_lowercase();
            if let Some(key) = request.header(&header_name) {
                return match self.api_key_store.validate(key) {
                    Ok(entry) => {
                        let identity =
                            AuthIdentity::new(&entry.user_id, entry.role, "api_key_header");
                        AuthResult::Authenticated(identity)
                    }
                    Err(msg) => AuthResult::Denied(format!("API key auth failed: {}", msg)),
                };
            }

            // Try API key from query param
            let param_name = self.config.api_key_param.clone();
            if let Some(key) = request.query_param(&param_name) {
                return match self.api_key_store.validate(key) {
                    Ok(entry) => {
                        let identity =
                            AuthIdentity::new(&entry.user_id, entry.role, "api_key_param");
                        AuthResult::Authenticated(identity)
                    }
                    Err(msg) => AuthResult::Denied(format!("API key auth failed: {}", msg)),
                };
            }
        }

        // 3. Try session token (from cookie header)
        if self.config.session_enabled {
            if let Some(cookie) = request.header("cookie") {
                if let Some(token) = extract_session_from_cookie(cookie) {
                    return match self.session_store.validate(&token, now_secs) {
                        Ok(session) => {
                            let identity =
                                AuthIdentity::new(&session.user_id, session.role, "session");
                            AuthResult::Authenticated(identity)
                        }
                        Err(msg) => AuthResult::Denied(format!("session auth failed: {}", msg)),
                    };
                }
            }
        }

        AuthResult::NoCredentials
    }

    /// Authorize an authenticated (or anonymous) request.
    pub fn authorize(&mut self, auth: &AuthResult, is_write: bool, now_ms: u64) -> bool {
        match auth {
            AuthResult::Authenticated(identity) => {
                // Check role
                let role_ok = if is_write {
                    identity.role.can_write()
                } else {
                    identity.role.can_read()
                };
                if !role_ok {
                    return false;
                }
                // Check per-user rate limit
                self.rate_limiter.check(&identity.user_id, now_ms)
            }
            AuthResult::NoCredentials => self.config.anonymous_policy.is_allowed(is_write),
            AuthResult::Denied(_) => false,
        }
    }
}

/// Extract a session token from a `Cookie` header value.
fn extract_session_from_cookie(cookie: &str) -> Option<String> {
    for part in cookie.split(';') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("session=") {
            let token = value.trim().to_string();
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Role ────────────────────────────────────────────────────────────────

    #[test]
    fn test_role_permissions() {
        assert!(Role::Admin.can_read());
        assert!(Role::Admin.can_write());
        assert!(Role::Admin.is_admin());

        assert!(Role::Reader.can_read());
        assert!(!Role::Reader.can_write());
        assert!(!Role::Reader.is_admin());

        assert!(Role::Writer.can_read());
        assert!(Role::Writer.can_write());
        assert!(!Role::Writer.is_admin());

        assert!(!Role::Anonymous.can_read());
        assert!(!Role::Anonymous.can_write());
        assert!(!Role::Anonymous.is_admin());
    }

    #[test]
    fn test_role_from_str_loose() {
        assert_eq!(Role::from_str_loose("admin"), Some(Role::Admin));
        assert_eq!(Role::from_str_loose("ADMIN"), Some(Role::Admin));
        assert_eq!(Role::from_str_loose("reader"), Some(Role::Reader));
        assert_eq!(Role::from_str_loose("read"), Some(Role::Reader));
        assert_eq!(Role::from_str_loose("writer"), Some(Role::Writer));
        assert_eq!(Role::from_str_loose("write"), Some(Role::Writer));
        assert_eq!(Role::from_str_loose("anonymous"), Some(Role::Anonymous));
        assert_eq!(Role::from_str_loose("anon"), Some(Role::Anonymous));
        assert_eq!(Role::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_role_labels() {
        assert_eq!(Role::Admin.label(), "admin");
        assert_eq!(Role::Reader.label(), "reader");
        assert_eq!(Role::Writer.label(), "writer");
        assert_eq!(Role::Anonymous.label(), "anonymous");
    }

    // ── Bearer validation ───────────────────────────────────────────────────

    #[test]
    fn test_bearer_valid_token() {
        let validator = BearerValidator::new();
        let token = "alice.1000.2000.admin";
        let claims = validator.validate(token, 1500).expect("should validate");
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.iat, 1000);
        assert_eq!(claims.exp, 2000);
        assert_eq!(claims.role, Role::Admin);
        assert!(claims.iss.is_none());
    }

    #[test]
    fn test_bearer_expired_token() {
        let validator = BearerValidator::new();
        let token = "alice.1000.1500.reader";
        let result = validator.validate(token, 2000);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(err.contains("expired"));
    }

    #[test]
    fn test_bearer_future_iat() {
        let validator = BearerValidator::new();
        let token = "alice.5000.6000.reader";
        let result = validator.validate(token, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_bearer_with_issuer() {
        let validator = BearerValidator::new().with_issuer("oxirs");
        let token = "alice.1000.2000.admin.oxirs";
        let claims = validator.validate(token, 1500).expect("should validate");
        assert_eq!(claims.iss, Some("oxirs".to_string()));
    }

    #[test]
    fn test_bearer_wrong_issuer() {
        let validator = BearerValidator::new().with_issuer("oxirs");
        let token = "alice.1000.2000.admin.other";
        let result = validator.validate(token, 1500);
        assert!(result.is_err());
    }

    #[test]
    fn test_bearer_missing_issuer_when_required() {
        let validator = BearerValidator::new().with_issuer("oxirs");
        let token = "alice.1000.2000.admin";
        let result = validator.validate(token, 1500);
        assert!(result.is_err());
    }

    #[test]
    fn test_bearer_invalid_format() {
        let validator = BearerValidator::new();
        let result = validator.validate("bad-token", 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_bearer_empty_subject() {
        let validator = BearerValidator::new();
        let result = validator.validate(".1000.2000.admin", 1500);
        assert!(result.is_err());
    }

    #[test]
    fn test_token_claims_ttl() {
        let claims = TokenClaims {
            sub: "user".into(),
            iat: 1000,
            exp: 2000,
            role: Role::Reader,
            iss: None,
        };
        assert_eq!(claims.ttl(1500), 500);
        assert_eq!(claims.ttl(2500), 0);
        assert!(!claims.is_expired(1500));
        assert!(claims.is_expired(2000));
    }

    // ── API key store ───────────────────────────────────────────────────────

    #[test]
    fn test_api_key_register_and_validate() {
        let mut store = ApiKeyStore::new();
        store.register(ApiKeyEntry::new("key-123", "alice", Role::Writer));

        let entry = store.validate("key-123").expect("should find");
        assert_eq!(entry.user_id, "alice");
        assert_eq!(entry.role, Role::Writer);
        assert!(entry.active);
    }

    #[test]
    fn test_api_key_unknown() {
        let store = ApiKeyStore::new();
        assert!(store.validate("nonexistent").is_err());
    }

    #[test]
    fn test_api_key_revoke() {
        let mut store = ApiKeyStore::new();
        store.register(ApiKeyEntry::new("key-abc", "bob", Role::Reader));

        assert!(store.revoke("key-abc"));
        let result = store.validate("key-abc");
        assert!(result.is_err());
        assert!(result.expect_err("should fail").contains("revoked"));
    }

    #[test]
    fn test_api_key_remove() {
        let mut store = ApiKeyStore::new();
        store.register(ApiKeyEntry::new("key-del", "carol", Role::Admin));

        assert!(store.remove("key-del"));
        assert!(!store.remove("key-del"));
        assert!(store.validate("key-del").is_err());
    }

    #[test]
    fn test_api_key_store_counts() {
        let mut store = ApiKeyStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.register(ApiKeyEntry::new("k1", "u1", Role::Reader));
        store.register(ApiKeyEntry::new("k2", "u2", Role::Writer));
        assert_eq!(store.len(), 2);
        assert_eq!(store.active_count(), 2);

        store.revoke("k1");
        assert_eq!(store.active_count(), 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_api_key_entry_builders() {
        let entry = ApiKeyEntry::new("k", "u", Role::Admin)
            .with_description("test key")
            .with_rate_limit(100);
        assert_eq!(entry.description, Some("test key".to_string()));
        assert_eq!(entry.rate_limit_rpm, 100);
    }

    // ── Session management ──────────────────────────────────────────────────

    #[test]
    fn test_session_create_and_validate() {
        let mut store = SessionStore::new(3600);
        let token = store.create("alice", Role::Writer, 1000);

        let session = store.validate(&token, 1500).expect("should validate");
        assert_eq!(session.user_id, "alice");
        assert_eq!(session.role, Role::Writer);
        assert!(session.is_valid(1500));
    }

    #[test]
    fn test_session_expired() {
        let mut store = SessionStore::new(100); // short TTL
        let token = store.create("bob", Role::Reader, 1000);

        let result = store.validate(&token, 2000);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_revoke() {
        let mut store = SessionStore::new(3600);
        let token = store.create("carol", Role::Admin, 1000);

        assert!(store.revoke(&token));
        let result = store.validate(&token, 1500);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_revoke_nonexistent() {
        let mut store = SessionStore::new(3600);
        assert!(!store.revoke("no-such-token"));
    }

    #[test]
    fn test_session_cleanup() {
        let mut store = SessionStore::new(100);
        store.create("u1", Role::Reader, 1000);
        store.create("u2", Role::Writer, 1000);
        store.create("u3", Role::Admin, 2000);

        assert_eq!(store.total_count(), 3);
        let removed = store.cleanup(1200);
        assert_eq!(removed, 2);
        assert_eq!(store.total_count(), 1);
    }

    #[test]
    fn test_session_active_count() {
        let mut store = SessionStore::new(3600);
        store.create("u1", Role::Reader, 1000);
        let t2 = store.create("u2", Role::Writer, 1000);
        store.revoke(&t2);

        assert_eq!(store.active_count(1500), 1);
    }

    #[test]
    fn test_session_custom_ttl() {
        let mut store = SessionStore::new(3600);
        let token = store.create_with_ttl("alice", Role::Admin, 1000, 60);

        let session = store.validate(&token, 1050).expect("valid");
        assert_eq!(session.remaining_secs(1050), 10);

        let result = store.validate(&token, 1070);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_unknown_token() {
        let mut store = SessionStore::new(3600);
        let result = store.validate("nonexistent", 1000);
        assert!(result.is_err());
    }

    // ── User rate limiter ───────────────────────────────────────────────────

    #[test]
    fn test_user_rate_limiter_basic() {
        let mut limiter = UserRateLimiter::new(2.0, 2);
        // Burst allows 2 immediate requests
        assert!(limiter.check("alice", 0));
        assert!(limiter.check("alice", 0));
        // Third should be denied
        assert!(!limiter.check("alice", 0));
    }

    #[test]
    fn test_user_rate_limiter_refill() {
        let mut limiter = UserRateLimiter::new(1.0, 1);
        assert!(limiter.check("alice", 0));
        assert!(!limiter.check("alice", 500)); // not enough time
        assert!(limiter.check("alice", 1500)); // 1 second refill
    }

    #[test]
    fn test_user_rate_limiter_separate_users() {
        let mut limiter = UserRateLimiter::new(1.0, 1);
        assert!(limiter.check("alice", 0));
        assert!(limiter.check("bob", 0)); // different user, different bucket
    }

    #[test]
    fn test_user_rate_limiter_counts() {
        let mut limiter = UserRateLimiter::new(1.0, 1);
        limiter.check("alice", 0);
        limiter.check("alice", 0);
        assert_eq!(limiter.allowed_count("alice"), 1);
        assert_eq!(limiter.denied_count("alice"), 1);
        assert_eq!(limiter.user_count(), 1);
    }

    // ── Anonymous policy ────────────────────────────────────────────────────

    #[test]
    fn test_anonymous_default_policy() {
        let policy = AnonymousPolicy::default();
        assert!(policy.is_allowed(false)); // read OK
        assert!(!policy.is_allowed(true)); // write denied
    }

    #[test]
    fn test_anonymous_dataset_check() {
        let policy = AnonymousPolicy {
            allowed_datasets: vec!["public".to_string()],
            ..AnonymousPolicy::default()
        };
        assert!(policy.dataset_allowed("public"));
        assert!(!policy.dataset_allowed("secret"));
    }

    #[test]
    fn test_anonymous_all_datasets_when_empty() {
        let policy = AnonymousPolicy::default();
        assert!(policy.dataset_allowed("anything"));
    }

    // ── AuthMiddleware integration ──────────────────────────────────────────

    #[test]
    fn test_middleware_bearer_auth() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let req = AuthRequest::new().with_header("Authorization", "Bearer alice.1000.2000.admin");

        let result = mw.authenticate(&req, 1500);
        match result {
            AuthResult::Authenticated(id) => {
                assert_eq!(id.user_id, "alice");
                assert_eq!(id.role, Role::Admin);
                assert_eq!(id.auth_method, "bearer");
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[test]
    fn test_middleware_api_key_header() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        mw.api_key_store_mut()
            .register(ApiKeyEntry::new("my-key", "bob", Role::Writer));

        let req = AuthRequest::new().with_header("X-API-Key", "my-key");
        let result = mw.authenticate(&req, 1500);
        match result {
            AuthResult::Authenticated(id) => {
                assert_eq!(id.user_id, "bob");
                assert_eq!(id.auth_method, "api_key_header");
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[test]
    fn test_middleware_api_key_query_param() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        mw.api_key_store_mut()
            .register(ApiKeyEntry::new("qkey", "carol", Role::Reader));

        let req = AuthRequest::new().with_query_param("api_key", "qkey");
        let result = mw.authenticate(&req, 1500);
        match result {
            AuthResult::Authenticated(id) => {
                assert_eq!(id.user_id, "carol");
                assert_eq!(id.auth_method, "api_key_param");
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[test]
    fn test_middleware_session_cookie() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let token = mw.session_store_mut().create("dave", Role::Admin, 1000);

        let req = AuthRequest::new().with_header("Cookie", format!("session={}", token));
        let result = mw.authenticate(&req, 1500);
        match result {
            AuthResult::Authenticated(id) => {
                assert_eq!(id.user_id, "dave");
                assert_eq!(id.auth_method, "session");
            }
            other => panic!("expected Authenticated, got {:?}", other),
        }
    }

    #[test]
    fn test_middleware_no_credentials() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let req = AuthRequest::new();
        let result = mw.authenticate(&req, 1500);
        assert_eq!(result, AuthResult::NoCredentials);
    }

    #[test]
    fn test_middleware_expired_bearer() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let req = AuthRequest::new().with_header("Authorization", "Bearer alice.1000.1500.admin");
        let result = mw.authenticate(&req, 2000);
        match result {
            AuthResult::Denied(msg) => assert!(msg.contains("expired")),
            other => panic!("expected Denied, got {:?}", other),
        }
    }

    #[test]
    fn test_middleware_revoked_api_key() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        mw.api_key_store_mut()
            .register(ApiKeyEntry::new("rk", "alice", Role::Admin));
        mw.api_key_store_mut().revoke("rk");

        let req = AuthRequest::new().with_header("X-API-Key", "rk");
        let result = mw.authenticate(&req, 1500);
        match result {
            AuthResult::Denied(msg) => assert!(msg.contains("revoked")),
            other => panic!("expected Denied, got {:?}", other),
        }
    }

    // ── Authorization ───────────────────────────────────────────────────────

    #[test]
    fn test_authorize_admin_write() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let auth = AuthResult::Authenticated(AuthIdentity::new("alice", Role::Admin, "bearer"));
        assert!(mw.authorize(&auth, true, 0));
    }

    #[test]
    fn test_authorize_reader_no_write() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let auth = AuthResult::Authenticated(AuthIdentity::new("bob", Role::Reader, "bearer"));
        assert!(!mw.authorize(&auth, true, 0));
        assert!(mw.authorize(&auth, false, 1000));
    }

    #[test]
    fn test_authorize_anonymous_read_default() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let auth = AuthResult::NoCredentials;
        assert!(mw.authorize(&auth, false, 0));
        assert!(!mw.authorize(&auth, true, 0));
    }

    #[test]
    fn test_authorize_denied_always_fails() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        let auth = AuthResult::Denied("bad".into());
        assert!(!mw.authorize(&auth, false, 0));
        assert!(!mw.authorize(&auth, true, 0));
    }

    #[test]
    fn test_authorize_rate_limit_enforced() {
        let mut mw = AuthMiddleware::new(AuthConfig::default());
        *mw.rate_limiter_mut() = UserRateLimiter::new(1.0, 1);
        let auth = AuthResult::Authenticated(AuthIdentity::new("alice", Role::Admin, "bearer"));

        assert!(mw.authorize(&auth, false, 0));
        assert!(!mw.authorize(&auth, false, 0)); // rate limited
    }

    // ── Cookie extraction ───────────────────────────────────────────────────

    #[test]
    fn test_extract_session_from_cookie() {
        assert_eq!(
            extract_session_from_cookie("session=abc123"),
            Some("abc123".to_string())
        );
        assert_eq!(
            extract_session_from_cookie("other=x; session=tok; more=y"),
            Some("tok".to_string())
        );
        assert_eq!(extract_session_from_cookie("other=x"), None);
        assert_eq!(extract_session_from_cookie("session="), None);
    }

    // ── AuthIdentity builder ────────────────────────────────────────────────

    #[test]
    fn test_auth_identity_display_name() {
        let id =
            AuthIdentity::new("alice", Role::Admin, "bearer").with_display_name("Alice Wonderland");
        assert_eq!(id.display_name, Some("Alice Wonderland".to_string()));
    }

    // ── AuthRequest builder ─────────────────────────────────────────────────

    #[test]
    fn test_auth_request_case_insensitive_header() {
        let req = AuthRequest::new().with_header("Content-Type", "application/json");
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_auth_request_default() {
        let req = AuthRequest::default();
        assert!(req.headers.is_empty());
        assert!(req.query_params.is_empty());
    }

    // ── Disabled auth methods ───────────────────────────────────────────────

    #[test]
    fn test_bearer_disabled() {
        let config = AuthConfig {
            bearer_enabled: false,
            ..AuthConfig::default()
        };
        let mut mw = AuthMiddleware::new(config);
        let req = AuthRequest::new().with_header("Authorization", "Bearer alice.1000.2000.admin");
        let result = mw.authenticate(&req, 1500);
        assert_eq!(result, AuthResult::NoCredentials);
    }

    #[test]
    fn test_api_key_disabled() {
        let config = AuthConfig {
            api_key_enabled: false,
            ..AuthConfig::default()
        };
        let mut mw = AuthMiddleware::new(config);
        mw.api_key_store_mut()
            .register(ApiKeyEntry::new("k", "u", Role::Admin));

        let req = AuthRequest::new().with_header("X-API-Key", "k");
        let result = mw.authenticate(&req, 1500);
        assert_eq!(result, AuthResult::NoCredentials);
    }
}
