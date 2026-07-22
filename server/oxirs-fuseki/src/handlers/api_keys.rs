//! API Key Management for OxiRS Fuseki
//!
//! This module provides comprehensive API key management with:
//! - Scoped API key authentication
//! - Role-based access control for API keys
//! - API key lifecycle management (create, revoke, expire)
//! - Usage analytics and rate limiting
//! - Secure key generation and validation

use crate::{
    auth::{AuthService, Permission, User},
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Path, Query, State},
    http::{
        header::{AUTHORIZATION, COOKIE},
        HeaderMap,
    },
    response::Json,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// API key scopes for fine-grained access control
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ApiKeyScope {
    /// Read access to SPARQL queries
    SparqlRead,
    /// Write access to SPARQL updates
    SparqlWrite,
    /// Access to graph store operations
    GraphStore,
    /// Access to dataset management
    DatasetManagement,
    /// Read access to system metrics
    MetricsRead,
    /// Administrative access
    Admin,
    /// Custom dataset-specific scopes
    DatasetRead(String),
    DatasetWrite(String),
    DatasetAdmin(String),
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    pub key_hash: String,
    pub scopes: Vec<ApiKeyScope>,
    pub owner: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
    pub rate_limit: Option<RateLimit>,
    pub allowed_ips: Vec<String>,
    pub description: Option<String>,
}

/// Rate limiting configuration for API keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
    pub burst_limit: u32,
}

/// API key creation request
#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub scopes: Vec<ApiKeyScope>,
    pub expires_in_days: Option<u32>,
    pub rate_limit: Option<RateLimit>,
    pub allowed_ips: Option<Vec<String>>,
    pub description: Option<String>,
}

/// API key creation response
#[derive(Debug, Serialize)]
pub struct CreateApiKeyResponse {
    pub success: bool,
    pub api_key: Option<ApiKeyInfo>,
    pub raw_key: Option<String>, // Only returned once at creation
    pub message: String,
}

/// API key information for responses (without sensitive data)
#[derive(Debug, Serialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub name: String,
    pub scopes: Vec<ApiKeyScope>,
    pub owner: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
    pub rate_limit: Option<RateLimit>,
    pub allowed_ips: Vec<String>,
    pub description: Option<String>,
}

/// API key usage statistics
#[derive(Debug, Serialize)]
pub struct ApiKeyUsageStats {
    pub total_requests: u64,
    pub requests_last_24h: u64,
    pub requests_last_7d: u64,
    pub requests_last_30d: u64,
    pub average_requests_per_day: f64,
    pub most_used_endpoints: Vec<EndpointUsage>,
    pub error_rate: f64,
}

/// Endpoint usage information
#[derive(Debug, Serialize)]
pub struct EndpointUsage {
    pub endpoint: String,
    pub count: u64,
    pub last_used: DateTime<Utc>,
}

/// API key update request
#[derive(Debug, Deserialize)]
pub struct UpdateApiKeyRequest {
    pub name: Option<String>,
    pub scopes: Option<Vec<ApiKeyScope>>,
    pub is_active: Option<bool>,
    pub rate_limit: Option<RateLimit>,
    pub allowed_ips: Option<Vec<String>>,
    pub description: Option<String>,
}

/// Query parameters for listing API keys
#[derive(Debug, Deserialize)]
pub struct ListApiKeysQuery {
    pub owner: Option<String>,
    pub active_only: Option<bool>,
    pub scope: Option<ApiKeyScope>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Create a new API key
#[instrument(skip(state, request))]
pub async fn create_api_key(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<CreateApiKeyRequest>,
) -> Result<Json<CreateApiKeyResponse>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    // Extract authenticated user
    let user = extract_authenticated_user(&headers, auth_service).await?;

    // Check if user has permission to create API keys
    if !has_api_key_management_permission(&user) {
        return Err(FusekiError::forbidden(
            "Insufficient permissions to manage API keys",
        ));
    }

    // Validate scopes
    validate_api_key_scopes(&request.scopes, &user)?;

    // Enforce the configured per-user key cap (security.api_keys.max_keys_per_user).
    let api_key_service = get_api_key_service(&state).await?;
    let existing = api_key_service.active_key_count_for(&user.username).await;
    if existing >= api_key_service.max_keys_per_user() as usize {
        return Err(FusekiError::forbidden(format!(
            "API key limit reached: user '{}' already holds {} active key(s) (max {})",
            user.username,
            existing,
            api_key_service.max_keys_per_user()
        )));
    }

    // Generate secure API key
    let raw_key = generate_api_key();
    let key_hash = hash_api_key(&raw_key)?;

    // Calculate expiration. When the request omits `expires_in_days`, fall back to
    // the configured default (security.api_keys.default_expiration_days) rather
    // than minting a non-expiring key.
    let effective_days = request
        .expires_in_days
        .unwrap_or_else(|| api_key_service.default_expiration_days());
    let expires_at = Some(Utc::now() + Duration::days(effective_days as i64));

    // Create API key record
    let api_key = ApiKey {
        id: Uuid::new_v4().to_string(),
        name: request.name.clone(),
        key_hash,
        scopes: request.scopes.clone(),
        owner: user.username.clone(),
        created_at: Utc::now(),
        expires_at,
        last_used: None,
        is_active: true,
        usage_count: 0,
        // Fall back to the configured `security.api_keys.default_rate_limit`
        // when the request omits an explicit limit, rather than minting an
        // unlimited key by default.
        rate_limit: request
            .rate_limit
            .clone()
            .or_else(|| api_key_service.default_rate_limit()),
        allowed_ips: request.allowed_ips.unwrap_or_default(),
        description: request.description.clone(),
    };

    // Store API key (service already resolved above for the per-user cap check).
    api_key_service.store_api_key(&api_key).await?;

    info!(
        "API key '{}' created for user '{}' with scopes: {:?}",
        request.name, user.username, request.scopes
    );

    Ok(Json(CreateApiKeyResponse {
        success: true,
        api_key: Some(api_key.into()),
        raw_key: Some(raw_key),
        message: "API key created successfully. Store the key securely - it won't be shown again."
            .to_string(),
    }))
}

/// List API keys for the authenticated user or all users (if admin)
#[instrument(skip(state))]
pub async fn list_api_keys(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListApiKeysQuery>,
    headers: HeaderMap,
) -> Result<Json<Vec<ApiKeyInfo>>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    let user = extract_authenticated_user(&headers, auth_service).await?;
    let api_key_service = get_api_key_service(&state).await?;

    // Determine owner filter
    let owner_filter = if has_admin_permission(&user) {
        query.owner
    } else {
        Some(user.username.clone())
    };

    let api_keys = api_key_service
        .list_api_keys(
            owner_filter.as_deref(),
            query.active_only.unwrap_or(false),
            query.scope.as_ref(),
            query.limit.unwrap_or(100),
            query.offset.unwrap_or(0),
        )
        .await?;

    let api_key_infos: Vec<ApiKeyInfo> = api_keys.into_iter().map(|key| key.into()).collect();

    Ok(Json(api_key_infos))
}

/// Get details of a specific API key
#[instrument(skip(state))]
pub async fn get_api_key(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<ApiKeyInfo>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    let user = extract_authenticated_user(&headers, auth_service).await?;
    let api_key_service = get_api_key_service(&state).await?;

    let api_key = api_key_service
        .get_api_key(&key_id)
        .await?
        .ok_or_else(|| FusekiError::not_found("API key not found"))?;

    // Check ownership or admin permission
    if api_key.owner != user.username && !has_admin_permission(&user) {
        return Err(FusekiError::forbidden("Access denied"));
    }

    Ok(Json(api_key.into()))
}

/// Update an existing API key
#[instrument(skip(state, request))]
pub async fn update_api_key(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<String>,
    headers: HeaderMap,
    Json(request): Json<UpdateApiKeyRequest>,
) -> Result<Json<ApiKeyInfo>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    let user = extract_authenticated_user(&headers, auth_service).await?;
    let api_key_service = get_api_key_service(&state).await?;

    let mut api_key = api_key_service
        .get_api_key(&key_id)
        .await?
        .ok_or_else(|| FusekiError::not_found("API key not found"))?;

    // Check ownership or admin permission
    if api_key.owner != user.username && !has_admin_permission(&user) {
        return Err(FusekiError::forbidden("Access denied"));
    }

    // Update fields
    if let Some(name) = request.name {
        api_key.name = name;
    }
    if let Some(scopes) = request.scopes {
        validate_api_key_scopes(&scopes, &user)?;
        api_key.scopes = scopes;
    }
    if let Some(is_active) = request.is_active {
        api_key.is_active = is_active;
    }
    if let Some(rate_limit) = request.rate_limit {
        api_key.rate_limit = Some(rate_limit);
    }
    if let Some(allowed_ips) = request.allowed_ips {
        api_key.allowed_ips = allowed_ips;
    }
    if let Some(description) = request.description {
        api_key.description = Some(description);
    }

    // Update in storage
    api_key_service.update_api_key(&api_key).await?;

    info!(
        "API key '{}' updated by user '{}'",
        api_key.name, user.username
    );

    Ok(Json(api_key.into()))
}

/// Revoke/delete an API key
#[instrument(skip(state))]
pub async fn revoke_api_key(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    let user = extract_authenticated_user(&headers, auth_service).await?;
    let api_key_service = get_api_key_service(&state).await?;

    let api_key = api_key_service
        .get_api_key(&key_id)
        .await?
        .ok_or_else(|| FusekiError::not_found("API key not found"))?;

    // Check ownership or admin permission
    if api_key.owner != user.username && !has_admin_permission(&user) {
        return Err(FusekiError::forbidden("Access denied"));
    }

    // Revoke the key
    api_key_service.revoke_api_key(&key_id).await?;

    info!(
        "API key '{}' revoked by user '{}'",
        api_key.name, user.username
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "API key revoked successfully"
    })))
}

/// Get usage statistics for an API key
#[instrument(skip(state))]
pub async fn get_api_key_usage(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<ApiKeyUsageStats>, FusekiError> {
    let auth_service = state
        .auth_service
        .as_ref()
        .ok_or_else(|| FusekiError::service_unavailable("Authentication service not available"))?;

    let user = extract_authenticated_user(&headers, auth_service).await?;
    let api_key_service = get_api_key_service(&state).await?;

    let api_key = api_key_service
        .get_api_key(&key_id)
        .await?
        .ok_or_else(|| FusekiError::not_found("API key not found"))?;

    // Check ownership or admin permission
    if api_key.owner != user.username && !has_admin_permission(&user) {
        return Err(FusekiError::forbidden("Access denied"));
    }

    let usage_stats = api_key_service.get_usage_stats(&key_id).await?;

    Ok(Json(usage_stats))
}

/// Validate API key and return associated permissions
pub async fn validate_api_key_auth(
    api_key: &str,
    api_key_service: &ApiKeyService,
    client_ip: Option<&str>,
) -> FusekiResult<Option<User>> {
    let key_hash = hash_api_key(api_key)?;

    // Find API key by hash
    if let Some(mut stored_key) = api_key_service.get_api_key_by_hash(&key_hash).await? {
        // Check if key is active
        if !stored_key.is_active {
            warn!("Attempt to use inactive API key: {}", stored_key.id);
            return Ok(None);
        }

        // Check expiration
        if let Some(expires_at) = stored_key.expires_at {
            if Utc::now() > expires_at {
                warn!("Attempt to use expired API key: {}", stored_key.id);
                return Ok(None);
            }
        }

        // Check IP restrictions
        if !stored_key.allowed_ips.is_empty() {
            if let Some(ip) = client_ip {
                if !stored_key.allowed_ips.contains(&ip.to_string()) {
                    warn!(
                        "API key {} used from unauthorized IP: {}",
                        stored_key.id, ip
                    );
                    return Ok(None);
                }
            } else {
                warn!(
                    "API key {} requires IP validation but no IP provided",
                    stored_key.id
                );
                return Ok(None);
            }
        }

        // Check rate limiting
        if let Some(rate_limit) = &stored_key.rate_limit {
            if !api_key_service
                .check_rate_limit(&stored_key.id, rate_limit)
                .await?
            {
                warn!("Rate limit exceeded for API key: {}", stored_key.id);
                return Err(FusekiError::RateLimit);
            }
        }

        // Update usage
        stored_key.last_used = Some(Utc::now());
        stored_key.usage_count += 1;
        api_key_service.update_usage(&stored_key).await?;

        // Convert scopes to permissions
        let permissions = convert_scopes_to_permissions(&stored_key.scopes);

        // Create user object for API key
        let user = User {
            username: format!("apikey:{}", stored_key.name),
            roles: vec!["api_user".to_string()],
            email: None,
            full_name: Some(format!("API Key: {}", stored_key.name)),
            last_login: Some(Utc::now()),
            permissions,
        };

        debug!("API key authentication successful: {}", stored_key.id);
        Ok(Some(user))
    } else {
        debug!("Invalid API key provided");
        Ok(None)
    }
}

// Helper Functions

fn generate_api_key() -> String {
    use scirs2_core::random::SecureRandom;

    const PREFIX: &str = "oxirs_";
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const KEY_LENGTH: usize = 32;

    // Rejection sampling avoids modulo bias: CHARSET.len() == 62 does not
    // evenly divide 256, so byte values in the trailing partial bucket are
    // discarded and re-drawn rather than reduced via `% CHARSET.len()`.
    let limit = (256 / CHARSET.len()) * CHARSET.len();
    let mut secure = SecureRandom::new();
    let mut key = String::with_capacity(KEY_LENGTH);
    while key.len() < KEY_LENGTH {
        let batch = secure.random_bytes(KEY_LENGTH - key.len());
        for b in batch {
            if (b as usize) < limit {
                key.push(CHARSET[(b as usize) % CHARSET.len()] as char);
            }
        }
    }

    format!("{PREFIX}{key}")
}

fn hash_api_key(api_key: &str) -> FusekiResult<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

fn validate_api_key_scopes(scopes: &[ApiKeyScope], user: &User) -> FusekiResult<()> {
    // Ensure user has permission to grant these scopes
    for scope in scopes {
        match scope {
            ApiKeyScope::Admin if !user.permissions.contains(&Permission::SystemConfig) => {
                return Err(FusekiError::forbidden("Cannot grant admin scope"));
            }
            ApiKeyScope::DatasetManagement
                if !user.permissions.contains(&Permission::GlobalAdmin) =>
            {
                return Err(FusekiError::forbidden(
                    "Cannot grant dataset management scope",
                ));
            }
            ApiKeyScope::DatasetAdmin(_)
                if !user.permissions.contains(&Permission::GlobalAdmin) =>
            {
                return Err(FusekiError::forbidden("Cannot grant dataset admin scope"));
            }
            _ => {} // Other scopes are generally allowed
        }
    }
    Ok(())
}

fn convert_scopes_to_permissions(scopes: &[ApiKeyScope]) -> Vec<Permission> {
    let mut permissions = Vec::new();

    for scope in scopes {
        match scope {
            ApiKeyScope::SparqlRead => permissions.push(Permission::SparqlQuery),
            ApiKeyScope::SparqlWrite => {
                permissions.extend(vec![Permission::SparqlQuery, Permission::SparqlUpdate]);
            }
            ApiKeyScope::GraphStore => permissions.push(Permission::GraphStore),
            ApiKeyScope::DatasetManagement => {
                permissions.extend(vec![
                    Permission::GlobalRead,
                    Permission::GlobalWrite,
                    Permission::GlobalAdmin,
                ]);
            }
            ApiKeyScope::MetricsRead => permissions.push(Permission::SystemMetrics),
            ApiKeyScope::Admin => {
                permissions.extend(vec![
                    Permission::GlobalAdmin,
                    Permission::SystemConfig,
                    Permission::UserManagement,
                    Permission::SystemMetrics,
                ]);
            }
            ApiKeyScope::DatasetRead(dataset) => {
                permissions.push(Permission::DatasetRead(dataset.clone()));
            }
            ApiKeyScope::DatasetWrite(dataset) => {
                permissions.extend(vec![
                    Permission::DatasetRead(dataset.clone()),
                    Permission::DatasetWrite(dataset.clone()),
                ]);
            }
            ApiKeyScope::DatasetAdmin(dataset) => {
                permissions.extend(vec![
                    Permission::DatasetRead(dataset.clone()),
                    Permission::DatasetWrite(dataset.clone()),
                    Permission::DatasetAdmin(dataset.clone()),
                ]);
            }
        }
    }

    permissions.sort();
    permissions.dedup();
    permissions
}

fn has_api_key_management_permission(user: &User) -> bool {
    user.permissions.contains(&Permission::SystemConfig)
        || user.permissions.contains(&Permission::GlobalAdmin)
        || user.roles.contains(&"admin".to_string())
}

fn has_admin_permission(user: &User) -> bool {
    user.permissions.contains(&Permission::GlobalAdmin) || user.roles.contains(&"admin".to_string())
}

/// Extract the real authenticated principal from the request.
///
/// Tries, in order: a `Bearer` JWT in the `Authorization` header, then a
/// `session_id` cookie. Fails closed (returns an authentication error)
/// rather than defaulting to any implicit identity when neither is present
/// or valid — API key management must never be reachable anonymously.
async fn extract_authenticated_user(
    headers: &HeaderMap,
    auth_service: &AuthService,
) -> FusekiResult<User> {
    if let Some(auth_header) = headers.get(AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if let Ok(validation) = auth_service.validate_jwt_token(token) {
                    return Ok(validation.user);
                }
            }
        }
    }

    if let Some(cookie_header) = headers.get(COOKIE) {
        if let Ok(cookie_str) = cookie_header.to_str() {
            for cookie in cookie_str.split(';') {
                let cookie = cookie.trim();
                if let Some(session_id) = cookie.strip_prefix("session_id=") {
                    if let Some(user) = auth_service.validate_session(session_id).await? {
                        return Ok(user);
                    }
                }
            }
        }
    }

    Err(FusekiError::authentication(
        "Authentication required to manage API keys",
    ))
}

// Conversion implementations
impl From<ApiKey> for ApiKeyInfo {
    fn from(api_key: ApiKey) -> Self {
        Self {
            id: api_key.id,
            name: api_key.name,
            scopes: api_key.scopes,
            owner: api_key.owner,
            created_at: api_key.created_at,
            expires_at: api_key.expires_at,
            last_used: api_key.last_used,
            is_active: api_key.is_active,
            usage_count: api_key.usage_count,
            rate_limit: api_key.rate_limit,
            allowed_ips: api_key.allowed_ips,
            description: api_key.description,
        }
    }
}

/// On-disk representation of the API key store (see [`ApiKeyService`]).
#[derive(Debug, Default, Serialize, Deserialize)]
struct ApiKeyFile {
    keys: Vec<ApiKey>,
}

/// Persistent API key service.
///
/// Keys created via `POST /$/api-keys` are written to a JSON file (default:
/// `<data_dir>/api_keys.json`) so they survive process restarts and can
/// authenticate subsequent requests via [`validate_api_key_auth`]. Every
/// mutation (create/update/revoke) is flushed to disk with a temp-file +
/// `fsync` + atomic `rename`, so a crash mid-write cannot corrupt the
/// previously-valid file.
///
/// Usage counters (`last_used`/`usage_count`) are updated in memory only on
/// the per-request hot path (see [`update_usage`](Self::update_usage)) to
/// avoid an `fsync` on every authenticated request; they are best-effort and
/// may reset to their last-flushed value across a crash, unlike identity and
/// active/revoked state which are always durably persisted.
pub struct ApiKeyService {
    path: std::path::PathBuf,
    keys: RwLock<HashMap<String, ApiKey>>,
    /// Maximum active keys a single user may hold (`security.api_keys.max_keys_per_user`).
    max_keys_per_user: u32,
    /// Default key TTL in days used when a create request omits `expires_in_days`
    /// (`security.api_keys.default_expiration_days`).
    default_expiration_days: u32,
    /// Default rate limit applied to a key when a create request omits an
    /// explicit `rate_limit` (`security.api_keys.default_rate_limit`). When
    /// `None`, keys created without an explicit rate limit remain
    /// unlimited, matching the pre-existing behaviour.
    default_rate_limit: Option<RateLimit>,
}

/// Default per-user key cap when no `security.api_keys` config is supplied.
const DEFAULT_MAX_KEYS_PER_USER: u32 = 100;
/// Default key TTL (days) when no `security.api_keys` config is supplied.
const DEFAULT_KEY_EXPIRATION_DAYS: u32 = 365;

impl ApiKeyService {
    /// Maximum active keys permitted per user.
    pub fn max_keys_per_user(&self) -> u32 {
        self.max_keys_per_user
    }

    /// Default key expiration (days) applied when a request omits `expires_in_days`.
    pub fn default_expiration_days(&self) -> u32 {
        self.default_expiration_days
    }

    /// Default rate limit applied when a create request omits an explicit
    /// `rate_limit` (`security.api_keys.default_rate_limit`).
    pub fn default_rate_limit(&self) -> Option<RateLimit> {
        self.default_rate_limit.clone()
    }

    /// Count the active (non-revoked) keys currently owned by `owner`.
    pub async fn active_key_count_for(&self, owner: &str) -> usize {
        self.keys
            .read()
            .await
            .values()
            .filter(|k| k.owner == owner && k.is_active)
            .count()
    }

    /// Open (or create) the API key store at `path` with policy limits taken
    /// from the `security.api_keys` configuration.
    pub async fn open_with_config(
        path: impl Into<std::path::PathBuf>,
        config: &crate::config::config_security::ApiKeyConfig,
    ) -> FusekiResult<Self> {
        let default_rate_limit = config.default_rate_limit.as_ref().map(|r| RateLimit {
            requests_per_minute: r.requests_per_minute,
            requests_per_hour: r.requests_per_hour,
            requests_per_day: r.requests_per_day,
            burst_limit: r.burst_limit,
        });
        Self::open_inner(
            path.into(),
            config.max_keys_per_user,
            config.default_expiration_days,
            default_rate_limit,
        )
        .await
    }

    /// Open (or create) the API key store at `path` with default policy limits.
    pub async fn open(path: impl Into<std::path::PathBuf>) -> FusekiResult<Self> {
        Self::open_inner(
            path.into(),
            DEFAULT_MAX_KEYS_PER_USER,
            DEFAULT_KEY_EXPIRATION_DAYS,
            None,
        )
        .await
    }

    async fn open_inner(
        path: std::path::PathBuf,
        max_keys_per_user: u32,
        default_expiration_days: u32,
        default_rate_limit: Option<RateLimit>,
    ) -> FusekiResult<Self> {
        let keys = if path.exists() {
            let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
                FusekiError::internal(format!("Failed to read API key store {path:?}: {e}"))
            })?;
            let file: ApiKeyFile = serde_json::from_str(&content).map_err(|e| {
                FusekiError::internal(format!("Failed to parse API key store {path:?}: {e}"))
            })?;
            file.keys.into_iter().map(|k| (k.id.clone(), k)).collect()
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        FusekiError::internal(format!(
                            "Failed to create API key store directory {parent:?}: {e}"
                        ))
                    })?;
                }
            }
            HashMap::new()
        };

        info!(
            "API key service initialized at {:?} with {} existing key(s)",
            path,
            keys.len()
        );

        Ok(Self {
            path,
            keys: RwLock::new(keys),
            max_keys_per_user,
            default_expiration_days,
            default_rate_limit,
        })
    }

    /// Atomically persist the full key set to disk.
    async fn flush(&self, keys: &HashMap<String, ApiKey>) -> FusekiResult<()> {
        let file = ApiKeyFile {
            keys: keys.values().cloned().collect(),
        };
        let content = serde_json::to_string_pretty(&file)
            .map_err(|e| FusekiError::internal(format!("Failed to serialize API keys: {e}")))?;

        let tmp_path = self.path.with_extension("json.tmp");
        {
            let mut tmp_file = tokio::fs::File::create(&tmp_path).await.map_err(|e| {
                FusekiError::internal(format!("Failed to write API key store: {e}"))
            })?;
            tmp_file.write_all(content.as_bytes()).await.map_err(|e| {
                FusekiError::internal(format!("Failed to write API key store: {e}"))
            })?;
            tmp_file.sync_all().await.map_err(|e| {
                FusekiError::internal(format!("Failed to fsync API key store: {e}"))
            })?;
        }
        tokio::fs::rename(&tmp_path, &self.path)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to finalize API key store: {e}")))?;
        Ok(())
    }

    pub async fn store_api_key(&self, api_key: &ApiKey) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;
        keys.insert(api_key.id.clone(), api_key.clone());
        self.flush(&keys).await
    }

    pub async fn get_api_key(&self, key_id: &str) -> FusekiResult<Option<ApiKey>> {
        let keys = self.keys.read().await;
        Ok(keys.get(key_id).cloned())
    }

    pub async fn get_api_key_by_hash(&self, key_hash: &str) -> FusekiResult<Option<ApiKey>> {
        let keys = self.keys.read().await;
        Ok(keys.values().find(|k| k.key_hash == key_hash).cloned())
    }

    pub async fn list_api_keys(
        &self,
        owner: Option<&str>,
        active_only: bool,
        scope: Option<&ApiKeyScope>,
        limit: usize,
        offset: usize,
    ) -> FusekiResult<Vec<ApiKey>> {
        let keys = self.keys.read().await;
        let mut result: Vec<ApiKey> = keys
            .values()
            .filter(|k| owner.map(|o| k.owner == o).unwrap_or(true))
            .filter(|k| !active_only || k.is_active)
            .filter(|k| scope.map(|s| k.scopes.contains(s)).unwrap_or(true))
            .cloned()
            .collect();
        result.sort_by_key(|b| std::cmp::Reverse(b.created_at));
        Ok(result.into_iter().skip(offset).take(limit).collect())
    }

    pub async fn update_api_key(&self, api_key: &ApiKey) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;
        if !keys.contains_key(&api_key.id) {
            return Err(FusekiError::not_found("API key not found"));
        }
        keys.insert(api_key.id.clone(), api_key.clone());
        self.flush(&keys).await
    }

    pub async fn revoke_api_key(&self, key_id: &str) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;
        {
            let key = keys
                .get_mut(key_id)
                .ok_or_else(|| FusekiError::not_found("API key not found"))?;
            key.is_active = false;
        }
        self.flush(&keys).await
    }

    /// Record key usage. In-memory only (see struct docs) to keep the
    /// per-request authentication hot path free of a synchronous `fsync`.
    pub async fn update_usage(&self, api_key: &ApiKey) -> FusekiResult<()> {
        let mut keys = self.keys.write().await;
        if let Some(existing) = keys.get_mut(&api_key.id) {
            existing.last_used = api_key.last_used;
            existing.usage_count = api_key.usage_count;
        }
        Ok(())
    }

    pub async fn get_usage_stats(&self, key_id: &str) -> FusekiResult<ApiKeyUsageStats> {
        let keys = self.keys.read().await;
        let key = keys
            .get(key_id)
            .ok_or_else(|| FusekiError::not_found("API key not found"))?;
        // Only the running total is tracked today (no per-request timestamp
        // log), so the windowed counters honestly report zero rather than
        // fabricating a distribution. See struct docs for what is durable.
        Ok(ApiKeyUsageStats {
            total_requests: key.usage_count,
            requests_last_24h: 0,
            requests_last_7d: 0,
            requests_last_30d: 0,
            average_requests_per_day: 0.0,
            most_used_endpoints: vec![],
            error_rate: 0.0,
        })
    }

    /// Check whether `key_id` is still within `rate_limit`.
    ///
    /// NOTE: sliding-window request counting is not implemented yet (would
    /// require a per-key timestamp ring buffer); this always allows the
    /// request. Tracked as a follow-up — do not rely on this for abuse
    /// protection until implemented.
    pub async fn check_rate_limit(
        &self,
        _key_id: &str,
        _rate_limit: &RateLimit,
    ) -> FusekiResult<bool> {
        Ok(true)
    }
}

/// Default location for the API key JSON store, relative to the process
/// working directory (never an absolute path — see COOLJAPAN policy).
pub fn default_api_key_store_path() -> std::path::PathBuf {
    std::path::PathBuf::from("./data/api_keys.json")
}

async fn get_api_key_service(state: &AppState) -> FusekiResult<Arc<ApiKeyService>> {
    state
        .api_key_service
        .clone()
        .ok_or_else(|| FusekiError::service_unavailable("API key service not available"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_generation() {
        let key = generate_api_key();
        assert!(key.starts_with("oxirs_"));
        assert_eq!(key.len(), 38); // "oxirs_" + 32 characters
    }

    #[test]
    fn test_api_key_generation_is_not_deterministic() {
        // Regression test: generate_api_key() previously used a fixed-seed
        // RNG (Random::seed(42)), so every generated key was byte-for-byte
        // identical. It must now use a cryptographically secure, unseeded
        // source of randomness.
        let a = generate_api_key();
        let b = generate_api_key();
        assert_ne!(a, b, "API keys must not be deterministic");
    }

    fn unique_temp_store_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxirs_fuseki_test_api_keys_{}_{}.json",
            label,
            uuid::Uuid::new_v4()
        ))
    }

    #[tokio::test]
    async fn regression_api_key_config_limits_applied() {
        use crate::config::config_security::{
            ApiKeyConfig, ApiKeyStorageBackend, ApiKeyStorageConfig,
        };
        let path = unique_temp_store_path("limits");
        let config = ApiKeyConfig {
            enabled: true,
            default_expiration_days: 7,
            max_keys_per_user: 2,
            default_rate_limit: None,
            usage_analytics: false,
            storage: ApiKeyStorageConfig {
                backend: ApiKeyStorageBackend::File,
                connection: path.to_string_lossy().to_string(),
                encryption_key: None,
            },
        };
        let service = ApiKeyService::open_with_config(&path, &config)
            .await
            .expect("service opens");

        // The configured limits are honoured (previously these fields had no effect).
        assert_eq!(service.max_keys_per_user(), 2);
        assert_eq!(service.default_expiration_days(), 7);

        // active_key_count_for reflects stored keys.
        assert_eq!(service.active_key_count_for("bob").await, 0);
        let key = ApiKey {
            id: "k".to_string(),
            name: "n".to_string(),
            key_hash: "h".to_string(),
            scopes: vec![ApiKeyScope::SparqlRead],
            owner: "bob".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            last_used: None,
            is_active: true,
            usage_count: 0,
            rate_limit: None,
            allowed_ips: vec![],
            description: None,
        };
        service.store_api_key(&key).await.expect("store");
        assert_eq!(service.active_key_count_for("bob").await, 1);

        let _ = std::fs::remove_file(&path);
    }

    /// Regression test for the `ApiKeyConfig.default_rate_limit` finding:
    /// the configured default must be surfaced through `default_rate_limit()`
    /// so `create_api_key` can apply it to keys created without an explicit
    /// `rate_limit`, instead of the field being silently unenforced.
    #[tokio::test]
    async fn regression_api_key_default_rate_limit_applied() {
        use crate::config::config_security::{
            ApiKeyConfig, ApiKeyRateLimit, ApiKeyStorageBackend, ApiKeyStorageConfig,
        };
        let path = unique_temp_store_path("rate_limit");
        let config = ApiKeyConfig {
            enabled: true,
            default_expiration_days: 30,
            max_keys_per_user: 10,
            default_rate_limit: Some(ApiKeyRateLimit {
                requests_per_minute: 60,
                requests_per_hour: 1000,
                requests_per_day: 10000,
                burst_limit: 10,
            }),
            usage_analytics: false,
            storage: ApiKeyStorageConfig {
                backend: ApiKeyStorageBackend::File,
                connection: path.to_string_lossy().to_string(),
                encryption_key: None,
            },
        };
        let service = ApiKeyService::open_with_config(&path, &config)
            .await
            .expect("service opens");

        let default_limit = service
            .default_rate_limit()
            .expect("configured default_rate_limit must be surfaced");
        assert_eq!(default_limit.requests_per_minute, 60);
        assert_eq!(default_limit.requests_per_hour, 1000);
        assert_eq!(default_limit.requests_per_day, 10000);
        assert_eq!(default_limit.burst_limit, 10);

        let _ = std::fs::remove_file(&path);
    }

    /// A service opened without an explicit `ApiKeyConfig` (or with
    /// `default_rate_limit: None`) must not fabricate a rate limit.
    #[tokio::test]
    async fn regression_api_key_no_default_rate_limit_when_unconfigured() {
        let path = unique_temp_store_path("no_rate_limit");
        let service = ApiKeyService::open(&path).await.expect("service opens");
        assert!(service.default_rate_limit().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_api_key_service_persists_across_reopen() {
        let path = unique_temp_store_path("persist");

        let api_key = ApiKey {
            id: "key-1".to_string(),
            name: "test key".to_string(),
            key_hash: "deadbeef".to_string(),
            scopes: vec![ApiKeyScope::SparqlRead],
            owner: "alice".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            last_used: None,
            is_active: true,
            usage_count: 0,
            rate_limit: None,
            allowed_ips: vec![],
            description: None,
        };

        {
            let service = ApiKeyService::open(&path)
                .await
                .expect("service should open a fresh store");
            service
                .store_api_key(&api_key)
                .await
                .expect("store should succeed");
        }

        // Reopen from disk: the created key must still be there and usable
        // to authenticate, which is the whole point of persistence.
        let reopened = ApiKeyService::open(&path)
            .await
            .expect("service should reopen the persisted store");
        let found = reopened
            .get_api_key_by_hash("deadbeef")
            .await
            .expect("lookup should succeed")
            .expect("key created before reopen must still be present");
        assert_eq!(found.id, "key-1");
        assert!(found.is_active);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_api_key_service_revoke_persists() {
        let path = unique_temp_store_path("revoke");
        let service = ApiKeyService::open(&path)
            .await
            .expect("service should open a fresh store");

        let api_key = ApiKey {
            id: "key-2".to_string(),
            name: "revoke me".to_string(),
            key_hash: "cafef00d".to_string(),
            scopes: vec![ApiKeyScope::SparqlRead],
            owner: "bob".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            last_used: None,
            is_active: true,
            usage_count: 0,
            rate_limit: None,
            allowed_ips: vec![],
            description: None,
        };
        service
            .store_api_key(&api_key)
            .await
            .expect("store should succeed");

        service
            .revoke_api_key("key-2")
            .await
            .expect("revoke should succeed");

        // A brand-new service instance loading the same file must observe
        // the revocation (i.e. it was actually flushed to disk).
        let reopened = ApiKeyService::open(&path)
            .await
            .expect("service should reopen the persisted store");
        let found = reopened
            .get_api_key("key-2")
            .await
            .expect("lookup should succeed")
            .expect("revoked key should still exist, just inactive");
        assert!(!found.is_active);

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn test_extract_authenticated_user_fails_closed_without_credentials() {
        let config = crate::config::SecurityConfig::default();
        let auth_service = AuthService::new(config)
            .await
            .expect("auth service should construct with default config");

        let headers = HeaderMap::new();
        let result = extract_authenticated_user(&headers, &auth_service).await;
        assert!(
            result.is_err(),
            "unauthenticated requests must be rejected, never default to an implicit identity"
        );
    }

    #[test]
    fn test_api_key_hashing() {
        let key = "oxirs_test123";
        let hash1 = hash_api_key(key).unwrap();
        let hash2 = hash_api_key(key).unwrap();
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, key);
    }

    #[test]
    fn test_scope_to_permission_conversion() {
        let scopes = vec![
            ApiKeyScope::SparqlRead,
            ApiKeyScope::SparqlWrite,
            ApiKeyScope::Admin,
        ];

        let permissions = convert_scopes_to_permissions(&scopes);

        assert!(permissions.contains(&Permission::SparqlQuery));
        assert!(permissions.contains(&Permission::SparqlUpdate));
        assert!(permissions.contains(&Permission::GlobalAdmin));
    }

    #[test]
    fn test_dataset_specific_scopes() {
        let scopes = vec![
            ApiKeyScope::DatasetRead("dataset1".to_string()),
            ApiKeyScope::DatasetWrite("dataset2".to_string()),
        ];

        let permissions = convert_scopes_to_permissions(&scopes);

        assert!(permissions.contains(&Permission::DatasetRead("dataset1".to_string())));
        assert!(permissions.contains(&Permission::DatasetRead("dataset2".to_string())));
        assert!(permissions.contains(&Permission::DatasetWrite("dataset2".to_string())));
    }
}
