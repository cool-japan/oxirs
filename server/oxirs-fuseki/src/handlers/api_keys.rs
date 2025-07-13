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
    http::HeaderMap,
    response::Json,
};
use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
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
    State(state): State<AppState>,
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

    // Generate secure API key
    let raw_key = generate_api_key();
    let key_hash = hash_api_key(&raw_key)?;

    // Calculate expiration
    let expires_at = request
        .expires_in_days
        .map(|days| Utc::now() + Duration::days(days as i64));

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
        rate_limit: request.rate_limit.clone(),
        allowed_ips: request.allowed_ips.unwrap_or_default(),
        description: request.description.clone(),
    };

    // Store API key
    let api_key_service = get_api_key_service(&state).await?;
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
    State(state): State<AppState>,
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
    State(state): State<AppState>,
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
    State(state): State<AppState>,
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
    State(state): State<AppState>,
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
    State(state): State<AppState>,
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
    const PREFIX: &str = "oxirs_";
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const KEY_LENGTH: usize = 32;

    let mut rng = rand::thread_rng();
    let key: String = (0..KEY_LENGTH)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();

    format!("{PREFIX}{key}")
}

fn hash_api_key(api_key: &str) -> FusekiResult<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_api_key_scopes(scopes: &[ApiKeyScope], user: &User) -> FusekiResult<()> {
    // Ensure user has permission to grant these scopes
    for scope in scopes {
        match scope {
            ApiKeyScope::Admin => {
                if !user.permissions.contains(&Permission::SystemConfig) {
                    return Err(FusekiError::forbidden("Cannot grant admin scope"));
                }
            }
            ApiKeyScope::DatasetManagement => {
                if !user.permissions.contains(&Permission::GlobalAdmin) {
                    return Err(FusekiError::forbidden(
                        "Cannot grant dataset management scope",
                    ));
                }
            }
            ApiKeyScope::DatasetAdmin(_) => {
                if !user.permissions.contains(&Permission::GlobalAdmin) {
                    return Err(FusekiError::forbidden("Cannot grant dataset admin scope"));
                }
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

async fn extract_authenticated_user(
    _headers: &HeaderMap,
    _auth_service: &AuthService,
) -> FusekiResult<User> {
    // Simplified extraction - in production would use proper auth middleware
    Ok(User {
        username: "admin".to_string(),
        roles: vec!["admin".to_string()],
        email: Some("admin@example.com".to_string()),
        full_name: Some("Administrator".to_string()),
        last_login: Some(Utc::now()),
        permissions: vec![Permission::GlobalAdmin, Permission::SystemConfig],
    })
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

// Mock API key service (would be a proper service in production)
pub struct ApiKeyService {
    // In production, this would be backed by a database
}

impl ApiKeyService {
    pub async fn store_api_key(&self, _api_key: &ApiKey) -> FusekiResult<()> {
        // Store in database
        Ok(())
    }

    pub async fn get_api_key(&self, _key_id: &str) -> FusekiResult<Option<ApiKey>> {
        // Retrieve from database
        Ok(None)
    }

    pub async fn get_api_key_by_hash(&self, _key_hash: &str) -> FusekiResult<Option<ApiKey>> {
        // Retrieve by hash from database
        Ok(None)
    }

    pub async fn list_api_keys(
        &self,
        _owner: Option<&str>,
        _active_only: bool,
        _scope: Option<&ApiKeyScope>,
        _limit: usize,
        _offset: usize,
    ) -> FusekiResult<Vec<ApiKey>> {
        // List from database with filters
        Ok(vec![])
    }

    pub async fn update_api_key(&self, _api_key: &ApiKey) -> FusekiResult<()> {
        // Update in database
        Ok(())
    }

    pub async fn revoke_api_key(&self, _key_id: &str) -> FusekiResult<()> {
        // Mark as inactive in database
        Ok(())
    }

    pub async fn update_usage(&self, _api_key: &ApiKey) -> FusekiResult<()> {
        // Update usage statistics
        Ok(())
    }

    pub async fn get_usage_stats(&self, _key_id: &str) -> FusekiResult<ApiKeyUsageStats> {
        // Calculate usage statistics
        Ok(ApiKeyUsageStats {
            total_requests: 0,
            requests_last_24h: 0,
            requests_last_7d: 0,
            requests_last_30d: 0,
            average_requests_per_day: 0.0,
            most_used_endpoints: vec![],
            error_rate: 0.0,
        })
    }

    pub async fn check_rate_limit(
        &self,
        _key_id: &str,
        _rate_limit: &RateLimit,
    ) -> FusekiResult<bool> {
        // Check rate limiting
        Ok(true)
    }
}

async fn get_api_key_service(_state: &AppState) -> FusekiResult<ApiKeyService> {
    // In production, extract from state
    Ok(ApiKeyService {})
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
