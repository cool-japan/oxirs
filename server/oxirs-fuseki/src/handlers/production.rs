//! Production management handlers for RC.1 features
//!
//! This module provides HTTP endpoints for managing production features:
//! - Load balancing configuration and statistics
//! - Edge caching management and purging
//! - CDN support for static assets

use crate::edge_caching::{EdgeCacheStatistics, InvalidationStrategy};
use crate::load_balancing::{Backend, BackendStatistics, LoadBalancingStrategy};
use crate::server::AppState;
use axum::{
    extract::{Path, State},
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;

// ============================================================================
// Load Balancing Endpoints
// ============================================================================

/// Load balancer status response
#[derive(Debug, Serialize)]
pub struct LoadBalancerStatus {
    pub enabled: bool,
    pub strategy: String,
    pub backend_count: usize,
    pub healthy_backends: usize,
    pub sticky_sessions: bool,
}

/// GET /$/load-balancer/status - Get load balancer status
#[instrument(skip(state))]
pub async fn load_balancer_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<LoadBalancerStatus>, StatusCode> {
    if let Some(ref lb) = state.load_balancer {
        let stats = lb.get_statistics();
        let healthy = stats.values().filter(|s| s.is_healthy).count();

        // Get strategy name from config (using default for now)
        let strategy_name = "round_robin".to_string();

        Ok(Json(LoadBalancerStatus {
            enabled: true,
            strategy: strategy_name,
            backend_count: stats.len(),
            healthy_backends: healthy,
            sticky_sessions: false,
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// GET /$/load-balancer/backends - List all backends
#[instrument(skip(state))]
pub async fn list_backends(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(ref lb) = state.load_balancer {
        Json(lb.get_statistics()).into_response()
    } else {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }
}

/// Request to add a backend
#[derive(Debug, Deserialize)]
pub struct AddBackendRequest {
    pub id: String,
    pub url: String,
    pub weight: Option<u32>,
    pub max_connections: Option<usize>,
    pub health_check_url: Option<String>,
}

/// POST /$/load-balancer/backends - Add a new backend
#[instrument(skip(state))]
pub async fn add_backend(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddBackendRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    if let Some(ref lb) = state.load_balancer {
        let backend = Backend {
            id: req.id,
            url: req.url,
            weight: req.weight.unwrap_or(1),
            max_connections: req.max_connections.unwrap_or(100),
            health_check_url: req.health_check_url,
            enabled: true,
        };

        lb.add_backend(backend)
            .map(|_| StatusCode::CREATED)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Load balancer not available".to_string(),
        ))
    }
}

/// DELETE /$/load-balancer/backends/:id - Remove a backend
#[instrument(skip(state))]
pub async fn remove_backend(
    Path(backend_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, (StatusCode, String)> {
    if let Some(ref lb) = state.load_balancer {
        lb.remove_backend(&backend_id)
            .map(|_| StatusCode::NO_CONTENT)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Load balancer not available".to_string(),
        ))
    }
}

/// Response for backend selection
#[derive(Debug, Serialize)]
pub struct SelectedBackend {
    pub backend_id: String,
    pub url: String,
}

/// POST /$/load-balancer/select - Select a backend (for testing)
#[instrument(skip(state))]
pub async fn select_backend(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SelectedBackend>, (StatusCode, String)> {
    if let Some(ref lb) = state.load_balancer {
        lb.select_backend(None, None)
            .map(|backend| {
                Json(SelectedBackend {
                    backend_id: backend.id,
                    url: backend.url,
                })
            })
            .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Load balancer not available".to_string(),
        ))
    }
}

// ============================================================================
// Edge Caching Endpoints
// ============================================================================

/// GET /$/edge-cache/status - Get edge cache status
#[instrument(skip(state))]
pub async fn edge_cache_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<EdgeCacheStatistics>, StatusCode> {
    if let Some(ref cache) = state.edge_cache_manager {
        Ok(Json(cache.get_statistics()))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Request to purge cache
#[derive(Debug, Deserialize)]
pub struct PurgeRequest {
    pub strategy: String,
    pub targets: Option<Vec<String>>,
}

/// POST /$/edge-cache/purge - Purge cache
#[instrument(skip(state))]
pub async fn purge_cache(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PurgeRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    if let Some(ref cache) = state.edge_cache_manager {
        match req.strategy.as_str() {
            "all" => cache
                .purge_all()
                .await
                .map(|_| StatusCode::OK)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
            "tags" => {
                if let Some(tags) = req.targets {
                    cache
                        .purge_by_tags(tags)
                        .await
                        .map(|_| StatusCode::OK)
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
                } else {
                    Err((StatusCode::BAD_REQUEST, "Tags required".to_string()))
                }
            }
            "dataset" => {
                if let Some(targets) = req.targets {
                    if let Some(dataset) = targets.first() {
                        cache
                            .purge_dataset(dataset)
                            .await
                            .map(|_| StatusCode::OK)
                            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
                    } else {
                        Err((StatusCode::BAD_REQUEST, "Dataset name required".to_string()))
                    }
                } else {
                    Err((StatusCode::BAD_REQUEST, "Dataset name required".to_string()))
                }
            }
            _ => Err((StatusCode::BAD_REQUEST, "Invalid strategy".to_string())),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Edge cache not available".to_string(),
        ))
    }
}

/// GET /$/edge-cache/headers - Get cache headers for a query
#[derive(Debug, Deserialize)]
pub struct CacheHeadersRequest {
    pub query: String,
    pub execution_time_ms: u64,
    pub response_size: usize,
}

/// POST /$/edge-cache/headers - Get recommended cache headers for a query
#[instrument(skip(state))]
pub async fn get_cache_headers(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CacheHeadersRequest>,
) -> Result<Json<HashMap<String, String>>, StatusCode> {
    if let Some(ref cache) = state.edge_cache_manager {
        match cache.get_cache_headers(&req.query, req.execution_time_ms, req.response_size) {
            Some(headers) => Ok(Json(headers)),
            None => Ok(Json(HashMap::new())),
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// ============================================================================
// CDN Static Asset Support
// ============================================================================

/// Static asset configuration
#[derive(Debug, Serialize)]
pub struct CdnConfig {
    pub enabled: bool,
    pub base_url: Option<String>,
    pub default_ttl_secs: u64,
    pub supported_types: Vec<String>,
}

/// GET /$/cdn/config - Get CDN configuration
#[instrument]
pub async fn cdn_config() -> Json<CdnConfig> {
    // CDN configuration (can be extended to read from config)
    Json(CdnConfig {
        enabled: true,
        base_url: None,          // Use same origin by default
        default_ttl_secs: 86400, // 24 hours
        supported_types: vec![
            "text/css".to_string(),
            "text/javascript".to_string(),
            "application/javascript".to_string(),
            "image/png".to_string(),
            "image/jpeg".to_string(),
            "image/svg+xml".to_string(),
            "font/woff2".to_string(),
            "font/woff".to_string(),
        ],
    })
}

/// Serve static asset with CDN headers.
///
/// Files are served from the directory configured at
/// `server.static_asset_dir`. When no root directory is configured the
/// endpoint fails loud with `503 Service Unavailable` rather than silently
/// pretending the asset does not exist. The requested path is resolved and
/// canonicalized against the configured root and rejected with `400 Bad
/// Request` if it would escape the root (path traversal).
#[instrument(skip(state))]
pub async fn serve_static_asset(
    Path(path): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let content_type = match path.rsplit('.').next() {
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("json") => "application/json",
        Some("html") => "text/html",
        _ => "application/octet-stream",
    }
    .to_string();

    let cache_control = "public, max-age=86400, stale-while-revalidate=3600".to_string();

    let Some(root) = state.config.server.static_asset_dir.clone() else {
        return static_asset_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Static asset serving is not configured (server.static_asset_dir is unset)",
        );
    };

    match resolve_static_asset_path(&root, &path) {
        Ok(resolved) => match tokio::fs::read(&resolved).await {
            Ok(bytes) => {
                let mut headers = axum::http::HeaderMap::new();
                if let Ok(v) = HeaderValue::from_str(&content_type) {
                    headers.insert(axum::http::header::CONTENT_TYPE, v);
                }
                if let Ok(v) = HeaderValue::from_str(&cache_control) {
                    headers.insert(axum::http::header::CACHE_CONTROL, v);
                }
                headers.insert(
                    axum::http::header::VARY,
                    HeaderValue::from_static("Accept-Encoding"),
                );
                headers.insert(
                    axum::http::header::HeaderName::from_static("x-content-type-options"),
                    HeaderValue::from_static("nosniff"),
                );
                (StatusCode::OK, headers, bytes).into_response()
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                static_asset_error(StatusCode::NOT_FOUND, "Static asset not found")
            }
            Err(err) => {
                tracing::warn!(error = %err, path = %path, "failed to read static asset");
                static_asset_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to read static asset",
                )
            }
        },
        Err(StaticAssetPathError::Traversal) => {
            static_asset_error(StatusCode::BAD_REQUEST, "Invalid asset path")
        }
        Err(StaticAssetPathError::IsDirectory) | Err(StaticAssetPathError::NotFound) => {
            static_asset_error(StatusCode::NOT_FOUND, "Static asset not found")
        }
    }
}

/// Error resolving a requested static-asset path against the configured root.
enum StaticAssetPathError {
    /// The requested path attempted to escape the configured root directory,
    /// or names a component that cannot possibly resolve under the root
    /// (e.g. its parent directory does not exist).
    Traversal,
    /// The resolved path is a directory, not a servable file.
    IsDirectory,
    /// The path does not name an existing entry at all.
    NotFound,
}

/// Resolve `requested` (the URL path captured after `/static/`) against
/// `root`, rejecting any attempt to escape the root via `..`, absolute
/// paths, or symlink traversal. Returns the resolved absolute filesystem
/// path on success. The returned path is guaranteed to exist and to live
/// under the canonicalized root.
fn resolve_static_asset_path(
    root: &std::path::Path,
    requested: &str,
) -> Result<std::path::PathBuf, StaticAssetPathError> {
    let mut relative = std::path::PathBuf::new();
    for component in std::path::Path::new(requested).components() {
        match component {
            std::path::Component::Normal(part) => relative.push(part),
            std::path::Component::CurDir => {}
            // Any parent-dir, root, or prefix component in a URL-decoded
            // request path is a traversal attempt (or nonsensical) — reject.
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => return Err(StaticAssetPathError::Traversal),
        }
    }

    let candidate = root.join(&relative);
    if !candidate.exists() {
        return Err(StaticAssetPathError::NotFound);
    }
    if candidate.is_dir() {
        return Err(StaticAssetPathError::IsDirectory);
    }

    // Canonicalize both sides and verify the resolved file still lives
    // under the root. This also catches traversal performed through
    // symlinks planted inside the root directory. If the root itself
    // cannot be canonicalized (e.g. misconfigured/missing), fail loud
    // rather than silently serving nothing.
    let canonical_root = root
        .canonicalize()
        .map_err(|_| StaticAssetPathError::NotFound)?;
    let canonical_candidate = candidate
        .canonicalize()
        .map_err(|_| StaticAssetPathError::NotFound)?;
    if !canonical_candidate.starts_with(&canonical_root) {
        return Err(StaticAssetPathError::Traversal);
    }

    Ok(canonical_candidate)
}

/// Build a minimal error response for the static asset endpoint that still
/// carries CDN-style security headers.
fn static_asset_error(status: StatusCode, message: &'static str) -> Response {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    (status, headers, message).into_response()
}

// ============================================================================
// Security Audit Endpoints
// ============================================================================

/// Security audit status
#[derive(Debug, Serialize)]
pub struct SecurityAuditStatus {
    pub enabled: bool,
    pub total_events: usize,
    pub critical_events: usize,
    pub last_scan: Option<String>,
    pub compliance_status: String,
}

/// GET /$/security/audit/status - Get security audit status
#[instrument(skip(state))]
pub async fn security_audit_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SecurityAuditStatus>, StatusCode> {
    if let Some(ref auditor) = state.security_auditor {
        // Get vulnerabilities to determine status
        let vulnerabilities = auditor.get_vulnerabilities().await;
        let critical_count = vulnerabilities
            .iter()
            .filter(|v| matches!(v.severity, crate::security_audit::Severity::Critical))
            .count();

        Ok(Json(SecurityAuditStatus {
            enabled: true,
            total_events: vulnerabilities.len(),
            critical_events: critical_count,
            last_scan: Some(chrono::Utc::now().to_rfc3339()),
            compliance_status: if critical_count == 0 {
                "compliant".to_string()
            } else if critical_count <= 2 {
                "partial".to_string()
            } else {
                "non-compliant".to_string()
            },
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// POST /$/security/audit/scan - Trigger a security scan
#[instrument(skip(state))]
pub async fn trigger_security_scan(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    if let Some(ref auditor) = state.security_auditor {
        // Trigger async scan
        let _ = auditor.perform_security_scan().await;
        Ok(StatusCode::ACCEPTED)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// ============================================================================
// DDoS Protection Endpoints
// ============================================================================

/// DDoS protection status
#[derive(Debug, Serialize)]
pub struct DDoSStatus {
    pub enabled: bool,
    pub blocked_ips: usize,
    pub rate_limited_requests: u64,
    pub total_requests: u64,
    pub current_connections: usize,
}

/// GET /$/security/ddos/status - Get DDoS protection status
#[instrument(skip(state))]
pub async fn ddos_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DDoSStatus>, StatusCode> {
    if let Some(ref protector) = state.ddos_protector {
        let stats = protector.get_statistics();
        Ok(Json(DDoSStatus {
            enabled: stats.enabled,
            blocked_ips: stats.blocked_ips_count,
            rate_limited_requests: stats.total_violations as u64,
            total_requests: stats.total_requests,
            current_connections: stats.total_ips_tracked,
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Request to manage IP
#[derive(Debug, Deserialize)]
pub struct IpManagementRequest {
    pub ip: String,
    pub action: String, // "whitelist", "blacklist", "unblock"
}

/// POST /$/security/ddos/manage-ip - Manage IP addresses
#[instrument(skip(state))]
pub async fn manage_ip(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IpManagementRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    if let Some(ref protector) = state.ddos_protector {
        let ip: std::net::IpAddr = req
            .ip
            .parse()
            .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid IP address".to_string()))?;

        match req.action.as_str() {
            "whitelist" => {
                protector
                    .whitelist_ip(ip)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                Ok(StatusCode::OK)
            }
            "blacklist" => {
                protector
                    .block_ip(ip, crate::ddos_protection::BlockReason::ManualBlock)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                Ok(StatusCode::OK)
            }
            "unblock" => {
                protector
                    .unblock_ip(ip)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                Ok(StatusCode::OK)
            }
            _ => Err((StatusCode::BAD_REQUEST, "Invalid action".to_string())),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "DDoS protection not available".to_string(),
        ))
    }
}

// ============================================================================
// Disaster Recovery Endpoints
// ============================================================================

/// Disaster recovery status
#[derive(Debug, Serialize)]
pub struct DisasterRecoveryStatus {
    pub enabled: bool,
    pub rpo_minutes: u64,
    pub rto_minutes: u64,
    pub last_recovery_point: Option<String>,
    pub health_status: String,
}

/// GET /$/recovery/status - Get disaster recovery status
#[instrument(skip(state))]
pub async fn disaster_recovery_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<DisasterRecoveryStatus>, StatusCode> {
    if let Some(ref recovery) = state.disaster_recovery {
        let status = recovery.get_status().await;
        Ok(Json(DisasterRecoveryStatus {
            enabled: status.enabled,
            rpo_minutes: status.rpo_minutes,
            rto_minutes: status.rto_minutes,
            last_recovery_point: status.last_recovery_test.map(|t| t.to_rfc3339()),
            health_status: if status.healthy {
                "healthy".to_string()
            } else {
                "degraded".to_string()
            },
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// POST /$/recovery/create-point - Create a recovery point
#[instrument(skip(state))]
pub async fn create_recovery_point(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, (StatusCode, String)> {
    if let Some(ref recovery) = state.disaster_recovery {
        recovery
            .create_recovery_point("Manual recovery point".to_string())
            .await
            .map(|_| StatusCode::CREATED)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Disaster recovery not available".to_string(),
        ))
    }
}

// ============================================================================
// Metrics Endpoint
// ============================================================================

/// GET /metrics - Prometheus metrics export
#[instrument(skip(state))]
pub async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> Result<String, (StatusCode, String)> {
    if let Some(ref metrics_service) = state.metrics_service {
        let result: Result<String, crate::error::FusekiError> =
            metrics_service.get_prometheus_metrics().await;
        result.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Metrics service not available".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdn_config() {
        let config = CdnConfig {
            enabled: true,
            base_url: None,
            default_ttl_secs: 86400,
            supported_types: vec!["text/css".to_string()],
        };
        assert!(config.enabled);
        assert_eq!(config.default_ttl_secs, 86400);
    }

    #[test]
    fn test_load_balancer_status() {
        let status = LoadBalancerStatus {
            enabled: true,
            strategy: "round_robin".to_string(),
            backend_count: 3,
            healthy_backends: 3,
            sticky_sessions: false,
        };
        assert!(status.enabled);
        assert_eq!(status.backend_count, 3);
    }
}
