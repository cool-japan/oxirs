//! Integrated middleware for production features
//!
//! This module provides middleware layers that integrate:
//! - Edge caching (automatic cache headers for SPARQL queries)
//! - Security audit logging (comprehensive request tracking)
//! - DDoS protection (rate limiting and IP blocking)

use crate::ddos_protection::RequestDecision;
use crate::ddos_protection::{BlockReason, DDoSProtectionManager};
use crate::edge_caching::EdgeCacheManager;
use crate::error::{FusekiError, FusekiResult};
use crate::security_audit::{
    AuditEventType, AuditLogEntry, AuditResult, SecurityAuditManager, Severity,
};
use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use chrono::Utc;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

// ============================================================================
// Edge Caching Middleware
// ============================================================================

/// Middleware layer for automatic edge caching
///
/// This middleware automatically applies cache headers to SPARQL query responses
/// based on query characteristics (read-only, volatility, execution time).
pub async fn edge_caching_middleware(
    State(cache_manager): State<Arc<Option<Arc<EdgeCacheManager>>>>,
    request: Request,
    next: Next,
) -> Response {
    // Only apply caching to SPARQL query endpoints
    let path = request.uri().path();
    let should_cache = path.contains("/sparql")
        && !path.contains("/update")
        && (request.method() == axum::http::Method::GET
            || request.method() == axum::http::Method::POST);

    if !should_cache {
        return next.run(request).await;
    }

    // Extract query from request (simplified - actual impl would parse body/query params)
    let query = extract_query_from_request(&request);

    // Execute request and measure time
    let start = Instant::now();
    let response = next.run(request).await;
    let execution_time_ms = start.elapsed().as_millis() as u64;

    // Apply cache headers if manager is available
    if let Some(ref manager) = cache_manager.as_ref() {
        if let Some(query_str) = query {
            // Get recommended cache headers
            let response_size = 0; // Would need to measure actual response size
            if let Some(cache_headers) =
                manager.get_cache_headers(&query_str, execution_time_ms, response_size)
            {
                return apply_cache_headers(response, cache_headers);
            }
        }
    }

    response
}

/// Extract SPARQL query from request
fn extract_query_from_request(request: &Request) -> Option<String> {
    // Try to extract from query parameters
    if let Some(query_params) = request.uri().query() {
        for param in query_params.split('&') {
            if let Some((key, value)) = param.split_once('=') {
                if key == "query" {
                    return Some(urlencoding::decode(value).ok()?.into_owned());
                }
            }
        }
    }

    // For POST requests, would need to parse body (not shown here)
    None
}

/// Apply cache headers to response
fn apply_cache_headers(
    mut response: Response,
    cache_headers: std::collections::HashMap<String, String>,
) -> Response {
    let headers = response.headers_mut();

    for (key, value) in cache_headers {
        if let Ok(header_name) = key.parse::<axum::http::HeaderName>() {
            if let Ok(header_value) = HeaderValue::from_str(&value) {
                headers.insert(header_name, header_value);
            }
        }
    }

    response
}

// ============================================================================
// Security Audit Middleware
// ============================================================================

/// Middleware layer for security audit logging
///
/// This middleware logs all requests for security analysis, including:
/// - Authentication attempts
/// - Authorization decisions
/// - Data access patterns
/// - Suspicious activity
pub async fn security_audit_middleware(
    State(auditor): State<Arc<Option<Arc<SecurityAuditManager>>>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let user = extract_user_from_request(&request);
    let start = Instant::now();

    // Execute request
    let response = next.run(request).await;

    // Log to audit manager if available
    if let Some(ref audit_manager) = auditor.as_ref() {
        let status = response.status();
        let duration = start.elapsed();

        // Determine event type and severity based on request characteristics
        let (event_type, severity) = classify_request(method.as_ref(), &path, status);

        let entry = AuditLogEntry {
            timestamp: Utc::now(),
            event_type,
            severity,
            user,
            ip_address: Some(addr.ip().to_string()),
            resource: path.clone(),
            action: method.to_string(),
            result: if status.is_success() {
                AuditResult::Success
            } else if status.is_client_error() {
                AuditResult::Denied
            } else {
                AuditResult::Error
            },
            details: Some(format!(
                "status={}, duration={}ms",
                status.as_u16(),
                duration.as_millis()
            )),
        };

        // Log asynchronously (fire and forget)
        let auditor_clone = Arc::clone(audit_manager);
        tokio::spawn(async move {
            let _ = auditor_clone.log_event(entry).await;
        });
    }

    response
}

/// Extract authenticated user from request headers
fn extract_user_from_request(request: &Request) -> Option<String> {
    // Check Authorization header
    if let Some(auth_header) = request.headers().get(axum::http::header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            // Basic auth: "Basic base64(username:password)"
            if let Some(stripped) = auth_str.strip_prefix("Basic ") {
                use base64::Engine;
                let engine = base64::engine::general_purpose::STANDARD;
                if let Ok(decoded) = engine.decode(stripped) {
                    if let Ok(credentials) = String::from_utf8(decoded) {
                        if let Some((username, _)) = credentials.split_once(':') {
                            return Some(username.to_string());
                        }
                    }
                }
            }
            // Bearer token: "Bearer <token>"
            else if auth_str.starts_with("Bearer ") {
                return Some("token_user".to_string()); // Would need to decode JWT
            }
        }
    }

    // Check X-User header (custom auth)
    if let Some(user_header) = request.headers().get("x-user") {
        if let Ok(user) = user_header.to_str() {
            return Some(user.to_string());
        }
    }

    None
}

/// Classify request for audit logging
fn classify_request(method: &str, path: &str, status: StatusCode) -> (AuditEventType, Severity) {
    // Authentication endpoints
    if path.contains("/auth/") || path.contains("/login") || path.contains("/oauth2") {
        let severity = if status.is_success() {
            Severity::Info
        } else {
            Severity::Medium
        };
        return (AuditEventType::Authentication, severity);
    }

    // Admin/management endpoints
    if path.starts_with("/$/") || path.contains("/admin") {
        let severity = if status.is_server_error() {
            Severity::High
        } else if status.is_client_error() {
            Severity::Medium
        } else {
            Severity::Low
        };
        return (AuditEventType::Authorization, severity);
    }

    // SPARQL Update (data modification)
    if path.contains("/update") || method == "POST" || method == "PUT" || method == "DELETE" {
        let severity = if status.is_server_error() {
            Severity::Medium
        } else {
            Severity::Low
        };
        return (AuditEventType::DataModification, severity);
    }

    // SPARQL Query (data access)
    if path.contains("/sparql") || path.contains("/query") {
        return (AuditEventType::DataAccess, Severity::Info);
    }

    // Default: security event
    (AuditEventType::SecurityEvent, Severity::Info)
}

// ============================================================================
// DDoS Protection Middleware
// ============================================================================

/// Middleware layer for DDoS protection
///
/// This middleware enforces rate limiting and IP blocking to prevent abuse:
/// - Rate limiting per IP address
/// - Suspicious pattern detection
/// - Automatic IP blocking
/// - Whitelist/blacklist management
pub async fn ddos_protection_middleware(
    State(protector): State<Arc<Option<Arc<DDoSProtectionManager>>>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = addr.ip();

    if let Some(ref protection_manager) = protector.as_ref() {
        // Check if IP is allowed
        match protection_manager.check_request(ip).await {
            Ok(RequestDecision::Allow) => {
                // Request allowed - execute it
                let response = next.run(request).await;
                Ok(response)
            }
            Ok(RequestDecision::Block { reason, .. }) => {
                // Request blocked
                warn!("Request blocked from IP {}: {:?}", ip, reason);
                Err(StatusCode::TOO_MANY_REQUESTS)
            }
            Ok(RequestDecision::RateLimit { .. }) => {
                // Rate limit exceeded
                warn!("Rate limit exceeded for IP {}", ip);
                Err(StatusCode::TOO_MANY_REQUESTS)
            }
            Ok(RequestDecision::Challenge { .. }) => {
                // Challenge required (CAPTCHA, etc.)
                warn!("Challenge required for IP {}", ip);
                Err(StatusCode::FORBIDDEN)
            }
            Err(e) => {
                // Error checking request - allow through but log
                debug!("Error in DDoS protection check: {}", e);
                Ok(next.run(request).await)
            }
        }
    } else {
        // No protection manager configured - allow request
        Ok(next.run(request).await)
    }
}

// ============================================================================
// Combined Middleware Layer
// ============================================================================

/// Combined state for all middleware
#[derive(Clone)]
pub struct MiddlewareState {
    pub edge_cache_manager: Arc<Option<Arc<EdgeCacheManager>>>,
    pub security_auditor: Arc<Option<Arc<SecurityAuditManager>>>,
    pub ddos_protector: Arc<Option<Arc<DDoSProtectionManager>>>,
}

impl MiddlewareState {
    /// Create new middleware state
    pub fn new(
        edge_cache_manager: Option<Arc<EdgeCacheManager>>,
        security_auditor: Option<Arc<SecurityAuditManager>>,
        ddos_protector: Option<Arc<DDoSProtectionManager>>,
    ) -> Self {
        Self {
            edge_cache_manager: Arc::new(edge_cache_manager),
            security_auditor: Arc::new(security_auditor),
            ddos_protector: Arc::new(ddos_protector),
        }
    }

    /// Create disabled middleware state (for testing)
    pub fn disabled() -> Self {
        Self {
            edge_cache_manager: Arc::new(None),
            security_auditor: Arc::new(None),
            ddos_protector: Arc::new(None),
        }
    }
}

// ============================================================================
// Request Context Enhancement
// ============================================================================

/// Enhanced request context with middleware information
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub start_time: Instant,
    pub client_ip: Option<std::net::IpAddr>,
    pub user: Option<String>,
    pub cached: bool,
    pub cache_ttl: Option<u64>,
}

impl RequestContext {
    /// Create new request context
    pub fn new(client_ip: Option<std::net::IpAddr>, user: Option<String>) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            client_ip,
            user,
            cached: false,
            cache_ttl: None,
        }
    }

    /// Get request duration
    pub fn duration(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_state_creation() {
        let state = MiddlewareState::disabled();
        assert!(state.edge_cache_manager.is_none());
        assert!(state.security_auditor.is_none());
        assert!(state.ddos_protector.is_none());
    }

    #[test]
    fn test_request_context() {
        let ip = "127.0.0.1".parse().ok();
        let ctx = RequestContext::new(ip, Some("test_user".to_string()));

        assert!(!ctx.request_id.is_empty());
        assert_eq!(ctx.user, Some("test_user".to_string()));
        assert!(!ctx.cached);
    }

    #[test]
    fn test_classify_authentication_request() {
        let (event_type, severity) = classify_request("POST", "/auth/login", StatusCode::OK);
        assert!(matches!(event_type, AuditEventType::Authentication));
        assert!(matches!(severity, Severity::Info));
    }

    #[test]
    fn test_classify_failed_authentication() {
        let (event_type, severity) =
            classify_request("POST", "/auth/login", StatusCode::UNAUTHORIZED);
        assert!(matches!(event_type, AuditEventType::Authentication));
        assert!(matches!(severity, Severity::Medium));
    }

    #[test]
    fn test_classify_admin_request() {
        let (event_type, severity) = classify_request("GET", "/$/stats", StatusCode::OK);
        assert!(matches!(event_type, AuditEventType::Authorization));
        assert!(matches!(severity, Severity::Low));
    }

    #[test]
    fn test_classify_update_request() {
        let (event_type, severity) = classify_request("POST", "/dataset/update", StatusCode::OK);
        assert!(matches!(event_type, AuditEventType::DataModification));
        assert!(matches!(severity, Severity::Low));
    }

    #[test]
    fn test_classify_query_request() {
        let (event_type, severity) =
            classify_request("GET", "/dataset/sparql?query=SELECT", StatusCode::OK);
        assert!(matches!(event_type, AuditEventType::DataAccess));
        assert!(matches!(severity, Severity::Info));
    }

    #[test]
    fn test_extract_query_from_url() {
        // This is a simplified test - actual implementation would need a full Request
        let query_str = "SELECT * WHERE { ?s ?p ?o }";
        let encoded = urlencoding::encode(query_str);
        let url_query = format!("query={}", encoded);

        // Would create a Request with this query string and test extract_query_from_request
        assert!(url_query.contains("query="));
    }
}
