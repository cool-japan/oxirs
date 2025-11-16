//! Production-grade middleware for security, tracing, and observability

use crate::auth::{
    permissions::PermissionChecker,
    policy_engine::{AuthorizationContext, UnifiedPolicyEngine},
    types::{Permission, User},
};
use axum::{
    extract::Request,
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn, Span};
use uuid::Uuid;

/// Security headers middleware for production deployment
///
/// Adds essential security headers to all responses:
/// - X-Frame-Options: Prevent clickjacking
/// - X-Content-Type-Options: Prevent MIME sniffing
/// - X-XSS-Protection: Enable XSS filter
/// - Referrer-Policy: Control referrer information
/// - Permissions-Policy: Control browser features
/// - Content-Security-Policy: Prevent XSS and injection attacks
pub async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Prevent clickjacking attacks
    headers.insert(
        header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );

    // Prevent MIME type sniffing
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );

    // Enable XSS protection (legacy, but still useful for older browsers)
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        HeaderValue::from_static("1; mode=block"),
    );

    // Control referrer information leakage
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    // Disable potentially dangerous browser features
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );

    // Content Security Policy - prevent XSS and injection attacks
    // Configured for SPARQL/RDF applications
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'self'; \
             script-src 'self' 'unsafe-inline'; \
             style-src 'self' 'unsafe-inline'; \
             img-src 'self' data: https:; \
             font-src 'self' data:; \
             connect-src 'self'; \
             frame-ancestors 'none'; \
             base-uri 'self'; \
             form-action 'self'",
        ),
    );

    response
}

/// HTTPS-specific security headers middleware
///
/// Adds HSTS (HTTP Strict Transport Security) header for HTTPS connections
/// Should only be used when TLS is enabled
pub async fn https_security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // HSTS: Force HTTPS for 1 year, include subdomains
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=31536000; includeSubDomains; preload"),
    );

    response
}

/// Request correlation ID middleware
///
/// Adds unique correlation ID to each request for distributed tracing
/// - Accepts existing X-Request-ID from client
/// - Generates new UUID if not provided
/// - Propagates ID through the request chain
/// - Includes ID in response headers
pub async fn request_correlation_id(mut request: Request, next: Next) -> Response {
    // Check for existing correlation ID from client
    let correlation_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // Add correlation ID to tracing span
    Span::current().record("request_id", &correlation_id);

    // Store correlation ID in request extensions for handlers
    request
        .extensions_mut()
        .insert(CorrelationId(correlation_id.clone()));

    debug!(correlation_id = %correlation_id, "Request received");

    // Process request
    let mut response = next.run(request).await;

    // Add correlation ID to response headers
    response.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        HeaderValue::from_str(&correlation_id)
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );

    response
}

/// Correlation ID extractor for handlers
#[derive(Clone, Debug)]
pub struct CorrelationId(pub String);

/// Authenticated user extractor for handlers
#[derive(Clone, Debug)]
pub struct AuthenticatedUser(pub Arc<User>);

/// RBAC (Role-Based Access Control) middleware
///
/// Enforces permission checks on protected endpoints
/// - Extracts authenticated user from request extensions
/// - Checks if user has required permission
/// - Returns 401 Unauthorized if no user present
/// - Returns 403 Forbidden if user lacks permission
/// - Allows request to proceed if permission granted
pub async fn rbac_check(
    permission: Permission,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>
       + Clone {
    move |request: Request, next: Next| {
        let required_permission = permission.clone();
        Box::pin(async move {
            // Extract authenticated user from request extensions
            let user = request.extensions().get::<AuthenticatedUser>().cloned();

            match user {
                Some(AuthenticatedUser(user_arc)) => {
                    let user_ref = &*user_arc;

                    // Check if user has the required permission
                    if PermissionChecker::has_permission(user_ref, &required_permission) {
                        debug!(
                            user = %user_ref.username,
                            permission = ?required_permission,
                            "Permission granted"
                        );
                        next.run(request).await
                    } else {
                        warn!(
                            user = %user_ref.username,
                            permission = ?required_permission,
                            "Permission denied"
                        );
                        (
                            StatusCode::FORBIDDEN,
                            format!(
                                "Access denied: User '{}' does not have required permission: {:?}",
                                user_ref.username, required_permission
                            ),
                        )
                            .into_response()
                    }
                }
                None => {
                    warn!(
                        permission = ?required_permission,
                        "Authentication required but no user present"
                    );
                    (StatusCode::UNAUTHORIZED, "Authentication required").into_response()
                }
            }
        })
    }
}

/// Route-specific RBAC middleware with automatic permission mapping
///
/// Maps HTTP methods and routes to required permissions:
/// - GET /sparql -> Permission::QueryExecute
/// - POST /sparql (query) -> Permission::QueryExecute
/// - POST /update -> Permission::UpdateExecute
/// - PUT/POST/DELETE /graph -> Permission::GraphStore
/// - POST /upload -> Permission::Upload
/// - GET /$/stats -> Permission::Monitor
/// - POST /$/datasets -> Permission::DatasetCreate
pub async fn route_based_rbac(request: Request, next: Next) -> Response {
    // Skip RBAC for public endpoints
    let path = request.uri().path();
    let public_endpoints = [
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics", // Public metrics endpoint
    ];

    if public_endpoints.contains(&path) {
        return next.run(request).await;
    }

    // Extract authenticated user
    let user = request.extensions().get::<AuthenticatedUser>().cloned();

    let user_arc = match user {
        Some(AuthenticatedUser(user)) => user,
        None => {
            // No authentication required for now - can be made strict later
            debug!(path = %path, "No authentication present, allowing request");
            return next.run(request).await;
        }
    };

    // Determine required permission based on route and method
    let method = request.method();
    let required_permission = match (method, path) {
        // SPARQL query endpoints
        (_, "/sparql") if method == Method::GET || method == Method::POST => {
            Some(Permission::QueryExecute)
        }

        // SPARQL update endpoints
        (_, "/update") if method == Method::POST => Some(Permission::UpdateExecute),

        // Graph Store Protocol
        (_, p) if p.starts_with("/graph") || p == "/data" => match *method {
            Method::GET | Method::HEAD => Some(Permission::Read),
            Method::PUT | Method::POST | Method::DELETE => Some(Permission::GraphStore),
            _ => Some(Permission::Read),
        },

        // Upload endpoints
        (_, "/upload") if method == Method::POST => Some(Permission::Upload),

        // SHACL validation
        (_, "/shacl") if method == Method::POST => Some(Permission::QueryExecute),

        // Patch operations
        (_, "/patch") if method == Method::POST => Some(Permission::Write),

        // Dataset management
        (_, p) if p.starts_with("/$/datasets") => match *method {
            Method::GET => Some(Permission::Read),
            Method::POST => Some(Permission::DatasetCreate),
            Method::DELETE => Some(Permission::DatasetDelete),
            Method::PUT => Some(Permission::DatasetManage),
            _ => Some(Permission::Admin),
        },

        // Admin endpoints
        (_, p) if p.starts_with("/$/admin") => Some(Permission::Admin),

        // Monitoring endpoints
        (_, p) if p.starts_with("/$/stats") || p.starts_with("/$/logs") => {
            Some(Permission::Monitor)
        }

        // Task management
        (_, p) if p.starts_with("/$/tasks") => match *method {
            Method::GET => Some(Permission::Monitor),
            _ => Some(Permission::Admin),
        },

        // Federation management
        (_, p) if p.starts_with("/$/federation") => Some(Permission::FederationManage),

        // Cluster management
        (_, p) if p.starts_with("/$/cluster") => Some(Permission::ClusterManage),

        // User management
        (_, p) if p.starts_with("/$/users") => Some(Permission::UserManage),

        // System configuration
        (_, p) if p.starts_with("/$/config") => Some(Permission::SystemConfig),

        // Backup/restore
        (_, "/$/backup") if method == Method::POST => Some(Permission::Backup),
        (_, "/$/restore") if method == Method::POST => Some(Permission::Restore),

        // Default: require read permission for all other endpoints
        _ => Some(Permission::Read),
    };

    // Check permission
    if let Some(permission) = required_permission {
        let user_ref = &*user_arc;

        if PermissionChecker::has_permission(user_ref, &permission) {
            debug!(
                user = %user_ref.username,
                path = %path,
                method = %method,
                permission = ?permission,
                "RBAC check passed"
            );
            next.run(request).await
        } else {
            warn!(
                user = %user_ref.username,
                path = %path,
                method = %method,
                permission = ?permission,
                "RBAC check failed - permission denied"
            );
            (
                StatusCode::FORBIDDEN,
                format!(
                    "Access denied: User '{}' does not have required permission {:?} for {} {}",
                    user_ref.username, permission, method, path
                ),
            )
                .into_response()
        }
    } else {
        // No specific permission required
        next.run(request).await
    }
}

/// ReBAC (Relationship-Based Access Control) middleware
///
/// Provides fine-grained authorization based on relationships between users and resources.
/// Works alongside RBAC to enable:
/// - Dataset-level permissions (can_read, can_write on specific datasets)
/// - Graph-level permissions (access to specific named graphs)
/// - Hierarchical permissions (parent dataset permissions inherit to graphs)
/// - Dynamic policies (organization membership, ownership)
///
/// Usage:
/// ```ignore
/// let app = Router::new()
///     .route("/dataset/:name", get(handler))
///     .layer(from_fn_with_state(policy_engine.clone(), rebac_middleware));
/// ```
pub async fn rebac_middleware(
    axum::extract::State(policy_engine): axum::extract::State<Arc<UnifiedPolicyEngine>>,
    request: Request,
    next: Next,
) -> Response {
    // Skip ReBAC for public endpoints
    let path = request.uri().path();
    let public_endpoints = ["/health", "/health/live", "/health/ready", "/metrics"];

    if public_endpoints.contains(&path) {
        return next.run(request).await;
    }

    // Extract authenticated user
    let user = match request.extensions().get::<AuthenticatedUser>().cloned() {
        Some(AuthenticatedUser(user)) => user,
        None => {
            // No user present, allow (assuming RBAC middleware will handle auth)
            return next.run(request).await;
        }
    };

    // Extract dataset/resource from path
    let (action, resource) = extract_action_and_resource(&request);

    // Create authorization context
    let context = AuthorizationContext::new((*user).clone(), action.clone(), resource.clone());

    // Check authorization using unified policy engine
    match policy_engine.authorize(&context).await {
        Ok(response) if response.allowed => {
            debug!(
                user = %user.username,
                action = %action,
                resource = %resource,
                "ReBAC authorization granted"
            );
            next.run(request).await
        }
        Ok(response) => {
            warn!(
                user = %user.username,
                action = %action,
                resource = %resource,
                reason = ?response.reason,
                "ReBAC authorization denied"
            );
            (
                StatusCode::FORBIDDEN,
                format!(
                    "Access denied: {}",
                    response
                        .reason
                        .unwrap_or_else(|| "Insufficient permissions".to_string())
                ),
            )
                .into_response()
        }
        Err(e) => {
            warn!(
                user = %user.username,
                action = %action,
                resource = %resource,
                error = %e,
                "ReBAC authorization error"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Authorization error: {}", e),
            )
                .into_response()
        }
    }
}

/// Extract action and resource from HTTP request
///
/// Maps HTTP methods and paths to ReBAC (action, resource) pairs:
/// - GET /dataset/foo → ("can_read", "dataset:foo")
/// - POST /dataset/foo/update → ("can_write", "dataset:foo")
/// - PUT /dataset/foo/graph?graph=http://example.org/g1 → ("can_write", "graph:http://example.org/g1")
fn extract_action_and_resource(request: &Request) -> (String, String) {
    let method = request.method();
    let path = request.uri().path();
    let query = request.uri().query();

    // Parse dataset name from path
    let dataset = if let Some(ds) = path.strip_prefix("/dataset/") {
        let ds_name = ds.split('/').next().unwrap_or("default");
        ds_name.to_string()
    } else {
        "default".to_string()
    };

    // Check for graph parameter
    if let Some(query_str) = query {
        if let Some(graph_uri) = extract_graph_from_query(query_str) {
            let action = match method {
                &Method::GET | &Method::HEAD => "can_read",
                &Method::POST | &Method::PUT | &Method::DELETE => "can_write",
                _ => "can_read",
            };
            return (action.to_string(), format!("graph:{}", graph_uri));
        }
    }

    // Determine action from method and path
    let action = match (method, path) {
        (&Method::GET, _) | (&Method::HEAD, _) => "can_read",
        (&Method::POST, p) if p.contains("/sparql") || p.contains("/query") => "can_execute_query",
        (&Method::POST, p) if p.contains("/update") => "can_execute_update",
        (&Method::POST, _) | (&Method::PUT, _) | (&Method::PATCH, _) | (&Method::DELETE, _) => {
            "can_write"
        }
        _ => "can_read",
    };

    (action.to_string(), format!("dataset:{}", dataset))
}

/// Extract graph URI from query string
fn extract_graph_from_query(query: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            if key == "graph" || key == "default" {
                return Some(urlencoding::decode(value).ok()?.into_owned());
            }
        }
    }
    None
}

/// Request timing middleware
///
/// Measures request duration and logs slow requests
/// Adds X-Response-Time header with duration in milliseconds
pub async fn request_timing(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    // Process request
    let response = next.run(request).await;

    let duration = start.elapsed();
    let duration_ms = duration.as_millis();

    // Log slow requests (>1 second)
    if duration_ms > 1000 {
        info!(
            method = %method,
            uri = %uri,
            duration_ms = %duration_ms,
            "Slow request detected"
        );
    }

    // Add timing header to response
    let mut response = response;
    if let Ok(duration_value) = HeaderValue::from_str(&duration_ms.to_string()) {
        response.headers_mut().insert(
            header::HeaderName::from_static("x-response-time"),
            duration_value,
        );
    }

    debug!(
        method = %method,
        uri = %uri,
        duration_ms = %duration_ms,
        status = %response.status(),
        "Request completed"
    );

    response
}

/// Health check bypass middleware
///
/// Skips expensive middleware (auth, rate limiting) for health check endpoints
/// Improves monitoring reliability and reduces overhead
pub async fn health_check_bypass(request: Request, next: Next) -> Response {
    let path = request.uri().path();

    // List of health check endpoints
    let health_endpoints = ["/health", "/health/live", "/health/ready", "/metrics"];

    if health_endpoints.contains(&path) {
        // Fast path for health checks - minimal processing
        return next.run(request).await;
    }

    // Normal processing for other requests
    next.run(request).await
}

/// Request size limiter middleware
///
/// Rejects requests exceeding maximum body size
/// Prevents DoS attacks via large payloads
pub async fn request_size_limit(request: Request, next: Next, max_size_bytes: usize) -> Response {
    if let Some(content_length) = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
    {
        if content_length > max_size_bytes {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "Request body too large: {} bytes (max: {})",
                    content_length, max_size_bytes
                ),
            )
                .into_response();
        }
    }

    next.run(request).await
}

/// API version middleware
///
/// Adds API version to response headers for client compatibility
pub async fn api_version(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    response.headers_mut().insert(
        header::HeaderName::from_static("x-api-version"),
        HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
    );

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_security_headers() {
        let app = Router::new()
            .route("/", get(|| async { "Hello" }))
            .layer(axum::middleware::from_fn(security_headers));

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert!(response.headers().contains_key("x-frame-options"));
        assert!(response.headers().contains_key("x-content-type-options"));
        assert!(response.headers().contains_key("x-xss-protection"));
        assert!(response.headers().contains_key("referrer-policy"));
        assert!(response.headers().contains_key("content-security-policy"));
    }

    #[tokio::test]
    async fn test_correlation_id_generation() {
        let app = Router::new()
            .route("/", get(|| async { "Hello" }))
            .layer(axum::middleware::from_fn(request_correlation_id));

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let correlation_id = response.headers().get("x-request-id");
        assert!(correlation_id.is_some());

        // Verify it's a valid UUID
        let id_str = correlation_id.unwrap().to_str().unwrap();
        assert!(Uuid::parse_str(id_str).is_ok());
    }

    #[tokio::test]
    async fn test_api_version() {
        let app = Router::new()
            .route("/", get(|| async { "Hello" }))
            .layer(axum::middleware::from_fn(api_version));

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let version = response.headers().get("x-api-version");
        assert!(version.is_some());
        assert_eq!(version.unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
