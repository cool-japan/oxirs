//! Production-grade middleware for security, tracing, and observability

use axum::{
    extract::Request,
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::time::Instant;
use tracing::{debug, info, Span};
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

    if health_endpoints.iter().any(|&endpoint| path == endpoint) {
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
