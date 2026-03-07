//! HTTP endpoint routing for SPARQL/GraphQL/REST paths.
//!
//! Provides path-pattern matching with `{param}` placeholders, method-based
//! dispatch, and structured error reporting for duplicate / invalid routes.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// HTTP request methods.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

impl HttpMethod {
    /// Parse a method string (case-insensitive).
    pub fn parse_method(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Self::GET),
            "POST" => Some(Self::POST),
            "PUT" => Some(Self::PUT),
            "DELETE" => Some(Self::DELETE),
            "PATCH" => Some(Self::PATCH),
            "HEAD" => Some(Self::HEAD),
            "OPTIONS" => Some(Self::OPTIONS),
            _ => None,
        }
    }

    /// Canonical uppercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GET => "GET",
            Self::POST => "POST",
            Self::PUT => "PUT",
            Self::DELETE => "DELETE",
            Self::PATCH => "PATCH",
            Self::HEAD => "HEAD",
            Self::OPTIONS => "OPTIONS",
        }
    }
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single named path parameter extracted from the URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteParam {
    pub name: String,
    pub value: String,
}

impl RouteParam {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A registered route descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub method: HttpMethod,
    /// Path pattern, e.g. `/sparql/{dataset}/query`.
    pub path_pattern: String,
    /// Logical name of the handler function or module.
    pub handler_name: String,
    pub requires_auth: bool,
}

impl Route {
    pub fn new(
        method: HttpMethod,
        path_pattern: impl Into<String>,
        handler_name: impl Into<String>,
    ) -> Self {
        Self {
            method,
            path_pattern: path_pattern.into(),
            handler_name: handler_name.into(),
            requires_auth: false,
        }
    }

    pub fn with_auth(mut self) -> Self {
        self.requires_auth = true;
        self
    }
}

/// The result of a successful route lookup.
#[derive(Debug, Clone)]
pub struct RouteMatch {
    pub route: Route,
    pub params: Vec<RouteParam>,
    pub query_string: Option<String>,
}

/// Errors produced by the router.
#[derive(Debug, Clone, PartialEq)]
pub enum RouterError {
    /// No route matched the given method + path.
    NotFound(String),
    /// A route exists for the path but not for this method.
    MethodNotAllowed {
        path: String,
        method: HttpMethod,
        allowed: Vec<HttpMethod>,
    },
    /// A route with the same method+pattern is already registered.
    DuplicateRoute(String),
    /// The path pattern contains invalid syntax.
    InvalidPattern(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(p) => write!(f, "No route found for '{p}'"),
            Self::MethodNotAllowed { path, method, .. } => {
                write!(f, "Method {method} not allowed for '{path}'")
            }
            Self::DuplicateRoute(k) => write!(f, "Duplicate route: '{k}'"),
            Self::InvalidPattern(p) => write!(f, "Invalid path pattern: '{p}'"),
        }
    }
}

impl std::error::Error for RouterError {}

// ────────────────────────────────────────────────────────────────────────────
// Pattern helpers
// ────────────────────────────────────────────────────────────────────────────

/// Validate that a path pattern is syntactically correct.
///
/// Rules:
/// - Each `{…}` placeholder must be non-empty and contain only word chars.
/// - Placeholders must not be nested.
fn validate_pattern(pattern: &str) -> Result<(), RouterError> {
    let mut depth: usize = 0;
    for ch in pattern.chars() {
        match ch {
            '{' => {
                depth += 1;
                if depth > 1 {
                    return Err(RouterError::InvalidPattern(pattern.to_string()));
                }
            }
            '}' => {
                if depth == 0 {
                    return Err(RouterError::InvalidPattern(pattern.to_string()));
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(RouterError::InvalidPattern(pattern.to_string()));
    }

    // Check each placeholder is non-empty and word-chars only
    let mut s = pattern;
    while let Some(open) = s.find('{') {
        s = &s[open + 1..];
        if let Some(close) = s.find('}') {
            let name = &s[..close];
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            {
                return Err(RouterError::InvalidPattern(pattern.to_string()));
            }
            s = &s[close + 1..];
        } else {
            return Err(RouterError::InvalidPattern(pattern.to_string()));
        }
    }

    Ok(())
}

/// A compiled path segment for efficient matching.
#[derive(Debug, Clone)]
enum Segment {
    Literal(String),
    Param(String),
}

/// Compile a pattern string into segments.
fn compile_pattern(pattern: &str) -> Vec<Segment> {
    pattern
        .split('/')
        .map(|seg| {
            if seg.starts_with('{') && seg.ends_with('}') {
                Segment::Param(seg[1..seg.len() - 1].to_string())
            } else {
                Segment::Literal(seg.to_string())
            }
        })
        .collect()
}

/// Try to match `path` against a compiled pattern.
///
/// Returns `Some(params)` on success, `None` otherwise.
fn match_segments(segments: &[Segment], path: &str) -> Option<Vec<RouteParam>> {
    let path_parts: Vec<&str> = path.split('/').collect();
    if path_parts.len() != segments.len() {
        return None;
    }

    let mut params = Vec::new();
    for (seg, part) in segments.iter().zip(path_parts.iter()) {
        match seg {
            Segment::Literal(lit) => {
                if lit != part {
                    return None;
                }
            }
            Segment::Param(name) => {
                params.push(RouteParam::new(name.clone(), *part));
            }
        }
    }
    Some(params)
}

// ────────────────────────────────────────────────────────────────────────────
// EndpointRouter
// ────────────────────────────────────────────────────────────────────────────

/// Internal entry stored per registered route.
#[derive(Debug, Clone)]
struct RouteEntry {
    route: Route,
    segments: Vec<Segment>,
}

/// HTTP endpoint router that supports `{param}` path placeholders.
#[derive(Debug, Default)]
pub struct EndpointRouter {
    /// Routes indexed by `"{METHOD} {pattern}"` for duplicate detection.
    route_keys: HashMap<String, usize>,
    entries: Vec<RouteEntry>,
}

impl EndpointRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a route.
    ///
    /// Returns `Err(DuplicateRoute)` if an identical method+pattern is already
    /// registered, or `Err(InvalidPattern)` if the pattern is malformed.
    pub fn register(&mut self, route: Route) -> Result<(), RouterError> {
        validate_pattern(&route.path_pattern)?;

        let key = format!("{} {}", route.method, route.path_pattern);
        if self.route_keys.contains_key(&key) {
            return Err(RouterError::DuplicateRoute(key));
        }

        let segments = compile_pattern(&route.path_pattern);
        let idx = self.entries.len();
        self.route_keys.insert(key, idx);
        self.entries.push(RouteEntry { route, segments });
        Ok(())
    }

    /// Resolve a method + path to the matching route.
    ///
    /// Splits query string from `path` before matching.
    pub fn route(&self, method: &HttpMethod, path: &str) -> Result<RouteMatch, RouterError> {
        // Split query string
        let (path_only, query_string) = match path.find('?') {
            Some(idx) => {
                let qs = path[idx + 1..].to_string();
                (&path[..idx], Some(qs))
            }
            None => (path, None),
        };

        let mut any_path_match = false;

        for entry in &self.entries {
            if let Some(params) = match_segments(&entry.segments, path_only) {
                any_path_match = true;
                if &entry.route.method == method {
                    return Ok(RouteMatch {
                        route: entry.route.clone(),
                        params,
                        query_string,
                    });
                }
            }
        }

        if any_path_match {
            let allowed = self.methods_for_path(path_only);
            return Err(RouterError::MethodNotAllowed {
                path: path_only.to_string(),
                method: method.clone(),
                allowed,
            });
        }

        Err(RouterError::NotFound(path_only.to_string()))
    }

    /// Number of registered routes.
    pub fn registered_count(&self) -> usize {
        self.entries.len()
    }

    /// All HTTP methods registered for a given concrete path.
    pub fn methods_for_path(&self, path: &str) -> Vec<HttpMethod> {
        self.entries
            .iter()
            .filter_map(|e| {
                if match_segments(&e.segments, path).is_some() {
                    Some(e.route.method.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return references to all registered routes.
    pub fn list_routes(&self) -> Vec<&Route> {
        self.entries.iter().map(|e| &e.route).collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn sparql_router() -> EndpointRouter {
        let mut r = EndpointRouter::new();
        r.register(Route::new(
            HttpMethod::GET,
            "/sparql/{dataset}/query",
            "sparql_query",
        ))
        .expect("register sparql query GET");
        r.register(Route::new(
            HttpMethod::POST,
            "/sparql/{dataset}/query",
            "sparql_query_post",
        ))
        .expect("register sparql query POST");
        r.register(Route::new(
            HttpMethod::GET,
            "/sparql/{dataset}/update",
            "sparql_update",
        ))
        .expect("register sparql update GET");
        r.register(Route::new(
            HttpMethod::POST,
            "/sparql/{dataset}/update",
            "sparql_update_post",
        ))
        .expect("register sparql update POST");
        r.register(Route::new(HttpMethod::GET, "/health", "health_check"))
            .expect("register health");
        r.register(Route::new(HttpMethod::POST, "/graphql", "graphql_handler").with_auth())
            .expect("register graphql");
        r
    }

    // ── HttpMethod ────────────────────────────────────────────────────────────

    #[test]
    fn test_http_method_from_str_get() {
        assert_eq!(HttpMethod::parse_method("GET"), Some(HttpMethod::GET));
    }

    #[test]
    fn test_http_method_from_str_lowercase() {
        assert_eq!(HttpMethod::parse_method("post"), Some(HttpMethod::POST));
    }

    #[test]
    fn test_http_method_from_str_unknown() {
        assert_eq!(HttpMethod::parse_method("CONNECT"), None);
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::DELETE.to_string(), "DELETE");
        assert_eq!(HttpMethod::OPTIONS.to_string(), "OPTIONS");
    }

    // ── RouteParam ────────────────────────────────────────────────────────────

    #[test]
    fn test_route_param_new() {
        let p = RouteParam::new("dataset", "mydb");
        assert_eq!(p.name, "dataset");
        assert_eq!(p.value, "mydb");
    }

    // ── Route ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_route_new_no_auth() {
        let r = Route::new(HttpMethod::GET, "/health", "health");
        assert!(!r.requires_auth);
    }

    #[test]
    fn test_route_with_auth() {
        let r = Route::new(HttpMethod::POST, "/admin", "admin").with_auth();
        assert!(r.requires_auth);
    }

    // ── EndpointRouter – registration ────────────────────────────────────────

    #[test]
    fn test_register_increases_count() {
        let router = sparql_router();
        assert_eq!(router.registered_count(), 6);
    }

    #[test]
    fn test_register_duplicate_returns_error() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::GET, "/health", "h1"))
            .expect("first");
        let err = router
            .register(Route::new(HttpMethod::GET, "/health", "h2"))
            .expect_err("duplicate");
        assert!(matches!(err, RouterError::DuplicateRoute(_)));
    }

    #[test]
    fn test_register_invalid_pattern_unclosed_brace() {
        let mut router = EndpointRouter::new();
        let err = router
            .register(Route::new(HttpMethod::GET, "/foo/{bar", "h"))
            .expect_err("invalid");
        assert!(matches!(err, RouterError::InvalidPattern(_)));
    }

    #[test]
    fn test_register_invalid_pattern_empty_placeholder() {
        let mut router = EndpointRouter::new();
        let err = router
            .register(Route::new(HttpMethod::GET, "/foo/{}/bar", "h"))
            .expect_err("invalid");
        assert!(matches!(err, RouterError::InvalidPattern(_)));
    }

    #[test]
    fn test_register_different_methods_same_pattern_ok() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::GET, "/data/{id}", "get_data"))
            .expect("GET");
        router
            .register(Route::new(HttpMethod::DELETE, "/data/{id}", "del_data"))
            .expect("DELETE");
        assert_eq!(router.registered_count(), 2);
    }

    #[test]
    fn test_list_routes() {
        let router = sparql_router();
        let routes = router.list_routes();
        assert_eq!(routes.len(), 6);
    }

    // ── EndpointRouter – routing ──────────────────────────────────────────────

    #[test]
    fn test_route_exact_path() {
        let router = sparql_router();
        let m = router.route(&HttpMethod::GET, "/health").expect("health");
        assert_eq!(m.route.handler_name, "health_check");
        assert!(m.params.is_empty());
    }

    #[test]
    fn test_route_with_param() {
        let router = sparql_router();
        let m = router
            .route(&HttpMethod::GET, "/sparql/mydb/query")
            .expect("sparql query");
        assert_eq!(m.route.handler_name, "sparql_query");
        assert_eq!(m.params.len(), 1);
        assert_eq!(m.params[0].name, "dataset");
        assert_eq!(m.params[0].value, "mydb");
    }

    #[test]
    fn test_route_post_with_param() {
        let router = sparql_router();
        let m = router
            .route(&HttpMethod::POST, "/sparql/testds/query")
            .expect("post query");
        assert_eq!(m.route.handler_name, "sparql_query_post");
        assert_eq!(m.params[0].value, "testds");
    }

    #[test]
    fn test_route_not_found() {
        let router = sparql_router();
        let err = router
            .route(&HttpMethod::GET, "/unknown/path")
            .expect_err("not found");
        assert!(matches!(err, RouterError::NotFound(_)));
    }

    #[test]
    fn test_route_method_not_allowed() {
        let router = sparql_router();
        // PUT /sparql/x/query is not registered
        let err = router
            .route(&HttpMethod::PUT, "/sparql/mydb/query")
            .expect_err("not allowed");
        match err {
            RouterError::MethodNotAllowed { allowed, .. } => {
                assert!(allowed.contains(&HttpMethod::GET));
                assert!(allowed.contains(&HttpMethod::POST));
            }
            other => panic!("Unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_route_query_string_stripped() {
        let router = sparql_router();
        let m = router
            .route(&HttpMethod::GET, "/health?verbose=true")
            .expect("health with qs");
        assert_eq!(m.route.handler_name, "health_check");
        assert_eq!(m.query_string, Some("verbose=true".to_string()));
    }

    #[test]
    fn test_route_no_query_string() {
        let router = sparql_router();
        let m = router.route(&HttpMethod::GET, "/health").expect("no qs");
        assert!(m.query_string.is_none());
    }

    #[test]
    fn test_route_auth_required() {
        let router = sparql_router();
        let m = router
            .route(&HttpMethod::POST, "/graphql")
            .expect("graphql");
        assert!(m.route.requires_auth);
    }

    #[test]
    fn test_route_no_auth_required() {
        let router = sparql_router();
        let m = router.route(&HttpMethod::GET, "/health").expect("health");
        assert!(!m.route.requires_auth);
    }

    // ── methods_for_path ──────────────────────────────────────────────────────

    #[test]
    fn test_methods_for_path_both() {
        let router = sparql_router();
        let methods = router.methods_for_path("/sparql/mydb/query");
        assert!(methods.contains(&HttpMethod::GET));
        assert!(methods.contains(&HttpMethod::POST));
        assert_eq!(methods.len(), 2);
    }

    #[test]
    fn test_methods_for_path_single() {
        let router = sparql_router();
        let methods = router.methods_for_path("/health");
        assert_eq!(methods, vec![HttpMethod::GET]);
    }

    #[test]
    fn test_methods_for_path_none() {
        let router = sparql_router();
        let methods = router.methods_for_path("/no/such/path");
        assert!(methods.is_empty());
    }

    // ── multi-segment patterns ────────────────────────────────────────────────

    #[test]
    fn test_register_three_segment_pattern() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(
                HttpMethod::GET,
                "/api/v1/{resource}/{id}",
                "resource_handler",
            ))
            .expect("register");
        let m = router
            .route(&HttpMethod::GET, "/api/v1/users/42")
            .expect("match");
        assert_eq!(m.params.len(), 2);
        let resource = m
            .params
            .iter()
            .find(|p| p.name == "resource")
            .expect("resource");
        let id = m.params.iter().find(|p| p.name == "id").expect("id");
        assert_eq!(resource.value, "users");
        assert_eq!(id.value, "42");
    }

    #[test]
    fn test_static_path_takes_precedence_when_registered_first() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::GET, "/data/special", "special"))
            .expect("special");
        router
            .register(Route::new(HttpMethod::GET, "/data/{id}", "generic"))
            .expect("generic");
        // Both match the path /data/special; the first registered wins
        let m = router
            .route(&HttpMethod::GET, "/data/special")
            .expect("match");
        assert_eq!(m.route.handler_name, "special");
    }

    #[test]
    fn test_param_captures_hyphenated_value() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::GET, "/dataset/{ds}", "ds_handler"))
            .expect("register");
        let m = router
            .route(&HttpMethod::GET, "/dataset/my-dataset")
            .expect("match");
        assert_eq!(m.params[0].value, "my-dataset");
    }

    #[test]
    fn test_empty_router_not_found() {
        let router = EndpointRouter::new();
        let err = router
            .route(&HttpMethod::GET, "/anything")
            .expect_err("not found");
        assert!(matches!(err, RouterError::NotFound(_)));
    }

    #[test]
    fn test_registered_count_zero_initially() {
        let router = EndpointRouter::new();
        assert_eq!(router.registered_count(), 0);
    }

    #[test]
    fn test_route_puts_and_deletes_not_confused() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::PUT, "/items/{id}", "put_item"))
            .expect("PUT");
        router
            .register(Route::new(HttpMethod::DELETE, "/items/{id}", "del_item"))
            .expect("DELETE");
        let m_put = router
            .route(&HttpMethod::PUT, "/items/99")
            .expect("PUT match");
        let m_del = router
            .route(&HttpMethod::DELETE, "/items/99")
            .expect("DELETE match");
        assert_eq!(m_put.route.handler_name, "put_item");
        assert_eq!(m_del.route.handler_name, "del_item");
    }

    #[test]
    fn test_route_patch_head_options() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(HttpMethod::PATCH, "/resource", "patch_h"))
            .expect("PATCH");
        router
            .register(Route::new(HttpMethod::HEAD, "/resource", "head_h"))
            .expect("HEAD");
        router
            .register(Route::new(HttpMethod::OPTIONS, "/resource", "opts_h"))
            .expect("OPTIONS");
        assert_eq!(
            router
                .route(&HttpMethod::PATCH, "/resource")
                .expect("patch")
                .route
                .handler_name,
            "patch_h"
        );
        assert_eq!(
            router
                .route(&HttpMethod::HEAD, "/resource")
                .expect("head")
                .route
                .handler_name,
            "head_h"
        );
        assert_eq!(
            router
                .route(&HttpMethod::OPTIONS, "/resource")
                .expect("opts")
                .route
                .handler_name,
            "opts_h"
        );
    }

    #[test]
    fn test_router_error_display_not_found() {
        let e = RouterError::NotFound("/foo".to_string());
        assert!(e.to_string().contains("/foo"));
    }

    #[test]
    fn test_router_error_display_method_not_allowed() {
        let e = RouterError::MethodNotAllowed {
            path: "/foo".to_string(),
            method: HttpMethod::DELETE,
            allowed: vec![HttpMethod::GET],
        };
        assert!(e.to_string().contains("DELETE"));
    }

    #[test]
    fn test_router_error_display_duplicate() {
        let e = RouterError::DuplicateRoute("GET /foo".to_string());
        assert!(e.to_string().contains("GET /foo"));
    }

    #[test]
    fn test_router_error_display_invalid_pattern() {
        let e = RouterError::InvalidPattern("/foo/{".to_string());
        assert!(e.to_string().contains("/foo/{"));
    }

    #[test]
    fn test_route_updates_path_with_multiple_params() {
        let mut router = EndpointRouter::new();
        router
            .register(Route::new(
                HttpMethod::GET,
                "/store/{store_id}/items/{item_id}",
                "store_item",
            ))
            .expect("register");
        let m = router
            .route(&HttpMethod::GET, "/store/shop1/items/widget42")
            .expect("match");
        let store = m
            .params
            .iter()
            .find(|p| p.name == "store_id")
            .expect("store");
        let item = m.params.iter().find(|p| p.name == "item_id").expect("item");
        assert_eq!(store.value, "shop1");
        assert_eq!(item.value, "widget42");
    }

    #[test]
    fn test_query_string_with_params() {
        let router = sparql_router();
        let m = router
            .route(
                &HttpMethod::GET,
                "/sparql/mydb/query?query=SELECT+*+WHERE{}",
            )
            .expect("match with qs");
        assert_eq!(m.params[0].value, "mydb");
        assert_eq!(m.query_string, Some("query=SELECT+*+WHERE{}".to_string()));
    }

    #[test]
    fn test_methods_for_path_with_query_string_not_counted() {
        let router = sparql_router();
        // methods_for_path should not be sensitive to the method count being stale
        let methods = router.methods_for_path("/sparql/anydb/update");
        assert!(methods.contains(&HttpMethod::GET));
        assert!(methods.contains(&HttpMethod::POST));
    }
}
