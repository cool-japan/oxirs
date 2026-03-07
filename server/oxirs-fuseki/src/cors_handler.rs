//! CORS (Cross-Origin Resource Sharing) header handling for SPARQL endpoints.
//!
//! Implements the W3C CORS specification relevant to SPARQL HTTP endpoints,
//! including preflight (`OPTIONS`) detection, origin validation, and header
//! construction.

/// Configuration for CORS policy.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins.  Use `["*"]` to allow any origin.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers.
    pub allowed_headers: Vec<String>,
    /// Headers exposed to the browser.
    pub expose_headers: Vec<String>,
    /// Max-Age in seconds for preflight caching.
    pub max_age_secs: u64,
    /// Whether to allow credentials (`cookies`, `Authorization`).
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Accept".to_string(),
                "Authorization".to_string(),
            ],
            expose_headers: Vec::new(),
            max_age_secs: 86400,
            allow_credentials: false,
        }
    }
}

/// The outcome of a CORS check.
#[derive(Debug, Clone)]
pub enum CorsDecision {
    /// The request is allowed; attach these headers to the response.
    Allow(CorsHeaders),
    /// The request is denied with a reason.
    Deny(String),
}

/// Collection of CORS response header name–value pairs.
#[derive(Debug, Clone, Default)]
pub struct CorsHeaders {
    /// Ordered list of `(header-name, value)` pairs to inject into the response.
    pub headers: Vec<(String, String)>,
}

impl CorsHeaders {
    /// Look up the value of a header by name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// CORS handler that validates requests and generates appropriate headers.
pub struct CorsHandler {
    config: CorsConfig,
}

impl CorsHandler {
    /// Create a handler from explicit configuration.
    pub fn new(config: CorsConfig) -> Self {
        Self { config }
    }

    /// Create a permissive handler that allows all origins and the common
    /// SPARQL-relevant HTTP methods.
    pub fn permissive() -> Self {
        Self {
            config: CorsConfig {
                allowed_origins: vec!["*".to_string()],
                allowed_methods: vec![
                    "GET".to_string(),
                    "POST".to_string(),
                    "PUT".to_string(),
                    "DELETE".to_string(),
                    "OPTIONS".to_string(),
                    "HEAD".to_string(),
                ],
                allowed_headers: vec![
                    "Content-Type".to_string(),
                    "Accept".to_string(),
                    "Authorization".to_string(),
                    "X-Requested-With".to_string(),
                ],
                expose_headers: vec![
                    "Content-Length".to_string(),
                    "X-Request-Id".to_string(),
                ],
                max_age_secs: 86400,
                allow_credentials: false,
            },
        }
    }

    /// Check a preflight (`OPTIONS`) request.
    ///
    /// * `origin` — the `Origin` request header value.
    /// * `method` — the `Access-Control-Request-Method` header value.
    /// * `req_headers` — the `Access-Control-Request-Headers` values.
    pub fn check_preflight(
        &self,
        origin: &str,
        method: &str,
        req_headers: &[String],
    ) -> CorsDecision {
        if !self.is_origin_allowed(origin) {
            return CorsDecision::Deny(format!("Origin '{}' is not allowed", origin));
        }
        if !self.is_method_allowed(method) {
            return CorsDecision::Deny(format!("Method '{}' is not allowed", method));
        }
        // Validate requested headers
        for hdr in req_headers {
            if !self
                .config
                .allowed_headers
                .iter()
                .any(|ah| ah.eq_ignore_ascii_case(hdr))
            {
                return CorsDecision::Deny(format!("Header '{}' is not allowed", hdr));
            }
        }

        let mut hdrs = self.build_headers(origin);
        hdrs.headers.push((
            "Access-Control-Allow-Methods".to_string(),
            self.config.allowed_methods.join(", "),
        ));
        hdrs.headers.push((
            "Access-Control-Allow-Headers".to_string(),
            self.config.allowed_headers.join(", "),
        ));
        hdrs.headers.push((
            "Access-Control-Max-Age".to_string(),
            self.config.max_age_secs.to_string(),
        ));

        CorsDecision::Allow(hdrs)
    }

    /// Check a simple (non-preflight) request from `origin`.
    pub fn check_simple(&self, origin: &str) -> CorsDecision {
        if !self.is_origin_allowed(origin) {
            return CorsDecision::Deny(format!("Origin '{}' is not allowed", origin));
        }
        CorsDecision::Allow(self.build_headers(origin))
    }

    /// Return `true` if the given origin is permitted by this configuration.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.config.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Return `true` if the given HTTP method is permitted.
    pub fn is_method_allowed(&self, method: &str) -> bool {
        self.config
            .allowed_methods
            .iter()
            .any(|m| m.eq_ignore_ascii_case(method))
    }

    /// Build the base CORS response headers for an allowed origin.
    pub fn build_headers(&self, origin: &str) -> CorsHeaders {
        let origin_value = if self.config.allowed_origins.iter().any(|o| o == "*")
            && !self.config.allow_credentials
        {
            "*".to_string()
        } else {
            origin.to_string()
        };

        let mut headers = vec![
            ("Access-Control-Allow-Origin".to_string(), origin_value),
        ];

        if self.config.allow_credentials {
            headers.push((
                "Access-Control-Allow-Credentials".to_string(),
                "true".to_string(),
            ));
        }

        if !self.config.expose_headers.is_empty() {
            headers.push((
                "Access-Control-Expose-Headers".to_string(),
                self.config.expose_headers.join(", "),
            ));
        }

        CorsHeaders { headers }
    }

    /// Return `true` if the given method and header set constitute an HTTP
    /// preflight request (method = OPTIONS **and** `Access-Control-Request-Method`
    /// is present in headers).
    pub fn is_preflight(method: &str, headers: &[String]) -> bool {
        method.eq_ignore_ascii_case("OPTIONS")
            && headers
                .iter()
                .any(|h| h.eq_ignore_ascii_case("Access-Control-Request-Method"))
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &CorsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed_origins(origins: &[&str]) -> CorsConfig {
        CorsConfig {
            allowed_origins: origins.iter().map(|s| s.to_string()).collect(),
            ..CorsConfig::default()
        }
    }

    // --- CorsHandler::permissive ---
    #[test]
    fn test_permissive_allows_any_origin() {
        let h = CorsHandler::permissive();
        assert!(h.is_origin_allowed("https://example.com"));
        assert!(h.is_origin_allowed("https://evil.example.com"));
    }

    #[test]
    fn test_permissive_allows_get_post_options() {
        let h = CorsHandler::permissive();
        assert!(h.is_method_allowed("GET"));
        assert!(h.is_method_allowed("POST"));
        assert!(h.is_method_allowed("OPTIONS"));
    }

    // --- is_origin_allowed ---
    #[test]
    fn test_wildcard_origin_allowed() {
        let h = CorsHandler::new(CorsConfig::default());
        assert!(h.is_origin_allowed("https://any.example.com"));
    }

    #[test]
    fn test_specific_origin_allowed() {
        let h = CorsHandler::new(allowed_origins(&["https://app.example.com"]));
        assert!(h.is_origin_allowed("https://app.example.com"));
    }

    #[test]
    fn test_specific_origin_denied() {
        let h = CorsHandler::new(allowed_origins(&["https://app.example.com"]));
        assert!(!h.is_origin_allowed("https://evil.example.com"));
    }

    #[test]
    fn test_multiple_allowed_origins() {
        let h = CorsHandler::new(allowed_origins(&[
            "https://a.example.com",
            "https://b.example.com",
        ]));
        assert!(h.is_origin_allowed("https://a.example.com"));
        assert!(h.is_origin_allowed("https://b.example.com"));
        assert!(!h.is_origin_allowed("https://c.example.com"));
    }

    // --- is_method_allowed ---
    #[test]
    fn test_method_allowed_case_insensitive() {
        let h = CorsHandler::new(CorsConfig::default());
        assert!(h.is_method_allowed("get"));
        assert!(h.is_method_allowed("POST"));
        assert!(h.is_method_allowed("options"));
    }

    #[test]
    fn test_method_not_allowed() {
        let h = CorsHandler::new(CorsConfig {
            allowed_methods: vec!["GET".to_string()],
            ..CorsConfig::default()
        });
        assert!(!h.is_method_allowed("DELETE"));
    }

    // --- build_headers ---
    #[test]
    fn test_build_headers_wildcard() {
        let h = CorsHandler::new(CorsConfig::default());
        let hdrs = h.build_headers("https://example.com");
        let acao = hdrs.get("Access-Control-Allow-Origin");
        assert_eq!(acao, Some("*"));
    }

    #[test]
    fn test_build_headers_specific_origin() {
        let h = CorsHandler::new(CorsConfig {
            allowed_origins: vec!["https://app.example.com".to_string()],
            allow_credentials: true,
            ..CorsConfig::default()
        });
        let hdrs = h.build_headers("https://app.example.com");
        assert_eq!(
            hdrs.get("Access-Control-Allow-Origin"),
            Some("https://app.example.com")
        );
        assert_eq!(
            hdrs.get("Access-Control-Allow-Credentials"),
            Some("true")
        );
    }

    #[test]
    fn test_build_headers_expose() {
        let h = CorsHandler::new(CorsConfig {
            expose_headers: vec!["X-Custom-Header".to_string()],
            ..CorsConfig::default()
        });
        let hdrs = h.build_headers("https://example.com");
        assert!(hdrs.get("Access-Control-Expose-Headers").is_some());
    }

    // --- check_simple ---
    #[test]
    fn test_check_simple_allowed() {
        let h = CorsHandler::new(CorsConfig::default());
        let decision = h.check_simple("https://example.com");
        assert!(matches!(decision, CorsDecision::Allow(_)));
    }

    #[test]
    fn test_check_simple_denied() {
        let h = CorsHandler::new(allowed_origins(&["https://good.example.com"]));
        let decision = h.check_simple("https://evil.example.com");
        assert!(matches!(decision, CorsDecision::Deny(_)));
    }

    #[test]
    fn test_check_simple_deny_message() {
        let h = CorsHandler::new(allowed_origins(&["https://good.example.com"]));
        if let CorsDecision::Deny(msg) = h.check_simple("https://bad.com") {
            assert!(msg.contains("bad.com"));
        } else {
            panic!("Expected Deny");
        }
    }

    // --- check_preflight ---
    #[test]
    fn test_check_preflight_allowed() {
        let h = CorsHandler::new(CorsConfig::default());
        let decision = h.check_preflight(
            "https://example.com",
            "POST",
            &["Content-Type".to_string()],
        );
        assert!(matches!(decision, CorsDecision::Allow(_)));
    }

    #[test]
    fn test_check_preflight_denied_origin() {
        let h = CorsHandler::new(allowed_origins(&["https://good.example.com"]));
        let decision = h.check_preflight("https://bad.com", "GET", &[]);
        assert!(matches!(decision, CorsDecision::Deny(_)));
    }

    #[test]
    fn test_check_preflight_denied_method() {
        let h = CorsHandler::new(CorsConfig {
            allowed_methods: vec!["GET".to_string()],
            ..CorsConfig::default()
        });
        let decision = h.check_preflight("https://example.com", "DELETE", &[]);
        assert!(matches!(decision, CorsDecision::Deny(_)));
    }

    #[test]
    fn test_check_preflight_denied_header() {
        let h = CorsHandler::new(CorsConfig {
            allowed_headers: vec!["Content-Type".to_string()],
            ..CorsConfig::default()
        });
        let decision = h.check_preflight(
            "https://example.com",
            "POST",
            &["X-Custom-Header".to_string()],
        );
        assert!(matches!(decision, CorsDecision::Deny(_)));
    }

    #[test]
    fn test_check_preflight_has_max_age() {
        let h = CorsHandler::new(CorsConfig {
            max_age_secs: 3600,
            ..CorsConfig::default()
        });
        if let CorsDecision::Allow(hdrs) =
            h.check_preflight("https://example.com", "GET", &[])
        {
            assert_eq!(hdrs.get("Access-Control-Max-Age"), Some("3600"));
        } else {
            panic!("Expected Allow");
        }
    }

    #[test]
    fn test_check_preflight_has_allow_methods() {
        let h = CorsHandler::new(CorsConfig::default());
        if let CorsDecision::Allow(hdrs) =
            h.check_preflight("https://example.com", "GET", &[])
        {
            assert!(hdrs.get("Access-Control-Allow-Methods").is_some());
        } else {
            panic!("Expected Allow");
        }
    }

    #[test]
    fn test_check_preflight_has_allow_headers() {
        let h = CorsHandler::new(CorsConfig::default());
        if let CorsDecision::Allow(hdrs) =
            h.check_preflight("https://example.com", "GET", &[])
        {
            assert!(hdrs.get("Access-Control-Allow-Headers").is_some());
        } else {
            panic!("Expected Allow");
        }
    }

    // --- is_preflight ---
    #[test]
    fn test_is_preflight_true() {
        let headers = vec!["Access-Control-Request-Method".to_string()];
        assert!(CorsHandler::is_preflight("OPTIONS", &headers));
    }

    #[test]
    fn test_is_preflight_false_wrong_method() {
        let headers = vec!["Access-Control-Request-Method".to_string()];
        assert!(!CorsHandler::is_preflight("GET", &headers));
    }

    #[test]
    fn test_is_preflight_false_missing_header() {
        assert!(!CorsHandler::is_preflight("OPTIONS", &[]));
    }

    #[test]
    fn test_is_preflight_case_insensitive() {
        let headers = vec!["access-control-request-method".to_string()];
        assert!(CorsHandler::is_preflight("options", &headers));
    }

    // --- credentials ---
    #[test]
    fn test_credentials_not_set_by_default() {
        let h = CorsHandler::new(CorsConfig::default());
        let hdrs = h.build_headers("https://example.com");
        assert!(hdrs.get("Access-Control-Allow-Credentials").is_none());
    }

    #[test]
    fn test_credentials_set_when_enabled() {
        let h = CorsHandler::new(CorsConfig {
            allow_credentials: true,
            allowed_origins: vec!["https://example.com".to_string()],
            ..CorsConfig::default()
        });
        let hdrs = h.build_headers("https://example.com");
        assert_eq!(hdrs.get("Access-Control-Allow-Credentials"), Some("true"));
    }

    // --- CorsHeaders::get ---
    #[test]
    fn test_cors_headers_get_case_insensitive() {
        let hdrs = CorsHeaders {
            headers: vec![("Access-Control-Allow-Origin".to_string(), "*".to_string())],
        };
        assert_eq!(hdrs.get("access-control-allow-origin"), Some("*"));
        assert_eq!(hdrs.get("ACCESS-CONTROL-ALLOW-ORIGIN"), Some("*"));
    }

    #[test]
    fn test_cors_headers_get_missing() {
        let hdrs = CorsHeaders::default();
        assert!(hdrs.get("X-Missing").is_none());
    }

    // --- config accessor ---
    #[test]
    fn test_config_accessor() {
        let cfg = CorsConfig::default();
        let h = CorsHandler::new(cfg.clone());
        assert_eq!(h.config().max_age_secs, cfg.max_age_secs);
    }

    // --- round-trip: simple allowed origin echoed back ---
    #[test]
    fn test_simple_echoes_specific_origin() {
        let h = CorsHandler::new(CorsConfig {
            allowed_origins: vec!["https://client.example.com".to_string()],
            allow_credentials: true,
            ..CorsConfig::default()
        });
        if let CorsDecision::Allow(hdrs) = h.check_simple("https://client.example.com") {
            assert_eq!(
                hdrs.get("Access-Control-Allow-Origin"),
                Some("https://client.example.com")
            );
        } else {
            panic!("Expected Allow");
        }
    }
}
