//! HTTP request validation for SPARQL endpoints.
//!
//! Performs pre-execution validation of incoming HTTP requests before they
//! reach the SPARQL query engine:
//!
//! - SPARQL query syntax pre-check (balanced braces, query type present)
//! - Graph URI well-formedness
//! - Request size limits (body size and query length)
//! - Malformed parameter detection
//! - HTTP method validation (GET and POST only)
//! - Content-Type validation for POST
//! - Basic query injection pattern detection
//!
//! Validation failures produce structured [`ValidationError`] values that
//! can be mapped directly to HTTP 400 responses.

use std::collections::{HashMap, HashSet};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the request validator.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Maximum allowed request body size in bytes.
    pub max_body_bytes: usize,
    /// Maximum allowed SPARQL query string length in bytes.
    pub max_query_length: usize,
    /// Whether to run the injection pattern check.
    pub enable_injection_check: bool,
    /// Allow only these HTTP methods (default: GET, POST).
    pub allowed_methods: Vec<String>,
    /// Recognized SPARQL protocol parameter names.
    pub known_params: HashSet<String>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        let known_params = [
            "query",
            "update",
            "default-graph-uri",
            "named-graph-uri",
            "using-graph-uri",
            "using-named-graph-uri",
            "format",
            "output",
            "callback",
            "force-accept",
            "timeout",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            max_body_bytes: 10 * 1024 * 1024, // 10 MiB
            max_query_length: 1_000_000,      // 1 MB
            enable_injection_check: true,
            allowed_methods: vec!["GET".to_string(), "POST".to_string()],
            known_params,
        }
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

/// A structured validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// HTTP method is not allowed.
    MethodNotAllowed {
        /// The method that was received.
        received: String,
        /// The methods that are permitted.
        allowed: Vec<String>,
    },
    /// Request body exceeds the size limit.
    BodyTooLarge {
        /// Actual body size in bytes.
        actual: usize,
        /// Maximum allowed size in bytes.
        limit: usize,
    },
    /// The SPARQL query string is longer than permitted.
    QueryTooLong {
        /// Actual query length in bytes.
        actual: usize,
        /// Maximum allowed length in bytes.
        limit: usize,
    },
    /// No recognizable SPARQL query type keyword was found.
    MissingQueryType,
    /// Unbalanced braces in the query.
    UnbalancedBraces {
        /// Net brace depth at end of query (positive = unclosed `{`, negative = extra `}`).
        depth: i64,
    },
    /// One or more graph URIs are malformed.
    InvalidGraphUri(String),
    /// An unknown query parameter was detected.
    UnknownParameter(String),
    /// A query parameter appears more than once.
    DuplicateParameter(String),
    /// Content-Type header is invalid or missing for a POST request.
    InvalidContentType {
        /// The content-type that was received (or empty if absent).
        received: String,
    },
    /// A suspicious injection pattern was detected.
    SuspiciousPattern(String),
    /// A query parameter value contains malformed percent-encoding.
    EncodingError {
        /// The parameter name with the encoding issue.
        param: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MethodNotAllowed { received, allowed } => {
                write!(
                    f,
                    "HTTP method '{}' not allowed; accepted: {}",
                    received,
                    allowed.join(", ")
                )
            }
            ValidationError::BodyTooLarge { actual, limit } => {
                write!(
                    f,
                    "Request body {actual} bytes exceeds limit of {limit} bytes"
                )
            }
            ValidationError::QueryTooLong { actual, limit } => {
                write!(
                    f,
                    "Query length {actual} bytes exceeds limit of {limit} bytes"
                )
            }
            ValidationError::MissingQueryType => {
                write!(
                    f,
                    "SPARQL query must contain SELECT, ASK, CONSTRUCT, or DESCRIBE"
                )
            }
            ValidationError::UnbalancedBraces { depth } => {
                if *depth > 0 {
                    write!(f, "Unbalanced braces: {depth} unclosed '{{' in query")
                } else {
                    write!(f, "Unbalanced braces: {} extra '}}' in query", depth.abs())
                }
            }
            ValidationError::InvalidGraphUri(uri) => {
                write!(f, "Invalid graph URI: '{uri}'")
            }
            ValidationError::UnknownParameter(p) => {
                write!(f, "Unknown query parameter: '{p}'")
            }
            ValidationError::DuplicateParameter(p) => {
                write!(f, "Duplicate query parameter: '{p}'")
            }
            ValidationError::InvalidContentType { received } => {
                if received.is_empty() {
                    write!(f, "POST request missing Content-Type header")
                } else {
                    write!(
                        f,
                        "Invalid Content-Type '{received}' for SPARQL POST request"
                    )
                }
            }
            ValidationError::SuspiciousPattern(msg) => {
                write!(f, "Suspicious query pattern detected: {msg}")
            }
            ValidationError::EncodingError { param } => {
                write!(f, "Malformed percent-encoding in parameter '{param}'")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

// ── Validation result ─────────────────────────────────────────────────────────

/// Outcome of running the full validation pipeline.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// All errors encountered during validation.
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    fn new() -> Self {
        Self { errors: Vec::new() }
    }

    fn push(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// Returns `true` when there are no validation errors.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the first error, if any.
    pub fn first_error(&self) -> Option<&ValidationError> {
        self.errors.first()
    }

    /// Returns the number of errors found.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ── Incoming request representation ──────────────────────────────────────────

/// A simplified representation of an incoming HTTP request for validation.
#[derive(Debug, Clone, Default)]
pub struct IncomingRequest {
    /// HTTP method, e.g. `"GET"` or `"POST"`.
    pub method: String,
    /// Content-Type header value (empty if absent).
    pub content_type: String,
    /// Raw query parameters as `(name, value)` pairs in the order received.
    ///
    /// Duplicate parameter names are intentionally kept as separate entries
    /// so that the validator can detect them.
    pub params: Vec<(String, String)>,
    /// Request body size in bytes (for size-limit checking).
    pub body_size: usize,
    /// The SPARQL query string (from the `query` parameter or POST body).
    pub query: String,
    /// Graph URIs from `default-graph-uri` or `named-graph-uri` parameters.
    pub graph_uris: Vec<String>,
}

impl IncomingRequest {
    /// Create a minimal GET request with a SPARQL query.
    pub fn get(query: impl Into<String>) -> Self {
        Self {
            method: "GET".to_string(),
            query: query.into(),
            ..Self::default()
        }
    }

    /// Create a minimal POST request with application/sparql-query content type.
    pub fn post(query: impl Into<String>, body_size: usize) -> Self {
        Self {
            method: "POST".to_string(),
            content_type: "application/sparql-query".to_string(),
            query: query.into(),
            body_size,
            ..Self::default()
        }
    }
}

// ── Validator ─────────────────────────────────────────────────────────────────

/// SPARQL HTTP request validator.
pub struct RequestValidator {
    config: ValidatorConfig,
}

/// Valid SPARQL POST Content-Type values.
const VALID_POST_CONTENT_TYPES: &[&str] = &[
    "application/sparql-query",
    "application/sparql-update",
    "application/x-www-form-urlencoded",
];

/// Keywords that indicate a valid SPARQL query type.
const QUERY_TYPE_KEYWORDS: &[&str] = &["SELECT", "ASK", "CONSTRUCT", "DESCRIBE"];

/// Simple injection patterns to reject (case-insensitive substrings in the raw query).
const INJECTION_PATTERNS: &[(&str, &str)] = &[
    ("DROP ALL", "DROP ALL graph operation detected"),
    ("DROP GRAPH", "DROP GRAPH operation detected"),
    ("CLEAR ALL", "CLEAR ALL operation detected"),
    ("DELETE WHERE { ?s ?p ?o }", "bulk delete pattern detected"),
];

impl RequestValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default configuration.
    pub fn default_config() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Run all validation checks against `request`.
    ///
    /// Collects all errors rather than short-circuiting on the first one.
    pub fn validate(&self, request: &IncomingRequest) -> ValidationResult {
        let mut result = ValidationResult::new();

        self.check_method(request, &mut result);
        self.check_body_size(request, &mut result);
        self.check_query_length(request, &mut result);
        self.check_content_type(request, &mut result);
        self.check_query_syntax(request, &mut result);
        self.check_graph_uris(request, &mut result);
        self.check_parameters(request, &mut result);

        if self.config.enable_injection_check {
            self.check_injection(request, &mut result);
        }

        result
    }

    // ── Individual checks ─────────────────────────────────────────────────────

    fn check_method(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        if !self
            .config
            .allowed_methods
            .contains(&request.method.to_uppercase())
        {
            result.push(ValidationError::MethodNotAllowed {
                received: request.method.clone(),
                allowed: self.config.allowed_methods.clone(),
            });
        }
    }

    fn check_body_size(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        if request.body_size > self.config.max_body_bytes {
            result.push(ValidationError::BodyTooLarge {
                actual: request.body_size,
                limit: self.config.max_body_bytes,
            });
        }
    }

    fn check_query_length(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        let len = request.query.len();
        if len > self.config.max_query_length {
            result.push(ValidationError::QueryTooLong {
                actual: len,
                limit: self.config.max_query_length,
            });
        }
    }

    fn check_content_type(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        if request.method.to_uppercase() != "POST" {
            return;
        }
        let ct = request.content_type.to_lowercase();
        let valid = VALID_POST_CONTENT_TYPES
            .iter()
            .any(|accepted| ct.starts_with(accepted));
        if !valid {
            result.push(ValidationError::InvalidContentType {
                received: request.content_type.clone(),
            });
        }
    }

    fn check_query_syntax(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        if request.query.is_empty() {
            return;
        }

        // Check for balanced braces.
        let depth = brace_depth(&request.query);
        if depth != 0 {
            result.push(ValidationError::UnbalancedBraces { depth });
        }

        // Check that a recognised query type keyword is present.
        let upper = request.query.to_uppercase();
        let has_type = QUERY_TYPE_KEYWORDS.iter().any(|kw| upper.contains(kw));
        if !has_type {
            result.push(ValidationError::MissingQueryType);
        }
    }

    fn check_graph_uris(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        for uri in &request.graph_uris {
            if !is_valid_iri(uri) {
                result.push(ValidationError::InvalidGraphUri(uri.clone()));
            }
        }
    }

    fn check_parameters(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        let mut seen: HashMap<&str, usize> = HashMap::new();

        for (name, value) in &request.params {
            // Check for unknown parameters.
            if !self.config.known_params.contains(name.as_str()) {
                result.push(ValidationError::UnknownParameter(name.clone()));
            }

            // Track duplicate occurrences.
            let count = seen.entry(name.as_str()).or_insert(0);
            *count += 1;
            if *count == 2 {
                result.push(ValidationError::DuplicateParameter(name.clone()));
            }

            // Check for malformed percent-encoding in value.
            if has_bad_percent_encoding(value) {
                result.push(ValidationError::EncodingError {
                    param: name.clone(),
                });
            }
        }
    }

    fn check_injection(&self, request: &IncomingRequest, result: &mut ValidationResult) {
        if request.query.is_empty() {
            return;
        }
        let upper = request.query.to_uppercase();
        for (pattern, description) in INJECTION_PATTERNS {
            if upper.contains(&pattern.to_uppercase()) {
                result.push(ValidationError::SuspiciousPattern(description.to_string()));
            }
        }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &ValidatorConfig {
        &self.config
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Count net brace depth in `query` (ignores braces inside string literals).
fn brace_depth(query: &str) -> i64 {
    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut string_char = '"';

    for ch in query.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
            }
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// Perform a basic IRI well-formedness check.
///
/// Accepts absolute IRIs of the form `scheme:path` where `scheme` starts with
/// a letter and contains only letters, digits, `+`, `-`, or `.`.
fn is_valid_iri(uri: &str) -> bool {
    if uri.is_empty() {
        return false;
    }
    // Must have a scheme (at least one char before ':')
    if let Some(colon_pos) = uri.find(':') {
        if colon_pos == 0 {
            return false;
        }
        let scheme = &uri[..colon_pos];
        let rest = &uri[colon_pos + 1..];

        // Scheme must start with a letter and contain only [a-zA-Z0-9+\-.].
        let scheme_valid = scheme.chars().enumerate().all(|(i, c)| {
            if i == 0 {
                c.is_ascii_alphabetic()
            } else {
                c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')
            }
        });

        // Rest must be non-empty (absolute IRI).
        scheme_valid && !rest.is_empty()
    } else {
        false
    }
}

/// Returns `true` if `value` contains a percent sign not followed by two hex digits.
fn has_bad_percent_encoding(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len()
                || !bytes[i + 1].is_ascii_hexdigit()
                || !bytes[i + 2].is_ascii_hexdigit()
            {
                return true;
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    false
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validator() -> RequestValidator {
        RequestValidator::default_config()
    }

    fn valid_select() -> &'static str {
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    }

    // ── ValidatorConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config_known_params() {
        let cfg = ValidatorConfig::default();
        assert!(cfg.known_params.contains("query"));
        assert!(cfg.known_params.contains("default-graph-uri"));
        assert!(cfg.known_params.contains("named-graph-uri"));
    }

    #[test]
    fn test_default_config_allowed_methods() {
        let cfg = ValidatorConfig::default();
        assert!(cfg.allowed_methods.contains(&"GET".to_string()));
        assert!(cfg.allowed_methods.contains(&"POST".to_string()));
        assert!(!cfg.allowed_methods.contains(&"DELETE".to_string()));
    }

    #[test]
    fn test_default_config_limits() {
        let cfg = ValidatorConfig::default();
        assert!(cfg.max_body_bytes > 0);
        assert!(cfg.max_query_length > 0);
    }

    // ── ValidationResult ─────────────────────────────────────────────────────

    #[test]
    fn test_validation_result_valid() {
        let r = ValidationResult::new();
        assert!(r.is_valid());
        assert_eq!(r.error_count(), 0);
        assert!(r.first_error().is_none());
    }

    #[test]
    fn test_validation_result_with_error() {
        let mut r = ValidationResult::new();
        r.push(ValidationError::MissingQueryType);
        assert!(!r.is_valid());
        assert_eq!(r.error_count(), 1);
        assert!(r.first_error().is_some());
    }

    // ── HTTP method ──────────────────────────────────────────────────────────

    #[test]
    fn test_get_allowed() {
        let req = IncomingRequest::get(valid_select());
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MethodNotAllowed { .. })));
    }

    #[test]
    fn test_post_allowed() {
        let req = IncomingRequest::post(valid_select(), 100);
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MethodNotAllowed { .. })));
    }

    #[test]
    fn test_delete_rejected() {
        let req = IncomingRequest {
            method: "DELETE".to_string(),
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MethodNotAllowed { .. })));
    }

    #[test]
    fn test_put_rejected() {
        let req = IncomingRequest {
            method: "PUT".to_string(),
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MethodNotAllowed { .. })));
    }

    // ── Body size ────────────────────────────────────────────────────────────

    #[test]
    fn test_body_size_within_limit() {
        let cfg = ValidatorConfig {
            max_body_bytes: 1000,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            body_size: 500,
            ..Default::default()
        };
        let r = v.validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::BodyTooLarge { .. })));
    }

    #[test]
    fn test_body_size_exceeds_limit() {
        let cfg = ValidatorConfig {
            max_body_bytes: 100,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            body_size: 200,
            ..Default::default()
        };
        let r = v.validate(&req);
        assert!(r.errors.iter().any(|e| matches!(
            e,
            ValidationError::BodyTooLarge {
                actual: 200,
                limit: 100
            }
        )));
    }

    // ── Query length ─────────────────────────────────────────────────────────

    #[test]
    fn test_query_length_ok() {
        let cfg = ValidatorConfig {
            max_query_length: 1000,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        let req = IncomingRequest::get(valid_select());
        let r = v.validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::QueryTooLong { .. })));
    }

    #[test]
    fn test_query_length_exceeds_limit() {
        let cfg = ValidatorConfig {
            max_query_length: 10,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        let req = IncomingRequest::get("SELECT ?s WHERE { ?s ?p ?o }");
        let r = v.validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::QueryTooLong { .. })));
    }

    // ── Content-Type ─────────────────────────────────────────────────────────

    #[test]
    fn test_post_sparql_query_content_type() {
        let req = IncomingRequest::post(valid_select(), 100);
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidContentType { .. })));
    }

    #[test]
    fn test_post_form_content_type() {
        let req = IncomingRequest {
            method: "POST".to_string(),
            content_type: "application/x-www-form-urlencoded".to_string(),
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidContentType { .. })));
    }

    #[test]
    fn test_post_invalid_content_type() {
        let req = IncomingRequest {
            method: "POST".to_string(),
            content_type: "text/plain".to_string(),
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidContentType { .. })));
    }

    #[test]
    fn test_post_missing_content_type() {
        let req = IncomingRequest {
            method: "POST".to_string(),
            content_type: String::new(),
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidContentType { .. })));
    }

    #[test]
    fn test_get_does_not_check_content_type() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            content_type: "text/plain".to_string(), // irrelevant for GET
            query: valid_select().to_string(),
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidContentType { .. })));
    }

    // ── Query syntax ─────────────────────────────────────────────────────────

    #[test]
    fn test_balanced_braces_ok() {
        let req = IncomingRequest::get("SELECT ?s WHERE { ?s ?p ?o }");
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnbalancedBraces { .. })));
    }

    #[test]
    fn test_unbalanced_open_brace() {
        let req = IncomingRequest::get("SELECT ?s WHERE { ?s ?p ?o");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnbalancedBraces { depth: 1 })));
    }

    #[test]
    fn test_unbalanced_close_brace() {
        let req = IncomingRequest::get("SELECT ?s WHERE { ?s ?p ?o }}");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnbalancedBraces { depth: -1 })));
    }

    #[test]
    fn test_missing_query_type() {
        let req = IncomingRequest::get("FILTER (?x > 5) WHERE { ?s ?p ?o }");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_select_query_type_accepted() {
        let req = IncomingRequest::get(valid_select());
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_ask_query_type_accepted() {
        let req = IncomingRequest::get("ASK { ?s ?p ?o }");
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_construct_query_type_accepted() {
        let req = IncomingRequest::get("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }");
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_describe_query_type_accepted() {
        let req = IncomingRequest::get("DESCRIBE <http://example.org>");
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_empty_query_no_syntax_error() {
        // An empty query produces no syntax errors (no query type, no braces).
        let req = IncomingRequest::get("");
        let r = validator().validate(&req);
        // No UnbalancedBraces or MissingQueryType for empty query.
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnbalancedBraces { .. })));
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingQueryType)));
    }

    #[test]
    fn test_braces_in_string_ignored() {
        // Braces inside string literals should not affect depth.
        let req = IncomingRequest::get("SELECT ?s WHERE { ?s ?p \"{this is fine}\" }");
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnbalancedBraces { .. })));
    }

    // ── Graph URIs ───────────────────────────────────────────────────────────

    #[test]
    fn test_valid_graph_uri() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            graph_uris: vec!["http://example.org/graph".to_string()],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidGraphUri(_))));
    }

    #[test]
    fn test_invalid_graph_uri_no_scheme() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            graph_uris: vec!["not-a-uri".to_string()],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidGraphUri(_))));
    }

    #[test]
    fn test_invalid_graph_uri_empty() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            graph_uris: vec!["".to_string()],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidGraphUri(_))));
    }

    #[test]
    fn test_valid_urn_uri() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            graph_uris: vec!["urn:example:graph".to_string()],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidGraphUri(_))));
    }

    // ── Parameters ───────────────────────────────────────────────────────────

    #[test]
    fn test_known_parameter_accepted() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            params: vec![("query".to_string(), "SELECT ?s WHERE {}".to_string())],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnknownParameter(_))));
    }

    #[test]
    fn test_unknown_parameter_rejected() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            params: vec![("foo".to_string(), "bar".to_string())],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnknownParameter(p) if p == "foo")));
    }

    #[test]
    fn test_duplicate_parameter_rejected() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            params: vec![
                ("query".to_string(), "SELECT ?a WHERE {}".to_string()),
                ("query".to_string(), "SELECT ?b WHERE {}".to_string()),
            ],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateParameter(p) if p == "query")));
    }

    #[test]
    fn test_bad_percent_encoding_detected() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            params: vec![("query".to_string(), "SELECT %ZZ WHERE {}".to_string())],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::EncodingError { param: p } if p == "query")));
    }

    #[test]
    fn test_valid_percent_encoding_accepted() {
        let req = IncomingRequest {
            method: "GET".to_string(),
            query: valid_select().to_string(),
            params: vec![(
                "query".to_string(),
                "SELECT%20%3Fs%20WHERE%20%7B%7D".to_string(),
            )],
            ..Default::default()
        };
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::EncodingError { .. })));
    }

    // ── Injection detection ──────────────────────────────────────────────────

    #[test]
    fn test_injection_drop_all() {
        let req = IncomingRequest::get("DROP ALL");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::SuspiciousPattern(_))));
    }

    #[test]
    fn test_injection_drop_graph() {
        let req = IncomingRequest::get("DROP GRAPH <http://example.org/g>");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::SuspiciousPattern(_))));
    }

    #[test]
    fn test_injection_clear_all() {
        let req = IncomingRequest::get("CLEAR ALL");
        let r = validator().validate(&req);
        assert!(r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::SuspiciousPattern(_))));
    }

    #[test]
    fn test_injection_disabled() {
        let cfg = ValidatorConfig {
            enable_injection_check: false,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        // Even a dangerous pattern should not trigger when disabled.
        let req = IncomingRequest::get("DROP ALL");
        let r = v.validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::SuspiciousPattern(_))));
    }

    #[test]
    fn test_normal_select_no_injection() {
        let req = IncomingRequest::get(valid_select());
        let r = validator().validate(&req);
        assert!(!r
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::SuspiciousPattern(_))));
    }

    // ── ValidationError display ──────────────────────────────────────────────

    #[test]
    fn test_error_display_method_not_allowed() {
        let e = ValidationError::MethodNotAllowed {
            received: "DELETE".to_string(),
            allowed: vec!["GET".to_string()],
        };
        let s = e.to_string();
        assert!(s.contains("DELETE"));
        assert!(s.contains("GET"));
    }

    #[test]
    fn test_error_display_body_too_large() {
        let e = ValidationError::BodyTooLarge {
            actual: 2000,
            limit: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("2000"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn test_error_display_query_too_long() {
        let e = ValidationError::QueryTooLong {
            actual: 500,
            limit: 100,
        };
        let s = e.to_string();
        assert!(s.contains("500"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_error_display_missing_query_type() {
        let s = ValidationError::MissingQueryType.to_string();
        assert!(s.contains("SELECT"));
    }

    #[test]
    fn test_error_display_unbalanced_open() {
        let e = ValidationError::UnbalancedBraces { depth: 2 };
        let s = e.to_string();
        assert!(s.contains("unclosed"));
    }

    #[test]
    fn test_error_display_unbalanced_close() {
        let e = ValidationError::UnbalancedBraces { depth: -1 };
        let s = e.to_string();
        assert!(s.contains("extra"));
    }

    #[test]
    fn test_error_display_invalid_graph_uri() {
        let e = ValidationError::InvalidGraphUri("bad-uri".to_string());
        assert!(e.to_string().contains("bad-uri"));
    }

    #[test]
    fn test_error_display_unknown_param() {
        let e = ValidationError::UnknownParameter("foo".to_string());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_error_display_duplicate_param() {
        let e = ValidationError::DuplicateParameter("query".to_string());
        assert!(e.to_string().contains("query"));
    }

    #[test]
    fn test_error_display_invalid_content_type() {
        let e = ValidationError::InvalidContentType {
            received: "text/plain".to_string(),
        };
        assert!(e.to_string().contains("text/plain"));
    }

    #[test]
    fn test_error_display_missing_content_type() {
        let e = ValidationError::InvalidContentType {
            received: String::new(),
        };
        assert!(e.to_string().contains("missing"));
    }

    #[test]
    fn test_error_display_suspicious_pattern() {
        let e = ValidationError::SuspiciousPattern("drop all".to_string());
        assert!(e.to_string().contains("drop all"));
    }

    #[test]
    fn test_error_display_encoding_error() {
        let e = ValidationError::EncodingError {
            param: "query".to_string(),
        };
        assert!(e.to_string().contains("query"));
    }

    // ── Utility functions ────────────────────────────────────────────────────

    #[test]
    fn test_brace_depth_balanced() {
        assert_eq!(brace_depth("{ { } }"), 0);
    }

    #[test]
    fn test_brace_depth_open() {
        assert_eq!(brace_depth("{ {"), 2);
    }

    #[test]
    fn test_brace_depth_close() {
        assert_eq!(brace_depth("}"), -1);
    }

    #[test]
    fn test_brace_depth_ignores_string() {
        assert_eq!(brace_depth("SELECT ?s WHERE { ?s ?p \"{{{\" }"), 0);
    }

    #[test]
    fn test_is_valid_iri_http() {
        assert!(is_valid_iri("http://example.org/"));
    }

    #[test]
    fn test_is_valid_iri_https() {
        assert!(is_valid_iri("https://example.org/graph"));
    }

    #[test]
    fn test_is_valid_iri_urn() {
        assert!(is_valid_iri("urn:example:a"));
    }

    #[test]
    fn test_is_valid_iri_empty() {
        assert!(!is_valid_iri(""));
    }

    #[test]
    fn test_is_valid_iri_no_scheme() {
        assert!(!is_valid_iri("example.org"));
    }

    #[test]
    fn test_is_valid_iri_colon_at_start() {
        assert!(!is_valid_iri(":path"));
    }

    #[test]
    fn test_has_bad_percent_encoding_valid() {
        assert!(!has_bad_percent_encoding("hello%20world%3F"));
    }

    #[test]
    fn test_has_bad_percent_encoding_invalid() {
        assert!(has_bad_percent_encoding("hello%ZZworld"));
    }

    #[test]
    fn test_has_bad_percent_encoding_truncated() {
        assert!(has_bad_percent_encoding("hello%2"));
    }

    #[test]
    fn test_has_bad_percent_encoding_no_percent() {
        assert!(!has_bad_percent_encoding("hello world"));
    }

    // ── Config accessor ──────────────────────────────────────────────────────

    #[test]
    fn test_config_accessor() {
        let v = validator();
        assert_eq!(v.config().allowed_methods, vec!["GET", "POST"]);
    }

    // ── Multiple errors collected ─────────────────────────────────────────────

    #[test]
    fn test_multiple_errors_collected() {
        let cfg = ValidatorConfig {
            max_body_bytes: 1,
            max_query_length: 5,
            ..ValidatorConfig::default()
        };
        let v = RequestValidator::new(cfg);
        let req = IncomingRequest {
            method: "PUT".to_string(),
            body_size: 100,
            query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            ..Default::default()
        };
        let r = v.validate(&req);
        // Should have method, body size, and query length errors
        assert!(r.error_count() >= 3);
    }
}
