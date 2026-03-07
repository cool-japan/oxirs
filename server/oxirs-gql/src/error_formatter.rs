//! GraphQL error formatting and classification.
//!
//! Provides error classification (validation, execution, internal), path
//! tracking, extension metadata (error codes, timestamps, trace IDs),
//! deduplication, severity levels, and structured error response building.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Error classification
// ────────────────────────────────────────────────────────────────────────────

/// Classification of a GraphQL error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    /// The query failed validation before execution.
    Validation,
    /// An error occurred during query execution (resolver failure, etc.).
    Execution,
    /// An internal server error not caused by the client.
    Internal,
    /// Authentication or authorization failure.
    Authorization,
    /// Rate limiting or resource exhaustion.
    RateLimit,
    /// Input coercion / type mismatch.
    InputCoercion,
}

impl ErrorClass {
    /// A short machine-readable code for this classification.
    pub fn code(&self) -> &'static str {
        match self {
            ErrorClass::Validation => "VALIDATION_ERROR",
            ErrorClass::Execution => "EXECUTION_ERROR",
            ErrorClass::Internal => "INTERNAL_ERROR",
            ErrorClass::Authorization => "AUTHORIZATION_ERROR",
            ErrorClass::RateLimit => "RATE_LIMIT_ERROR",
            ErrorClass::InputCoercion => "INPUT_COERCION_ERROR",
        }
    }

    /// A human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ErrorClass::Validation => "Validation Error",
            ErrorClass::Execution => "Execution Error",
            ErrorClass::Internal => "Internal Error",
            ErrorClass::Authorization => "Authorization Error",
            ErrorClass::RateLimit => "Rate Limit Error",
            ErrorClass::InputCoercion => "Input Coercion Error",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Error severity
// ────────────────────────────────────────────────────────────────────────────

/// Severity level for a GraphQL error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational; the request may have partially succeeded.
    Info,
    /// A warning; the response is still usable.
    Warning,
    /// An error; the affected field is null.
    Error,
    /// Critical; the entire request failed.
    Critical,
}

impl Severity {
    pub fn label(&self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
            Severity::Critical => "critical",
        }
    }

    /// Map a classification to its default severity.
    pub fn from_class(class: ErrorClass) -> Self {
        match class {
            ErrorClass::Validation => Severity::Error,
            ErrorClass::Execution => Severity::Error,
            ErrorClass::Internal => Severity::Critical,
            ErrorClass::Authorization => Severity::Error,
            ErrorClass::RateLimit => Severity::Warning,
            ErrorClass::InputCoercion => Severity::Error,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Path segment
// ────────────────────────────────────────────────────────────────────────────

/// A segment in the field path where an error occurred.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PathSegment {
    /// A named field.
    Field(String),
    /// An index into a list.
    Index(usize),
}

impl PathSegment {
    pub fn field(name: impl Into<String>) -> Self {
        PathSegment::Field(name.into())
    }

    pub fn index(i: usize) -> Self {
        PathSegment::Index(i)
    }
}

impl std::fmt::Display for PathSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathSegment::Field(name) => write!(f, "{}", name),
            PathSegment::Index(i) => write!(f, "{}", i),
        }
    }
}

/// Format a path as a dotted string (e.g., `user.friends.0.name`).
pub fn format_path(path: &[PathSegment]) -> String {
    path.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

// ────────────────────────────────────────────────────────────────────────────
// Extension metadata
// ────────────────────────────────────────────────────────────────────────────

/// Extension metadata attached to a GraphQL error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorExtensions {
    /// Machine-readable error code (e.g., "VALIDATION_ERROR").
    pub code: String,
    /// ISO-8601 timestamp of when the error occurred.
    pub timestamp: Option<String>,
    /// Distributed trace identifier.
    pub trace_id: Option<String>,
    /// Extra key-value pairs.
    pub extra: HashMap<String, String>,
}

impl ErrorExtensions {
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            timestamp: None,
            trace_id: None,
            extra: HashMap::new(),
        }
    }

    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }

    pub fn with_trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GraphQL error
// ────────────────────────────────────────────────────────────────────────────

/// A single GraphQL error.
#[derive(Debug, Clone)]
pub struct GraphQLError {
    /// Human-readable error message.
    pub message: String,
    /// Classification.
    pub classification: ErrorClass,
    /// Severity level.
    pub severity: Severity,
    /// Path in the query where the error occurred.
    pub path: Vec<PathSegment>,
    /// Source locations in the query document.
    pub locations: Vec<SourceLocation>,
    /// Extension metadata.
    pub extensions: Option<ErrorExtensions>,
}

/// A source location in the query document.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

impl GraphQLError {
    /// Create a new error with a message and classification.
    pub fn new(message: impl Into<String>, classification: ErrorClass) -> Self {
        Self {
            message: message.into(),
            classification,
            severity: Severity::from_class(classification),
            path: Vec::new(),
            locations: Vec::new(),
            extensions: None,
        }
    }

    /// Attach a field path.
    pub fn with_path(mut self, path: Vec<PathSegment>) -> Self {
        self.path = path;
        self
    }

    /// Attach source locations.
    pub fn with_locations(mut self, locations: Vec<SourceLocation>) -> Self {
        self.locations = locations;
        self
    }

    /// Override the default severity.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Attach extension metadata.
    pub fn with_extensions(mut self, ext: ErrorExtensions) -> Self {
        self.extensions = Some(ext);
        self
    }

    /// Convenience: attach a simple code extension.
    pub fn with_code(self, code: impl Into<String>) -> Self {
        self.with_extensions(ErrorExtensions::new(code))
    }

    /// Format the path as a dot-separated string.
    pub fn path_string(&self) -> String {
        format_path(&self.path)
    }

    /// Produce a deduplication key for grouping similar errors.
    pub fn dedup_key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.classification.code(),
            self.path_string(),
            self.message
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Custom error messages per type
// ────────────────────────────────────────────────────────────────────────────

/// Registry of custom error messages keyed by error code.
pub struct ErrorMessageRegistry {
    /// Maps error codes to custom message templates.
    messages: HashMap<String, String>,
}

impl Default for ErrorMessageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorMessageRegistry {
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
        }
    }

    /// Register a custom message for an error code.
    pub fn register(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.messages.insert(code.into(), message.into());
    }

    /// Look up a custom message for the given error code.
    pub fn get(&self, code: &str) -> Option<&str> {
        self.messages.get(code).map(|s| s.as_str())
    }

    /// Apply the registry to an error: if a custom message exists for the
    /// error's code, replace the error's message.
    pub fn apply(&self, error: &mut GraphQLError) {
        if let Some(ext) = &error.extensions {
            if let Some(custom) = self.messages.get(&ext.code) {
                error.message = custom.clone();
            }
        }
    }

    /// Number of registered custom messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Error aggregation (dedup)
// ────────────────────────────────────────────────────────────────────────────

/// Aggregates GraphQL errors, deduplicating identical ones and tracking counts.
pub struct ErrorAggregator {
    /// Deduplicated errors keyed by dedup_key.
    errors: HashMap<String, AggregatedError>,
    /// Insertion order for deterministic output.
    order: Vec<String>,
}

/// An error that may have been seen multiple times.
#[derive(Debug, Clone)]
pub struct AggregatedError {
    pub error: GraphQLError,
    pub count: usize,
}

impl Default for ErrorAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorAggregator {
    pub fn new() -> Self {
        Self {
            errors: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add an error to the aggregator.
    pub fn add(&mut self, error: GraphQLError) {
        let key = error.dedup_key();
        if let Some(existing) = self.errors.get_mut(&key) {
            existing.count += 1;
        } else {
            self.order.push(key.clone());
            self.errors.insert(key, AggregatedError { error, count: 1 });
        }
    }

    /// Number of unique errors.
    pub fn unique_count(&self) -> usize {
        self.errors.len()
    }

    /// Total errors (including duplicates).
    pub fn total_count(&self) -> usize {
        self.errors.values().map(|e| e.count).sum()
    }

    /// Retrieve unique errors in insertion order.
    pub fn errors(&self) -> Vec<&AggregatedError> {
        self.order
            .iter()
            .filter_map(|k| self.errors.get(k))
            .collect()
    }

    /// Drain all aggregated errors, consuming the aggregator.
    pub fn into_errors(self) -> Vec<AggregatedError> {
        self.order
            .into_iter()
            .filter_map(|k| self.errors.get(&k).cloned())
            .collect()
    }

    /// Clear all accumulated errors.
    pub fn clear(&mut self) {
        self.errors.clear();
        self.order.clear();
    }

    /// Filter errors by minimum severity.
    pub fn errors_above_severity(&self, min: Severity) -> Vec<&AggregatedError> {
        self.errors()
            .into_iter()
            .filter(|ae| ae.error.severity >= min)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Structured response builder
// ────────────────────────────────────────────────────────────────────────────

/// A structured GraphQL error response (matching the GraphQL spec shape).
#[derive(Debug, Clone)]
pub struct ErrorResponse {
    pub errors: Vec<ErrorEntry>,
}

/// One entry in the `errors` array of a GraphQL response.
#[derive(Debug, Clone)]
pub struct ErrorEntry {
    pub message: String,
    pub path: Option<Vec<String>>,
    pub locations: Option<Vec<SourceLocation>>,
    pub extensions: Option<HashMap<String, String>>,
}

/// Builder for constructing a complete error response.
pub struct ErrorResponseBuilder {
    entries: Vec<ErrorEntry>,
    default_timestamp: Option<String>,
    default_trace_id: Option<String>,
}

impl Default for ErrorResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorResponseBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            default_timestamp: None,
            default_trace_id: None,
        }
    }

    /// Set a default timestamp for all entries.
    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.default_timestamp = Some(ts.into());
        self
    }

    /// Set a default trace ID for all entries.
    pub fn with_trace_id(mut self, id: impl Into<String>) -> Self {
        self.default_trace_id = Some(id.into());
        self
    }

    /// Add a single GraphQL error.
    pub fn add_error(&mut self, error: &GraphQLError) {
        let path = if error.path.is_empty() {
            None
        } else {
            Some(error.path.iter().map(|s| s.to_string()).collect())
        };

        let locations = if error.locations.is_empty() {
            None
        } else {
            Some(error.locations.clone())
        };

        let mut ext_map = HashMap::new();
        ext_map.insert("code".to_string(), error.classification.code().to_string());
        ext_map.insert("severity".to_string(), error.severity.label().to_string());
        ext_map.insert(
            "classification".to_string(),
            error.classification.label().to_string(),
        );

        if let Some(ref ts) = self.default_timestamp {
            ext_map.insert("timestamp".to_string(), ts.clone());
        }
        if let Some(ref tid) = self.default_trace_id {
            ext_map.insert("traceId".to_string(), tid.clone());
        }

        if let Some(ref exts) = error.extensions {
            // Error-level code overrides classification code
            if !exts.code.is_empty() {
                ext_map.insert("code".to_string(), exts.code.clone());
            }
            if let Some(ref ts) = exts.timestamp {
                ext_map.insert("timestamp".to_string(), ts.clone());
            }
            if let Some(ref tid) = exts.trace_id {
                ext_map.insert("traceId".to_string(), tid.clone());
            }
            for (k, v) in &exts.extra {
                ext_map.insert(k.clone(), v.clone());
            }
        }

        self.entries.push(ErrorEntry {
            message: error.message.clone(),
            path,
            locations,
            extensions: Some(ext_map),
        });
    }

    /// Add errors from an aggregator (with dedup counts).
    pub fn add_aggregated(&mut self, aggregator: &ErrorAggregator) {
        for ae in aggregator.errors() {
            self.add_error(&ae.error);
            // Include count in extensions if > 1
            if ae.count > 1 {
                if let Some(entry) = self.entries.last_mut() {
                    if let Some(ref mut ext) = entry.extensions {
                        ext.insert("occurrences".to_string(), ae.count.to_string());
                    }
                }
            }
        }
    }

    /// Build the final response.
    pub fn build(self) -> ErrorResponse {
        ErrorResponse {
            errors: self.entries,
        }
    }

    /// Number of entries accumulated.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Serialize an [`ErrorResponse`] to a JSON-like string representation.
pub fn format_response_json(response: &ErrorResponse) -> String {
    let mut out = String::from("{\"errors\":[");
    for (i, entry) in response.errors.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"message\":");
        out.push_str(&json_escape(&entry.message));

        if let Some(ref path) = entry.path {
            out.push_str(",\"path\":[");
            for (j, seg) in path.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                // Try to parse as number for array indices
                if seg.parse::<usize>().is_ok() {
                    out.push_str(seg);
                } else {
                    out.push_str(&json_escape(seg));
                }
            }
            out.push(']');
        }

        if let Some(ref locs) = entry.locations {
            out.push_str(",\"locations\":[");
            for (j, loc) in locs.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "{{\"line\":{},\"column\":{}}}",
                    loc.line, loc.column
                ));
            }
            out.push(']');
        }

        if let Some(ref ext) = entry.extensions {
            out.push_str(",\"extensions\":{");
            let mut sorted_keys: Vec<&String> = ext.keys().collect();
            sorted_keys.sort();
            for (j, key) in sorted_keys.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                if let Some(val) = ext.get(*key) {
                    out.push_str(&format!("{}:{}", json_escape(key), json_escape(val)));
                }
            }
            out.push('}');
        }

        out.push('}');
    }
    out.push_str("]}");
    out
}

/// Simple JSON string escaping.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ErrorClass ──────────────────────────────────────────────────────────

    #[test]
    fn test_error_class_codes() {
        assert_eq!(ErrorClass::Validation.code(), "VALIDATION_ERROR");
        assert_eq!(ErrorClass::Execution.code(), "EXECUTION_ERROR");
        assert_eq!(ErrorClass::Internal.code(), "INTERNAL_ERROR");
        assert_eq!(ErrorClass::Authorization.code(), "AUTHORIZATION_ERROR");
        assert_eq!(ErrorClass::RateLimit.code(), "RATE_LIMIT_ERROR");
        assert_eq!(ErrorClass::InputCoercion.code(), "INPUT_COERCION_ERROR");
    }

    #[test]
    fn test_error_class_labels() {
        assert_eq!(ErrorClass::Validation.label(), "Validation Error");
        assert_eq!(ErrorClass::Internal.label(), "Internal Error");
    }

    // ── Severity ────────────────────────────────────────────────────────────

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(Severity::Info.label(), "info");
        assert_eq!(Severity::Warning.label(), "warning");
        assert_eq!(Severity::Error.label(), "error");
        assert_eq!(Severity::Critical.label(), "critical");
    }

    #[test]
    fn test_severity_from_class() {
        assert_eq!(
            Severity::from_class(ErrorClass::Validation),
            Severity::Error
        );
        assert_eq!(
            Severity::from_class(ErrorClass::Internal),
            Severity::Critical
        );
        assert_eq!(
            Severity::from_class(ErrorClass::RateLimit),
            Severity::Warning
        );
    }

    // ── PathSegment ─────────────────────────────────────────────────────────

    #[test]
    fn test_path_segment_display() {
        assert_eq!(PathSegment::Field("user".into()).to_string(), "user");
        assert_eq!(PathSegment::Index(3).to_string(), "3");
    }

    #[test]
    fn test_format_path() {
        let path = vec![
            PathSegment::field("user"),
            PathSegment::field("friends"),
            PathSegment::index(0),
            PathSegment::field("name"),
        ];
        assert_eq!(format_path(&path), "user.friends.0.name");
    }

    #[test]
    fn test_format_path_empty() {
        assert_eq!(format_path(&[]), "");
    }

    // ── ErrorExtensions ─────────────────────────────────────────────────────

    #[test]
    fn test_extensions_builder() {
        let ext = ErrorExtensions::new("CUSTOM_CODE")
            .with_timestamp("2026-03-03T00:00:00Z")
            .with_trace_id("trace-abc-123")
            .with_extra("service", "gql-server");

        assert_eq!(ext.code, "CUSTOM_CODE");
        assert_eq!(ext.timestamp, Some("2026-03-03T00:00:00Z".to_string()));
        assert_eq!(ext.trace_id, Some("trace-abc-123".to_string()));
        assert_eq!(ext.extra.get("service"), Some(&"gql-server".to_string()));
    }

    // ── GraphQLError ────────────────────────────────────────────────────────

    #[test]
    fn test_error_basic() {
        let err = GraphQLError::new("field not found", ErrorClass::Validation);
        assert_eq!(err.message, "field not found");
        assert_eq!(err.classification, ErrorClass::Validation);
        assert_eq!(err.severity, Severity::Error);
        assert!(err.path.is_empty());
    }

    #[test]
    fn test_error_with_path() {
        let err = GraphQLError::new("null value", ErrorClass::Execution)
            .with_path(vec![PathSegment::field("user"), PathSegment::field("name")]);
        assert_eq!(err.path_string(), "user.name");
    }

    #[test]
    fn test_error_with_locations() {
        let err = GraphQLError::new("syntax error", ErrorClass::Validation)
            .with_locations(vec![SourceLocation::new(3, 10)]);
        assert_eq!(err.locations.len(), 1);
        assert_eq!(err.locations[0].line, 3);
        assert_eq!(err.locations[0].column, 10);
    }

    #[test]
    fn test_error_with_severity_override() {
        let err = GraphQLError::new("deprecated field", ErrorClass::Execution)
            .with_severity(Severity::Warning);
        assert_eq!(err.severity, Severity::Warning);
    }

    #[test]
    fn test_error_with_code() {
        let err = GraphQLError::new("test", ErrorClass::Internal).with_code("CUSTOM");
        assert!(err.extensions.is_some());
        let ext = err.extensions.as_ref().expect("should have extensions");
        assert_eq!(ext.code, "CUSTOM");
    }

    #[test]
    fn test_error_dedup_key() {
        let err1 = GraphQLError::new("bad field", ErrorClass::Validation)
            .with_path(vec![PathSegment::field("user")]);
        let err2 = GraphQLError::new("bad field", ErrorClass::Validation)
            .with_path(vec![PathSegment::field("user")]);
        assert_eq!(err1.dedup_key(), err2.dedup_key());
    }

    #[test]
    fn test_error_dedup_key_different() {
        let err1 = GraphQLError::new("bad field", ErrorClass::Validation);
        let err2 = GraphQLError::new("other error", ErrorClass::Validation);
        assert_ne!(err1.dedup_key(), err2.dedup_key());
    }

    // ── ErrorMessageRegistry ────────────────────────────────────────────────

    #[test]
    fn test_message_registry_register_and_get() {
        let mut reg = ErrorMessageRegistry::new();
        reg.register("AUTH_FAIL", "Please login to continue.");
        assert_eq!(reg.get("AUTH_FAIL"), Some("Please login to continue."));
        assert_eq!(reg.get("OTHER"), None);
    }

    #[test]
    fn test_message_registry_apply() {
        let mut reg = ErrorMessageRegistry::new();
        reg.register("CUSTOM", "A friendly error message");

        let mut err = GraphQLError::new("raw error", ErrorClass::Execution)
            .with_extensions(ErrorExtensions::new("CUSTOM"));
        reg.apply(&mut err);
        assert_eq!(err.message, "A friendly error message");
    }

    #[test]
    fn test_message_registry_no_match() {
        let mut reg = ErrorMessageRegistry::new();
        reg.register("OTHER", "something");

        let mut err = GraphQLError::new("original", ErrorClass::Execution)
            .with_extensions(ErrorExtensions::new("NOMATCH"));
        reg.apply(&mut err);
        assert_eq!(err.message, "original");
    }

    #[test]
    fn test_message_registry_counts() {
        let mut reg = ErrorMessageRegistry::new();
        assert!(reg.is_empty());
        reg.register("A", "msg A");
        reg.register("B", "msg B");
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_message_registry_default() {
        let reg = ErrorMessageRegistry::default();
        assert!(reg.is_empty());
    }

    // ── ErrorAggregator ─────────────────────────────────────────────────────

    #[test]
    fn test_aggregator_dedup() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("same error", ErrorClass::Execution));
        agg.add(GraphQLError::new("same error", ErrorClass::Execution));
        agg.add(GraphQLError::new("different", ErrorClass::Execution));

        assert_eq!(agg.unique_count(), 2);
        assert_eq!(agg.total_count(), 3);
    }

    #[test]
    fn test_aggregator_order_preserved() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("first", ErrorClass::Validation));
        agg.add(GraphQLError::new("second", ErrorClass::Execution));
        agg.add(GraphQLError::new("third", ErrorClass::Internal));

        let errors = agg.errors();
        assert_eq!(errors[0].error.message, "first");
        assert_eq!(errors[1].error.message, "second");
        assert_eq!(errors[2].error.message, "third");
    }

    #[test]
    fn test_aggregator_clear() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("e", ErrorClass::Execution));
        agg.clear();
        assert_eq!(agg.unique_count(), 0);
        assert_eq!(agg.total_count(), 0);
    }

    #[test]
    fn test_aggregator_into_errors() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("e1", ErrorClass::Execution));
        agg.add(GraphQLError::new("e2", ErrorClass::Execution));
        let errors = agg.into_errors();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_aggregator_severity_filter() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("info", ErrorClass::Execution).with_severity(Severity::Info));
        agg.add(GraphQLError::new("warn", ErrorClass::RateLimit).with_severity(Severity::Warning));
        agg.add(GraphQLError::new("err", ErrorClass::Execution));
        agg.add(GraphQLError::new("crit", ErrorClass::Internal));

        let critical = agg.errors_above_severity(Severity::Critical);
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].error.message, "crit");

        let warnings = agg.errors_above_severity(Severity::Warning);
        assert_eq!(warnings.len(), 3);
    }

    #[test]
    fn test_aggregator_default() {
        let agg = ErrorAggregator::default();
        assert_eq!(agg.unique_count(), 0);
    }

    // ── ErrorResponseBuilder ────────────────────────────────────────────────

    #[test]
    fn test_response_builder_single_error() {
        let mut builder = ErrorResponseBuilder::new();
        let err = GraphQLError::new("not found", ErrorClass::Execution)
            .with_path(vec![PathSegment::field("user")]);
        builder.add_error(&err);

        let response = builder.build();
        assert_eq!(response.errors.len(), 1);
        assert_eq!(response.errors[0].message, "not found");
        assert!(response.errors[0].path.is_some());
    }

    #[test]
    fn test_response_builder_with_timestamp() {
        let mut builder = ErrorResponseBuilder::new().with_timestamp("2026-03-03T00:00:00Z");
        builder.add_error(&GraphQLError::new("err", ErrorClass::Internal));

        let response = builder.build();
        let ext = response.errors[0].extensions.as_ref().expect("extensions");
        assert_eq!(
            ext.get("timestamp"),
            Some(&"2026-03-03T00:00:00Z".to_string())
        );
    }

    #[test]
    fn test_response_builder_with_trace_id() {
        let mut builder = ErrorResponseBuilder::new().with_trace_id("trace-42");
        builder.add_error(&GraphQLError::new("err", ErrorClass::Execution));

        let response = builder.build();
        let ext = response.errors[0].extensions.as_ref().expect("extensions");
        assert_eq!(ext.get("traceId"), Some(&"trace-42".to_string()));
    }

    #[test]
    fn test_response_builder_aggregated() {
        let mut agg = ErrorAggregator::new();
        agg.add(GraphQLError::new("dup", ErrorClass::Execution));
        agg.add(GraphQLError::new("dup", ErrorClass::Execution));

        let mut builder = ErrorResponseBuilder::new();
        builder.add_aggregated(&agg);

        let response = builder.build();
        assert_eq!(response.errors.len(), 1);
        let ext = response.errors[0].extensions.as_ref().expect("extensions");
        assert_eq!(ext.get("occurrences"), Some(&"2".to_string()));
    }

    #[test]
    fn test_response_builder_entry_count() {
        let mut builder = ErrorResponseBuilder::new();
        assert_eq!(builder.entry_count(), 0);
        builder.add_error(&GraphQLError::new("a", ErrorClass::Execution));
        builder.add_error(&GraphQLError::new("b", ErrorClass::Validation));
        assert_eq!(builder.entry_count(), 2);
    }

    #[test]
    fn test_response_builder_default() {
        let builder = ErrorResponseBuilder::default();
        assert_eq!(builder.entry_count(), 0);
    }

    // ── JSON formatting ─────────────────────────────────────────────────────

    #[test]
    fn test_format_response_json_basic() {
        let mut builder = ErrorResponseBuilder::new();
        builder.add_error(&GraphQLError::new("oops", ErrorClass::Execution));
        let response = builder.build();

        let json = format_response_json(&response);
        assert!(json.contains("\"errors\""));
        assert!(json.contains("oops"));
        assert!(json.contains("EXECUTION_ERROR"));
    }

    #[test]
    fn test_format_response_json_with_path() {
        let mut builder = ErrorResponseBuilder::new();
        builder.add_error(
            &GraphQLError::new("null", ErrorClass::Execution)
                .with_path(vec![PathSegment::field("user"), PathSegment::index(0)]),
        );
        let response = builder.build();

        let json = format_response_json(&response);
        assert!(json.contains("\"path\""));
        assert!(json.contains("\"user\""));
        assert!(json.contains("0"));
    }

    #[test]
    fn test_format_response_json_with_locations() {
        let mut builder = ErrorResponseBuilder::new();
        builder.add_error(
            &GraphQLError::new("parse error", ErrorClass::Validation)
                .with_locations(vec![SourceLocation::new(5, 12)]),
        );
        let response = builder.build();

        let json = format_response_json(&response);
        assert!(json.contains("\"locations\""));
        assert!(json.contains("\"line\":5"));
        assert!(json.contains("\"column\":12"));
    }

    #[test]
    fn test_format_response_json_empty() {
        let builder = ErrorResponseBuilder::new();
        let response = builder.build();
        let json = format_response_json(&response);
        assert_eq!(json, "{\"errors\":[]}");
    }

    #[test]
    fn test_format_response_json_multiple_errors() {
        let mut builder = ErrorResponseBuilder::new();
        builder.add_error(&GraphQLError::new("e1", ErrorClass::Execution));
        builder.add_error(&GraphQLError::new("e2", ErrorClass::Validation));
        let response = builder.build();

        let json = format_response_json(&response);
        assert!(json.contains("e1"));
        assert!(json.contains("e2"));
    }

    // ── json_escape ─────────────────────────────────────────────────────────

    #[test]
    fn test_json_escape_simple() {
        assert_eq!(json_escape("hello"), "\"hello\"");
    }

    #[test]
    fn test_json_escape_special_chars() {
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_escape("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_escape("a\nb"), "\"a\\nb\"");
        assert_eq!(json_escape("a\tb"), "\"a\\tb\"");
        assert_eq!(json_escape("a\rb"), "\"a\\rb\"");
    }

    // ── SourceLocation ──────────────────────────────────────────────────────

    #[test]
    fn test_source_location_new() {
        let loc = SourceLocation::new(10, 20);
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 20);
    }

    // ── GraphQLError extensions chain ───────────────────────────────────────

    #[test]
    fn test_error_extensions_override_builder_defaults() {
        let mut builder = ErrorResponseBuilder::new().with_timestamp("builder-ts");

        let err = GraphQLError::new("test", ErrorClass::Execution)
            .with_extensions(ErrorExtensions::new("CODE").with_timestamp("error-ts"));
        builder.add_error(&err);

        let response = builder.build();
        let ext = response.errors[0].extensions.as_ref().expect("extensions");
        // Error-level timestamp overrides builder-level
        assert_eq!(ext.get("timestamp"), Some(&"error-ts".to_string()));
    }

    #[test]
    fn test_error_path_string_empty() {
        let err = GraphQLError::new("e", ErrorClass::Execution);
        assert_eq!(err.path_string(), "");
    }

    // ── Integration: full pipeline ──────────────────────────────────────────

    #[test]
    fn test_full_pipeline_aggregation_to_response() {
        let mut agg = ErrorAggregator::new();

        // Add validation error
        agg.add(
            GraphQLError::new("field 'xyz' not found", ErrorClass::Validation)
                .with_path(vec![PathSegment::field("query"), PathSegment::field("xyz")])
                .with_locations(vec![SourceLocation::new(1, 5)])
                .with_extensions(ErrorExtensions::new("UNKNOWN_FIELD").with_trace_id("t-001")),
        );

        // Add execution error (duplicate)
        agg.add(
            GraphQLError::new("resolver failed", ErrorClass::Execution)
                .with_path(vec![PathSegment::field("user"), PathSegment::field("name")]),
        );
        agg.add(
            GraphQLError::new("resolver failed", ErrorClass::Execution)
                .with_path(vec![PathSegment::field("user"), PathSegment::field("name")]),
        );

        assert_eq!(agg.unique_count(), 2);
        assert_eq!(agg.total_count(), 3);

        let mut builder = ErrorResponseBuilder::new()
            .with_timestamp("2026-03-03T12:00:00Z")
            .with_trace_id("global-trace");

        builder.add_aggregated(&agg);
        let response = builder.build();

        assert_eq!(response.errors.len(), 2);

        // First error should have trace-id from extensions (overriding global)
        let ext0 = response.errors[0].extensions.as_ref().expect("extensions");
        assert_eq!(ext0.get("traceId"), Some(&"t-001".to_string()));

        // Second error should have duplicate count
        let ext1 = response.errors[1].extensions.as_ref().expect("extensions");
        assert_eq!(ext1.get("occurrences"), Some(&"2".to_string()));

        let json = format_response_json(&response);
        assert!(json.contains("UNKNOWN_FIELD"));
        assert!(json.contains("resolver failed"));
    }
}
