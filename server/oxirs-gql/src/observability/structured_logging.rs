//! Structured Logging with Query Context
//!
//! Provides structured logging capabilities with rich GraphQL query context
//! for improved debugging and monitoring.
//!
//! # Features
//!
//! - Structured JSON log output
//! - Rich query context (operation, variables, fields)
//! - Request ID tracking across the stack
//! - User and client identification
//! - Performance metrics in logs
//! - Error context and stack traces
//! - Configurable log levels
//! - Log filtering and sampling

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Query context for logging
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub operation_name: Option<String>,
    pub operation_type: Option<String>, // query, mutation, subscription
    pub query_hash: Option<String>,
    pub variables: HashMap<String, String>,
    pub requested_fields: Vec<String>,
    pub query_depth: usize,
    pub query_complexity: Option<f64>,
}

/// Request context for logging
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub client_name: Option<String>,
    pub client_version: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
}

/// Performance metrics for logging
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub duration_ms: f64,
    pub resolver_count: usize,
    pub database_queries: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub bytes_sent: usize,
    pub bytes_received: usize,
}

/// Error context for logging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error_message: String,
    pub error_type: String,
    pub error_code: Option<String>,
    pub stack_trace: Option<String>,
    pub field_path: Option<Vec<String>>,
    pub recoverable: bool,
}

/// Structured log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub request_context: Option<RequestContext>,
    pub query_context: Option<QueryContext>,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub error_context: Option<ErrorContext>,
    pub custom_fields: HashMap<String, String>,
}

impl LogEntry {
    /// Convert log entry to JSON string
    pub fn to_json(&self) -> String {
        let mut json = String::from("{");

        // Timestamp
        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        json.push_str(&format!("\"timestamp\":{},", timestamp));

        // Level
        json.push_str(&format!("\"level\":\"{}\",", self.level));

        // Message
        json.push_str(&format!(
            "\"message\":{},",
            Self::escape_json(&self.message)
        ));

        // Request context
        if let Some(ref ctx) = self.request_context {
            json.push_str("\"request\":{");
            json.push_str(&format!("\"id\":{},", Self::escape_json(&ctx.request_id)));
            if let Some(ref user_id) = ctx.user_id {
                json.push_str(&format!("\"user_id\":{},", Self::escape_json(user_id)));
            }
            if let Some(ref client) = ctx.client_name {
                json.push_str(&format!("\"client\":{},", Self::escape_json(client)));
            }
            if let Some(ref trace_id) = ctx.trace_id {
                json.push_str(&format!("\"trace_id\":{},", Self::escape_json(trace_id)));
            }
            if let Some(ref span_id) = ctx.span_id {
                json.push_str(&format!("\"span_id\":{},", Self::escape_json(span_id)));
            }
            // Remove trailing comma
            if json.ends_with(',') {
                json.pop();
            }
            json.push_str("},");
        }

        // Query context
        if let Some(ref ctx) = self.query_context {
            json.push_str("\"query\":{");
            if let Some(ref op_name) = ctx.operation_name {
                json.push_str(&format!("\"operation\":{},", Self::escape_json(op_name)));
            }
            if let Some(ref op_type) = ctx.operation_type {
                json.push_str(&format!("\"type\":{},", Self::escape_json(op_type)));
            }
            json.push_str(&format!("\"depth\":{},", ctx.query_depth));
            if let Some(complexity) = ctx.query_complexity {
                json.push_str(&format!("\"complexity\":{},", complexity));
            }
            // Remove trailing comma
            if json.ends_with(',') {
                json.pop();
            }
            json.push_str("},");
        }

        // Performance metrics
        if let Some(ref metrics) = self.performance_metrics {
            json.push_str("\"performance\":{");
            json.push_str(&format!("\"duration_ms\":{},", metrics.duration_ms));
            json.push_str(&format!("\"resolvers\":{},", metrics.resolver_count));
            json.push_str(&format!("\"db_queries\":{},", metrics.database_queries));
            json.push_str(&format!("\"cache_hits\":{},", metrics.cache_hits));
            json.push_str(&format!("\"cache_misses\":{},", metrics.cache_misses));
            // Remove trailing comma
            if json.ends_with(',') {
                json.pop();
            }
            json.push_str("},");
        }

        // Error context
        if let Some(ref err) = self.error_context {
            json.push_str("\"error\":{");
            json.push_str(&format!(
                "\"message\":{},",
                Self::escape_json(&err.error_message)
            ));
            json.push_str(&format!("\"type\":{},", Self::escape_json(&err.error_type)));
            if let Some(ref code) = err.error_code {
                json.push_str(&format!("\"code\":{},", Self::escape_json(code)));
            }
            json.push_str(&format!("\"recoverable\":{},", err.recoverable));
            // Remove trailing comma
            if json.ends_with(',') {
                json.pop();
            }
            json.push_str("},");
        }

        // Custom fields
        if !self.custom_fields.is_empty() {
            json.push_str("\"custom\":{");
            for (key, value) in &self.custom_fields {
                json.push_str(&format!("\"{}\":{},", key, Self::escape_json(value)));
            }
            // Remove trailing comma
            if json.ends_with(',') {
                json.pop();
            }
            json.push_str("},");
        }

        // Remove trailing comma and close
        if json.ends_with(',') {
            json.pop();
        }
        json.push('}');

        json
    }

    /// Escape string for JSON
    fn escape_json(s: &str) -> String {
        let escaped = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        format!("\"{}\"", escaped)
    }
}

/// Structured logger configuration
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    pub min_level: LogLevel,
    pub include_timestamp: bool,
    pub include_request_context: bool,
    pub include_query_context: bool,
    pub include_performance: bool,
    pub sanitize_variables: bool,
    pub max_field_count: usize,
    pub sample_rate: f64, // 0.0 to 1.0
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            include_timestamp: true,
            include_request_context: true,
            include_query_context: true,
            include_performance: true,
            sanitize_variables: true,
            max_field_count: 100,
            sample_rate: 1.0,
        }
    }
}

/// Structured logger
pub struct StructuredLogger {
    config: LoggerConfig,
    entries: Arc<Mutex<Vec<LogEntry>>>,
    max_entries: usize,
}

impl StructuredLogger {
    /// Create a new structured logger
    pub fn new(config: LoggerConfig) -> Self {
        Self {
            config,
            entries: Arc::new(Mutex::new(Vec::new())),
            max_entries: 10_000,
        }
    }

    /// Log an entry
    pub fn log(&self, mut entry: LogEntry) {
        // Check log level
        if entry.level < self.config.min_level {
            return;
        }

        // Check sampling
        if self.config.sample_rate < 1.0 {
            let random_value = fastrand::f64();
            if random_value >= self.config.sample_rate {
                return;
            }
        }

        // Sanitize if needed
        if self.config.sanitize_variables {
            if let Some(ref mut query_ctx) = entry.query_context {
                for (key, value) in query_ctx.variables.iter_mut() {
                    if Self::is_sensitive_field(key) {
                        *value = "[REDACTED]".to_string();
                    }
                }
            }
        }

        // Limit field count
        if let Some(ref mut query_ctx) = entry.query_context {
            if query_ctx.requested_fields.len() > self.config.max_field_count {
                query_ctx
                    .requested_fields
                    .truncate(self.config.max_field_count);
            }
        }

        // Store entry
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);

        // Trim if too many entries
        if entries.len() > self.max_entries {
            let excess = entries.len() - self.max_entries;
            entries.drain(0..excess);
        }
    }

    /// Log with trace level
    pub fn trace(&self, message: String) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Trace,
            message,
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        });
    }

    /// Log with debug level
    pub fn debug(&self, message: String) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Debug,
            message,
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        });
    }

    /// Log with info level
    pub fn info(&self, message: String) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Info,
            message,
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        });
    }

    /// Log with warn level
    pub fn warn(&self, message: String) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Warn,
            message,
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        });
    }

    /// Log with error level
    pub fn error(&self, message: String, error_ctx: Option<ErrorContext>) {
        self.log(LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Error,
            message,
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: error_ctx,
            custom_fields: HashMap::new(),
        });
    }

    /// Get recent log entries
    pub fn get_recent_entries(&self, count: usize) -> Vec<LogEntry> {
        let entries = self.entries.lock().unwrap();
        let start = if entries.len() > count {
            entries.len() - count
        } else {
            0
        };
        entries[start..].to_vec()
    }

    /// Get entries by level
    pub fn get_entries_by_level(&self, level: LogLevel) -> Vec<LogEntry> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }

    /// Get entries by request ID
    pub fn get_entries_by_request(&self, request_id: &str) -> Vec<LogEntry> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| {
                e.request_context
                    .as_ref()
                    .map(|ctx| ctx.request_id == request_id)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Export logs as JSON array
    pub fn export_json(&self) -> String {
        let entries = self.entries.lock().unwrap();
        let json_entries: Vec<String> = entries.iter().map(|e| e.to_json()).collect();
        format!("[{}]", json_entries.join(","))
    }

    /// Clear all log entries
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// Get total log count
    pub fn count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Get count by level
    pub fn count_by_level(&self, level: LogLevel) -> usize {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.level == level)
            .count()
    }

    // Helper methods

    fn is_sensitive_field(field_name: &str) -> bool {
        let sensitive_patterns = [
            "password",
            "token",
            "secret",
            "api_key",
            "apikey",
            "auth",
            "credential",
            "private",
        ];

        let lower_name = field_name.to_lowercase();
        sensitive_patterns
            .iter()
            .any(|pattern| lower_name.contains(pattern))
    }
}

/// Log entry builder for fluent API
pub struct LogEntryBuilder {
    entry: LogEntry,
}

impl LogEntryBuilder {
    pub fn new(level: LogLevel, message: String) -> Self {
        Self {
            entry: LogEntry {
                timestamp: SystemTime::now(),
                level,
                message,
                request_context: None,
                query_context: None,
                performance_metrics: None,
                error_context: None,
                custom_fields: HashMap::new(),
            },
        }
    }

    pub fn with_request_context(mut self, ctx: RequestContext) -> Self {
        self.entry.request_context = Some(ctx);
        self
    }

    pub fn with_query_context(mut self, ctx: QueryContext) -> Self {
        self.entry.query_context = Some(ctx);
        self
    }

    pub fn with_performance(mut self, metrics: PerformanceMetrics) -> Self {
        self.entry.performance_metrics = Some(metrics);
        self
    }

    pub fn with_error(mut self, error: ErrorContext) -> Self {
        self.entry.error_context = Some(error);
        self
    }

    pub fn with_custom_field(mut self, key: String, value: String) -> Self {
        self.entry.custom_fields.insert(key, value);
        self
    }

    pub fn build(self) -> LogEntry {
        self.entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_basic_logging() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.info("Test message".to_string());

        assert_eq!(logger.count(), 1);
        let entries = logger.get_recent_entries(1);
        assert_eq!(entries[0].level, LogLevel::Info);
        assert_eq!(entries[0].message, "Test message");
    }

    #[test]
    fn test_log_level_filtering() {
        let config = LoggerConfig {
            min_level: LogLevel::Warn,
            ..Default::default()
        };
        let logger = StructuredLogger::new(config);

        logger.debug("Debug message".to_string());
        logger.info("Info message".to_string());
        logger.warn("Warn message".to_string());
        logger.error("Error message".to_string(), None);

        assert_eq!(logger.count(), 2); // Only warn and error
    }

    #[test]
    fn test_log_with_request_context() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        let entry = LogEntryBuilder::new(LogLevel::Info, "Request logged".to_string())
            .with_request_context(RequestContext {
                request_id: "req-123".to_string(),
                user_id: Some("user-456".to_string()),
                client_name: Some("mobile-app".to_string()),
                client_version: Some("1.0.0".to_string()),
                ip_address: None,
                user_agent: None,
                trace_id: Some("trace-789".to_string()),
                span_id: Some("span-abc".to_string()),
            })
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        assert!(entries[0].request_context.is_some());
        assert_eq!(
            entries[0].request_context.as_ref().unwrap().request_id,
            "req-123"
        );
    }

    #[test]
    fn test_log_with_query_context() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        let entry = LogEntryBuilder::new(LogLevel::Info, "Query executed".to_string())
            .with_query_context(QueryContext {
                operation_name: Some("GetUser".to_string()),
                operation_type: Some("query".to_string()),
                query_hash: None,
                variables: HashMap::new(),
                requested_fields: vec!["id".to_string(), "name".to_string()],
                query_depth: 2,
                query_complexity: Some(10.5),
            })
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        assert!(entries[0].query_context.is_some());
        assert_eq!(
            entries[0].query_context.as_ref().unwrap().operation_name,
            Some("GetUser".to_string())
        );
    }

    #[test]
    fn test_log_with_performance_metrics() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        let entry = LogEntryBuilder::new(LogLevel::Info, "Request completed".to_string())
            .with_performance(PerformanceMetrics {
                duration_ms: 123.45,
                resolver_count: 10,
                database_queries: 5,
                cache_hits: 3,
                cache_misses: 2,
                bytes_sent: 1024,
                bytes_received: 512,
            })
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        assert!(entries[0].performance_metrics.is_some());
        assert_eq!(
            entries[0].performance_metrics.as_ref().unwrap().duration_ms,
            123.45
        );
    }

    #[test]
    fn test_log_with_error_context() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.error(
            "Query failed".to_string(),
            Some(ErrorContext {
                error_message: "Invalid input".to_string(),
                error_type: "ValidationError".to_string(),
                error_code: Some("E001".to_string()),
                stack_trace: None,
                field_path: Some(vec!["user".to_string(), "email".to_string()]),
                recoverable: true,
            }),
        );

        let entries = logger.get_recent_entries(1);
        assert!(entries[0].error_context.is_some());
        assert_eq!(
            entries[0].error_context.as_ref().unwrap().error_message,
            "Invalid input"
        );
    }

    #[test]
    fn test_json_export() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.info("Test message".to_string());

        let json = logger.export_json();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("Test message"));
    }

    #[test]
    fn test_log_entry_to_json() {
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Info,
            message: "Test".to_string(),
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        };

        let json = entry.to_json();
        assert!(json.contains("\"level\":\"INFO\""));
        assert!(json.contains("\"message\":\"Test\""));
    }

    #[test]
    fn test_get_entries_by_request() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        let entry1 = LogEntryBuilder::new(LogLevel::Info, "Request 1".to_string())
            .with_request_context(RequestContext {
                request_id: "req-1".to_string(),
                user_id: None,
                client_name: None,
                client_version: None,
                ip_address: None,
                user_agent: None,
                trace_id: None,
                span_id: None,
            })
            .build();

        let entry2 = LogEntryBuilder::new(LogLevel::Info, "Request 2".to_string())
            .with_request_context(RequestContext {
                request_id: "req-2".to_string(),
                user_id: None,
                client_name: None,
                client_version: None,
                ip_address: None,
                user_agent: None,
                trace_id: None,
                span_id: None,
            })
            .build();

        logger.log(entry1);
        logger.log(entry2);

        let entries = logger.get_entries_by_request("req-1");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Request 1");
    }

    #[test]
    fn test_count_by_level() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.info("Info 1".to_string());
        logger.info("Info 2".to_string());
        logger.warn("Warn 1".to_string());
        logger.error("Error 1".to_string(), None);

        assert_eq!(logger.count_by_level(LogLevel::Info), 2);
        assert_eq!(logger.count_by_level(LogLevel::Warn), 1);
        assert_eq!(logger.count_by_level(LogLevel::Error), 1);
    }

    #[test]
    fn test_clear_logs() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.info("Test".to_string());
        assert_eq!(logger.count(), 1);

        logger.clear();
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_sensitive_field_detection() {
        assert!(StructuredLogger::is_sensitive_field("password"));
        assert!(StructuredLogger::is_sensitive_field("api_key"));
        assert!(StructuredLogger::is_sensitive_field("auth_token"));
        assert!(!StructuredLogger::is_sensitive_field("username"));
    }

    #[test]
    fn test_variable_sanitization() {
        let config = LoggerConfig {
            sanitize_variables: true,
            ..Default::default()
        };
        let logger = StructuredLogger::new(config);

        let mut variables = HashMap::new();
        variables.insert("username".to_string(), "john".to_string());
        variables.insert("password".to_string(), "secret123".to_string());

        let entry = LogEntryBuilder::new(LogLevel::Info, "Login attempt".to_string())
            .with_query_context(QueryContext {
                operation_name: Some("login".to_string()),
                operation_type: Some("mutation".to_string()),
                query_hash: None,
                variables,
                requested_fields: vec![],
                query_depth: 1,
                query_complexity: None,
            })
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        let vars = &entries[0].query_context.as_ref().unwrap().variables;
        assert_eq!(vars.get("username").unwrap(), "john");
        assert_eq!(vars.get("password").unwrap(), "[REDACTED]");
    }

    #[test]
    fn test_log_sampling() {
        let config = LoggerConfig {
            sample_rate: 0.0, // Log nothing
            ..Default::default()
        };
        let logger = StructuredLogger::new(config);

        for _ in 0..100 {
            logger.info("Test".to_string());
        }

        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_custom_fields() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        let entry = LogEntryBuilder::new(LogLevel::Info, "Custom".to_string())
            .with_custom_field("environment".to_string(), "production".to_string())
            .with_custom_field("region".to_string(), "us-west-2".to_string())
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        assert_eq!(entries[0].custom_fields.len(), 2);
        assert_eq!(
            entries[0].custom_fields.get("environment").unwrap(),
            "production"
        );
    }

    #[test]
    fn test_max_entries_limit() {
        let mut logger = StructuredLogger::new(LoggerConfig::default());
        logger.max_entries = 10;

        for i in 0..20 {
            logger.info(format!("Message {}", i));
        }

        assert_eq!(logger.count(), 10);
    }

    #[test]
    fn test_json_escaping() {
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Info,
            message: "Test \"quote\" and\nnewline".to_string(),
            request_context: None,
            query_context: None,
            performance_metrics: None,
            error_context: None,
            custom_fields: HashMap::new(),
        };

        let json = entry.to_json();
        assert!(json.contains("\\\"quote\\\""));
        assert!(json.contains("\\n"));
    }

    #[test]
    fn test_get_entries_by_level() {
        let logger = StructuredLogger::new(LoggerConfig::default());

        logger.info("Info 1".to_string());
        logger.warn("Warn 1".to_string());
        logger.info("Info 2".to_string());

        let info_entries = logger.get_entries_by_level(LogLevel::Info);
        assert_eq!(info_entries.len(), 2);

        let warn_entries = logger.get_entries_by_level(LogLevel::Warn);
        assert_eq!(warn_entries.len(), 1);
    }

    #[test]
    fn test_field_count_limit() {
        let config = LoggerConfig {
            max_field_count: 3,
            ..Default::default()
        };
        let logger = StructuredLogger::new(config);

        let entry = LogEntryBuilder::new(LogLevel::Info, "Test".to_string())
            .with_query_context(QueryContext {
                operation_name: None,
                operation_type: None,
                query_hash: None,
                variables: HashMap::new(),
                requested_fields: vec![
                    "field1".to_string(),
                    "field2".to_string(),
                    "field3".to_string(),
                    "field4".to_string(),
                    "field5".to_string(),
                ],
                query_depth: 1,
                query_complexity: None,
            })
            .build();

        logger.log(entry);

        let entries = logger.get_recent_entries(1);
        let fields = &entries[0].query_context.as_ref().unwrap().requested_fields;
        assert_eq!(fields.len(), 3); // Truncated to max_field_count
    }
}
