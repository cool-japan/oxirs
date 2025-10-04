//! Request Logging Handler
//!
//! Provides structured logging for HTTP requests and SPARQL queries.
//! Based on Apache Jena Fuseki's request logging system.
//!
//! GET /$/logs - Retrieve recent request logs
//! GET /$/logs/statistics - Get logging statistics

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tracing::{debug, info};

/// Log entry for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unique request ID
    pub request_id: String,

    /// Timestamp when request started
    pub timestamp: SystemTime,

    /// HTTP method
    pub method: String,

    /// Request path
    pub path: String,

    /// Query string (if any)
    pub query_string: Option<String>,

    /// Remote client address
    pub remote_addr: Option<String>,

    /// User agent
    pub user_agent: Option<String>,

    /// Request content type
    pub content_type: Option<String>,

    /// Request body size in bytes
    pub request_size: Option<usize>,

    /// HTTP status code
    pub status_code: u16,

    /// Response body size in bytes
    pub response_size: Option<usize>,

    /// Request duration in milliseconds
    pub duration_ms: u64,

    /// SPARQL query (if applicable)
    pub sparql_query: Option<String>,

    /// Operation type (query, update, upload, etc.)
    pub operation_type: Option<String>,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl LogEntry {
    /// Create new log entry
    pub fn new(request_id: String, method: String, path: String) -> Self {
        Self {
            request_id,
            timestamp: SystemTime::now(),
            method,
            path,
            query_string: None,
            remote_addr: None,
            user_agent: None,
            content_type: None,
            request_size: None,
            status_code: 200,
            response_size: None,
            duration_ms: 0,
            sparql_query: None,
            operation_type: None,
            error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Format log entry as JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format log entry as text string
    pub fn to_text(&self) -> String {
        format!(
            "{} {} {} {} {}ms [{}]",
            self.request_id,
            self.method,
            self.path,
            self.status_code,
            self.duration_ms,
            self.timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        )
    }

    /// Check if request was successful (2xx status)
    pub fn is_success(&self) -> bool {
        self.status_code >= 200 && self.status_code < 300
    }

    /// Check if request had error (4xx or 5xx status)
    pub fn is_error(&self) -> bool {
        self.status_code >= 400
    }
}

/// Request logger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggerConfig {
    /// Enable request logging
    pub enabled: bool,

    /// Maximum number of log entries to keep in memory
    pub max_entries: usize,

    /// Log format (json or text)
    pub format: LogFormat,

    /// Minimum duration in ms to log slow queries
    pub slow_query_threshold_ms: Option<u64>,

    /// Log request bodies
    pub log_request_body: bool,

    /// Log response bodies
    pub log_response_body: bool,

    /// Log SPARQL queries
    pub log_sparql: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            format: LogFormat::Json,
            slow_query_threshold_ms: Some(1000),
            log_request_body: false,
            log_response_body: false,
            log_sparql: true,
        }
    }
}

/// Log output format
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Text,
}

/// Request logger for tracking HTTP requests
#[derive(Clone)]
pub struct RequestLogger {
    config: Arc<RwLock<LoggerConfig>>,
    entries: Arc<RwLock<VecDeque<LogEntry>>>,
    statistics: Arc<RwLock<LogStatistics>>,
}

impl RequestLogger {
    /// Create new request logger with default config
    pub fn new() -> Self {
        Self::with_config(LoggerConfig::default())
    }

    /// Create request logger with custom config
    pub fn with_config(config: LoggerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            entries: Arc::new(RwLock::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(LogStatistics::default())),
        }
    }

    /// Log a request
    pub fn log_request(&self, entry: LogEntry) -> Result<(), LogError> {
        let config = self
            .config
            .read()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        if !config.enabled {
            return Ok(());
        }

        // Check slow query threshold
        if let Some(threshold) = config.slow_query_threshold_ms {
            if entry.duration_ms >= threshold {
                info!(
                    "Slow request: {} {} {}ms",
                    entry.method, entry.path, entry.duration_ms
                );
            }
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

            stats.total_requests += 1;

            if entry.is_success() {
                stats.successful_requests += 1;
            } else if entry.is_error() {
                stats.failed_requests += 1;
            }

            stats.total_duration_ms += entry.duration_ms;
            if entry.duration_ms > stats.max_duration_ms {
                stats.max_duration_ms = entry.duration_ms;
            }
            if entry.duration_ms < stats.min_duration_ms || stats.min_duration_ms == 0 {
                stats.min_duration_ms = entry.duration_ms;
            }

            if let Some(size) = entry.request_size {
                stats.total_request_bytes += size;
            }
            if let Some(size) = entry.response_size {
                stats.total_response_bytes += size;
            }
        }

        // Store entry
        let mut entries = self
            .entries
            .write()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        // Remove oldest if at capacity
        if entries.len() >= config.max_entries {
            entries.pop_front();
        }

        // Format and log
        match config.format {
            LogFormat::Json => debug!("Request: {}", entry.to_json()),
            LogFormat::Text => debug!("Request: {}", entry.to_text()),
        }

        entries.push_back(entry);

        Ok(())
    }

    /// Get recent log entries
    pub fn get_logs(
        &self,
        limit: Option<usize>,
        filter: Option<LogFilter>,
    ) -> Result<Vec<LogEntry>, LogError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        let mut logs: Vec<LogEntry> = entries.iter().cloned().collect();

        // Apply filters
        if let Some(filter) = filter {
            if let Some(method) = filter.method {
                logs.retain(|e| e.method.eq_ignore_ascii_case(&method));
            }
            if let Some(min_duration) = filter.min_duration_ms {
                logs.retain(|e| e.duration_ms >= min_duration);
            }
            if let Some(status_code) = filter.status_code {
                logs.retain(|e| e.status_code == status_code);
            }
            if filter.errors_only.unwrap_or(false) {
                logs.retain(|e| e.is_error());
            }
        }

        // Apply limit (take most recent)
        let limit = limit.unwrap_or(logs.len());
        logs.reverse(); // Most recent first
        logs.truncate(limit);

        Ok(logs)
    }

    /// Get logging statistics
    pub fn get_statistics(&self) -> Result<LogStatistics, LogError> {
        let stats = self
            .statistics
            .read()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        Ok(stats.clone())
    }

    /// Clear all log entries
    pub fn clear_logs(&self) -> Result<(), LogError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        entries.clear();
        info!("Cleared all log entries");

        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> Result<LoggerConfig, LogError> {
        let config = self
            .config
            .read()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        Ok(config.clone())
    }

    /// Update configuration
    pub fn update_config(&self, new_config: LoggerConfig) -> Result<(), LogError> {
        let mut config = self
            .config
            .write()
            .map_err(|e| LogError::Internal(format!("Lock error: {}", e)))?;

        *config = new_config;
        info!("Updated logger configuration");

        Ok(())
    }
}

impl Default for RequestLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Logging statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStatistics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration_ms: u64,
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,
    pub total_request_bytes: usize,
    pub total_response_bytes: usize,
}

impl Default for LogStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration_ms: 0,
            min_duration_ms: 0,
            max_duration_ms: 0,
            total_request_bytes: 0,
            total_response_bytes: 0,
        }
    }
}

impl LogStatistics {
    /// Calculate average request duration
    pub fn avg_duration_ms(&self) -> f64 {
        if self.total_requests > 0 {
            self.total_duration_ms as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }

    /// Calculate success rate percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Log filter criteria
#[derive(Debug, Clone, Deserialize)]
pub struct LogFilter {
    pub method: Option<String>,
    pub min_duration_ms: Option<u64>,
    pub status_code: Option<u16>,
    pub errors_only: Option<bool>,
}

/// Log error types
#[derive(Debug, thiserror::Error)]
pub enum LogError {
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid filter: {0}")]
    InvalidFilter(String),
}

impl LogError {
    fn status_code(&self) -> StatusCode {
        match self {
            LogError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            LogError::InvalidFilter(_) => StatusCode::BAD_REQUEST,
        }
    }
}

impl IntoResponse for LogError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();

        (
            status,
            Json(serde_json::json!({
                "error": message,
                "status": status.as_u16(),
            })),
        )
            .into_response()
    }
}

/// Query parameters for log retrieval
#[derive(Debug, Clone, Deserialize)]
pub struct LogQuery {
    pub limit: Option<usize>,
    pub method: Option<String>,
    pub min_duration_ms: Option<u64>,
    pub status_code: Option<u16>,
    pub errors_only: Option<bool>,
}

impl From<LogQuery> for Option<LogFilter> {
    fn from(query: LogQuery) -> Self {
        if query.method.is_none()
            && query.min_duration_ms.is_none()
            && query.status_code.is_none()
            && query.errors_only.is_none()
        {
            None
        } else {
            Some(LogFilter {
                method: query.method,
                min_duration_ms: query.min_duration_ms,
                status_code: query.status_code,
                errors_only: query.errors_only,
            })
        }
    }
}

/// Get recent request logs
///
/// GET /$/logs?limit=100&errors_only=true
pub async fn get_logs(
    Query(params): Query<LogQuery>,
    State(logger): State<Arc<RequestLogger>>,
) -> Result<Response, LogError> {
    info!("Get logs request (limit: {:?})", params.limit);

    let filter = params.clone().into();
    let logs = logger.get_logs(params.limit, filter)?;

    debug!("Returning {} log entries", logs.len());

    Ok((StatusCode::OK, Json(logs)).into_response())
}

/// Get logging statistics
///
/// GET /$/logs/statistics
pub async fn get_log_statistics(
    State(logger): State<Arc<RequestLogger>>,
) -> Result<Response, LogError> {
    info!("Get log statistics request");

    let stats = logger.get_statistics()?;

    debug!("Log statistics: {:?}", stats);

    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Clear all logs
///
/// DELETE /$/logs
pub async fn clear_logs(State(logger): State<Arc<RequestLogger>>) -> Result<Response, LogError> {
    info!("Clear logs request");

    logger.clear_logs()?;

    Ok((StatusCode::NO_CONTENT, ()).into_response())
}

/// Get logger configuration
///
/// GET /$/logs/config
pub async fn get_log_config(
    State(logger): State<Arc<RequestLogger>>,
) -> Result<Response, LogError> {
    info!("Get log config request");

    let config = logger.get_config()?;

    Ok((StatusCode::OK, Json(config)).into_response())
}

/// Update logger configuration
///
/// PUT /$/logs/config
pub async fn update_log_config(
    State(logger): State<Arc<RequestLogger>>,
    Json(config): Json<LoggerConfig>,
) -> Result<Response, LogError> {
    info!("Update log config request");

    logger.update_config(config.clone())?;

    debug!("Updated config: {:?}", config);

    Ok((StatusCode::OK, Json(config)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new(
            "req-123".to_string(),
            "GET".to_string(),
            "/query".to_string(),
        );

        assert_eq!(entry.request_id, "req-123");
        assert_eq!(entry.method, "GET");
        assert_eq!(entry.path, "/query");
        assert_eq!(entry.status_code, 200);
    }

    #[test]
    fn test_log_entry_success_check() {
        let mut entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/".to_string());

        entry.status_code = 200;
        assert!(entry.is_success());
        assert!(!entry.is_error());

        entry.status_code = 404;
        assert!(!entry.is_success());
        assert!(entry.is_error());

        entry.status_code = 500;
        assert!(!entry.is_success());
        assert!(entry.is_error());
    }

    #[tokio::test]
    async fn test_logger_creation() {
        let logger = RequestLogger::new();
        let config = logger.get_config().unwrap();

        assert!(config.enabled);
        assert_eq!(config.format, LogFormat::Json);
    }

    #[tokio::test]
    async fn test_log_request() {
        let logger = RequestLogger::new();

        let entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());

        logger.log_request(entry).unwrap();

        let logs = logger.get_logs(None, None).unwrap();
        assert_eq!(logs.len(), 1);
    }

    #[tokio::test]
    async fn test_log_statistics() {
        let logger = RequestLogger::new();

        // Log successful request
        let mut entry1 =
            LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
        entry1.duration_ms = 100;
        entry1.status_code = 200;
        logger.log_request(entry1).unwrap();

        // Log failed request
        let mut entry2 = LogEntry::new(
            "req-2".to_string(),
            "POST".to_string(),
            "/update".to_string(),
        );
        entry2.duration_ms = 50;
        entry2.status_code = 500;
        logger.log_request(entry2).unwrap();

        let stats = logger.get_statistics().unwrap();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.total_duration_ms, 150);
        assert_eq!(stats.max_duration_ms, 100);
        assert_eq!(stats.min_duration_ms, 50);
    }

    #[tokio::test]
    async fn test_log_limit() {
        let config = LoggerConfig {
            max_entries: 5,
            ..Default::default()
        };
        let logger = RequestLogger::with_config(config);

        // Log 10 entries
        for i in 0..10 {
            let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
            logger.log_request(entry).unwrap();
        }

        let logs = logger.get_logs(None, None).unwrap();
        assert_eq!(logs.len(), 5); // Should only keep last 5
    }

    #[tokio::test]
    async fn test_log_filtering() {
        let logger = RequestLogger::new();

        // Log various requests
        let mut entry1 =
            LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
        entry1.duration_ms = 100;
        logger.log_request(entry1).unwrap();

        let mut entry2 = LogEntry::new(
            "req-2".to_string(),
            "POST".to_string(),
            "/update".to_string(),
        );
        entry2.duration_ms = 500;
        logger.log_request(entry2).unwrap();

        let mut entry3 =
            LogEntry::new("req-3".to_string(), "GET".to_string(), "/query".to_string());
        entry3.duration_ms = 50;
        logger.log_request(entry3).unwrap();

        // Filter by method
        let filter = LogFilter {
            method: Some("GET".to_string()),
            min_duration_ms: None,
            status_code: None,
            errors_only: None,
        };
        let logs = logger.get_logs(None, Some(filter)).unwrap();
        assert_eq!(logs.len(), 2);

        // Filter by duration
        let filter = LogFilter {
            method: None,
            min_duration_ms: Some(100),
            status_code: None,
            errors_only: None,
        };
        let logs = logger.get_logs(None, Some(filter)).unwrap();
        assert_eq!(logs.len(), 2);
    }

    #[tokio::test]
    async fn test_clear_logs() {
        let logger = RequestLogger::new();

        // Log some entries
        for i in 0..5 {
            let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
            logger.log_request(entry).unwrap();
        }

        assert_eq!(logger.get_logs(None, None).unwrap().len(), 5);

        logger.clear_logs().unwrap();

        assert_eq!(logger.get_logs(None, None).unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_statistics_calculations() {
        let logger = RequestLogger::new();

        // Log requests with known durations
        for i in 0..10 {
            let mut entry =
                LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
            entry.duration_ms = 100;
            entry.status_code = if i < 8 { 200 } else { 500 };
            logger.log_request(entry).unwrap();
        }

        let stats = logger.get_statistics().unwrap();
        assert_eq!(stats.avg_duration_ms(), 100.0);
        assert_eq!(stats.success_rate(), 80.0);
    }
}
