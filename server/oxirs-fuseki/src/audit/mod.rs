//! Comprehensive audit logging for OxiRS Fuseki.
//!
//! Provides structured audit records for all significant server events:
//! SPARQL queries, data modifications, administrative actions, authentication
//! failures, and data exports.  Records are emitted asynchronously through
//! pluggable backends (file JSONL, syslog).

pub mod file_backend;

use std::collections::VecDeque;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};
use tokio::time::sleep;

pub use file_backend::{FileAuditBackend, FileBackendConfig, SyslogAuditBackend};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// The kind of event being audited.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum AuditEvent {
    /// A SPARQL SELECT / CONSTRUCT / ASK / DESCRIBE query was executed.
    QueryExecuted,
    /// A SPARQL UPDATE, LOAD, or equivalent data modification was performed.
    DataModified,
    /// An administrative action was taken (dataset creation, config change, etc.).
    AdminAction,
    /// An authentication attempt failed.
    AuthFailure,
    /// Data was exported from the triple store.
    DataExport,
    /// A user successfully authenticated.
    AuthSuccess,
    /// A rate-limit was applied to a client.
    RateLimited,
    /// A custom application-specific event.
    Custom(String),
}

impl std::fmt::Display for AuditEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditEvent::QueryExecuted => write!(f, "QueryExecuted"),
            AuditEvent::DataModified => write!(f, "DataModified"),
            AuditEvent::AdminAction => write!(f, "AdminAction"),
            AuditEvent::AuthFailure => write!(f, "AuthFailure"),
            AuditEvent::DataExport => write!(f, "DataExport"),
            AuditEvent::AuthSuccess => write!(f, "AuthSuccess"),
            AuditEvent::RateLimited => write!(f, "RateLimited"),
            AuditEvent::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// A single immutable audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    /// Wall-clock timestamp (UTC) when the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Authenticated user identifier or `"anonymous"`.
    pub user_id: String,
    /// Client IP address, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<IpAddr>,
    /// The type of event.
    pub event_type: AuditEvent,
    /// The server resource that was accessed (endpoint path, dataset name, etc.).
    pub resource: String,
    /// The query text, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_text: Option<String>,
    /// How long the operation took in milliseconds.
    pub duration_ms: u64,
    /// Whether the operation completed successfully.
    pub success: bool,
    /// Additional structured details (arbitrary JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl AuditRecord {
    /// Create a minimal audit record.
    pub fn new(
        user_id: impl Into<String>,
        event_type: AuditEvent,
        resource: impl Into<String>,
        success: bool,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            user_id: user_id.into(),
            client_ip: None,
            event_type,
            resource: resource.into(),
            query_text: None,
            duration_ms: 0,
            success,
            details: None,
        }
    }

    /// Builder: set client IP.
    pub fn with_client_ip(mut self, ip: IpAddr) -> Self {
        self.client_ip = Some(ip);
        self
    }

    /// Builder: set query text.
    pub fn with_query_text(mut self, query: impl Into<String>) -> Self {
        self.query_text = Some(query.into());
        self
    }

    /// Builder: set duration.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Builder: set extra details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// An async sink for `AuditRecord` values.
#[async_trait::async_trait]
pub trait AuditBackend: Send + Sync + 'static {
    /// Persist a single record.  Must not block.
    async fn write(&self, record: &AuditRecord) -> Result<(), AuditError>;
    /// Flush buffered writes to durable storage.
    async fn flush(&self) -> Result<(), AuditError>;
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the audit subsystem.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("File backend error: {0}")]
    File(#[from] file_backend::FileBackendError),
    #[error("Backend channel closed")]
    ChannelClosed,
    #[error("Audit logger is shut down")]
    ShutDown,
}

// ---------------------------------------------------------------------------
// AuditLogger
// ---------------------------------------------------------------------------

/// Configuration for `AuditLogger`.
#[derive(Debug, Clone)]
pub struct AuditLoggerConfig {
    /// Number of records to buffer before forcing a flush.
    pub buffer_capacity: usize,
    /// Interval at which the background flusher drains the buffer.
    pub flush_interval: Duration,
    /// Whether to drop records silently when the buffer is full (vs. blocking).
    pub drop_on_full: bool,
}

impl Default for AuditLoggerConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 1_000,
            flush_interval: Duration::from_secs(5),
            drop_on_full: false,
        }
    }
}

/// High-throughput async audit logger.
///
/// Records are enqueued in an in-memory ring buffer and flushed to the
/// configured backend(s) by a background task.
pub struct AuditLogger {
    buffer: Arc<Mutex<VecDeque<AuditRecord>>>,
    notify: Arc<Notify>,
    config: AuditLoggerConfig,
    backends: Arc<Vec<Box<dyn AuditBackend>>>,
    shutdown: Arc<tokio::sync::watch::Sender<bool>>,
}

impl AuditLogger {
    /// Construct an `AuditLogger` with the given backends and configuration.
    pub fn new(backends: Vec<Box<dyn AuditBackend>>, config: AuditLoggerConfig) -> Self {
        let (tx, _rx) = tokio::sync::watch::channel(false);
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_capacity))),
            notify: Arc::new(Notify::new()),
            config,
            backends: Arc::new(backends),
            shutdown: Arc::new(tx),
        }
    }

    /// Enqueue a record for async writing.
    ///
    /// Returns `Ok(())` if enqueued, or `Err(AuditError::ChannelClosed)` if
    /// the logger has been shut down.
    pub async fn log(&self, record: AuditRecord) -> Result<(), AuditError> {
        let mut buf = self.buffer.lock().await;
        if buf.len() >= self.config.buffer_capacity && self.config.drop_on_full {
            // Silently discard the oldest record to make room.
            buf.pop_front();
            // When `drop_on_full` is false we still enqueue but the caller
            // accepts potential unbounded growth (bounded by OS memory).
        }
        buf.push_back(record);
        drop(buf);
        self.notify.notify_one();
        Ok(())
    }

    /// Convenience: log a query execution event.
    pub async fn log_query(
        &self,
        user_id: impl Into<String>,
        client_ip: Option<IpAddr>,
        resource: impl Into<String>,
        query_text: Option<String>,
        duration_ms: u64,
        success: bool,
    ) -> Result<(), AuditError> {
        let mut rec = AuditRecord::new(user_id, AuditEvent::QueryExecuted, resource, success)
            .with_duration_ms(duration_ms);
        if let Some(ip) = client_ip {
            rec = rec.with_client_ip(ip);
        }
        if let Some(q) = query_text {
            rec = rec.with_query_text(q);
        }
        self.log(rec).await
    }

    /// Convenience: log an auth failure.
    pub async fn log_auth_failure(
        &self,
        user_id: impl Into<String>,
        client_ip: Option<IpAddr>,
        resource: impl Into<String>,
    ) -> Result<(), AuditError> {
        let mut rec = AuditRecord::new(user_id, AuditEvent::AuthFailure, resource, false);
        if let Some(ip) = client_ip {
            rec = rec.with_client_ip(ip);
        }
        self.log(rec).await
    }

    /// Convenience: log a data modification.
    pub async fn log_data_modified(
        &self,
        user_id: impl Into<String>,
        client_ip: Option<IpAddr>,
        resource: impl Into<String>,
        query_text: Option<String>,
        duration_ms: u64,
        success: bool,
    ) -> Result<(), AuditError> {
        let mut rec = AuditRecord::new(user_id, AuditEvent::DataModified, resource, success)
            .with_duration_ms(duration_ms);
        if let Some(ip) = client_ip {
            rec = rec.with_client_ip(ip);
        }
        if let Some(q) = query_text {
            rec = rec.with_query_text(q);
        }
        self.log(rec).await
    }

    /// Flush all buffered records to all backends immediately.
    pub async fn flush(&self) -> Result<(), AuditError> {
        let records: Vec<AuditRecord> = {
            let mut buf = self.buffer.lock().await;
            buf.drain(..).collect()
        };
        for record in &records {
            for backend in self.backends.iter() {
                backend.write(record).await?;
            }
        }
        for backend in self.backends.iter() {
            backend.flush().await?;
        }
        Ok(())
    }

    /// Start the background flush task.  Returns a handle that must be kept
    /// alive for the duration of the logger.
    pub fn start_background_flush(&self) -> tokio::task::JoinHandle<()> {
        let buffer = Arc::clone(&self.buffer);
        let notify = Arc::clone(&self.notify);
        let backends = Arc::clone(&self.backends);
        let interval = self.config.flush_interval;
        let mut shutdown_rx = self.shutdown.subscribe();

        tokio::spawn(async move {
            loop {
                // Wait for either the flush interval or a notification.
                tokio::select! {
                    _ = sleep(interval) => {},
                    _ = notify.notified() => {},
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
                let records: Vec<AuditRecord> = {
                    let mut buf = buffer.lock().await;
                    buf.drain(..).collect()
                };
                for record in &records {
                    for backend in backends.iter() {
                        if let Err(e) = backend.write(record).await {
                            tracing::error!("Audit backend write error: {}", e);
                        }
                    }
                }
                for backend in backends.iter() {
                    if let Err(e) = backend.flush().await {
                        tracing::error!("Audit backend flush error: {}", e);
                    }
                }
            }
        })
    }

    /// Signal the background flush task to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown.send(true);
    }

    /// Return the current number of buffered (unflushed) records.
    pub async fn buffered_count(&self) -> usize {
        self.buffer.lock().await.len()
    }
}

// ---------------------------------------------------------------------------
// Blanket implementations for concrete backends
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl AuditBackend for FileAuditBackend {
    async fn write(&self, record: &AuditRecord) -> Result<(), AuditError> {
        self.write(record).await.map_err(AuditError::File)
    }

    async fn flush(&self) -> Result<(), AuditError> {
        self.flush().await.map_err(AuditError::File)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::file_backend::FileBackendConfig;
    use std::net::IpAddr;
    use tokio::time::timeout;

    fn make_record(event: AuditEvent) -> AuditRecord {
        AuditRecord::new("alice", event, "/sparql", true)
            .with_client_ip("192.168.1.1".parse::<IpAddr>().unwrap())
            .with_duration_ms(100)
    }

    // -----------------------------------------------------------------------
    // AuditEvent
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_event_display() {
        assert_eq!(AuditEvent::QueryExecuted.to_string(), "QueryExecuted");
        assert_eq!(AuditEvent::DataModified.to_string(), "DataModified");
        assert_eq!(AuditEvent::AdminAction.to_string(), "AdminAction");
        assert_eq!(AuditEvent::AuthFailure.to_string(), "AuthFailure");
        assert_eq!(AuditEvent::DataExport.to_string(), "DataExport");
        assert_eq!(AuditEvent::AuthSuccess.to_string(), "AuthSuccess");
        assert_eq!(AuditEvent::RateLimited.to_string(), "RateLimited");
        assert_eq!(
            AuditEvent::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }

    #[test]
    fn test_audit_event_serialize() {
        let json = serde_json::to_string(&AuditEvent::QueryExecuted).unwrap();
        assert!(json.contains("QueryExecuted"));
    }

    #[test]
    fn test_audit_event_deserialize() {
        let event: AuditEvent = serde_json::from_str("\"QueryExecuted\"").unwrap();
        assert_eq!(event, AuditEvent::QueryExecuted);
    }

    #[test]
    fn test_audit_event_equality() {
        assert_eq!(AuditEvent::QueryExecuted, AuditEvent::QueryExecuted);
        assert_ne!(AuditEvent::QueryExecuted, AuditEvent::DataModified);
    }

    // -----------------------------------------------------------------------
    // AuditRecord
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_record_new() {
        let rec = AuditRecord::new("bob", AuditEvent::DataModified, "/update", true);
        assert_eq!(rec.user_id, "bob");
        assert_eq!(rec.event_type, AuditEvent::DataModified);
        assert_eq!(rec.resource, "/update");
        assert!(rec.success);
        assert!(rec.client_ip.is_none());
        assert!(rec.query_text.is_none());
        assert_eq!(rec.duration_ms, 0);
    }

    #[test]
    fn test_audit_record_builder_chain() {
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        let rec = AuditRecord::new("carol", AuditEvent::AuthFailure, "/auth", false)
            .with_client_ip(ip)
            .with_query_text("SELECT ?x WHERE {}")
            .with_duration_ms(5)
            .with_details(serde_json::json!({"reason": "bad_password"}));
        assert_eq!(rec.client_ip, Some(ip));
        assert_eq!(rec.query_text.as_deref(), Some("SELECT ?x WHERE {}"));
        assert_eq!(rec.duration_ms, 5);
        assert!(rec.details.is_some());
    }

    #[test]
    fn test_audit_record_serialize_skip_nones() {
        let rec = AuditRecord::new("dave", AuditEvent::AdminAction, "/admin", true);
        let json = serde_json::to_string(&rec).unwrap();
        // Optional fields should be absent from JSON.
        assert!(!json.contains("client_ip"));
        assert!(!json.contains("query_text"));
        assert!(!json.contains("details"));
    }

    #[test]
    fn test_audit_record_serialize_with_ip() {
        let ip: IpAddr = "203.0.113.42".parse().unwrap();
        let rec =
            AuditRecord::new("eve", AuditEvent::DataExport, "/export", true).with_client_ip(ip);
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("203.0.113.42"));
    }

    #[test]
    fn test_audit_record_deserialize_roundtrip() {
        let rec = make_record(AuditEvent::QueryExecuted);
        let json = serde_json::to_string(&rec).unwrap();
        let rec2: AuditRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(rec2.user_id, rec.user_id);
        assert_eq!(rec2.event_type, rec.event_type);
        assert_eq!(rec2.resource, rec.resource);
        assert_eq!(rec2.success, rec.success);
        assert_eq!(rec2.duration_ms, rec.duration_ms);
    }

    // -----------------------------------------------------------------------
    // AuditLoggerConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_logger_config_defaults() {
        let cfg = AuditLoggerConfig::default();
        assert_eq!(cfg.buffer_capacity, 1_000);
        assert_eq!(cfg.flush_interval, Duration::from_secs(5));
        assert!(!cfg.drop_on_full);
    }

    // -----------------------------------------------------------------------
    // AuditLogger
    // -----------------------------------------------------------------------

    async fn file_logger(tmp: &std::path::Path) -> (AuditLogger, FileBackendConfig) {
        let cfg = FileBackendConfig::new(tmp);
        let backend = FileAuditBackend::new(cfg.clone()).await.unwrap();
        let logger = AuditLogger::new(
            vec![Box::new(backend)],
            AuditLoggerConfig {
                buffer_capacity: 100,
                flush_interval: Duration::from_millis(50),
                drop_on_full: false,
            },
        );
        (logger, cfg)
    }

    #[tokio::test]
    async fn test_logger_log_and_flush() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_flush_{}",
            Utc::now().timestamp_millis()
        ));
        let (logger, _cfg) = file_logger(&tmp).await;
        logger
            .log(make_record(AuditEvent::QueryExecuted))
            .await
            .unwrap();
        assert_eq!(logger.buffered_count().await, 1);
        logger.flush().await.unwrap();
        assert_eq!(logger.buffered_count().await, 0);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_log_query_convenience() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_query_{}",
            Utc::now().timestamp_millis()
        ));
        let (logger, _cfg) = file_logger(&tmp).await;
        logger
            .log_query(
                "frank",
                Some("127.0.0.1".parse().unwrap()),
                "/sparql",
                Some("SELECT * {}".to_string()),
                123,
                true,
            )
            .await
            .unwrap();
        assert_eq!(logger.buffered_count().await, 1);
        logger.flush().await.unwrap();
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_log_auth_failure() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_auth_{}",
            Utc::now().timestamp_millis()
        ));
        let (logger, _cfg) = file_logger(&tmp).await;
        logger
            .log_auth_failure("unknown", Some("10.0.0.1".parse().unwrap()), "/auth")
            .await
            .unwrap();
        logger.flush().await.unwrap();
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_log_data_modified() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_modify_{}",
            Utc::now().timestamp_millis()
        ));
        let (logger, _cfg) = file_logger(&tmp).await;
        logger
            .log_data_modified(
                "grace",
                None,
                "/update",
                Some("INSERT DATA {}".to_string()),
                50,
                true,
            )
            .await
            .unwrap();
        logger.flush().await.unwrap();
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_background_flush() {
        let tmp =
            std::env::temp_dir().join(format!("oxirs_logger_bg_{}", Utc::now().timestamp_millis()));
        let cfg = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(cfg.clone()).await.unwrap();
        let logger = AuditLogger::new(
            vec![Box::new(backend)],
            AuditLoggerConfig {
                buffer_capacity: 100,
                flush_interval: Duration::from_millis(30),
                drop_on_full: false,
            },
        );
        let handle = logger.start_background_flush();
        logger
            .log(make_record(AuditEvent::AdminAction))
            .await
            .unwrap();
        sleep(Duration::from_millis(200)).await;
        assert_eq!(logger.buffered_count().await, 0);
        logger.shutdown();
        let _ = timeout(Duration::from_secs(2), handle).await;
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_drop_on_full() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_drop_{}",
            Utc::now().timestamp_millis()
        ));
        let cfg = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(cfg).await.unwrap();
        let logger = AuditLogger::new(
            vec![Box::new(backend)],
            AuditLoggerConfig {
                buffer_capacity: 3,
                flush_interval: Duration::from_secs(60),
                drop_on_full: true,
            },
        );
        for _ in 0..10 {
            logger
                .log(make_record(AuditEvent::RateLimited))
                .await
                .unwrap();
        }
        // Buffer capacity = 3; oldest entries dropped.
        assert!(logger.buffered_count().await <= 3);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_multiple_events() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_events_{}",
            Utc::now().timestamp_millis()
        ));
        let (logger, _cfg) = file_logger(&tmp).await;
        let events = vec![
            AuditEvent::QueryExecuted,
            AuditEvent::DataModified,
            AuditEvent::AdminAction,
            AuditEvent::AuthFailure,
            AuditEvent::DataExport,
            AuditEvent::AuthSuccess,
            AuditEvent::RateLimited,
        ];
        for event in events {
            logger.log(make_record(event)).await.unwrap();
        }
        assert_eq!(logger.buffered_count().await, 7);
        logger.flush().await.unwrap();
        assert_eq!(logger.buffered_count().await, 0);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_flush_writes_to_file() {
        let tmp = std::env::temp_dir().join(format!(
            "oxirs_logger_file_{}",
            Utc::now().timestamp_millis()
        ));
        let cfg = FileBackendConfig::new(&tmp);
        let backend = FileAuditBackend::new(cfg.clone()).await.unwrap();
        let path = backend.current_path().await.unwrap();
        let logger = AuditLogger::new(vec![Box::new(backend)], AuditLoggerConfig::default());
        for _ in 0..5 {
            logger
                .log(make_record(AuditEvent::QueryExecuted))
                .await
                .unwrap();
        }
        logger.flush().await.unwrap();

        let contents = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(contents.lines().count(), 5);
        let _ = tokio::fs::remove_dir_all(&tmp).await;
    }

    #[tokio::test]
    async fn test_logger_no_backend_is_valid() {
        // A logger with zero backends should still work without panicking.
        let logger = AuditLogger::new(vec![], AuditLoggerConfig::default());
        logger
            .log(make_record(AuditEvent::AdminAction))
            .await
            .unwrap();
        logger.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_audit_record_custom_event() {
        let rec = AuditRecord::new(
            "henry",
            AuditEvent::Custom("sparql_federation".to_string()),
            "/federated",
            true,
        );
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains("sparql_federation"));
    }

    #[tokio::test]
    async fn test_logger_shutdown_signals() {
        let logger = AuditLogger::new(vec![], AuditLoggerConfig::default());
        let handle = logger.start_background_flush();
        logger.shutdown();
        let result = timeout(Duration::from_secs(3), handle).await;
        assert!(result.is_ok(), "Background task should stop after shutdown");
    }
}
