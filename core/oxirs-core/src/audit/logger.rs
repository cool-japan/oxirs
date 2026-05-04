//! Audit logger sinks: in-memory, NDJSON file, and composite fan-out.

use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};

use super::event::AuditEvent;

/// Error type for audit logging operations.
#[derive(Debug, thiserror::Error)]
pub enum AuditLogError {
    /// The in-memory logger has reached its maximum capacity.
    #[error("logger capacity exceeded ({0} events)")]
    CapacityExceeded(usize),
    /// An underlying I/O error occurred (e.g., disk write failure).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization of the event failed.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Trait for audit event sinks.
///
/// Implementors are required to be `Send + Sync` so they can be used behind
/// `Arc` in multi-threaded services.
pub trait AuditLogger: Send + Sync {
    /// Record a single audit event.
    ///
    /// Implementors must not panic; all errors must be returned as
    /// [`AuditLogError`] variants.
    fn log(&self, event: AuditEvent) -> Result<(), AuditLogError>;

    /// Flush any buffered events to the underlying sink.
    ///
    /// The default implementation is a no-op, suitable for loggers that
    /// write synchronously.
    fn flush(&self) -> Result<(), AuditLogError> {
        Ok(())
    }
}

// ─────────────────────────────────────────────
// InMemoryAuditLogger
// ─────────────────────────────────────────────

/// In-memory audit logger, backed by a `Vec<AuditEvent>`.
///
/// Suitable for tests and short-lived processes. When capacity is exceeded,
/// [`log`](AuditLogger::log) returns [`AuditLogError::CapacityExceeded`] rather
/// than evicting events — preserving audit trail completeness.
pub struct InMemoryAuditLogger {
    events: Arc<RwLock<Vec<AuditEvent>>>,
    max_capacity: usize,
}

impl Default for InMemoryAuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryAuditLogger {
    /// Create a new logger with the default capacity of 10,000 events.
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Create a new logger with a custom maximum capacity.
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_capacity,
        }
    }

    /// Return a snapshot of all logged events.
    ///
    /// Acquires a read lock; panics (poison recovery) if the lock is poisoned.
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Return the number of events currently stored.
    pub fn len(&self) -> usize {
        self.events.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Return `true` if no events are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all stored events.
    pub fn clear(&self) {
        self.events
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }
}

impl AuditLogger for InMemoryAuditLogger {
    fn log(&self, event: AuditEvent) -> Result<(), AuditLogError> {
        let mut guard = self.events.write().unwrap_or_else(|p| p.into_inner());
        if guard.len() >= self.max_capacity {
            return Err(AuditLogError::CapacityExceeded(self.max_capacity));
        }
        guard.push(event);
        Ok(())
    }
}

// ─────────────────────────────────────────────
// JsonLineAuditLogger
// ─────────────────────────────────────────────

/// NDJSON (newline-delimited JSON) audit logger.
///
/// Writes one compact JSON object per line to an underlying `Write` sink.
/// Thread-safe: concurrent calls to [`log`](AuditLogger::log) are serialised
/// via an internal `Mutex`. Suitable for log aggregators that tail files
/// (Fluentd, Filebeat, etc.).
pub struct JsonLineAuditLogger {
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl JsonLineAuditLogger {
    /// Create a logger that writes to any `Write` sink.
    pub fn new(writer: impl Write + Send + 'static) -> Self {
        Self {
            writer: Arc::new(Mutex::new(Box::new(writer))),
        }
    }

    /// Create a logger that appends to a file at the given path.
    ///
    /// The file is created if it does not exist. Existing content is preserved
    /// (append semantics) to maintain audit trail integrity.
    pub fn to_file(path: &std::path::Path) -> Result<Self, AuditLogError> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self::new(file))
    }

    /// Create a logger that writes to standard error.
    pub fn to_stderr() -> Self {
        Self::new(std::io::stderr())
    }
}

impl AuditLogger for JsonLineAuditLogger {
    fn log(&self, event: AuditEvent) -> Result<(), AuditLogError> {
        let line = serde_json::to_string(&event)
            .map_err(|e| AuditLogError::Serialization(e.to_string()))?;
        let mut guard = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        guard.write_all(line.as_bytes())?;
        guard.write_all(b"\n")?;
        Ok(())
    }

    fn flush(&self) -> Result<(), AuditLogError> {
        let mut guard = self.writer.lock().unwrap_or_else(|p| p.into_inner());
        guard.flush()?;
        Ok(())
    }
}

// ─────────────────────────────────────────────
// CompositeAuditLogger
// ─────────────────────────────────────────────

/// Fan-out audit logger that dispatches every event to multiple child loggers.
///
/// All loggers receive every event. If one or more loggers return errors, the
/// composite logger still delivers to all remaining loggers before returning
/// the first error encountered. This ensures maximum coverage even in partial
/// failure scenarios.
pub struct CompositeAuditLogger {
    loggers: Vec<Arc<dyn AuditLogger>>,
}

impl CompositeAuditLogger {
    /// Create a composite logger from a list of child logger arcs.
    pub fn new(loggers: Vec<Arc<dyn AuditLogger>>) -> Self {
        Self { loggers }
    }
}

impl AuditLogger for CompositeAuditLogger {
    fn log(&self, event: AuditEvent) -> Result<(), AuditLogError> {
        // Fan out to all loggers, collecting the first error while still
        // delivering to all remaining loggers (fail-partial semantics).
        let mut first_err: Option<AuditLogError> = None;
        for logger in &self.loggers {
            if let Err(e) = logger.log(event.clone()) {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn flush(&self) -> Result<(), AuditLogError> {
        let mut first_err: Option<AuditLogError> = None;
        for logger in &self.loggers {
            if let Err(e) = logger.flush() {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
