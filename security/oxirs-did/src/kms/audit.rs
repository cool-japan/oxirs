//! Audit log for HSM cryptographic operations.
//!
//! Records sign, verify, key-generation, and key-destruction events
//! with timestamps for compliance and forensics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// Audit event types
// ─────────────────────────────────────────────────────────────────────────────

/// The category of an auditable event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventKind {
    /// A signing operation was requested.
    Sign,
    /// A signature verification was requested.
    Verify,
    /// A key pair was generated.
    KeyGenerate,
    /// A key was destroyed / deleted.
    KeyDestroy,
    /// A key was exported (public key retrieval).
    KeyExport,
    /// Authentication / session open.
    SessionOpen,
    /// Authentication / session close.
    SessionClose,
}

impl AuditEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sign => "SIGN",
            Self::Verify => "VERIFY",
            Self::KeyGenerate => "KEY_GENERATE",
            Self::KeyDestroy => "KEY_DESTROY",
            Self::KeyExport => "KEY_EXPORT",
            Self::SessionOpen => "SESSION_OPEN",
            Self::SessionClose => "SESSION_CLOSE",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AuditEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// UTC timestamp of the event.
    pub timestamp: DateTime<Utc>,
    /// Event category.
    pub kind: AuditEventKind,
    /// Slot or key identifier.
    pub key_id: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Optional human-readable detail.
    pub detail: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event recorded at the current UTC time.
    pub fn now(
        kind: AuditEventKind,
        key_id: impl Into<String>,
        success: bool,
        detail: Option<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            kind,
            key_id: key_id.into(),
            success,
            detail,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AuditLog
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe append-only audit log for HSM operations.
#[derive(Clone)]
pub struct AuditLog {
    events: Arc<RwLock<Vec<AuditEvent>>>,
}

impl AuditLog {
    /// Create an empty audit log.
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Append an event to the log.
    ///
    /// Returns `Err` only if the internal lock is poisoned (extremely rare).
    pub fn append(&self, event: AuditEvent) -> Result<(), String> {
        self.events
            .write()
            .map_err(|e| format!("AuditLog lock poisoned: {e}"))?
            .push(event);
        Ok(())
    }

    /// Convenience helper: log a sign attempt.
    pub fn log_sign(&self, key_id: &str, success: bool, detail: Option<String>) {
        let _ = self.append(AuditEvent::now(
            AuditEventKind::Sign,
            key_id,
            success,
            detail,
        ));
    }

    /// Convenience helper: log a verify attempt.
    pub fn log_verify(&self, key_id: &str, success: bool, detail: Option<String>) {
        let _ = self.append(AuditEvent::now(
            AuditEventKind::Verify,
            key_id,
            success,
            detail,
        ));
    }

    /// Convenience helper: log a key generation event.
    pub fn log_key_generate(&self, key_id: &str, success: bool) {
        let _ = self.append(AuditEvent::now(
            AuditEventKind::KeyGenerate,
            key_id,
            success,
            None,
        ));
    }

    /// Convenience helper: log a key destruction event.
    pub fn log_key_destroy(&self, key_id: &str, success: bool) {
        let _ = self.append(AuditEvent::now(
            AuditEventKind::KeyDestroy,
            key_id,
            success,
            None,
        ));
    }

    /// Return all events in insertion order.
    pub fn all_events(&self) -> Vec<AuditEvent> {
        self.events
            .read()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Return events matching the given kind.
    pub fn events_by_kind(&self, kind: AuditEventKind) -> Vec<AuditEvent> {
        self.all_events()
            .into_iter()
            .filter(|e| e.kind == kind)
            .collect()
    }

    /// Return events for a specific key_id.
    pub fn events_for_key(&self, key_id: &str) -> Vec<AuditEvent> {
        self.all_events()
            .into_iter()
            .filter(|e| e.key_id == key_id)
            .collect()
    }

    /// Total number of recorded events.
    pub fn len(&self) -> usize {
        self.events.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Return `true` if the log has no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Count failed events (for alerting / monitoring).
    pub fn failed_event_count(&self) -> usize {
        self.all_events().iter().filter(|e| !e.success).count()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_and_retrieve() {
        let log = AuditLog::new();
        log.log_sign("key-1", true, None);
        log.log_verify("key-1", false, Some("bad sig".into()));
        log.log_key_generate("key-2", true);

        assert_eq!(log.len(), 3);
        let events = log.all_events();
        assert_eq!(events[0].kind, AuditEventKind::Sign);
        assert!(events[0].success);
        assert_eq!(events[1].kind, AuditEventKind::Verify);
        assert!(!events[1].success);
    }

    #[test]
    fn test_filter_by_kind() {
        let log = AuditLog::new();
        log.log_sign("k1", true, None);
        log.log_sign("k2", false, None);
        log.log_verify("k1", true, None);

        let signs = log.events_by_kind(AuditEventKind::Sign);
        assert_eq!(signs.len(), 2);
        let verifies = log.events_by_kind(AuditEventKind::Verify);
        assert_eq!(verifies.len(), 1);
    }

    #[test]
    fn test_filter_by_key() {
        let log = AuditLog::new();
        log.log_sign("alpha", true, None);
        log.log_sign("beta", true, None);
        log.log_verify("alpha", false, None);

        let alpha = log.events_for_key("alpha");
        assert_eq!(alpha.len(), 2);
    }

    #[test]
    fn test_failed_event_count() {
        let log = AuditLog::new();
        log.log_sign("k", true, None);
        log.log_sign("k", false, None);
        log.log_verify("k", false, None);

        assert_eq!(log.failed_event_count(), 2);
    }

    #[test]
    fn test_empty_log() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.failed_event_count(), 0);
    }

    #[test]
    fn test_event_kind_as_str() {
        assert_eq!(AuditEventKind::Sign.as_str(), "SIGN");
        assert_eq!(AuditEventKind::KeyGenerate.as_str(), "KEY_GENERATE");
        assert_eq!(AuditEventKind::SessionClose.as_str(), "SESSION_CLOSE");
    }

    #[test]
    fn test_clone_log_shared_state() {
        let log = AuditLog::new();
        let log2 = log.clone();
        log.log_sign("shared", true, None);
        // Clone should see the event appended by the original
        assert_eq!(log2.len(), 1, "cloned log shares state");
    }

    #[test]
    fn test_audit_event_serialization() {
        let event = AuditEvent::now(AuditEventKind::Sign, "key-123", true, None);
        let json = serde_json::to_string(&event).unwrap();
        let decoded: AuditEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.key_id, "key-123");
        assert_eq!(decoded.kind, AuditEventKind::Sign);
    }
}
