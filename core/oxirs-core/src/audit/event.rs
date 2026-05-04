//! Audit event types for SOC2/GDPR-compliant structured event logging.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The type of actor who performed an action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActorType {
    /// An authenticated human user.
    User,
    /// An automated service account (e.g., a pipeline or integration).
    ServiceAccount,
    /// An internal system process (e.g., scheduled maintenance).
    System,
    /// An unauthenticated caller.
    Anonymous,
}

/// The actor who performed an action.
///
/// Captures identity information for the party responsible for the event.
/// Distinct from the data subject (who the action concerns).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditActor {
    /// Stable identifier for the actor (e.g., user ID, service account name).
    pub actor_id: String,
    /// The class of actor.
    pub actor_type: ActorType,
    /// Client IP address at the time of the action, if known.
    pub ip_address: Option<String>,
    /// Session or token identifier, if applicable.
    pub session_id: Option<String>,
}

/// The resource that was acted upon.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditResource {
    /// Resource class, e.g. `"dataset"`, `"graph"`, `"query"`, `"user"`.
    pub resource_type: String,
    /// Stable identifier for the specific resource instance.
    pub resource_id: String,
    /// Tenant scope, for multi-tenant deployments.
    pub tenant_id: Option<String>,
}

/// Outcome of the audited action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AuditOutcome {
    /// The action completed successfully.
    Success,
    /// The action failed outright.
    Failure {
        /// Human-readable description of why the action failed.
        reason: String,
    },
    /// The action partially succeeded.
    PartialSuccess {
        /// Details about what succeeded and what did not.
        details: String,
    },
}

/// High-level category of the audited event.
///
/// Used for compliance filtering (e.g., SOC2 CC6, GDPR Article 30).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventKind {
    /// Login, logout, token refresh, MFA challenge.
    Authentication,
    /// Access check, RBAC decision, policy evaluation.
    Authorization,
    /// SPARQL SELECT, graph read, dataset list.
    DataAccess,
    /// SPARQL UPDATE, INSERT, DELETE, CLEAR.
    DataModification,
    /// User management, dataset creation/deletion, configuration change.
    Admin,
    /// Failed authentication, rate limiting, suspicious access pattern.
    Security,
    /// Startup, shutdown, backup, restore.
    System,
}

/// A single immutable audit event record.
///
/// Once constructed via [`AuditEvent::new`], the core identity and timing fields
/// are fixed. Builder methods add optional enrichment (`with_duration`,
/// `with_metadata`, `with_data_subject`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// UUID v4 uniquely identifying this event record.
    pub event_id: String,
    /// Wall-clock time the event was recorded (UTC).
    pub timestamp: DateTime<Utc>,
    /// High-level event category.
    pub kind: AuditEventKind,
    /// Machine-readable action label, e.g. `"sparql.select"`, `"auth.login"`.
    pub action: String,
    /// Who performed the action.
    pub actor: AuditActor,
    /// What the action was performed on.
    pub resource: AuditResource,
    /// Whether the action succeeded.
    pub outcome: AuditOutcome,
    /// Elapsed time for the action, if measured.
    pub duration_ms: Option<u64>,
    /// Arbitrary key-value metadata for context (e.g., query text, bytes transferred).
    pub metadata: HashMap<String, String>,
    /// GDPR: which data subject does this event concern?
    ///
    /// Distinct from the `actor`. For example, an admin modifying another user's
    /// profile: `actor.actor_id` is the admin, `data_subject_id` is the target user.
    pub data_subject_id: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event with a fresh UUID and the current UTC timestamp.
    ///
    /// The `event_id` is generated with [`Uuid::new_v4`] and `timestamp` is set
    /// to [`Utc::now()`]. All optional fields default to `None`/empty.
    pub fn new(
        kind: AuditEventKind,
        action: impl Into<String>,
        actor: AuditActor,
        resource: AuditResource,
        outcome: AuditOutcome,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            kind,
            action: action.into(),
            actor,
            resource,
            outcome,
            duration_ms: None,
            metadata: HashMap::new(),
            data_subject_id: None,
        }
    }

    /// Attach a measured duration to this event.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Attach a single key-value metadata entry.
    ///
    /// Can be chained to add multiple entries.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Associate this event with a GDPR data subject.
    ///
    /// Required for [`crate::audit::gdpr::GdprService`] to generate correct
    /// data subject reports and pseudonymisation.
    #[must_use]
    pub fn with_data_subject(mut self, subject_id: impl Into<String>) -> Self {
        self.data_subject_id = Some(subject_id.into());
        self
    }
}
