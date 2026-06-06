use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, TdbError};

/// Opaque, URL-safe identifier for a tenant.
///
/// Internal representation is a validated string containing only
/// `[a-zA-Z0-9_-]` characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantId(Arc<str>);

impl TenantId {
    /// Create a new [`TenantId`] from a string slice.
    ///
    /// Returns an error if the string is empty or contains disallowed characters.
    pub fn new(id: impl AsRef<str>) -> Result<Self> {
        let s = id.as_ref();
        if s.is_empty() {
            return Err(TdbError::InvalidInput(
                "TenantId must not be empty".to_string(),
            ));
        }
        if !s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(TdbError::InvalidInput(format!(
                "TenantId '{}' contains disallowed characters (only [a-zA-Z0-9_-] allowed)",
                s
            )));
        }
        if s.len() > 128 {
            return Err(TdbError::InvalidInput(
                "TenantId must be at most 128 characters".to_string(),
            ));
        }
        Ok(Self(Arc::from(s)))
    }

    /// Return the string representation of the tenant ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Create the namespace prefix used to isolate this tenant's triples.
    ///
    /// The prefix is `"tenant:<id>:"` — prepended to every subject IRI before
    /// storage so that tenants cannot accidentally read each other's data.
    pub fn namespace_prefix(&self) -> String {
        format!("tenant:{}:", self.0)
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Resource limits and access policy for a single tenant.
#[derive(Debug, Clone)]
pub struct TenantConfig {
    /// Maximum number of triples this tenant may store (0 = unlimited).
    pub max_triples: u64,
    /// Maximum number of named graphs (0 = unlimited).
    pub max_graphs: u64,
    /// Approximate maximum storage in bytes (0 = unlimited).
    pub quota_bytes: u64,
    /// Predicate IRIs this tenant is allowed to use.
    /// An empty vec means all predicates are permitted.
    pub allowed_predicates: Vec<String>,
    /// Named graph IRI prefixes this tenant is allowed to write to.
    /// Each graph IRI must start with one of these prefixes.
    /// An empty vec means all graph IRIs are permitted.
    pub allowed_prefixes: Vec<String>,
    /// Whether this tenant is active. Inactive tenants reject all writes.
    pub active: bool,
}

impl Default for TenantConfig {
    fn default() -> Self {
        Self {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec![],
            allowed_prefixes: vec![],
            active: true,
        }
    }
}

impl TenantConfig {
    /// Create an unlimited configuration suitable for trusted tenants.
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Create a restricted configuration with the given limits.
    pub fn with_limits(max_triples: u64, max_graphs: u64, quota_bytes: u64) -> Self {
        Self {
            max_triples,
            max_graphs,
            quota_bytes,
            allowed_predicates: vec![],
            allowed_prefixes: vec![],
            active: true,
        }
    }
}

/// Runtime statistics for a single tenant.
#[derive(Debug, Clone, Default)]
pub struct TenantStats {
    /// Current number of stored triples.
    pub triple_count: u64,
    /// Current number of distinct named graphs (graph IRIs seen).
    pub graph_count: u64,
    /// Approximate bytes consumed (estimated at 100 bytes / triple).
    pub bytes_used: u64,
    /// Total successful write operations.
    pub writes: u64,
    /// Total read operations (queries).
    pub reads: u64,
    /// Total write operations rejected due to quota exceeded.
    pub quota_rejections: u64,
    /// Total write operations rejected due to disallowed predicates.
    pub predicate_rejections: u64,
}

impl TenantStats {
    /// Estimate bytes from current triple count.
    pub fn estimate_bytes(triple_count: u64) -> u64 {
        triple_count * 100
    }
}

#[derive(Debug)]
pub(crate) struct TenantEntry {
    pub(crate) config: TenantConfig,
    pub(crate) stats: TenantStats,
    pub(crate) graphs: HashSet<String>,
}

impl TenantEntry {
    pub(crate) fn new(config: TenantConfig) -> Self {
        Self {
            config,
            stats: TenantStats::default(),
            graphs: HashSet::new(),
        }
    }
}

/// Error variants specific to multi-tenant operations.
#[derive(Debug, thiserror::Error)]
pub enum TenantError {
    /// The specified tenant does not exist.
    #[error("Tenant not found: {0}")]
    NotFound(String),

    /// A tenant with the same ID already exists.
    #[error("Tenant already exists: {0}")]
    AlreadyExists(String),

    /// The tenant is inactive and rejects operations.
    #[error("Tenant is inactive: {0}")]
    Inactive(String),

    /// Triple count quota exceeded.
    #[error("Triple quota exceeded for tenant '{tenant}': {current}/{limit}")]
    QuotaTriples {
        /// Tenant ID
        tenant: String,
        /// Current triple count
        current: u64,
        /// Configured limit
        limit: u64,
    },

    /// Byte storage quota exceeded.
    #[error("Storage quota exceeded for tenant '{tenant}': {current}/{limit} bytes")]
    QuotaBytes {
        /// Tenant ID
        tenant: String,
        /// Current bytes used
        current: u64,
        /// Configured limit
        limit: u64,
    },

    /// Graph count quota exceeded.
    #[error("Graph quota exceeded for tenant '{tenant}': {current}/{limit}")]
    QuotaGraphs {
        /// Tenant ID
        tenant: String,
        /// Current graph count
        current: u64,
        /// Configured limit
        limit: u64,
    },

    /// The predicate is not in the tenant's allowed list.
    #[error("Predicate '{predicate}' not allowed for tenant '{tenant}'")]
    PredicateNotAllowed {
        /// Predicate IRI that was rejected
        predicate: String,
        /// Tenant whose policy rejected it
        tenant: String,
    },

    /// Cross-tenant access was detected and blocked.
    #[error("Cross-tenant access denied: tenant '{accessor}' tried to access '{target}'")]
    CrossTenantAccess {
        /// Tenant making the access
        accessor: String,
        /// Tenant whose data was targeted
        target: String,
    },

    /// The graph IRI does not match any allowed prefix for this tenant.
    #[error("Graph IRI '{graph}' not allowed for tenant '{tenant}': must match an allowed prefix")]
    GraphPrefixNotAllowed {
        /// Graph IRI that was rejected
        graph: String,
        /// Tenant whose policy rejected it
        tenant: String,
    },
}

impl From<TenantError> for TdbError {
    fn from(e: TenantError) -> Self {
        TdbError::Other(e.to_string())
    }
}

/// Result type for tenant operations.
pub type TenantResult<T> = std::result::Result<T, TenantError>;

/// A single audit event for cross-tenant or anomalous access.
#[derive(Debug, Clone)]
pub struct TenantAuditEvent {
    /// Unix timestamp in seconds when the event occurred.
    pub timestamp_secs: u64,
    /// The tenant that attempted the action.
    pub accessor: TenantId,
    /// The tenant whose data was targeted (if applicable).
    pub target: Option<TenantId>,
    /// Human-readable description of the event.
    pub description: String,
    /// Whether access was blocked.
    pub blocked: bool,
}

/// Append-only log of tenant audit events, capped at a configurable maximum.
#[derive(Debug)]
pub struct TenantAuditLog {
    events: Arc<Mutex<VecDeque<TenantAuditEvent>>>,
    max_events: usize,
}

impl TenantAuditLog {
    /// Create a new audit log with the given maximum event capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::with_capacity(max_events.min(1024)))),
            max_events,
        }
    }

    /// Record an event. If the log is full, the oldest event is evicted.
    pub fn record(&self, event: TenantAuditEvent) {
        let Ok(mut guard) = self.events.lock() else {
            return;
        };
        if guard.len() >= self.max_events {
            guard.pop_front();
        }
        guard.push_back(event);
    }

    /// Record a cross-tenant access attempt.
    pub fn record_cross_tenant(&self, accessor: TenantId, target: TenantId, blocked: bool) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.record(TenantAuditEvent {
            timestamp_secs: ts,
            accessor,
            target: Some(target),
            description: "Cross-tenant data access attempt".to_string(),
            blocked,
        });
    }

    /// Record a quota violation attempt.
    pub fn record_quota_violation(&self, accessor: TenantId, description: String) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.record(TenantAuditEvent {
            timestamp_secs: ts,
            accessor,
            target: None,
            description,
            blocked: true,
        });
    }

    /// Return a snapshot of all logged events (oldest first).
    pub fn events(&self) -> Vec<TenantAuditEvent> {
        self.events
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return only events involving a specific tenant (as accessor or target).
    pub fn events_for_tenant(&self, tenant: &TenantId) -> Vec<TenantAuditEvent> {
        self.events()
            .into_iter()
            .filter(|e| {
                &e.accessor == tenant || e.target.as_ref().map(|t| t == tenant).unwrap_or(false)
            })
            .collect()
    }

    /// Count total events in the log.
    pub fn len(&self) -> usize {
        self.events.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Return `true` if the log contains no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all events.
    pub fn clear(&self) {
        if let Ok(mut g) = self.events.lock() {
            g.clear();
        }
    }
}
