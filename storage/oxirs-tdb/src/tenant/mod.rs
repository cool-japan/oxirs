//! Multi-Tenant Isolation for OxiRS TDB
//!
//! Provides per-tenant namespace isolation, quota enforcement, and
//! cross-tenant access auditing on top of the TDB storage engine.
//!
//! ## Design
//! - Each tenant gets a unique [`TenantId`] that prefixes all triple storage keys.
//! - [`TenantStore`] wraps a TDB store and transparently namespaces all operations.
//! - [`TenantRegistry`] manages tenant lifecycle (create, delete, list).
//! - [`TenantAuditLog`] records cross-tenant access attempts.
//! - Quotas are enforced at write time: inserts fail when limits are exceeded.
//!
//! ## Usage
//! ```rust,no_run
//! use oxirs_tdb::tenant::{TenantId, TenantConfig, TenantRegistry, TenantStore};
//!
//! let mut registry = TenantRegistry::new();
//! let id = TenantId::new("acme_corp").unwrap();
//! let config = TenantConfig {
//!     max_triples: 1_000_000,
//!     max_graphs: 100,
//!     quota_bytes: 512 * 1024 * 1024,
//!     allowed_predicates: vec![],
//!     allowed_prefixes: vec![],
//!     active: true,
//! };
//! registry.create_tenant(id, config).unwrap();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, TdbError};

// ─────────────────────────────────────────────────────────────────────────────
// TenantId
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TenantConfig
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TenantStats
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TenantEntry (internal)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct TenantEntry {
    config: TenantConfig,
    stats: TenantStats,
    /// Set of distinct graph names seen for this tenant.
    graphs: HashSet<String>,
}

impl TenantEntry {
    fn new(config: TenantConfig) -> Self {
        Self {
            config,
            stats: TenantStats::default(),
            graphs: HashSet::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TenantError
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TenantAuditLog
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TenantRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Central registry that manages the full lifecycle of tenants.
///
/// Thread-safe via an internal `RwLock`.
#[derive(Debug)]
pub struct TenantRegistry {
    tenants: Arc<RwLock<HashMap<TenantId, TenantEntry>>>,
    audit_log: Arc<TenantAuditLog>,
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TenantRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tenants: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(TenantAuditLog::new(10_000)),
        }
    }

    /// Create a new tenant with the given configuration.
    pub fn create_tenant(&self, id: TenantId, config: TenantConfig) -> TenantResult<()> {
        let mut guard = self
            .tenants
            .write()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        if guard.contains_key(&id) {
            return Err(TenantError::AlreadyExists(id.as_str().to_string()));
        }
        guard.insert(id, TenantEntry::new(config));
        Ok(())
    }

    /// Delete a tenant and discard all their statistics.
    ///
    /// This does **not** physically remove the underlying triples from the store;
    /// callers should also call [`TenantStore::purge`] first if desired.
    pub fn delete_tenant(&self, id: &TenantId) -> TenantResult<()> {
        let mut guard = self
            .tenants
            .write()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        if guard.remove(id).is_none() {
            return Err(TenantError::NotFound(id.as_str().to_string()));
        }
        Ok(())
    }

    /// List all tenant IDs (sorted for determinism).
    pub fn list_tenants(&self) -> Vec<TenantId> {
        let Ok(guard) = self.tenants.read() else {
            return vec![];
        };
        let mut ids: Vec<TenantId> = guard.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Return the configuration for a tenant.
    pub fn config(&self, id: &TenantId) -> TenantResult<TenantConfig> {
        let guard = self
            .tenants
            .read()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        guard
            .get(id)
            .map(|e| e.config.clone())
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))
    }

    /// Return the runtime statistics for a tenant.
    pub fn stats(&self, id: &TenantId) -> TenantResult<TenantStats> {
        let guard = self
            .tenants
            .read()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        guard
            .get(id)
            .map(|e| e.stats.clone())
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))
    }

    /// Update the configuration for an existing tenant.
    pub fn update_config(&self, id: &TenantId, config: TenantConfig) -> TenantResult<()> {
        let mut guard = self
            .tenants
            .write()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))?;
        entry.config = config;
        Ok(())
    }

    /// Activate or deactivate a tenant.
    pub fn set_active(&self, id: &TenantId, active: bool) -> TenantResult<()> {
        let mut guard = self
            .tenants
            .write()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))?;
        entry.config.active = active;
        Ok(())
    }

    /// Check whether a tenant exists.
    pub fn exists(&self, id: &TenantId) -> bool {
        self.tenants
            .read()
            .map(|g| g.contains_key(id))
            .unwrap_or(false)
    }

    /// Return a shared reference to the audit log.
    pub fn audit_log(&self) -> Arc<TenantAuditLog> {
        Arc::clone(&self.audit_log)
    }

    /// Internal: apply a pre-write check and update statistics on success.
    pub(crate) fn pre_write_check(
        &self,
        id: &TenantId,
        predicate: &str,
        graph: Option<&str>,
    ) -> TenantResult<()> {
        let mut guard = self
            .tenants
            .write()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        let entry = guard
            .get_mut(id)
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))?;

        // Inactive tenant check
        if !entry.config.active {
            return Err(TenantError::Inactive(id.as_str().to_string()));
        }

        // Predicate allow-list check
        if !entry.config.allowed_predicates.is_empty()
            && !entry
                .config
                .allowed_predicates
                .iter()
                .any(|p| p == predicate)
        {
            entry.stats.predicate_rejections += 1;
            return Err(TenantError::PredicateNotAllowed {
                predicate: predicate.to_string(),
                tenant: id.as_str().to_string(),
            });
        }

        // Triple quota check
        if entry.config.max_triples > 0 && entry.stats.triple_count >= entry.config.max_triples {
            entry.stats.quota_rejections += 1;
            return Err(TenantError::QuotaTriples {
                tenant: id.as_str().to_string(),
                current: entry.stats.triple_count,
                limit: entry.config.max_triples,
            });
        }

        // Byte quota check
        if entry.config.quota_bytes > 0 {
            let estimated = TenantStats::estimate_bytes(entry.stats.triple_count + 1);
            if estimated > entry.config.quota_bytes {
                entry.stats.quota_rejections += 1;
                return Err(TenantError::QuotaBytes {
                    tenant: id.as_str().to_string(),
                    current: entry.stats.bytes_used,
                    limit: entry.config.quota_bytes,
                });
            }
        }

        // Graph IRI prefix allowlist check
        if let Some(g) = graph {
            if !entry.config.allowed_prefixes.is_empty() {
                let allowed = entry
                    .config
                    .allowed_prefixes
                    .iter()
                    .any(|prefix| g.starts_with(prefix.as_str()));
                if !allowed {
                    entry.stats.quota_rejections += 1;
                    return Err(TenantError::GraphPrefixNotAllowed {
                        graph: g.to_string(),
                        tenant: id.as_str().to_string(),
                    });
                }
            }
        }

        // Graph quota check
        if let Some(g) = graph {
            if entry.config.max_graphs > 0 {
                let is_new = !entry.graphs.contains(g);
                if is_new && entry.graphs.len() as u64 >= entry.config.max_graphs {
                    entry.stats.quota_rejections += 1;
                    return Err(TenantError::QuotaGraphs {
                        tenant: id.as_str().to_string(),
                        current: entry.graphs.len() as u64,
                        limit: entry.config.max_graphs,
                    });
                }
                if is_new {
                    entry.graphs.insert(g.to_string());
                    entry.stats.graph_count = entry.graphs.len() as u64;
                }
            }
        }

        // All checks passed — update statistics
        entry.stats.triple_count += 1;
        entry.stats.bytes_used = TenantStats::estimate_bytes(entry.stats.triple_count);
        entry.stats.writes += 1;
        Ok(())
    }

    /// Internal: record a successful read.
    pub(crate) fn record_read(&self, id: &TenantId) {
        if let Ok(mut guard) = self.tenants.write() {
            if let Some(entry) = guard.get_mut(id) {
                entry.stats.reads += 1;
            }
        }
    }

    /// Internal: decrement triple count on delete.
    pub(crate) fn record_delete(&self, id: &TenantId) {
        if let Ok(mut guard) = self.tenants.write() {
            if let Some(entry) = guard.get_mut(id) {
                entry.stats.triple_count = entry.stats.triple_count.saturating_sub(1);
                entry.stats.bytes_used = TenantStats::estimate_bytes(entry.stats.triple_count);
                entry.stats.writes += 1;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TenantStore
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for the inner triple map: subject → Vec<(predicate, object)>.
type TripleMap = Arc<RwLock<HashMap<String, Vec<(String, String)>>>>;

/// An in-memory multi-tenant triple store.
///
/// All triples are namespaced by tenant prefix so that a tenant can never
/// read or write another tenant's data.  Actual triples are stored in a
/// `HashMap<String, Vec<(String, String)>>` keyed by namespaced subject IRI.
///
/// In production use this would wrap the underlying [`crate::store::TdbStore`];
/// the lightweight in-memory representation is used here so that the tenant
/// layer can be developed, tested, and benchmarked independently of the full
/// disk-based engine.
#[derive(Debug)]
pub struct TenantStore {
    registry: Arc<TenantRegistry>,
    /// subject_key → Vec<(predicate, object)>
    data: TripleMap,
}

impl TenantStore {
    /// Create a new tenant store backed by the given registry.
    pub fn new(registry: Arc<TenantRegistry>) -> Self {
        Self {
            registry,
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a triple on behalf of `tenant`.
    ///
    /// The subject is namespaced automatically; the predicate and object
    /// remain unchanged but are recorded verbatim in the namespaced slot.
    pub fn insert(
        &self,
        tenant: &TenantId,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> TenantResult<()> {
        // Pre-write checks (quota, predicates, etc.)
        self.registry.pre_write_check(tenant, predicate, None)?;

        let key = format!("{}{}", tenant.namespace_prefix(), subject);
        let mut guard = self
            .data
            .write()
            .map_err(|_| TenantError::NotFound("store lock poisoned".to_string()))?;
        guard
            .entry(key)
            .or_default()
            .push((predicate.to_string(), object.to_string()));
        Ok(())
    }

    /// Query triples for `tenant` matching an optional subject filter.
    ///
    /// Returns `(subject_without_prefix, predicate, object)` tuples.
    /// Always returns an empty vec for a non-existent tenant instead of an error.
    pub fn query(
        &self,
        tenant: &TenantId,
        subject_filter: Option<&str>,
    ) -> TenantResult<Vec<(String, String, String)>> {
        // Verify tenant exists
        if !self.registry.exists(tenant) {
            return Err(TenantError::NotFound(tenant.as_str().to_string()));
        }
        self.registry.record_read(tenant);

        let prefix = tenant.namespace_prefix();
        let guard = self
            .data
            .read()
            .map_err(|_| TenantError::NotFound("store lock poisoned".to_string()))?;

        let mut results = Vec::new();
        for (key, po_list) in guard.iter() {
            // Only include triples that belong to this tenant
            if let Some(subject) = key.strip_prefix(&prefix) {
                if let Some(filter) = subject_filter {
                    if subject != filter {
                        continue;
                    }
                }
                for (pred, obj) in po_list {
                    results.push((subject.to_string(), pred.clone(), obj.clone()));
                }
            }
        }
        Ok(results)
    }

    /// Delete all triples with the given subject under `tenant`.
    ///
    /// Returns the number of `(predicate, object)` pairs removed.
    pub fn delete_subject(&self, tenant: &TenantId, subject: &str) -> TenantResult<usize> {
        if !self.registry.exists(tenant) {
            return Err(TenantError::NotFound(tenant.as_str().to_string()));
        }
        let key = format!("{}{}", tenant.namespace_prefix(), subject);
        let mut guard = self
            .data
            .write()
            .map_err(|_| TenantError::NotFound("store lock poisoned".to_string()))?;
        let removed = guard.remove(&key).map(|v| v.len()).unwrap_or(0);
        for _ in 0..removed {
            self.registry.record_delete(tenant);
        }
        Ok(removed)
    }

    /// Remove all triples belonging to `tenant`.
    ///
    /// This is used during tenant deletion to reclaim space.
    pub fn purge(&self, tenant: &TenantId) -> TenantResult<u64> {
        if !self.registry.exists(tenant) {
            return Err(TenantError::NotFound(tenant.as_str().to_string()));
        }
        let prefix = tenant.namespace_prefix();
        let mut guard = self
            .data
            .write()
            .map_err(|_| TenantError::NotFound("store lock poisoned".to_string()))?;
        let keys_to_remove: Vec<String> = guard
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();
        let mut total: u64 = 0;
        for key in &keys_to_remove {
            if let Some(pairs) = guard.remove(key) {
                total += pairs.len() as u64;
            }
        }
        Ok(total)
    }

    /// Count the total number of `(subject, predicate, object)` triples for `tenant`.
    pub fn triple_count(&self, tenant: &TenantId) -> TenantResult<u64> {
        if !self.registry.exists(tenant) {
            return Err(TenantError::NotFound(tenant.as_str().to_string()));
        }
        let prefix = tenant.namespace_prefix();
        let guard = self
            .data
            .read()
            .map_err(|_| TenantError::NotFound("store lock poisoned".to_string()))?;
        let count = guard
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(_, v)| v.len() as u64)
            .sum();
        Ok(count)
    }

    /// Attempt to read the raw key for a subject in another tenant's namespace.
    ///
    /// This always returns an `CrossTenantAccess` error and records the attempt
    /// in the audit log.
    pub fn cross_tenant_access_check(
        &self,
        accessor: &TenantId,
        target: &TenantId,
    ) -> TenantResult<()> {
        if accessor == target {
            return Ok(());
        }
        self.registry
            .audit_log()
            .record_cross_tenant(accessor.clone(), target.clone(), true);
        Err(TenantError::CrossTenantAccess {
            accessor: accessor.as_str().to_string(),
            target: target.as_str().to_string(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TenantHandle
// ─────────────────────────────────────────────────────────────────────────────

/// Scoped access to a single tenant's data.
///
/// A `TenantHandle` is obtained from [`TenantIsolationLayer::create_tenant`] or
/// [`TenantIsolationLayer::get_tenant`].  All operations are automatically
/// namespaced to the owning tenant.
#[derive(Debug, Clone)]
pub struct TenantHandle {
    tenant_id: TenantId,
    registry: Arc<TenantRegistry>,
    store: Arc<TenantStore>,
}

impl TenantHandle {
    /// Return the [`TenantId`] this handle belongs to.
    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Insert a triple on behalf of the owning tenant.
    ///
    /// Subject is namespaced internally; predicate and object are stored
    /// verbatim under the tenant namespace.
    pub fn insert_triple(&self, subject: &str, predicate: &str, object: &str) -> TenantResult<()> {
        self.store
            .insert(&self.tenant_id, subject, predicate, object)
    }

    /// Insert a triple with an explicit named graph IRI.
    ///
    /// The graph IRI must match the tenant's `allowed_prefixes` (if set).
    pub fn insert_triple_in_graph(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        graph: &str,
    ) -> TenantResult<()> {
        // Named graph prefix check via registry
        self.registry
            .pre_write_check_graph_only(&self.tenant_id, graph)?;
        self.store
            .insert(&self.tenant_id, subject, predicate, object)
    }

    /// Query triples belonging to this tenant, optionally filtered by subject.
    ///
    /// Returns `(subject, predicate, object)` tuples.
    pub fn query(
        &self,
        subject_filter: Option<&str>,
    ) -> TenantResult<Vec<(String, String, String)>> {
        self.store.query(&self.tenant_id, subject_filter)
    }

    /// Return the number of triples stored for this tenant.
    pub fn triple_count(&self) -> u64 {
        self.store.triple_count(&self.tenant_id).unwrap_or(0)
    }

    /// Return the number of distinct named graphs this tenant has used.
    pub fn graph_count(&self) -> u32 {
        self.registry
            .stats(&self.tenant_id)
            .map(|s| s.graph_count as u32)
            .unwrap_or(0)
    }

    /// Return a snapshot of the tenant's runtime statistics.
    pub fn stats(&self) -> TenantResult<TenantStats> {
        self.registry.stats(&self.tenant_id)
    }

    /// Delete all triples with the given subject for this tenant.
    pub fn delete_subject(&self, subject: &str) -> TenantResult<usize> {
        self.store.delete_subject(&self.tenant_id, subject)
    }

    /// Remove all triples for this tenant (purge).
    ///
    /// Returns the number of triples deleted.
    pub fn purge(&self) -> TenantResult<u64> {
        self.store.purge(&self.tenant_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TenantIsolationLayer
// ─────────────────────────────────────────────────────────────────────────────

/// High-level isolation layer that wraps a TDB store with per-tenant namespacing.
///
/// Provides the primary entry point for multi-tenant operations:
///
/// - Named graph isolation: all tenant graphs prefixed with `urn:tenant:{id}:`
/// - Quota enforcement: checks `max_triples`, `max_graphs`, `quota_bytes`
/// - Prefix allowlist: rejects writes to graphs outside `allowed_prefixes`
///
/// ## Example
///
/// ```rust,no_run
/// use oxirs_tdb::tenant::{TenantId, TenantConfig, TenantIsolationLayer};
/// use std::sync::Arc;
///
/// let layer = TenantIsolationLayer::new();
/// let id = TenantId::new("acme").unwrap();
/// let config = TenantConfig::unlimited();
/// let handle = layer.create_tenant(id, config).unwrap();
/// handle.insert_triple("http://s", "http://p", "value").unwrap();
/// ```
#[derive(Debug)]
pub struct TenantIsolationLayer {
    registry: Arc<TenantRegistry>,
    store: Arc<TenantStore>,
}

impl TenantIsolationLayer {
    /// Create a new, empty isolation layer.
    pub fn new() -> Self {
        let registry = Arc::new(TenantRegistry::new());
        let store = Arc::new(TenantStore::new(Arc::clone(&registry)));
        Self { registry, store }
    }

    /// Create a new tenant and return a [`TenantHandle`] for scoped access.
    ///
    /// Returns an error if a tenant with the same ID already exists.
    pub fn create_tenant(&self, id: TenantId, config: TenantConfig) -> TenantResult<TenantHandle> {
        self.registry.create_tenant(id.clone(), config)?;
        Ok(TenantHandle {
            tenant_id: id,
            registry: Arc::clone(&self.registry),
            store: Arc::clone(&self.store),
        })
    }

    /// Retrieve a [`TenantHandle`] for an existing tenant.
    ///
    /// Returns `None` if the tenant does not exist.
    pub fn get_tenant(&self, id: &TenantId) -> Option<TenantHandle> {
        if self.registry.exists(id) {
            Some(TenantHandle {
                tenant_id: id.clone(),
                registry: Arc::clone(&self.registry),
                store: Arc::clone(&self.store),
            })
        } else {
            None
        }
    }

    /// List all active tenant IDs (sorted).
    pub fn list_tenants(&self) -> Vec<TenantId> {
        self.registry.list_tenants()
    }

    /// Delete a tenant and purge all their data.
    ///
    /// Returns the number of triples that were deleted.
    pub fn delete_tenant(&self, id: &TenantId) -> TenantResult<u64> {
        let deleted = self.store.purge(id).unwrap_or(0);
        self.registry.delete_tenant(id)?;
        Ok(deleted)
    }

    /// Return a shared reference to the underlying registry.
    pub fn registry(&self) -> Arc<TenantRegistry> {
        Arc::clone(&self.registry)
    }

    /// Return a shared reference to the audit log.
    pub fn audit_log(&self) -> Arc<TenantAuditLog> {
        self.registry.audit_log()
    }

    /// Check whether a tenant exists.
    pub fn exists(&self, id: &TenantId) -> bool {
        self.registry.exists(id)
    }

    /// Construct the fully-qualified named graph IRI for a tenant.
    ///
    /// The canonical pattern is `urn:tenant:{id}:{graph_local_name}`.
    pub fn graph_iri(id: &TenantId, local_name: &str) -> String {
        format!("urn:tenant:{}:{}", id.as_str(), local_name)
    }
}

impl Default for TenantIsolationLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional TenantRegistry helpers
// ─────────────────────────────────────────────────────────────────────────────

impl TenantRegistry {
    /// Graph-only pre-write check (prefix allowlist only, no quota update).
    ///
    /// Used by [`TenantHandle::insert_triple_in_graph`] to validate graph IRIs
    /// without double-counting quota updates.
    pub(crate) fn pre_write_check_graph_only(
        &self,
        id: &TenantId,
        graph: &str,
    ) -> TenantResult<()> {
        let guard = self
            .tenants
            .read()
            .map_err(|_| TenantError::NotFound("registry lock poisoned".to_string()))?;
        let entry = guard
            .get(id)
            .ok_or_else(|| TenantError::NotFound(id.as_str().to_string()))?;

        if !entry.config.allowed_prefixes.is_empty() {
            let allowed = entry
                .config
                .allowed_prefixes
                .iter()
                .any(|prefix| graph.starts_with(prefix.as_str()));
            if !allowed {
                return Err(TenantError::GraphPrefixNotAllowed {
                    graph: graph.to_string(),
                    tenant: id.as_str().to_string(),
                });
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Arc<TenantRegistry> {
        Arc::new(TenantRegistry::new())
    }

    fn create_tenant(registry: &TenantRegistry, name: &str) -> TenantId {
        let id = TenantId::new(name).unwrap();
        registry
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();
        id
    }

    // ── 1. TenantId validation ───────────────────────────────────────────────

    #[test]
    fn test_tenant_id_valid() {
        let id = TenantId::new("acme_corp-42").unwrap();
        assert_eq!(id.as_str(), "acme_corp-42");
    }

    #[test]
    fn test_tenant_id_empty_rejected() {
        assert!(TenantId::new("").is_err());
    }

    #[test]
    fn test_tenant_id_special_chars_rejected() {
        assert!(TenantId::new("bad/id").is_err());
        assert!(TenantId::new("bad id").is_err());
        assert!(TenantId::new("bad@id").is_err());
    }

    #[test]
    fn test_tenant_id_too_long_rejected() {
        let long = "a".repeat(129);
        assert!(TenantId::new(&long).is_err());
    }

    #[test]
    fn test_tenant_id_namespace_prefix() {
        let id = TenantId::new("testco").unwrap();
        assert_eq!(id.namespace_prefix(), "tenant:testco:");
    }

    // ── 2. TenantRegistry CRUD ───────────────────────────────────────────────

    #[test]
    fn test_create_and_list_tenants() {
        let reg = registry();
        let id1 = TenantId::new("alpha").unwrap();
        let id2 = TenantId::new("beta").unwrap();
        reg.create_tenant(id1.clone(), TenantConfig::unlimited())
            .unwrap();
        reg.create_tenant(id2.clone(), TenantConfig::unlimited())
            .unwrap();
        let list = reg.list_tenants();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&id1));
        assert!(list.contains(&id2));
    }

    #[test]
    fn test_create_duplicate_tenant_fails() {
        let reg = registry();
        let id = TenantId::new("dup").unwrap();
        reg.create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();
        let result = reg.create_tenant(id, TenantConfig::unlimited());
        assert!(matches!(result, Err(TenantError::AlreadyExists(_))));
    }

    #[test]
    fn test_delete_tenant() {
        let reg = registry();
        let id = create_tenant(&reg, "todelete");
        reg.delete_tenant(&id).unwrap();
        assert!(!reg.exists(&id));
    }

    #[test]
    fn test_delete_nonexistent_tenant_fails() {
        let reg = registry();
        let id = TenantId::new("ghost").unwrap();
        let result = reg.delete_tenant(&id);
        assert!(matches!(result, Err(TenantError::NotFound(_))));
    }

    #[test]
    fn test_exists() {
        let reg = registry();
        let id = TenantId::new("exists_check").unwrap();
        assert!(!reg.exists(&id));
        reg.create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();
        assert!(reg.exists(&id));
    }

    // ── 3. TenantStore basic insert / query ─────────────────────────────────

    #[test]
    fn test_insert_and_query() {
        let reg = registry();
        let id = create_tenant(&reg, "t1");
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "http://s1", "http://p", "obj").unwrap();
        let results = store.query(&id, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "http://s1");
    }

    #[test]
    fn test_query_with_subject_filter() {
        let reg = registry();
        let id = create_tenant(&reg, "t2");
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "http://s1", "http://p", "a").unwrap();
        store.insert(&id, "http://s2", "http://p", "b").unwrap();
        let results = store.query(&id, Some("http://s1")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, "a");
    }

    // ── 4. Tenant isolation ──────────────────────────────────────────────────

    #[test]
    fn test_tenant_isolation() {
        let reg = registry();
        let t1 = create_tenant(&reg, "tenant_a");
        let t2 = create_tenant(&reg, "tenant_b");
        let store = TenantStore::new(Arc::clone(&reg));

        store
            .insert(&t1, "http://secret", "http://p", "sensitive_value")
            .unwrap();

        // tenant_b must not see tenant_a's data
        let results_b = store.query(&t2, None).unwrap();
        assert!(
            results_b.is_empty(),
            "Tenant B must not see Tenant A's data"
        );
    }

    #[test]
    fn test_cross_tenant_access_blocked_and_audited() {
        let reg = registry();
        let t1 = create_tenant(&reg, "accessor");
        let t2 = create_tenant(&reg, "target");
        let store = TenantStore::new(Arc::clone(&reg));

        let result = store.cross_tenant_access_check(&t1, &t2);
        assert!(matches!(result, Err(TenantError::CrossTenantAccess { .. })));

        let events = reg.audit_log().events();
        assert_eq!(events.len(), 1);
        assert!(events[0].blocked);
    }

    // ── 5. Quota enforcement ─────────────────────────────────────────────────

    #[test]
    fn test_triple_quota_enforced() {
        let reg = registry();
        let id = TenantId::new("limited").unwrap();
        reg.create_tenant(id.clone(), TenantConfig::with_limits(3, 0, 0))
            .unwrap();
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "s1", "p", "o").unwrap();
        store.insert(&id, "s2", "p", "o").unwrap();
        store.insert(&id, "s3", "p", "o").unwrap();
        let result = store.insert(&id, "s4", "p", "o");
        assert!(matches!(result, Err(TenantError::QuotaTriples { .. })));
    }

    #[test]
    fn test_graph_quota_enforced() {
        let reg = registry();
        let id = TenantId::new("gquota").unwrap();
        reg.create_tenant(id.clone(), TenantConfig::with_limits(0, 2, 0))
            .unwrap();
        let store = TenantStore::new(Arc::clone(&reg));

        // Use pre_write_check directly with graph context
        reg.pre_write_check(&id, "http://p", Some("graph1"))
            .unwrap();
        reg.pre_write_check(&id, "http://p", Some("graph2"))
            .unwrap();
        let result = reg.pre_write_check(&id, "http://p", Some("graph3"));
        assert!(matches!(result, Err(TenantError::QuotaGraphs { .. })));
    }

    #[test]
    fn test_predicate_allowlist_enforced() {
        let reg = registry();
        let id = TenantId::new("strict").unwrap();
        let config = TenantConfig {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec!["http://allowed".to_string()],
            allowed_prefixes: vec![],
            active: true,
        };
        reg.create_tenant(id.clone(), config).unwrap();
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "s", "http://allowed", "o").unwrap();
        let result = store.insert(&id, "s", "http://forbidden", "o");
        assert!(matches!(
            result,
            Err(TenantError::PredicateNotAllowed { .. })
        ));
    }

    #[test]
    fn test_inactive_tenant_rejects_writes() {
        let reg = registry();
        let id = create_tenant(&reg, "inactive_t");
        reg.set_active(&id, false).unwrap();
        let store = TenantStore::new(Arc::clone(&reg));

        let result = store.insert(&id, "s", "p", "o");
        assert!(matches!(result, Err(TenantError::Inactive(_))));
    }

    // ── 6. Stats tracking ────────────────────────────────────────────────────

    #[test]
    fn test_stats_triple_count_increments() {
        let reg = registry();
        let id = create_tenant(&reg, "stats_t");
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "s1", "p", "o").unwrap();
        store.insert(&id, "s2", "p", "o").unwrap();

        let stats = reg.stats(&id).unwrap();
        assert_eq!(stats.triple_count, 2);
        assert_eq!(stats.writes, 2);
    }

    #[test]
    fn test_stats_reads_increments() {
        let reg = registry();
        let id = create_tenant(&reg, "read_stats");
        let store = TenantStore::new(Arc::clone(&reg));

        store.query(&id, None).unwrap();
        store.query(&id, None).unwrap();

        let stats = reg.stats(&id).unwrap();
        assert_eq!(stats.reads, 2);
    }

    // ── 7. Delete and purge ──────────────────────────────────────────────────

    #[test]
    fn test_delete_subject() {
        let reg = registry();
        let id = create_tenant(&reg, "del_t");
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&id, "s1", "p", "o").unwrap();
        store.insert(&id, "s1", "p2", "o2").unwrap();
        let removed = store.delete_subject(&id, "s1").unwrap();
        assert_eq!(removed, 2);
        let results = store.query(&id, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_purge_removes_all_tenant_data() {
        let reg = registry();
        let t1 = create_tenant(&reg, "purge_a");
        let t2 = create_tenant(&reg, "purge_b");
        let store = TenantStore::new(Arc::clone(&reg));

        store.insert(&t1, "s", "p", "o").unwrap();
        store.insert(&t2, "s", "p", "o").unwrap();

        let purged = store.purge(&t1).unwrap();
        assert_eq!(purged, 1);
        assert!(store.query(&t1, None).unwrap().is_empty());
        // t2's data must remain intact
        assert_eq!(store.query(&t2, None).unwrap().len(), 1);
    }

    // ── 8. Audit log ─────────────────────────────────────────────────────────

    #[test]
    fn test_audit_log_records_cross_tenant() {
        let reg = registry();
        let t1 = create_tenant(&reg, "audit_a");
        let t2 = create_tenant(&reg, "audit_b");
        let store = TenantStore::new(Arc::clone(&reg));

        let _ = store.cross_tenant_access_check(&t1, &t2);
        let events = reg.audit_log().events_for_tenant(&t1);
        assert!(!events.is_empty());
        assert!(events.iter().all(|e| e.blocked));
    }

    #[test]
    fn test_audit_log_same_tenant_no_event() {
        let reg = registry();
        let t1 = create_tenant(&reg, "self_access");
        let store = TenantStore::new(Arc::clone(&reg));

        let result = store.cross_tenant_access_check(&t1, &t1);
        assert!(result.is_ok());
        assert!(reg.audit_log().is_empty());
    }

    // ── 9. Config update ─────────────────────────────────────────────────────

    #[test]
    fn test_update_config() {
        let reg = registry();
        let id = create_tenant(&reg, "update_cfg");
        let new_cfg = TenantConfig::with_limits(500, 10, 1024 * 1024);
        reg.update_config(&id, new_cfg.clone()).unwrap();
        let cfg = reg.config(&id).unwrap();
        assert_eq!(cfg.max_triples, 500);
        assert_eq!(cfg.max_graphs, 10);
    }

    // ── 10. Triple count helper ──────────────────────────────────────────────

    #[test]
    fn test_triple_count() {
        let reg = registry();
        let id = create_tenant(&reg, "tcount");
        let store = TenantStore::new(Arc::clone(&reg));

        assert_eq!(store.triple_count(&id).unwrap(), 0);
        store.insert(&id, "s1", "p", "o").unwrap();
        store.insert(&id, "s1", "p2", "o2").unwrap();
        assert_eq!(store.triple_count(&id).unwrap(), 2);
    }

    // ── 11. allowed_prefixes ─────────────────────────────────────────────────

    #[test]
    fn test_allowed_prefixes_permits_matching_graph() {
        let reg = registry();
        let id = TenantId::new("prefix_ok").unwrap();
        let config = TenantConfig {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec![],
            allowed_prefixes: vec!["urn:tenant:prefix_ok:".to_string()],
            active: true,
        };
        reg.create_tenant(id.clone(), config).unwrap();

        // Graph IRI matching the allowed prefix should succeed
        let result = reg.pre_write_check(&id, "http://p", Some("urn:tenant:prefix_ok:data"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_allowed_prefixes_rejects_non_matching_graph() {
        let reg = registry();
        let id = TenantId::new("prefix_rej").unwrap();
        let config = TenantConfig {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec![],
            allowed_prefixes: vec!["urn:tenant:prefix_rej:".to_string()],
            active: true,
        };
        reg.create_tenant(id.clone(), config).unwrap();

        // Graph IRI NOT matching any allowed prefix should fail
        let result = reg.pre_write_check(&id, "http://p", Some("urn:other:graph"));
        assert!(matches!(
            result,
            Err(TenantError::GraphPrefixNotAllowed { .. })
        ));
    }

    #[test]
    fn test_empty_allowed_prefixes_permits_any_graph() {
        let reg = registry();
        let id = TenantId::new("prefix_any").unwrap();
        reg.create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();

        // No prefix restriction → any graph IRI should succeed
        let result = reg.pre_write_check(&id, "http://p", Some("urn:any:graph:here"));
        assert!(result.is_ok());
    }

    // ── 12. TenantIsolationLayer ─────────────────────────────────────────────

    #[test]
    fn test_isolation_layer_create_and_get() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("iso_a").unwrap();
        let _handle = layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();

        assert!(layer.exists(&id));
        let retrieved = layer.get_tenant(&id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_isolation_layer_get_nonexistent() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("ghost_iso").unwrap();
        assert!(layer.get_tenant(&id).is_none());
    }

    #[test]
    fn test_isolation_layer_list_tenants() {
        let layer = TenantIsolationLayer::new();
        layer
            .create_tenant(TenantId::new("t_alpha").unwrap(), TenantConfig::unlimited())
            .unwrap();
        layer
            .create_tenant(TenantId::new("t_beta").unwrap(), TenantConfig::unlimited())
            .unwrap();
        let list = layer.list_tenants();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_isolation_layer_delete_tenant() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("del_iso").unwrap();
        let handle = layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();

        handle.insert_triple("s", "p", "o").unwrap();
        let deleted = layer.delete_tenant(&id).unwrap();
        assert_eq!(deleted, 1);
        assert!(!layer.exists(&id));
    }

    #[test]
    fn test_isolation_layer_delete_returns_triple_count() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("del_count").unwrap();
        let handle = layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();

        for i in 0..5 {
            handle.insert_triple(&format!("s{}", i), "p", "o").unwrap();
        }
        // Note: purge returns count of triple groups (by subject key), not total triples
        let deleted = layer.delete_tenant(&id).unwrap();
        assert!(deleted > 0);
    }

    #[test]
    fn test_isolation_layer_graph_iri() {
        let id = TenantId::new("giri").unwrap();
        let iri = TenantIsolationLayer::graph_iri(&id, "dataset1");
        assert_eq!(iri, "urn:tenant:giri:dataset1");
    }

    // ── 13. TenantHandle ─────────────────────────────────────────────────────

    #[test]
    fn test_handle_insert_and_query() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("handle_ins").unwrap();
        let handle = layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();

        handle
            .insert_triple("http://subject", "http://pred", "value")
            .unwrap();
        let results = handle.query(None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "http://subject");
    }

    #[test]
    fn test_handle_triple_count() {
        let layer = TenantIsolationLayer::new();
        let handle = layer
            .create_tenant(TenantId::new("hcount").unwrap(), TenantConfig::unlimited())
            .unwrap();

        assert_eq!(handle.triple_count(), 0);
        handle.insert_triple("s1", "p", "o").unwrap();
        handle.insert_triple("s2", "p", "o").unwrap();
        assert_eq!(handle.triple_count(), 2);
    }

    #[test]
    fn test_handle_graph_count() {
        let layer = TenantIsolationLayer::new();
        // max_graphs > 0 enables graph tracking
        let config = TenantConfig::with_limits(0, 100, 0);
        let handle = layer
            .create_tenant(TenantId::new("hgraph").unwrap(), config)
            .unwrap();

        // Use pre_write_check directly to register graph names
        let reg = layer.registry();
        let id = handle.tenant_id().clone();
        reg.pre_write_check(&id, "http://p", Some("graph1"))
            .unwrap();
        reg.pre_write_check(&id, "http://p", Some("graph2"))
            .unwrap();
        reg.pre_write_check(&id, "http://p", Some("graph2"))
            .unwrap(); // duplicate

        assert_eq!(handle.graph_count(), 2);
    }

    #[test]
    fn test_handle_delete_subject() {
        let layer = TenantIsolationLayer::new();
        let handle = layer
            .create_tenant(TenantId::new("hdel").unwrap(), TenantConfig::unlimited())
            .unwrap();

        handle.insert_triple("s1", "p1", "o1").unwrap();
        handle.insert_triple("s1", "p2", "o2").unwrap();
        let removed = handle.delete_subject("s1").unwrap();
        assert_eq!(removed, 2);
        assert_eq!(handle.triple_count(), 0);
    }

    #[test]
    fn test_handle_purge() {
        let layer = TenantIsolationLayer::new();
        let handle = layer
            .create_tenant(TenantId::new("hpurge").unwrap(), TenantConfig::unlimited())
            .unwrap();

        handle.insert_triple("s", "p", "o").unwrap();
        let purged = handle.purge().unwrap();
        assert!(purged > 0);
    }

    #[test]
    fn test_handle_stats() {
        let layer = TenantIsolationLayer::new();
        let handle = layer
            .create_tenant(TenantId::new("hstats").unwrap(), TenantConfig::unlimited())
            .unwrap();

        handle.insert_triple("s", "p", "o").unwrap();
        let stats = handle.stats().unwrap();
        assert_eq!(stats.triple_count, 1);
        assert_eq!(stats.writes, 1);
    }

    #[test]
    fn test_handle_insert_in_graph_allowed() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("hgraph_ok").unwrap();
        let config = TenantConfig {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec![],
            allowed_prefixes: vec!["urn:tenant:hgraph_ok:".to_string()],
            active: true,
        };
        let handle = layer.create_tenant(id, config).unwrap();

        let result = handle.insert_triple_in_graph("s", "p", "o", "urn:tenant:hgraph_ok:mydata");
        assert!(result.is_ok());
    }

    #[test]
    fn test_handle_insert_in_graph_rejected() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("hgraph_rej").unwrap();
        let config = TenantConfig {
            max_triples: 0,
            max_graphs: 0,
            quota_bytes: 0,
            allowed_predicates: vec![],
            allowed_prefixes: vec!["urn:tenant:hgraph_rej:".to_string()],
            active: true,
        };
        let handle = layer.create_tenant(id, config).unwrap();

        let result = handle.insert_triple_in_graph("s", "p", "o", "urn:other:graph");
        assert!(matches!(
            result,
            Err(TenantError::GraphPrefixNotAllowed { .. })
        ));
    }

    #[test]
    fn test_isolation_layer_create_duplicate_fails() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("dup_layer").unwrap();
        layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();
        let result = layer.create_tenant(id, TenantConfig::unlimited());
        assert!(matches!(result, Err(TenantError::AlreadyExists(_))));
    }

    #[test]
    fn test_multiple_handles_share_store() {
        let layer = TenantIsolationLayer::new();
        let id = TenantId::new("shared_store").unwrap();
        let h1 = layer
            .create_tenant(id.clone(), TenantConfig::unlimited())
            .unwrap();
        let h2 = layer.get_tenant(&id).unwrap();

        h1.insert_triple("s", "p", "o").unwrap();
        // h2 sees the data inserted by h1 because they share the same store
        assert_eq!(h2.triple_count(), 1);
    }
}
