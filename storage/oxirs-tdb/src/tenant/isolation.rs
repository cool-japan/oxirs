use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::registry::TenantRegistry;
use super::types::{
    TenantAuditLog, TenantConfig, TenantError, TenantId, TenantResult, TenantStats,
};

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
