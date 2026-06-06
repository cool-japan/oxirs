use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::types::{
    TenantAuditLog, TenantConfig, TenantEntry, TenantError, TenantId, TenantResult, TenantStats,
};

/// Central registry that manages the full lifecycle of tenants.
///
/// Thread-safe via an internal `RwLock`.
#[derive(Debug)]
pub struct TenantRegistry {
    pub(crate) tenants: Arc<RwLock<HashMap<TenantId, TenantEntry>>>,
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

        if !entry.config.active {
            return Err(TenantError::Inactive(id.as_str().to_string()));
        }

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

        if entry.config.max_triples > 0 && entry.stats.triple_count >= entry.config.max_triples {
            entry.stats.quota_rejections += 1;
            return Err(TenantError::QuotaTriples {
                tenant: id.as_str().to_string(),
                current: entry.stats.triple_count,
                limit: entry.config.max_triples,
            });
        }

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

        entry.stats.triple_count += 1;
        entry.stats.bytes_used = TenantStats::estimate_bytes(entry.stats.triple_count);
        entry.stats.writes += 1;
        Ok(())
    }

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

    pub(crate) fn record_read(&self, id: &TenantId) {
        if let Ok(mut guard) = self.tenants.write() {
            if let Some(entry) = guard.get_mut(id) {
                entry.stats.reads += 1;
            }
        }
    }

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
