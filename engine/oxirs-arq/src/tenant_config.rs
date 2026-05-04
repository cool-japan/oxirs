//! Per-tenant configuration for the ARQ query executor.
//!
//! This module ties together SLA classification ([`oxirs_core::sla::SlaClass`]),
//! query budget hints (timeouts, concurrency), and admission control state for
//! a single logical tenant of the SPARQL endpoint.
//!
//! It is the input feed for [`crate::sla_integration::ArqSlaGate`], which in
//! turn coordinates with [`oxirs_core::sla::AdmissionController`] and
//! [`oxirs_core::sla::PriorityDispatcher`].

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use oxirs_core::sla::{SlaClass, SlaThresholds};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// TenantConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration assigned to a single SPARQL tenant.
///
/// Each tenant is mapped to exactly one [`SlaClass`] which determines the
/// thresholds applied when admitting and dispatching their queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Logical tenant identifier (matches the `X-Tenant-Id` header or similar).
    pub tenant_id: String,
    /// SLA tier this tenant belongs to.
    pub sla_class: SlaClass,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Whether queries from this tenant may be rejected at admission time.
    ///
    /// When `false`, admission failures are converted to soft warnings rather
    /// than errors.  Useful for canary tenants and integration tests.
    pub strict_admission: bool,
    /// Maximum cost factor in token units that a single query may consume.
    ///
    /// Override-able per tenant.  Defaults to 1.0 (one token = one query).
    pub max_query_cost: f64,
    /// Optional per-tenant timeout overriding the global executor timeout.
    pub query_timeout: Option<Duration>,
}

impl TenantConfig {
    /// Construct a tenant config with sensible defaults.
    pub fn new(tenant_id: impl Into<String>, sla_class: SlaClass) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            sla_class,
            label: None,
            strict_admission: true,
            max_query_cost: 1.0,
            query_timeout: None,
        }
    }

    /// Attach a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Override the per-query token cost.
    pub fn with_max_query_cost(mut self, cost: f64) -> Self {
        self.max_query_cost = cost;
        self
    }

    /// Override the per-query timeout.
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = Some(timeout);
        self
    }

    /// Toggle strict admission.
    pub fn with_strict_admission(mut self, strict: bool) -> Self {
        self.strict_admission = strict;
        self
    }

    /// Resolve the SLA threshold bundle for this tenant.
    pub fn thresholds(&self) -> SlaThresholds {
        self.sla_class.thresholds()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TenantConfigRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe registry of [`TenantConfig`] entries.
///
/// Cheap to clone — backed by an `Arc<RwLock<...>>`.  Read-heavy workloads
/// hit the read lock; write operations (register / update / remove) take the
/// write lock.
#[derive(Debug, Clone, Default)]
pub struct TenantConfigRegistry {
    inner: Arc<RwLock<HashMap<String, TenantConfig>>>,
}

impl TenantConfigRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace a tenant config.
    pub fn register(&self, config: TenantConfig) {
        let mut map = self.inner.write().unwrap_or_else(|e| e.into_inner());
        map.insert(config.tenant_id.clone(), config);
    }

    /// Look up a tenant config (clone returned to avoid lifetime ties).
    pub fn get(&self, tenant_id: &str) -> Option<TenantConfig> {
        let map = self.inner.read().unwrap_or_else(|e| e.into_inner());
        map.get(tenant_id).cloned()
    }

    /// Convenience: fetch the SLA class for a tenant, or `None` if unregistered.
    pub fn sla_class(&self, tenant_id: &str) -> Option<SlaClass> {
        self.get(tenant_id).map(|c| c.sla_class)
    }

    /// Remove a tenant.  Returns the removed config, if present.
    pub fn remove(&self, tenant_id: &str) -> Option<TenantConfig> {
        let mut map = self.inner.write().unwrap_or_else(|e| e.into_inner());
        map.remove(tenant_id)
    }

    /// Number of registered tenants.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Snapshot all tenant IDs (handy for diagnostics).
    pub fn tenant_ids(&self) -> Vec<String> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tenant_config() {
        let cfg = TenantConfig::new("alpha", SlaClass::Gold);
        assert_eq!(cfg.tenant_id, "alpha");
        assert_eq!(cfg.sla_class, SlaClass::Gold);
        assert!(cfg.label.is_none());
        assert!(cfg.strict_admission);
        assert!((cfg.max_query_cost - 1.0).abs() < 1e-9);
        assert!(cfg.query_timeout.is_none());

        let thresholds = cfg.thresholds();
        assert_eq!(thresholds.max_concurrent_queries, 20);
    }

    #[test]
    fn test_builder_pattern() {
        let cfg = TenantConfig::new("beta", SlaClass::Platinum)
            .with_label("beta-corp")
            .with_max_query_cost(2.5)
            .with_query_timeout(Duration::from_millis(500))
            .with_strict_admission(false);
        assert_eq!(cfg.label.as_deref(), Some("beta-corp"));
        assert!((cfg.max_query_cost - 2.5).abs() < 1e-9);
        assert_eq!(cfg.query_timeout, Some(Duration::from_millis(500)));
        assert!(!cfg.strict_admission);
    }

    #[test]
    fn test_registry_roundtrip() {
        let registry = TenantConfigRegistry::new();
        assert!(registry.is_empty());
        registry.register(TenantConfig::new("gamma", SlaClass::Silver));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.sla_class("gamma"), Some(SlaClass::Silver));
        assert_eq!(registry.sla_class("delta"), None);

        let removed = registry.remove("gamma");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_replaces_on_reregister() {
        let registry = TenantConfigRegistry::new();
        registry.register(TenantConfig::new("epsilon", SlaClass::Bronze));
        assert_eq!(registry.sla_class("epsilon"), Some(SlaClass::Bronze));
        registry.register(TenantConfig::new("epsilon", SlaClass::Platinum));
        assert_eq!(registry.sla_class("epsilon"), Some(SlaClass::Platinum));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_clone_shares_state() {
        let r1 = TenantConfigRegistry::new();
        let r2 = r1.clone();
        r1.register(TenantConfig::new("zeta", SlaClass::Gold));
        assert_eq!(r2.sla_class("zeta"), Some(SlaClass::Gold));
    }

    #[test]
    fn test_serde_roundtrip_tenant_config() {
        let cfg = TenantConfig::new("eta", SlaClass::Gold).with_label("eta-corp");
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: TenantConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.tenant_id, "eta");
        assert_eq!(back.sla_class, SlaClass::Gold);
        assert_eq!(back.label.as_deref(), Some("eta-corp"));
    }

    #[test]
    fn test_tenant_ids() {
        let registry = TenantConfigRegistry::new();
        registry.register(TenantConfig::new("a", SlaClass::Bronze));
        registry.register(TenantConfig::new("b", SlaClass::Silver));
        let mut ids = registry.tenant_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
