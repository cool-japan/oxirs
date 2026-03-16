//! Per-tenant GraphQL schema management.
//!
//! Each tenant can register an independent GraphQL schema string.  The
//! registry stores these schemas by tenant ID and exposes methods for
//! registration, retrieval, listing, and removal.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};

/// A stored GraphQL schema for a single tenant.
#[derive(Debug, Clone)]
pub struct TenantSchema {
    /// The tenant this schema belongs to.
    pub tenant_id: String,
    /// The SDL (Schema Definition Language) source text.
    pub sdl: String,
    /// An opaque version tag updated on each `register_schema` call.
    pub version: u64,
}

impl TenantSchema {
    fn new(tenant_id: impl Into<String>, sdl: impl Into<String>, version: u64) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            sdl: sdl.into(),
            version,
        }
    }
}

/// Thread-safe per-tenant GraphQL schema registry.
///
/// Tenants may have completely different schemas — the registry keeps them
/// isolated. Registration is idempotent: re-registering an existing tenant
/// replaces the schema and increments the version counter.
pub struct TenantSchemaRegistry {
    schemas: Arc<RwLock<HashMap<String, TenantSchema>>>,
    versions: Arc<RwLock<HashMap<String, u64>>>,
}

impl std::fmt::Debug for TenantSchemaRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.schemas.read().map(|s| s.len()).unwrap_or(0);
        f.debug_struct("TenantSchemaRegistry")
            .field("tenant_count", &count)
            .finish()
    }
}

impl TenantSchemaRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register (or replace) a tenant's schema.
    ///
    /// Returns an error only if the internal lock is poisoned.
    pub fn register_schema(&self, tenant_id: &str, sdl: &str) -> Result<()> {
        let mut versions = self
            .versions
            .write()
            .map_err(|_| anyhow!("TenantSchemaRegistry versions lock poisoned"))?;
        let version = versions
            .entry(tenant_id.to_string())
            .and_modify(|v| *v += 1)
            .or_insert(1);
        let schema = TenantSchema::new(tenant_id, sdl, *version);

        self.schemas
            .write()
            .map_err(|_| anyhow!("TenantSchemaRegistry schemas lock poisoned"))?
            .insert(tenant_id.to_string(), schema);
        Ok(())
    }

    /// Retrieve a tenant's schema by ID.
    ///
    /// Returns `None` if the tenant has no registered schema.
    pub fn get_schema(&self, tenant_id: &str) -> Option<TenantSchema> {
        self.schemas.read().ok()?.get(tenant_id).cloned()
    }

    /// Return the IDs of all tenants with a registered schema.
    pub fn list_tenants(&self) -> Vec<String> {
        self.schemas
            .read()
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Remove a tenant's schema from the registry.
    ///
    /// Returns `true` if the tenant had a schema that was removed.
    pub fn remove_schema(&self, tenant_id: &str) -> bool {
        self.schemas
            .write()
            .map(|mut s| s.remove(tenant_id).is_some())
            .unwrap_or(false)
    }

    /// Return the number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.schemas.read().map(|s| s.len()).unwrap_or(0)
    }
}

impl Default for TenantSchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sdl() -> &'static str {
        r#"type Query { hello: String }"#
    }

    #[test]
    fn test_register_and_get_schema() {
        let reg = TenantSchemaRegistry::new();
        reg.register_schema("acme", sample_sdl())
            .expect("register ok");
        let schema = reg.get_schema("acme").expect("should exist");
        assert_eq!(schema.tenant_id, "acme");
        assert_eq!(schema.sdl, sample_sdl());
        assert_eq!(schema.version, 1);
    }

    #[test]
    fn test_get_missing_schema_returns_none() {
        let reg = TenantSchemaRegistry::new();
        assert!(reg.get_schema("ghost").is_none());
    }

    #[test]
    fn test_register_increments_version() {
        let reg = TenantSchemaRegistry::new();
        reg.register_schema("t1", "type Query { v1: String }")
            .expect("should succeed");
        reg.register_schema("t1", "type Query { v2: String }")
            .expect("should succeed");
        let schema = reg.get_schema("t1").expect("should succeed");
        assert_eq!(schema.version, 2);
        assert!(schema.sdl.contains("v2"));
    }

    #[test]
    fn test_list_tenants() {
        let reg = TenantSchemaRegistry::new();
        reg.register_schema("a", sample_sdl())
            .expect("should succeed");
        reg.register_schema("b", sample_sdl())
            .expect("should succeed");
        let mut tenants = reg.list_tenants();
        tenants.sort();
        assert_eq!(tenants, vec!["a", "b"]);
    }

    #[test]
    fn test_remove_schema() {
        let reg = TenantSchemaRegistry::new();
        reg.register_schema("gone", sample_sdl())
            .expect("should succeed");
        assert!(reg.remove_schema("gone"));
        assert!(reg.get_schema("gone").is_none());
        assert!(!reg.remove_schema("gone")); // already removed
    }

    #[test]
    fn test_tenant_count() {
        let reg = TenantSchemaRegistry::new();
        assert_eq!(reg.tenant_count(), 0);
        reg.register_schema("t1", sample_sdl())
            .expect("should succeed");
        assert_eq!(reg.tenant_count(), 1);
        reg.register_schema("t2", sample_sdl())
            .expect("should succeed");
        assert_eq!(reg.tenant_count(), 2);
        reg.remove_schema("t1");
        assert_eq!(reg.tenant_count(), 1);
    }

    #[test]
    fn test_multiple_tenants_isolated() {
        let reg = TenantSchemaRegistry::new();
        reg.register_schema("tenant_a", "type Query { a: String }")
            .expect("should succeed");
        reg.register_schema("tenant_b", "type Query { b: String }")
            .expect("should succeed");

        let a = reg.get_schema("tenant_a").expect("should succeed");
        let b = reg.get_schema("tenant_b").expect("should succeed");
        assert!(a.sdl.contains("a: String"));
        assert!(b.sdl.contains("b: String"));
        assert_ne!(a.sdl, b.sdl);
    }
}
