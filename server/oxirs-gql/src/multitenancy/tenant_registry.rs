//! Multi-tenant Schema Registry for GraphQL over RDF
//!
//! Each tenant has isolated schema definitions, dataset access lists, and access
//! policies. The registry is thread-safe and designed for concurrent access from
//! multiple request handlers.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Operations that a tenant may be permitted to perform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TenantOperation {
    /// Allow GraphQL query execution.
    Query,
    /// Allow GraphQL mutation execution.
    Mutation,
    /// Allow GraphQL subscription registration.
    Subscription,
}

/// A custom GraphQL field mapped to an RDF predicate.
#[derive(Debug, Clone)]
pub struct TenantField {
    /// GraphQL field name (camelCase).
    pub field_name: String,
    /// The RDF predicate IRI this field maps to.
    pub rdf_predicate: String,
    /// GraphQL type name for this field (e.g. `"String"`, `"Int"`).
    pub field_type: String,
    /// Whether the field is non-nullable in the generated schema.
    pub is_required: bool,
    /// Whether the field returns a list value.
    pub is_list: bool,
}

/// A custom GraphQL object type mapped to an RDF class.
#[derive(Debug, Clone)]
pub struct TenantCustomType {
    /// GraphQL type name (PascalCase).
    pub type_name: String,
    /// The RDF class IRI this type maps to.
    pub rdf_class: String,
    /// Fields available on this type.
    pub fields: Vec<TenantField>,
}

impl TenantCustomType {
    /// Create a new custom type with no fields.
    pub fn new(type_name: impl Into<String>, rdf_class: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            rdf_class: rdf_class.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field to this type, returning `self` for chaining.
    pub fn with_field(mut self, field: TenantField) -> Self {
        self.fields.push(field);
        self
    }
}

/// Configuration for a single tenant.
#[derive(Debug, Clone)]
pub struct TenantConfig {
    /// Unique tenant identifier.
    pub tenant_id: String,
    /// Human-readable tenant name.
    pub display_name: String,
    /// Named graph IRIs that this tenant can access.
    pub datasets: Vec<String>,
    /// Maximum allowed query depth for this tenant.
    pub max_query_depth: u32,
    /// Maximum allowed query complexity for this tenant.
    pub max_query_complexity: u32,
    /// Maximum number of GraphQL requests per minute.
    pub rate_limit_rpm: u32,
    /// Set of operations this tenant is allowed to perform.
    pub allowed_operations: Vec<TenantOperation>,
    /// Custom GraphQL types exposed to this tenant.
    pub custom_types: Vec<TenantCustomType>,
}

impl TenantConfig {
    /// Create a minimal configuration for a new tenant.
    pub fn new(tenant_id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            display_name: display_name.into(),
            datasets: Vec::new(),
            max_query_depth: 10,
            max_query_complexity: 1_000,
            rate_limit_rpm: 60,
            allowed_operations: vec![TenantOperation::Query],
            custom_types: Vec::new(),
        }
    }

    /// Allow a specific dataset (named graph IRI).
    pub fn with_dataset(mut self, graph_iri: impl Into<String>) -> Self {
        self.datasets.push(graph_iri.into());
        self
    }

    /// Allow a specific operation.
    pub fn with_operation(mut self, op: TenantOperation) -> Self {
        if !self.allowed_operations.contains(&op) {
            self.allowed_operations.push(op);
        }
        self
    }

    /// Set the requests-per-minute rate limit.
    pub fn with_rate_limit(mut self, rpm: u32) -> Self {
        self.rate_limit_rpm = rpm;
        self
    }

    /// Add a custom type definition.
    pub fn with_custom_type(mut self, t: TenantCustomType) -> Self {
        self.custom_types.push(t);
        self
    }

    /// Check whether a given operation is allowed for this tenant.
    pub fn allows(&self, op: &TenantOperation) -> bool {
        self.allowed_operations.contains(op)
    }

    /// Check whether a given named graph IRI is accessible to this tenant.
    pub fn can_access_dataset(&self, graph_iri: &str) -> bool {
        self.datasets.is_empty() || self.datasets.iter().any(|d| d == graph_iri)
    }
}

/// Per-request tenant isolation context.
///
/// Attached to every request after the tenant has been authenticated.
#[derive(Debug, Clone)]
pub struct TenantContext {
    /// The resolved tenant identifier.
    pub tenant_id: String,
    /// The full tenant configuration.
    pub config: Arc<TenantConfig>,
    /// A unique request ID for tracing.
    pub request_id: String,
    /// The authenticated user, if available.
    pub authenticated_user: Option<String>,
}

impl TenantContext {
    /// Create a new tenant context for a request.
    pub fn new(
        tenant_id: impl Into<String>,
        config: Arc<TenantConfig>,
        request_id: impl Into<String>,
        authenticated_user: Option<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            config,
            request_id: request_id.into(),
            authenticated_user,
        }
    }

    /// Convenience: check whether a given operation is permitted.
    pub fn can_perform(&self, op: &TenantOperation) -> bool {
        self.config.allows(op)
    }

    /// Convenience: check dataset accessibility.
    pub fn can_access_dataset(&self, graph_iri: &str) -> bool {
        self.config.can_access_dataset(graph_iri)
    }
}

/// Thread-safe registry of all tenant configurations.
pub struct TenantRegistry {
    tenants: Arc<RwLock<HashMap<String, Arc<TenantConfig>>>>,
    default_config: Option<TenantConfig>,
}

impl TenantRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tenants: Arc::new(RwLock::new(HashMap::new())),
            default_config: None,
        }
    }

    /// Set the configuration used when no tenant header is present.
    pub fn with_default_config(mut self, config: TenantConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    /// Register a new tenant.
    ///
    /// Returns an error if a tenant with the same ID already exists.
    pub fn register_tenant(&self, config: TenantConfig) -> Result<()> {
        let tenant_id = config.tenant_id.clone();
        let mut tenants = self
            .tenants
            .write()
            .map_err(|_| anyhow!("TenantRegistry lock poisoned"))?;

        if tenants.contains_key(&tenant_id) {
            return Err(anyhow!("Tenant '{}' is already registered", tenant_id));
        }
        tenants.insert(tenant_id, Arc::new(config));
        Ok(())
    }

    /// Update an existing tenant's configuration.
    ///
    /// Returns an error if the tenant does not exist.
    pub fn update_tenant(&self, config: TenantConfig) -> Result<()> {
        let tenant_id = config.tenant_id.clone();
        let mut tenants = self
            .tenants
            .write()
            .map_err(|_| anyhow!("TenantRegistry lock poisoned"))?;

        if !tenants.contains_key(&tenant_id) {
            return Err(anyhow!("Tenant '{}' not found", tenant_id));
        }
        tenants.insert(tenant_id, Arc::new(config));
        Ok(())
    }

    /// Remove a tenant from the registry.
    ///
    /// Returns `true` if the tenant existed and was removed, `false` otherwise.
    pub fn deregister_tenant(&self, tenant_id: &str) -> bool {
        self.tenants
            .write()
            .map(|mut t| t.remove(tenant_id).is_some())
            .unwrap_or(false)
    }

    /// Retrieve a tenant's configuration by ID.
    pub fn get_tenant(&self, tenant_id: &str) -> Option<Arc<TenantConfig>> {
        self.tenants
            .read()
            .ok()
            .and_then(|t| t.get(tenant_id).cloned())
    }

    /// Return a list of all registered tenant IDs.
    pub fn list_tenants(&self) -> Vec<String> {
        self.tenants
            .read()
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Return the number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().map(|t| t.len()).unwrap_or(0)
    }

    /// Build a `TenantContext` for a request.
    ///
    /// If `tenant_id` is `None` and a default configuration has been set, the
    /// default tenant is used.  Returns an error if the tenant is unknown and
    /// no default is configured.
    pub fn create_context(
        &self,
        tenant_id: Option<&str>,
        request_id: String,
    ) -> Result<TenantContext> {
        let config = match tenant_id {
            Some(id) => self
                .get_tenant(id)
                .ok_or_else(|| anyhow!("Unknown tenant '{}'", id))?,
            None => {
                // Try default configuration
                match &self.default_config {
                    Some(default) => Arc::new(default.clone()),
                    None => {
                        return Err(anyhow!(
                            "No tenant ID provided and no default tenant configured"
                        ))
                    }
                }
            }
        };

        let resolved_id = config.tenant_id.clone();
        Ok(TenantContext::new(resolved_id, config, request_id, None))
    }

    /// Create a context with an authenticated user attached.
    pub fn create_authenticated_context(
        &self,
        tenant_id: &str,
        request_id: String,
        user: String,
    ) -> Result<TenantContext> {
        let config = self
            .get_tenant(tenant_id)
            .ok_or_else(|| anyhow!("Unknown tenant '{}'", tenant_id))?;

        Ok(TenantContext::new(
            tenant_id.to_string(),
            config,
            request_id,
            Some(user),
        ))
    }

    /// Extract a tenant ID from HTTP request headers.
    ///
    /// Checks the following headers in order:
    /// 1. `X-Tenant-ID`
    /// 2. `X-Tenant`
    /// 3. `Tenant-ID`
    pub fn extract_tenant_from_headers(headers: &HashMap<String, String>) -> Option<String> {
        const HEADER_NAMES: &[&str] = &["x-tenant-id", "x-tenant", "tenant-id"];
        for header_name in HEADER_NAMES {
            // Check both exact case and lower-cased variants
            if let Some(value) = headers.get(*header_name) {
                return Some(value.trim().to_string());
            }
            // Try capitalised versions
            let capitalised = header_name
                .split('-')
                .map(|part| {
                    let mut chars = part.chars();
                    match chars.next() {
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join("-");
            if let Some(value) = headers.get(&capitalised) {
                return Some(value.trim().to_string());
            }
        }
        None
    }

    /// Generate a new unique request ID suitable for use in a `TenantContext`.
    pub fn generate_request_id() -> String {
        Uuid::new_v4().to_string()
    }
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TenantRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantRegistry")
            .field("tenant_count", &self.tenant_count())
            .field("has_default_config", &self.default_config.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tenant(id: &str) -> TenantConfig {
        TenantConfig::new(id, format!("Tenant {}", id))
            .with_dataset("http://ex.org/data")
            .with_operation(TenantOperation::Mutation)
            .with_rate_limit(120)
    }

    #[test]
    fn test_register_and_retrieve_tenant() {
        let registry = TenantRegistry::new();
        let config = make_tenant("acme");
        registry
            .register_tenant(config)
            .expect("register should succeed");

        let retrieved = registry.get_tenant("acme").expect("should be found");
        assert_eq!(retrieved.tenant_id, "acme");
        assert_eq!(retrieved.display_name, "Tenant acme");
        assert_eq!(retrieved.rate_limit_rpm, 120);
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let registry = TenantRegistry::new();
        registry
            .register_tenant(make_tenant("dup"))
            .expect("first ok");
        let result = registry.register_tenant(make_tenant("dup"));
        assert!(result.is_err());
    }

    #[test]
    fn test_deregister_tenant() {
        let registry = TenantRegistry::new();
        registry.register_tenant(make_tenant("temp")).expect("ok");
        assert!(registry.deregister_tenant("temp"));
        assert!(!registry.deregister_tenant("temp")); // already gone
        assert!(registry.get_tenant("temp").is_none());
    }

    #[test]
    fn test_list_tenants() {
        let registry = TenantRegistry::new();
        registry.register_tenant(make_tenant("t1")).expect("ok");
        registry.register_tenant(make_tenant("t2")).expect("ok");
        let mut tenants = registry.list_tenants();
        tenants.sort();
        assert_eq!(tenants, vec!["t1", "t2"]);
    }

    #[test]
    fn test_create_context_known_tenant() {
        let registry = TenantRegistry::new();
        registry
            .register_tenant(make_tenant("ctx_tenant"))
            .expect("ok");

        let ctx = registry
            .create_context(Some("ctx_tenant"), "req-1".to_string())
            .expect("context creation should succeed");

        assert_eq!(ctx.tenant_id, "ctx_tenant");
        assert_eq!(ctx.request_id, "req-1");
        assert!(ctx.authenticated_user.is_none());
    }

    #[test]
    fn test_create_context_unknown_tenant_fails() {
        let registry = TenantRegistry::new();
        let result = registry.create_context(Some("unknown"), "req".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_create_context_default_tenant() {
        let default_cfg = TenantConfig::new("default", "Default Tenant");
        let registry = TenantRegistry::new().with_default_config(default_cfg);

        let ctx = registry
            .create_context(None, "req-default".to_string())
            .expect("should use default");
        assert_eq!(ctx.tenant_id, "default");
    }

    #[test]
    fn test_tenant_operation_permission() {
        let config = make_tenant("perm_test");
        // make_tenant adds Mutation; Query is already there by default
        assert!(config.allows(&TenantOperation::Query));
        assert!(config.allows(&TenantOperation::Mutation));
        assert!(!config.allows(&TenantOperation::Subscription));
    }

    #[test]
    fn test_dataset_access() {
        let config = TenantConfig::new("ds_test", "DS Test").with_dataset("http://ex.org/allowed");

        assert!(config.can_access_dataset("http://ex.org/allowed"));
        assert!(!config.can_access_dataset("http://ex.org/forbidden"));
    }

    #[test]
    fn test_dataset_access_empty_allows_all() {
        let config = TenantConfig::new("open", "Open Tenant");
        // No datasets configured — all graphs accessible
        assert!(config.can_access_dataset("http://anything.example.org/graph"));
    }

    #[test]
    fn test_extract_tenant_from_headers() {
        let mut headers = HashMap::new();
        headers.insert("x-tenant-id".to_string(), "  acme  ".to_string());

        let tenant = TenantRegistry::extract_tenant_from_headers(&headers);
        assert_eq!(tenant.as_deref(), Some("acme"));
    }

    #[test]
    fn test_extract_tenant_from_headers_capitalised() {
        let mut headers = HashMap::new();
        headers.insert("X-Tenant-Id".to_string(), "widget-corp".to_string());

        let tenant = TenantRegistry::extract_tenant_from_headers(&headers);
        assert_eq!(tenant.as_deref(), Some("widget-corp"));
    }

    #[test]
    fn test_extract_tenant_missing_header_returns_none() {
        let headers = HashMap::new();
        assert!(TenantRegistry::extract_tenant_from_headers(&headers).is_none());
    }

    #[test]
    fn test_update_tenant() {
        let registry = TenantRegistry::new();
        registry.register_tenant(make_tenant("upd")).expect("ok");

        let updated = TenantConfig::new("upd", "Updated Name").with_rate_limit(999);
        registry
            .update_tenant(updated)
            .expect("update should succeed");

        let retrieved = registry.get_tenant("upd").expect("should exist");
        assert_eq!(retrieved.display_name, "Updated Name");
        assert_eq!(retrieved.rate_limit_rpm, 999);
    }

    #[test]
    fn test_update_nonexistent_tenant_fails() {
        let registry = TenantRegistry::new();
        let result = registry.update_tenant(make_tenant("ghost"));
        assert!(result.is_err());
    }

    #[test]
    fn test_authenticated_context() {
        let registry = TenantRegistry::new();
        registry
            .register_tenant(make_tenant("auth_tenant"))
            .expect("ok");

        let ctx = registry
            .create_authenticated_context(
                "auth_tenant",
                "req-auth".to_string(),
                "alice@example.com".to_string(),
            )
            .expect("ok");

        assert_eq!(ctx.authenticated_user.as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn test_tenant_context_can_perform() {
        let registry = TenantRegistry::new();
        registry.register_tenant(make_tenant("perm")).expect("ok");

        let ctx = registry
            .create_context(Some("perm"), TenantRegistry::generate_request_id())
            .expect("ok");

        assert!(ctx.can_perform(&TenantOperation::Query));
        assert!(!ctx.can_perform(&TenantOperation::Subscription));
    }
}
