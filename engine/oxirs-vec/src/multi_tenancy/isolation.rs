//! Data isolation strategies for multi-tenancy

use crate::multi_tenancy::types::{MultiTenancyError, MultiTenancyResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Level of tenant isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// Shared index with namespace prefixes (lowest isolation, highest efficiency)
    Namespace,

    /// Separate indices per tenant (medium isolation)
    SeparateIndex,

    /// Separate databases/instances per tenant (highest isolation, highest cost)
    Dedicated,
}

impl IsolationLevel {
    /// Get isolation strength (0-10, higher = stronger)
    pub fn strength(&self) -> u8 {
        match self {
            Self::Namespace => 3,
            Self::SeparateIndex => 7,
            Self::Dedicated => 10,
        }
    }

    /// Get performance efficiency (0-10, higher = more efficient)
    pub fn efficiency(&self) -> u8 {
        match self {
            Self::Namespace => 10,
            Self::SeparateIndex => 6,
            Self::Dedicated => 3,
        }
    }

    /// Recommended for tenant tier
    pub fn for_tier(tier: &str) -> Self {
        match tier {
            "free" | "trial" => Self::Namespace,
            "pro" | "business" => Self::SeparateIndex,
            "enterprise" | "dedicated" => Self::Dedicated,
            _ => Self::Namespace,
        }
    }
}

/// Isolation strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationStrategy {
    /// Base isolation level
    pub level: IsolationLevel,

    /// Whether to encrypt tenant data at rest
    pub encryption_at_rest: bool,

    /// Whether to encrypt tenant data in transit
    pub encryption_in_transit: bool,

    /// Custom namespace separator
    pub namespace_separator: String,

    /// Maximum namespace depth
    pub max_namespace_depth: usize,
}

impl IsolationStrategy {
    /// Create new isolation strategy
    pub fn new(level: IsolationLevel) -> Self {
        Self {
            level,
            encryption_at_rest: false,
            encryption_in_transit: true,
            namespace_separator: ":".to_string(),
            max_namespace_depth: 5,
        }
    }

    /// Create strategy for free tier
    pub fn free_tier() -> Self {
        Self::new(IsolationLevel::Namespace)
    }

    /// Create strategy for pro tier
    pub fn pro_tier() -> Self {
        let mut strategy = Self::new(IsolationLevel::SeparateIndex);
        strategy.encryption_at_rest = true;
        strategy
    }

    /// Create strategy for enterprise tier
    pub fn enterprise_tier() -> Self {
        let mut strategy = Self::new(IsolationLevel::Dedicated);
        strategy.encryption_at_rest = true;
        strategy
    }

    /// Enable encryption
    pub fn with_encryption(mut self, at_rest: bool, in_transit: bool) -> Self {
        self.encryption_at_rest = at_rest;
        self.encryption_in_transit = in_transit;
        self
    }

    /// Set custom namespace separator
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.namespace_separator = separator.into();
        self
    }
}

/// Namespace manager for tenant data isolation
pub struct NamespaceManager {
    /// Tenant namespaces
    namespaces: Arc<RwLock<HashMap<String, Namespace>>>,

    /// Isolation strategy
    strategy: IsolationStrategy,
}

/// Namespace for tenant data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Namespace {
    /// Tenant identifier
    pub tenant_id: String,

    /// Namespace prefix
    pub prefix: String,

    /// Sub-namespaces
    pub sub_namespaces: Vec<String>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Namespace {
    /// Create new namespace
    pub fn new(tenant_id: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            prefix: prefix.into(),
            sub_namespaces: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create sub-namespace
    pub fn create_sub_namespace(&mut self, name: impl Into<String>, separator: &str) -> String {
        let sub = format!("{}{}{}", self.prefix, separator, name.into());
        self.sub_namespaces.push(sub.clone());
        sub
    }

    /// Get fully qualified key
    pub fn qualify_key(&self, key: &str, separator: &str) -> String {
        format!("{}{}{}", self.prefix, separator, key)
    }

    /// Check if key belongs to this namespace
    pub fn owns_key(&self, key: &str) -> bool {
        key.starts_with(&self.prefix)
    }
}

impl NamespaceManager {
    /// Create new namespace manager
    pub fn new(strategy: IsolationStrategy) -> Self {
        Self {
            namespaces: Arc::new(RwLock::new(HashMap::new())),
            strategy,
        }
    }

    /// Register namespace for tenant
    pub fn register_tenant(&self, tenant_id: impl Into<String>) -> MultiTenancyResult<String> {
        let tenant_id = tenant_id.into();
        let prefix = self.generate_namespace_prefix(&tenant_id);

        let namespace = Namespace::new(tenant_id.clone(), prefix.clone());

        let mut namespaces =
            self.namespaces
                .write()
                .map_err(|e| MultiTenancyError::InternalError {
                    message: format!("Lock error: {}", e),
                })?;

        if namespaces.contains_key(&tenant_id) {
            return Err(MultiTenancyError::TenantAlreadyExists {
                tenant_id: tenant_id.clone(),
            });
        }

        namespaces.insert(tenant_id, namespace);

        Ok(prefix)
    }

    /// Unregister tenant namespace
    pub fn unregister_tenant(&self, tenant_id: &str) -> MultiTenancyResult<()> {
        let mut namespaces =
            self.namespaces
                .write()
                .map_err(|e| MultiTenancyError::InternalError {
                    message: format!("Lock error: {}", e),
                })?;

        namespaces
            .remove(tenant_id)
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })?;

        Ok(())
    }

    /// Get namespace prefix for tenant
    pub fn get_prefix(&self, tenant_id: &str) -> MultiTenancyResult<String> {
        let namespaces = self
            .namespaces
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        namespaces
            .get(tenant_id)
            .map(|ns| ns.prefix.clone())
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })
    }

    /// Qualify key with tenant namespace
    pub fn qualify_key(&self, tenant_id: &str, key: &str) -> MultiTenancyResult<String> {
        let namespaces = self
            .namespaces
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        let namespace =
            namespaces
                .get(tenant_id)
                .ok_or_else(|| MultiTenancyError::TenantNotFound {
                    tenant_id: tenant_id.to_string(),
                })?;

        Ok(namespace.qualify_key(key, &self.strategy.namespace_separator))
    }

    /// Extract tenant ID from namespaced key
    pub fn extract_tenant_id(&self, namespaced_key: &str) -> MultiTenancyResult<String> {
        let namespaces = self
            .namespaces
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        for (tenant_id, namespace) in namespaces.iter() {
            if namespace.owns_key(namespaced_key) {
                return Ok(tenant_id.clone());
            }
        }

        Err(MultiTenancyError::IsolationViolation {
            message: format!("No tenant owns key: {}", namespaced_key),
        })
    }

    /// Validate that a key belongs to a tenant
    pub fn validate_access(&self, tenant_id: &str, key: &str) -> MultiTenancyResult<bool> {
        let namespaces = self
            .namespaces
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        let namespace =
            namespaces
                .get(tenant_id)
                .ok_or_else(|| MultiTenancyError::TenantNotFound {
                    tenant_id: tenant_id.to_string(),
                })?;

        Ok(namespace.owns_key(key))
    }

    /// Create sub-namespace for tenant
    pub fn create_sub_namespace(
        &self,
        tenant_id: &str,
        name: impl Into<String>,
    ) -> MultiTenancyResult<String> {
        let mut namespaces =
            self.namespaces
                .write()
                .map_err(|e| MultiTenancyError::InternalError {
                    message: format!("Lock error: {}", e),
                })?;

        let namespace =
            namespaces
                .get_mut(tenant_id)
                .ok_or_else(|| MultiTenancyError::TenantNotFound {
                    tenant_id: tenant_id.to_string(),
                })?;

        if namespace.sub_namespaces.len() >= self.strategy.max_namespace_depth {
            return Err(MultiTenancyError::InvalidConfiguration {
                message: "Maximum namespace depth exceeded".to_string(),
            });
        }

        Ok(namespace.create_sub_namespace(name, &self.strategy.namespace_separator))
    }

    /// Get all namespaces
    pub fn list_namespaces(&self) -> MultiTenancyResult<Vec<String>> {
        let namespaces = self
            .namespaces
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(namespaces.keys().cloned().collect())
    }

    /// Generate namespace prefix from tenant ID
    fn generate_namespace_prefix(&self, tenant_id: &str) -> String {
        // Sanitize tenant ID for use as namespace
        let sanitized: String = tenant_id
            .chars()
            .map(|c| match c {
                '-' | '.' | '/' => '_',
                c => c,
            })
            .collect();

        format!("tenant_{}", sanitized)
    }

    /// Get isolation strategy
    pub fn strategy(&self) -> &IsolationStrategy {
        &self.strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_levels() {
        assert_eq!(IsolationLevel::Namespace.strength(), 3);
        assert_eq!(IsolationLevel::SeparateIndex.strength(), 7);
        assert_eq!(IsolationLevel::Dedicated.strength(), 10);

        assert_eq!(IsolationLevel::Namespace.efficiency(), 10);
        assert_eq!(IsolationLevel::Dedicated.efficiency(), 3);
    }

    #[test]
    fn test_isolation_level_for_tier() {
        assert_eq!(IsolationLevel::for_tier("free"), IsolationLevel::Namespace);
        assert_eq!(
            IsolationLevel::for_tier("pro"),
            IsolationLevel::SeparateIndex
        );
        assert_eq!(
            IsolationLevel::for_tier("enterprise"),
            IsolationLevel::Dedicated
        );
    }

    #[test]
    fn test_isolation_strategy() {
        let strategy = IsolationStrategy::free_tier();
        assert_eq!(strategy.level, IsolationLevel::Namespace);
        assert!(!strategy.encryption_at_rest);

        let strategy = IsolationStrategy::pro_tier();
        assert_eq!(strategy.level, IsolationLevel::SeparateIndex);
        assert!(strategy.encryption_at_rest);

        let strategy = IsolationStrategy::enterprise_tier();
        assert_eq!(strategy.level, IsolationLevel::Dedicated);
        assert!(strategy.encryption_at_rest);
    }

    #[test]
    fn test_namespace_creation() {
        let ns = Namespace::new("tenant1", "tenant_tenant1");
        assert_eq!(ns.tenant_id, "tenant1");
        assert_eq!(ns.prefix, "tenant_tenant1");
        assert!(ns.sub_namespaces.is_empty());
    }

    #[test]
    fn test_namespace_qualification() {
        let ns = Namespace::new("tenant1", "tenant_tenant1");
        let qualified = ns.qualify_key("vector123", ":");
        assert_eq!(qualified, "tenant_tenant1:vector123");
        assert!(ns.owns_key(&qualified));
        assert!(!ns.owns_key("other_tenant:vector123"));
    }

    #[test]
    fn test_namespace_manager() {
        let strategy = IsolationStrategy::new(IsolationLevel::Namespace);
        let manager = NamespaceManager::new(strategy);

        // Register tenant
        let prefix = manager.register_tenant("tenant1").unwrap();
        assert!(prefix.contains("tenant_tenant1"));

        // Get prefix
        let retrieved_prefix = manager.get_prefix("tenant1").unwrap();
        assert_eq!(prefix, retrieved_prefix);

        // Qualify key
        let qualified = manager.qualify_key("tenant1", "vector123").unwrap();
        assert!(qualified.starts_with(&prefix));
        assert!(qualified.contains("vector123"));

        // Validate access
        assert!(manager.validate_access("tenant1", &qualified).unwrap());
        assert!(!manager
            .validate_access("tenant1", "other_tenant:key")
            .unwrap());

        // Extract tenant ID
        let extracted = manager.extract_tenant_id(&qualified).unwrap();
        assert_eq!(extracted, "tenant1");

        // Unregister tenant
        manager.unregister_tenant("tenant1").unwrap();
        assert!(manager.get_prefix("tenant1").is_err());
    }

    #[test]
    fn test_sub_namespaces() {
        let strategy = IsolationStrategy::new(IsolationLevel::Namespace);
        let manager = NamespaceManager::new(strategy);

        manager.register_tenant("tenant1").unwrap();

        // Create sub-namespace
        let sub = manager.create_sub_namespace("tenant1", "vectors").unwrap();
        assert!(sub.contains("tenant_tenant1"));
        assert!(sub.contains("vectors"));

        // Create another sub-namespace
        let sub2 = manager
            .create_sub_namespace("tenant1", "embeddings")
            .unwrap();
        assert!(sub2.contains("embeddings"));
        assert_ne!(sub, sub2);
    }

    #[test]
    fn test_namespace_manager_errors() {
        let strategy = IsolationStrategy::new(IsolationLevel::Namespace);
        let manager = NamespaceManager::new(strategy);

        // Getting prefix for non-existent tenant should fail
        assert!(manager.get_prefix("nonexistent").is_err());

        // Qualifying key for non-existent tenant should fail
        assert!(manager.qualify_key("nonexistent", "key").is_err());

        // Registering same tenant twice should fail
        manager.register_tenant("tenant1").unwrap();
        assert!(manager.register_tenant("tenant1").is_err());

        // Unregistering non-existent tenant should fail
        assert!(manager.unregister_tenant("nonexistent").is_err());
    }

    #[test]
    fn test_list_namespaces() {
        let strategy = IsolationStrategy::new(IsolationLevel::Namespace);
        let manager = NamespaceManager::new(strategy);

        assert_eq!(manager.list_namespaces().unwrap().len(), 0);

        manager.register_tenant("tenant1").unwrap();
        manager.register_tenant("tenant2").unwrap();

        let namespaces = manager.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);
        assert!(namespaces.contains(&"tenant1".to_string()));
        assert!(namespaces.contains(&"tenant2".to_string()));
    }
}
