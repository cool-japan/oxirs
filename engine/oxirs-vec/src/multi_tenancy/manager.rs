//! Multi-tenant manager - main interface for multi-tenancy

use crate::multi_tenancy::{
    access_control::AccessControl,
    billing::{BillingEngine, BillingMetrics, BillingPeriod, PricingModel},
    isolation::{IsolationLevel, IsolationStrategy, NamespaceManager},
    quota::{QuotaEnforcer, QuotaLimits, RateLimiter},
    tenant::{Tenant, TenantId, TenantMetadata, TenantStatus},
    types::{
        MultiTenancyError, MultiTenancyResult, TenantContext, TenantOperation, TenantStatistics,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Configuration for tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant metadata
    pub metadata: TenantMetadata,

    /// Isolation strategy
    pub isolation: IsolationStrategy,

    /// Quota limits
    pub quotas: QuotaLimits,

    /// Pricing model
    pub pricing: PricingModel,

    /// Rate limit (requests per second)
    pub rate_limit: Option<f64>,
}

impl TenantConfig {
    /// Create config for free tier
    pub fn free_tier(tenant_id: impl Into<String>, name: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        Self {
            metadata: TenantMetadata::new(name, "free"),
            isolation: IsolationStrategy::free_tier(),
            quotas: QuotaLimits::free_tier(&tenant_id),
            pricing: PricingModel::PerRequest {
                cost_per_request: 0.001,
            },
            rate_limit: Some(10.0),
        }
    }

    /// Create config for pro tier
    pub fn pro_tier(tenant_id: impl Into<String>, name: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        Self {
            metadata: TenantMetadata::new(name, "pro"),
            isolation: IsolationStrategy::pro_tier(),
            quotas: QuotaLimits::pro_tier(&tenant_id),
            pricing: PricingModel::PerComputeUnit {
                cost_per_unit: 0.01,
            },
            rate_limit: Some(100.0),
        }
    }

    /// Create config for enterprise tier
    pub fn enterprise_tier(tenant_id: impl Into<String>, name: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        Self {
            metadata: TenantMetadata::new(name, "enterprise"),
            isolation: IsolationStrategy::enterprise_tier(),
            quotas: QuotaLimits::enterprise_tier(&tenant_id),
            pricing: PricingModel::Subscription {
                monthly_fee: 1000.0,
                included_requests: 1_000_000,
                overage_cost: 0.005,
            },
            rate_limit: None, // Unlimited
        }
    }
}

/// Configuration for multi-tenant manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantManagerConfig {
    /// Default isolation level
    pub default_isolation: IsolationLevel,

    /// Billing period
    pub billing_period: BillingPeriod,

    /// Enable strict quota enforcement
    pub strict_quotas: bool,

    /// Enable access control
    pub enable_access_control: bool,

    /// Enable billing/metering
    pub enable_billing: bool,

    /// Auto-suspend tenants on quota exceeded
    pub auto_suspend_on_quota_exceeded: bool,
}

impl TenantManagerConfig {
    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            default_isolation: IsolationLevel::Namespace,
            billing_period: BillingPeriod::Monthly,
            strict_quotas: true,
            enable_access_control: true,
            enable_billing: true,
            auto_suspend_on_quota_exceeded: false,
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self {
            default_isolation: IsolationLevel::SeparateIndex,
            billing_period: BillingPeriod::Monthly,
            strict_quotas: true,
            enable_access_control: true,
            enable_billing: true,
            auto_suspend_on_quota_exceeded: true,
        }
    }
}

/// Multi-tenant manager
pub struct MultiTenantManager {
    /// Configuration
    config: TenantManagerConfig,

    /// Tenant registry
    tenants: Arc<RwLock<HashMap<TenantId, Tenant>>>,

    /// Tenant statistics
    statistics: Arc<RwLock<HashMap<TenantId, TenantStatistics>>>,

    /// Namespace manager
    namespace_manager: Arc<NamespaceManager>,

    /// Quota enforcer
    quota_enforcer: Arc<QuotaEnforcer>,

    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,

    /// Access control
    access_control: Arc<AccessControl>,

    /// Billing engine
    billing_engine: Arc<BillingEngine>,
}

impl MultiTenantManager {
    /// Create new multi-tenant manager
    pub fn new(config: TenantManagerConfig) -> Self {
        let isolation_strategy = IsolationStrategy::new(config.default_isolation);
        let namespace_manager = Arc::new(NamespaceManager::new(isolation_strategy));
        let quota_enforcer = Arc::new(QuotaEnforcer::new());
        let rate_limiter = Arc::new(RateLimiter::new());
        let access_control = Arc::new(AccessControl::new());
        let billing_engine = Arc::new(BillingEngine::new(config.billing_period));

        Self {
            config,
            tenants: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            namespace_manager,
            quota_enforcer,
            rate_limiter,
            access_control,
            billing_engine,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TenantManagerConfig::default_config())
    }

    /// Create tenant with configuration
    pub fn create_tenant(
        &self,
        tenant_id: impl Into<String>,
        config: TenantConfig,
    ) -> MultiTenancyResult<()> {
        let tenant_id = tenant_id.into();
        let tenant = Tenant::new(tenant_id.clone(), config.metadata);

        // Register namespace
        self.namespace_manager.register_tenant(&tenant_id)?;

        // Set quota limits
        self.quota_enforcer.set_limits(config.quotas)?;

        // Set rate limit
        if let Some(rate) = config.rate_limit {
            self.rate_limiter.set_rate(&tenant_id, rate)?;
        }

        // Set pricing model
        self.billing_engine
            .set_pricing(&tenant_id, config.pricing)?;

        // Create default access policy
        if self.config.enable_access_control {
            self.access_control.create_default_policy(&tenant_id)?;
        }

        // Store tenant
        self.tenants
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(tenant_id.clone(), tenant);

        // Initialize statistics
        self.statistics
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(tenant_id.clone(), TenantStatistics::new(tenant_id));

        Ok(())
    }

    /// Get tenant by ID
    pub fn get_tenant(&self, tenant_id: &str) -> MultiTenancyResult<Tenant> {
        self.tenants
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })
    }

    /// Update tenant status
    pub fn update_tenant_status(
        &self,
        tenant_id: &str,
        status: TenantStatus,
    ) -> MultiTenancyResult<()> {
        let mut tenants = self
            .tenants
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        let tenant =
            tenants
                .get_mut(tenant_id)
                .ok_or_else(|| MultiTenancyError::TenantNotFound {
                    tenant_id: tenant_id.to_string(),
                })?;

        tenant.set_status(status);
        Ok(())
    }

    /// Delete tenant
    pub fn delete_tenant(&self, tenant_id: &str) -> MultiTenancyResult<()> {
        // Remove from registry
        self.tenants
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .remove(tenant_id);

        // Remove namespace
        self.namespace_manager.unregister_tenant(tenant_id)?;

        // Remove statistics
        self.statistics
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .remove(tenant_id);

        Ok(())
    }

    /// Check if tenant can execute operation
    pub fn check_operation(
        &self,
        context: &TenantContext,
        _operation: TenantOperation,
        resource_delta: Option<(crate::multi_tenancy::quota::ResourceType, u64)>,
    ) -> MultiTenancyResult<()> {
        let tenant_id = &context.tenant_id;

        // Check tenant status
        let tenant = self.get_tenant(tenant_id)?;
        if !tenant.is_operational() {
            if tenant.status == TenantStatus::Suspended {
                return Err(MultiTenancyError::TenantSuspended {
                    tenant_id: tenant_id.clone(),
                });
            }
            return Err(MultiTenancyError::InternalError {
                message: format!("Tenant {} is not operational", tenant_id),
            });
        }

        // Check rate limit
        if !self.rate_limiter.allow_request(tenant_id)? {
            return Err(MultiTenancyError::RateLimitExceeded {
                tenant_id: tenant_id.clone(),
            });
        }

        // Check resource quota
        if let Some((resource_type, amount)) = resource_delta {
            if self.config.strict_quotas
                && !self
                    .quota_enforcer
                    .check_quota(tenant_id, resource_type, amount)?
            {
                if self.config.auto_suspend_on_quota_exceeded {
                    self.update_tenant_status(tenant_id, TenantStatus::Suspended)?;
                }
                return Err(MultiTenancyError::QuotaExceeded {
                    tenant_id: tenant_id.clone(),
                    resource: resource_type.name(),
                });
            }
        }

        Ok(())
    }

    /// Execute operation with full checks
    pub fn execute_operation<F, R>(
        &self,
        context: &TenantContext,
        operation: TenantOperation,
        func: F,
    ) -> MultiTenancyResult<R>
    where
        F: FnOnce() -> MultiTenancyResult<R>,
    {
        // Pre-execution checks
        self.check_operation(context, operation, None)?;

        // Execute operation
        let start = chrono::Utc::now();
        let result = func()?;
        let latency_ms = (chrono::Utc::now() - start).num_milliseconds() as f64;

        // Post-execution: record statistics and billing
        self.record_operation_completed(context, operation, latency_ms)?;

        Ok(result)
    }

    /// Record operation completion
    fn record_operation_completed(
        &self,
        context: &TenantContext,
        operation: TenantOperation,
        latency_ms: f64,
    ) -> MultiTenancyResult<()> {
        let tenant_id = &context.tenant_id;

        // Update statistics
        let mut stats = self
            .statistics
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        stats
            .entry(tenant_id.clone())
            .or_insert_with(|| TenantStatistics::new(tenant_id))
            .record_operation(operation);

        if operation == TenantOperation::VectorSearch {
            stats.get_mut(tenant_id).unwrap().record_query(latency_ms);
        }

        // Record billing
        if self.config.enable_billing {
            self.billing_engine.record_usage(tenant_id, operation, 1)?;
        }

        Ok(())
    }

    /// Get tenant statistics
    pub fn get_statistics(&self, tenant_id: &str) -> MultiTenancyResult<TenantStatistics> {
        self.statistics
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })
    }

    /// Get billing metrics
    pub fn get_billing_metrics(&self, tenant_id: &str) -> MultiTenancyResult<BillingMetrics> {
        self.billing_engine.get_metrics(tenant_id)
    }

    /// List all tenants
    pub fn list_tenants(&self) -> MultiTenancyResult<Vec<Tenant>> {
        Ok(self
            .tenants
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .values()
            .cloned()
            .collect())
    }

    /// Get namespace manager
    pub fn namespace_manager(&self) -> &NamespaceManager {
        &self.namespace_manager
    }

    /// Get quota enforcer
    pub fn quota_enforcer(&self) -> &QuotaEnforcer {
        &self.quota_enforcer
    }

    /// Get access control
    pub fn access_control(&self) -> &AccessControl {
        &self.access_control
    }

    /// Get billing engine
    pub fn billing_engine(&self) -> &BillingEngine {
        &self.billing_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_config_tiers() {
        let free = TenantConfig::free_tier("t1", "Free Tenant");
        assert_eq!(free.metadata.tier, "free");
        assert!(free.rate_limit.is_some());

        let pro = TenantConfig::pro_tier("t2", "Pro Tenant");
        assert_eq!(pro.metadata.tier, "pro");

        let enterprise = TenantConfig::enterprise_tier("t3", "Enterprise");
        assert_eq!(enterprise.metadata.tier, "enterprise");
        assert!(enterprise.rate_limit.is_none()); // Unlimited
    }

    #[test]
    fn test_manager_creation() {
        let manager = MultiTenantManager::with_defaults();
        assert_eq!(manager.list_tenants().unwrap().len(), 0);
    }

    #[test]
    fn test_create_and_get_tenant() {
        let manager = MultiTenantManager::with_defaults();
        let config = TenantConfig::free_tier("tenant1", "Test Tenant");

        manager.create_tenant("tenant1", config).unwrap();

        let tenant = manager.get_tenant("tenant1").unwrap();
        assert_eq!(tenant.id, "tenant1");
        assert_eq!(tenant.metadata.name, "Test Tenant");
        assert_eq!(tenant.status, TenantStatus::Active);
    }

    #[test]
    fn test_tenant_operations() {
        let manager = MultiTenantManager::with_defaults();
        let config = TenantConfig::free_tier("tenant1", "Test");
        manager.create_tenant("tenant1", config).unwrap();

        let context = TenantContext::new("tenant1");

        // Should allow operation
        manager
            .check_operation(&context, TenantOperation::VectorSearch, None)
            .unwrap();

        // Execute operation with closure
        let result: MultiTenancyResult<i32> =
            manager.execute_operation(&context, TenantOperation::VectorSearch, || Ok(42));
        assert_eq!(result.unwrap(), 42);

        // Check statistics updated
        let stats = manager.get_statistics("tenant1").unwrap();
        assert_eq!(stats.total_queries, 1);
    }

    #[test]
    fn test_tenant_status_changes() {
        let manager = MultiTenantManager::with_defaults();
        let config = TenantConfig::free_tier("tenant1", "Test");
        manager.create_tenant("tenant1", config).unwrap();

        // Suspend tenant
        manager
            .update_tenant_status("tenant1", TenantStatus::Suspended)
            .unwrap();

        let tenant = manager.get_tenant("tenant1").unwrap();
        assert_eq!(tenant.status, TenantStatus::Suspended);

        // Operations should fail when suspended
        let context = TenantContext::new("tenant1");
        assert!(manager
            .check_operation(&context, TenantOperation::VectorSearch, None)
            .is_err());
    }

    #[test]
    fn test_delete_tenant() {
        let manager = MultiTenantManager::with_defaults();
        let config = TenantConfig::free_tier("tenant1", "Test");
        manager.create_tenant("tenant1", config).unwrap();

        assert!(manager.get_tenant("tenant1").is_ok());

        manager.delete_tenant("tenant1").unwrap();

        assert!(manager.get_tenant("tenant1").is_err());
    }

    #[test]
    fn test_list_tenants() {
        let manager = MultiTenantManager::with_defaults();

        manager
            .create_tenant("tenant1", TenantConfig::free_tier("tenant1", "T1"))
            .unwrap();
        manager
            .create_tenant("tenant2", TenantConfig::pro_tier("tenant2", "T2"))
            .unwrap();
        manager
            .create_tenant("tenant3", TenantConfig::enterprise_tier("tenant3", "T3"))
            .unwrap();

        let tenants = manager.list_tenants().unwrap();
        assert_eq!(tenants.len(), 3);
    }

    #[test]
    fn test_billing_integration() {
        let manager = MultiTenantManager::with_defaults();
        let config = TenantConfig::free_tier("tenant1", "Test");
        manager.create_tenant("tenant1", config).unwrap();

        let context = TenantContext::new("tenant1");

        // Execute some operations
        for _ in 0..10 {
            let _ = manager.execute_operation(&context, TenantOperation::VectorSearch, || Ok(()));
        }

        // Check billing metrics
        let metrics = manager.get_billing_metrics("tenant1").unwrap();
        assert_eq!(metrics.total_requests, 10);
        assert!(metrics.total_cost > 0.0);
    }

    #[test]
    fn test_manager_config() {
        let config = TenantManagerConfig::default_config();
        assert!(config.strict_quotas);
        assert!(config.enable_access_control);
        assert!(config.enable_billing);

        let prod_config = TenantManagerConfig::production();
        assert_eq!(prod_config.default_isolation, IsolationLevel::SeparateIndex);
        assert!(prod_config.auto_suspend_on_quota_exceeded);
    }
}
