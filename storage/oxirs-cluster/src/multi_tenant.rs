//! Multi-tenant isolation and resource management for cluster storage
//!
//! This module provides comprehensive tenant isolation, resource quotas,
//! and per-tenant metrics for secure multi-tenant deployments.

use crate::error::{ClusterError, Result};
use scirs2_core::metrics::{Counter, Gauge, Histogram};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tenant identifier with validation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(String);

impl TenantId {
    /// Create a new tenant ID with validation
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();

        // Validate format: alphanumeric + hyphens only
        if !id
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(ClusterError::InvalidTenant(format!(
                "Tenant ID must contain only alphanumeric characters, hyphens, and underscores: {}",
                id
            )));
        }

        // Validate length (1-64 characters)
        if id.is_empty() || id.len() > 64 {
            return Err(ClusterError::InvalidTenant(format!(
                "Tenant ID must be between 1 and 64 characters: {} (length: {})",
                id,
                id.len()
            )));
        }

        // Check for reserved tenant IDs
        if id == "system" || id == "admin" || id == "root" {
            return Err(ClusterError::InvalidTenant(format!(
                "Reserved tenant ID: {}",
                id
            )));
        }

        Ok(Self(id))
    }

    /// Get the tenant ID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Resource limits for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum storage usage in MB
    pub max_storage_mb: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Maximum query rate (queries per second)
    pub max_query_rate: f64,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,    // 1GB
            max_storage_mb: 10_240, // 10GB
            max_connections: 100,
            max_query_rate: 100.0,
            max_cpu_percent: 50.0,
        }
    }
}

/// Current resource usage for a tenant
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Current memory usage in MB
    pub memory_mb: f64,
    /// Current storage usage in MB
    pub storage_mb: f64,
    /// Current connection count
    pub connections: u32,
    /// Current query rate (queries per second)
    pub query_rate: f64,
    /// Current CPU usage percentage
    pub cpu_percent: f64,
}

/// Per-tenant metrics collector
#[derive(Clone)]
pub struct TenantMetrics {
    /// Total requests
    requests_total: Arc<Counter>,
    /// Failed requests
    requests_failed: Arc<Counter>,
    /// Active connections
    active_connections: Arc<Gauge>,
    /// Memory usage in bytes
    memory_usage_bytes: Arc<Gauge>,
    /// Storage usage in bytes
    storage_usage_bytes: Arc<Gauge>,
    /// CPU usage percentage
    cpu_usage_percent: Arc<Gauge>,
    /// Request latency histogram
    request_latency_ms: Arc<Histogram>,
    /// Query rate (requests/sec)
    query_rate: Arc<Gauge>,
}

impl TenantMetrics {
    /// Create new tenant metrics
    pub fn new(tenant_id: &TenantId) -> Self {
        let prefix = format!("tenant_{}", tenant_id.as_str());

        Self {
            requests_total: Arc::new(Counter::new(format!("{}_requests_total", prefix))),
            requests_failed: Arc::new(Counter::new(format!("{}_requests_failed", prefix))),
            active_connections: Arc::new(Gauge::new(format!("{}_active_connections", prefix))),
            memory_usage_bytes: Arc::new(Gauge::new(format!("{}_memory_usage_bytes", prefix))),
            storage_usage_bytes: Arc::new(Gauge::new(format!("{}_storage_usage_bytes", prefix))),
            cpu_usage_percent: Arc::new(Gauge::new(format!("{}_cpu_usage_percent", prefix))),
            request_latency_ms: Arc::new(Histogram::new(format!("{}_request_latency_ms", prefix))),
            query_rate: Arc::new(Gauge::new(format!("{}_query_rate", prefix))),
        }
    }

    /// Record a successful request
    pub fn record_request(&self, latency_ms: f64) {
        self.requests_total.inc();
        self.request_latency_ms.observe(latency_ms);
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.requests_total.inc();
        self.requests_failed.inc();
    }

    /// Update resource usage
    pub fn update_usage(&self, usage: &ResourceUsage) {
        self.memory_usage_bytes
            .set(usage.memory_mb * 1024.0 * 1024.0);
        self.storage_usage_bytes
            .set(usage.storage_mb * 1024.0 * 1024.0);
        self.active_connections.set(usage.connections as f64);
        self.cpu_usage_percent.set(usage.cpu_percent);
        self.query_rate.set(usage.query_rate);
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> TenantMetricsSnapshot {
        let stats = self.request_latency_ms.get_stats();
        TenantMetricsSnapshot {
            total_requests: self.requests_total.get(),
            failed_requests: self.requests_failed.get(),
            active_connections: self.active_connections.get() as u32,
            memory_usage_mb: self.memory_usage_bytes.get() / (1024.0 * 1024.0),
            storage_usage_mb: self.storage_usage_bytes.get() / (1024.0 * 1024.0),
            cpu_usage_percent: self.cpu_usage_percent.get(),
            avg_latency_ms: stats.mean,
            p95_latency_ms: stats.mean * 1.5, // Approximate p95 as 1.5x mean
            query_rate: self.query_rate.get(),
        }
    }
}

/// Snapshot of tenant metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantMetricsSnapshot {
    pub total_requests: u64,
    pub failed_requests: u64,
    pub active_connections: u32,
    pub memory_usage_mb: f64,
    pub storage_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub query_rate: f64,
}

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant identifier
    pub id: TenantId,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Enable strict isolation
    pub strict_isolation: bool,
    /// Enable resource monitoring
    pub enable_monitoring: bool,
}

/// Tenant context for request processing
#[derive(Debug, Clone)]
pub struct TenantContext {
    /// Tenant identifier
    pub tenant_id: TenantId,
    /// Current resource usage
    pub usage: ResourceUsage,
    /// Resource limits
    pub limits: ResourceLimits,
}

/// Multi-tenant isolation manager
pub struct TenantIsolation {
    /// Registered tenants
    tenants: Arc<RwLock<HashMap<TenantId, TenantConfig>>>,
    /// Per-tenant metrics
    metrics: Arc<RwLock<HashMap<TenantId, TenantMetrics>>>,
    /// Current resource usage per tenant
    usage: Arc<RwLock<HashMap<TenantId, ResourceUsage>>>,
    /// Global configuration
    config: IsolationConfig,
}

/// Global isolation configuration
#[derive(Debug, Clone)]
pub struct IsolationConfig {
    /// Enable strict isolation (no resource sharing)
    pub strict_isolation: bool,
    /// Enable quota enforcement
    pub enforce_quotas: bool,
    /// Enable automatic resource scaling
    pub auto_scaling: bool,
    /// Default resource limits for new tenants
    pub default_limits: ResourceLimits,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            strict_isolation: true,
            enforce_quotas: true,
            auto_scaling: false,
            default_limits: ResourceLimits::default(),
        }
    }
}

impl TenantIsolation {
    /// Create new tenant isolation manager
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            tenants: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a new tenant
    pub async fn register_tenant(&self, config: TenantConfig) -> Result<()> {
        let mut tenants = self.tenants.write().await;
        let mut metrics = self.metrics.write().await;
        let mut usage = self.usage.write().await;

        // Check if tenant already exists
        if tenants.contains_key(&config.id) {
            return Err(ClusterError::InvalidTenant(format!(
                "Tenant already registered: {}",
                config.id
            )));
        }

        // Create metrics for the tenant
        let tenant_metrics = TenantMetrics::new(&config.id);

        // Insert tenant configuration
        tenants.insert(config.id.clone(), config.clone());
        metrics.insert(config.id.clone(), tenant_metrics);
        usage.insert(config.id.clone(), ResourceUsage::default());

        Ok(())
    }

    /// Unregister a tenant
    pub async fn unregister_tenant(&self, tenant_id: &TenantId) -> Result<()> {
        let mut tenants = self.tenants.write().await;
        let mut metrics = self.metrics.write().await;
        let mut usage = self.usage.write().await;

        // Remove tenant
        tenants.remove(tenant_id).ok_or_else(|| {
            ClusterError::InvalidTenant(format!("Tenant not found: {}", tenant_id))
        })?;
        metrics.remove(tenant_id);
        usage.remove(tenant_id);

        Ok(())
    }

    /// Check if a tenant exists
    pub async fn tenant_exists(&self, tenant_id: &TenantId) -> bool {
        self.tenants.read().await.contains_key(tenant_id)
    }

    /// Get tenant configuration
    pub async fn get_tenant_config(&self, tenant_id: &TenantId) -> Result<TenantConfig> {
        self.tenants
            .read()
            .await
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| ClusterError::InvalidTenant(format!("Tenant not found: {}", tenant_id)))
    }

    /// Validate resource request against limits
    pub async fn validate_resource_request(
        &self,
        tenant_id: &TenantId,
        requested: &ResourceUsage,
    ) -> Result<()> {
        if !self.config.enforce_quotas {
            return Ok(());
        }

        let tenants = self.tenants.read().await;
        let config = tenants.get(tenant_id).ok_or_else(|| {
            ClusterError::InvalidTenant(format!("Tenant not found: {}", tenant_id))
        })?;

        // Check memory limit
        if requested.memory_mb > config.limits.max_memory_mb as f64 {
            return Err(ClusterError::ResourceLimit(format!(
                "Memory limit exceeded for tenant {}: {} MB > {} MB",
                tenant_id, requested.memory_mb, config.limits.max_memory_mb
            )));
        }

        // Check storage limit
        if requested.storage_mb > config.limits.max_storage_mb as f64 {
            return Err(ClusterError::ResourceLimit(format!(
                "Storage limit exceeded for tenant {}: {} MB > {} MB",
                tenant_id, requested.storage_mb, config.limits.max_storage_mb
            )));
        }

        // Check connection limit
        if requested.connections > config.limits.max_connections {
            return Err(ClusterError::ResourceLimit(format!(
                "Connection limit exceeded for tenant {}: {} > {}",
                tenant_id, requested.connections, config.limits.max_connections
            )));
        }

        // Check query rate limit
        if requested.query_rate > config.limits.max_query_rate {
            return Err(ClusterError::ResourceLimit(format!(
                "Query rate limit exceeded for tenant {}: {:.2} > {:.2} queries/sec",
                tenant_id, requested.query_rate, config.limits.max_query_rate
            )));
        }

        // Check CPU limit
        if requested.cpu_percent > config.limits.max_cpu_percent {
            return Err(ClusterError::ResourceLimit(format!(
                "CPU limit exceeded for tenant {}: {:.1}% > {:.1}%",
                tenant_id, requested.cpu_percent, config.limits.max_cpu_percent
            )));
        }

        Ok(())
    }

    /// Update tenant resource usage
    pub async fn update_usage(&self, tenant_id: &TenantId, usage: ResourceUsage) -> Result<()> {
        // Validate against limits
        if self.config.enforce_quotas {
            self.validate_resource_request(tenant_id, &usage).await?;
        }

        // Update usage
        let mut usage_map = self.usage.write().await;
        usage_map.insert(tenant_id.clone(), usage.clone());

        // Update metrics
        let metrics = self.metrics.read().await;
        if let Some(tenant_metrics) = metrics.get(tenant_id) {
            tenant_metrics.update_usage(&usage);
        }

        Ok(())
    }

    /// Get current resource usage for a tenant
    pub async fn get_usage(&self, tenant_id: &TenantId) -> Result<ResourceUsage> {
        self.usage
            .read()
            .await
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| ClusterError::InvalidTenant(format!("Tenant not found: {}", tenant_id)))
    }

    /// Record a request for a tenant
    pub async fn record_request(&self, tenant_id: &TenantId, latency_ms: f64) -> Result<()> {
        let metrics = self.metrics.read().await;
        if let Some(tenant_metrics) = metrics.get(tenant_id) {
            tenant_metrics.record_request(latency_ms);
            Ok(())
        } else {
            Err(ClusterError::InvalidTenant(format!(
                "Tenant not found: {}",
                tenant_id
            )))
        }
    }

    /// Record a failure for a tenant
    pub async fn record_failure(&self, tenant_id: &TenantId) -> Result<()> {
        let metrics = self.metrics.read().await;
        if let Some(tenant_metrics) = metrics.get(tenant_id) {
            tenant_metrics.record_failure();
            Ok(())
        } else {
            Err(ClusterError::InvalidTenant(format!(
                "Tenant not found: {}",
                tenant_id
            )))
        }
    }

    /// Get metrics snapshot for a tenant
    pub async fn get_metrics(&self, tenant_id: &TenantId) -> Result<TenantMetricsSnapshot> {
        let metrics = self.metrics.read().await;
        metrics
            .get(tenant_id)
            .map(|m| m.snapshot())
            .ok_or_else(|| ClusterError::InvalidTenant(format!("Tenant not found: {}", tenant_id)))
    }

    /// Get all registered tenant IDs
    pub async fn list_tenants(&self) -> Vec<TenantId> {
        self.tenants.read().await.keys().cloned().collect()
    }

    /// Get total number of registered tenants
    pub async fn tenant_count(&self) -> usize {
        self.tenants.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id_validation_valid() {
        assert!(TenantId::new("tenant1").is_ok());
        assert!(TenantId::new("tenant-123").is_ok());
        assert!(TenantId::new("test_tenant").is_ok());
        assert!(TenantId::new("a").is_ok());
        assert!(TenantId::new("ABC-123_test").is_ok());
    }

    #[test]
    fn test_tenant_id_validation_invalid_chars() {
        assert!(TenantId::new("tenant@123").is_err());
        assert!(TenantId::new("tenant.123").is_err());
        assert!(TenantId::new("tenant 123").is_err());
        assert!(TenantId::new("tenant/123").is_err());
    }

    #[test]
    fn test_tenant_id_validation_length() {
        assert!(TenantId::new("").is_err());
        assert!(TenantId::new("a".repeat(65)).is_err());
        assert!(TenantId::new("a".repeat(64)).is_ok());
    }

    #[test]
    fn test_tenant_id_validation_reserved() {
        assert!(TenantId::new("system").is_err());
        assert!(TenantId::new("admin").is_err());
        assert!(TenantId::new("root").is_err());
    }

    #[tokio::test]
    async fn test_register_tenant() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits::default(),
            strict_isolation: true,
            enable_monitoring: true,
        };

        assert!(isolation.register_tenant(config).await.is_ok());
        assert!(isolation.tenant_exists(&tenant_id).await);
    }

    #[tokio::test]
    async fn test_register_duplicate_tenant() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits::default(),
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config.clone())
            .await
            .expect("First registration");
        let result = isolation.register_tenant(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unregister_tenant() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits::default(),
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config)
            .await
            .expect("Registration");
        assert!(isolation.unregister_tenant(&tenant_id).await.is_ok());
        assert!(!isolation.tenant_exists(&tenant_id).await);
    }

    #[tokio::test]
    async fn test_resource_limit_validation_memory() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits {
                max_memory_mb: 100,
                ..Default::default()
            },
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config)
            .await
            .expect("Registration");

        let usage = ResourceUsage {
            memory_mb: 150.0,
            ..Default::default()
        };

        let result = isolation
            .validate_resource_request(&tenant_id, &usage)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Memory limit exceeded"));
    }

    #[tokio::test]
    async fn test_resource_limit_validation_connections() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits {
                max_connections: 10,
                ..Default::default()
            },
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config)
            .await
            .expect("Registration");

        let usage = ResourceUsage {
            connections: 15,
            ..Default::default()
        };

        let result = isolation
            .validate_resource_request(&tenant_id, &usage)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Connection limit exceeded"));
    }

    #[tokio::test]
    async fn test_update_usage() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits::default(),
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config)
            .await
            .expect("Registration");

        let usage = ResourceUsage {
            memory_mb: 500.0,
            storage_mb: 1000.0,
            connections: 50,
            query_rate: 50.0,
            cpu_percent: 25.0,
        };

        assert!(isolation
            .update_usage(&tenant_id, usage.clone())
            .await
            .is_ok());

        let retrieved_usage = isolation.get_usage(&tenant_id).await.expect("Get usage");
        assert_eq!(retrieved_usage.memory_mb, 500.0);
        assert_eq!(retrieved_usage.connections, 50);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let isolation = TenantIsolation::new(IsolationConfig::default());
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        let config = TenantConfig {
            id: tenant_id.clone(),
            limits: ResourceLimits::default(),
            strict_isolation: true,
            enable_monitoring: true,
        };

        isolation
            .register_tenant(config)
            .await
            .expect("Registration");

        // Record some requests
        isolation
            .record_request(&tenant_id, 100.0)
            .await
            .expect("Record request");
        isolation
            .record_request(&tenant_id, 150.0)
            .await
            .expect("Record request");
        isolation
            .record_failure(&tenant_id)
            .await
            .expect("Record failure");

        let metrics = isolation
            .get_metrics(&tenant_id)
            .await
            .expect("Get metrics");
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn test_list_tenants() {
        let isolation = TenantIsolation::new(IsolationConfig::default());

        let tenant1 = TenantId::new("tenant1").expect("Valid tenant ID");
        let tenant2 = TenantId::new("tenant2").expect("Valid tenant ID");

        isolation
            .register_tenant(TenantConfig {
                id: tenant1.clone(),
                limits: ResourceLimits::default(),
                strict_isolation: true,
                enable_monitoring: true,
            })
            .await
            .expect("Registration");

        isolation
            .register_tenant(TenantConfig {
                id: tenant2.clone(),
                limits: ResourceLimits::default(),
                strict_isolation: true,
                enable_monitoring: true,
            })
            .await
            .expect("Registration");

        let tenants = isolation.list_tenants().await;
        assert_eq!(tenants.len(), 2);
        assert!(tenants.contains(&tenant1));
        assert!(tenants.contains(&tenant2));
    }

    #[tokio::test]
    async fn test_quota_enforcement_disabled() {
        let mut config = IsolationConfig::default();
        config.enforce_quotas = false;

        let isolation = TenantIsolation::new(config);
        let tenant_id = TenantId::new("test-tenant").expect("Valid tenant ID");

        isolation
            .register_tenant(TenantConfig {
                id: tenant_id.clone(),
                limits: ResourceLimits {
                    max_memory_mb: 100,
                    ..Default::default()
                },
                strict_isolation: true,
                enable_monitoring: true,
            })
            .await
            .expect("Registration");

        // This should succeed even though it exceeds limits
        let usage = ResourceUsage {
            memory_mb: 150.0,
            ..Default::default()
        };

        let result = isolation
            .validate_resource_request(&tenant_id, &usage)
            .await;
        assert!(result.is_ok());
    }
}
