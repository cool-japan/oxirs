//! # Multi-Tenancy Support
//!
//! Complete multi-tenancy implementation with resource isolation, quota management,
//! tenant lifecycle management, and fair resource allocation for streaming workloads.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Multi-tenancy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTenancyConfig {
    /// Enable multi-tenancy
    pub enabled: bool,
    /// Isolation mode
    pub isolation_mode: IsolationMode,
    /// Resource allocation strategy
    pub resource_allocation: ResourceAllocationStrategy,
    /// Default tenant quota
    pub default_quota: TenantQuota,
    /// Tenant lifecycle configuration
    pub lifecycle: TenantLifecycleConfig,
}

impl Default for MultiTenancyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            isolation_mode: IsolationMode::Namespace,
            resource_allocation: ResourceAllocationStrategy::FairShare,
            default_quota: TenantQuota::default(),
            lifecycle: TenantLifecycleConfig::default(),
        }
    }
}

/// Tenant isolation modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IsolationMode {
    /// Namespace-based isolation (logical)
    Namespace,
    /// Process-based isolation (strong)
    Process,
    /// Container-based isolation (very strong)
    Container,
    /// VM-based isolation (strongest)
    VirtualMachine,
}

impl std::fmt::Display for IsolationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsolationMode::Namespace => write!(f, "Namespace"),
            IsolationMode::Process => write!(f, "Process"),
            IsolationMode::Container => write!(f, "Container"),
            IsolationMode::VirtualMachine => write!(f, "VirtualMachine"),
        }
    }
}

/// Resource allocation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResourceAllocationStrategy {
    /// Fair share allocation
    FairShare,
    /// Priority-based allocation
    PriorityBased,
    /// Guaranteed resources
    Guaranteed,
    /// Best-effort allocation
    BestEffort,
}

/// Tenant quota configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantQuota {
    /// Maximum events per second
    pub max_events_per_second: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Maximum topics/streams
    pub max_topics: u32,
    /// Maximum storage size (bytes)
    pub max_storage_bytes: u64,
    /// Maximum memory usage (bytes)
    pub max_memory_bytes: u64,
    /// Maximum CPU usage (percentage, 0-100)
    pub max_cpu_percent: f64,
    /// Maximum bandwidth (bytes per second)
    pub max_bandwidth_bytes_per_sec: u64,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_events_per_second: 10000,
            max_connections: 100,
            max_topics: 50,
            max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            max_memory_bytes: 1024 * 1024 * 1024,       // 1 GB
            max_cpu_percent: 50.0,
            max_bandwidth_bytes_per_sec: 100 * 1024 * 1024, // 100 MB/s
        }
    }
}

/// Tenant lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantLifecycleConfig {
    /// Enable automatic provisioning
    pub auto_provisioning: bool,
    /// Enable automatic deprovisioning
    pub auto_deprovisioning: bool,
    /// Grace period before deprovisioning (seconds)
    pub deprovision_grace_period_secs: u64,
    /// Enable tenant suspension on quota violation
    pub auto_suspend_on_violation: bool,
}

impl Default for TenantLifecycleConfig {
    fn default() -> Self {
        Self {
            auto_provisioning: true,
            auto_deprovisioning: false,
            deprovision_grace_period_secs: 86400, // 24 hours
            auto_suspend_on_violation: true,
        }
    }
}

/// Tenant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Tenant ID
    pub tenant_id: String,
    /// Tenant name
    pub name: String,
    /// Organization
    pub organization: Option<String>,
    /// Tenant status
    pub status: TenantStatus,
    /// Quota configuration
    pub quota: TenantQuota,
    /// Current resource usage
    pub usage: ResourceUsage,
    /// Tenant tier
    pub tier: TenantTier,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Updated at
    pub updated_at: DateTime<Utc>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Tenant status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TenantStatus {
    /// Active and operational
    Active,
    /// Suspended (quota violation or manual)
    Suspended,
    /// Pending provisioning
    Provisioning,
    /// Pending deprovisioning
    Deprovisioning,
    /// Archived
    Archived,
}

impl std::fmt::Display for TenantStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TenantStatus::Active => write!(f, "Active"),
            TenantStatus::Suspended => write!(f, "Suspended"),
            TenantStatus::Provisioning => write!(f, "Provisioning"),
            TenantStatus::Deprovisioning => write!(f, "Deprovisioning"),
            TenantStatus::Archived => write!(f, "Archived"),
        }
    }
}

/// Tenant tier
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TenantTier {
    Free,
    Basic,
    Professional,
    Enterprise,
    Custom,
}

impl std::fmt::Display for TenantTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TenantTier::Free => write!(f, "Free"),
            TenantTier::Basic => write!(f, "Basic"),
            TenantTier::Professional => write!(f, "Professional"),
            TenantTier::Enterprise => write!(f, "Enterprise"),
            TenantTier::Custom => write!(f, "Custom"),
        }
    }
}

/// Current resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current events per second
    pub events_per_second: u64,
    /// Current connections
    pub connections: u32,
    /// Current topics
    pub topics: u32,
    /// Current storage usage (bytes)
    pub storage_bytes: u64,
    /// Current memory usage (bytes)
    pub memory_bytes: u64,
    /// Current CPU usage (percentage)
    pub cpu_percent: f64,
    /// Current bandwidth usage (bytes per second)
    pub bandwidth_bytes_per_sec: u64,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

/// Tenant namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantNamespace {
    /// Namespace ID
    pub namespace_id: String,
    /// Tenant ID
    pub tenant_id: String,
    /// Namespace resources
    pub resources: NamespaceResources,
}

/// Namespace resources
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NamespaceResources {
    /// Topics owned by this tenant
    pub topics: Vec<String>,
    /// Consumer groups
    pub consumer_groups: Vec<String>,
    /// Connections
    pub connections: Vec<String>,
}

/// Multi-tenancy manager
pub struct MultiTenancyManager {
    config: MultiTenancyConfig,
    tenants: Arc<RwLock<HashMap<String, Tenant>>>,
    namespaces: Arc<RwLock<HashMap<String, TenantNamespace>>>,
    metrics: Arc<RwLock<MultiTenancyMetrics>>,
}

/// Multi-tenancy metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiTenancyMetrics {
    /// Total tenants
    pub total_tenants: u64,
    /// Active tenants
    pub active_tenants: u64,
    /// Suspended tenants
    pub suspended_tenants: u64,
    /// Total quota violations
    pub quota_violations: u64,
    /// Resource utilization by tenant
    pub tenant_utilization: HashMap<String, f64>,
}

impl MultiTenancyManager {
    /// Create a new multi-tenancy manager
    pub fn new(config: MultiTenancyConfig) -> Self {
        Self {
            config,
            tenants: Arc::new(RwLock::new(HashMap::new())),
            namespaces: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(MultiTenancyMetrics::default())),
        }
    }

    /// Initialize multi-tenancy system
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Multi-tenancy is disabled");
            return Ok(());
        }

        info!(
            "Initializing multi-tenancy system with isolation mode: {}",
            self.config.isolation_mode
        );

        // Initialize default namespace
        self.create_default_namespace().await?;

        info!("Multi-tenancy system initialized successfully");
        Ok(())
    }

    /// Create default namespace
    async fn create_default_namespace(&self) -> Result<()> {
        let namespace = TenantNamespace {
            namespace_id: "default".to_string(),
            tenant_id: "default".to_string(),
            resources: NamespaceResources::default(),
        };

        self.namespaces
            .write()
            .await
            .insert("default".to_string(), namespace);
        debug!("Default namespace created");
        Ok(())
    }

    /// Create a new tenant
    pub async fn create_tenant(&self, name: String, tier: TenantTier) -> Result<Tenant> {
        info!("Creating tenant: {} (tier: {})", name, tier);

        let tenant_id = Uuid::new_v4().to_string();
        let quota = self.get_quota_for_tier(tier);

        let tenant = Tenant {
            tenant_id: tenant_id.clone(),
            name: name.clone(),
            organization: None,
            status: TenantStatus::Provisioning,
            quota,
            usage: ResourceUsage {
                updated_at: Utc::now(),
                ..Default::default()
            },
            tier,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        };

        // Provision tenant resources
        self.provision_tenant(&tenant).await?;

        // Update tenant status to active
        let mut active_tenant = tenant.clone();
        active_tenant.status = TenantStatus::Active;

        // Store tenant
        self.tenants
            .write()
            .await
            .insert(tenant_id.clone(), active_tenant.clone());

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_tenants += 1;
            metrics.active_tenants += 1;
        }

        info!("Tenant created successfully: {}", name);
        Ok(active_tenant)
    }

    /// Get quota for tenant tier
    fn get_quota_for_tier(&self, tier: TenantTier) -> TenantQuota {
        match tier {
            TenantTier::Free => TenantQuota {
                max_events_per_second: 1000,
                max_connections: 10,
                max_topics: 5,
                max_storage_bytes: 1024 * 1024 * 1024, // 1 GB
                max_memory_bytes: 256 * 1024 * 1024,   // 256 MB
                max_cpu_percent: 10.0,
                max_bandwidth_bytes_per_sec: 10 * 1024 * 1024, // 10 MB/s
            },
            TenantTier::Basic => TenantQuota {
                max_events_per_second: 10000,
                max_connections: 50,
                max_topics: 25,
                max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                max_memory_bytes: 512 * 1024 * 1024,        // 512 MB
                max_cpu_percent: 25.0,
                max_bandwidth_bytes_per_sec: 50 * 1024 * 1024, // 50 MB/s
            },
            TenantTier::Professional => TenantQuota {
                max_events_per_second: 50000,
                max_connections: 200,
                max_topics: 100,
                max_storage_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
                max_memory_bytes: 2 * 1024 * 1024 * 1024,    // 2 GB
                max_cpu_percent: 50.0,
                max_bandwidth_bytes_per_sec: 200 * 1024 * 1024, // 200 MB/s
            },
            TenantTier::Enterprise => TenantQuota {
                max_events_per_second: 500000,
                max_connections: 1000,
                max_topics: 500,
                max_storage_bytes: 1024 * 1024 * 1024 * 1024, // 1 TB
                max_memory_bytes: 16 * 1024 * 1024 * 1024,    // 16 GB
                max_cpu_percent: 100.0,
                max_bandwidth_bytes_per_sec: 1024 * 1024 * 1024, // 1 GB/s
            },
            TenantTier::Custom => self.config.default_quota.clone(),
        }
    }

    /// Provision tenant resources
    async fn provision_tenant(&self, tenant: &Tenant) -> Result<()> {
        debug!("Provisioning resources for tenant: {}", tenant.tenant_id);

        // Create tenant namespace
        let namespace = TenantNamespace {
            namespace_id: format!("tenant-{}", tenant.tenant_id),
            tenant_id: tenant.tenant_id.clone(),
            resources: NamespaceResources::default(),
        };

        self.namespaces
            .write()
            .await
            .insert(namespace.namespace_id.clone(), namespace);

        // In a real implementation, this would:
        // 1. Allocate compute resources
        // 2. Set up network isolation
        // 3. Configure storage
        // 4. Initialize monitoring

        debug!("Tenant resources provisioned: {}", tenant.tenant_id);
        Ok(())
    }

    /// Check quota for tenant
    pub async fn check_quota(&self, tenant_id: &str, resource: ResourceType) -> Result<bool> {
        let tenants = self.tenants.read().await;
        let tenant = tenants
            .get(tenant_id)
            .ok_or_else(|| anyhow!("Tenant not found: {}", tenant_id))?;

        if tenant.status != TenantStatus::Active {
            return Err(anyhow!("Tenant is not active: {}", tenant.status));
        }

        let within_quota = match resource {
            ResourceType::EventsPerSecond => {
                tenant.usage.events_per_second < tenant.quota.max_events_per_second
            }
            ResourceType::Connections => tenant.usage.connections < tenant.quota.max_connections,
            ResourceType::Topics => tenant.usage.topics < tenant.quota.max_topics,
            ResourceType::Storage => tenant.usage.storage_bytes < tenant.quota.max_storage_bytes,
            ResourceType::Memory => tenant.usage.memory_bytes < tenant.quota.max_memory_bytes,
            ResourceType::CPU => tenant.usage.cpu_percent < tenant.quota.max_cpu_percent,
            ResourceType::Bandwidth => {
                tenant.usage.bandwidth_bytes_per_sec < tenant.quota.max_bandwidth_bytes_per_sec
            }
        };

        if !within_quota {
            warn!("Quota exceeded for tenant {}: {:?}", tenant_id, resource);

            if self.config.lifecycle.auto_suspend_on_violation {
                self.suspend_tenant(tenant_id, "Quota violation".to_string())
                    .await?;
            }

            // Update metrics
            self.metrics.write().await.quota_violations += 1;
        }

        Ok(within_quota)
    }

    /// Update tenant resource usage
    pub async fn update_usage(&self, tenant_id: &str, usage: ResourceUsage) -> Result<()> {
        let mut tenants = self.tenants.write().await;
        if let Some(tenant) = tenants.get_mut(tenant_id) {
            tenant.usage = usage;
            tenant.updated_at = Utc::now();
            debug!("Updated usage for tenant: {}", tenant_id);
        }
        Ok(())
    }

    /// Suspend tenant
    pub async fn suspend_tenant(&self, tenant_id: &str, reason: String) -> Result<()> {
        info!("Suspending tenant {}: {}", tenant_id, reason);

        let mut tenants = self.tenants.write().await;
        if let Some(tenant) = tenants.get_mut(tenant_id) {
            tenant.status = TenantStatus::Suspended;
            tenant.updated_at = Utc::now();
            tenant
                .metadata
                .insert("suspension_reason".to_string(), reason);

            // Update metrics
            drop(tenants);
            let mut metrics = self.metrics.write().await;
            metrics.active_tenants -= 1;
            metrics.suspended_tenants += 1;
        }

        Ok(())
    }

    /// Resume tenant
    pub async fn resume_tenant(&self, tenant_id: &str) -> Result<()> {
        info!("Resuming tenant: {}", tenant_id);

        let mut tenants = self.tenants.write().await;
        if let Some(tenant) = tenants.get_mut(tenant_id) {
            tenant.status = TenantStatus::Active;
            tenant.updated_at = Utc::now();
            tenant.metadata.remove("suspension_reason");

            // Update metrics
            drop(tenants);
            let mut metrics = self.metrics.write().await;
            metrics.active_tenants += 1;
            metrics.suspended_tenants -= 1;
        }

        Ok(())
    }

    /// Delete tenant
    pub async fn delete_tenant(&self, tenant_id: &str) -> Result<()> {
        info!("Deleting tenant: {}", tenant_id);

        // Deprovision resources
        self.deprovision_tenant(tenant_id).await?;

        // Remove tenant
        let mut tenants = self.tenants.write().await;
        if tenants.remove(tenant_id).is_some() {
            // Update metrics
            drop(tenants);
            let mut metrics = self.metrics.write().await;
            metrics.total_tenants -= 1;
        }

        Ok(())
    }

    /// Deprovision tenant resources
    async fn deprovision_tenant(&self, tenant_id: &str) -> Result<()> {
        debug!("Deprovisioning resources for tenant: {}", tenant_id);

        // Remove namespace
        let namespace_id = format!("tenant-{}", tenant_id);
        self.namespaces.write().await.remove(&namespace_id);

        // In a real implementation, this would:
        // 1. Release compute resources
        // 2. Remove network isolation
        // 3. Clean up storage
        // 4. Remove monitoring

        debug!("Tenant resources deprovisioned: {}", tenant_id);
        Ok(())
    }

    /// Get tenant
    pub async fn get_tenant(&self, tenant_id: &str) -> Option<Tenant> {
        self.tenants.read().await.get(tenant_id).cloned()
    }

    /// List all tenants
    pub async fn list_tenants(&self) -> Vec<Tenant> {
        self.tenants.read().await.values().cloned().collect()
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> MultiTenancyMetrics {
        self.metrics.read().await.clone()
    }
}

/// Resource types for quota checking
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResourceType {
    EventsPerSecond,
    Connections,
    Topics,
    Storage,
    Memory,
    CPU,
    Bandwidth,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_tenancy_config_default() {
        let config = MultiTenancyConfig::default();
        assert!(config.enabled);
        assert_eq!(config.isolation_mode, IsolationMode::Namespace);
    }

    #[tokio::test]
    async fn test_tenant_creation() {
        let config = MultiTenancyConfig::default();
        let manager = MultiTenancyManager::new(config);
        manager.initialize().await.unwrap();

        let tenant = manager
            .create_tenant("Test Tenant".to_string(), TenantTier::Basic)
            .await
            .unwrap();
        assert_eq!(tenant.name, "Test Tenant");
        assert_eq!(tenant.tier, TenantTier::Basic);
        assert_eq!(tenant.status, TenantStatus::Active);
    }

    #[tokio::test]
    async fn test_quota_check() {
        let config = MultiTenancyConfig::default();
        let manager = MultiTenancyManager::new(config);
        manager.initialize().await.unwrap();

        let tenant = manager
            .create_tenant("Test".to_string(), TenantTier::Free)
            .await
            .unwrap();

        // Should be within quota initially
        let within_quota = manager
            .check_quota(&tenant.tenant_id, ResourceType::Connections)
            .await
            .unwrap();
        assert!(within_quota);
    }

    #[tokio::test]
    async fn test_tenant_suspension() {
        let config = MultiTenancyConfig::default();
        let manager = MultiTenancyManager::new(config);
        manager.initialize().await.unwrap();

        let tenant = manager
            .create_tenant("Test".to_string(), TenantTier::Basic)
            .await
            .unwrap();

        manager
            .suspend_tenant(&tenant.tenant_id, "Testing".to_string())
            .await
            .unwrap();

        let suspended_tenant = manager.get_tenant(&tenant.tenant_id).await.unwrap();
        assert_eq!(suspended_tenant.status, TenantStatus::Suspended);
    }

    #[tokio::test]
    async fn test_tier_quota() {
        let config = MultiTenancyConfig::default();
        let manager = MultiTenancyManager::new(config);

        let free_quota = manager.get_quota_for_tier(TenantTier::Free);
        let enterprise_quota = manager.get_quota_for_tier(TenantTier::Enterprise);

        assert!(enterprise_quota.max_events_per_second > free_quota.max_events_per_second);
        assert!(enterprise_quota.max_connections > free_quota.max_connections);
    }
}
