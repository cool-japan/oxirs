//! Multi-Cloud Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub struct MultiCloudConfig {
    /// Enable multi-cloud deployment
    pub enabled: bool,
    /// Primary cloud provider
    pub primary_provider: CloudProvider,
    /// Secondary cloud providers
    pub secondary_providers: Vec<CloudProvider>,
    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,
    /// Failover configuration
    pub failover: MultiCloudFailoverConfig,
}

impl Default for MultiCloudConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            primary_provider: CloudProvider::AWS,
            secondary_providers: vec![CloudProvider::GCP, CloudProvider::Azure],
            replication_strategy: ReplicationStrategy::ActivePassive,
            failover: MultiCloudFailoverConfig::default(),
        }
    }
}

/// Cloud providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS,
    GCP,
    Azure,
    DigitalOcean,
    Linode,
    OnPremise,
}

/// Replication strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    ActiveActive,
    ActivePassive,
    MultiMaster,
}

/// Multi-cloud failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCloudFailoverConfig {
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Failover threshold
    pub failover_threshold: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Failback enabled
    pub failback_enabled: bool,
}

impl Default for MultiCloudFailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover: true,
            failover_threshold: Duration::from_secs(300),
            health_check_interval: Duration::from_secs(30),
            failback_enabled: true,
        }
    }
}
