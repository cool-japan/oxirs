//! Configuration for hot/warm/cold tiering system

use super::policies::TieringPolicy;
use super::types::{GradualTransitionConfig, TierCostModel};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Main configuration for tiering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Hot tier capacity in gigabytes
    pub hot_tier_capacity_gb: f64,
    /// Warm tier capacity in gigabytes
    pub warm_tier_capacity_gb: f64,
    /// Cold tier capacity in gigabytes
    pub cold_tier_capacity_gb: f64,
    /// Tiering policy
    pub policy: TieringPolicy,
    /// Base directory for storage
    pub storage_base_path: PathBuf,
    /// Enable automatic tier management
    pub auto_tier_management: bool,
    /// Tier evaluation interval
    pub evaluation_interval: Duration,
    /// Hot tier utilization threshold for demotion (0.0 - 1.0)
    pub hot_tier_utilization_threshold: f64,
    /// Warm tier utilization threshold for demotion (0.0 - 1.0)
    pub warm_tier_utilization_threshold: f64,
    /// Access frequency threshold for hot tier (queries/second)
    pub hot_tier_qps_threshold: f64,
    /// Access frequency threshold for warm tier (queries/second)
    pub warm_tier_qps_threshold: f64,
    /// Minimum time in tier before transition (prevents thrashing)
    pub min_time_in_tier: Duration,
    /// Gradual transition configuration
    pub gradual_transition: GradualTransitionConfig,
    /// Cost model
    pub cost_model: TierCostModel,
    /// Enable predictive tier management
    pub enable_predictive_management: bool,
    /// Enable multi-tenancy support
    pub enable_multi_tenancy: bool,
    /// Maximum concurrent tier transitions
    pub max_concurrent_transitions: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics retention period
    pub metrics_retention: Duration,
    /// Enable compression in warm tier
    pub warm_tier_compression: bool,
    /// Enable compression in cold tier
    pub cold_tier_compression: bool,
    /// Warm tier compression level (1-22)
    pub warm_tier_compression_level: i32,
    /// Cold tier compression level (1-22)
    pub cold_tier_compression_level: i32,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            hot_tier_capacity_gb: 16.0,
            warm_tier_capacity_gb: 128.0,
            cold_tier_capacity_gb: 1024.0,
            policy: TieringPolicy::Adaptive,
            storage_base_path: PathBuf::from("/var/lib/oxirs/tiered-storage"),
            auto_tier_management: true,
            evaluation_interval: Duration::from_secs(300), // 5 minutes
            hot_tier_utilization_threshold: 0.8,           // 80%
            warm_tier_utilization_threshold: 0.9,          // 90%
            hot_tier_qps_threshold: 10.0,                  // 10 queries/sec
            warm_tier_qps_threshold: 1.0,                  // 1 query/sec
            min_time_in_tier: Duration::from_secs(3600),   // 1 hour
            gradual_transition: GradualTransitionConfig::default(),
            cost_model: TierCostModel::default(),
            enable_predictive_management: true,
            enable_multi_tenancy: false,
            max_concurrent_transitions: 4,
            enable_metrics: true,
            metrics_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
            warm_tier_compression: true,
            cold_tier_compression: true,
            warm_tier_compression_level: 6,
            cold_tier_compression_level: 19,
        }
    }
}

impl TieringConfig {
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.hot_tier_capacity_gb <= 0.0 {
            anyhow::bail!("Hot tier capacity must be positive");
        }
        if self.warm_tier_capacity_gb <= 0.0 {
            anyhow::bail!("Warm tier capacity must be positive");
        }
        if self.cold_tier_capacity_gb <= 0.0 {
            anyhow::bail!("Cold tier capacity must be positive");
        }
        if self.hot_tier_utilization_threshold <= 0.0 || self.hot_tier_utilization_threshold > 1.0 {
            anyhow::bail!("Hot tier utilization threshold must be in (0.0, 1.0]");
        }
        if self.warm_tier_utilization_threshold <= 0.0 || self.warm_tier_utilization_threshold > 1.0
        {
            anyhow::bail!("Warm tier utilization threshold must be in (0.0, 1.0]");
        }
        if self.hot_tier_qps_threshold < 0.0 {
            anyhow::bail!("Hot tier QPS threshold must be non-negative");
        }
        if self.warm_tier_qps_threshold < 0.0 {
            anyhow::bail!("Warm tier QPS threshold must be non-negative");
        }
        if self.max_concurrent_transitions == 0 {
            anyhow::bail!("Max concurrent transitions must be at least 1");
        }
        if self.warm_tier_compression_level < 1 || self.warm_tier_compression_level > 22 {
            anyhow::bail!("Warm tier compression level must be in [1, 22]");
        }
        if self.cold_tier_compression_level < 1 || self.cold_tier_compression_level > 22 {
            anyhow::bail!("Cold tier compression level must be in [1, 22]");
        }
        Ok(())
    }

    /// Create a development configuration (smaller capacities)
    pub fn development() -> Self {
        Self {
            hot_tier_capacity_gb: 1.0,
            warm_tier_capacity_gb: 8.0,
            cold_tier_capacity_gb: 64.0,
            storage_base_path: PathBuf::from("/tmp/oxirs-tiered-storage"),
            evaluation_interval: Duration::from_secs(60), // 1 minute
            ..Default::default()
        }
    }

    /// Create a production configuration (larger capacities, conservative settings)
    pub fn production() -> Self {
        Self {
            hot_tier_capacity_gb: 64.0,
            warm_tier_capacity_gb: 512.0,
            cold_tier_capacity_gb: 4096.0,
            evaluation_interval: Duration::from_secs(600), // 10 minutes
            min_time_in_tier: Duration::from_secs(7200),   // 2 hours
            enable_predictive_management: true,
            ..Default::default()
        }
    }

    /// Calculate total capacity in bytes
    pub fn total_capacity_bytes(&self) -> u64 {
        ((self.hot_tier_capacity_gb + self.warm_tier_capacity_gb + self.cold_tier_capacity_gb)
            * 1024.0
            * 1024.0
            * 1024.0) as u64
    }

    /// Get hot tier capacity in bytes
    pub fn hot_tier_capacity_bytes(&self) -> u64 {
        (self.hot_tier_capacity_gb * 1024.0 * 1024.0 * 1024.0) as u64
    }

    /// Get warm tier capacity in bytes
    pub fn warm_tier_capacity_bytes(&self) -> u64 {
        (self.warm_tier_capacity_gb * 1024.0 * 1024.0 * 1024.0) as u64
    }

    /// Get cold tier capacity in bytes
    pub fn cold_tier_capacity_bytes(&self) -> u64 {
        (self.cold_tier_capacity_gb * 1024.0 * 1024.0 * 1024.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = TieringConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_development_config_is_valid() {
        let config = TieringConfig::development();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_production_config_is_valid() {
        let config = TieringConfig::production();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_capacity() {
        let config = TieringConfig {
            hot_tier_capacity_gb: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_threshold() {
        let config = TieringConfig {
            hot_tier_utilization_threshold: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_capacity_calculations() {
        let config = TieringConfig {
            hot_tier_capacity_gb: 1.0,
            warm_tier_capacity_gb: 2.0,
            cold_tier_capacity_gb: 3.0,
            ..Default::default()
        };

        assert_eq!(config.hot_tier_capacity_bytes(), 1073741824); // 1 GB
        assert_eq!(config.warm_tier_capacity_bytes(), 2147483648); // 2 GB
        assert_eq!(config.cold_tier_capacity_bytes(), 3221225472); // 3 GB
        assert_eq!(config.total_capacity_bytes(), 6442450944); // 6 GB
    }
}
