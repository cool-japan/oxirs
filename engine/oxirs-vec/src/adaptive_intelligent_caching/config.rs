//! Configuration for the adaptive intelligent caching system

use serde::{Deserialize, Serialize};

/// Configuration for the adaptive cache system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfiguration {
    /// Maximum total cache size in bytes
    pub max_total_size_bytes: u64,
    /// Number of cache tiers
    pub num_tiers: u32,
    /// Tier size distribution
    pub tier_size_ratios: Vec<f64>,
    /// Default TTL for cached items
    pub default_ttl_seconds: u64,
    /// Optimization frequency
    pub optimization_interval_seconds: u64,
    /// ML model update frequency
    pub ml_update_interval_seconds: u64,
    /// Enable predictive prefetching
    pub enable_prefetching: bool,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Performance monitoring settings
    pub monitoring_config: MonitoringConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfiguration {
    pub enable_detailed_metrics: bool,
    pub metrics_retention_days: u32,
    pub alert_thresholds: AlertThresholds,
    pub export_prometheus: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub min_hit_rate: f64,
    pub max_latency_p99_ms: f64,
    pub max_memory_utilization: f64,
    pub min_cache_efficiency: f64,
}

impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            max_total_size_bytes: 1024 * 1024 * 1024, // 1GB
            num_tiers: 3,
            tier_size_ratios: vec![0.5, 0.3, 0.2], // 50%, 30%, 20%
            default_ttl_seconds: 3600,             // 1 hour
            optimization_interval_seconds: 300,    // 5 minutes
            ml_update_interval_seconds: 900,       // 15 minutes
            enable_prefetching: true,
            enable_adaptive_optimization: true,
            monitoring_config: MonitoringConfiguration {
                enable_detailed_metrics: true,
                metrics_retention_days: 7,
                alert_thresholds: AlertThresholds {
                    min_hit_rate: 0.8,
                    max_latency_p99_ms: 100.0,
                    max_memory_utilization: 0.9,
                    min_cache_efficiency: 0.7,
                },
                export_prometheus: true,
            },
        }
    }
}
