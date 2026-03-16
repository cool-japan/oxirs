//! # CloudIntegrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `CloudIntegrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{
    AutoScalingConfig, CloudIntegrationConfig, CloudProvider, CostOptimizationConfig,
    MonitoringConfig, SecurityConfig,
};

impl Default for CloudIntegrationConfig {
    fn default() -> Self {
        Self {
            default_provider: CloudProvider::AWS,
            auto_scaling: AutoScalingConfig {
                enabled: true,
                min_instances: 1,
                max_instances: 10,
                target_cpu_utilization: 70.0,
                target_memory_utilization: 80.0,
                scale_up_threshold: 80.0,
                scale_down_threshold: 30.0,
                cooldown_period_seconds: 300,
            },
            cost_optimization: CostOptimizationConfig {
                enabled: true,
                max_hourly_cost_usd: 50.0,
                use_spot_instances: false,
                auto_shutdown_idle: true,
                idle_threshold_minutes: 30,
                reserved_capacity_percentage: 20.0,
            },
            security: SecurityConfig {
                encryption_at_rest: true,
                encryption_in_transit: true,
                vpc_config: None,
                iam_config: None,
                network_acl: vec![],
            },
            monitoring: MonitoringConfig {
                enabled: true,
                collection_interval_seconds: 60,
                alert_thresholds: HashMap::from([
                    ("cpu_utilization".to_string(), 85.0),
                    ("memory_utilization".to_string(), 90.0),
                    ("error_rate".to_string(), 0.05),
                ]),
                notification_endpoints: vec![],
            },
        }
    }
}
