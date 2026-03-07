//! # ElasticScalingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ElasticScalingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CloudProvider, ElasticScalingConfig, InstanceType};

impl Default for ElasticScalingConfig {
    fn default() -> Self {
        Self {
            min_nodes: 3,
            max_nodes: 100,
            target_cpu_utilization: 0.70,
            target_memory_utilization: 0.75,
            scale_up_threshold: 0.80,
            scale_down_threshold: 0.30,
            cooldown_seconds: 300,
            use_spot_instances: true,
            max_spot_ratio: 0.50,
            instance_types: vec![
                InstanceType {
                    name: "small".to_string(),
                    vcpus: 2,
                    memory_gb: 4,
                    hourly_cost: 0.05,
                    spot_hourly_cost: 0.015,
                },
                InstanceType {
                    name: "medium".to_string(),
                    vcpus: 4,
                    memory_gb: 8,
                    hourly_cost: 0.10,
                    spot_hourly_cost: 0.03,
                },
                InstanceType {
                    name: "large".to_string(),
                    vcpus: 8,
                    memory_gb: 16,
                    hourly_cost: 0.20,
                    spot_hourly_cost: 0.06,
                },
            ],
            provider: CloudProvider::AWS,
        }
    }
}
