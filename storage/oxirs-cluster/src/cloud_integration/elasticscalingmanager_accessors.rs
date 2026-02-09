//! # ElasticScalingManager - accessors Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ElasticScalingStatus, ScalingEvent};

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Get scaling status
    pub async fn get_status(&self) -> ElasticScalingStatus {
        let nodes = self.current_nodes.read().await;
        let events = self.scaling_events.read().await;
        let recent_events: Vec<ScalingEvent> = events.iter().rev().take(10).cloned().collect();
        ElasticScalingStatus {
            current_node_count: nodes.len() as u32,
            min_nodes: self.config.min_nodes,
            max_nodes: self.config.max_nodes,
            spot_count: nodes.iter().filter(|n| n.is_spot).count() as u32,
            on_demand_count: nodes.iter().filter(|n| !n.is_spot).count() as u32,
            target_cpu: self.config.target_cpu_utilization,
            target_memory: self.config.target_memory_utilization,
            cooldown_seconds: self.config.cooldown_seconds,
            recent_events,
        }
    }
}
