//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;
use anyhow::Result;
use tracing::debug;

use super::types::{RegionHealth, RegionStatus};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get region health status
    pub async fn get_region_health(&self, region_id: &str) -> Result<RegionHealth> {
        let topology = self.topology.read().await;
        let metrics = self.performance_metrics.read().await;
        let _region = topology
            .regions
            .get(region_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown region: {}", region_id))?;
        drop(topology);
        let (healthy_count, total_count) = self.check_nodes_health(region_id).await?;
        let throughput = metrics
            .region_throughput
            .get(region_id)
            .cloned()
            .unwrap_or_default();
        let error_rate = metrics
            .region_error_rates
            .get(region_id)
            .cloned()
            .unwrap_or_default();
        let status = if error_rate.error_rate < 0.01 && healthy_count > 0 {
            RegionStatus::Healthy
        } else if healthy_count > 0 {
            RegionStatus::Degraded
        } else {
            RegionStatus::Unavailable
        };
        Ok(RegionHealth {
            region_id: region_id.to_string(),
            total_nodes: total_count,
            healthy_nodes: healthy_count,
            throughput,
            error_rate,
            status,
        })
    }
    /// Check health of nodes in a region
    /// Returns the count of healthy nodes out of total nodes
    ///
    /// Note: This currently counts all registered nodes as healthy.
    /// For production use, this should be integrated with a proper health monitoring system
    /// that tracks node availability and responsiveness.
    pub async fn check_nodes_health(&self, region_id: &str) -> Result<(usize, usize)> {
        let topology = self.topology.read().await;
        let nodes_in_region: Vec<OxirsNodeId> = topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.region_id == region_id)
            .map(|(node_id, _)| *node_id)
            .collect();
        let total_nodes = nodes_in_region.len();
        let healthy_count = total_nodes;
        debug!(
            "Health check for region {}: {}/{} nodes healthy",
            region_id, healthy_count, total_nodes
        );
        Ok((healthy_count, total_nodes))
    }
}
