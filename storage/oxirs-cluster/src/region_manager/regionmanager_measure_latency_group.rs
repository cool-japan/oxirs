//! # RegionManager - measure_latency_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;
use crate::raft::OxirsNodeId;
use std::time::Instant;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Measure latency between two regions
    pub async fn measure_latency(&self, from: &str, to: &str) -> ClusterResult<f64> {
        let topology = self.topology.read().await;
        let from_nodes: Vec<OxirsNodeId> = topology
            .node_placements
            .iter()
            .filter(|(_, p)| p.region_id == from)
            .map(|(id, _)| *id)
            .collect();
        let to_nodes: Vec<OxirsNodeId> = topology
            .node_placements
            .iter()
            .filter(|(_, p)| p.region_id == to)
            .map(|(id, _)| *id)
            .collect();
        if from_nodes.is_empty() || to_nodes.is_empty() {
            return Err(crate::error::ClusterError::Config(format!(
                "No nodes available in region {} or {}",
                from, to
            )));
        }
        let to_node = to_nodes[0];
        let start = Instant::now();
        self.ping_node_by_id(to_node).await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        drop(topology);
        let mut topology = self.topology.write().await;
        topology
            .latency_matrix
            .insert((from.to_string(), to.to_string()), latency_ms);
        Ok(latency_ms)
    }
}
