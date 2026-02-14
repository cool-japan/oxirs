//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;
use anyhow::Result;
use tracing::{debug, warn};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get nodes in a specific region
    pub async fn get_nodes_in_region(&self, region_id: &str) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;
        topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.region_id == region_id)
            .map(|(node_id, _)| *node_id)
            .collect()
    }
    /// Send data to a specific region
    pub(super) async fn send_data_to_region(
        &self,
        data: &[u8],
        target_region: &str,
    ) -> Result<Option<Vec<u8>>> {
        let target_nodes = self.get_nodes_in_region(target_region).await;
        if target_nodes.is_empty() {
            return Err(anyhow::anyhow!(
                "No nodes available in target region {}",
                target_region
            ));
        }
        let mut last_error = None;
        for &node_id in target_nodes.iter().take(3) {
            match self.send_data_to_node(data, node_id).await {
                Ok(response) => {
                    debug!(
                        "Successfully sent data to node {} in region {}",
                        node_id, target_region
                    );
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        "Failed to send data to node {} in region {}: {}",
                        node_id, target_region, e
                    );
                    last_error = Some(e);
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("All nodes in region {} failed", target_region)))
    }
}
