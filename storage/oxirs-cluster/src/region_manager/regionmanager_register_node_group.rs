//! # RegionManager - register_node_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;
use anyhow::Result;

use super::types::NodePlacement;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Register a node in a specific region and availability zone
    pub async fn register_node(
        &self,
        node_id: OxirsNodeId,
        region_id: String,
        availability_zone_id: String,
        data_center: Option<String>,
        rack: Option<String>,
    ) -> Result<()> {
        let mut topology = self.topology.write().await;
        if !topology.regions.contains_key(&region_id) {
            return Err(anyhow::anyhow!("Unknown region: {}", region_id));
        }
        let region = topology
            .regions
            .get(&region_id)
            .expect("region should exist after contains_key check");
        if !region
            .availability_zones
            .iter()
            .any(|az| az.id == availability_zone_id)
        {
            return Err(anyhow::anyhow!(
                "Unknown availability zone: {} in region: {}",
                availability_zone_id,
                region_id
            ));
        }
        let placement = NodePlacement {
            node_id,
            region_id: region_id.clone(),
            availability_zone_id: availability_zone_id.clone(),
            data_center,
            rack,
        };
        topology.node_placements.insert(node_id, placement);
        tracing::info!(
            "Registered node {} in region {} AZ {}",
            node_id,
            region_id,
            availability_zone_id
        );
        Ok(())
    }
}
