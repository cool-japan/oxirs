//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get nodes in a specific availability zone
    pub async fn get_nodes_in_availability_zone(
        &self,
        availability_zone_id: &str,
    ) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;
        topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.availability_zone_id == availability_zone_id)
            .map(|(node_id, _)| *node_id)
            .collect()
    }
}
