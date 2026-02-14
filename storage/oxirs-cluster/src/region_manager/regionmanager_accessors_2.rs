//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::raft::OxirsNodeId;

use super::types::RegionTopology;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get optimal leader candidates for a region
    pub async fn get_leader_candidates(&self, region_id: &str) -> Vec<OxirsNodeId> {
        let topology = self.topology.read().await;
        let mut candidates = topology
            .node_placements
            .iter()
            .filter(|(_, placement)| placement.region_id == region_id)
            .map(|(node_id, _)| *node_id)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            let nearby_regions = self.get_nearby_regions(region_id, &topology);
            for nearby_region in nearby_regions {
                let nearby_candidates: Vec<_> = topology
                    .node_placements
                    .iter()
                    .filter(|(_, placement)| placement.region_id == nearby_region)
                    .map(|(node_id, _)| *node_id)
                    .collect();
                candidates.extend(nearby_candidates);
            }
        }
        candidates
    }
    /// Get nearby regions sorted by latency
    pub(crate) fn get_nearby_regions(
        &self,
        region_id: &str,
        topology: &RegionTopology,
    ) -> Vec<String> {
        let mut region_latencies: Vec<_> = topology
            .latency_matrix
            .iter()
            .filter(|((from, to), _)| from == region_id && to != region_id)
            .map(|((_, to), latency)| (to.clone(), *latency))
            .collect();
        region_latencies
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        region_latencies
            .into_iter()
            .map(|(region, _)| region)
            .collect()
    }
}
