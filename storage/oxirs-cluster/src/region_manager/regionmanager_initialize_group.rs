//! # RegionManager - initialize_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{ConnectivityStatus, Region, RegionTopology};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Initialize the region manager with topology configuration
    pub async fn initialize(&self, regions: Vec<Region>) -> Result<()> {
        let mut topology = self.topology.write().await;
        for region in regions {
            topology.regions.insert(region.id.clone(), region);
        }
        let region_ids: Vec<_> = topology.regions.keys().cloned().collect();
        for i in 0..region_ids.len() {
            for j in 0..region_ids.len() {
                let region_pair = (region_ids[i].clone(), region_ids[j].clone());
                if i == j {
                    topology.latency_matrix.insert(region_pair.clone(), 0.0);
                    topology
                        .connectivity_status
                        .insert(region_pair, ConnectivityStatus::Optimal);
                } else {
                    let latency = self.estimate_latency(&region_ids[i], &region_ids[j], &topology);
                    topology.latency_matrix.insert(region_pair.clone(), latency);
                    topology.connectivity_status.insert(
                        region_pair,
                        if latency < 50.0 {
                            ConnectivityStatus::Optimal
                        } else {
                            ConnectivityStatus::Degraded {
                                latency_ms: latency as u64,
                            }
                        },
                    );
                }
            }
        }
        tracing::info!(
            "Initialized multi-region topology with {} regions",
            topology.regions.len()
        );
        Ok(())
    }
    /// Estimate latency between regions based on coordinates
    fn estimate_latency(&self, region_a: &str, region_b: &str, topology: &RegionTopology) -> f64 {
        let region_a_info = topology.regions.get(region_a);
        let region_b_info = topology.regions.get(region_b);
        match (region_a_info, region_b_info) {
            (Some(a), Some(b)) => {
                if let (Some(coord_a), Some(coord_b)) = (&a.coordinates, &b.coordinates) {
                    let distance = self.calculate_distance(coord_a, coord_b);
                    (distance / 200.0) + 10.0
                } else {
                    100.0
                }
            }
            _ => 1000.0,
        }
    }
}
