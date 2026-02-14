//! # RegionManager - new_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;
use crate::raft::OxirsNodeId;
use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use super::types::{
    ConsensusStrategy, CrossRegionStrategy, MultiRegionReplicationStrategy,
    RegionPerformanceMetrics, RegionTopology, VectorClock,
};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Create a new multi-region manager
    pub fn new(
        local_region: String,
        local_availability_zone: String,
        consensus_strategy: ConsensusStrategy,
        replication_strategy: MultiRegionReplicationStrategy,
    ) -> Self {
        Self {
            topology: RwLock::new(RegionTopology {
                regions: HashMap::new(),
                node_placements: HashMap::new(),
                latency_matrix: HashMap::new(),
                connectivity_status: HashMap::new(),
            }),
            local_region,
            local_availability_zone,
            consensus_strategy,
            replication_strategy,
            performance_metrics: RwLock::new(RegionPerformanceMetrics::default()),
        }
    }
    /// Calculate replication targets for a given region
    pub async fn calculate_replication_targets(&self, source_region: &str) -> Result<Vec<String>> {
        let topology = self.topology.read().await;
        let source_region_config = topology
            .regions
            .get(source_region)
            .ok_or_else(|| anyhow::anyhow!("Unknown source region: {}", source_region))?;
        let mut targets = Vec::new();
        match &self.replication_strategy.cross_region {
            CrossRegionStrategy::AsyncAll => {
                for region_id in topology.regions.keys() {
                    if region_id != source_region {
                        targets.push(region_id.clone());
                    }
                }
            }
            CrossRegionStrategy::SelectiveSync { target_regions } => {
                for target_region in target_regions {
                    if target_region != source_region
                        && topology.regions.contains_key(target_region)
                    {
                        targets.push(target_region.clone());
                    }
                }
            }
            CrossRegionStrategy::EventualConsistency { .. } => {
                let nearby_regions = self.get_nearby_regions(source_region, &topology);
                targets.extend(
                    nearby_regions
                        .into_iter()
                        .take(source_region_config.config.cross_region_replication_factor),
                );
            }
            CrossRegionStrategy::ChainReplication { replication_chain } => {
                if let Some(pos) = replication_chain.iter().position(|r| r == source_region) {
                    if pos + 1 < replication_chain.len() {
                        targets.push(replication_chain[pos + 1].clone());
                    }
                }
            }
        }
        Ok(targets)
    }
    /// Measure actual inter-region latency
    pub(super) async fn measure_inter_region_latency(
        &self,
        region_a: &str,
        region_b: &str,
    ) -> Result<f64> {
        use std::time::Instant;
        use tokio::time::timeout;
        let nodes_a = self.get_nodes_in_region(region_a).await;
        let nodes_b = self.get_nodes_in_region(region_b).await;
        if nodes_a.is_empty() || nodes_b.is_empty() {
            let topology = self.topology.read().await;
            return Ok(topology
                .latency_matrix
                .get(&(region_a.to_string(), region_b.to_string()))
                .copied()
                .unwrap_or(1000.0));
        }
        let node_addresses_a = self.get_node_addresses(&nodes_a).await?;
        let node_addresses_b = self.get_node_addresses(&nodes_b).await?;
        let mut measurements = Vec::new();
        let samples_per_pair = 3;
        let measurement_timeout = Duration::from_secs(5);
        for _addr_a in node_addresses_a.iter().take(3) {
            for addr_b in node_addresses_b.iter().take(3) {
                for _ in 0..samples_per_pair {
                    let start = Instant::now();
                    match timeout(measurement_timeout, self.ping_node(*addr_b)).await {
                        Ok(Ok(_)) => {
                            let latency = start.elapsed().as_secs_f64() * 1000.0;
                            measurements.push(latency);
                        }
                        Ok(Err(_)) | Err(_) => {
                            continue;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
        if measurements.is_empty() {
            warn!(
                "All latency measurements failed between {} and {}, using estimated latency",
                region_a, region_b
            );
            let topology = self.topology.read().await;
            return Ok(topology
                .latency_matrix
                .get(&(region_a.to_string(), region_b.to_string()))
                .copied()
                .unwrap_or(1000.0));
        }
        measurements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let avg_latency = measurements.iter().sum::<f64>() / measurements.len() as f64;
        {
            let mut topology = self.topology.write().await;
            topology
                .latency_matrix
                .insert((region_a.to_string(), region_b.to_string()), avg_latency);
            topology
                .latency_matrix
                .insert((region_b.to_string(), region_a.to_string()), avg_latency);
        }
        debug!(
            "Measured latency between {} and {}: {}ms (from {} samples)",
            region_a,
            region_b,
            avg_latency,
            measurements.len()
        );
        Ok(avg_latency)
    }
    /// Get network addresses for the given node IDs
    pub(super) async fn get_node_addresses(
        &self,
        node_ids: &[OxirsNodeId],
    ) -> Result<Vec<SocketAddr>> {
        let mut addresses = Vec::new();
        for &node_id in node_ids {
            if let Some(addr) = self.get_node_address(node_id).await? {
                addresses.push(addr);
            }
        }
        Ok(addresses)
    }
    /// Generate vector clock for eventual consistency
    pub(super) async fn generate_vector_clock(&self) -> Result<VectorClock> {
        let topology = self.topology.read().await;
        let mut clock = HashMap::new();
        for region_id in topology.regions.keys() {
            clock.insert(
                region_id.clone(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_millis() as u64,
            );
        }
        Ok(VectorClock { clocks: clock })
    }
    /// Reconstruct path from predecessors
    pub(super) fn reconstruct_path(
        &self,
        predecessors: &[Option<usize>],
        source: usize,
        dest: usize,
        region_ids: &[String],
    ) -> ClusterResult<Vec<String>> {
        let mut path = Vec::new();
        let mut current = dest;
        while current != source {
            path.push(region_ids[current].clone());
            match predecessors[current] {
                Some(pred) => current = pred,
                None => {
                    return Err(crate::error::ClusterError::Network(format!(
                        "No path found from {} to {}",
                        region_ids[source], region_ids[dest]
                    )));
                }
            }
        }
        path.push(region_ids[source].clone());
        path.reverse();
        Ok(path)
    }
}
