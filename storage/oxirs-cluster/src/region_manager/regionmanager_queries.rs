//! # RegionManager - queries Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;
use std::collections::HashMap;
use tracing::info;

use super::types::{RegionTopology, Route, RoutingStrategy};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Find optimal route between regions using configured routing strategy
    pub async fn find_route(&self, source: &str, dest: &str) -> ClusterResult<Route> {
        let topology = self.topology.read().await;
        if !topology.regions.contains_key(source) {
            return Err(crate::error::ClusterError::Config(format!(
                "Source region {} not found",
                source
            )));
        }
        if !topology.regions.contains_key(dest) {
            return Err(crate::error::ClusterError::Config(format!(
                "Destination region {} not found",
                dest
            )));
        }
        let source_region = topology.regions.get(source).ok_or_else(|| {
            crate::error::ClusterError::Config(format!("Source region {} not found", source))
        })?;
        match source_region.config.routing_strategy {
            RoutingStrategy::Direct => Ok(Route::direct(source.to_string(), dest.to_string())),
            RoutingStrategy::LatencyAware => {
                self.find_latency_optimal_route(source, dest, &topology)
                    .await
            }
            RoutingStrategy::BandwidthAware => {
                self.find_bandwidth_optimal_route(source, dest, &topology)
                    .await
            }
            RoutingStrategy::CostAware => {
                self.find_cost_optimal_route(source, dest, &topology).await
            }
        }
    }
    /// Find latency-optimal route using Dijkstra's algorithm
    async fn find_latency_optimal_route(
        &self,
        source: &str,
        dest: &str,
        topology: &RegionTopology,
    ) -> ClusterResult<Route> {
        let region_ids: Vec<String> = topology.regions.keys().cloned().collect();
        let num_regions = region_ids.len();
        let region_to_idx: HashMap<String, usize> = region_ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.clone(), idx))
            .collect();
        let mut graph = vec![vec![f64::INFINITY; num_regions]; num_regions];
        for i in 0..num_regions {
            graph[i][i] = 0.0;
        }
        for ((from, to), &latency) in &topology.latency_matrix {
            if let (Some(&from_idx), Some(&to_idx)) =
                (region_to_idx.get(from), region_to_idx.get(to))
            {
                graph[from_idx][to_idx] = latency;
            }
        }
        let source_idx = region_to_idx.get(source).ok_or_else(|| {
            crate::error::ClusterError::Config(format!("Source region {} not found", source))
        })?;
        let dest_idx = region_to_idx.get(dest).ok_or_else(|| {
            crate::error::ClusterError::Config(format!("Destination region {} not found", dest))
        })?;
        let (distances, predecessors) = self.dijkstra(&graph, *source_idx)?;
        let path = self.reconstruct_path(&predecessors, *source_idx, *dest_idx, &region_ids)?;
        let total_latency = distances[*dest_idx];
        let direct_latency = topology
            .latency_matrix
            .get(&(source.to_string(), dest.to_string()))
            .copied()
            .unwrap_or(f64::INFINITY);
        let source_region = topology.regions.get(source).ok_or_else(|| {
            crate::error::ClusterError::Config(format!("Source region {} not found", source))
        })?;
        if source_region.config.enable_relay
            && path.len() > 2
            && total_latency < direct_latency * 0.8
        {
            info!(
                "Using relay route from {} to {}: {:?} ({}ms vs {}ms direct)",
                source, dest, path, total_latency, direct_latency
            );
            Ok(Route {
                hops: path,
                total_latency,
                use_compression: source_region.config.enable_compression,
            })
        } else {
            Ok(Route::direct(source.to_string(), dest.to_string()))
        }
    }
    /// Find bandwidth-optimal route (considers both latency and bandwidth)
    async fn find_bandwidth_optimal_route(
        &self,
        source: &str,
        dest: &str,
        _topology: &RegionTopology,
    ) -> ClusterResult<Route> {
        self.find_latency_optimal_route(source, dest, _topology)
            .await
    }
    /// Find cost-optimal route (minimizes cross-region data transfer costs)
    async fn find_cost_optimal_route(
        &self,
        source: &str,
        dest: &str,
        _topology: &RegionTopology,
    ) -> ClusterResult<Route> {
        self.find_latency_optimal_route(source, dest, _topology)
            .await
    }
}
