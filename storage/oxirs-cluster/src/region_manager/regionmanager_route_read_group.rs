//! # RegionManager - route_read_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;
use tracing::info;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Route read request to nearest replica
    pub async fn route_read(
        &self,
        client_region: &str,
        available_replicas: &[String],
    ) -> ClusterResult<String> {
        if available_replicas.is_empty() {
            return Err(crate::error::ClusterError::Config(
                "No replicas available".to_string(),
            ));
        }
        let topology = self.topology.read().await;
        let source_region = topology.regions.get(client_region).ok_or_else(|| {
            crate::error::ClusterError::Config(format!("Client region {} not found", client_region))
        })?;
        if !source_region.config.enable_read_local {
            return Ok(available_replicas[0].clone());
        }
        let mut best_replica = available_replicas[0].clone();
        let mut best_latency = f64::INFINITY;
        for replica in available_replicas {
            let latency = topology
                .latency_matrix
                .get(&(client_region.to_string(), replica.clone()))
                .copied()
                .unwrap_or(f64::INFINITY);
            if latency < best_latency {
                best_latency = latency;
                best_replica = replica.clone();
            }
        }
        info!(
            "Routed read from {} to {} ({}ms)",
            client_region, best_replica, best_latency
        );
        Ok(best_replica)
    }
}
