//! # RegionManager - perform_region_failover_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::ConnectivityStatus;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Perform region failover
    pub async fn perform_region_failover(
        &self,
        failed_region: &str,
        target_region: &str,
    ) -> Result<()> {
        tracing::warn!(
            "Performing region failover from {} to {}",
            failed_region,
            target_region
        );
        let topology = self.topology.read().await;
        if !topology.regions.contains_key(target_region) {
            return Err(anyhow::anyhow!("Invalid target region: {}", target_region));
        }
        let region_ids: Vec<String> = topology.regions.keys().cloned().collect();
        drop(topology);
        let mut topology = self.topology.write().await;
        for region_id in region_ids {
            topology.connectivity_status.insert(
                (failed_region.to_string(), region_id.clone()),
                ConnectivityStatus::Disconnected,
            );
            topology.connectivity_status.insert(
                (region_id, failed_region.to_string()),
                ConnectivityStatus::Disconnected,
            );
        }
        tracing::info!(
            "Region failover completed from {} to {}",
            failed_region,
            target_region
        );
        Ok(())
    }
}
