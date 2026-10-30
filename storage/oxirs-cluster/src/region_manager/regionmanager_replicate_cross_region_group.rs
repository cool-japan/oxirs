//! # RegionManager - replicate_cross_region_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::time::{Duration, SystemTime};
use tracing::{debug, warn};

use super::types::{
    ConnectivityStatus, CrossRegionStrategy, EventualConsistencyMetadata,
    MultiRegionReplicationStrategy, ReplicationPackage,
};

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Perform cross-region replication of data
    pub async fn replicate_cross_region(
        &self,
        data: &[u8],
        source_region: &str,
        replication_strategy: &MultiRegionReplicationStrategy,
    ) -> Result<()> {
        let targets = self.calculate_replication_targets(source_region).await?;
        if targets.is_empty() {
            debug!(
                "No cross-region replication targets for region {}",
                source_region
            );
            return Ok(());
        }
        match &replication_strategy.cross_region {
            CrossRegionStrategy::AsyncAll => self.replicate_async_all(data, &targets).await,
            CrossRegionStrategy::SelectiveSync { .. } => {
                self.replicate_selective_sync(data, &targets).await
            }
            CrossRegionStrategy::EventualConsistency {
                reconciliation_interval_ms,
            } => {
                self.replicate_eventual_consistency(data, &targets, *reconciliation_interval_ms)
                    .await
            }
            CrossRegionStrategy::ChainReplication { .. } => {
                self.replicate_chain(data, &targets).await
            }
        }
    }
    /// Asynchronous replication to all target regions
    ///
    /// Every region is tried; the call fails if any of them was not reached.
    async fn replicate_async_all(&self, data: &[u8], targets: &[String]) -> Result<()> {
        let mut failed = Vec::new();
        for target_region in targets {
            match self.send_data_to_region(data, target_region).await {
                Ok(_) => {
                    debug!("Successfully replicated data to region {}", target_region);
                }
                Err(e) => {
                    warn!(
                        "Failed to replicate data to region {}: {}",
                        target_region, e
                    );
                    failed.push(format!("{target_region}: {e}"));
                }
            }
        }
        replication_outcome(&failed)
    }
    /// Selective synchronization replication
    async fn replicate_selective_sync(&self, data: &[u8], targets: &[String]) -> Result<()> {
        let topology = self.topology.read().await;
        let mut prioritized_targets: Vec<_> = targets
            .iter()
            .map(|region| {
                let connectivity = topology
                    .connectivity_status
                    .get(&("local".to_string(), region.clone()))
                    .unwrap_or(&ConnectivityStatus::Disconnected);
                (region, connectivity)
            })
            .collect();
        prioritized_targets.sort_by(|(_, a), (_, b)| {
            use ConnectivityStatus::*;
            match (a, b) {
                (Optimal, _) => std::cmp::Ordering::Less,
                (_, Optimal) => std::cmp::Ordering::Greater,
                (Degraded { latency_ms: a_lat }, Degraded { latency_ms: b_lat }) => {
                    a_lat.cmp(b_lat)
                }
                (Degraded { .. }, _) => std::cmp::Ordering::Less,
                (_, Degraded { .. }) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            }
        });
        let mut failed = Vec::new();
        for (target_region, _) in prioritized_targets {
            match self.send_data_to_region(data, target_region).await {
                Ok(_) => {
                    debug!(
                        "Successfully replicated data to region {} (selective sync)",
                        target_region
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to replicate data to region {} (selective sync): {}",
                        target_region, e
                    );
                    failed.push(format!("{target_region}: {e}"));
                }
            }
        }
        replication_outcome(&failed)
    }
    /// Eventual consistency replication with reconciliation
    ///
    /// Not implemented: there is no transport that delivers the package to
    /// the target regions. Earlier releases spawned tasks that only logged
    /// and reported success.
    async fn replicate_eventual_consistency(
        &self,
        data: &[u8],
        targets: &[String],
        reconciliation_interval_ms: u64,
    ) -> Result<()> {
        let timestamp = SystemTime::now();
        let vector_clock = self.generate_vector_clock().await?;
        let metadata = EventualConsistencyMetadata {
            timestamp,
            vector_clock,
            source_region: self.local_region.clone(),
            reconciliation_interval: Duration::from_millis(reconciliation_interval_ms),
        };
        let package = ReplicationPackage {
            data: data.to_vec(),
            metadata,
        };
        Err(anyhow::anyhow!(
            "eventual-consistency cross-region replication is not implemented: a package of \
             {} byte(s) from {} was not delivered to {}",
            package.data.len(),
            package.metadata.source_region,
            targets.join(", ")
        ))
    }
    /// Chain replication implementation
    async fn replicate_chain(&self, data: &[u8], targets: &[String]) -> Result<()> {
        let mut current_data = data.to_vec();
        for target_region in targets {
            match self.send_data_to_region(&current_data, target_region).await {
                Ok(response_data) => {
                    debug!(
                        "Successfully replicated data to region {} in chain",
                        target_region
                    );
                    current_data = response_data.unwrap_or(current_data);
                }
                Err(e) => {
                    warn!(
                        "Chain replication failed at region {}: {}",
                        target_region, e
                    );
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

/// `Ok` when every target region was reached, else an error naming the
/// regions that were not.
fn replication_outcome(failed: &[String]) -> Result<()> {
    if failed.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "cross-region replication failed for {} region(s): {}",
            failed.len(),
            failed.join("; ")
        ))
    }
}
