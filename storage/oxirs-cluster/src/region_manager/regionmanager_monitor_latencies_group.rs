//! # RegionManager - monitor_latencies_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::time::SystemTime;

use super::types::LatencyStats;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Monitor and update inter-region latencies
    pub async fn monitor_latencies(&self) -> Result<()> {
        let topology = self.topology.read().await;
        let mut metrics = self.performance_metrics.write().await;
        for (region_a, region_b) in topology.latency_matrix.keys() {
            if region_a != region_b {
                let latency = self
                    .measure_inter_region_latency(region_a, region_b)
                    .await?;
                let stats = metrics
                    .inter_region_latencies
                    .entry((region_a.clone(), region_b.clone()))
                    .or_default();
                self.update_latency_stats(stats, latency);
            }
        }
        metrics.last_updated = SystemTime::now();
        Ok(())
    }
    /// Update latency statistics
    fn update_latency_stats(&self, stats: &mut LatencyStats, new_latency: f64) {
        if stats.sample_count == 0 {
            stats.min_ms = new_latency;
            stats.max_ms = new_latency;
            stats.avg_ms = new_latency;
            stats.p95_ms = new_latency;
            stats.p99_ms = new_latency;
        } else {
            stats.min_ms = stats.min_ms.min(new_latency);
            stats.max_ms = stats.max_ms.max(new_latency);
            stats.avg_ms = (stats.avg_ms * stats.sample_count as f64 + new_latency)
                / (stats.sample_count as f64 + 1.0);
            if new_latency > stats.p95_ms {
                stats.p95_ms = (stats.p95_ms * 19.0 + new_latency) / 20.0;
            }
            if new_latency > stats.p99_ms {
                stats.p99_ms = (stats.p99_ms * 99.0 + new_latency) / 100.0;
            }
        }
        stats.sample_count += 1;
    }
}
