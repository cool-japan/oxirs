//! # RegionManager - enable_monitoring_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Enable monitoring
    pub async fn enable_monitoring(&self) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.monitoring_enabled = true;
        tracing::info!("Multi-region monitoring enabled");
    }
}
