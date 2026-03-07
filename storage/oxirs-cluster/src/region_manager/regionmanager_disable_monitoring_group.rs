//! # RegionManager - disable_monitoring_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Disable monitoring
    pub async fn disable_monitoring(&self) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.monitoring_enabled = false;
        tracing::info!("Multi-region monitoring disabled");
    }
}
