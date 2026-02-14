//! # RegionManager - predicates Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Check if monitoring is currently active
    pub async fn is_monitoring_active(&self) -> bool {
        let metrics = self.performance_metrics.read().await;
        metrics.monitoring_enabled
    }
}
