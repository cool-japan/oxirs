//! # RegionManager - schedule_reconciliation_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;
use tracing::debug;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Schedule reconciliation for eventual consistency
    pub(super) async fn schedule_reconciliation(&self, interval_ms: u64) {
        let interval = Duration::from_millis(interval_ms);
        tokio::spawn(async move {
            tokio::time::sleep(interval).await;
            debug!("Performing scheduled reconciliation");
        });
    }
}
