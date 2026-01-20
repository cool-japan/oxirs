//! # ServiceRegistry - stop_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::Result;
use tracing::info;

impl ServiceRegistry {
    /// Stop the service registry
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping service registry");
        if let Some(handle) = self.health_monitor_handle.take() {
            handle.abort();
        }
        Ok(())
    }
}
