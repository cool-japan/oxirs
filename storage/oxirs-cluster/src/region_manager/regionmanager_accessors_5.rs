//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RegionTopology;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get current region topology
    pub async fn get_topology(&self) -> RegionTopology {
        self.topology.read().await.clone()
    }
}
