//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use tokio::sync::RwLock;

use super::types::{
    ConsensusStrategy, MultiRegionReplicationStrategy, RegionPerformanceMetrics, RegionTopology,
};

/// Multi-region cluster manager
#[derive(Debug)]
pub struct RegionManager {
    /// Current topology configuration
    pub(super) topology: RwLock<RegionTopology>,
    /// Local node's region information
    pub(super) local_region: String,
    pub(super) local_availability_zone: String,
    /// Consensus strategy configuration
    #[allow(dead_code)]
    pub(super) consensus_strategy: ConsensusStrategy,
    /// Replication strategy configuration
    pub(super) replication_strategy: MultiRegionReplicationStrategy,
    /// Performance monitoring data
    pub(super) performance_metrics: RwLock<RegionPerformanceMetrics>,
}
