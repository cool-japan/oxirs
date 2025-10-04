//! Cache tier implementation

use super::types::AccessTracker;
use super::eviction::EvictionPolicy;
use super::storage::CacheStorage;
use super::types::{TierConfiguration, TierStatistics};

/// Individual cache tier with specific characteristics
#[derive(Debug)]
pub struct CacheTier {
    /// Tier identifier
    #[allow(dead_code)]
    pub(crate) tier_id: u32,
    /// Storage implementation
    pub(crate) storage: Box<dyn CacheStorage>,
    /// Eviction policy
    pub(crate) eviction_policy: Box<dyn EvictionPolicy>,
    /// Access frequency tracker
    pub(crate) access_tracker: AccessTracker,
    /// Tier-specific configuration
    pub(crate) config: TierConfiguration,
    /// Performance statistics
    pub(crate) stats: TierStatistics,
}
