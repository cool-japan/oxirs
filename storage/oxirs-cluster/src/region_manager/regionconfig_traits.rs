//! # RegionConfig - Trait Implementations
//!
//! This module contains trait implementations for `RegionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Hash`
//! - `Eq`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{RegionConfig, RoutingStrategy};

impl std::hash::Hash for RegionConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.local_replication_factor.hash(state);
        self.cross_region_replication_factor.hash(state);
        self.max_regional_latency_ms.hash(state);
        self.prefer_local_leader.hash(state);
        self.enable_cross_region_backup.hash(state);
        let mut sorted_props: Vec<_> = self.properties.iter().collect();
        sorted_props.sort_by_key(|(k, _)| *k);
        sorted_props.hash(state);
    }
}

impl Eq for RegionConfig {}

impl Default for RegionConfig {
    fn default() -> Self {
        Self {
            local_replication_factor: 3,
            cross_region_replication_factor: 1,
            max_regional_latency_ms: 100,
            prefer_local_leader: true,
            enable_cross_region_backup: true,
            enable_relay: true,
            relay_latency_threshold_ms: 200.0,
            enable_compression: true,
            enable_read_local: true,
            routing_strategy: RoutingStrategy::default(),
            properties: HashMap::new(),
        }
    }
}
