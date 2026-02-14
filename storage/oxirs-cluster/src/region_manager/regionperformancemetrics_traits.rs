//! # RegionPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `RegionPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::SystemTime;

use super::types::RegionPerformanceMetrics;

impl Default for RegionPerformanceMetrics {
    fn default() -> Self {
        Self {
            inter_region_latencies: HashMap::new(),
            region_throughput: HashMap::new(),
            region_error_rates: HashMap::new(),
            last_updated: SystemTime::UNIX_EPOCH,
            monitoring_enabled: true,
        }
    }
}
