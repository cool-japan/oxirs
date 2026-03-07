//! # ThroughputStats - Trait Implementations
//!
//! This module contains trait implementations for `ThroughputStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::SystemTime;

use super::types::ThroughputStats;

impl Default for ThroughputStats {
    fn default() -> Self {
        Self {
            operations_per_second: 0.0,
            bytes_per_second: 0.0,
            peak_ops_per_second: 0.0,
            last_measurement: SystemTime::UNIX_EPOCH,
        }
    }
}
