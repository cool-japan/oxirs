//! # BaselineTrackerConfig - Trait Implementations
//!
//! This module contains trait implementations for `BaselineTrackerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_monitor::BaselineTrackerConfig;

impl Default for BaselineTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            regression_threshold: 0.2,
            min_samples: 10,
            auto_update_baseline: true,
        }
    }
}
