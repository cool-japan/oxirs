//! # RoutingStrategy - Trait Implementations
//!
//! This module contains trait implementations for `RoutingStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RoutingStrategy;

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::LatencyAware
    }
}
