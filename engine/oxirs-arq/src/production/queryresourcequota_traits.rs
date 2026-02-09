//! # QueryResourceQuota - Trait Implementations
//!
//! This module contains trait implementations for `QueryResourceQuota`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::QueryResourceQuota;

impl Default for QueryResourceQuota {
    fn default() -> Self {
        Self::new(1_000_000, Duration::from_secs(300), 1000)
    }
}
