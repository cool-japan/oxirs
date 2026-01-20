//! # PathCacheConfig - Trait Implementations
//!
//! This module contains trait implementations for `PathCacheConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl Default for PathCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 5000,
            max_cache_age: std::time::Duration::from_secs(3600),
            intelligent_eviction: true,
            min_access_threshold: 2,
            cache_negative_results: false,
        }
    }
}
