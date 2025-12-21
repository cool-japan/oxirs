//! # PropertyPathEvaluator - accessors Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> PathCacheStats {
        PathCacheStats {
            entries: self.cache.len(),
            total_values: self.cache.values().map(|v| v.values.len()).sum(),
        }
    }
}
