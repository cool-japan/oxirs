//! # PropertyPathEvaluator - accessors Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Get performance statistics for path evaluation
    pub fn get_performance_stats(&self) -> PropertyPathStats {
        PropertyPathStats {
            cache_entries: self.cache.len(),
            total_cached_results: self.cache.values().map(|v| v.values.len()).sum(),
        }
    }
}
