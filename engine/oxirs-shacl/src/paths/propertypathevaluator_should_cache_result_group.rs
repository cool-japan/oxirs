//! # PropertyPathEvaluator - should_cache_result_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Check if a result should be cached
    pub(super) fn should_cache_result(&self, result: &[Term]) -> bool {
        if result.len() > 10000 {
            return false;
        }
        if result.is_empty() && !self.cache_config.cache_negative_results {
            return false;
        }
        true
    }
}
