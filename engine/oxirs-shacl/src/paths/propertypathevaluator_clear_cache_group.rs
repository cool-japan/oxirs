//! # PropertyPathEvaluator - clear_cache_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Clear the evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.query_plan_cache.clear();
    }
}
