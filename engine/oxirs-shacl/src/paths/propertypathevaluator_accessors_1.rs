//! # PropertyPathEvaluator - accessors Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Get maximum recursion depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Set maximum recursion depth
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }
}
