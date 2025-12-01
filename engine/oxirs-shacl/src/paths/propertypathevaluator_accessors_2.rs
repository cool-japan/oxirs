//! # PropertyPathEvaluator - accessors Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Get maximum intermediate results limit
    pub fn max_intermediate_results(&self) -> usize {
        self.max_intermediate_results
    }

    /// Set maximum intermediate results limit
    pub fn set_max_intermediate_results(&mut self, max_results: usize) {
        self.max_intermediate_results = max_results;
    }
}
