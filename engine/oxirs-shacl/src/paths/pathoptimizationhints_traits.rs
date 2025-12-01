//! # PathOptimizationHints - Trait Implementations
//!
//! This module contains trait implementations for `PathOptimizationHints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl Default for PathOptimizationHints {
    fn default() -> Self {
        Self {
            cache_simple_paths: true,
            cache_complex_paths: false,
            max_cache_size: 5000,
            parallel_threshold: 100,
            max_recursion_depth: 50,
            max_intermediate_results: 10000,
        }
    }
}
