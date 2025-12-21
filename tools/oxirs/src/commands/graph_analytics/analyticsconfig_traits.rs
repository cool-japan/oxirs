//! # AnalyticsConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalyticsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AnalyticsConfig, AnalyticsOperation};

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            operation: AnalyticsOperation::PageRank,
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            source_node: None,
            target_node: None,
            top_k: 20,
            katz_alpha: 0.1,
            katz_beta: 1.0,
            k_core_value: None,
            enable_simd: true,
            enable_parallel: true,
            enable_gpu: false,
            enable_cache: true,
            export_path: None,
            enable_benchmarking: false,
        }
    }
}
