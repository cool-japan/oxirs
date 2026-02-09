//! # CostEstimatorConfig - Trait Implementations
//!
//! This module contains trait implementations for `CostEstimatorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CostEstimatorConfig;

impl Default for CostEstimatorConfig {
    fn default() -> Self {
        Self {
            pattern_weight: 10.0,
            join_weight: 50.0,
            filter_weight: 20.0,
            aggregate_weight: 30.0,
            path_weight: 100.0,
            enable_ml_prediction: false,
        }
    }
}
