//! # ErrorRateStats - Trait Implementations
//!
//! This module contains trait implementations for `ErrorRateStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ErrorRateStats;

impl Default for ErrorRateStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            failed_operations: 0,
            error_rate: 0.0,
            last_error: None,
        }
    }
}
