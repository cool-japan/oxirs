//! # PerformanceStats - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PerformanceStats;

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            last_latency_ms: None,
            freshness_score: 1.0,
            reliability_score: 1.0,
        }
    }
}
