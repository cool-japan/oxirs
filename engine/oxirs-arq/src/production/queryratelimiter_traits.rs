//! # QueryRateLimiter - Trait Implementations
//!
//! This module contains trait implementations for `QueryRateLimiter`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::RwLock;

use super::types::QueryRateLimiter;

impl Default for QueryRateLimiter {
    fn default() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            requests_per_second: AtomicU32::new(100),
            burst_size: AtomicU32::new(200),
            enabled: AtomicBool::new(true),
        }
    }
}
