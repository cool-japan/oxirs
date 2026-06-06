//! # QueryEngineHealth - Trait Implementations
//!
//! This module contains trait implementations for `QueryEngineHealth`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types_monitor::QueryEngineHealth;

impl Default for QueryEngineHealth {
    fn default() -> Self {
        let health = Self::new();
        health.register_check("parser");
        health.register_check("executor");
        health.register_check("optimizer");
        health
    }
}
