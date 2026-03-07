//! # EnhancedReteStats - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedReteStats`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::EnhancedReteStats;

impl fmt::Display for EnhancedReteStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} | Beta joins: {}/{} ({:.1}% success), Evictions: {}, Peak mem: {}, Enhanced: {}",
            self.basic,
            self.successful_beta_joins,
            self.total_beta_joins,
            self.join_success_rate * 100.0,
            self.memory_evictions,
            self.peak_memory_usage,
            self.enhanced_nodes
        )
    }
}
