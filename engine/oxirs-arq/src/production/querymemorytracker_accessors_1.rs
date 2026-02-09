//! # QueryMemoryTracker - accessors Methods
//!
//! This module contains method implementations for `QueryMemoryTracker`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::querymemorytracker_type::QueryMemoryTracker;

impl QueryMemoryTracker {
    pub fn set_pressure_threshold(&self, threshold: f64) {
        *self.pressure_threshold.write().expect("lock poisoned") = threshold.clamp(0.0, 1.0);
    }
}
