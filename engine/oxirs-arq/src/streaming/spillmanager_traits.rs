//! # SpillManager - Trait Implementations
//!
//! This module contains trait implementations for `SpillManager`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SpillManager;

impl Clone for SpillManager {
    fn clone(&self) -> Self {
        Self {
            spill_directory: self.spill_directory.clone(),
            active_spills: self.active_spills.clone(),
            spill_counter: self.spill_counter,
            compression_enabled: self.compression_enabled,
            compression_level: self.compression_level,
        }
    }
}
