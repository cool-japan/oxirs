//! # VectorClock - Trait Implementations
//!
//! This module contains trait implementations for `VectorClock`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::VectorClock;

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}
