//! # RifConverter - Trait Implementations
//!
//! This module contains trait implementations for `RifConverter`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::RifConverter;

impl Default for RifConverter {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}
