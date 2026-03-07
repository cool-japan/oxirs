//! # RifDocument - Trait Implementations
//!
//! This module contains trait implementations for `RifDocument`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{RifDialect, RifDocument};

impl Default for RifDocument {
    fn default() -> Self {
        Self::new(RifDialect::default())
    }
}
