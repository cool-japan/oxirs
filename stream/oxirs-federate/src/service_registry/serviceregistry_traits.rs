//! # ServiceRegistry - Trait Implementations
//!
//! This module contains trait implementations for `ServiceRegistry`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
