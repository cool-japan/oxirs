//! # Store - Trait Implementations
//!
//! This module contains trait implementations for `Store`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("default_store", &"CoreStore")
            .field("datasets", &"<datasets>")
            .field("query_engine", &"QueryEngine")
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
