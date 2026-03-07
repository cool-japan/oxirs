//! # QueryCancellationToken - Trait Implementations
//!
//! This module contains trait implementations for `QueryCancellationToken`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QueryCancellationToken;

impl Default for QueryCancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for QueryCancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCancellationToken")
            .field("cancelled", &self.is_cancelled())
            .field("reason", &self.get_reason())
            .finish()
    }
}
