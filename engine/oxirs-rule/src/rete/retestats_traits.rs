//! # ReteStats - Trait Implementations
//!
//! This module contains trait implementations for `ReteStats`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::ReteStats;

impl fmt::Display for ReteStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Nodes: {} (α:{}, β:{}, P:{}), Tokens: {}",
            self.total_nodes,
            self.alpha_nodes,
            self.beta_nodes,
            self.production_nodes,
            self.total_tokens
        )
    }
}
