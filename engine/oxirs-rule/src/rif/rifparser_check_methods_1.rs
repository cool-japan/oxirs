//! # RifParser - check_methods Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::rifparser_type::RifParser;

impl RifParser {
    pub(super) fn check_alpha(&self) -> bool {
        self.pos < self.input.len()
            && self
                .input
                .chars()
                .nth(self.pos)
                .map(|c| c.is_alphabetic())
                .unwrap_or(false)
    }
}
