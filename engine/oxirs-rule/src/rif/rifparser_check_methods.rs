//! # RifParser - check_methods Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::rifparser_type::RifParser;

impl RifParser {
    pub(super) fn check_eof(&self) -> bool {
        self.pos >= self.input.len()
    }
}
