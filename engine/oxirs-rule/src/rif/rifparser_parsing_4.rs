//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::rifparser_type::RifParser;

impl RifParser {
    /// Parse a number
    pub(super) fn parse_number(&mut self) -> Result<String> {
        let start = self.pos;
        if self.check("-") || self.check("+") {
            self.pos += 1;
        }
        while self.pos < self.input.len() && self.check_digit() {
            self.pos += 1;
        }
        if self.check(".") {
            self.pos += 1;
            while self.pos < self.input.len() && self.check_digit() {
                self.pos += 1;
            }
        }
        Ok(self.input[start..self.pos].to_string())
    }
    pub(super) fn check(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }
    pub(super) fn check_digit(&self) -> bool {
        self.pos < self.input.len()
            && self
                .input
                .chars()
                .nth(self.pos)
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
    }
}
