//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::rifparser_type::RifParser;

impl RifParser {
    /// Parse a name
    pub(super) fn parse_name(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self
                .input
                .chars()
                .nth(self.pos)
                .expect("position is within input bounds");
            if c.is_alphanumeric() || c == '_' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(anyhow!("Expected name at position {}", self.pos));
        }
        Ok(self.input[start..self.pos].to_string())
    }
    /// Parse a prefixed name
    pub(super) fn parse_prefixed_name(&mut self) -> Result<String> {
        let first = self.parse_name()?;
        if self.try_consume(":") {
            let local = self.parse_name().unwrap_or_default();
            if let Some(base) = self.prefixes.get(&first) {
                Ok(format!("{}{}", base, local))
            } else {
                Ok(format!("{}:{}", first, local))
            }
        } else {
            Ok(first)
        }
    }
}
