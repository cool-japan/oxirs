//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::PropertyPath;
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse property path expression
    pub(super) fn parse_property_path(&mut self) -> Result<PropertyPath> {
        self.parse_property_path_alternative()
    }
    /// Parse property path alternatives (highest precedence)
    pub(super) fn parse_property_path_alternative(&mut self) -> Result<PropertyPath> {
        let mut left = self.parse_property_path_sequence()?;
        while self.match_token(&Token::Pipe) {
            let right = self.parse_property_path_sequence()?;
            left = PropertyPath::alternative(left, right);
        }
        Ok(left)
    }
}
