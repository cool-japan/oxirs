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
    /// Parse property path sequences
    pub(super) fn parse_property_path_sequence(&mut self) -> Result<PropertyPath> {
        let mut left = self.parse_property_path_postfix()?;
        // The lexer emits `/` as `Token::Divide` (its arithmetic spelling); in a
        // property-path position it is the sequence operator, never division, so
        // accept it here. (`Token::Slash` is retained for safety though the
        // current lexer never produces it.)
        while matches!(self.peek(), Some(Token::Slash) | Some(Token::Divide)) {
            self.advance();
            let right = self.parse_property_path_postfix()?;
            left = PropertyPath::sequence(left, right);
        }
        Ok(left)
    }
    /// Parse property path with postfix operators (*, +, ?)
    pub(super) fn parse_property_path_postfix(&mut self) -> Result<PropertyPath> {
        let mut path = self.parse_property_path_primary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    path = PropertyPath::zero_or_more(path);
                }
                Some(Token::Plus) => {
                    self.advance();
                    path = PropertyPath::one_or_more(path);
                }
                Some(Token::Question) => {
                    self.advance();
                    path = PropertyPath::zero_or_one(path);
                }
                _ => break,
            }
        }
        Ok(path)
    }
}
