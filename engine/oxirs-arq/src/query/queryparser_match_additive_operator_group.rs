//! # QueryParser - match_additive_operator_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::BinaryOperator;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn match_additive_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Plus) => {
                self.advance();
                Some(BinaryOperator::Add)
            }
            Some(Token::Minus_) => {
                self.advance();
                Some(BinaryOperator::Subtract)
            }
            _ => None,
        }
    }
}
