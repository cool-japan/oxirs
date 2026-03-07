//! # QueryParser - match_equality_operator_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::BinaryOperator;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Equal) => {
                self.advance();
                Some(BinaryOperator::Equal)
            }
            Some(Token::NotEqual) => {
                self.advance();
                Some(BinaryOperator::NotEqual)
            }
            _ => None,
        }
    }
}
