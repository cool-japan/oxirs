//! # QueryParser - match_multiplicative_operator_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::BinaryOperator;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn match_multiplicative_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Star) => {
                self.advance();
                Some(BinaryOperator::Multiply)
            }
            Some(Token::Divide) => {
                self.advance();
                Some(BinaryOperator::Divide)
            }
            _ => None,
        }
    }
}
