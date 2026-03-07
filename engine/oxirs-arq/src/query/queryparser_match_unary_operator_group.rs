//! # QueryParser - match_unary_operator_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::UnaryOperator;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        match self.peek() {
            Some(Token::Not) => {
                self.advance();
                Some(UnaryOperator::Not)
            }
            Some(Token::Plus) => {
                self.advance();
                Some(UnaryOperator::Plus)
            }
            Some(Token::Minus_) => {
                self.advance();
                Some(UnaryOperator::Minus)
            }
            _ => None,
        }
    }
}
