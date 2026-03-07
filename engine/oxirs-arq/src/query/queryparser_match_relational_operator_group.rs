//! # QueryParser - match_relational_operator_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::BinaryOperator;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn match_relational_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Less) => {
                self.advance();
                Some(BinaryOperator::Less)
            }
            Some(Token::LessEqual) => {
                self.advance();
                Some(BinaryOperator::LessEqual)
            }
            Some(Token::Greater) => {
                self.advance();
                Some(BinaryOperator::Greater)
            }
            Some(Token::GreaterEqual) => {
                self.advance();
                Some(BinaryOperator::GreaterEqual)
            }
            _ => None,
        }
    }
}
