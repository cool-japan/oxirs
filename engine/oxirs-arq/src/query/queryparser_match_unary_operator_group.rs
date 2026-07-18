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
            // Both the `NOT` keyword and the `!` operator are logical negation
            // (effective-boolean-value negation). `!` is the usual SPARQL
            // spelling: `!BOUND(?x)`, `!EXISTS { … }`, `!(?a = ?b)`.
            Some(Token::Not) | Some(Token::Bang) => {
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
