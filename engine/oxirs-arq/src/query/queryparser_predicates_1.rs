//! # QueryParser - predicates Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Check if current position starts a property path
    pub(super) fn is_property_path_start(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Caret)
                | Some(Token::Iri(_))
                | Some(Token::PrefixedName(_, _))
                | Some(Token::LeftParen)
                | Some(Token::Bang)
        )
    }
}
