//! # QueryParser - predicates Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn is_solution_modifier_end(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Limit)
                | Some(Token::Offset)
                | Some(Token::OrderBy)
                | Some(Token::GroupBy)
                | Some(Token::Having)
                | Some(Token::Eof)
        )
    }
}
