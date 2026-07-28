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
                // A closing `}` ends a SubSelect's solution-modifier list
                // (`{ SELECT … GROUP BY ?s }`): without this the GROUP BY /
                // ORDER BY condition loop reads past the brace and fails with
                // "Expected primary expression". A top-level query never reaches
                // a `}` here (its WHERE braces close before the modifiers), so
                // this only affects the nested-subquery case.
                | Some(Token::RightBrace)
                | Some(Token::Eof)
        )
    }
}
