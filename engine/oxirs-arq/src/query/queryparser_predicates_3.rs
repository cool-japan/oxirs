//! # QueryParser - predicates Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn is_pattern_end(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::RightBrace)
                // A `{` starts a new graph-pattern element (nested group,
                // `GRAPH`, subquery, …); a triple block can never continue past
                // it, so end the BGP here even after a separating `.` — e.g.
                // `{ ?s ?p ?o . { FILTER(?o > 5) } }`.
                | Some(Token::LeftBrace)
                | Some(Token::Optional)
                | Some(Token::Union)
                | Some(Token::Minus)
                | Some(Token::Filter)
                | Some(Token::Bind)
                | Some(Token::Service)
                | Some(Token::Graph)
                | Some(Token::Values)
                | Some(Token::Eof)
        )
    }
}
