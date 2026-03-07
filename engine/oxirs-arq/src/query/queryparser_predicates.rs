//! # QueryParser - predicates Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Check if the current pattern contains UNION by looking ahead
    pub(super) fn has_union_pattern(&self) -> bool {
        let mut pos = self.position;
        let mut brace_depth = 0;
        while pos < self.tokens.len() {
            match &self.tokens[pos] {
                Token::LeftBrace => brace_depth += 1,
                Token::RightBrace => {
                    if brace_depth == 0 {
                        break;
                    }
                    brace_depth -= 1;
                }
                Token::Union if brace_depth == 0 => return true,
                _ => {}
            }
            pos += 1;
        }
        false
    }
}
