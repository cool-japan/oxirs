//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_graph_pattern(&mut self) -> Result<Algebra> {
        self.skip_whitespace_and_newlines();
        match self.peek() {
            Some(Token::Optional) => self.parse_optional_pattern(),
            Some(Token::Union) => self.parse_union_pattern(),
            Some(Token::Minus) => self.parse_minus_pattern(),
            Some(Token::Filter) => self.parse_filter_pattern(),
            Some(Token::Bind) => self.parse_bind_pattern(),
            Some(Token::Service) => self.parse_service_pattern(),
            Some(Token::Graph) => self.parse_graph_pattern_named(),
            Some(Token::Values) => self.parse_values_pattern(),
            Some(Token::LeftBrace) => {
                self.advance();
                let pattern = self.parse_group_graph_pattern()?;
                self.skip_whitespace_and_newlines();
                self.expect_token(Token::RightBrace)?;
                Ok(pattern)
            }
            _ => self.parse_basic_graph_pattern(),
        }
    }
    pub(super) fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    /// Skip whitespace and newlines more comprehensively
    pub(super) fn skip_whitespace_and_newlines(&mut self) {
        while let Some(token) = self.peek() {
            match token {
                Token::Newline => {
                    self.advance();
                }
                _ => break,
            }
        }
    }
    pub(super) fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.position += 1;
        }
        self.tokens.get(self.position - 1)
    }
}
