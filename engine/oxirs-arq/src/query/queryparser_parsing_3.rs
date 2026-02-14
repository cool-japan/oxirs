//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{GroupCondition, OrderCondition};
use anyhow::Result;

use super::types::{Query, Token};

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse a SPARQL query string into a Query AST
    pub fn parse(&mut self, query_str: &str) -> Result<Query> {
        self.tokenize(query_str)?;
        self.parse_query()
    }
    pub(super) fn parse_solution_modifiers(&mut self, query: &mut Query) -> Result<()> {
        if self.match_token(&Token::GroupBy) {
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                let expr = self.parse_expression()?;
                let alias = if self.match_token(&Token::As) {
                    Some(self.expect_variable()?)
                } else {
                    None
                };
                query.group_by.push(GroupCondition { expr, alias });
            }
        }
        if self.match_token(&Token::Having) {
            query.having = Some(self.parse_expression()?);
        }
        if self.match_token(&Token::OrderBy) {
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                let ascending = if self.match_token(&Token::Desc) {
                    false
                } else {
                    self.match_token(&Token::Asc);
                    true
                };
                let expr = self.parse_expression()?;
                query.order_by.push(OrderCondition { expr, ascending });
            }
        }
        if self.match_token(&Token::Limit) {
            if let Some(Token::NumericLiteral(num)) = self.peek() {
                query.limit = num.parse().ok();
                self.advance();
            }
        }
        if self.match_token(&Token::Offset) {
            if let Some(Token::NumericLiteral(num)) = self.peek() {
                query.offset = num.parse().ok();
                self.advance();
            }
        }
        Ok(())
    }
}
