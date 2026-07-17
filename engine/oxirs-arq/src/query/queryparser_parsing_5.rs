//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{Query, Token};

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_ask_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Ask)?;
        self.parse_dataset_clause(&mut query.dataset)?;
        // The `WHERE` keyword is optional for ASK (`ASK { … }` is valid SPARQL);
        // only the group graph pattern braces are required.
        self.match_token(&Token::Where);
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(())
    }
}
