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
        // Mirror the CONSTRUCT/SELECT/DESCRIBE hardening: the tokenizer emits
        // an explicit `Token::Newline` for every line break (and for a line
        // comment's terminating newline), and nothing upstream filters those
        // out of the stream — every lookahead below must skip them explicitly
        // or a query that wraps onto a new line anywhere between `ASK` and
        // the group's `{` (including a blank line, or a CRLF line ending,
        // whose `\r` the tokenizer already drops as plain whitespace) will
        // fail with a spurious "Expected X, found Newline" parse error.
        self.skip_whitespace_and_newlines();
        self.parse_dataset_clause(&mut query.dataset)?;
        self.skip_whitespace_and_newlines();
        // The `WHERE` keyword is optional for ASK (`ASK { … }` is valid SPARQL);
        // only the group graph pattern braces are required.
        self.match_token(&Token::Where);
        self.skip_whitespace_and_newlines();
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(())
    }
}
