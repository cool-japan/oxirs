//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::update::UpdateOperation;
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse DELETE operation (DELETE DATA, DELETE WHERE, or DELETE/INSERT/WHERE)
    pub(super) fn parse_delete_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Delete)?;
        // A newline may separate `DELETE` from `DATA`/`WHERE`/the template's
        // opening `{` — none of `match_token`/`peek`/`check` skip the
        // tokenizer's explicit `Token::Newline`, so without this a
        // `DELETE\nWHERE { … }` shorthand would misdispatch into
        // `parse_delete_insert_where` (the `Where` token hidden behind the
        // newline) instead of `parse_delete_where`.
        self.skip_whitespace_and_newlines();
        if self.match_token(&Token::Data) {
            self.parse_delete_data()
        } else if self.peek() == Some(&Token::Where) {
            self.parse_delete_where()
        } else {
            self.parse_delete_insert_where()
        }
    }
    /// Parse DELETE DATA operation
    pub(super) fn parse_delete_data(&mut self) -> Result<UpdateOperation> {
        // `DELETE DATA\n{ … }`: a newline between `DATA` and the block's `{`.
        self.skip_whitespace_and_newlines();
        self.expect_token(Token::LeftBrace)?;
        let quads = self.parse_quad_data()?;
        self.expect_token(Token::RightBrace)?;
        Ok(UpdateOperation::DeleteData { data: quads })
    }
}
