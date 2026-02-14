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
    /// Parse INSERT operation (INSERT DATA or INSERT WHERE)
    pub(super) fn parse_insert_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Insert)?;
        if self.match_token(&Token::Data) {
            self.parse_insert_data()
        } else {
            self.parse_insert_where()
        }
    }
    /// Parse INSERT DATA operation
    pub(super) fn parse_insert_data(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let quads = self.parse_quad_data()?;
        self.expect_token(Token::RightBrace)?;
        Ok(UpdateOperation::InsertData { data: quads })
    }
}
