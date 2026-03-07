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
    /// Parse DELETE operation (DELETE DATA or DELETE WHERE)
    pub(super) fn parse_delete_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Delete)?;
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
        self.expect_token(Token::LeftBrace)?;
        let quads = self.parse_quad_data()?;
        self.expect_token(Token::RightBrace)?;
        Ok(UpdateOperation::DeleteData { data: quads })
    }
}
