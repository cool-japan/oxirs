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
    /// Parse ADD operation
    pub(super) fn parse_add_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Add)?;
        let silent = self.match_token(&Token::Silent);
        let from = self.parse_graph_ref()?;
        self.expect_token(Token::To)?;
        let to = self.parse_graph_ref()?;
        Ok(UpdateOperation::Add { from, to, silent })
    }
}
