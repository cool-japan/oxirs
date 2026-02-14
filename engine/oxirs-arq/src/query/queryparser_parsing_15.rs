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
    /// Parse DROP operation
    pub(super) fn parse_drop_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Drop)?;
        let silent = self.match_token(&Token::Silent);
        let target = self.parse_graph_ref_all()?;
        Ok(UpdateOperation::Drop { target, silent })
    }
}
