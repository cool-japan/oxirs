//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::update::{GraphTarget, UpdateOperation};
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse CLEAR operation
    pub(super) fn parse_clear_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Clear)?;
        let silent = self.match_token(&Token::Silent);
        let target = self.parse_graph_ref_all()?;
        Ok(UpdateOperation::Clear { target, silent })
    }
    /// Parse graph reference with ALL option
    pub(super) fn parse_graph_ref_all(&mut self) -> Result<GraphTarget> {
        match self.peek() {
            Some(Token::All) => {
                self.advance();
                Ok(GraphTarget::All)
            }
            _ => self.parse_graph_ref(),
        }
    }
}
