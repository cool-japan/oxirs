//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::update::{GraphReference, UpdateOperation};
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse LOAD operation
    pub(super) fn parse_load_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Load)?;
        let silent = self.match_token(&Token::Silent);
        let source = self.expect_iri()?;
        let graph = if matches!(self.peek(), Some(Token::Iri(_)))
            || matches!(self.peek(), Some(Token::PrefixedName(_, _)))
        {
            Some(GraphReference::Iri(self.expect_iri()?))
        } else {
            None
        };
        Ok(UpdateOperation::Load {
            source,
            graph,
            silent,
        })
    }
}
