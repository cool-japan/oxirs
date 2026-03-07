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
    /// Parse CREATE operation
    pub(super) fn parse_create_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Create)?;
        let silent = self.match_token(&Token::Silent);
        let graph = self.expect_iri()?;
        Ok(UpdateOperation::Create {
            graph: GraphReference::Iri(graph),
            silent,
        })
    }
}
