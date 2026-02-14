//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::update::{GraphReference, GraphTarget, UpdateOperation};
use anyhow::{bail, Result};

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse COPY operation
    pub(super) fn parse_copy_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Copy)?;
        let silent = self.match_token(&Token::Silent);
        let from = self.parse_graph_ref()?;
        self.expect_token(Token::To)?;
        let to = self.parse_graph_ref()?;
        Ok(UpdateOperation::Copy { from, to, silent })
    }
    /// Parse graph reference (DEFAULT, NAMED, or IRI)
    pub(super) fn parse_graph_ref(&mut self) -> Result<GraphTarget> {
        match self.peek() {
            Some(Token::Default) => {
                self.advance();
                Ok(GraphTarget::Default)
            }
            Some(Token::Named) => {
                self.advance();
                Ok(GraphTarget::Named)
            }
            Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _)) => {
                let iri = self.expect_iri()?;
                Ok(GraphTarget::Graph(GraphReference::Iri(iri)))
            }
            _ => bail!("Expected graph reference"),
        }
    }
}
