//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Term;
use crate::update::{GraphReference, QuadPattern};
use anyhow::Result;

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse a variable or term (used for quad parsing)
    pub(super) fn parse_var_or_term(&mut self) -> Result<Term> {
        self.parse_term()
    }
    /// Parse a single quad
    pub(super) fn parse_quad(&mut self) -> Result<QuadPattern> {
        let subject = self.parse_var_or_term()?;
        let predicate = self.parse_var_or_term()?;
        let object = self.parse_var_or_term()?;
        let graph = if matches!(
            self.peek(),
            Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _))
        ) && !matches!(self.peek(), Some(Token::Dot) | Some(Token::RightBrace))
        {
            let iri = self.expect_iri()?;
            Some(GraphReference::Iri(iri))
        } else {
            None
        };
        Ok(QuadPattern {
            subject,
            predicate,
            object,
            graph,
        })
    }
}
