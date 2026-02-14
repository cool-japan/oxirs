//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;

use super::types::{DatasetClause, Query, Token};

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_construct_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Construct)?;
        if self.match_token(&Token::LeftBrace) {
            query.construct_template = self.parse_construct_template()?;
            self.expect_token(Token::RightBrace)?;
        }
        self.parse_dataset_clause(&mut query.dataset)?;
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        self.parse_solution_modifiers(query)?;
        Ok(())
    }
    pub(super) fn parse_dataset_clause(&mut self, dataset: &mut DatasetClause) -> Result<()> {
        while self.match_token(&Token::From) {
            if self.match_token(&Token::Named) {
                let iri = self.expect_iri()?;
                dataset.named_graphs.push(NamedNode::new_unchecked(iri));
            } else {
                let iri = self.expect_iri()?;
                dataset.default_graphs.push(NamedNode::new_unchecked(iri));
            }
        }
        Ok(())
    }
    pub(super) fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }
    pub(super) fn expect_token(&mut self, token: Token) -> Result<()> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            bail!("Expected {token:?}, found {:?}", self.peek())
        }
    }
}
