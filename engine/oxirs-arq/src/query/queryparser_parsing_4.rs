//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;
use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;

use super::types::{DatasetClause, Query, Token};

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_construct_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Construct)?;
        // The tokenizer emits a `Token::Newline` for every line break (and for
        // a line comment's terminating newline), and nothing upstream filters
        // those out of the stream — every lookahead in this function must skip
        // them explicitly or a query that wraps onto a new line anywhere
        // between `CONSTRUCT` and the WHERE group's `{` (including a blank
        // line, or a CRLF line ending, whose `\r` the tokenizer already drops
        // as plain whitespace) will fail with a spurious "Expected X, found
        // Newline" parse error. See `parse_group_graph_pattern`, which already
        // does this at each of its own lookaheads.
        self.skip_whitespace_and_newlines();
        // Two forms (SPARQL 1.1 §16.2):
        //   * explicit template — `CONSTRUCT { tmpl } WHERE { … }`
        //   * shorthand         — `CONSTRUCT WHERE { BGP }`, where the WHERE
        //     block IS the template. The shorthand only permits a basic graph
        //     pattern in WHERE.
        let explicit_template = self.match_token(&Token::LeftBrace);
        if explicit_template {
            query.construct_template = self.parse_construct_template()?;
            self.skip_whitespace_and_newlines();
            self.expect_token(Token::RightBrace)?;
        }
        self.skip_whitespace_and_newlines();
        self.parse_dataset_clause(&mut query.dataset)?;
        self.skip_whitespace_and_newlines();
        // The explicit-template form (`CONSTRUCT { tmpl } [WHERE] { … }`) may omit
        // the `WHERE` keyword in SPARQL 1.1. The shorthand form
        // (`CONSTRUCT WHERE { BGP }`) must keep it: there the WHERE block *is* the
        // template, so the keyword is grammatically mandatory and dropping it
        // would make `CONSTRUCT { … }` ambiguous with the explicit form.
        if explicit_template {
            self.match_token(&Token::Where);
        } else {
            self.expect_token(Token::Where)?;
        }
        self.skip_whitespace_and_newlines();
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        self.parse_solution_modifiers(query)?;
        if !explicit_template {
            // Shorthand: the template is the WHERE block's BGP. Anything other
            // than a plain BGP (FILTER, OPTIONAL, GRAPH, UNION, BIND, …) is a
            // syntax error for this form — fail loud rather than silently drop.
            match &query.where_clause {
                Algebra::Bgp(triples) => {
                    query.construct_template = triples.clone();
                }
                // An empty WHERE group parses to `Table`; the template is then
                // simply empty (an empty BGP), which is well-formed.
                Algebra::Table => {}
                other => bail!(
                    "CONSTRUCT WHERE shorthand permits only a basic graph pattern in the \
                     WHERE clause (no FILTER, OPTIONAL, GRAPH, UNION, BIND, MINUS, …); \
                     found: {other:?}"
                ),
            }
        }
        Ok(())
    }
    pub(super) fn parse_dataset_clause(&mut self, dataset: &mut DatasetClause) -> Result<()> {
        loop {
            // Skip a line break before each `FROM` lookahead, both ahead of the
            // first clause and between repeated ones (`FROM <a>\nFROM NAMED
            // <b>`), so a dataset clause spread across lines still parses.
            self.skip_whitespace_and_newlines();
            if !self.match_token(&Token::From) {
                break;
            }
            self.skip_whitespace_and_newlines();
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
