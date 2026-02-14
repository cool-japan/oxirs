//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::UpdateRequest;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse a SPARQL UPDATE request string into an UpdateRequest AST
    pub fn parse_update(&mut self, update_str: &str) -> Result<UpdateRequest> {
        self.tokenize(update_str)?;
        self.parse_update_request()
    }
}
