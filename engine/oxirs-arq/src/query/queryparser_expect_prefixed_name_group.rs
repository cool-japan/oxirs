//! # QueryParser - expect_prefixed_name_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{bail, Result};

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn expect_prefixed_name(&mut self) -> Result<(String, String)> {
        if let Some(Token::PrefixedName(prefix, local)) = self.peek() {
            let result = (prefix.clone(), local.clone());
            self.advance();
            Ok(result)
        } else {
            bail!("Expected prefixed name")
        }
    }
}
