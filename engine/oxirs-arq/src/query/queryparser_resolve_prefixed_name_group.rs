//! # QueryParser - resolve_prefixed_name_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{bail, Result};

use super::types::Token;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Resolve a prefixed name to its full IRI
    pub(super) fn resolve_prefixed_name(&self, prefix: &str, local: &str) -> Result<String> {
        if let Some(base) = self.prefixes.get(prefix) {
            Ok(format!("{base}{local}"))
        } else {
            bail!("Undefined prefix '{prefix}' in prefixed name '{prefix}:{local}'")
        }
    }
    pub(super) fn expect_iri(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(iri)
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                self.advance();
                self.resolve_prefixed_name(&prefix, &local)
            }
            _ => bail!("Expected IRI"),
        }
    }
}
