//! # QueryParser - resolve_prefixed_name_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;

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
    /// Resolve the raw datatype of an `"…"^^dt` literal to a `NamedNode`.
    ///
    /// `raw` is the datatype exactly as written after `^^`: either an absolute
    /// IRI (the `<iri>` form, already unwrapped by the lexer) or a
    /// `prefix:local` name. An absolute IRI is recognised by an authority
    /// separator (`scheme://…`); anything else with a colon is treated as a
    /// prefixed name and resolved against the declared prefixes.
    pub(super) fn resolve_datatype(&self, raw: &str) -> Result<NamedNode> {
        if let Some(colon) = raw.find(':') {
            if raw[colon + 1..].starts_with("//") {
                return Ok(NamedNode::new_unchecked(raw.to_string()));
            }
            let prefix = &raw[..colon];
            let local = &raw[colon + 1..];
            let full = self.resolve_prefixed_name(prefix, local)?;
            Ok(NamedNode::new_unchecked(full))
        } else {
            Ok(NamedNode::new_unchecked(raw.to_string()))
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
