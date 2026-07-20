//! # QueryParser - resolve_prefixed_name_group Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;

use super::types::{DatatypeRef, Token};

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
    /// Resolve the datatype of an `"…"^^dt` literal to a `NamedNode`, honouring
    /// how it was written (see [`DatatypeRef`]):
    ///
    /// * [`DatatypeRef::Iri`] — the `^^<iri>` form — is an absolute IRI used
    ///   verbatim. This is what fixes `"5"^^<urn:myint>` / `"x"^^<tag:…>`: an
    ///   authority-less scheme is no longer mis-resolved as a prefix.
    /// * [`DatatypeRef::Prefixed`] — the `^^prefix:local` form — is resolved
    ///   against the declared prefixes. A colon-free name (which cannot arise
    ///   from a well-formed prefixed datatype) is used verbatim rather than
    ///   failing, matching the prior lenient behaviour.
    pub(super) fn resolve_datatype(&self, raw: &DatatypeRef) -> Result<NamedNode> {
        match raw {
            DatatypeRef::Iri(iri) => Ok(NamedNode::new_unchecked(iri.clone())),
            DatatypeRef::Prefixed(name) => {
                if let Some(colon) = name.find(':') {
                    let prefix = &name[..colon];
                    let local = &name[colon + 1..];
                    let full = self.resolve_prefixed_name(prefix, local)?;
                    Ok(NamedNode::new_unchecked(full))
                } else {
                    Ok(NamedNode::new_unchecked(name.clone()))
                }
            }
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
