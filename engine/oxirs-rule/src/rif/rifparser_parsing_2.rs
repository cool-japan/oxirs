//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::types::{RifDocument, RifImport};

use super::rifparser_type::RifParser;

impl RifParser {
    /// Parse Document wrapper
    pub(super) fn parse_document_wrapper(&mut self, doc: &mut RifDocument) -> Result<()> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        while !self.try_consume(")") {
            self.skip_ws();
            if self.try_consume("Base") {
                doc.base = Some(self.parse_iri()?);
            } else if self.try_consume("Prefix") {
                let (prefix, iri) = self.parse_prefix_decl()?;
                doc.add_prefix(&prefix, &iri);
                self.prefixes.insert(prefix, iri);
            } else if self.try_consume("Import") {
                doc.imports.push(self.parse_import()?);
            } else if self.try_consume("Group") {
                doc.add_group(self.parse_group()?);
            } else {
                break;
            }
            self.skip_ws();
        }
        Ok(())
    }
    /// Parse prefix declaration: Prefix(prefix <iri>)
    pub(super) fn parse_prefix_decl(&mut self) -> Result<(String, String)> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let prefix = self.parse_name()?;
        self.skip_ws();
        let iri = self.parse_iri()?;
        self.skip_ws();
        self.expect(")")?;
        Ok((prefix, iri))
    }
    /// Parse import directive
    pub(super) fn parse_import(&mut self) -> Result<RifImport> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let location = self.parse_iri()?;
        self.skip_ws();
        let profile = if !self.check(")") {
            Some(self.parse_iri()?)
        } else {
            None
        };
        self.skip_ws();
        self.expect(")")?;
        Ok(RifImport { location, profile })
    }
    /// Parse an IRI: <...>
    pub(super) fn parse_iri(&mut self) -> Result<String> {
        self.skip_ws();
        self.expect("<")?;
        let start = self.pos;
        while self.pos < self.input.len() && !self.check(">") {
            self.pos += 1;
        }
        let iri = self.input[start..self.pos].to_string();
        self.expect(">")?;
        Ok(iri)
    }
    pub(super) fn skip_ws(&mut self) {
        while self.pos < self.input.len() {
            let c = self
                .input
                .chars()
                .nth(self.pos)
                .expect("position is within input bounds");
            if c.is_whitespace() {
                self.pos += 1;
            } else if c == '(' && self.input[self.pos..].starts_with("(*") {
                self.pos += 2;
                while self.pos < self.input.len() - 1 {
                    if self.input[self.pos..].starts_with("*)") {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }
    pub(super) fn try_consume(&mut self, s: &str) -> bool {
        if self.check(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }
    pub(super) fn expect(&mut self, s: &str) -> Result<()> {
        if self.try_consume(s) {
            Ok(())
        } else {
            Err(anyhow!(
                "Expected '{}' at position {}, found '{}'",
                s,
                self.pos,
                &self.input[self.pos..self.pos.saturating_add(10).min(self.input.len())]
            ))
        }
    }
}
