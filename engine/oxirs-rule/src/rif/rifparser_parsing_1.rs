//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{RifFrame, RifTerm};

use super::rifparser_type::RifParser;

impl RifParser {
    /// Parse a list
    pub(super) fn parse_list(&mut self) -> Result<RifTerm> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let mut items = Vec::new();
        while !self.check(")") {
            items.push(self.parse_term()?);
            self.skip_ws();
        }
        self.expect(")")?;
        Ok(RifTerm::List(items))
    }
    /// Parse a frame: obj[slot->value, ...]
    pub(super) fn parse_frame(&mut self) -> Result<RifFrame> {
        let object = self.parse_term()?;
        self.skip_ws();
        self.expect("[")?;
        self.skip_ws();
        let mut slots = Vec::new();
        while !self.check("]") {
            let slot = self.parse_term()?;
            self.skip_ws();
            self.expect("->")?;
            self.skip_ws();
            let value = self.parse_term()?;
            slots.push((slot, value));
            self.skip_ws();
            self.try_consume(",");
            self.skip_ws();
        }
        self.expect("]")?;
        Ok(RifFrame { object, slots })
    }
}
