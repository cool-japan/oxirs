//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{RifFormula, RifTerm};

use super::rifparser_type::RifParser;

impl RifParser {
    /// Parse atomic formula
    pub(super) fn parse_atom_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        if self.peek_frame() {
            return Ok(RifFormula::Frame(self.parse_frame()?));
        }
        if self.peek_member() {
            let (term, class) = self.parse_member()?;
            return Ok(RifFormula::Member(term, class));
        }
        if self.peek_subclass() {
            let (sub, sup) = self.parse_subclass()?;
            return Ok(RifFormula::Subclass(sub, sup));
        }
        let atom = self.parse_atom()?;
        Ok(RifFormula::Atom(atom))
    }
    /// Parse membership: term#class
    pub(super) fn parse_member(&mut self) -> Result<(RifTerm, RifTerm)> {
        let term = self.parse_term()?;
        self.expect("#")?;
        let class = self.parse_term()?;
        Ok((term, class))
    }
    /// Parse subclass: sub##super
    pub(super) fn parse_subclass(&mut self) -> Result<(RifTerm, RifTerm)> {
        let sub = self.parse_term()?;
        self.expect("##")?;
        let sup = self.parse_term()?;
        Ok((sub, sup))
    }
    fn peek_frame(&self) -> bool {
        let mut i = self.pos;
        while i < self.input.len() {
            let c = self
                .input
                .chars()
                .nth(i)
                .expect("index is within input bounds");
            if c == '[' {
                return true;
            }
            if c == '(' || c.is_whitespace() || c == ':' {
                break;
            }
            i += 1;
        }
        false
    }
    fn peek_member(&self) -> bool {
        let mut i = self.pos;
        let mut depth = 0;
        while i < self.input.len() {
            let c = self
                .input
                .chars()
                .nth(i)
                .expect("index is within input bounds");
            if c == '(' {
                depth += 1;
            }
            if c == ')' {
                depth -= 1;
            }
            if depth == 0 && c == '#' && !self.input[i..].starts_with("##") {
                return true;
            }
            if depth == 0 && (c.is_whitespace() || c == ')' || c == ',') {
                break;
            }
            i += 1;
        }
        false
    }
    fn peek_subclass(&self) -> bool {
        let mut i = self.pos;
        while i < self.input.len() - 1 {
            if self.input[i..].starts_with("##") {
                return true;
            }
            let c = self
                .input
                .chars()
                .nth(i)
                .expect("index is within input bounds");
            if c.is_whitespace() || c == ')' || c == ',' {
                break;
            }
            i += 1;
        }
        false
    }
}
