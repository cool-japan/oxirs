//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::update::QuadPattern;
use anyhow::Result;

use super::queryparser_type::QueryParser;

impl QueryParser {
    /// Parse quad pattern data for INSERT/DELETE templates
    pub(super) fn parse_quad_pattern_data(&mut self) -> Result<Vec<QuadPattern>> {
        self.parse_quad_data()
    }
}
