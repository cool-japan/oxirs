//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Expression;
use anyhow::Result;

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or_expression()
    }
}
