//! N3 format parser (stub implementation)

use crate::error::TurtleResult;
use crate::toolkit::Parser;
use oxirs_core::model::Triple;
use std::io::{BufRead, Read};

/// N3 parser (stub)
#[derive(Debug, Clone)]
pub struct N3Parser;

impl N3Parser {
    pub fn new() -> Self {
        Self
    }
}

impl Parser<Triple> for N3Parser {
    fn parse<R: Read>(&self, _reader: R) -> TurtleResult<Vec<Triple>> {
        todo!("N3 parser implementation")
    }

    fn for_reader<R: BufRead>(&self, _reader: R) -> Box<dyn Iterator<Item = TurtleResult<Triple>>> {
        todo!("N3 streaming parser implementation")
    }
}
