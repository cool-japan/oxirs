//! N-Quads format parser and serializer (stub implementation)

use crate::error::TurtleResult;
use crate::toolkit::{Parser, Serializer};
use oxirs_core::model::Quad;
use std::io::{BufRead, Read, Write};

/// N-Quads parser (stub)
#[derive(Debug, Clone)]
pub struct NQuadsParser;

impl NQuadsParser {
    pub fn new() -> Self {
        Self
    }
}

impl Parser<Quad> for NQuadsParser {
    fn parse<R: Read>(&self, _reader: R) -> TurtleResult<Vec<Quad>> {
        todo!("N-Quads parser implementation")
    }

    fn for_reader<R: BufRead>(&self, _reader: R) -> Box<dyn Iterator<Item = TurtleResult<Quad>>> {
        todo!("N-Quads streaming parser implementation")
    }
}

/// N-Quads serializer (stub)
#[derive(Debug, Clone)]
pub struct NQuadsSerializer;

impl NQuadsSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer<Quad> for NQuadsSerializer {
    fn serialize<W: Write>(&self, _quads: &[Quad], _writer: W) -> TurtleResult<()> {
        todo!("N-Quads serializer implementation")
    }

    fn serialize_item<W: Write>(&self, _quad: &Quad, _writer: W) -> TurtleResult<()> {
        todo!("N-Quads item serializer implementation")
    }
}
