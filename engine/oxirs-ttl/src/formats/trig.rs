//! TriG format parser and serializer (stub implementation)

use crate::error::TurtleResult;
use crate::toolkit::{Parser, Serializer};
use oxirs_core::model::Quad;
use std::io::{BufRead, Read, Write};

/// TriG parser (stub)
#[derive(Debug, Clone)]
pub struct TriGParser;

impl TriGParser {
    pub fn new() -> Self {
        Self
    }
}

impl Parser<Quad> for TriGParser {
    fn parse<R: Read>(&self, _reader: R) -> TurtleResult<Vec<Quad>> {
        todo!("TriG parser implementation")
    }
    
    fn for_reader<R: BufRead>(&self, _reader: R) -> Box<dyn Iterator<Item = TurtleResult<Quad>>> {
        todo!("TriG streaming parser implementation")
    }
}

/// TriG serializer (stub)
#[derive(Debug, Clone)]
pub struct TriGSerializer;

impl TriGSerializer {
    pub fn new() -> Self {
        Self
    }
}

impl Serializer<Quad> for TriGSerializer {
    fn serialize<W: Write>(&self, _quads: &[Quad], _writer: W) -> TurtleResult<()> {
        todo!("TriG serializer implementation")
    }
    
    fn serialize_item<W: Write>(&self, _quad: &Quad, _writer: W) -> TurtleResult<()> {
        todo!("TriG item serializer implementation")
    }
}