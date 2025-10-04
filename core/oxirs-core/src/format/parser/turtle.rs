//! Turtle format parsing implementation
//!
//! Uses oxttl for parsing until native implementation is complete

use super::helpers::convert_quad;
use super::{RdfParser, ReaderQuadParser, SliceQuadParser};
use crate::format::error::RdfParseError;
use oxttl::turtle::TurtleParser;
use std::io::Read;

pub(super) fn parse_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let mut oxttl_parser = if let Some(base) = parser.base_iri() {
        TurtleParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| TurtleParser::new())
    } else {
        TurtleParser::new()
    };

    // Enable unchecked mode if lenient parsing is requested
    if parser.is_lenient() {
        oxttl_parser = oxttl_parser.unchecked();
    }

    // Parse the reader
    let lenient = parser.is_lenient();
    let iter = oxttl_parser.for_reader(reader).filter_map(move |result| {
        match result {
            Ok(triple) => {
                // Convert Triple to Quad (add default graph)
                let quad = oxrdf::Quad::new(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    oxrdf::GraphName::DefaultGraph,
                );
                Some(convert_quad(quad))
            }
            Err(e) => {
                // In lenient mode, skip errors; otherwise propagate them
                if lenient {
                    None
                } else {
                    Some(Err(RdfParseError::syntax(e.to_string())))
                }
            }
        }
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let mut oxttl_parser = if let Some(base) = parser.base_iri() {
        TurtleParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| TurtleParser::new())
    } else {
        TurtleParser::new()
    };

    // Enable unchecked mode if lenient parsing is requested
    if parser.is_lenient() {
        oxttl_parser = oxttl_parser.unchecked();
    }

    // Parse the slice
    let lenient = parser.is_lenient();
    let iter = oxttl_parser.for_slice(slice).filter_map(move |result| {
        match result {
            Ok(triple) => {
                // Convert Triple to Quad (add default graph)
                let quad = oxrdf::Quad::new(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    oxrdf::GraphName::DefaultGraph,
                );
                Some(convert_quad(quad))
            }
            Err(e) => {
                // In lenient mode, skip errors; otherwise propagate them
                if lenient {
                    None
                } else {
                    Some(Err(RdfParseError::syntax(e.to_string())))
                }
            }
        }
    });

    SliceQuadParser::new(Box::new(iter))
}
