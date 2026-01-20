//! JSON-LD format parsing implementation
//!
//! Uses oxjsonld for parsing until native implementation is complete

use super::{RdfParser, ReaderQuadParser, SliceQuadParser};
use oxjsonld::JsonLdParser;
use std::io::Read;

use super::helpers::convert_quad;
// convert_quad imported from helpers

pub(super) fn parse_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxjsonld_parser = if let Some(base) = parser.base_iri() {
        JsonLdParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| JsonLdParser::new())
    } else {
        JsonLdParser::new()
    };

    // Parse the reader
    let iter = oxjsonld_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxjsonld_parser = if let Some(base) = parser.base_iri() {
        JsonLdParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| JsonLdParser::new())
    } else {
        JsonLdParser::new()
    };

    // Parse the slice
    let iter = oxjsonld_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    SliceQuadParser::new(Box::new(iter))
}
