//! RDF/XML format parsing implementation
//!
//! Uses oxrdfxml for parsing until native implementation is complete

use super::{RdfParser, ReaderQuadParser, SliceQuadParser};
use oxrdfxml::RdfXmlParser;
use std::io::Read;

use super::helpers::convert_quad;
// convert_quad imported from helpers

pub(super) fn parse_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxrdfxml_parser = if let Some(base) = parser.base_iri() {
        RdfXmlParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| RdfXmlParser::new())
    } else {
        RdfXmlParser::new()
    };

    // Parse the reader
    let iter = oxrdfxml_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|triple| {
                // Convert Triple to Quad (add default graph)
                let quad = oxrdf::Quad::new(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    oxrdf::GraphName::DefaultGraph,
                );
                convert_quad(quad)
            })
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxrdfxml_parser = if let Some(base) = parser.base_iri() {
        RdfXmlParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| RdfXmlParser::new())
    } else {
        RdfXmlParser::new()
    };

    // Parse the slice
    let iter = oxrdfxml_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|triple| {
                // Convert Triple to Quad (add default graph)
                let quad = oxrdf::Quad::new(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                    oxrdf::GraphName::DefaultGraph,
                );
                convert_quad(quad)
            })
    });

    SliceQuadParser::new(Box::new(iter))
}
