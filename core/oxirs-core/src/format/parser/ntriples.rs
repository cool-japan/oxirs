//! N-Triples, N-Quads, TriG, and N3 format parsing implementation
//!
//! Uses oxttl for parsing until native implementation is complete

use super::{RdfParser, ReaderQuadParser, SliceQuadParser};
use oxttl::n3::N3Parser;
use oxttl::nquads::NQuadsParser;
use oxttl::ntriples::NTriplesParser;
use oxttl::trig::TriGParser;
use std::io::Read;

use super::helpers::convert_quad;
// convert_quad imported from helpers

// N-Triples parsers
pub(super) fn parse_ntriples_reader<R: Read + Send + 'static>(
    _parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxttl_parser = NTriplesParser::new();

    let iter = oxttl_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|triple| {
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

pub(super) fn parse_ntriples_slice<'a>(_parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxttl_parser = NTriplesParser::new();

    let iter = oxttl_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|triple| {
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

// N-Quads parsers
pub(super) fn parse_nquads_reader<R: Read + Send + 'static>(
    _parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxttl_parser = NQuadsParser::new();

    let iter = oxttl_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_nquads_slice<'a>(_parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxttl_parser = NQuadsParser::new();

    let iter = oxttl_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    SliceQuadParser::new(Box::new(iter))
}

// TriG parsers
pub(super) fn parse_trig_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxttl_parser = if let Some(base) = parser.base_iri() {
        TriGParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| TriGParser::new())
    } else {
        TriGParser::new()
    };

    let iter = oxttl_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_trig_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxttl_parser = if let Some(base) = parser.base_iri() {
        TriGParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| TriGParser::new())
    } else {
        TriGParser::new()
    };

    let iter = oxttl_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    SliceQuadParser::new(Box::new(iter))
}

// N3 parsers
pub(super) fn parse_n3_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    let oxttl_parser = if let Some(base) = parser.base_iri() {
        N3Parser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| N3Parser::new())
    } else {
        N3Parser::new()
    };

    let iter = oxttl_parser.for_reader(reader).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|n3_quad| {
                // N3 can have extended terms, convert to standard RDF
                // For now, we just handle standard RDF terms
                let quad = oxrdf::Quad::new(
                    match n3_quad.subject {
                        oxttl::n3::N3Term::NamedNode(n) => oxrdf::NamedOrBlankNode::NamedNode(n),
                        oxttl::n3::N3Term::BlankNode(b) => oxrdf::NamedOrBlankNode::BlankNode(b),
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 extended terms not yet supported",
                            ))
                        }
                    },
                    match n3_quad.predicate {
                        oxttl::n3::N3Term::NamedNode(n) => n,
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 non-IRI predicates not yet supported",
                            ))
                        }
                    },
                    match n3_quad.object {
                        oxttl::n3::N3Term::NamedNode(n) => oxrdf::Term::NamedNode(n),
                        oxttl::n3::N3Term::BlankNode(b) => oxrdf::Term::BlankNode(b),
                        oxttl::n3::N3Term::Literal(l) => oxrdf::Term::Literal(l),
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 extended terms not yet supported",
                            ))
                        }
                    },
                    n3_quad.graph_name,
                );
                convert_quad(quad)
            })
    });

    ReaderQuadParser::new(Box::new(iter))
}

pub(super) fn parse_n3_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxttl_parser = if let Some(base) = parser.base_iri() {
        N3Parser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| N3Parser::new())
    } else {
        N3Parser::new()
    };

    let iter = oxttl_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(|n3_quad| {
                let quad = oxrdf::Quad::new(
                    match n3_quad.subject {
                        oxttl::n3::N3Term::NamedNode(n) => oxrdf::NamedOrBlankNode::NamedNode(n),
                        oxttl::n3::N3Term::BlankNode(b) => oxrdf::NamedOrBlankNode::BlankNode(b),
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 extended terms not yet supported",
                            ))
                        }
                    },
                    match n3_quad.predicate {
                        oxttl::n3::N3Term::NamedNode(n) => n,
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 non-IRI predicates not yet supported",
                            ))
                        }
                    },
                    match n3_quad.object {
                        oxttl::n3::N3Term::NamedNode(n) => oxrdf::Term::NamedNode(n),
                        oxttl::n3::N3Term::BlankNode(b) => oxrdf::Term::BlankNode(b),
                        oxttl::n3::N3Term::Literal(l) => oxrdf::Term::Literal(l),
                        _ => {
                            return Err(crate::format::error::RdfParseError::unsupported(
                                "N3 extended terms not yet supported",
                            ))
                        }
                    },
                    n3_quad.graph_name,
                );
                convert_quad(quad)
            })
    });

    SliceQuadParser::new(Box::new(iter))
}
