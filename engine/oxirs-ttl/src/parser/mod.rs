//! Standalone RDF parsers operating on lightweight [`crate::writer::RdfTerm`] types
//!
//! These parsers do not depend on `oxirs-core` model types and can be used in
//! isolation for simple N-Triples / N-Quads parsing tasks.

pub mod ntriples_parser;

pub use ntriples_parser::{NQuad, NQuadsLiteParser, NTriple, NTriplesLiteParser, ParseError};
