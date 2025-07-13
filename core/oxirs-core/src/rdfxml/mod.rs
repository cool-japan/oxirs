//! RDF/XML parsing and serialization support for OxiRS
//!
//! This module provides complete RDF/XML format support ported from Oxigraph
//! with ultra-high performance DOM-free streaming capabilities.

mod error;
mod parser;
mod serializer;
mod streaming;
mod utils;
pub mod wrapper;

pub use error::{RdfXmlParseError, RdfXmlSyntaxError};
#[cfg(feature = "async-tokio")]
pub use parser::TokioAsyncReaderRdfXmlParser;
pub use parser::{RdfXmlParser, RdfXmlPrefixesIter, ReaderRdfXmlParser, SliceRdfXmlParser};
#[cfg(feature = "async-tokio")]
pub use serializer::TokioAsyncWriterRdfXmlSerializer;
pub use serializer::{RdfXmlSerializer, WriterRdfXmlSerializer};
pub use streaming::{
    DomFreeStreamingRdfXmlParser, ElementContext, ElementType, MemoryRdfXmlSink, NamespaceContext,
    ParseType, RdfXmlSinkStatistics, RdfXmlStreamingConfig, RdfXmlStreamingSink,
    RdfXmlStreamingStatistics,
};
pub use wrapper::parse_rdfxml;
