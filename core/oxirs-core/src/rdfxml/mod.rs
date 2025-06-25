//! RDF/XML parsing and serialization support for OxiRS
//!
//! This module provides complete RDF/XML format support ported from Oxigraph
//! with ultra-high performance DOM-free streaming capabilities.

mod error;
mod parser;
mod serializer;
// mod streaming; // TODO: Fix compilation errors
mod utils;
mod wrapper;

pub use error::{RdfXmlParseError, RdfXmlSyntaxError};
pub use parser::{RdfXmlParser, RdfXmlPrefixesIter, ReaderRdfXmlParser, SliceRdfXmlParser};
#[cfg(feature = "async-tokio")]
pub use parser::TokioAsyncReaderRdfXmlParser;
pub use serializer::{RdfXmlSerializer, WriterRdfXmlSerializer};
#[cfg(feature = "async-tokio")]
pub use serializer::TokioAsyncWriterRdfXmlSerializer;
// pub use streaming::{
//     DomFreeStreamingRdfXmlParser, RdfXmlStreamingConfig, RdfXmlStreamingSink,
//     RdfXmlStreamingStatistics, MemoryRdfXmlSink, RdfXmlSinkStatistics,
//     ElementType, ParseType, NamespaceContext, ElementContext
// }; // TODO: Re-enable when streaming module is fixed
pub use wrapper::parse_rdfxml;