//! RDF/XML parsing and serialization support for OxiRS
//!
//! This module provides complete RDF/XML format support ported from Oxigraph.

mod error;
mod parser;
mod serializer;
mod utils;

pub use error::{RdfXmlParseError, RdfXmlSyntaxError};
pub use parser::{RdfXmlParser, RdfXmlPrefixesIter, ReaderRdfXmlParser, SliceRdfXmlParser};
#[cfg(feature = "async")]
pub use parser::TokioAsyncReaderRdfXmlParser;
pub use serializer::{RdfXmlSerializer, WriterRdfXmlSerializer};
#[cfg(feature = "async")]
pub use serializer::TokioAsyncWriterRdfXmlSerializer;