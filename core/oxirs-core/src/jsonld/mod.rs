//! JSON-LD processing functionality for OxiRS Core
//!
//! This module provides JSON-LD parsing and serialization capabilities,
//! ported from Oxigraph's oxjsonld implementation with ultra-high performance
//! streaming enhancements.

mod context;
mod error;
mod expansion;
mod from_rdf;
mod profile;
mod streaming;
mod to_rdf;

pub use context::{JsonLdLoadDocumentOptions, JsonLdRemoteDocument};
pub use error::{JsonLdErrorCode, JsonLdParseError, JsonLdSyntaxError, TextPosition};
#[cfg(feature = "async")]
pub use from_rdf::TokioAsyncWriterJsonLdSerializer;
pub use from_rdf::{JsonLdSerializer, WriterJsonLdSerializer};
#[doc(hidden)]
pub use profile::JsonLdProcessingMode;
pub use profile::{JsonLdProfile, JsonLdProfileSet};
pub use streaming::{
    MemoryStreamingSink, SinkStatistics, StreamingConfig, StreamingSink, StreamingStatistics,
    UltraStreamingJsonLdParser, ZeroCopyLevel,
};
#[cfg(feature = "async")]
pub use to_rdf::TokioAsyncReaderJsonLdParser;
pub use to_rdf::{JsonLdParser, JsonLdPrefixesIter, ReaderJsonLdParser, SliceJsonLdParser};

pub const MAX_CONTEXT_RECURSION: usize = 8;
