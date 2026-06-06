//! JSON-LD processing functionality for OxiRS Core
//!
//! This module provides JSON-LD parsing and serialization capabilities,
//! ported from Oxigraph's oxjsonld implementation with ultra-high performance
//! streaming enhancements.

pub mod compaction;
mod context;
mod context_core;
mod context_types;
mod error;
mod expansion;
pub(crate) mod expansion_algorithm;
pub mod expansion_algorithm_ctx;
pub mod expansion_algorithm_step;
mod expansion_algorithm_tests;
pub mod expansion_algorithm_types;
pub(crate) mod expansion_algorithm_value;
pub(crate) mod expansion_context;
#[allow(unused_imports)]
mod expansion_tests;
pub mod flattening;
pub mod framing;
pub mod framing_embed;
pub mod framing_match;
mod framing_tests;
mod from_rdf;
mod profile;
mod streaming;
pub mod to_rdf;
mod to_rdf_converter;
mod to_rdf_parser;
mod to_rdf_readers;
mod to_rdf_tests;

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
