//! I/O utilities for RDF data processing
//!
//! This module provides various I/O utilities including async streaming support
//! for parsing and serializing RDF data.

#[cfg(feature = "async-tokio")]
pub mod async_streaming;

pub mod zero_copy;

#[cfg(feature = "async-tokio")]
pub use async_streaming::{
    AsyncRdfParser, AsyncRdfSerializer, AsyncStreamingConfig, AsyncStreamingParser,
    AsyncStreamingSerializer, BackpressureReader, ProgressCallback, StreamingProgress,
};

pub use zero_copy::{
    MmapReader, MmapWriter, ZeroCopyDeserialize, ZeroCopySerialize, ZeroCopyStr,
    ZeroCopyTerm, ZeroCopyTriple, ZeroCopyQuad, ZeroCopyIri, ZeroCopyBlankNode,
    ZeroCopyLiteral,
};