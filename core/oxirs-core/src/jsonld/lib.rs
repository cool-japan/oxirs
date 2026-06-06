#![doc = include_str!("../README.md")]
#![doc(test(attr(deny(warnings))))]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/oxigraph/oxigraph/main/logo.svg")]
#![doc(html_logo_url = "https://raw.githubusercontent.com/oxigraph/oxigraph/main/logo.svg")]

mod context;
mod context_core;
mod context_types;
mod error;
mod expansion;
mod from_rdf;
mod profile;
mod to_rdf;
mod to_rdf_converter;
mod to_rdf_parser;
mod to_rdf_readers;
mod to_rdf_tests;

pub use context::{JsonLdLoadDocumentOptions, JsonLdRemoteDocument};
pub use error::{JsonLdErrorCode, JsonLdParseError, JsonLdSyntaxError, TextPosition};
#[cfg(feature = "async-tokio")]
pub use from_rdf::TokioAsyncWriterJsonLdSerializer;
pub use from_rdf::{JsonLdSerializer, WriterJsonLdSerializer};
#[doc(hidden)]
pub use profile::JsonLdProcessingMode;
pub use profile::{JsonLdProfile, JsonLdProfileSet};
#[cfg(feature = "async-tokio")]
pub use to_rdf::TokioAsyncReaderJsonLdParser;
pub use to_rdf::{JsonLdParser, JsonLdPrefixesIter, ReaderJsonLdParser, SliceJsonLdParser};

const MAX_CONTEXT_RECURSION: usize = 8;
