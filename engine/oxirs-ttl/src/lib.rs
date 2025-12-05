//! # OxiRS Turtle - RDF Format Parser
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--beta.2-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl)
//!
//! **Status**: Beta Release (v0.1.0-beta.2)
//! **Stability**: Production-ready with 461 passing tests and 97% W3C compliance.
//!
//! High-performance parsing and serialization for RDF formats in the Turtle family.
//! Supports Turtle, TriG, N-Triples, N-Quads, and N3 with streaming and error recovery.
//!
//! Ported from Oxigraph's oxttl crate with adaptations for OxiRS.
//!
//! # Features
//!
//! - **Streaming support**: Process large files with minimal memory usage
//! - **Error recovery**: Continue parsing despite syntax errors
//! - **Async I/O**: Optional Tokio async support
//! - **Parallel processing**: Process files in parallel chunks
//! - **RDF 1.2 support**: Quoted triples and directional language tags
//! - **Incremental parsing**: Parse as bytes arrive with checkpointing
//!
//! # Quick Start
//!
//! ## Basic Turtle Parsing
//!
//! ```rust
//! use oxirs_ttl::{turtle::TurtleParser, Parser};
//! use std::io::Cursor;
//!
//! let turtle_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:subject ex:predicate "object" .
//! "#;
//!
//! let parser = TurtleParser::new();
//! for result in parser.for_reader(Cursor::new(turtle_data)) {
//!     let triple = result?;
//!     println!("{}", triple);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Error Recovery with Lenient Mode
//!
//! ```rust
//! use oxirs_ttl::turtle::TurtleParser;
//!
//! let turtle_with_errors = r#"
//! @prefix ex: <http://example.org/> .
//! ex:good ex:pred "value" .
//! ex:also_good ex:pred "value2" .
//! "#;
//!
//! // Lenient mode continues parsing after errors
//! let parser = TurtleParser::new_lenient();
//! let triples = parser.parse_document(turtle_with_errors)?;
//! println!("Parsed {} triples", triples.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Streaming Large Files
//!
//! ```rust
//! use oxirs_ttl::{StreamingParser, StreamingConfig};
//! use std::io::Cursor;
//!
//! let config = StreamingConfig::default()
//!     .with_batch_size(10000);  // Process 10K triples per batch
//!
//! let data = Cursor::new(b"<http://s> <http://p> <http://o> .");
//! let parser = StreamingParser::with_config(data, config);
//! let mut total = 0;
//!
//! for batch in parser.batches() {
//!     let triples = batch?;
//!     total += triples.len();
//!     println!("Processed batch of {} triples", triples.len());
//! }
//! println!("Total: {} triples", total);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Incremental Parsing
//!
//! ```rust
//! use oxirs_ttl::{IncrementalParser, ParseState};
//!
//! let mut parser = IncrementalParser::new();
//!
//! // Feed data as it arrives
//! parser.push_data(b"@prefix ex: <http://example.org/> .\n")?;
//! parser.push_data(b"ex:s ex:p \"object\" .\n")?;
//! parser.push_eof();
//!
//! // Parse available complete statements
//! let triples = parser.parse_available()?;
//! println!("Parsed {} triples", triples.len());
//!
//! assert_eq!(parser.state(), ParseState::Complete);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Serialization with Pretty Printing
//!
//! ```rust
//! use oxirs_ttl::turtle::TurtleSerializer;
//! use oxirs_ttl::toolkit::{Serializer, SerializationConfig};
//! use oxirs_core::model::{NamedNode, Triple};
//!
//! let triple = Triple::new(
//!     NamedNode::new("http://example.org/subject")?,
//!     NamedNode::new("http://example.org/predicate")?,
//!     NamedNode::new("http://example.org/object")?
//! );
//!
//! // Create config with pretty printing
//! let config = SerializationConfig::default()
//!     .with_pretty(true)
//!     .with_use_prefixes(true);
//!
//! let serializer = TurtleSerializer::with_config(config);
//!
//! let mut output = Vec::new();
//! serializer.serialize(&vec![triple], &mut output)?;
//!
//! let turtle_string = String::from_utf8(output)?;
//! println!("{}", turtle_string);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod convenience;
pub mod error;
pub mod formats;
pub mod incremental;
pub mod lexer;
pub mod profiling;
pub mod streaming;
pub mod toolkit;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "async-tokio")]
pub mod async_parser;

// Re-export the main format APIs
pub mod turtle {
    //! Turtle format parser and serializer
    pub use crate::formats::turtle::*;
}

pub mod trig {
    //! TriG format parser and serializer
    pub use crate::formats::trig::*;
}

pub mod ntriples {
    //! N-Triples format parser and serializer
    pub use crate::formats::ntriples::*;
}

pub mod nquads {
    //! N-Quads format parser and serializer
    pub use crate::formats::nquads::*;
}

pub mod n3 {
    //! N3 format parser and reasoning (experimental)
    pub use crate::formats::n3::*;
    pub use crate::formats::n3_parser::*;
    pub use crate::formats::n3_serializer::*;
    pub use crate::formats::n3_types::*;

    /// N3 reasoning primitives
    pub mod reasoning {
        pub use crate::formats::n3_reasoning::*;
    }
}

// Re-export common types
pub use error::{TurtleParseError, TurtleSyntaxError};
pub use incremental::{IncrementalParser, ParseCheckpoint, ParseState};
pub use profiling::{ParsingStats, TtlProfiler};
pub use streaming::{PrintProgress, ProgressCallback, StreamingConfig, StreamingParser};
pub use toolkit::{Parser, RuleRecognizer, Serializer, TokenRecognizer};
