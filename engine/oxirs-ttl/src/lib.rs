//! # OxiRS Turtle - RDF Format Parser
//!
//! [![Version](https://img.shields.io/badge/version-0.2.4-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl)
//!
//! **Status**: Production Release (v0.2.4)
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
pub mod diff;
pub mod error;
pub mod formats;
pub mod incremental;
pub mod lexer;
pub mod parser;
pub mod patch;
pub mod profiling;
pub mod streaming;
pub mod toolkit;
pub mod writer;

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

    /// N3 reasoning primitives (forward chaining)
    pub mod reasoning {
        pub use crate::formats::n3_reasoning::*;
    }

    /// N3 backward chaining engine and proof tracing
    pub mod backward_chaining {
        pub use crate::formats::n3_backward_chaining::*;
    }

    /// N3 built-in predicate evaluators (math, string, list, logic)
    pub mod builtins {
        pub use crate::formats::n3_builtins::*;
    }

    /// Enhanced N3 rule file parser and serializer
    pub mod rule_parser {
        pub use crate::formats::n3_rule_parser::*;
    }
}

// Re-export common types
pub use diff::{compute_diff, parse_patch, RdfDiff};
pub use error::{TurtleParseError, TurtleSyntaxError};
pub use incremental::{IncrementalParser, ParseCheckpoint, ParseState};
pub use parser::{
    NQuad, NQuadsLiteParser, NTriple, NTriplesLiteParser, ParseError as NtParseError,
};
pub use profiling::{ParsingStats, TtlProfiler};
pub use streaming::{PrintProgress, ProgressCallback, StreamingConfig, StreamingParser};
pub use toolkit::{Parser, RuleRecognizer, Serializer, TokenRecognizer};
pub use writer::{RdfTerm, TermType, TurtleWriter, TurtleWriterConfig};

pub use patch::{
    apply_patch, diff_to_patch, Graph, PatchChange, PatchError, PatchHeader, PatchParser,
    PatchQuad, PatchResult, PatchSerializer, PatchStats, PatchTerm, PatchTriple, RdfPatch,
};

pub mod mapping;
/// Turtle pretty printer with prefix analysis.
pub mod pretty_printer;

/// @base / @prefix IRI resolution for Turtle/TriG (v1.1.0 round 6)
pub mod base_directive;

/// Incremental/streaming Turtle parser for large files (v1.1.0 round 7)
pub mod streaming_parser;

/// Namespace/prefix management for Turtle and SPARQL serialization (v1.1.0 round 8)
pub mod namespace_mapper;

/// Turtle/TriG document syntax validation (v1.1.0 round 9)
pub mod turtle_validator;

/// Prefix/CURIE resolver for Turtle and TriG documents (v1.1.0 round 10)
pub mod prefix_resolver;

/// JSON-LD framing: apply a frame template to a node set (v1.1.0 round 11)
pub mod json_ld_framing;

/// IRI prefix catalog with CURIE expansion and compression (v1.1.0 round 13)
pub mod iri_catalog;

/// Compact Turtle serialization with subject/predicate grouping (v1.1.0 round 12)
pub mod compact_serializer;

/// N-Triples/N-Quads serialization with proper escaping (v1.1.0 round 11)
pub mod ntriples_writer;

/// Basic RDFa 1.1 Lite parser: property/typeof/resource/about/prefix attributes,
/// literal extraction, rel/rev links, context inheritance (v1.1.0 round 13)
pub mod rdfa_parser;

/// N-Triples and N-Quads streaming parser: IRI/blank-node/literal tokens,
/// comment/blank line skip, typed literals, language tags, Unicode escapes (v1.1.0 round 14)
pub mod nt_parser;

pub use mapping::{DataSource, MappingEngine, MappingRule, MappingRuleBuilder, ObjectSpec};

pub mod jsonld {
    //! JSON-LD 1.1 format parser, serializer, and processing algorithms
    pub use crate::formats::jsonld::*;
}

pub use formats::jsonld::{
    JsonLdContext, JsonLdError, JsonLdProcessor, JsonLdQuad, JsonLdResult, JsonLdTerm,
    JsonLdWriter, TermDefinition,
};

pub mod rdf_thrift {
    //! RDF Binary (Thrift) format — read + write support
    //!
    //! Implements the Jena-compatible RDF Binary Thrift encoding as described at
    //! <https://jena.apache.org/documentation/io/rdf-binary.html>.
    pub use crate::formats::rdf_thrift::*;
}

/// JSON-LD compaction: converts expanded JSON-LD to compact form using a context (v1.1.0 round 15)
pub mod jsonld_compactor;

/// TriG named-graph Turtle parser: @prefix/PREFIX declarations, GRAPH blocks,
/// default-graph triples, prefix expansion, graph_sizes, named_graphs (v1.1.0 round 16)
pub mod trig_parser;
