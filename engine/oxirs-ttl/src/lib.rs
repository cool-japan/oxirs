//! OxiRS Turtle-family RDF Parser and Serializer
//!
//! This crate provides high-performance parsing and serialization for RDF formats
//! in the Turtle family: Turtle, TriG, N-Triples, N-Quads, and N3.
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
//!
//! # Quick Start
//!
//! ```rust
//! use oxirs_ttl::turtle::TurtleParser;
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

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;
pub mod toolkit;
pub mod lexer;
pub mod formats;

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
    //! N3 format parser (experimental)
    pub use crate::formats::n3::*;
}

// Re-export common types
pub use error::{TurtleParseError, TurtleSyntaxError};
pub use toolkit::{TokenRecognizer, RuleRecognizer, Parser, Serializer};