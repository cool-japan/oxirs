//! Turtle format parser and serializer
//!
//! Full implementation of the Turtle RDF serialization format with support for:
//! - Prefix declarations (@prefix)
//! - Base IRI declarations (@base)
//! - Abbreviated syntax (a for rdf:type, semicolons, commas)
//! - Collection syntax []
//! - List syntax ()
//! - Comments and whitespace handling
//! - RDF 1.2 features (quoted triples, directional language tags)
//!
//! # Module Organization
//!
//! - `types` - Common types (TurtleParsingContext, TurtleStatement, Token, TokenKind)
//! - `tokenizer` - TurtleTokenizer for lexical analysis
//! - `parser` - TurtleParser for parsing Turtle documents
//! - `serializer` - TurtleSerializer for serializing to Turtle format
//!
//! # Quick Start
//!
//! ```rust
//! use oxirs_ttl::turtle::{TurtleParser, TurtleSerializer};
//! use oxirs_ttl::toolkit::{Parser, Serializer};
//! use std::io::Cursor;
//!
//! // Parse Turtle
//! let turtle_data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:subject ex:predicate "object" .
//! "#;
//!
//! let parser = TurtleParser::new();
//! for result in parser.for_reader(Cursor::new(turtle_data)) {
//!     let triple = result.unwrap();
//!     println!("{}", triple);
//! }
//!
//! // Serialize to Turtle
//! let serializer = TurtleSerializer::new();
//! let mut output: Vec<u8> = Vec::new();
//! // serializer.serialize(&triples, &mut output).unwrap();
//! ```

mod parser;
mod serializer;
pub(crate) mod tokenizer;
mod types;

// Re-export public types
pub use parser::TurtleParser;
pub use serializer::TurtleSerializer;
pub use types::{Token, TokenKind, TurtleParsingContext, TurtleStatement};
