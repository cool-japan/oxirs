//! TriG Streaming Parser — pull-parser for the W3C TriG format.
//!
//! TriG extends Turtle with named graph support, allowing triples to be grouped
//! in named or anonymous graph blocks. This module provides a pull-parser
//! `TriGStreamingParser` that implements `Iterator<Item=Result<StreamedQuad, TriGParseError>>`.
//!
//! # Specification
//!
//! TriG is defined by the W3C at <https://www.w3.org/TR/trig/>
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::trig_streaming::{TriGStreamingParser, StreamedQuad};
//! use std::io::Cursor;
//!
//! let data = b"@prefix ex: <http://example.org/> .\nex:g1 { ex:s ex:p ex:o . }\n";
//! let parser = TriGStreamingParser::new(Cursor::new(data));
//! let quads: Vec<_> = parser.collect();
//! assert!(!quads.is_empty());
//! ```

pub mod lexer;
pub mod parser;
#[cfg(test)]
mod tests;

pub use lexer::{TriGLexer, TriGToken};
pub use parser::TriGParser;

use std::io::{BufReader, Read};
use thiserror::Error;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur when parsing TriG documents.
#[derive(Debug, Error)]
pub enum TriGParseError {
    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A token could not be recognised at the given line.
    #[error("Invalid token at line {line}: {message}")]
    InvalidToken {
        /// 1-based line number.
        line: usize,
        /// Human-readable description.
        message: String,
    },

    /// A triple statement was syntactically malformed.
    #[error("Invalid triple at line {line}: {message}")]
    InvalidTriple {
        /// 1-based line number.
        line: usize,
        /// Human-readable description.
        message: String,
    },

    /// A graph block was syntactically malformed.
    #[error("Invalid graph at line {line}: {name}")]
    InvalidGraph {
        /// 1-based line number.
        line: usize,
        /// Graph name or context description.
        name: String,
    },

    /// A graph block was opened but never closed.
    #[error("Unclosed graph opened at line {opened_at}")]
    UnclosedGraph {
        /// 1-based line at which `{` was encountered.
        opened_at: usize,
    },
}

// ============================================================================
// Term types
// ============================================================================

/// A literal value with optional datatype IRI and/or language tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriGLiteral {
    /// The lexical form of the literal value.
    pub value: String,
    /// Optional datatype IRI (e.g. `"http://www.w3.org/2001/XMLSchema#integer"`).
    pub datatype: Option<String>,
    /// Optional BCP 47 language tag (e.g. `"en"`, `"fr-CA"`).
    pub language: Option<String>,
}

/// An RDF term in a streamed quad from a TriG document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriGTerm {
    /// An IRI reference.
    NamedNode(String),
    /// A blank node with a label.
    BlankNode(String),
    /// An RDF literal.
    Literal(TriGLiteral),
}

impl TriGTerm {
    /// Return the IRI string if this is a `NamedNode`.
    pub fn as_iri(&self) -> Option<&str> {
        match self {
            Self::NamedNode(iri) => Some(iri),
            _ => None,
        }
    }

    /// Return the blank node label if this is a `BlankNode`.
    pub fn as_blank(&self) -> Option<&str> {
        match self {
            Self::BlankNode(label) => Some(label),
            _ => None,
        }
    }

    /// Return the literal if this is a `Literal`.
    pub fn as_literal(&self) -> Option<&TriGLiteral> {
        match self {
            Self::Literal(lit) => Some(lit),
            _ => None,
        }
    }
}

// ============================================================================
// Quad
// ============================================================================

/// A quad streamed from a TriG document.
///
/// Triples that appear outside any graph block have `graph_name = None`
/// (the default graph).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedQuad {
    /// The quad subject.
    pub subject: TriGTerm,
    /// The quad predicate (must be a named node).
    pub predicate: TriGTerm,
    /// The quad object.
    pub object: TriGTerm,
    /// Optional named graph; `None` means the default graph.
    pub graph_name: Option<TriGTerm>,
}

// ============================================================================
// Streaming parser
// ============================================================================

/// A streaming pull-parser for TriG documents.
///
/// Implements `Iterator<Item=Result<StreamedQuad, TriGParseError>>`.  Each call
/// to `next()` drives the recursive-descent parser to produce the next quad.
pub struct TriGStreamingParser<R: Read> {
    inner: TriGParser<BufReader<R>>,
    /// Pending quads from the current statement.
    pending: Vec<StreamedQuad>,
    /// Whether the underlying parser has been exhausted.
    done: bool,
}

impl<R: Read> TriGStreamingParser<R> {
    /// Create a new streaming parser wrapping any `Read` source.
    pub fn new(reader: R) -> Self {
        Self {
            inner: TriGParser::new(BufReader::new(reader)),
            pending: Vec::new(),
            done: false,
        }
    }
}

impl<R: Read> Iterator for TriGStreamingParser<R> {
    type Item = Result<StreamedQuad, TriGParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Drain pending quads first.
            if !self.pending.is_empty() {
                return Some(Ok(self.pending.remove(0)));
            }

            if self.done {
                return None;
            }

            // Parse the next statement.
            match self.inner.parse_statement() {
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Ok(Some(quads)) => {
                    if quads.is_empty() {
                        continue; // directive — no quads, try again
                    }
                    self.pending.extend(quads);
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}
