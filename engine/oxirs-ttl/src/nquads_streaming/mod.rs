//! N-Quads streaming parser: pull-parse N-Quads format line by line.
//!
//! Implements a zero-copy-friendly [`NQuadsStreamingParser`] that wraps a
//! `BufReader<R>` and acts as an iterator over [`StreamedQuad`] values.
//! Each call to `next()` reads exactly one line from the underlying reader
//! and either returns a parsed quad, skips blank/comment lines, or returns
//! an error.
//!
//! # Specification
//!
//! N-Quads is defined by the W3C at
//! <https://www.w3.org/TR/n-quads/>
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::nquads_streaming::{NQuadsStreamingParser, StreamedQuad};
//! use std::io::Cursor;
//!
//! let data = b"<http://s> <http://p> <http://o> .\n";
//! let mut parser = NQuadsStreamingParser::new(Cursor::new(data));
//! let quad = parser.next().expect("line").expect("valid quad");
//! assert!(quad.graph_name.is_none());
//! ```

pub mod lexer;
pub mod parser;
mod tests;

pub use lexer::{NQuadsLexer, Token};
pub use parser::{parse_line, parse_term};

use std::io::{BufRead, BufReader, Read};
use thiserror::Error;

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur when parsing N-Quads.
#[derive(Debug, Error)]
pub enum NQuadsParseError {
    /// I/O error from the underlying reader.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A line could not be parsed as a valid N-Quads statement.
    #[error("Invalid N-Quads line {line}: {message}")]
    InvalidLine {
        /// 1-based line number.
        line: usize,
        /// Human-readable description of the parse failure.
        message: String,
    },

    /// An IRI reference was syntactically invalid.
    #[error("Invalid IRI at line {line}: <{iri}>")]
    InvalidIri {
        /// 1-based line number.
        line: usize,
        /// The invalid IRI text (without angle brackets).
        iri: String,
    },

    /// A literal value was syntactically invalid.
    #[error("Invalid literal at line {line}: {message}")]
    InvalidLiteral {
        /// 1-based line number.
        line: usize,
        /// Human-readable description of the literal error.
        message: String,
    },

    /// A blank node label was syntactically invalid.
    #[error("Invalid blank node at line {line}: _:{name}")]
    InvalidBlankNode {
        /// 1-based line number.
        line: usize,
        /// The invalid blank node label (without `_:` prefix).
        name: String,
    },
}

// ============================================================================
// Term types
// ============================================================================

/// A literal value with optional datatype IRI or language tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedLiteral {
    /// The lexical form of the literal.
    pub value: String,
    /// Optional datatype IRI (e.g., `"http://www.w3.org/2001/XMLSchema#integer"`).
    pub datatype: Option<String>,
    /// Optional language tag (e.g., `"en"`, `"fr"`).
    pub language: Option<String>,
}

/// An RDF term in a streamed quad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamedTerm {
    /// An IRI reference, e.g. `<http://example.org/>`.
    NamedNode(String),
    /// A blank node, e.g. `_:b0`.
    BlankNode(String),
    /// An RDF literal.
    Literal(StreamedLiteral),
}

impl StreamedTerm {
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
    pub fn as_literal(&self) -> Option<&StreamedLiteral> {
        match self {
            Self::Literal(lit) => Some(lit),
            _ => None,
        }
    }
}

// ============================================================================
// Quad
// ============================================================================

/// A streamed N-Quads quad (subject, predicate, object, optional graph name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedQuad {
    /// Subject: NamedNode or BlankNode.
    pub subject: StreamedTerm,
    /// Predicate: NamedNode only.
    pub predicate: StreamedTerm,
    /// Object: NamedNode, BlankNode, or Literal.
    pub object: StreamedTerm,
    /// Optional named graph: NamedNode or BlankNode.
    pub graph_name: Option<StreamedTerm>,
}

// ============================================================================
// Streaming parser
// ============================================================================

/// A pull-parser for N-Quads format data from any [`Read`] source.
///
/// Implements [`Iterator<Item = Result<StreamedQuad, NQuadsParseError>>`].
/// Blank lines and comment lines (starting with `#`) are skipped transparently.
pub struct NQuadsStreamingParser<R: Read> {
    reader: BufReader<R>,
    line_num: usize,
}

impl<R: Read> NQuadsStreamingParser<R> {
    /// Create a new streaming parser wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_num: 0,
        }
    }

    /// Return the number of lines read so far (including blank/comment lines).
    pub fn lines_read(&self) -> usize {
        self.line_num
    }
}

impl<R: Read> Iterator for NQuadsStreamingParser<R> {
    type Item = Result<StreamedQuad, NQuadsParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Err(e) => return Some(Err(NQuadsParseError::Io(e))),
                Ok(0) => return None, // EOF
                Ok(_) => {
                    self.line_num += 1;
                    let trimmed = line.trim_end_matches('\n').trim_end_matches('\r').trim();
                    match parse_line(trimmed, self.line_num) {
                        Ok(None) => {
                            // blank or comment – read the next line
                            continue;
                        }
                        Ok(Some(quad)) => return Some(Ok(quad)),
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
        }
    }
}
