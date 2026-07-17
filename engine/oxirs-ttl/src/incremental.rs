//! Incremental parsing support for RDF formats
//!
//! This module provides incremental parsing capabilities that allow:
//! - Parsing as bytes arrive (non-blocking)
//! - Resume parsing from checkpoints
//! - Partial document handling
//!
//! # Use Cases
//!
//! - Network streaming where data arrives in chunks
//! - Large file processing with progress tracking
//! - Interactive parsing with partial results
//! - Memory-constrained environments
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::incremental::{IncrementalParser, ParseState};
//!
//! let mut parser = IncrementalParser::new();
//!
//! // Feed data in chunks
//! parser.push_data(b"@prefix ex: <http://example.org/> .\n")?;
//! parser.push_data(b"ex:subject ex:predicate \"object\" .\n")?;
//!
//! // Parse available triples
//! let triples = parser.parse_available()?;
//!
//! // Check if more data is expected
//! if parser.state() == ParseState::Incomplete {
//!     // Wait for more data...
//! }
//! # Ok::<(), oxirs_ttl::error::TurtleParseError>(())
//! ```

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::turtle::{TurtleParser, TurtleParsingContext};
use oxirs_core::model::Triple;
use std::collections::HashMap;

/// State of the incremental parser
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseState {
    /// Ready to receive data
    Ready,
    /// Parser has incomplete data, waiting for more
    Incomplete,
    /// Parser has data ready to parse
    HasData,
    /// Parsing complete (EOF received)
    Complete,
    /// Parser encountered an error
    Error,
}

/// Checkpoint for resuming parsing
#[derive(Debug, Clone)]
pub struct ParseCheckpoint {
    /// Position in the input stream
    pub byte_offset: usize,
    /// Number of triples parsed so far
    pub triple_count: usize,
    /// Current prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI if set
    pub base_iri: Option<String>,
    /// Pending incomplete data
    pub pending_data: Vec<u8>,
    /// State at checkpoint
    pub state: ParseState,
    /// Blank node counter at checkpoint (see `IncrementalParser::blank_node_counter`)
    pub blank_node_counter: usize,
    /// Explicitly-seen blank node labels at checkpoint
    pub explicit_blank_labels: std::collections::HashSet<String>,
}

impl ParseCheckpoint {
    /// Create a new empty checkpoint
    pub fn new() -> Self {
        Self {
            byte_offset: 0,
            triple_count: 0,
            prefixes: HashMap::new(),
            base_iri: None,
            pending_data: Vec::new(),
            state: ParseState::Ready,
            blank_node_counter: 0,
            explicit_blank_labels: std::collections::HashSet::new(),
        }
    }
}

impl Default for ParseCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental parser for RDF formats
///
/// Supports feeding data in chunks and parsing as data becomes available.
/// Maintains state between chunks for seamless continuation.
pub struct IncrementalParser {
    /// Accumulated data buffer
    buffer: Vec<u8>,
    /// Current parser state
    state: ParseState,
    /// Prefix declarations collected so far
    prefixes: HashMap<String, String>,
    /// Base IRI if set
    base_iri: Option<String>,
    /// Total bytes processed
    bytes_processed: usize,
    /// Total triples parsed
    triples_parsed: usize,
    /// Whether in lenient mode
    lenient: bool,
    /// Whether EOF has been signaled
    eof: bool,
    /// Errors collected in lenient mode
    errors: Vec<TurtleParseError>,
    /// Blank node counter carried across `parse_available()` calls.
    ///
    /// Without this, each call would create a brand-new [`TurtleParsingContext`]
    /// starting its counter at zero, so an anonymous blank node (`[...]`) in one
    /// chunk and another in a later chunk would both mint `_:genid-b0` and get
    /// silently unified once the returned triples are merged by the caller.
    blank_node_counter: usize,
    /// Blank node labels explicitly written by the user, carried across calls so
    /// that generated IDs (see `blank_node_counter`) never collide with them
    /// either, no matter which chunk the explicit label appeared in.
    explicit_blank_labels: std::collections::HashSet<String>,
}

impl IncrementalParser {
    /// Create a new incremental parser
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            state: ParseState::Ready,
            prefixes: HashMap::new(),
            base_iri: None,
            bytes_processed: 0,
            triples_parsed: 0,
            lenient: false,
            eof: false,
            errors: Vec::new(),
            blank_node_counter: 0,
            explicit_blank_labels: std::collections::HashSet::new(),
        }
    }

    /// Create a lenient incremental parser
    pub fn new_lenient() -> Self {
        let mut parser = Self::new();
        parser.lenient = true;
        parser
    }

    /// Set lenient mode
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// Get current parser state
    pub fn state(&self) -> ParseState {
        self.state
    }

    /// Get number of bytes processed
    pub fn bytes_processed(&self) -> usize {
        self.bytes_processed
    }

    /// Get number of triples parsed
    pub fn triples_parsed(&self) -> usize {
        self.triples_parsed
    }

    /// Get collected errors (in lenient mode)
    pub fn errors(&self) -> &[TurtleParseError] {
        &self.errors
    }

    /// Clear collected errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Push new data to the parser
    pub fn push_data(&mut self, data: &[u8]) -> TurtleResult<()> {
        if self.eof {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Cannot push data after EOF".to_string(),
                position: TextPosition::new(1, 1, self.bytes_processed),
            }));
        }

        self.buffer.extend_from_slice(data);
        self.bytes_processed += data.len();

        // Update state
        if self.buffer.is_empty() {
            self.state = ParseState::Ready;
        } else {
            self.state = ParseState::HasData;
        }

        Ok(())
    }

    /// Signal end of input
    pub fn push_eof(&mut self) {
        self.eof = true;
        if self.buffer.is_empty() {
            self.state = ParseState::Complete;
        }
    }

    /// Check if parser has data to parse
    pub fn has_data(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Check if parsing is complete
    pub fn is_complete(&self) -> bool {
        self.state == ParseState::Complete
    }

    /// Parse all available complete statements
    ///
    /// Returns triples that were successfully parsed. Incomplete statements
    /// are kept in the buffer for the next call.
    pub fn parse_available(&mut self) -> TurtleResult<Vec<Triple>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Convert buffer to string
        let content = match std::str::from_utf8(&self.buffer) {
            Ok(s) => s.to_string(),
            Err(e) => {
                // Handle partial UTF-8: find valid prefix
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: "Invalid UTF-8 data".to_string(),
                        position: TextPosition::new(1, 1, 0),
                    }));
                }
                std::str::from_utf8(&self.buffer[..valid_up_to])
                    .expect("valid UTF-8")
                    .to_string()
            }
        };

        // Find the last complete statement
        let (parseable, remaining) = self.find_statement_boundary(&content);

        if parseable.is_empty() {
            // No complete statements yet
            if self.eof {
                // At EOF, try to parse whatever we have
                self.state = ParseState::Complete;
                return self.parse_final();
            } else {
                self.state = ParseState::Incomplete;
                return Ok(Vec::new());
            }
        }

        // Parse the complete statements
        let triples = self.parse_content(&parseable)?;
        self.triples_parsed += triples.len();

        // Keep the remaining incomplete data
        self.buffer = remaining.as_bytes().to_vec();

        if self.buffer.is_empty() {
            if self.eof {
                self.state = ParseState::Complete;
            } else {
                self.state = ParseState::Ready;
            }
        } else {
            self.state = ParseState::Incomplete;
        }

        Ok(triples)
    }

    /// Parse remaining data at end of stream
    fn parse_final(&mut self) -> TurtleResult<Vec<Triple>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let content = match std::str::from_utf8(&self.buffer) {
            Ok(s) => s.to_string(),
            Err(_) => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Invalid UTF-8 at end of stream".to_string(),
                    position: TextPosition::new(1, 1, self.bytes_processed),
                }));
            }
        };

        // Try to parse even incomplete data
        let triples = self.parse_content(&content)?;
        self.triples_parsed += triples.len();
        self.buffer.clear();
        self.state = ParseState::Complete;

        Ok(triples)
    }

    /// Find the boundary between complete and incomplete statements.
    ///
    /// Delegates to the shared [`crate::statement_boundary`] scanner so this
    /// crate maintains a single (tested) implementation of the string/long-
    /// string tracking used to recognize a top-level statement terminator.
    fn find_statement_boundary(&self, content: &str) -> (String, String) {
        let (parseable, remaining) =
            crate::statement_boundary::find_last_statement_boundary(content);
        (parseable.to_string(), remaining.to_string())
    }

    /// Parse content string
    fn parse_content(&mut self, content: &str) -> TurtleResult<Vec<Triple>> {
        // Extract new prefix/base declarations FIRST, before parsing
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("@prefix") {
                // Parse @prefix ex: <http://example.org/> .
                let rest = rest.trim();
                if let Some(colon_pos) = rest.find(':') {
                    let prefix = rest[..colon_pos].trim().to_string();
                    let after_colon = rest[colon_pos + 1..].trim();
                    if let Some(iri_start) = after_colon.find('<') {
                        if let Some(iri_end) = after_colon.find('>') {
                            let iri = after_colon[iri_start + 1..iri_end].to_string();
                            self.prefixes.insert(prefix, iri);
                        }
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("@base") {
                // Parse @base <http://example.org/> .
                let rest = rest.trim();
                if let Some(iri_start) = rest.find('<') {
                    if let Some(iri_end) = rest.find('>') {
                        let iri = rest[iri_start + 1..iri_end].to_string();
                        self.base_iri = Some(iri);
                    }
                }
            }
        }

        // Create parser with all collected state (including newly extracted prefixes)
        let mut parser = TurtleParser::new();
        if self.lenient {
            parser.lenient = true;
        }

        for (prefix, iri) in &self.prefixes {
            parser.prefixes.insert(prefix.clone(), iri.clone());
        }

        if let Some(base) = &self.base_iri {
            parser.base_iri = Some(base.clone());
        }

        // Build a parsing context seeded with the blank-node state accumulated
        // across all previous `parse_available()` calls, so that anonymous blank
        // nodes generated while parsing this chunk cannot collide with ones
        // generated for an earlier chunk (see `blank_node_counter`).
        let mut context = TurtleParsingContext::new();
        context.prefixes = parser.prefixes.clone();
        context.base_iri = parser.base_iri.clone();
        context.blank_node_counter = self.blank_node_counter;
        context.explicit_blank_labels = self.explicit_blank_labels.clone();

        // Parse
        let result = parser.parse_document_with_context(content, &mut context);

        // Persist blank-node state regardless of success/failure so that partial
        // progress (e.g. lenient-mode recovery) still avoids future collisions.
        self.blank_node_counter = context.blank_node_counter;
        self.explicit_blank_labels = context.explicit_blank_labels;

        match result {
            Ok(triples) => Ok(triples),
            Err(e) => {
                if self.lenient {
                    self.errors.push(e);
                    self.state = ParseState::Error;
                    Ok(Vec::new())
                } else {
                    self.state = ParseState::Error;
                    Err(e)
                }
            }
        }
    }

    /// Create a checkpoint of current state
    pub fn checkpoint(&self) -> ParseCheckpoint {
        ParseCheckpoint {
            byte_offset: self.bytes_processed,
            triple_count: self.triples_parsed,
            prefixes: self.prefixes.clone(),
            base_iri: self.base_iri.clone(),
            pending_data: self.buffer.clone(),
            state: self.state,
            blank_node_counter: self.blank_node_counter,
            explicit_blank_labels: self.explicit_blank_labels.clone(),
        }
    }

    /// Restore parser state from a checkpoint
    pub fn restore(&mut self, checkpoint: ParseCheckpoint) {
        self.bytes_processed = checkpoint.byte_offset;
        self.triples_parsed = checkpoint.triple_count;
        self.prefixes = checkpoint.prefixes;
        self.base_iri = checkpoint.base_iri;
        self.buffer = checkpoint.pending_data;
        self.state = checkpoint.state;
        self.eof = checkpoint.state == ParseState::Complete;
        self.errors.clear();
        self.blank_node_counter = checkpoint.blank_node_counter;
        self.explicit_blank_labels = checkpoint.explicit_blank_labels;
    }

    /// Reset parser to initial state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = ParseState::Ready;
        self.prefixes.clear();
        self.base_iri = None;
        self.bytes_processed = 0;
        self.triples_parsed = 0;
        self.eof = false;
        self.errors.clear();
        self.blank_node_counter = 0;
        self.explicit_blank_labels.clear();
    }

    /// Get pending data size
    pub fn pending_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get current prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Get base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }
}

impl Default for IncrementalParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_basic() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:s ex:p \"object\" .\n")
            .expect("push data should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
        assert!(parser.is_complete());
    }

    #[test]
    fn test_incremental_chunks() {
        let mut parser = IncrementalParser::new();

        // Send data in small chunks
        parser
            .push_data(b"@prefix ex: <")
            .expect("push data should succeed");
        parser
            .push_data(b"http://example.org/> .\n")
            .expect("push data should succeed");
        parser
            .push_data(b"ex:s ex:p ")
            .expect("push data should succeed");
        parser
            .push_data(b"\"object\" .\n")
            .expect("push data should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_incremental_incomplete() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:s ex:p")
            .expect("push data should succeed"); // Incomplete statement

        let triples = parser.parse_available().expect("parsing should succeed");
        assert!(triples.is_empty());
        assert_eq!(parser.state(), ParseState::Incomplete);

        // Complete the statement
        parser
            .push_data(b" \"object\" .\n")
            .expect("push data should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_incremental_multiple_triples() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:a ex:p \"1\" .\nex:b ex:p \"2\" .\nex:c ex:p \"3\" .\n")
            .expect("operation should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:a ex:p \"1\" .\n")
            .expect("push data should succeed");
        parser.parse_available().expect("parsing should succeed");

        // Create checkpoint
        let checkpoint = parser.checkpoint();

        // Parse more
        parser
            .push_data(b"ex:b ex:p \"2\" .\n")
            .expect("push data should succeed");
        parser.push_eof();
        parser.parse_available().expect("parsing should succeed");
        assert_eq!(parser.triples_parsed(), 2);

        // Restore to checkpoint
        parser.restore(checkpoint);
        assert_eq!(parser.triples_parsed(), 1);
    }

    #[test]
    fn test_lenient_mode() {
        let mut parser = IncrementalParser::new_lenient();

        // In lenient mode, parser collects errors but doesn't skip invalid statements
        // The parser attempts to parse the entire content as one document
        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:s ex:p \"object\" .\n")
            .expect("push data should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
        assert!(parser.errors().is_empty());

        // Now test with errors - errors are collected
        let mut parser2 = IncrementalParser::new_lenient();
        parser2
            .push_data(b"invalid syntax here\n")
            .expect("push data should succeed");
        parser2.push_eof();

        let _ = parser2.parse_available().expect("parsing should succeed"); // Should not panic
                                                                            // Errors may or may not be collected depending on parsing behavior
    }

    #[test]
    fn test_prefix_accumulation() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser.parse_available().expect("parsing should succeed");

        assert!(parser.prefixes().contains_key("ex"));

        parser
            .push_data(b"@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n")
            .expect("operation should succeed");
        parser.parse_available().expect("parsing should succeed");

        assert!(parser.prefixes().contains_key("ex"));
        assert!(parser.prefixes().contains_key("foaf"));
    }

    #[test]
    fn test_multiline_string() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:s ex:p \"\"\"hello\nworld\"\"\" .\n")
            .expect("operation should succeed");
        parser.push_eof();

        let triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        parser
            .push_data(b"ex:s ex:p \"object\" .\n")
            .expect("push data should succeed");
        parser.push_eof();
        parser.parse_available().expect("parsing should succeed");

        assert_eq!(parser.triples_parsed(), 1);

        parser.reset();

        assert_eq!(parser.triples_parsed(), 0);
        assert_eq!(parser.bytes_processed(), 0);
        assert!(!parser.is_complete());
    }

    #[test]
    fn test_blank_node_counter_persists_across_chunks() {
        // Regression test: two chunks, each introducing an anonymous blank node
        // property list, must not reuse the same generated blank node ID — that
        // would silently unify two logically-distinct blank nodes once the
        // triples returned by successive `parse_available()` calls are merged.
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("push data should succeed");
        parser
            .push_data(b"ex:s1 ex:p [ ex:q \"1\" ] .\n")
            .expect("push data should succeed");

        let first_triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(first_triples.len(), 2);
        let first_blank = first_triples
            .iter()
            .find_map(|t| match t.subject() {
                oxirs_core::model::Subject::BlankNode(b) => Some(b.clone()),
                _ => None,
            })
            .expect("first chunk should contain a blank node subject");

        parser
            .push_data(b"ex:s2 ex:p [ ex:q \"2\" ] .\n")
            .expect("push data should succeed");
        parser.push_eof();

        let second_triples = parser.parse_available().expect("parsing should succeed");
        assert_eq!(second_triples.len(), 2);
        let second_blank = second_triples
            .iter()
            .find_map(|t| match t.subject() {
                oxirs_core::model::Subject::BlankNode(b) => Some(b.clone()),
                _ => None,
            })
            .expect("second chunk should contain a blank node subject");

        assert_ne!(
            first_blank, second_blank,
            "blank nodes generated in different chunks must not collide"
        );
    }

    #[test]
    fn test_eof_error() {
        let mut parser = IncrementalParser::new();
        parser.push_eof();

        let result = parser.push_data(b"data after eof");
        assert!(result.is_err());
    }

    #[test]
    fn test_progress_tracking() {
        let mut parser = IncrementalParser::new();

        // "@prefix ex: <http://example.org/> .\n" is 36 bytes
        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .expect("operation should succeed");
        assert_eq!(parser.bytes_processed(), 36);

        // "ex:s ex:p \"object\" .\n" is 21 bytes
        parser
            .push_data(b"ex:s ex:p \"object\" .\n")
            .expect("push data should succeed");
        assert_eq!(parser.bytes_processed(), 57);

        parser.push_eof();
        parser.parse_available().expect("parsing should succeed");

        assert_eq!(parser.triples_parsed(), 1);
    }
}
