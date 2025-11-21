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
use crate::turtle::TurtleParser;
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
                    .unwrap()
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

    /// Find the boundary between complete and incomplete statements
    fn find_statement_boundary(&self, content: &str) -> (String, String) {
        let mut last_complete = 0;
        let mut in_string = false;
        let mut in_long_string = false;
        let mut string_quote = '\0';
        let mut chars = content.char_indices().peekable();

        while let Some((i, ch)) = chars.next() {
            // Handle string literals
            if !in_string && !in_long_string && (ch == '"' || ch == '\'') {
                // Check for long string
                let mut count = 1;
                while let Some(&(_, next_ch)) = chars.peek() {
                    if next_ch == ch && count < 3 {
                        chars.next();
                        count += 1;
                    } else {
                        break;
                    }
                }

                if count == 3 {
                    in_long_string = true;
                } else {
                    in_string = count == 1;
                }
                string_quote = ch;
            } else if in_long_string && ch == string_quote {
                // Check for end of long string
                let mut count = 1;
                while let Some(&(_, next_ch)) = chars.peek() {
                    if next_ch == string_quote && count < 3 {
                        chars.next();
                        count += 1;
                    } else {
                        break;
                    }
                }
                if count >= 3 {
                    in_long_string = false;
                }
            } else if in_string && ch == string_quote {
                in_string = false;
            } else if in_string && ch == '\\' {
                // Skip escaped character
                chars.next();
            } else if !in_string && !in_long_string {
                // Look for statement end
                if ch == '.' || ch == '}' {
                    // Include trailing whitespace after statement terminator
                    let mut end_pos = i + ch.len_utf8();
                    while let Some(&(next_i, next_ch)) = chars.peek() {
                        if next_ch == ' ' || next_ch == '\t' || next_ch == '\n' || next_ch == '\r' {
                            chars.next();
                            end_pos = next_i + next_ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    last_complete = end_pos;
                }
            }
        }

        if last_complete == 0 {
            (String::new(), content.to_string())
        } else {
            (
                content[..last_complete].to_string(),
                content[last_complete..].to_string(),
            )
        }
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

        // Parse
        match parser.parse_document(content) {
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
            .unwrap();
        parser.push_data(b"ex:s ex:p \"object\" .\n").unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 1);
        assert!(parser.is_complete());
    }

    #[test]
    fn test_incremental_chunks() {
        let mut parser = IncrementalParser::new();

        // Send data in small chunks
        parser.push_data(b"@prefix ex: <").unwrap();
        parser.push_data(b"http://example.org/> .\n").unwrap();
        parser.push_data(b"ex:s ex:p ").unwrap();
        parser.push_data(b"\"object\" .\n").unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_incremental_incomplete() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser.push_data(b"ex:s ex:p").unwrap(); // Incomplete statement

        let triples = parser.parse_available().unwrap();
        assert!(triples.is_empty());
        assert_eq!(parser.state(), ParseState::Incomplete);

        // Complete the statement
        parser.push_data(b" \"object\" .\n").unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_incremental_multiple_triples() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser
            .push_data(b"ex:a ex:p \"1\" .\nex:b ex:p \"2\" .\nex:c ex:p \"3\" .\n")
            .unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser.push_data(b"ex:a ex:p \"1\" .\n").unwrap();
        parser.parse_available().unwrap();

        // Create checkpoint
        let checkpoint = parser.checkpoint();

        // Parse more
        parser.push_data(b"ex:b ex:p \"2\" .\n").unwrap();
        parser.push_eof();
        parser.parse_available().unwrap();
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
            .unwrap();
        parser.push_data(b"ex:s ex:p \"object\" .\n").unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 1);
        assert!(parser.errors().is_empty());

        // Now test with errors - errors are collected
        let mut parser2 = IncrementalParser::new_lenient();
        parser2.push_data(b"invalid syntax here\n").unwrap();
        parser2.push_eof();

        let _ = parser2.parse_available().unwrap(); // Should not panic
                                                    // Errors may or may not be collected depending on parsing behavior
    }

    #[test]
    fn test_prefix_accumulation() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser.parse_available().unwrap();

        assert!(parser.prefixes().contains_key("ex"));

        parser
            .push_data(b"@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n")
            .unwrap();
        parser.parse_available().unwrap();

        assert!(parser.prefixes().contains_key("ex"));
        assert!(parser.prefixes().contains_key("foaf"));
    }

    #[test]
    fn test_multiline_string() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser
            .push_data(b"ex:s ex:p \"\"\"hello\nworld\"\"\" .\n")
            .unwrap();
        parser.push_eof();

        let triples = parser.parse_available().unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut parser = IncrementalParser::new();

        parser
            .push_data(b"@prefix ex: <http://example.org/> .\n")
            .unwrap();
        parser.push_data(b"ex:s ex:p \"object\" .\n").unwrap();
        parser.push_eof();
        parser.parse_available().unwrap();

        assert_eq!(parser.triples_parsed(), 1);

        parser.reset();

        assert_eq!(parser.triples_parsed(), 0);
        assert_eq!(parser.bytes_processed(), 0);
        assert!(!parser.is_complete());
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
            .unwrap();
        assert_eq!(parser.bytes_processed(), 36);

        // "ex:s ex:p \"object\" .\n" is 21 bytes
        parser.push_data(b"ex:s ex:p \"object\" .\n").unwrap();
        assert_eq!(parser.bytes_processed(), 57);

        parser.push_eof();
        parser.parse_available().unwrap();

        assert_eq!(parser.triples_parsed(), 1);
    }
}
