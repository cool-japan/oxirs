//! Async I/O support for RDF parsing using Tokio
//!
//! This module provides async parsing capabilities for non-blocking I/O operations.

#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};

use crate::error::{TurtleParseError, TurtleResult};
use crate::toolkit::Parser;
use oxirs_core::model::Triple;

/// Async parser trait for non-blocking RDF parsing
#[cfg(feature = "async-tokio")]
#[async_trait::async_trait]
pub trait AsyncParser: Send + Sync {
    /// Parse a document from an async reader
    async fn parse_async<R: AsyncRead + Unpin + Send>(
        &self,
        reader: R,
    ) -> TurtleResult<Vec<Triple>>;

    /// Parse a document from a string
    async fn parse_str_async(&self, content: &str) -> TurtleResult<Vec<Triple>>;
}

/// Async streaming parser for processing large files
#[cfg(feature = "async-tokio")]
pub struct AsyncStreamingParser<R: AsyncBufRead + Unpin> {
    reader: R,
    buffer: String,
    prefix_declarations: String,
    triples_parsed: usize,
    bytes_read: usize,
    batch_size: usize,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin + Send> AsyncStreamingParser<BufReader<R>> {
    /// Create a new async streaming parser
    pub fn new(reader: R) -> Self {
        Self::with_batch_size(reader, 10_000)
    }

    /// Create a streaming parser with a specific batch size
    pub fn with_batch_size(reader: R, batch_size: usize) -> Self {
        Self {
            reader: BufReader::new(reader),
            buffer: String::new(),
            prefix_declarations: String::new(),
            triples_parsed: 0,
            bytes_read: 0,
            batch_size,
        }
    }
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncBufRead + Unpin> AsyncStreamingParser<R> {
    /// Create from an existing async BufRead
    pub fn from_buf_reader(reader: R, batch_size: usize) -> Self {
        Self {
            reader,
            buffer: String::new(),
            prefix_declarations: String::new(),
            triples_parsed: 0,
            bytes_read: 0,
            batch_size,
        }
    }

    /// Get the number of triples parsed so far
    pub fn triples_parsed(&self) -> usize {
        self.triples_parsed
    }

    /// Get the number of bytes read so far
    pub fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Parse the next batch of triples asynchronously
    pub async fn next_batch_async(&mut self) -> TurtleResult<Option<Vec<Triple>>> {
        use crate::turtle::TurtleParser;

        // Read up to batch_size lines
        self.buffer.clear();
        let mut lines_read = 0;
        let target_lines = self.batch_size / 10; // Rough estimate: ~10 triples per line

        while lines_read < target_lines {
            let mut line = String::new();
            match self.reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(n) => {
                    self.bytes_read += n;
                    self.buffer.push_str(&line);
                    lines_read += 1;
                }
                Err(e) => return Err(TurtleParseError::io(e)),
            }
        }

        if self.buffer.is_empty() {
            return Ok(None); // EOF
        }

        // Extract prefix declarations
        for line in self.buffer.lines() {
            let trimmed = line.trim();
            if (trimmed.starts_with("@prefix") || trimmed.starts_with("@base"))
                && !self.prefix_declarations.contains(trimmed)
            {
                self.prefix_declarations.push_str(trimmed);
                self.prefix_declarations.push('\n');
            }
        }

        // Parse the complete document (prefixes + current batch)
        let document = format!("{}{}", self.prefix_declarations, self.buffer);

        let parser = TurtleParser::new();
        match parser.parse_document(&document) {
            Ok(triples) => {
                self.triples_parsed += triples.len();
                Ok(Some(triples))
            }
            Err(e) => Err(e),
        }
    }

    /// Stream all triples with a callback
    pub async fn stream_with_callback<F>(&mut self, mut callback: F) -> TurtleResult<usize>
    where
        F: FnMut(Vec<Triple>),
    {
        let mut total = 0;

        while let Some(batch) = self.next_batch_async().await? {
            total += batch.len();
            callback(batch);
        }

        Ok(total)
    }

    /// Collect all triples into a vector
    pub async fn collect_all_async(&mut self) -> TurtleResult<Vec<Triple>> {
        let mut all_triples = Vec::new();

        while let Some(batch) = self.next_batch_async().await? {
            all_triples.extend(batch);
        }

        Ok(all_triples)
    }
}

/// Async Turtle parser implementation
#[cfg(feature = "async-tokio")]
pub struct AsyncTurtleParser {
    lenient: bool,
}

#[cfg(feature = "async-tokio")]
impl AsyncTurtleParser {
    /// Create a new async Turtle parser
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Create a new lenient async Turtle parser
    pub fn new_lenient() -> Self {
        Self { lenient: true }
    }
}

#[cfg(feature = "async-tokio")]
impl Default for AsyncTurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async-tokio")]
#[async_trait::async_trait]
impl AsyncParser for AsyncTurtleParser {
    async fn parse_async<R: AsyncRead + Unpin + Send>(
        &self,
        mut reader: R,
    ) -> TurtleResult<Vec<Triple>> {
        use crate::turtle::TurtleParser;

        // Read entire content
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .await
            .map_err(TurtleParseError::io)?;

        // Parse using sync parser
        let parser = if self.lenient {
            TurtleParser::new_lenient()
        } else {
            TurtleParser::new()
        };

        parser.parse_document(&content)
    }

    async fn parse_str_async(&self, content: &str) -> TurtleResult<Vec<Triple>> {
        use crate::turtle::TurtleParser;

        let parser = if self.lenient {
            TurtleParser::new_lenient()
        } else {
            TurtleParser::new()
        };

        parser.parse_document(content)
    }
}

/// Async N-Triples parser implementation
#[cfg(feature = "async-tokio")]
pub struct AsyncNTriplesParser {
    lenient: bool,
}

#[cfg(feature = "async-tokio")]
impl AsyncNTriplesParser {
    /// Create a new async N-Triples parser
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Create a new lenient async N-Triples parser
    pub fn new_lenient() -> Self {
        Self { lenient: true }
    }

    /// Parse lines from an async reader
    pub async fn parse_lines<R: AsyncBufRead + Unpin>(
        &self,
        reader: R,
    ) -> TurtleResult<Vec<Triple>> {
        use crate::ntriples::NTriplesParser;

        let mut lines = reader.lines();
        let mut all_triples = Vec::new();
        let parser = NTriplesParser::new();
        let mut line_number = 0;

        while let Some(line) = lines.next_line().await.map_err(TurtleParseError::io)? {
            line_number += 1;

            match parser.parse_line(&line, line_number) {
                Ok(Some(triple)) => all_triples.push(triple),
                Ok(None) => {} // Empty line or comment
                Err(e) if self.lenient => {
                    eprintln!("Warning: Parse error on line {}: {}", line_number, e);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(all_triples)
    }
}

#[cfg(feature = "async-tokio")]
impl Default for AsyncNTriplesParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "async-tokio")]
#[async_trait::async_trait]
impl AsyncParser for AsyncNTriplesParser {
    async fn parse_async<R: AsyncRead + Unpin + Send>(
        &self,
        reader: R,
    ) -> TurtleResult<Vec<Triple>> {
        self.parse_lines(BufReader::new(reader)).await
    }

    async fn parse_str_async(&self, content: &str) -> TurtleResult<Vec<Triple>> {
        use crate::ntriples::NTriplesParser;
        use std::io::Cursor;

        let parser = NTriplesParser::new();
        let content_owned = content.to_string();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(content_owned)).collect();
        result
    }
}

#[cfg(not(feature = "async-tokio"))]
compile_error!("Async parsing requires the 'async-tokio' feature to be enabled");

#[cfg(all(test, feature = "async-tokio"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_turtle_parser() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        "#;

        let parser = AsyncTurtleParser::new();
        let result = parser.parse_str_async(turtle).await;

        assert!(result.is_ok());
        let triples = result.expect("result should be Ok");
        assert_eq!(triples.len(), 2);
    }

    #[tokio::test]
    async fn test_async_ntriples_parser() {
        let nt = "<http://example.org/s> <http://example.org/p> \"o\" .\n\
                  <http://example.org/s2> <http://example.org/p2> \"o2\" .";

        let parser = AsyncNTriplesParser::new();
        let result = parser.parse_str_async(nt).await;

        assert!(result.is_ok());
        let triples = result.expect("result should be Ok");
        assert_eq!(triples.len(), 2);
    }

    #[tokio::test]
    async fn test_async_streaming_parser() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
            ex:charlie ex:name "Charlie" .
        "#;

        let cursor = std::io::Cursor::new(turtle);
        let async_reader = tokio::io::BufReader::new(cursor);
        let mut parser = AsyncStreamingParser::from_buf_reader(async_reader, 10);

        let triples = parser.collect_all_async().await;

        assert!(triples.is_ok());
        assert_eq!(triples.expect("operation should succeed").len(), 3);
    }

    #[tokio::test]
    async fn test_async_streaming_with_callback() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        "#;

        let cursor = std::io::Cursor::new(turtle);
        let async_reader = tokio::io::BufReader::new(cursor);
        let mut parser = AsyncStreamingParser::from_buf_reader(async_reader, 10);

        let mut count = 0;
        let result = parser
            .stream_with_callback(|batch| {
                count += batch.len();
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(count, 2);
    }
}
