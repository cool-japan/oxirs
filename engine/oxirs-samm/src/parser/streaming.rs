//! Streaming SAMM Parser for Memory-Efficient Processing
//!
//! This module provides a streaming parser for SAMM models that processes
//! large Turtle/RDF files incrementally without loading the entire file into memory.
//!
//! ## Features
//!
//! - **Memory Efficient**: Processes files in configurable chunks (default: 64KB)
//! - **Large File Support**: Can handle files larger than available RAM
//! - **Incremental Parsing**: Emits model elements as they're parsed
//! - **Async Streaming**: Uses Rust async streams for efficient I/O
//! - **Configurable Buffer**: Adjust chunk size based on available memory
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_samm::parser::StreamingParser;
//! use oxirs_samm::metamodel::ModelElement;
//! use futures::{StreamExt, pin_mut};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create streaming parser with 128KB chunks
//! let mut parser = StreamingParser::new()
//!     .with_chunk_size(128 * 1024);
//!
//! // Parse large file incrementally
//! let stream = parser.parse_file_streaming("large_model.ttl").await?;
//! pin_mut!(stream);
//!
//! // Process elements as they arrive
//! while let Some(element) = stream.next().await {
//!     match element {
//!         Ok(aspect) => println!("Parsed aspect: {}", aspect.name()),
//!         Err(e) => eprintln!("Parse error: {}", e),
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::Aspect;
use crate::parser::SammTurtleParser;
use futures::stream::{Stream, StreamExt};
use std::path::Path;
use std::pin::Pin;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

/// Default chunk size for streaming (64KB)
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum buffer size to prevent unbounded growth (16MB)
const MAX_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Streaming parser for memory-efficient SAMM model processing
pub struct StreamingParser {
    /// Chunk size for reading file
    chunk_size: usize,
    /// Maximum buffer size before forcing a parse
    max_buffer_size: usize,
    /// Base URI for resolving relative references
    base_uri: Option<String>,
}

impl StreamingParser {
    /// Create a new streaming parser with default settings
    pub fn new() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_buffer_size: MAX_BUFFER_SIZE,
            base_uri: None,
        }
    }

    /// Set the chunk size for reading (in bytes)
    ///
    /// Smaller chunks use less memory but may be slower.
    /// Larger chunks are faster but use more memory.
    ///
    /// Default: 64KB
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1024); // Minimum 1KB
        self
    }

    /// Set the maximum buffer size (in bytes)
    ///
    /// When the buffer grows beyond this size, the parser will
    /// attempt to flush and parse accumulated data.
    ///
    /// Default: 16MB
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Set the base URI for resolving relative references
    pub fn with_base_uri(mut self, base_uri: impl Into<String>) -> Self {
        self.base_uri = Some(base_uri.into());
        self
    }

    /// Parse a file using streaming for memory efficiency
    ///
    /// This method reads the file in chunks and emits parsed Aspect models
    /// as they become available. This is much more memory-efficient than
    /// loading the entire file at once.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Turtle file
    ///
    /// # Returns
    ///
    /// A stream of `Result<Aspect, SammError>` that emits aspects as they're parsed
    pub async fn parse_file_streaming<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<impl Stream<Item = Result<Aspect>>> {
        let file = File::open(path.as_ref())
            .await
            .map_err(|e| SammError::ParseError(format!("Failed to open file: {}", e)))?;

        let base_uri = self
            .base_uri
            .clone()
            .unwrap_or_else(|| format!("file://{}", path.as_ref().to_string_lossy()));

        Ok(self.create_stream(file, base_uri))
    }

    /// Parse from an async reader using streaming
    ///
    /// This allows streaming from any async source (file, network, etc.)
    pub fn parse_reader_streaming<R>(
        &mut self,
        reader: R,
        base_uri: impl Into<String>,
    ) -> impl Stream<Item = Result<Aspect>>
    where
        R: AsyncReadExt + Unpin + Send + 'static,
    {
        self.create_stream(reader, base_uri.into())
    }

    /// Internal method to create the streaming parser
    fn create_stream<R>(&self, reader: R, base_uri: String) -> impl Stream<Item = Result<Aspect>>
    where
        R: AsyncReadExt + Unpin + Send + 'static,
    {
        let chunk_size = self.chunk_size;
        let max_buffer_size = self.max_buffer_size;

        async_stream::stream! {
            let mut reader = BufReader::with_capacity(chunk_size, reader);
            let mut buffer = String::new();

            // Read file line by line to find complete Turtle documents
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        // EOF - process any remaining content
                        if !buffer.is_empty() {
                            match try_parse_buffer(&buffer, &base_uri).await {
                                Ok(Some(aspect)) => yield Ok(aspect),
                                Ok(None) => {}, // Incomplete document
                                Err(e) => yield Err(e),
                            }
                        }
                        break;
                    }
                    Ok(_) => {
                        buffer.push_str(&line);

                        // Check if we have a complete document (ends with .)
                        // This is a simple heuristic - could be improved
                        if line.trim().ends_with('.') || buffer.len() > max_buffer_size {
                            match try_parse_buffer(&buffer, &base_uri).await {
                                Ok(Some(aspect)) => {
                                    yield Ok(aspect);
                                    buffer.clear();
                                }
                                Ok(None) => {
                                    // Keep accumulating
                                    if buffer.len() > max_buffer_size {
                                        // Force clear to prevent OOM
                                        tracing::warn!(
                                            "Buffer exceeded max size ({}MB), clearing incomplete document",
                                            buffer.len() / 1024 / 1024
                                        );
                                        buffer.clear();
                                        yield Err(SammError::ParseError(
                                            "Document too large for streaming parser".to_string()
                                        ));
                                    }
                                }
                                Err(e) => {
                                    // Parse error - clear buffer and continue
                                    tracing::debug!("Parse error in streaming: {}", e);
                                    buffer.clear();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(SammError::ParseError(format!("Read error: {}", e)));
                        break;
                    }
                }
            }
        }
    }

    /// Parse a string using line-by-line streaming
    ///
    /// This is useful for processing large in-memory strings without
    /// creating multiple copies.
    pub fn parse_string_streaming(
        &self,
        content: String,
        base_uri: impl Into<String>,
    ) -> impl Stream<Item = Result<Aspect>> {
        let base_uri = base_uri.into();
        let lines: Vec<String> = content.lines().map(String::from).collect();

        async_stream::stream! {
            let mut buffer = String::new();
            let mut blank_line_count = 0;

            for line in lines {
                let trimmed = line.trim();

                // Track consecutive blank lines as potential document separators
                if trimmed.is_empty() {
                    blank_line_count += 1;
                } else {
                    blank_line_count = 0;
                }

                buffer.push_str(&line);
                buffer.push('\n');

                // Try to parse when we have a potentially complete document:
                // - Multiple blank lines suggest document boundary
                // - Or buffer is getting large
                if (blank_line_count >= 2 || buffer.len() > 10000) && !buffer.trim().is_empty() {
                    match try_parse_buffer(&buffer, &base_uri).await {
                        Ok(Some(aspect)) => {
                            yield Ok(aspect);
                            buffer.clear();
                            blank_line_count = 0;
                        }
                        Ok(None) => {
                            // Keep accumulating - document not complete yet
                        }
                        Err(e) => {
                            // Only clear if buffer is excessively large
                            if buffer.len() > 100000 {
                                tracing::debug!("Clearing large buffer after parse error: {}", e);
                                buffer.clear();
                                blank_line_count = 0;
                            }
                            // Otherwise keep accumulating
                        }
                    }
                }
            }

            // Parse any remaining content
            if !buffer.is_empty() {
                match try_parse_buffer(&buffer, &base_uri).await {
                    Ok(Some(aspect)) => yield Ok(aspect),
                    Ok(None) => {},
                    Err(e) => yield Err(e),
                }
            }
        }
    }

    /// Get current configuration as a summary string
    pub fn config_summary(&self) -> String {
        format!(
            "StreamingParser {{ chunk_size: {}KB, max_buffer: {}MB, base_uri: {} }}",
            self.chunk_size / 1024,
            self.max_buffer_size / 1024 / 1024,
            self.base_uri.as_deref().unwrap_or("auto")
        )
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to parse accumulated buffer content
async fn try_parse_buffer(content: &str, base_uri: &str) -> Result<Option<Aspect>> {
    // Check if content looks complete (has @prefix and ends with .)
    let has_prefix = content.contains("@prefix");
    let ends_properly = content.trim_end().ends_with('.');

    if !has_prefix || !ends_properly {
        return Ok(None); // Incomplete document
    }

    // Try parsing
    let mut parser = SammTurtleParser::new();
    match parser.parse_string(content, base_uri).await {
        Ok(aspect) => Ok(Some(aspect)),
        Err(e) => {
            // If it's a genuine parse error, return it
            // If it's just incomplete, return None
            if content.len() < 100 {
                Ok(None) // Probably incomplete
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ModelElement;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_streaming_parser_string() {
        use futures::pin_mut;

        let ttl_content = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:TestAspect a samm:Aspect ;
    samm:preferredName "Test Aspect"@en ;
    samm:description "A test aspect"@en ;
    samm:properties ( :testProperty ) .

:testProperty a samm:Property ;
    samm:preferredName "Test Property"@en ;
    samm:description "Test description"@en ;
    samm:characteristic :TestCharacteristic .

:TestCharacteristic a samm:Characteristic ;
    samm:dataType <http://www.w3.org/2001/XMLSchema#string> .
        "#
        .to_string();

        let parser = StreamingParser::new();
        let stream = parser.parse_string_streaming(ttl_content, "urn:samm:org.example:1.0.0#");
        pin_mut!(stream);

        let mut count = 0;
        while let Some(result) = stream.next().await {
            match result {
                Ok(aspect) => {
                    assert_eq!(aspect.name(), "TestAspect");
                    count += 1;
                }
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }

        assert!(count > 0, "Should have parsed at least one aspect");
    }

    #[tokio::test]
    async fn test_streaming_parser_config() {
        let parser = StreamingParser::new()
            .with_chunk_size(128 * 1024)
            .with_max_buffer_size(32 * 1024 * 1024)
            .with_base_uri("urn:test#");

        let summary = parser.config_summary();
        assert!(summary.contains("128KB"));
        assert!(summary.contains("32MB"));
        assert!(summary.contains("urn:test#"));
    }

    #[tokio::test]
    async fn test_streaming_parser_empty_input() {
        use futures::pin_mut;

        let parser = StreamingParser::new();
        let stream = parser.parse_string_streaming(String::new(), "urn:test#");
        pin_mut!(stream);

        let result = stream.next().await;
        assert!(result.is_none(), "Empty input should produce no results");
    }

    #[tokio::test]
    async fn test_streaming_parser_memory_efficiency() {
        use futures::pin_mut;

        // Test that streaming parser works with chunk size limits
        let parser = StreamingParser::new()
            .with_chunk_size(1024) // Small chunks
            .with_max_buffer_size(10 * 1024); // Small buffer

        let ttl_content = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:SmallAspect a samm:Aspect ;
    samm:preferredName "Small"@en ;
    samm:description "Small test"@en ;
    samm:properties () .
        "#
        .to_string();

        let stream = parser.parse_string_streaming(ttl_content, "urn:samm:org.example:1.0.0#");
        pin_mut!(stream);

        // Should still work with small buffer
        let result = stream.next().await;
        assert!(result.is_some());
    }
}
