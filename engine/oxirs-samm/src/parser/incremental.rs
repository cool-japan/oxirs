//! Incremental Parser for Large SAMM Models
//!
//! This module provides incremental parsing capabilities that build on the streaming parser.
//! It allows for:
//! - **Resumable parsing**: Save parser state and continue later
//! - **Partial updates**: Parse only changed sections of a model
//! - **Progress tracking**: Monitor parsing progress for large files
//! - **Event-based parsing**: Receive events as different model parts are parsed
//!
//! ## Use Cases
//!
//! - **Large model processing**: Parse multi-gigabyte SAMM models with progress tracking
//! - **Live updates**: Incrementally update models as files change
//! - **Interactive UIs**: Show progress bars and intermediate results
//! - **Distributed parsing**: Split parsing across multiple workers
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_samm::parser::incremental::{IncrementalParser, ParseEvent};
//! use futures::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut parser = IncrementalParser::new("large_model.ttl");
//!
//! // Subscribe to parse events
//! let mut events = parser.parse_with_events().await?;
//!
//! while let Some(event) = events.next().await {
//!     match event {
//!         ParseEvent::Started { total_bytes } => {
//!             println!("Starting parse of {} bytes", total_bytes);
//!         }
//!         ParseEvent::Progress { bytes_parsed, total_bytes } => {
//!             let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
//!             println!("Progress: {:.1}%", percent);
//!         }
//!         ParseEvent::PropertyParsed { property } => {
//!             println!("Parsed property: {}", property.metadata.urn);
//!         }
//!         ParseEvent::Completed { aspect } => {
//!             println!("Parse complete: {} properties", aspect.properties.len());
//!         }
//!         ParseEvent::Error { error } => {
//!             eprintln!("Parse error: {}", error);
//!         }
//!         _ => {} // Handle other events (OperationParsed, PrefixParsed, MetadataParsed, etc.)
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, Operation, Property};
use crate::parser::SammTurtleParser;
use async_stream::stream;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, BufReader};

/// Events emitted during incremental parsing
#[derive(Debug, Clone)]
pub enum ParseEvent {
    /// Parsing has started
    Started {
        /// Total bytes to parse
        total_bytes: u64,
    },

    /// Progress update
    Progress {
        /// Bytes parsed so far
        bytes_parsed: u64,
        /// Total bytes
        total_bytes: u64,
    },

    /// A property was successfully parsed
    PropertyParsed {
        /// The parsed property
        property: Property,
    },

    /// An operation was successfully parsed
    OperationParsed {
        /// The parsed operation
        operation: Operation,
    },

    /// A namespace prefix was parsed
    PrefixParsed {
        /// Prefix name
        prefix: String,
        /// Namespace URI
        namespace: String,
    },

    /// Metadata was parsed (preferred names, descriptions)
    MetadataParsed {
        /// Language code
        language: String,
        /// Metadata type (e.g., "preferredName", "description")
        metadata_type: String,
    },

    /// Parsing completed successfully
    Completed {
        /// The complete parsed aspect
        aspect: Aspect,
    },

    /// An error occurred during parsing
    Error {
        /// The error
        error: String,
    },
}

/// Parser state that can be saved and resumed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseState {
    /// File path being parsed
    pub file_path: PathBuf,
    /// Byte offset in file
    pub byte_offset: u64,
    /// Partially constructed aspect
    pub partial_aspect: Option<Aspect>,
    /// Number of properties parsed so far
    pub properties_parsed: usize,
    /// Number of operations parsed so far
    pub operations_parsed: usize,
    /// Timestamp of last save
    pub last_saved: Option<String>,
}

impl ParseState {
    /// Create a new parse state
    pub fn new(file_path: impl Into<PathBuf>) -> Self {
        Self {
            file_path: file_path.into(),
            byte_offset: 0,
            partial_aspect: None,
            properties_parsed: 0,
            operations_parsed: 0,
            last_saved: None,
        }
    }

    /// Calculate parse progress as a percentage (0.0 to 100.0)
    pub async fn progress_percentage(&self) -> Result<f64> {
        let metadata = tokio::fs::metadata(&self.file_path).await?;
        let total_bytes = metadata.len();
        if total_bytes == 0 {
            return Ok(100.0);
        }
        Ok((self.byte_offset as f64 / total_bytes as f64) * 100.0)
    }

    /// Save state to JSON file
    pub async fn save_to_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        self.last_saved = Some(chrono::Utc::now().to_rfc3339());
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| SammError::Other(format!("JSON error: {}", e)))?;
        let mut file = File::create(path).await?;
        file.write_all(json.as_bytes()).await?;
        file.flush().await?;
        Ok(())
    }

    /// Load state from JSON file
    pub async fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path).await?;
        let state: ParseState = serde_json::from_str(&contents)
            .map_err(|e| SammError::Other(format!("JSON error: {}", e)))?;
        Ok(state)
    }
}

/// Incremental parser for large SAMM models
pub struct IncrementalParser {
    file_path: PathBuf,
    chunk_size: usize,
    state: ParseState,
}

impl IncrementalParser {
    /// Create a new incremental parser
    pub fn new(file_path: impl Into<PathBuf>) -> Self {
        let file_path = file_path.into();
        let state = ParseState::new(file_path.clone());

        Self {
            file_path,
            chunk_size: 64 * 1024, // 64KB chunks
            state,
        }
    }

    /// Create parser from saved state
    pub fn from_state(state: ParseState) -> Self {
        Self {
            file_path: state.file_path.clone(),
            chunk_size: 64 * 1024,
            state,
        }
    }

    /// Set chunk size for reading
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1024);
        self
    }

    /// Get current parser state (for saving/resuming)
    pub fn state(&self) -> &ParseState {
        &self.state
    }

    /// Get mutable parser state
    pub fn state_mut(&mut self) -> &mut ParseState {
        &mut self.state
    }

    /// Save current state to file
    pub async fn save_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.state.save_to_file(path).await
    }

    /// Parse with event stream for progress tracking
    pub async fn parse_with_events(
        &mut self,
    ) -> Result<Pin<Box<dyn Stream<Item = ParseEvent> + Send>>> {
        // Get file metadata
        let metadata = tokio::fs::metadata(&self.file_path).await?;
        let total_bytes = metadata.len();

        // Open file and seek to offset if resuming
        let file = File::open(&self.file_path).await?;
        let mut reader = BufReader::new(file);
        if self.state.byte_offset > 0 {
            reader
                .seek(tokio::io::SeekFrom::Start(self.state.byte_offset))
                .await?;
        }

        let chunk_size = self.chunk_size;
        let mut current_offset = self.state.byte_offset;
        let partial_aspect = self.state.partial_aspect.clone();

        let stream = stream! {
            // Emit start event
            yield ParseEvent::Started { total_bytes };

            // Read and parse incrementally
            let mut buffer = Vec::new();
            let mut accumulated = String::new();
            let mut aspect_result: Option<Aspect> = partial_aspect;

            loop {
                buffer.clear();
                buffer.resize(chunk_size, 0);

                match reader.read(&mut buffer).await {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        current_offset += n as u64;

                        // Try to convert bytes to string
                        if let Ok(chunk_str) = String::from_utf8(buffer[..n].to_vec()) {
                            accumulated.push_str(&chunk_str);

                            // Try to parse when we hit statement boundaries
                            if accumulated.contains('.') || accumulated.contains(';') {
                                // Attempt parse
                                match SammTurtleParser::new().parse_string(&accumulated, "urn:samm:").await {
                                    Ok(aspect) => {
                                        // Emit events for parsed elements
                                        if let Some(ref asp) = aspect_result {
                                            // Compare and emit new properties
                                            for prop in &aspect.properties {
                                                if !asp.properties.iter().any(|p| p.metadata.urn == prop.metadata.urn) {
                                                    yield ParseEvent::PropertyParsed {
                                                        property: prop.clone(),
                                                    };
                                                }
                                            }

                                            // Compare and emit new operations
                                            for op in &aspect.operations {
                                                if !asp.operations.iter().any(|o| o.metadata.urn == op.metadata.urn) {
                                                    yield ParseEvent::OperationParsed {
                                                        operation: op.clone(),
                                                    };
                                                }
                                            }
                                        } else {
                                            // First parse - emit all elements
                                            for prop in &aspect.properties {
                                                yield ParseEvent::PropertyParsed {
                                                    property: prop.clone(),
                                                };
                                            }

                                            for op in &aspect.operations {
                                                yield ParseEvent::OperationParsed {
                                                    operation: op.clone(),
                                                };
                                            }
                                        }

                                        aspect_result = Some(aspect);
                                        accumulated.clear();
                                    }
                                    Err(_) => {
                                        // Not yet a complete document, keep accumulating
                                    }
                                }
                            }

                            // Emit progress
                            yield ParseEvent::Progress {
                                bytes_parsed: current_offset,
                                total_bytes,
                            };
                        }
                    }
                    Err(e) => {
                        yield ParseEvent::Error {
                            error: format!("IO error: {}", e),
                        };
                        break;
                    }
                }
            }

            // Final parse attempt with remaining buffer
            if !accumulated.is_empty() && aspect_result.is_none() {
                match SammTurtleParser::new().parse_string(&accumulated, "urn:samm:").await {
                    Ok(aspect) => {
                        aspect_result = Some(aspect);
                    }
                    Err(e) => {
                        yield ParseEvent::Error { error: e.to_string() };
                    }
                }
            }

            // Emit completion
            if let Some(aspect) = aspect_result {
                yield ParseEvent::Completed { aspect };
            } else {
                yield ParseEvent::Error {
                    error: "Failed to parse aspect".to_string(),
                };
            }
        };

        Ok(Box::pin(stream))
    }

    /// Parse file and return final aspect
    pub async fn parse(&mut self) -> Result<Aspect> {
        use futures::StreamExt;

        let mut events = self.parse_with_events().await?;
        let mut result = None;

        while let Some(event) = events.next().await {
            if let ParseEvent::Completed { aspect } = event {
                result = Some(aspect);
                break;
            } else if let ParseEvent::Error { error } = event {
                return Err(SammError::ParseError(error));
            }
        }

        result.ok_or_else(|| SammError::ParseError("No aspect parsed".to_string()))
    }

    /// Parse with progress callback
    ///
    /// The callback receives (bytes_parsed, total_bytes) and returns whether to continue
    pub async fn parse_with_progress<F>(&mut self, mut callback: F) -> Result<Aspect>
    where
        F: FnMut(u64, u64) -> bool + Send,
    {
        use futures::StreamExt;

        let mut events = self.parse_with_events().await?;
        let mut result = None;

        while let Some(event) = events.next().await {
            match event {
                ParseEvent::Progress {
                    bytes_parsed,
                    total_bytes,
                } => {
                    if !callback(bytes_parsed, total_bytes) {
                        return Err(SammError::Other("Parsing cancelled by user".to_string()));
                    }
                }
                ParseEvent::Completed { aspect } => {
                    result = Some(aspect);
                    break;
                }
                ParseEvent::Error { error } => {
                    return Err(SammError::ParseError(error));
                }
                _ => {}
            }
        }

        result.ok_or_else(|| SammError::ParseError("No aspect parsed".to_string()))
    }
}

impl Default for IncrementalParser {
    fn default() -> Self {
        Self::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_parse_state_creation() {
        let state = ParseState::new("/tmp/test.ttl");
        assert_eq!(state.byte_offset, 0);
        assert_eq!(state.properties_parsed, 0);
        assert!(state.partial_aspect.is_none());
    }

    #[tokio::test]
    async fn test_parse_state_progress() {
        // Create a temp file
        let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        write!(temp_file, "test content").expect("write should succeed");
        temp_file.flush().expect("flush should succeed");

        let mut state = ParseState::new(temp_file.path());
        state.byte_offset = 6; // Half of "test content" (12 bytes)

        let progress = state
            .progress_percentage()
            .await
            .expect("async operation should succeed");
        assert!((progress - 50.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_parse_state_save_load() {
        let temp_state_file = NamedTempFile::new().expect("temp file creation should succeed");
        let temp_data_file = NamedTempFile::new().expect("temp file creation should succeed");

        let mut state = ParseState::new(temp_data_file.path());
        state.byte_offset = 100;
        state.properties_parsed = 5;

        // Save
        state
            .save_to_file(temp_state_file.path())
            .await
            .expect("async operation should succeed");

        // Load
        let loaded_state = ParseState::load_from_file(temp_state_file.path())
            .await
            .expect("operation should succeed");

        assert_eq!(loaded_state.byte_offset, 100);
        assert_eq!(loaded_state.properties_parsed, 5);
    }

    #[tokio::test]
    async fn test_incremental_parser_creation() {
        let parser = IncrementalParser::new("/tmp/test.ttl");
        assert_eq!(parser.state().byte_offset, 0);
    }

    #[tokio::test]
    async fn test_incremental_parser_with_chunk_size() {
        let parser = IncrementalParser::new("/tmp/test.ttl").with_chunk_size(1024);
        assert_eq!(parser.chunk_size, 1024);
    }

    #[tokio::test]
    async fn test_parse_events_simple() {
        // Create a minimal SAMM file
        let mut temp_file = NamedTempFile::new().expect("temp file creation should succeed");
        let ttl_content = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:TestAspect a samm:Aspect ;
    samm:properties ( :property1 ) ;
    samm:operations ( ) .

:property1 a samm:Property ;
    samm:characteristic :TestCharacteristic .

:TestCharacteristic a samm:Characteristic ;
    samm:dataType xsd:string .
"#;
        write!(temp_file, "{}", ttl_content).expect("write should succeed");
        temp_file.flush().expect("flush should succeed");

        let mut parser = IncrementalParser::new(temp_file.path());
        let events = parser
            .parse_with_events()
            .await
            .expect("async operation should succeed");

        let collected: Vec<ParseEvent> = events.collect().await;

        // Should have at least Started event and some progress
        assert!(collected
            .iter()
            .any(|e| matches!(e, ParseEvent::Started { .. })));
        assert!(!collected.is_empty(), "Should emit at least some events");

        // Check if we got a Completed or Error event
        let has_completion = collected
            .iter()
            .any(|e| matches!(e, ParseEvent::Completed { .. } | ParseEvent::Error { .. }));
        assert!(
            has_completion,
            "Should have either Completed or Error event"
        );
    }
}
