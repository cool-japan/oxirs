//! Async streaming support for RDF parsing and serialization
//!
//! This module provides async/await compatible streaming interfaces for RDF data processing,
//! with support for backpressure, progress reporting, and cancellation.
//!
//! # Examples
//!
//! ## Async Parsing with Progress Reporting
//!
//! ```no_run
//! use oxirs_core::io::{AsyncRdfParser, AsyncStreamingParser, AsyncStreamingConfig};
//! use oxirs_core::parser::RdfFormat;
//! use tokio::fs::File;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let file = File::open("data.nt").await?;
//!     let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
//!     
//!     let config = AsyncStreamingConfig {
//!         chunk_size: 65536,  // 64KB chunks
//!         ..Default::default()
//!     };
//!     
//!     let progress = Box::new(|p: &oxirs_core::io::StreamingProgress| {
//!         println!("Parsed {} quads ({} bytes)", p.items_processed, p.bytes_processed);
//!     });
//!     
//!     let quads = parser.parse_async(file, config, Some(progress), None).await?;
//!     println!("Total: {} quads", quads.len());
//!     Ok(())
//! }
//! ```
//!
//! ## Async Serialization with Cancellation
//!
//! ```no_run
//! use oxirs_core::io::{AsyncRdfSerializer, AsyncStreamingSerializer, AsyncStreamingConfig};
//! use oxirs_core::parser::RdfFormat;
//! use oxirs_core::model::*;
//! use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let serializer = AsyncStreamingSerializer::new(RdfFormat::NTriples);
//!     let cancel_token = Arc::new(AtomicBool::new(false));
//!     
//!     // Create some quads
//!     let quads = vec![
//!         // ... your quads here
//!     ];
//!     
//!     let mut output = Vec::new();
//!     serializer.serialize_quads_async(
//!         &mut output,
//!         quads.into_iter(),
//!         AsyncStreamingConfig::default(),
//!         None,
//!         Some(cancel_token.clone()),
//!     ).await?;
//!     
//!     Ok(())
//! }
//! ```

use crate::{
    model::{Quad, Triple},
    parser::{Parser, ParserConfig, RdfFormat},
    serializer::Serializer,
    OxirsError, Result,
};
use futures::future::BoxFuture;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Progress information for async operations
#[derive(Debug, Clone)]
pub struct StreamingProgress {
    /// Total bytes processed
    pub bytes_processed: usize,
    /// Total items (quads/triples) processed
    pub items_processed: usize,
    /// Errors encountered (if error tolerance is enabled)
    pub errors_encountered: usize,
    /// Estimated total bytes (if known)
    pub total_bytes: Option<usize>,
    /// Processing rate (items per second)
    pub items_per_second: Option<f64>,
}

impl StreamingProgress {
    /// Create new progress info
    pub fn new() -> Self {
        StreamingProgress {
            bytes_processed: 0,
            items_processed: 0,
            errors_encountered: 0,
            total_bytes: None,
            items_per_second: None,
        }
    }

    /// Calculate completion percentage if total size is known
    pub fn completion_percentage(&self) -> Option<f64> {
        self.total_bytes.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_processed as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Callback for progress reporting
pub type ProgressCallback = Box<dyn Fn(&StreamingProgress) + Send + Sync>;

/// Configuration for async streaming operations
#[derive(Clone)]
pub struct AsyncStreamingConfig {
    /// Size of chunks to read/write at a time
    pub chunk_size: usize,
    /// Size of the internal buffer
    pub buffer_size: usize,
    /// Whether to continue on parse errors
    pub ignore_errors: bool,
    /// Maximum number of errors before stopping
    pub max_errors: Option<usize>,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
}

impl Default for AsyncStreamingConfig {
    fn default() -> Self {
        AsyncStreamingConfig {
            chunk_size: 8192,      // 8KB chunks
            buffer_size: 65536,    // 64KB buffer
            ignore_errors: false,
            max_errors: None,
            base_iri: None,
        }
    }
}

/// Trait for async RDF parsing with streaming input
pub trait AsyncRdfParser: Send + Sync {
    /// Parse RDF data from an async reader
    fn parse_async<'a, R>(
        &'a self,
        reader: R,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<Vec<Quad>>>
    where
        R: AsyncRead + Unpin + Send + 'a;

    /// Parse RDF data from an async reader with a custom handler
    fn parse_with_handler_async<'a, R, F, Fut>(
        &'a self,
        reader: R,
        handler: F,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        R: AsyncRead + Unpin + Send + 'a,
        F: FnMut(Quad) -> Fut + Send + 'a,
        Fut: std::future::Future<Output = Result<()>> + Send + 'a;
}

/// Trait for async RDF serialization with streaming output
pub trait AsyncRdfSerializer: Send + Sync {
    /// Serialize quads to an async writer
    fn serialize_quads_async<'a, W, I>(
        &'a self,
        writer: W,
        quads: I,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        W: AsyncWrite + Unpin + Send + 'a,
        I: Iterator<Item = Quad> + Send + 'a;

    /// Serialize triples to an async writer
    fn serialize_triples_async<'a, W, I>(
        &'a self,
        writer: W,
        triples: I,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        W: AsyncWrite + Unpin + Send + 'a,
        I: Iterator<Item = Triple> + Send + 'a;
}

/// Async streaming parser implementation
pub struct AsyncStreamingParser {
    format: RdfFormat,
}

impl AsyncStreamingParser {
    /// Create a new async streaming parser
    pub fn new(format: RdfFormat) -> Self {
        AsyncStreamingParser { format }
    }

    /// Check if cancellation was requested
    fn check_cancelled(cancel_token: &Option<Arc<AtomicBool>>) -> Result<()> {
        if let Some(token) = cancel_token {
            if token.load(Ordering::Relaxed) {
                return Err(OxirsError::Io(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "Operation cancelled",
                )));
            }
        }
        Ok(())
    }

    /// Report progress
    fn report_progress(
        progress_callback: &Option<ProgressCallback>,
        progress_info: &StreamingProgress,
    ) {
        if let Some(callback) = progress_callback {
            callback(progress_info);
        }
    }

    /// Parse line-based formats (N-Triples, N-Quads)
    async fn parse_line_based<R, F, Fut>(
        &self,
        mut reader: R,
        mut handler: F,
        config: AsyncStreamingConfig,
        progress_callback: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> Result<()>
    where
        R: AsyncRead + Unpin + Send,
        F: FnMut(Quad) -> Fut + Send,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        let parser_config = ParserConfig {
            base_iri: config.base_iri.clone(),
            ignore_errors: config.ignore_errors,
            max_errors: config.max_errors,
        };
        let parser = Parser::with_config(self.format, parser_config);

        let mut buffer = vec![0u8; config.chunk_size];
        let mut accumulated = Vec::new();
        let mut line_buffer = String::new();
        let mut progress = StreamingProgress::new();
        let start_time = std::time::Instant::now();

        loop {
            Self::check_cancelled(&cancel_token)?;

            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                break; // EOF
            }

            progress.bytes_processed += bytes_read;
            accumulated.extend_from_slice(&buffer[..bytes_read]);

            // Process complete lines
            while let Some(newline_pos) = accumulated.iter().position(|&b| b == b'\n') {
                let line_bytes = accumulated.drain(..=newline_pos).collect::<Vec<_>>();
                
                // Convert to string, handling UTF-8 errors gracefully
                match String::from_utf8(line_bytes) {
                    Ok(mut line) => {
                        // Remove newline
                        if line.ends_with('\n') {
                            line.pop();
                            if line.ends_with('\r') {
                                line.pop();
                            }
                        }
                        
                        line_buffer.push_str(&line);
                        
                        // Try to parse the line
                        match self.parse_single_line(&parser, &line_buffer) {
                            Ok(Some(quad)) => {
                                handler(quad).await?;
                                progress.items_processed += 1;
                                line_buffer.clear();
                            }
                            Ok(None) => {
                                // Empty line or comment
                                line_buffer.clear();
                            }
                            Err(e) => {
                                if config.ignore_errors {
                                    progress.errors_encountered += 1;
                                    if let Some(max_errors) = config.max_errors {
                                        if progress.errors_encountered >= max_errors {
                                            return Err(e);
                                        }
                                    }
                                    tracing::warn!("Parse error: {}", e);
                                    line_buffer.clear();
                                } else {
                                    return Err(e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if config.ignore_errors {
                            progress.errors_encountered += 1;
                            tracing::warn!("UTF-8 error: {}", e);
                        } else {
                            return Err(OxirsError::Parse(format!("UTF-8 error: {}", e)));
                        }
                    }
                }
            }

            // Calculate processing rate
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                progress.items_per_second = Some(progress.items_processed as f64 / elapsed);
            }

            Self::report_progress(&progress_callback, &progress);
        }

        // Process any remaining data
        if !accumulated.is_empty() {
            match String::from_utf8(accumulated) {
                Ok(remaining) => {
                    line_buffer.push_str(&remaining);
                    if !line_buffer.trim().is_empty() {
                        if let Ok(Some(quad)) = self.parse_single_line(&parser, &line_buffer) {
                            handler(quad).await?;
                            progress.items_processed += 1;
                        }
                    }
                }
                Err(e) => {
                    if !config.ignore_errors {
                        return Err(OxirsError::Parse(format!("UTF-8 error: {}", e)));
                    }
                }
            }
        }

        Self::report_progress(&progress_callback, &progress);
        Ok(())
    }

    /// Parse a single line for line-based formats
    fn parse_single_line(&self, parser: &Parser, line: &str) -> Result<Option<Quad>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        match self.format {
            RdfFormat::NTriples => parser.parse_ntriples_line(line),
            RdfFormat::NQuads => parser.parse_nquads_line(line),
            _ => Err(OxirsError::Parse(
                "Format not supported for line-based parsing".to_string(),
            )),
        }
    }

    /// Parse document-based formats (Turtle, TriG, RDF/XML, JSON-LD)
    async fn parse_document_based<R, F, Fut>(
        &self,
        mut reader: R,
        mut handler: F,
        config: AsyncStreamingConfig,
        progress_callback: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> Result<()>
    where
        R: AsyncRead + Unpin + Send,
        F: FnMut(Quad) -> Fut + Send,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        // For document-based formats, we need to read the entire document
        let mut buffer = Vec::new();
        let mut chunk = vec![0u8; config.chunk_size];
        let mut progress = StreamingProgress::new();

        loop {
            Self::check_cancelled(&cancel_token)?;

            let bytes_read = reader.read(&mut chunk).await?;
            if bytes_read == 0 {
                break;
            }

            buffer.extend_from_slice(&chunk[..bytes_read]);
            progress.bytes_processed += bytes_read;

            Self::report_progress(&progress_callback, &progress);
        }

        // Parse the complete document
        let document = String::from_utf8(buffer)
            .map_err(|e| OxirsError::Parse(format!("UTF-8 error: {}", e)))?;

        let parser_config = ParserConfig {
            base_iri: config.base_iri,
            ignore_errors: config.ignore_errors,
            max_errors: config.max_errors,
        };
        let parser = Parser::with_config(self.format, parser_config);

        // Parse the document and collect quads
        let quads = parser.parse_str_to_quads(&document)?;
        
        // Process each quad with the async handler
        for quad in quads {
            Self::check_cancelled(&cancel_token)?;
            handler(quad).await?;
            progress.items_processed += 1;
        }

        Self::report_progress(&progress_callback, &progress);
        Ok(())
    }
}

impl AsyncRdfParser for AsyncStreamingParser {
    fn parse_async<'a, R>(
        &'a self,
        reader: R,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<Vec<Quad>>>
    where
        R: AsyncRead + Unpin + Send + 'a,
    {
        Box::pin(async move {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            
            // Spawn a task to collect quads
            let collector = tokio::spawn(async move {
                let mut quads = Vec::new();
                while let Some(quad) = rx.recv().await {
                    quads.push(quad);
                }
                quads
            });
            
            // Parse with handler that sends quads to the channel
            let parse_result = self.parse_with_handler_async(
                reader,
                |quad| {
                    let tx = tx.clone();
                    async move {
                        tx.send(quad).map_err(|_| OxirsError::Parse("Channel send error".to_string()))?;
                        Ok(())
                    }
                },
                config,
                progress,
                cancel_token,
            )
            .await;
            
            // Close the channel
            drop(tx);
            
            // Check for parse errors
            parse_result?;
            
            // Collect the quads
            let quads = collector.await.map_err(|_| OxirsError::Parse("Failed to collect quads".to_string()))?;
            Ok(quads)
        })
    }

    fn parse_with_handler_async<'a, R, F, Fut>(
        &'a self,
        reader: R,
        handler: F,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        R: AsyncRead + Unpin + Send + 'a,
        F: FnMut(Quad) -> Fut + Send + 'a,
        Fut: std::future::Future<Output = Result<()>> + Send + 'a,
    {
        Box::pin(async move {
            match self.format {
                RdfFormat::NTriples | RdfFormat::NQuads => {
                    self.parse_line_based(reader, handler, config, progress, cancel_token)
                        .await
                }
                _ => {
                    self.parse_document_based(reader, handler, config, progress, cancel_token)
                        .await
                }
            }
        })
    }
}

/// Async streaming serializer implementation
pub struct AsyncStreamingSerializer {
    format: RdfFormat,
}

impl AsyncStreamingSerializer {
    /// Create a new async streaming serializer
    pub fn new(format: RdfFormat) -> Self {
        AsyncStreamingSerializer { format }
    }

    /// Check if cancellation was requested
    fn check_cancelled(cancel_token: &Option<Arc<AtomicBool>>) -> Result<()> {
        if let Some(token) = cancel_token {
            if token.load(Ordering::Relaxed) {
                return Err(OxirsError::Io(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "Operation cancelled",
                )));
            }
        }
        Ok(())
    }

    /// Serialize a single quad to a string
    fn serialize_quad(&self, quad: &Quad) -> Result<String> {
        let serializer = Serializer::new(self.format);
        match self.format {
            RdfFormat::NTriples => {
                // For N-Triples, we only serialize if it's in the default graph
                if quad.is_default_graph() {
                    let mut graph = crate::model::Graph::new();
                    graph.insert(quad.to_triple());
                    serializer.serialize_graph(&graph)
                } else {
                    Ok(String::new())
                }
            }
            RdfFormat::NQuads => serializer.serialize_quad_to_nquads(quad),
            _ => Err(OxirsError::Serialize(
                "Format not supported for streaming serialization".to_string(),
            )),
        }
    }

    /// Serialize a single triple to a string
    fn serialize_triple(&self, triple: &Triple) -> Result<String> {
        let quad = Quad::from_triple(triple.clone());
        self.serialize_quad(&quad)
    }
}

impl AsyncRdfSerializer for AsyncStreamingSerializer {
    fn serialize_quads_async<'a, W, I>(
        &'a self,
        mut writer: W,
        quads: I,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        W: AsyncWrite + Unpin + Send + 'a,
        I: Iterator<Item = Quad> + Send + 'a,
    {
        Box::pin(async move {
            let mut buffer = String::with_capacity(config.buffer_size);
            let mut progress_info = StreamingProgress::new();
            let start_time = std::time::Instant::now();

            for quad in quads {
                Self::check_cancelled(&cancel_token)?;

                // Serialize the quad
                let serialized = self.serialize_quad(&quad)?;
                buffer.push_str(&serialized);
                progress_info.items_processed += 1;

                // Write buffer if it's getting full
                if buffer.len() >= config.chunk_size {
                    writer.write_all(buffer.as_bytes()).await?;
                    progress_info.bytes_processed += buffer.len();
                    buffer.clear();
                }

                // Update progress
                let elapsed = start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    progress_info.items_per_second = Some(progress_info.items_processed as f64 / elapsed);
                }

                if let Some(ref callback) = progress {
                    callback(&progress_info);
                }
            }

            // Write any remaining data
            if !buffer.is_empty() {
                writer.write_all(buffer.as_bytes()).await?;
                progress_info.bytes_processed += buffer.len();
                
                // Report final progress
                if let Some(ref callback) = progress {
                    callback(&progress_info);
                }
            }

            // Flush the writer
            writer.flush().await?;

            Ok(())
        })
    }

    fn serialize_triples_async<'a, W, I>(
        &'a self,
        writer: W,
        triples: I,
        config: AsyncStreamingConfig,
        progress: Option<ProgressCallback>,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> BoxFuture<'a, Result<()>>
    where
        W: AsyncWrite + Unpin + Send + 'a,
        I: Iterator<Item = Triple> + Send + 'a,
    {
        let quads = triples.map(Quad::from_triple);
        self.serialize_quads_async(writer, quads, config, progress, cancel_token)
    }
}

/// Buffered async reader with backpressure support
pub struct BackpressureReader<R> {
    inner: R,
    buffer: Vec<u8>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
}

impl<R: AsyncRead + Unpin> BackpressureReader<R> {
    /// Create a new backpressure reader with specified buffer capacity
    pub fn new(inner: R, capacity: usize) -> Self {
        BackpressureReader {
            inner,
            buffer: vec![0; capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
        }
    }

    /// Get the number of bytes available in the buffer
    pub fn available(&self) -> usize {
        self.write_pos - self.read_pos
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.available() == self.capacity
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for BackpressureReader<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let me = self.get_mut();
        
        // If we have data in the buffer, return it
        if me.available() > 0 {
            let to_read = std::cmp::min(buf.remaining(), me.available());
            buf.put_slice(&me.buffer[me.read_pos..me.read_pos + to_read]);
            me.read_pos += to_read;

            // Reset positions if buffer is empty
            if me.read_pos == me.write_pos {
                me.read_pos = 0;
                me.write_pos = 0;
            }

            return Poll::Ready(Ok(()));
        }

        // Otherwise, try to fill the buffer
        let write_pos = me.write_pos;
        let mut read_buf = tokio::io::ReadBuf::new(&mut me.buffer[write_pos..]);
        match Pin::new(&mut me.inner).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => {
                let bytes_read = read_buf.filled().len();
                if bytes_read == 0 {
                    // EOF
                    return Poll::Ready(Ok(()));
                }

                me.write_pos += bytes_read;

                // Now serve from buffer
                let to_read = std::cmp::min(buf.remaining(), me.available());
                buf.put_slice(&me.buffer[me.read_pos..me.read_pos + to_read]);
                me.read_pos += to_read;

                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_async_ntriples_parsing() {
        let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
"#;

        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
        let reader = std::io::Cursor::new(ntriples_data.as_bytes());
        
        let quads = parser
            .parse_async(reader, AsyncStreamingConfig::default(), None, None)
            .await
            .unwrap();

        assert_eq!(quads.len(), 2);
        assert!(quads[0].is_default_graph());
        assert!(quads[1].is_default_graph());
    }

    #[tokio::test]
    async fn test_async_parsing_with_progress() {
        let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
<http://example.org/charlie> <http://xmlns.com/foaf/0.1/name> "Charlie" .
"#;

        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
        let reader = std::io::Cursor::new(ntriples_data.as_bytes());
        
        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();
        
        let progress_callback = Box::new(move |progress: &StreamingProgress| {
            progress_count_clone.fetch_add(1, Ordering::Relaxed);
            assert!(progress.bytes_processed > 0);
            assert!(progress.items_processed <= 3);
        });

        let quads = parser
            .parse_async(
                reader,
                AsyncStreamingConfig::default(),
                Some(progress_callback),
                None,
            )
            .await
            .unwrap();

        assert_eq!(quads.len(), 3);
        assert!(progress_count.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_async_parsing_with_cancellation() {
        let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
<http://example.org/charlie> <http://xmlns.com/foaf/0.1/name> "Charlie" .
"#;

        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
        let reader = std::io::Cursor::new(ntriples_data.as_bytes());
        
        let cancel_token = Arc::new(AtomicBool::new(false));
        let cancel_token_clone = cancel_token.clone();

        // Set up handler that cancels after first quad
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        let result = parser
            .parse_with_handler_async(
                reader,
                |_quad| {
                    let token = cancel_token_clone.clone();
                    let count = count_clone.clone();
                    async move {
                        let current = count.fetch_add(1, Ordering::Relaxed);
                        if current == 0 {
                            token.store(true, Ordering::Relaxed);
                        }
                        Ok(())
                    }
                },
                AsyncStreamingConfig::default(),
                None,
                Some(cancel_token),
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_async_ntriples_serialization() {
        let mut quads = Vec::new();
        
        let alice = NamedNode::new("http://example.org/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let alice_name = Literal::new("Alice");
        let triple1 = Triple::new(alice.clone(), name_pred.clone(), alice_name);
        quads.push(Quad::from_triple(triple1));

        let bob = NamedNode::new("http://example.org/bob").unwrap();
        let bob_name = Literal::new("Bob");
        let triple2 = Triple::new(bob, name_pred, bob_name);
        quads.push(Quad::from_triple(triple2));

        let serializer = AsyncStreamingSerializer::new(RdfFormat::NTriples);
        let mut output = Vec::new();
        
        serializer
            .serialize_quads_async(
                &mut output,
                quads.into_iter(),
                AsyncStreamingConfig::default(),
                None,
                None,
            )
            .await
            .unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("http://example.org/alice"));
        assert!(result.contains("http://example.org/bob"));
        assert!(result.contains("\"Alice\""));
        assert!(result.contains("\"Bob\""));
    }

    #[tokio::test]
    async fn test_async_serialization_with_progress() {
        let mut triples = Vec::new();
        
        for i in 0..10 {
            let subject = NamedNode::new(&format!("http://example.org/item{}", i)).unwrap();
            let pred = NamedNode::new("http://example.org/value").unwrap();
            let obj = Literal::new(i.to_string());
            triples.push(Triple::new(subject, pred, obj));
        }

        let serializer = AsyncStreamingSerializer::new(RdfFormat::NTriples);
        let mut output = Vec::new();
        
        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();
        let last_bytes = Arc::new(AtomicUsize::new(0));
        let last_bytes_clone = last_bytes.clone();
        
        let progress_callback = Box::new(move |progress: &StreamingProgress| {
            progress_count_clone.fetch_add(1, Ordering::Relaxed);
            assert!(progress.items_processed <= 10);
            // bytes should be monotonically increasing or stay the same
            let prev_bytes = last_bytes_clone.load(Ordering::Relaxed);
            assert!(progress.bytes_processed >= prev_bytes);
            last_bytes_clone.store(progress.bytes_processed, Ordering::Relaxed);
        });

        serializer
            .serialize_triples_async(
                &mut output,
                triples.into_iter(),
                AsyncStreamingConfig::default(),
                Some(progress_callback),
                None,
            )
            .await
            .unwrap();

        assert!(progress_count.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_async_error_tolerance() {
        let invalid_ntriples = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
INVALID LINE HERE
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
"#;

        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
        let reader = std::io::Cursor::new(invalid_ntriples.as_bytes());
        
        let mut config = AsyncStreamingConfig::default();
        config.ignore_errors = true;
        
        let quads = parser
            .parse_async(reader, config, None, None)
            .await
            .unwrap();

        // Should parse the two valid lines
        assert_eq!(quads.len(), 2);
    }

    #[tokio::test]
    async fn test_backpressure_reader() {
        let data = b"Hello, World!";
        let cursor = std::io::Cursor::new(data);
        let mut reader = BackpressureReader::new(cursor, 16); // Buffer size larger than data

        let mut output = Vec::new();
        reader.read_to_end(&mut output).await.unwrap();

        assert_eq!(output, data);
    }

    #[tokio::test]
    async fn test_async_nquads_parsing() {
        let nquads_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" <http://example.org/graph1> .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" <http://example.org/graph2> .
"#;

        let parser = AsyncStreamingParser::new(RdfFormat::NQuads);
        let reader = std::io::Cursor::new(nquads_data.as_bytes());
        
        let quads = parser
            .parse_async(reader, AsyncStreamingConfig::default(), None, None)
            .await
            .unwrap();

        assert_eq!(quads.len(), 2);
        assert!(!quads[0].is_default_graph());
        assert!(!quads[1].is_default_graph());
    }

    #[tokio::test]
    async fn test_large_chunk_parsing() {
        // Create a large dataset
        let mut ntriples_data = String::new();
        for i in 0..1000 {
            ntriples_data.push_str(&format!(
                "<http://example.org/item{}> <http://example.org/value> \"{}\" .\n",
                i, i
            ));
        }

        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);
        let reader = std::io::Cursor::new(ntriples_data.as_bytes());
        
        let mut config = AsyncStreamingConfig::default();
        config.chunk_size = 1024; // Small chunks to test buffering
        
        let quads = parser
            .parse_async(reader, config, None, None)
            .await
            .unwrap();

        assert_eq!(quads.len(), 1000);
    }

    #[tokio::test]
    async fn test_custom_base_iri() {
        let turtle_data = r#"@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob ."#;

        let parser = AsyncStreamingParser::new(RdfFormat::Turtle);
        let reader = std::io::Cursor::new(turtle_data.as_bytes());
        
        let mut config = AsyncStreamingConfig::default();
        config.base_iri = Some("http://example.org/".to_string());
        
        let quads = parser
            .parse_async(reader, config, None, None)
            .await
            .unwrap();

        assert_eq!(quads.len(), 1);
        let triple = quads[0].to_triple();
        
        // Should contain the example.org namespace
        if let Subject::NamedNode(subj) = triple.subject() {
            assert!(subj.as_str().contains("example.org"));
        }
    }
}