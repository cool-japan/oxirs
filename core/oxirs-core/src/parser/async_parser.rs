//! Async RDF streaming parser for high-performance large file processing

use super::{Parser, ParserConfig, RdfFormat};
use crate::model::Quad;
use crate::{OxirsError, Result};
use std::future::Future;
use std::pin::Pin;

/// Async RDF streaming parser for high-performance large file processing
#[cfg(feature = "async")]
pub struct AsyncStreamingParser {
    format: RdfFormat,
    config: ParserConfig,
    progress_callback: Option<Box<dyn Fn(usize) + Send + Sync>>,
    chunk_size: usize,
}

#[cfg(feature = "async")]
impl AsyncStreamingParser {
    /// Create a new async streaming parser
    pub fn new(format: RdfFormat) -> Self {
        AsyncStreamingParser {
            format,
            config: ParserConfig::default(),
            progress_callback: None,
            chunk_size: 8192, // 8KB default chunk size
        }
    }

    /// Set a progress callback that reports the number of bytes processed
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Set the chunk size for streaming processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Configure error tolerance
    pub fn with_error_tolerance(mut self, ignore_errors: bool) -> Self {
        self.config.ignore_errors = ignore_errors;
        self
    }

    /// Parse from an async readable stream
    pub async fn parse_stream<R, F, Fut>(&self, mut reader: R, mut handler: F) -> Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        use tokio::io::AsyncReadExt;

        let mut buffer = Vec::with_capacity(self.chunk_size);
        let mut accumulated_data = String::new();
        let mut bytes_processed = 0usize;
        let mut line_buffer = String::new();

        loop {
            buffer.clear();
            buffer.resize(self.chunk_size, 0);

            let bytes_read = reader.read(&mut buffer).await?;

            if bytes_read == 0 {
                break; // End of stream
            }

            buffer.truncate(bytes_read);
            bytes_processed += bytes_read;

            // Convert bytes to string and append to accumulated data
            let chunk_str = String::from_utf8_lossy(&buffer);
            accumulated_data.push_str(&chunk_str);

            // Process complete lines for line-based formats (N-Triples, N-Quads)
            if matches!(self.format, RdfFormat::NTriples | RdfFormat::NQuads) {
                self.process_lines_async(&mut accumulated_data, &mut line_buffer, &mut handler)
                    .await?;
            }

            // Report progress if callback is set
            if let Some(ref callback) = self.progress_callback {
                callback(bytes_processed);
            }
        }

        // Process any remaining data
        if !accumulated_data.is_empty() {
            match self.format {
                RdfFormat::NTriples | RdfFormat::NQuads => {
                    // Process final lines
                    accumulated_data.push_str(&line_buffer);
                    self.process_lines_async(
                        &mut accumulated_data,
                        &mut String::new(),
                        &mut handler,
                    )
                    .await?;
                }
                _ => {
                    // For other formats, parse the complete document
                    let parser = Parser::with_config(self.format, self.config.clone());
                    parser.parse_str_with_handler(&accumulated_data, |quad| {
                        // Convert sync closure to async - this is a simplified approach
                        // In a real implementation, you'd want to use proper async handling
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(handler(quad))
                        })
                    })?;
                }
            }
        }

        Ok(())
    }

    /// Process lines asynchronously for line-based formats
    async fn process_lines_async<F, Fut>(
        &self,
        accumulated_data: &mut String,
        line_buffer: &mut String,
        handler: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        // Combine line buffer with new data
        let mut full_data = line_buffer.clone();
        full_data.push_str(accumulated_data);

        let mut last_newline_pos = 0;

        // Find complete lines
        for (pos, _) in full_data.match_indices('\n') {
            let line = &full_data[last_newline_pos..pos];
            last_newline_pos = pos + 1;

            // Parse the line
            if let Some(quad) = self.parse_line(line)? {
                handler(quad).await?;
            }
        }

        // Keep incomplete line for next iteration
        line_buffer.clear();
        if last_newline_pos < full_data.len() {
            line_buffer.push_str(&full_data[last_newline_pos..]);
        }

        accumulated_data.clear();
        Ok(())
    }

    /// Parse a single line (for N-Triples/N-Quads)
    fn parse_line(&self, line: &str) -> Result<Option<Quad>> {
        let parser = Parser::with_config(self.format, self.config.clone());

        match self.format {
            RdfFormat::NTriples => parser.parse_ntriples_line(line),
            RdfFormat::NQuads => {
                // For N-Quads, we need a more sophisticated parser
                // This is a simplified implementation
                parser.parse_ntriples_line(line)
            }
            _ => Err(OxirsError::Parse(
                "Unsupported format for line parsing".to_string(),
            )),
        }
    }

    /// Parse from bytes asynchronously
    pub async fn parse_bytes<F, Fut>(&self, data: &[u8], mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        use std::io::Cursor;
        let cursor = Cursor::new(data);
        self.parse_stream(cursor, handler).await
    }

    /// Parse from string asynchronously
    pub async fn parse_str_async<F, Fut>(&self, data: &str, mut handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let bytes = data.as_bytes();
        self.parse_bytes(bytes, handler).await
    }

    /// Convenience method to parse to a vector asynchronously
    pub async fn parse_str_to_quads_async(&self, data: &str) -> Result<Vec<Quad>> {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let quads = Arc::new(Mutex::new(Vec::new()));
        let quads_clone = Arc::clone(&quads);

        self.parse_str_async(data, move |quad| {
            let quads = Arc::clone(&quads_clone);
            async move {
                quads.lock().await.push(quad);
                Ok(())
            }
        })
        .await?;

        let result = quads.lock().await;
        Ok(result.clone())
    }
}

/// Progress information for async parsing
#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub struct ParseProgress {
    pub bytes_processed: usize,
    pub quads_parsed: usize,
    pub errors_encountered: usize,
    pub estimated_total_bytes: Option<usize>,
}

#[cfg(feature = "async")]
impl ParseProgress {
    /// Calculate completion percentage if total size is known
    pub fn completion_percentage(&self) -> Option<f64> {
        self.estimated_total_bytes.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.bytes_processed as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Async streaming sink for writing parsed RDF data
#[cfg(feature = "async")]
pub trait AsyncRdfSink: Send + Sync {
    /// Process a parsed quad asynchronously
    fn process_quad(&mut self, quad: Quad)
        -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    /// Finalize processing (called when parsing is complete)
    fn finalize(&mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

/// Memory-based async sink that collects quads
#[cfg(feature = "async")]
pub struct MemoryAsyncSink {
    quads: Vec<Quad>,
}

#[cfg(feature = "async")]
impl MemoryAsyncSink {
    pub fn new() -> Self {
        MemoryAsyncSink { quads: Vec::new() }
    }

    pub fn into_quads(self) -> Vec<Quad> {
        self.quads
    }
}

#[cfg(feature = "async")]
impl AsyncRdfSink for MemoryAsyncSink {
    fn process_quad(
        &mut self,
        quad: Quad,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            self.quads.push(quad);
            Ok(())
        })
    }

    fn finalize(&mut self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move { Ok(()) })
    }
}
