//! Async RDF streaming parser for high-performance large file processing
//!
//! **Honest streaming status**: only [`RdfFormat::NTriples`] and
//! [`RdfFormat::NQuads`] are truly streamed here -- they are line-based
//! formats, so each chunk read from the source is processed line-by-line as
//! it arrives and the accumulation buffer is cleared after every batch of
//! complete lines (bounded memory, independent of document size).
//!
//! Turtle, TriG, RDF/XML, JSON-LD and N3 are **not** incrementally streamed:
//! their grammars allow constructs (multi-line statements, blank-node
//! property lists, forward references inside a single document) that the
//! underlying oxttl/oxrdfxml/oxjsonld parsers only expose through a
//! synchronous [`std::io::Read`]-based API, which cannot be safely driven
//! from an arbitrary [`tokio::io::AsyncRead`] without either spawning a
//! blocking bridge thread or fully materializing the document first. This
//! parser takes the second option: it accumulates chunks into memory and
//! parses the complete document once the source is exhausted. To keep that
//! bounded and fail loudly instead of exhausting memory on an unexpectedly
//! large source, accumulation is capped at [`AsyncStreamingParser::max_buffer_size`]
//! (configurable via [`AsyncStreamingParser::with_max_buffer_size`]); once
//! the cap is exceeded, [`AsyncStreamingParser::parse_stream`] returns an
//! explicit [`OxirsError::Parse`] rather than continuing to buffer.

use super::{Parser, ParserConfig, RdfFormat};
use crate::model::Quad;
use crate::{OxirsError, Result};
use std::future::Future;
use std::pin::Pin;

/// Default cap (256 MiB) on the in-memory accumulation buffer used for
/// formats that are not truly streamed (see module docs).
#[cfg(feature = "async")]
const DEFAULT_MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Async RDF streaming parser for high-performance large file processing
///
/// See the module-level docs for which formats are genuinely streamed
/// (N-Triples/N-Quads) versus buffered-then-parsed (everything else).
#[cfg(feature = "async")]
pub struct AsyncStreamingParser {
    format: RdfFormat,
    config: ParserConfig,
    progress_callback: Option<Box<dyn Fn(usize) + Send + Sync>>,
    chunk_size: usize,
    max_buffer_size: usize,
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
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
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

    /// Set the maximum number of bytes this parser will accumulate in
    /// memory before returning an error, for formats that are not
    /// incrementally streamed (anything other than N-Triples/N-Quads; see
    /// module docs). Has no effect on N-Triples/N-Quads, which always
    /// stream with bounded per-chunk memory regardless of document size.
    pub fn with_max_buffer_size(mut self, max_buffer_size: usize) -> Self {
        self.max_buffer_size = max_buffer_size;
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

            // Process complete lines for line-based formats (N-Triples, N-Quads):
            // these are truly streamed and accumulated_data is drained below.
            if matches!(self.format, RdfFormat::NTriples | RdfFormat::NQuads) {
                self.process_lines_async(&mut accumulated_data, &mut line_buffer, &mut handler)
                    .await?;
            } else if accumulated_data.len() > self.max_buffer_size {
                // Every other format is not incrementally streamed (see
                // module docs): fail loudly instead of silently buffering
                // an unbounded amount of the source into memory.
                return Err(OxirsError::Parse(format!(
                    "AsyncStreamingParser: {:?} is not incrementally streamed and the \
                     source exceeded the configured max_buffer_size of {} bytes \
                     (buffered {} bytes so far). Increase the limit via \
                     with_max_buffer_size(), split the document, or use \
                     RdfFormat::NTriples/NQuads for true streaming.",
                    self.format,
                    self.max_buffer_size,
                    accumulated_data.len()
                )));
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
                    // For other formats (see module docs), the document is
                    // fully buffered (bounded by max_buffer_size above) and
                    // parsed once via the synchronous Parser API. Results
                    // are then handed to the async handler directly -- no
                    // nested runtime/block_on bridge is needed since we are
                    // not calling an async fn from inside a sync callback.
                    let parser = Parser::with_config(self.format, self.config.clone());
                    let quads = parser.parse_str_to_quads(&accumulated_data)?;
                    for quad in quads {
                        handler(quad).await?;
                    }
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
    pub async fn parse_bytes<F, Fut>(&self, data: &[u8], handler: F) -> Result<()>
    where
        F: FnMut(Quad) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        use std::io::Cursor;
        let cursor = Cursor::new(data);
        self.parse_stream(cursor, handler).await
    }

    /// Parse from string asynchronously
    pub async fn parse_str_async<F, Fut>(&self, data: &str, handler: F) -> Result<()>
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
impl Default for MemoryAsyncSink {
    fn default() -> Self {
        Self::new()
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

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    /// N-Triples must remain truly line-streamed: this is a baseline
    /// correctness check for the fast path that `process_lines_async`
    /// already handles (untouched by the buffered-format fix below).
    #[tokio::test]
    async fn test_ntriples_streams_correctly() {
        let data = "<http://example.org/s1> <http://example.org/p> \"o1\" .\n\
                     <http://example.org/s2> <http://example.org/p> \"o2\" .\n";
        let reader = Cursor::new(data.as_bytes());
        let parser = AsyncStreamingParser::new(RdfFormat::NTriples);

        let quads = Arc::new(Mutex::new(Vec::new()));
        let quads_clone = Arc::clone(&quads);
        parser
            .parse_stream(reader, move |quad| {
                let quads = Arc::clone(&quads_clone);
                async move {
                    quads
                        .lock()
                        .map_err(|_| OxirsError::Parse("poisoned".into()))?
                        .push(quad);
                    Ok(())
                }
            })
            .await
            .expect("N-Triples streaming should succeed");

        assert_eq!(quads.lock().expect("lock").len(), 2);
    }

    /// Regression test for the P0 finding: parsing a buffered (non
    /// line-based) format must no longer nest a `block_in_place` +
    /// `block_on` call inside the already-async `parse_stream` future (this
    /// used to panic/deadlock risk if called from certain runtime flavors,
    /// and is the P2 concurrency finding at async_parser.rs:112). We assert
    /// the happy path still produces correct quads through the async
    /// handler without any nested runtime call.
    #[tokio::test]
    async fn test_turtle_buffered_parsing_produces_correct_quads() {
        let data = r#"@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob ."#;
        let reader = Cursor::new(data.as_bytes());
        let parser = AsyncStreamingParser::new(RdfFormat::Turtle);

        let quads = Arc::new(Mutex::new(Vec::new()));
        let quads_clone = Arc::clone(&quads);
        parser
            .parse_stream(reader, move |quad| {
                let quads = Arc::clone(&quads_clone);
                async move {
                    quads
                        .lock()
                        .map_err(|_| OxirsError::Parse("poisoned".into()))?
                        .push(quad);
                    Ok(())
                }
            })
            .await
            .expect("buffered Turtle parsing should succeed");

        assert_eq!(quads.lock().expect("lock").len(), 1);
    }

    /// Regression test for the P0 finding: non-line-based formats used to
    /// buffer the entire document with no bound. Now, exceeding the
    /// configured `max_buffer_size` must fail loudly with a clear error
    /// instead of continuing to grow memory unboundedly.
    #[tokio::test]
    async fn test_non_streaming_format_exceeding_max_buffer_errors_loudly() {
        let data = r#"@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob, ex:carol, ex:dave, ex:eve, ex:frank ."#;
        let reader = Cursor::new(data.as_bytes());
        // Deliberately tiny cap so the (~100+ byte) document overflows it.
        let parser = AsyncStreamingParser::new(RdfFormat::Turtle)
            .with_chunk_size(16)
            .with_max_buffer_size(8);

        let result = parser.parse_stream(reader, |_quad| async { Ok(()) }).await;

        assert!(
            result.is_err(),
            "exceeding max_buffer_size must be a loud error, not silent unbounded buffering"
        );
        let message = result.unwrap_err().to_string();
        assert!(
            message.contains("max_buffer_size") || message.contains("not incrementally streamed"),
            "error should clearly explain the buffering limitation: {message}"
        );
    }

    /// N-Triples must be exempt from the buffer cap: it is genuinely
    /// streamed line-by-line and `accumulated_data` never grows unbounded,
    /// so an arbitrarily large N-Triples document must succeed even with a
    /// tiny `max_buffer_size`.
    #[tokio::test]
    async fn test_ntriples_ignores_max_buffer_size_cap() {
        let mut data = String::new();
        for i in 0..50 {
            data.push_str(&format!(
                "<http://example.org/s{i}> <http://example.org/p> \"o{i}\" .\n"
            ));
        }
        let reader = Cursor::new(data.as_bytes());
        let parser = AsyncStreamingParser::new(RdfFormat::NTriples)
            .with_chunk_size(16)
            .with_max_buffer_size(8);

        let quads = Arc::new(Mutex::new(Vec::new()));
        let quads_clone = Arc::clone(&quads);
        parser
            .parse_stream(reader, move |quad| {
                let quads = Arc::clone(&quads_clone);
                async move {
                    quads
                        .lock()
                        .map_err(|_| OxirsError::Parse("poisoned".into()))?
                        .push(quad);
                    Ok(())
                }
            })
            .await
            .expect("true streaming formats must ignore the buffered-format cap");

        assert_eq!(quads.lock().expect("lock").len(), 50);
    }
}
